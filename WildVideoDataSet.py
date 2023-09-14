import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json
import os
import os.path as osp
import numpy as np
import copy
from mmengine import Config
import random
from utils import get_high_frequency_map, get_cond_color

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

class WildVideoDataSet(data.Dataset):
    """
        Test Dataset for CP-VTON.
    """
    def __init__(self, opt, tokenizer=None):
        super(WildVideoDataSet, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize([0.5], [0.5]),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        im_names = os.listdir(osp.join(self.root, 'image'))
        im_names = sorted(im_names)
        print(im_names)
        self.im_names = im_names
        self.tokenizer = tokenizer
        self.is_atr = opt.is_atr

    def name(self):
        return "WildVideoDataSet"

    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
        if not self.is_atr:
            parse_hair = ((parse_array == 1).astype(np.float32) +
                        (parse_array == 2).astype(np.float32))
            parse_head = ((parse_array == 4).astype(np.float32) +
                        (parse_array == 13).astype(np.float32))
            parse_lower = ((parse_array == 9).astype(np.float32) +
                        (parse_array == 12).astype(np.float32) +
                        (parse_array == 16).astype(np.float32) +
                        (parse_array == 17).astype(np.float32) +
                        (parse_array == 18).astype(np.float32) +
                        (parse_array == 19).astype(np.float32))
        else:
            parse_hair = ((parse_array == 1).astype(np.float32) +
                        (parse_array == 2).astype(np.float32))
            parse_head = ((parse_array == 3).astype(np.float32) +
                        (parse_array == 11).astype(np.float32))
            parse_lower = ((parse_array == 5).astype(np.float32) +
                        (parse_array == 6).astype(np.float32) +
                        (parse_array == 8).astype(np.float32) +
                        (parse_array == 12).astype(np.float32) +
                        (parse_array == 13).astype(np.float32) +
                        (parse_array == 9).astype(np.float32) +
                        (parse_array == 10).astype(np.float32))
            
        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        hand_masks = []
        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            mask_arm = Image.new('L', (parse_array.shape[1], parse_array.shape[0]), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
            mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
            hand_masks.append(parse_arm)

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_hair * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        hand_mask = (hand_masks[0] + hand_masks[1]).clip(0,1)
        return agnostic, hand_mask

    def single_get(self, index):

        self.data_path = '/root/autodl-tmp/zalando-hd-resized/test/'
        # self.data_path = '/data1/hzj/DressCode/upper_body/images'
        c_name = {}
        c_name['paired'] = '07148_00.jpg'
        high_frequency_map = {}
        color_map = {}
        c = {}
        for key in ['paired']:
            # c[key] = Image.open(osp.join(self.data_path, 'cloth', '12137_00.jpg')).convert('RGB')
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')

            # get high frequency map
            high_frequency_map[key] = get_high_frequency_map(osp.join(self.data_path, 'cloth', c_name[key]))
            high_frequency_map[key] = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(high_frequency_map[key])
            high_frequency_map[key] = self.transform(high_frequency_map[key])  # [-1,1]

            # get color map
            color_map[key] = get_cond_color(c[key])
            color_map[key] = transforms.Resize(self.fine_width, interpolation=2)(color_map[key])
            color_map[key] = self.transform(color_map[key])  # [-1,1]

            c[key] = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(c[key])
            c[key] = self.transform(c[key])  # [-1,1]

        # load_dino_fea
        fea_name = c_name[key].replace('.jpg', '.pt')
        dino_fea = torch.load(osp.join(self.data_path, 'dino_fea', fea_name),map_location='cpu')

        im_name = self.im_names[index]

        # person image
        im_pil_big = Image.open(osp.join(self.root, 'image', im_name))
        im_pil = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(im_pil_big)
        # print(self.fine_height,self.fine_width,im_pil_big.size, im_pil.size)
        im = self.transform(im_pil)

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.root, 'image-parse-v3', parse_name))
        im_parse = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse)[None]).long()
        im_parse = self.transform(im_parse.convert('RGB'))

        if not self.is_atr:
            labels = {
                0:  ['background',  [0, 10]],
                1:  ['hair',        [1, 2]],
                2:  ['face',        [4, 13]],
                3:  ['upper',       [5, 6, 7]],
                4:  ['bottom',      [9, 12]],
                5:  ['left_arm',    [14]],
                6:  ['right_arm',   [15]],
                7:  ['left_leg',    [16]],
                8:  ['right_leg',   [17]],
                9:  ['left_shoe',   [18]],
                10: ['right_shoe',  [19]],
                11: ['socks',       [8]],
                12: ['noise',       [3, 11]]
            }
        else:
            labels = {
                0:  ['background',  [0]],
                1:  ['hair',        [1, 2]],
                2:  ['face',        [3,11]],
                3:  ['upper',       [4, 7]],
                4:  ['bottom',      [5, 6, 8]],
                5:  ['left_arm',    [14]],
                6:  ['right_arm',   [15]],
                7:  ['left_leg',    [12]],
                8:  ['right_leg',   [13]],
                9:  ['left_shoe',   [9]],
                10: ['right_shoe',  [10]],
                11: ['scarf',       [17]],
                12: ['bag',       [16]]
            }
        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        # parse cloth & parse cloth mask
        pcm = new_parse_map[3:4]
        im_c = im * pcm + (1 - pcm)

        # parse head
        phead = new_parse_map[1] + new_parse_map[2]
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        if self.opt.with_one_hot:
            parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse_onehot[0] += parse_map[label] * i
        else:
            parse_onehot = None

        # if self.opt.with_parse_agnostic: 
        #     # load image-parse-agnostic
        #     image_parse_agnostic = Image.open(osp.join(self.root, 'image-parse-agnostic-v3.2', parse_name))
        #     image_parse_agnostic = transforms.Resize((self.fine_height,self.fine_width), interpolation=0)(image_parse_agnostic)
        #     parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        #     image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        #     parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        #     parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        #     new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        #     for i in range(len(labels)):
        #         for label in labels[i][1]:
        #             new_parse_agnostic_map[i] += parse_agnostic_map[label]
        # else:
        #     new_parse_agnostic_map = None

        # load pose points
        pose_name = im_name.replace('.jpg', '_rendered.png')
        pose_map = Image.open(osp.join(self.root, 'openpose_img', pose_name))
        pose_map = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]

        # pose name
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        h,w = im_pil_big.size
        with open(osp.join(self.root, 'openpose_json',pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
            # pose_data[1] = pose_data[1] * self.fine_height / h
            # pose_data[0] = pose_data[0] * self.fine_width / w

        # # load densepose
        # densepose_name = im_name.replace('image', 'image-densepose')
        # densepose_map = Image.open(osp.join(self.root, 'image-densepose', densepose_name))
        # densepose_map = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(densepose_map)
        # densepose_map = self.transform(densepose_map)  # [-1,1]

        agnostic, hand_mask = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        # parse other
        hand_mask = Image.fromarray(np.uint8(hand_mask), 'L')
        hand_mask = transforms.Resize((self.fine_height,self.fine_width), interpolation=0)(hand_mask)
        hand_mask = torch.tensor(np.array(hand_mask))
        parse_other = new_parse_map[1] + new_parse_map[2] + new_parse_map[4] + new_parse_map[7] + new_parse_map[8] + \
                            new_parse_map[9] + new_parse_map[10] + new_parse_map[11]
        parse_other += hand_mask
        parse_other = parse_other.clamp(0,1)
        im_other = im * parse_other + (1 - parse_other)
        parse_upper_mask = (1 - parse_other).unsqueeze(0)

        result = {
            'c_name': c_name,       # for visualization or ground truth
            'im_name':  im_name,    # for visualization or ground truth
            # intput 2 (segnet)
            #'parse_agnostic': new_parse_agnostic_map,
            #'densepose': densepose_map,
            'pose': pose_map,       # for conditioning
            # generator input
            'agnostic' : agnostic,
            # GT
            'parse_onehot' : parse_onehot,  # Cross Entropy
            'parse': new_parse_map, # GAN Loss real
            'pcm': pcm,             # L1 Loss & vis
            'parse_cloth': im_c,    # VGG Loss & vis
            'parse_other':im_other,
            'parse_upper_mask':parse_upper_mask,
            # visualization
            'image':    im,         # for visualization
            'head': im_h,  # for conditioning and visualization
            
            'cloth':    c,
            'high_frequency_map':high_frequency_map,
            'dino_fea':dino_fea,
            'color_map':color_map
            }
        
        return result

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, index):
        if index > 0:
            pre_index = index - 1
        else:
            pre_index = index
        result = self.single_get(index)
        pr = self.single_get(index-1)
        result['previous_pose'] = pr['pose']
        result['previous_image'] = pr['image']
        result['previous_parse_cloth'] = pr['parse_cloth']
        return result

class CPDataLoader(object):
    def __init__(self, batch_size, workers, shuffle, dataset, collate_fn=None):
        super(CPDataLoader, self).__init__()

        if shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = torch.utils.data.SequentialSampler(dataset)

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True, drop_last=True, sampler=train_sampler,collate_fn=collate_fn)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == '__main__':

    config = Config.fromfile('wild_config.py')
    opt = copy.deepcopy(config)
    opt.data_list = config.infer_data_list


    def ndim_tensor2im(image_tensor, imtype=np.uint8, batch=0):
        image_numpy = image_tensor[batch].cpu().float().numpy()
        result = np.argmax(image_numpy, axis=0)
        return result.astype(imtype)

    def visualize_segmap(input, multi_channel=True, tensor_out=True, batch=0) :
        palette = [
            0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51,
            254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85,
            85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220,
            0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0
        ]
        input = input.detach()
        if multi_channel :
            input = ndim_tensor2im(input,batch=batch)
        else :
            input = input[batch][0].cpu()
            input = np.asarray(input)
            input = input.astype(np.uint8)
        input = Image.fromarray(input, 'P')
        input.putpalette(palette)

        if tensor_out :
            trans = transforms.ToTensor()
            return trans(input.convert('RGB'))

        return input

    dataset = WildVideoDataSet(opt)

    p = dataset[5]

    agnostic = p['agnostic'].permute(1,2,0).numpy()
    agnostic *=255
    agnostic = agnostic.astype(np.uint8)
    agnostic= Image.fromarray(agnostic)
    agnostic.save('agnostic.jpg')

    image = p['image'].permute(1,2,0).numpy()
    print(image.shape)
    image *=255
    image = image.astype(np.uint8)
    image= Image.fromarray(image)
    image.save('image.jpg')

    parse_cloth = p['pcm'].permute(1,2,0).numpy()
    parse_cloth *=255
    parse_cloth = parse_cloth.astype(np.uint8)
    parse_cloth = parse_cloth[:,:,0]
    parse_cloth= Image.fromarray(parse_cloth)
    parse_cloth.save('pcm.jpg')

    parse = p['parse'].unsqueeze(0)
    parse = visualize_segmap(parse,tensor_out=False)
    parse.convert('RGB').save('parse.jpg')

    parse_other = p['parse_other'].permute(1,2,0).numpy()
    parse_other *=255
    parse_other = parse_other.astype(np.uint8)
    parse_other= Image.fromarray(parse_other)
    parse_other.save('parse_other.jpg')

    pose = p['pose'].permute(1,2,0).numpy()
    pose *=255
    pose = pose.astype(np.uint8)
    pose= Image.fromarray(pose)
    pose.save('pose.jpg')
    
    # densepose = p['densepose'].permute(1,2,0).numpy()
    # densepose *=255
    # densepose = densepose.astype(np.uint8)
    # densepose= Image.fromarray(densepose)
    # densepose.save('densepose.jpg')
    
    # parse_agnostic = p['parse_agnostic'].unsqueeze(0)
    # parse_agnostic = visualize_segmap(parse_agnostic,tensor_out=False)
    # parse_agnostic.convert('RGB').save('parse_agnostic.jpg')

    head = p['head'].permute(1,2,0).numpy()
    head *=255
    head = head.astype(np.uint8)
    head= Image.fromarray(head)
    head.save('head.jpg')

    parse_cloth = p['parse_cloth'].permute(1,2,0).numpy()
    parse_cloth *=255
    parse_cloth = parse_cloth.astype(np.uint8)
    parse_cloth= Image.fromarray(parse_cloth)
    parse_cloth.save('parse_cloth.jpg')

    image = p['previous_image'].permute(1,2,0).numpy()
    image *=255
    image = image.astype(np.uint8)
    image= Image.fromarray(image)
    image.save('previous_image.jpg')

    pose = p['previous_pose'].permute(1,2,0).numpy()
    pose *=255
    pose = pose.astype(np.uint8)
    pose= Image.fromarray(pose)
    pose.save('previous_pose.jpg')

    parse_cloth = p['previous_parse_cloth'].permute(1,2,0).numpy()
    parse_cloth *=255
    parse_cloth = parse_cloth.astype(np.uint8)
    parse_cloth= Image.fromarray(parse_cloth)
    parse_cloth.save('previous_parse_cloth.jpg')

    cloth = p['cloth']['paired'].permute(1,2,0).numpy()
    cloth *=255
    cloth = cloth.astype(np.uint8)
    cloth= Image.fromarray(cloth)
    cloth.save('cloth.jpg')