#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json

import os.path as osp
import numpy as np
import copy
from mmengine import Config
import random 
import cv2
from utils import get_cond_color, get_high_frequency_map
from skimage import feature as ft

class CPDataset(data.Dataset):
    """
        Dataset for CP-VTON.
    """
    def __init__(self, opt, random_pair=False):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize([0.5], [0.5]),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dino_transform = transforms.Compose([
            transforms.Resize((1022,756), interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.Resize((280,224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names
        self.random_pair = random_pair

    def name(self):
        return "CPDataset"
    
    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
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
            mask_arm = Image.new('L', (768, 1024), 'white')
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

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = 'image/' + im_name
        c_name = {}
        c = {}
        cm = {}
        high_frequency_map = {}
        dino_c = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]       
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')

            # get high frequency map
            high_frequency_map[key] = get_high_frequency_map(osp.join(self.data_path, 'cloth', c_name[key]))
            # high_frequency_map[key] = transforms.Resize(self.fine_width, interpolation=2)(high_frequency_map[key])
            # high_frequency_map[key] = self.transform(high_frequency_map[key])  # [-1,1]
            high_frequency_map[key] = self.dino_transform(high_frequency_map[key])  # [-1,1]

            dino_c[key] = self.dino_transform(c[key])

            c[key] = transforms.Resize(self.fine_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize(self.fine_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

        # load background
        foreground_name = im_name.replace('image', 'foreground').replace('.jpg', '_mask.png')
        foreground = Image.open(osp.join(self.data_path,  foreground_name))
        foreground = transforms.Resize(self.fine_width, interpolation=2)(foreground)
        foreground = np.array(foreground)
        foreground_mask = torch.tensor(foreground / 255)

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)
        # im = im * foreground_mask

        # load parsing image
        parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))

        # parse map
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
        
        if self.opt.with_parse_agnostic:
            # load image-parse-agnostic
            image_parse_agnostic = Image.open(osp.join(self.data_path, parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
            image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
            parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
            image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

            parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
            parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
            new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    new_parse_agnostic_map[i] += parse_agnostic_map[label]        
        else:
            new_parse_agnostic_map = None

        # load pose points
        pose_name = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')
        pose_map = Image.open(osp.join(self.data_path, pose_name))
        pose_map = transforms.Resize(self.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]
        
        # pose name
        pose_name = im_name.replace('image', 'openpose_json').replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load densepose
        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(osp.join(self.data_path, densepose_name))
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)  # [-1,1]

        # agnostic
        agnostic, hand_mask = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)
        # agnostic = agnostic * (1-pcm) + pcm * torch.zeros_like(agnostic)
        # agnostic = agnostic * (1-pcm) + pcm * 0.5 * torch.ones_like(agnostic)
        agnostic = agnostic * (1-pcm) + pcm * torch.ones_like(agnostic) * 0.00392163
        
        # parse other
        hand_mask = Image.fromarray(np.uint8(hand_mask), 'L')
        hand_mask = transforms.Resize(self.fine_width, interpolation=0)(hand_mask)

        parse_other = new_parse_map[1] + new_parse_map[2] + new_parse_map[4] + new_parse_map[7] + new_parse_map[8] + \
                            new_parse_map[9] + new_parse_map[10] + new_parse_map[11]+ new_parse_map[12]
        parse_other = parse_other + torch.tensor(np.array(hand_mask))
        parse_other = parse_other.clamp(0,1)
        im_other = im * parse_other
        parse_upper_mask = (1 - parse_other).unsqueeze(0)


        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name.split('/')[1],    # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
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
            # visualization & GT
            'image':    im,         # for visualization
            'head': im_h,  # for conditioning and visualization
            # mine
            'original_pixel_values':im_c, 
            'edited_pixel_values':im,
            'high_frequency_map':high_frequency_map,
            'dino_c':dino_c,
            }

        return result

    def __len__(self):
        return len(self.im_names)
    
class CPDataLoader(object):
    def __init__(self, batch_size, workers, shuffle, dataset, collate_fn=None):
        super(CPDataLoader, self).__init__()

        if shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            # train_sampler = torch.utils.data.SequentialSampler(dataset)
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=(train_sampler is None),
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

    config = Config.fromfile('config.py')
    opt = copy.deepcopy(config)
    opt.datamode = config.infer_datamode
    opt.data_list = config.infer_data_list
    opt.datasetting = config.infer_datasetting

    class UnNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            """
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be unnormalized.
            Returns:
                Tensor: UnNormalized image.
            """
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # 以下代码与上面等效，但更易读
                # t = t * s + m
            return tensor

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

    dataset = CPDataset(opt)

    p = dataset[8]
    print(p['c_name'], p['im_name'])
    agnostic = p['agnostic'].permute(1,2,0).numpy()
    agnostic = agnostic 
    agnostic *=255
    agnostic = agnostic.astype(np.uint8)
    agnostic= Image.fromarray(agnostic)
    agnostic.save('agnostic.jpg')

    image = p['image'].permute(1,2,0).numpy()
    image = image 
    image *=255
    image = image.astype(np.uint8)
    image= Image.fromarray(image)
    image.save('image.jpg')

    parse_upper_mask = p['parse_upper_mask'].permute(1,2,0).numpy()
    parse_upper_mask *=255
    parse_upper_mask = parse_upper_mask.astype(np.uint8)
    parse_upper_mask = parse_upper_mask[:,:,0]
    parse_upper_mask= Image.fromarray(parse_upper_mask)
    parse_upper_mask.save('parse_upper_mask.jpg')

    parse_cloth = p['pcm'].permute(1,2,0).numpy()
    parse_cloth *=255
    parse_cloth = parse_cloth.astype(np.uint8)
    parse_cloth = parse_cloth[:,:,0]
    parse_cloth= Image.fromarray(parse_cloth)
    parse_cloth.save('pcm.jpg')
        
    cloth = p['cloth']['unpaired'].permute(1,2,0).numpy()
    cloth *=255
    cloth = cloth.astype(np.uint8)
    cloth= Image.fromarray(cloth)
    cloth.save('cloth.jpg')

    cloth_mask = p['cloth_mask']['unpaired'].permute(1,2,0).numpy()
    cloth_mask *=255
    cloth_mask = cloth_mask.astype(np.uint8)
    cloth_mask = cloth_mask[:,:,0]
    cloth_mask= Image.fromarray(cloth_mask)
    cloth_mask.save('cloth_mask.jpg')

    unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    high_frequency_map = p['high_frequency_map']['unpaired']
    high_frequency_map = unnormalize(high_frequency_map)
    high_frequency_map = transforms.Resize((512,384))(high_frequency_map)
    high_frequency_map = high_frequency_map.permute(1,2,0).numpy()
    high_frequency_map *= 255
    high_frequency_map = Image.fromarray(high_frequency_map.astype(np.uint8))
    high_frequency_map.save('high_frequency_map.jpg')

    pose = p['pose'].permute(1,2,0).numpy()
    pose *=255
    pose = pose.astype(np.uint8)
    pose= Image.fromarray(pose)
    pose.save('pose.jpg')
    
    densepose = p['densepose'].permute(1,2,0).numpy()
    densepose *=255
    densepose = densepose.astype(np.uint8)
    densepose= Image.fromarray(densepose)
    densepose.save('densepose.jpg')
    
    parse_agnostic = p['parse_agnostic'].unsqueeze(0)
    parse_agnostic = visualize_segmap(parse_agnostic,tensor_out=False)
    parse_agnostic.convert('RGB').save('parse_agnostic.jpg')

    parse = p['parse'].unsqueeze(0)
    parse = visualize_segmap(parse,tensor_out=False)
    parse.convert('RGB').save('parse.jpg')

    parse_other = p['parse_other'].permute(1,2,0).numpy()
    parse_other = parse_other / 2 + 0.5
    parse_other *=255
    parse_other = parse_other.astype(np.uint8)
    parse_other= Image.fromarray(parse_other)
    parse_other.save('parse_other.jpg')

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
    