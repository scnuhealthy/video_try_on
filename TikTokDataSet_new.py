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
from utils import get_high_frequency_map
import random

class TikTokDataSet(data.Dataset):
    """
        Dataset for CP-VTON.
    """
    def __init__(self, opt, level='image'):
        super(TikTokDataSet, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot4
        self.datamode = opt.datamode # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        # self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.data_path = self.root
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize([0.5], [0.5]),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dino_transform = transforms.Compose([
            transforms.Resize((1022,756), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot4, self.data_list), 'r') as f:
            for line in f.readlines()[96:]:
            # for line in f.readlines():
                c_name, im_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names
        self.level = level
        if self.level == 'video':
            self.group_frames()
        # elif self.level == 'image':
        #     self.group_frames()
        #     self.set_group(0)
        self.f = open('wrong_id.txt', 'w')
        self.is_atr = False

    def merge_dict(self, dict_list):
        batch = {}
        for key in dict_list[0].keys():
            if key in ['c_name', 'im_name']:
                continue
            key_list = [dict_list[i][key] for i in range(len(dict_list))]
            batch[key] = torch.stack(key_list)
        batch['c_name'] = dict_list[0]['c_name']
        batch['im_name'] = dict_list[0]['im_name']
        return batch
            
    def getitem_video(self, index, length=16):
        group = self.groups[index]
        begin_idx, end_idx = group
        # a = 100
        a = random.randint(begin_idx, end_idx - length)
        # print(begin_idx, end_idx, a)
        image_batchs = []
        for image_idx in range(a, a + length):
            image_batchs.append(self.getitem_image(image_idx))
        batch = self.merge_dict(image_batchs)
        return batch

    def set_group(self, group_idx):
        group = self.groups[group_idx]
        print(group)
        self.im_names = self.im_names[group[0]:group[1]+1]
        self.c_names = self.c_names[group[0]:group[1]+1]

    def group_frames(self):
        last_c_name = None
        last_idx = 0
        groups = []
        with open(osp.join(self.root, self.data_list), 'r') as f:
            lines = f.readlines()
        for idx in range(len(lines)):
            line = lines[idx]
            c_name, im_name = line.strip().split()
            c_name = c_name.split('/')[1]
            if idx == 0:
                last_c_name = c_name
            elif c_name == last_c_name:
                pass
            else:
                groups.append((last_idx, idx-1))
                last_idx = idx 
                last_c_name = c_name
        groups.append((last_idx, idx))
        self.groups = groups

    def name(self):
        return "TikTokDataSet"

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

        # r = int(length_a / 16) + 1 
        r = int(length_a / 16)

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

    def getitem_image(self, index):
        im_name = self.im_names[index]
        c_name = self.c_names[index]
        # c_name = 'cloth/00130.png'
        file_name_noex = im_name.split('/')[-1][:-4]
        video_dir = im_name.split('/')[1]
        if not self.is_atr:
            parse_name = 'parse_lip/' + video_dir + '/' + file_name_noex +  '.png'
        else:
            parse_name = 'parse/' + video_dir + '/' + file_name_noex +  '.png'
        pose_json_name = 'openpose_json/' + video_dir + '/' + file_name_noex + '_keypoints.json'
        pose_name = 'openpose_img/' + video_dir + '/' + file_name_noex + '_rendered.png'

        c_name = osp.join(self.root, c_name)
        parse_name = osp.join(self.root, parse_name)
        pose_json_name = osp.join(self.root, pose_json_name)
        pose_name = osp.join(self.root, pose_name)

        # load_dino_fea
        # fea_name = c_name.replace('.jpg', '.pt').replace('img','dino_fea')
        name = c_name.split('/')[-1].replace('.png', '.pt')
        fea_name = osp.join(self.root, 'dino_fea', name)
        # print(fea_name)
        # c_name = '/data1/hzj/DressCode/upper_body/images/009328_1.jpg'
        #fea_name = '/data1/hzj/DressCode/upper_body/dino_fea/009328_1.pt'
        # c_name = '/root/autodl-tmp/zalando-hd-resized/test/cloth/07421_00.jpg'
        # fea_name = '/root/autodl-tmp/zalando-hd-resized/test/dino_fea/07421_00.pt'
        # dino_fea = torch.load(fea_name, map_location='cpu')

        # cloth
        c = Image.open(c_name).convert('RGB')
        dino_c = self.dino_transform(c)
        width, height = c.size
        target_ratio = 192 / 256
        new_height = int(width / target_ratio)
        left = 0
        top = (height - new_height) / 2
        right = width
        bottom = (height + new_height) / 2
        crop_area = (left, top, right, bottom)
        c = c.crop(crop_area)
        c = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(c)
        c = self.transform(c)  # [-1,1]

        # get high frequency map
        high_frequency_map = get_high_frequency_map(c_name)
        # high_frequency_map = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(high_frequency_map)
        # high_frequency_map = self.transform(high_frequency_map)  # [-1,1]
        high_frequency_map = self.dino_transform(high_frequency_map)  # [-1,1]

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # load parsing image
        # parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize((self.fine_height,self.fine_width), interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))

        # parse map
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

        # # background mask
        # background_mask = new_parse_map[0:1]
        # im = im * (1 - background_mask) + background_mask

        # parse_big = torch.from_numpy(np.array(im_parse_pil_big)[None]).long()
        # parse_map_big = torch.FloatTensor(20, im_pil_big.size[1], im_pil_big.size[0]).zero_()
        # parse_map_big = parse_map_big.scatter_(0, parse_big, 1.0)
        # # print(parse_map_big.shape)
        # background_mask_big = parse_map_big[0]

        # im_pil_big_tmp = torch.tensor(np.array(im_pil_big)).permute(2,0,1)
        # # print(im_pil_big_tmp.shape, background_mask_big.shape)
        # im_pil_big_tmp = im_pil_big_tmp * (1 - background_mask_big) + background_mask_big*255
        # im_pil_big_tmp = im_pil_big_tmp.permute(1,2,0).numpy().astype(np.uint8)
        # im_pil_big = Image.fromarray(im_pil_big_tmp)

        # parse agnostic v3.2
        trans_dict = {
            0:0, 1:1, 2:2,
            3:3,4:4,
            5:0, 6:0, 7:0, 
            8:8,9:9,10:10,11:11,
            12:12,13:13,
            14:0,15:0,
            16:16,17:17,18:18,19:19
        }
        parse_base = np.asarray(im_parse_pil_big, dtype=np.uint8)
        new_arr = np.full(parse_base.shape, 7)
        for old, new in trans_dict.items():
            new_arr = np.where(parse_base == old, new, new_arr)
        new_arr = Image.fromarray(new_arr.astype(np.uint8))
        image_parse_agnostic = transforms.Resize((self.fine_height,self.fine_width), interpolation=0)(new_arr)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]    

        # load pose points
        pose_map = Image.open(pose_name)
        pose_map = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]
        
        # pose name
        # h,w = im_pil_big.size
        with open(pose_json_name, 'r') as f:
            pose_label = json.load(f)
            if len(pose_label['people'])>0:
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
                # pose_data[1] = pose_data[1] * self.fine_height / h
                # pose_data[0] = pose_data[0] * self.fine_width / w
            else:
                self.f.write(pose_name)
                self.f.flush()
                print('111',pose_name)
                return self.getitem_image(index+1)

        # agnostic
        agnostic, hand_mask = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize((self.fine_height,self.fine_width), interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)
        agnostic = agnostic * (1-pcm) + pcm * torch.ones_like(agnostic) * 0.00392163
        # agnostic = agnostic * (1-pcm) + pcm * torch.zeros_like(agnostic)
        # pam = new_parse_map[5:6]
        # agnostic = agnostic * (1-pam) + pam * torch.zeros_like(agnostic)
        # pam = new_parse_map[6:7]
        # agnostic = agnostic * (1-pam) + pam * torch.zeros_like(agnostic)

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

        c = {'paired':c}
        high_frequency_map = {"paired":high_frequency_map}
        dino_c = {'paired':dino_c}
        c_name = {'paired':c_name}

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth':    c,          # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_map,       # for conditioning
            # generator input
            'agnostic' : agnostic,
            # GT
            'parse': new_parse_map, # GAN Loss real
            'pcm': pcm,             # L1 Loss & vis
            'parse_cloth': im_c,    # VGG Loss & vis
            'parse_other':im_other,
            'parse_upper_mask':parse_upper_mask,
            # visualization & GT
            'image':    im,         # for visualization
            'head': im_h,  # for conditioning and visualization
            'high_frequency_map':high_frequency_map,
            'dino_c':dino_c,
            }

        return result

    def __getitem__(self, index):
        if self.level == 'video':
            return self.getitem_video(index)
        else:
            return self.getitem_image(index)    

    def __len__(self):
        if self.level == 'video':
            return len(self.groups)
        else:
            return len(self.im_names)



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

    config = Config.fromfile('config.py')
    opt = copy.deepcopy(config)
    opt.datamode = config.train_datamode
    opt.data_list = config.train_data_list
    opt.datasetting = config.train_datasetting


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

    dataset = TikTokDataSet(opt,level='image')
    print(len(dataset))
    # for i in range(len(dataset)):
    #     p=dataset[i]
    sp = random.randint(0,len(dataset)-1)
    #sp = 6958
    #print('index',sp)
    p = dataset[sp]

    agnostic = p['agnostic'].permute(1,2,0).numpy()
    h,w,c = agnostic.shape
    print(agnostic[h//2,w//2,:])
    agnostic = agnostic / 2 + 0.5
    agnostic *=255
    agnostic = agnostic.astype(np.uint8)
    agnostic= Image.fromarray(agnostic)
    agnostic.save('agnostic.jpg')

    image = p['image'].permute(1,2,0).numpy()
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
    
    cloth = p['cloth']['paired'].permute(1,2,0).numpy()
    cloth *=255
    cloth = cloth.astype(np.uint8)
    cloth= Image.fromarray(cloth)
    cloth.save('cloth.jpg')

    high_frequency_map = p['high_frequency_map']['paired'].permute(1,2,0).numpy()
    print(high_frequency_map.shape)
    high_frequency_map *= 255
    high_frequency_map = Image.fromarray(high_frequency_map.astype(np.uint8))
    high_frequency_map.save('high_frequency_map.jpg')

    pose = p['pose'].permute(1,2,0).numpy()
    pose *=255
    pose = pose.astype(np.uint8)
    pose= Image.fromarray(pose)
    pose.save('pose.jpg')
    
    head = p['head'].permute(1,2,0).numpy()
    head *=255
    head = head.astype(np.uint8)
    head= Image.fromarray(head)
    head.save('head.jpg')

    parse = p['parse'].unsqueeze(0)
    parse = visualize_segmap(parse,tensor_out=False)
    parse.convert('RGB').save('parse.jpg')

    parse_cloth = p['parse_cloth'].permute(1,2,0).numpy()
    parse_cloth *=255
    parse_cloth = parse_cloth.astype(np.uint8)
    parse_cloth= Image.fromarray(parse_cloth)
    parse_cloth.save('parse_cloth.jpg')

    parse_agnostic = p['parse_agnostic'].unsqueeze(0)
    parse_agnostic = visualize_segmap(parse_agnostic,tensor_out=False)
    parse_agnostic.convert('RGB').save('parse_agnostic.jpg')

    parse_upper_mask = p['parse_upper_mask'].permute(1,2,0).numpy()
    parse_upper_mask *=255
    parse_upper_mask = parse_upper_mask.astype(np.uint8)
    parse_upper_mask = parse_upper_mask[:,:,0]
    parse_upper_mask= Image.fromarray(parse_upper_mask)
    parse_upper_mask.save('parse_upper_mask.jpg')