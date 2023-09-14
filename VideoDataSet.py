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

class VideoDataSet(data.Dataset):
    """
        Dataset for CP-VTON.
    """
    def __init__(self, opt, random_pair=False,begin_idx=0):
        super(VideoDataSet, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot3
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
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot3, opt.data_list), 'r') as f:
            for line in f.readlines()[begin_idx:]:
                c_name, im_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names
        self.random_pair = random_pair

    def name(self):
        return "VideoDataSet"
    
    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
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

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            # mask_arm = Image.new('L', (768, 1024), 'white')
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

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic

    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = self.c_names[index]
        info = im_name.split('/')
        video_name, file_name = info[1], info[2]
        file_name_noex = file_name.split('.')[0]
        c_mask_name = c_name[:-4]+'_mask.jpg'
        parse_name = f'{self.datamode}_frames_parsing' + '/' + video_name + '/' + file_name_noex +  '_label.png'
        pose_json_name = f'{self.datamode}_openpose_json' + '/' + video_name + '/' +  file_name_noex + '_keypoints.json'
        pose_name = f'{self.datamode}_openpose_img' + '/' + video_name + '/' + file_name_noex + '_rendered.png'
        densepose_name = f'{self.datamode}_densepose' + '/' + video_name + '/' + file_name_noex + '.png'

        # previous info
        file_name_info = file_name_noex.split('_')
        # pre_inter = random.randint(1,4)
        pre_inter = 1
        file_name_noex_previous = file_name_info[0] + '_'+ file_name_info[1] + '_'+  str(int(file_name_info[2])-pre_inter).zfill(3)
        previous_im_name = f'{self.datamode}_frames' + '/' + video_name + '/' + file_name_noex_previous + '.png'
        previous_pose_name = f'{self.datamode}_openpose_img' + '/' + video_name + '/' + file_name_noex_previous + '_rendered.png'
        previous_parse_name = f'{self.datamode}_frames_parsing' + '/' + video_name + '/' + file_name_noex_previous +  '_label.png'
        previous_im_name = osp.join(self.root, previous_im_name)
        previous_pose_name = osp.join(self.root, previous_pose_name)
        previous_parse_name = osp.join(self.root, previous_parse_name)

        c_name = osp.join(self.root, c_name)
        c_mask_name = osp.join(self.root, c_mask_name)
        parse_name = osp.join(self.root, parse_name)
        pose_json_name = osp.join(self.root, pose_json_name)
        pose_name = osp.join(self.root, pose_name)
        densepose_name = osp.join(self.root, densepose_name)

        # cloth
        c = Image.open(c_name).convert('RGB')
        c = transforms.Resize(self.fine_width, interpolation=2)(c)
        cm = Image.open(c_mask_name)
        cm = transforms.Resize(self.fine_width, interpolation=0)(cm)
        c = self.transform(c)  # [-1,1]
        cm = np.array(cm)
        cm = (cm >= 128).astype(np.float32)
        cm = torch.from_numpy(cm)  # [0,1]
        cm.unsqueeze_(0)

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # load parsing image
        # parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
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
        image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(new_arr)
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
        pose_map = transforms.Resize(self.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]
        
        # pose name
        with open(pose_json_name, 'r') as f:
            pose_label = json.load(f)
            if len(pose_label['people'])>0:
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
            else:
                print(pose_name)
                return self.__getitem__(index+300)

        # load densepose
        densepose_map = Image.open(densepose_name)
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)  # [-1,1]

        if self.opt.with_agnostic:
            # agnostic
            agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
            agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
            agnostic = self.transform(agnostic)
        else:
            agnostic = None


        # previous info
        # load pose points
        previous_pose_map = Image.open(previous_pose_name)
        previous_pose_map = transforms.Resize(self.fine_width, interpolation=2)(previous_pose_map)
        previous_pose_map = self.transform(previous_pose_map)  # [-1,1]

        # person image
        previous_im_pil_big = Image.open(osp.join(self.data_path, previous_im_name))
        previous_im_pil = transforms.Resize(self.fine_width, interpolation=2)(previous_im_pil_big)
        previous_im = self.transform(previous_im_pil)

        # load parsing image
        # parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))

        # parse map
        previous_parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        previous_parse_map = previous_parse_map.scatter_(0, parse, 1.0)
        new_previous_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_previous_parse_map[i] += previous_parse_map[label]

        # parse cloth & parse cloth mask
        previous_pcm = new_previous_parse_map[3:4]
        previous_im_c = previous_im * previous_pcm + (1 - previous_pcm)

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
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
            # visualization & GT
            'image':    im,         # for visualization
            'head': im_h,  # for conditioning and visualization
            # previous info
            'previous_pose': previous_pose_map,
            'previous_image': previous_im,
            'previous_parse_cloth':previous_im_c
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
    opt.datamode = config.infer_datamode
    opt.data_list = config.infer_data_list
    opt.datasetting = config.infer_datasetting


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

    dataset = VideoDataSet(opt)

    p = dataset[0]

    agnostic = p['agnostic'].permute(1,2,0).numpy()
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
    
    cloth = p['cloth'].permute(1,2,0).numpy()
    cloth *=255
    cloth = cloth.astype(np.uint8)
    cloth= Image.fromarray(cloth)
    cloth.save('cloth.jpg')

    cloth_mask = p['cloth_mask'].permute(1,2,0).numpy()
    cloth_mask *=255
    cloth_mask = cloth_mask.astype(np.uint8)
    cloth_mask = cloth_mask[:,:,0]
    cloth_mask= Image.fromarray(cloth_mask)
    cloth_mask.save('cloth_mask.jpg')

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

    parse_agnostic = p['parse_agnostic'].unsqueeze(0)
    parse_agnostic = visualize_segmap(parse_agnostic,tensor_out=False)
    parse_agnostic.convert('RGB').save('parse_agnostic.jpg')

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