import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import os

class ClothDataSet(data.Dataset):
    def __init__(self, root):
        files = os.listdir(root)
        self.files = []
        for onefile in files:
            #if onefile[-5] == '1':
            self.files.append(onefile)
        print(len(self.files))
        self.root = root
        self.transform = T.Compose([
            T.Resize((280,224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    def __getitem__(self,index):
        cloth_file = self.files[index]
        cloth_path = os.path.join(self.root, cloth_file)
        img = Image.open(cloth_path).convert('RGB')
        img = self.transform(img)
        return cloth_file, img

    def __len__(self):
        return len(self.files)

class CPDataLoader(object):
    def __init__(self, batch_size, workers, shuffle, dataset, collate_fn=None):
        super(CPDataLoader, self).__init__()

        if shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = torch.utils.data.SequentialSampler(dataset)

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True, drop_last=False, sampler=train_sampler,collate_fn=collate_fn)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

def visualize(features, h, w):
    # vis code from https://github.com/dichotomies/N3F/tree/master/feature_extractor
    dim = features.shape[-1]
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    all_features = features.cpu()
    pca = PCA(n_components=3)
    N, C, H, W = all_features.shape
    all_features = all_features.permute(0, 2, 3, 1).view(-1, C).numpy()
    pca_features = pca.fit_transform(all_features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min()) 
    pca_features = pca_features * 255
    pca_features = pca_features.reshape(h, w, 3)
    vis_img = Image.fromarray(pca_features.astype(np.uint8))
    vis_img.save('vis.jpg')


if  __name__ == '__main__':
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.cuda()

    # d = ClothDataSet('/data1/hzj/zalando-hd-resized/test/cloth/')
    # feature_save_root = '/data1/hzj/zalando-hd-resized/test/dino_fea/'
    # d = ClothDataSet('/data1/hzj/DressCode/upper_body/images')
    # feature_save_root = '/data1/hzj/DressCode/upper_body/dino_fea'
    d = ClothDataSet('/data1/hzj/TikTok_dataset2/cloth/')
    feature_save_root = '/data1/hzj/TikTok_dataset2/dino_fea/'
    if not os.path.exists(feature_save_root):
        os.mkdir(feature_save_root)
    dataloader = CPDataLoader(4,4,False,d)

    for i, batch in enumerate(dataloader.data_loader):
        names = batch[0]
        imgs_tenosr = batch[1]
        with torch.no_grad():
            features_dict = model.forward_features(imgs_tenosr.cuda())
        features_patchtokens = features_dict['x_norm_patchtokens']
        feature_clstoken = features_dict['x_norm_clstoken'].unsqueeze(1)
        features = torch.cat([feature_clstoken,features_patchtokens],1)
        ## save tensor
        for j in range(len(names)):
            save_feature = features[j]
            save_name = names[j][:-4] + '.pt'
            save_path = os.path.join(feature_save_root, save_name)
            torch.save(save_feature, save_path)
        ## visualize
        # h, w = int(imgs_tenosr.shape[2] / model.patch_embed.patch_size[0]), int(
        #     imgs_tenosr.shape[3] / model.patch_embed.patch_size[1]
        # )
        # visualize(features_patchtokens[1], h, w)
        
    # img = Image.open('/data1/hzj/zalando-hd-resized/train/cloth/00024_00.jpg')
    # transform = T.Compose([
    #     # T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    #     # T.CenterCrop(224),
    #     # T.Resize((574,448), interpolation=T.InterpolationMode.BICUBIC),
    #     T.Resize((280,224), interpolation=T.InterpolationMode.BICUBIC),
    #     T.ToTensor(),
    #     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ])

    # img = transform(img)[:3].unsqueeze(0)
    # img = img.cuda()
    
    # h, w = int(img.shape[2] / model.patch_embed.patch_size[0]), int(
    #     img.shape[3] / model.patch_embed.patch_size[1]
    # )
    # with torch.no_grad():
    #     # 将图像张量传递给dinov2_vits14模型获取特征
    #     features_dict = model.forward_features(img)
    #     features_patchtokens = features_dict['x_norm_patchtokens']
    #     feature_clstoken = features_dict['x_norm_clstoken'].unsqueeze(1)
    #     print(feature_clstoken.shape,features_patchtokens.shape)
    #     features = torch.cat([feature_clstoken,features_patchtokens],1)
    #     print(features.shape)

    # visualize(features_patchtokens, h, w)