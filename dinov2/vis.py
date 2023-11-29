import torch
from PIL import Image
import torchvision.transforms as T
import hubconf
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np

def visualize(features, model):
    # vis code from https://github.com/dichotomies/N3F/tree/master/feature_extractor
    h, w = int(img.shape[2] / model.patch_embed.patch_size[0]), int(
        img.shape[3] / model.patch_embed.patch_size[1]
    )
    dim = features.shape[-1]
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    print(features.shape)

    all_features = features.cpu()
    pca = PCA(n_components=3)
    N, C, H, W = all_features.shape
    all_features = all_features.permute(0, 2, 3, 1).view(-1, C).numpy()
    print("Features shape: ", all_features.shape)
    pca_features = pca.fit_transform(all_features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min()) 
    pca_features = pca_features * 255
    pca_features = pca_features.reshape(h, w, 3)
    print(pca_features.shape)
    vis_img = Image.fromarray(pca_features.astype(np.uint8))
    vis_img.save('vis.jpg')


model = hubconf.dinov2_vitl14()

img = Image.open('00012_00.jpg')
transform = T.Compose([
    # T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    # T.CenterCrop(224),
    # T.Resize((574,448), interpolation=T.InterpolationMode.BICUBIC),
    T.Resize((280,224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

img = transform(img)[:3].unsqueeze(0)
img = img.cuda()
model = model.cuda()
with torch.no_grad():
    # 将图像张量传递给dinov2_vits14模型获取特征
    features_dict = model.forward_features(img)
    features_patchtokens = features_dict['x_norm_patchtokens']
    feature_clstoken = features_dict['x_norm_clstoken'].unsqueeze(1)
    print(feature_clstoken.shape,features_patchtokens.shape)
    features = torch.cat([feature_clstoken,features_patchtokens],1)
    print(features.shape)

visualize(features_patchtokens, model)