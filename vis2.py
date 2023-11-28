import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('./mae')

from mae.util.decoder.utils import tensor_normalize, spatial_sampling
import random
# seed 
seed = 4
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)


def plot_input(tensor):
    tensor = tensor.float()
    f, ax = plt.subplots(nrows=tensor.shape[0], ncols=tensor.shape[1], figsize=(50, 20))

    tensor = tensor * torch.tensor(STD).view(3, 1, 1)
    tensor = tensor + torch.tensor(MEAN).view(3, 1, 1)
    tensor = torch.clip(tensor * 255, 0, 255).int()

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            ax[i][j].axis("off")
            ax[i][j].imshow(tensor[i][j].permute(1, 2, 0))
    plt.show()


from mae.models_mae import mae_vit_large_patch16
model = mae_vit_large_patch16(decoder_embed_dim=512, decoder_depth=4, mask_type="st", t_patch_size=2, img_size=224)

checkpoint = torch.load("./video-mae-100x4-joint.pth", map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)
model = model.cuda().to(dtype=torch.float16)

# load data into tensor
file_path = 'mae_test/07148_00_smooth'
frames = []
frame_names = sorted(os.listdir(file_path))
for franme_name in frame_names:
    if franme_name[-3:] == 'gif':
        continue
    frame_path = os.path.join(file_path, franme_name)
    frame = Image.open(frame_path)
    frame = torch.tensor(np.array(frame))
    frames.append(frame)
frames = torch.stack(frames)
print(frames.shape)
frames = tensor_normalize(
    frames, 
    torch.tensor(MEAN), 
    torch.tensor(STD),
).permute(3, 0, 1, 2)
print(frames.shape)
frames = spatial_sampling(
    frames,
    spatial_idx=1,
    min_scale=256,
    max_scale=256,
    crop_size=224,
    random_horizontal_flip=False,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
)
print(frames.shape)
frames = frames.cuda().to(dtype=torch.float16)
for ratio in [0.3, 0.5, 0.7, 0.9]:
    loss, _, _, vis = model(frames.unsqueeze(0), 1, mask_ratio=ratio, visualize=True)
    vis = vis.detach().cpu()
    print(ratio, loss)
plot_input(vis[0].permute(0, 2, 1, 3, 4))