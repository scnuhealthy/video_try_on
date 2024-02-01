## from anydoor

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

sys.path.append("./dinov2")
import hubconf
from omegaconf import OmegaConf
# DINOv2_weight_path = '/root/autodl-tmp/AnyDoor/trained_models/dinov2_vitg14_pretrain.pth'
DINOv2_weight_path = '/root/autodl-tmp/video_try_on/trained_models/dinov2_vitl14_pretrain.pth'

class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, freeze=True):
        super().__init__()
        # dinov2 = hubconf.dinov2_vitg14() 
        dinov2 = hubconf.dinov2_vitl14()
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2
        if freeze:
            self.freeze()

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        features = self.model.forward_features(image)
        tokens = features["x_norm_patchtokens"]
        image_features  = features["x_norm_clstoken"]
        image_features = image_features.unsqueeze(1)
        hint = torch.cat([image_features,tokens],1) # b,257,1536
        return hint

if __name__ == '__main__':
    p = FrozenDinoV2Encoder()
