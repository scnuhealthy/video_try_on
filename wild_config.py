# data
dataroot = '/root/autodl-tmp/my_clothes_dataset/'
fine_height = 512
fine_width = 384
semantic_nc = 13
with_one_hot= False
is_atr = False

# infer 
# model_path = '/data0/hzj/sd_models/df-1.5'
model_path = 'runwayml/stable-diffusion-v1-5'
unet_path = '/root/autodl-tmp/agnostic_norm_hair_have_background/checkpoint-50000/'
vae_path = '/root/autodl-tmp/HR_VITON_vae'
out_dir = 'test_guide2'

test_dataset = 'Wild'
infer_datamode = 'test'
infer_data_list = 'test_pairs.txt'
infer_datasetting = 'unpaired'

# train data
train_datamode = 'train'
train_data_list = 'train_pairs.txt'
train_datasetting = 'paired'