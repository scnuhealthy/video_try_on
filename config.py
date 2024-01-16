# data
dataroot = '/root/autodl-tmp/zalando-hd-resized'
dataroot2 = '/data1/hzj/DressCode/upper_body/'
dataroot3 = '/root/autodl-tmp/192_256/'
dataroot4 = '/root/autodl-tmp/TikTok_dataset2/'
vtt_data_list = '/data1/hzj/192_256/custom_test_pairs.txt'
fine_height = 512
fine_width = 384
with_one_hot = False
with_parse_agnostic = True
with_agnostic = True
semantic_nc = 13

# infer 
model_path = '/root/autodl-tmp/animate/pretrained_models/stable-diffusion-v1-5'  # basic stable diffusion

# # VITON 
# unet_path = 'trained_models/model_VITON_512_fixbug/checkpoint-120000'
# # unet_path = '/data0/hzj/anydoor/trained_models/agnostic_norm_hair_have_background/checkpoint-50000/'  # paper result
# vae_path = 'trained_models/HR_VITON_vae'
# test_dataset = 'VITON'
# infer_datamode = 'test'
# infer_data_list = 'test_pairs.txt'
# infer_datasetting = 'unpaired'
# out_dir = 'gen_test_VITON'

# # DressCode
# unet_path = '/data0/hzj/anydoor/trained_models/model_VITON_512_fixbug/checkpoint_120000'
# vae_path = '/data0/hzj/anydoor/trained_models/HR_VITON_vae'
# test_dataset = 'DressCode'
# infer_datamode = 'test'
# infer_data_list = 'test_pairs.txt'
# infer_datasetting = 'unpaired'
# out_dir = 'gen_test_DressCode'

# TikTok
# unet_path = 'trained_models/model_TikTok_512_fixbug_1109_lip/checkpoint-150000'  # tiktok model
# vae_path = 'trained_models/HR_VITON_vae'      # VITON vae
# out_dir = 'gen_test_TikTOk'
# test_dataset = 'TikTok'
# infer_datamode = 'test'
# infer_data_list = 'train_unpairs_sp_267_44.txt'
# infer_datasetting = 'unpaired'

# VTT
unet_path = 'trained_models/model_VTT_192_256_1030_fixbug/checkpoint-120000'  # tiktok model
vae_path = 'trained_models/HR_VITON_vae'      # VITON vae
output_root = 'gen_test_VTT'
test_dataset = 'VTT'
infer_datamode = 'test'
infer_data_list = 'test_pairs_sp.txt'
infer_datasetting = 'unpaired'
fine_height = 256
fine_width = 192

# # unet_path = 'model_TikTok_512_fixbug_1109_atr/checkpoint-60000'
# unet_path = 'model_TikTok_512_fixbug_1109_lip/checkpoint-150000'  # tiktok model
# # unet_path = 'model_VTT_192_256_1030_fixbug/checkpoint-80000'    # VVT model
# # unet_path = 'model_VTT_192_256_1023/checkpoint-62000'           # VVT model
# # unet_path = '/data1/hzj/agnostic_norm_hair_have_background/checkpoint-50000/'  # VITON-HD and DressCode model
# # vae_path = 'model_TikTok_vae_512_fixbug/checkpoint-4000'         # TikTok vae, not use
# vae_path = '../virtual_try_on_code/save_models/HR_VITON_vae'       # VITON vae
# # vae_path = 'model_VTT_vae/checkpoint-8000'                       # VTT vae, not use
# # vae_path = 'parse_other_norm_nobackground_vae/checkpoint-14000'
# out_dir = 'test_TikTok_video_demo/test1'

# train data
train_datamode = 'train'
train_data_list = 'train_pairs.txt'
# train_data_list = 'train_unpairs_sp_50_38.txt'
train_datasetting = 'paired'

# train
pretrained_model_name_or_path = '/data0/hzj/sd_models/df-1.5'
output_dir = 'model_TikTok_512_fixbug_1109_lip'
# output_dir = 'model_VTT_vae_fixhand'
revision = None
validation_prompt = None
num_validation_images = 1
validation_epoches = 1
max_train_samples = None
cache_dir = None
seed = 42
resolution = 256
center_crop = False
random_flip = True
train_batch_size = 4
max_train_steps = None
num_train_epochs = 1000
gradient_accumulation_steps = 2
gradient_checkpointing = True
learning_rate = 5e-05
scale_lr = False
lr_scheduler = 'constant'
lr_warmup_steps = 0
conditioning_dropout_prob = 0.00
use_8bit_adam = False
allow_tf32 = True
use_ema = False
non_ema_revision = None
dataloader_num_workers = 10
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08
max_grad_norm = 1.0
logging_dir = 'logs'
mixed_precision = 'fp16'
report_to = 'tensorboard'
local_rank = -1
checkpointing_steps = 15000
checkpoints_total_limit = None
resume_from_checkpoint = 'latest'
# resume_from_checkpoint = None
enable_xformers_memory_efficient_attention = True
