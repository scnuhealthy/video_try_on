# data
dataroot = '/data1/hzj/zalando-hd-resized'
dataroot2 = '/data1/hzj/DressCode/upper_body/'
dataroot3 = '/data1/hzj/192_256/'
fine_height = 512
fine_width = 384
with_one_hot = False
with_parse_agnostic = True
with_agnostic = True
semantic_nc = 13

# infer 
# model_path = 'HR_VITON_LaDIVTON_small_model_sr/'
model_path = '/data0/hzj/sd_models/df-1.5'
unet_path = '/data1/hzj/agnostic_norm_hair_have_background_hog/checkpoint-120000'
# unet_path = '/data1/hzj/agnostic_norm_hair_have_background/checkpoint-50000/'
vae_path = '../virtual_try_on_code/save_models/HR_VITON_vae'
# vae_path = 'parse_other_norm_nobackground_vae/checkpoint-14000'
# out_dir = '/data1/hzj/zalando-hd-resized/train/low_resolution_image'
out_dir = 'test'

test_dataset = 'VITON'
infer_datamode = 'test'
infer_data_list = 'test_pairs.txt'
infer_datasetting = 'unpaired'

# train data
train_datamode = 'train'
train_data_list = 'train_pairs.txt'
train_datasetting = 'paired'

# train
# pretrained_model_name_or_path = 'save_models/HR_VITON_LaDIVTON'
pretrained_model_name_or_path = '/data0/hzj/sd_models/df-1.5'
# output_dir = 'parse_other_norm_nobackground'
output_dir = '/data1/hzj/agnostic_norm_hair_have_background_hog' 
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
train_batch_size = 16
max_train_steps = None
num_train_epochs = 500
gradient_accumulation_steps = 1
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
checkpointing_steps = 30000
checkpoints_total_limit = None
resume_from_checkpoint = 'latest'
# resume_from_checkpoint = None
enable_xformers_memory_efficient_attention = True
