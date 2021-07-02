# Constant
L1_KNOWLEDGE_DISTILLATION_MODE = ['Output_Only', 'Intermediate']
PERCEPT_KNOWLEDGE_DISTILLATION_MODE = ['LPIPS', 'VGG']
GENERATOR_SHAPE_256PX = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128]
PRUNING_MODE = ['Global_Number', 'Layer_Uniform']
PERCEPT_LOSS_IMAGE_SIZE = 256

# Params
data_folder = '/home/code-base/user_space/Dataset/FFHQ_Amazon/'
generated_img_size = 256
channel_multiplier = 2
latent = 512
n_mlp = 8
ckpt = './Model_StyleGAN2/550000.pt'
load_train_state = False

gpu_device_ids = [0,1]
primary_device = 'cuda:0'

training_iters = 200001
batch_size = 16
init_lr = 0.002
discriminator_r1 = 10
generator_path_reg_weight = 2
path_reg_batch_shrink = 2
g_reg_freq = 4
d_reg_freq = 16
noise_mixing = 0.9

sparsity_eta = 1e-5
init_step = 0
lay_rmve_ratio = 0.1
num_rmve_channel = 588
model_prune_freq = 500000
prune_metric = 'l1-style'
pruning_mode = 'Global_Number'

val_sample_num = 9
val_sample_freq = 1000
model_save_freq = 10000
fid_n_sample = 50000
fid_batch = 64

teacher = './Model_StyleGAN2/550000.pt'
kd_l1_lambda = 0
kd_percept_lambda = 3
kd_l1_mode = L1_KNOWLEDGE_DISTILLATION_MODE[1]
kd_percept_mode = PERCEPT_KNOWLEDGE_DISTILLATION_MODE[1]
