import torch
import torch.nn as nn
from PIL import Image
import numpy as np

import time
import argparse
from matplotlib import pyplot as plt

from Util.network_util import Build_Generator_From_Dict, Convert_Tensor_To_Image
from Evaluation.image_projection.image_projector import Image_Projector, im2tensor, Get_LPIPS_Model_Image, Get_PSNR_Model_Image
import lpips

device = 'cuda:0'
gpu_device_ids = [0,1]

# Arg Parsing

parser = argparse.ArgumentParser()

parser.add_argument('--generated_img_size', type=int, default=256)
parser.add_argument('--ckpt', type=str, default='''./Model/full_size_model/256px_full_size.pt''')
parser.add_argument('--image_file', type=str, default='''/image/file/path''')
parser.add_argument('--num_iters', type=int, default=800)
parser.add_argument('--info_print', action='store_true', default=False)

args = parser.parse_args()

# Model Preparation
model_dict = torch.load(args.ckpt, map_location=device)
g_ema = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size).to(device)
g_ema = nn.DataParallel(g_ema, device_ids=gpu_device_ids)
g_ema.eval();

# Image Preparation
target_img = Image.open(args.image_file).convert("RGB").resize((args.generated_img_size, args.generated_img_size))

# Optimization

if args.info_print:
    print_iters = 100
else:
    print_iters = np.inf
input_img_tensor, output_img_tensor = Image_Projector(generator = g_ema, 
                device = device, 
                per_layer_W = True, 
                target_im = target_img, 
                opt = 'LBFGS',
                num_iters = args.num_iters,
                print_iters = print_iters)



# PSNR, LPIPS

lpips_percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[int(device[-1])])
lpips_percept.eval();
target_img = np.array(target_img)
target_img_tensor = im2tensor(target_img)
lpips_score = Get_LPIPS_Model_Image([output_img_tensor], [target_img_tensor], lpips_percept)[0][0]
print('LPIPS Score: ' + str(round(lpips_score, 4)))

output_img = np.array(Convert_Tensor_To_Image(output_img_tensor))
psnr_score = Get_PSNR_Model_Image([output_img], [target_img])[0][0]
print('PSNR Score: ' + str(round(psnr_score, 4)))


# Image Visualization for real image and projected image

wspace = 0.1
hspace = 0.1
n_row = 1
n_col = 2 
title_size = 5

dpi = 500
figsize = (n_col,n_row)

plt.figure(dpi = dpi, figsize=figsize)

plt.subplot(1, 2, 1)
plt.imshow(target_img)
plt.axis('off')
plt.title('Real Image', fontsize = title_size)

plt.subplot(1, 2, 2)
plt.imshow(output_img)
plt.axis('off')
plt.title('Projected Image', fontsize = title_size)

plt.subplots_adjust(wspace=wspace, hspace=hspace)

plt.savefig('./Image_Projection_Visualization.png')

