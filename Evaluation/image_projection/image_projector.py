import torch
from torch import nn
from torchvision import transforms
import numpy as np
import math

#import sys
#sys.path.append('/home/code-base/user_space/beacon') # To use the beacon code
from Evaluation.image_projection import LBFGS
from Evaluation.image_projection import project

# Image transformation function for current StyleGAN2 design
img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def Get_Avg_W_as_Latent(generator, device, per_layer_W, is_generator_DPmodule=False):
    '''
    Usage:
        Return the average latent_W of a generator based on 1000 samples estimation
    
    Args:
        generator:   (nn.Module) a generator placed on a single device without need of nn.DataParallel()
        device:      (str or torch.device()) the device to place the tensor 
        per_layer_W: (bool) whether we would like to return a tensor of shape [1,num_layer,512] or [1,512]
    '''
    
    num_sample = 1000
    latent_dim = 512
    num_layer = generator.module.num_layers + 1 if is_generator_DPmodule else generator.num_layers + 1
    
    with torch.no_grad():
        noise_z = torch.randn(num_sample, latent_dim)
        noise_z = noise_z.to(device)
        if is_generator_DPmodule:
            style_DP_module = nn.DataParallel(generator.module.style, device_ids=generator.device_ids) 
            latent_W = style_DP_module(noise_z)
        else:
            latent_W = generator.style(noise_z)
        avg_W = torch.mean(latent_W, axis=0)
        if per_layer_W is True:
            avg_W = avg_W.repeat((num_layer, 1)).unsqueeze(0)
        else:
            avg_W = avg_W.reshape(1,-1)
    
    return avg_W

def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad



def Image_Projector(generator, device, per_layer_W, target_im, opt, num_iters=500, print_iters=20, criterion=None):
    '''
    Usage:
        Project an target image with a StyleGAN2 model
    
    Args:
        generator:   (nn.Module) a generator placed on a single device without need of nn.DataParallel()
        device:      (str or torch.device()) the device to place the tensor
        per_layer_W: (bool) whether we would like to return a tensor of shape [1,num_layer,512] or [1,512]
        target_im:   (PIL.Image) or (list of PIL.Image) the target image we want to project
        opt:         (str) the optimizer used for projection
        num_iters:   (int) number of optimization iterations
        print_iters: (int) loss printing frequency 
        criterion:   (nn.Module) loss criterion of the image projection
    '''
    
    set_requires_grad(generator, False)
    generator.eval()
    is_generator_DPmodule = ('module' in list(generator.state_dict().keys())[0])

    if isinstance(target_im, list):    # Image Batch
        target_im_tensor_list = [img_transform(im) for im in target_im]
        target_im_tensor = torch.stack(target_im_tensor_list).to(device)

    else:   # Single Image
        target_im_tensor = img_transform(target_im)
        target_im_tensor = target_im_tensor.reshape((1,) + tuple(target_im_tensor.shape)).to(device) # transform the tensor from 3D to 4D
    
    # Average W
    avg_W = Get_Avg_W_as_Latent(generator, device, per_layer_W, is_generator_DPmodule)
    avg_W = torch.repeat_interleave(avg_W, target_im_tensor.shape[0], dim=0) # Copy Avg_W to have the same batch size as image batch
    avg_W.requires_grad = True
    
    # Noises
    if is_generator_DPmodule:
        noises = generator.module.make_noise()
    else:
        noises = generator.make_noise()
    for noise in noises:
        noise.requires_grad = True
            
    # Input Params
    input_kwargs = {'noise_z': None, 'input_is_latent': True, 'latent_styles' : [avg_W], 'noise':noises}
    input_params = [avg_W] + noises
    
    # Input Image
    input_img = generator(**input_kwargs)
    
    # Target Params
    target_kwargs = {'target':target_im_tensor, 'mask': None}
    
    # Losses for optimization
    if criterion is None:
        loss = 'mse+lpips'
        criterion = project.ImageReconstructionLoss(device=device, loss=loss)
    
    # Optimization 
    if opt == 'LBFGS':
        optimizer = LBFGS.FullBatchLBFGS(input_params, lr=1)
    elif opt == 'Adam':
        optimizer = torch.optim.Adam(input_params, lr=0.01)
    
    project.optimize(model=generator,
                 input_kwargs=input_kwargs,
                 targets=target_kwargs,
                 criterion=criterion,
                 optimizer=optimizer,
                 iterations=num_iters,
                 print_iterations=print_iters,
                 device=device)

    del criterion
    del optimizer.state
    del optimizer
    
    # Output Image
    output_img = generator(**input_kwargs).detach().cpu()
    input_img = input_img.detach().cpu()
    del avg_W 
    
    return (input_img, output_img)

# ---------------------------------------- Image Projection Evaluation ----------------------------------------

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def Downsample_Image_256(im_tensor):
    PERCEPTUAL_SIZE = 256
    while im_tensor.shape[2] > PERCEPTUAL_SIZE:
        im_tensor = torch.nn.functional.interpolate(im_tensor, scale_factor=1/2, mode='bilinear', align_corners=False)
    return im_tensor

def Get_LPIPS_Model_Image(output_img_tensor_list, target_img_tensor_list, lpips_percept):
    '''
    Usage:
        Obtain the LPIPS each pair of output image and target image
    
    Args:
        output_img_tensor_list: (list) of (torch.Tensor) each tensor is a [1, C, H, W] output image
        target_img_tensor_list: (list) of (torch.Tensor) each tensor is a [1, C, H, W] target image
    '''
    
    num_model = len(output_img_tensor_list) // len(target_img_tensor_list)
    assert len(output_img_tensor_list) % len(target_img_tensor_list) == 0
    num_img = len(target_img_tensor_list)
    
    lpips_score_list = []
    with torch.no_grad():
        for i in range(num_model):
            model_lpips_score_list = []
            for j in range(num_img):
                lpips_score = lpips_percept(Downsample_Image_256(target_img_tensor_list[j]), 
                                       Downsample_Image_256(output_img_tensor_list[j + num_img * i]))
                model_lpips_score_list.append(float(lpips_score))
            lpips_score_list.append(model_lpips_score_list)
    
    return lpips_score_list


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def Get_PSNR_Model_Image(output_img_list, target_img_list):
    '''
    Usage:
        Obtain the PSNR each pair of output image and target image
    
    Args:
        output_img_list: (list) of (np.array) each element is a [1, C, H, W] output image
        target_img_list: (list) of (np.array) each element is a [1, C, H, W] target image
    '''
    
    num_model = len(output_img_list) // len(target_img_list)
    assert len(output_img_list) % len(target_img_list) == 0
    num_img = len(target_img_list)
    
    psnr_score_list = []
    with torch.no_grad():
        for i in range(num_model):
            model_psnr_score_list = []
            for j in range(num_img):
                psnr_score = psnr(target_img_list[j], output_img_list[j + num_img * i])
                model_psnr_score_list.append(float(psnr_score))
            psnr_score_list.append(model_psnr_score_list)
    
    return psnr_score_list
