import torch
from torch import nn
import numpy as np
from PIL import Image

from beacon_projector import set_requires_grad, img_transform, Get_Avg_W_as_Latent, project, LBFGS

def Image_Projector(generator, device, per_layer_W, target_im, opt, optimize_noise = True, fixed_noise = None,
                    num_iters=500, print_iters=20, criterion=None):
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
    if fixed_noise is not None:
        noises = fixed_noise
    else:
        if is_generator_DPmodule:
            noises = generator.module.make_noise()
        else:
            noises = generator.make_noise()
    
    # Noise optimization
    if optimize_noise: 
        for noise in noises:
            noise.requires_grad = True
            
    # Input Params
    input_kwargs = {'noise_z': None, 'input_is_latent': True, 'latent_styles' : [avg_W], 'noise':noises}
    if optimize_noise:
        input_params = [avg_W] + noises
    else:
        input_params = [avg_W]
    
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
    input_kwargs['latent_styles'][0] = input_kwargs['latent_styles'][0].detach()
    del avg_W 
    
    return output_img, input_kwargs

def Latent_Style_Mixing(img_latent, inject_index):
    '''
    Usage:
        Mix the latent style code 
    
    Args:
        img_latent:   (list) of 2 image latent codes, each image latent code is of shape [N_sample, N_layer, DIM]
        inject_index: (int) The layer index alternate the latent code
    '''
    
    mixed_latent = torch.zeros_like(img_latent[0])
    mixed_latent[:, :inject_index, :] = img_latent[0][:, :inject_index, :]
    mixed_latent[:, inject_index:, :] = img_latent[1][:, inject_index:, :]
    return mixed_latent


def Noise_Style_Mixing(noises, inject_index):
    '''
    Usage:
        Cross over operation for the noises 
    
    Args:
        noises:       (list) of 2 noises, each noise is of len N_layer - 1
        inject_index: (int) The layer index alternate the latent code
    '''
    
    # Would do a -1 here as the first layer doesn't have noises
    mixed_noise = noises[0][ :inject_index - 1] + noises[1][inject_index - 1:] 
    return mixed_noise