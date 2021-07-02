import argparse

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import lpips
from model import Generator


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t



def Generate_Interpolated_Image(generator, batch_size, eps, device, latent_dim, gpu_device_ids = None):
    '''
    Usage:
        To generate a bunch of images that are interpolated in W space for Perceptual Path Length evaluation
    
    Args:
        generator:      (nn.Module) the StyleGAN generator
        batch_size:     (int) size of the batch
        eps:            (float) the interpolation distance
        device:         (str) device to put tensor the primary gpu
        latent_dim:     (int) dimension of the latent noise z
        gpu_device_ids: (list) of the gpu indices for DataParallel
    '''
    
    if 'module' in list(generator.state_dict().keys())[0]:
        g_module = generator.module
    else:
        g_module = generator  
        
    with torch.no_grad():

        noise_z = torch.randn([batch_size * 2, latent_dim], device=device)
        lerp_t = torch.rand(batch_size, device=device)

        if gpu_device_ids is None:
            style_module = nn.DataParallel(g_module.style)
        else:
            style_module = nn.DataParallel(g_module.style, device_ids=gpu_device_ids)
            
        latent = style_module(noise_z)
        latent_t0, latent_t1 = latent[::2], latent[1::2]
        latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
        latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + eps)
        latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)
        
        image = generator(noise_z = None, latent_styles = [latent_e], input_is_latent=True, noise=None)
    
    return image



def Get_PPL_Score(generator, n_sample, batch_size, eps, latent_dim, device, gpu_device_ids, info_print = False):
    '''
    Usage:
        A unified function to get the PPL score of a model
        
    Args:
        generator:      (nn.Module) of the generator
        n_sample:       (int) # of generated image pairs
        batch_size:     (int) size of the minibatch for inference
        eps:            (float) the pertubation in latent codes
        latent_dim:     (int) the dimension of the noise_z for generator's mapping network
        device:         (str) primary gpu to place the Tensor
        gpu_device_ids: (list) of (int) of the gpu devices to parallel on
        info_print:     (bool) whether or not to print the evaluation information
    '''
    
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch - 1) * batch_size
    batch_sizes = [batch_size]*(n_batch - 1) + [resid]

    distances = []
    with torch.no_grad():
       
        # Default initialization of the perceptual loss
        percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[gpu_device_ids[0]])

        for idx, batch in enumerate(batch_sizes):
            if info_print:
                print('Evaluating Batch: ' + str(idx))
                
            image = Generate_Interpolated_Image(generator, batch_size = batch_size, eps = eps, 
                                        device = device, latent_dim = latent_dim, gpu_device_ids=gpu_device_ids)      

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode='bilinear', align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) 
            distances.append(dist.to('cpu').numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )
    
    ppl = filtered_dist.mean()
    del percept
    
    return ppl




if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--space', choices=['z', 'w'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=5000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('ckpt', metavar='CHECKPOINT')

    args = parser.parse_args()

    latent_dim = 512

    ckpt = torch.load(args.ckpt)

    g = Generator(args.size, latent_dim, 8).to(device)
    g.load_state_dict(ckpt['g_ema'])
    g.eval()

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    )

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]

    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            noise = g.make_noise()

            inputs = torch.randn([batch * 2, latent_dim], device=device)
            lerp_t = torch.rand(batch, device=device)

            if args.space == 'w':
                latent = g.get_latent(inputs)
                latent_t0, latent_t1 = latent[::2], latent[1::2]
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            image, _ = g([latent_e], input_is_latent=True, noise=noise)

            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode='bilinear', align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to('cpu').numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print('ppl:', filtered_dist.mean())
