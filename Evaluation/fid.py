import argparse
import pickle as pkl
import time

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from .calc_inception import load_patched_inception_v3

from pathlib import Path
file_path = Path(__file__).parent

INCEPTION_FFHQ_256_FILE = str((file_path / '''./inception_ffhq_embed/self_ffhq_256_inception_embeddings_eval_mode.pkl''').resolve())
INCEPTION_FFHQ_1024_FILE = str((file_path / '''./inception_ffhq_embed/self_ffhq_1024_inception_embeddings_eval_mode.pkl''').resolve())

def extract_feature_from_samples(generator, inception, truncation, truncation_latent, batch_size, n_sample, device, info_print = False):
    with torch.no_grad():
        generator.eval()
        inception.eval()
        n_batch = n_sample // batch_size
        resid = n_sample - (n_batch - 1) * batch_size
        batch_sizes = [batch_size] * (n_batch - 1) + [resid]
        features = []

        for idx, batch in enumerate(batch_sizes):
            if info_print:
                print('Processing Batch: ' + str(idx))
            latent = torch.randn(batch, 512, device=device)
            img = generator([latent], truncation=truncation, truncation_latent=truncation_latent)
            feat = inception(img)[0].view(img.shape[0], -1)
            features.append(feat.to('cpu'))

        features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


def Get_Model_FID_Score(generator, load_inception_net = True, device = 'cuda', gpu_device_ids = None,
                        truncation = 1, mean_latent = None, batch_size = 100, num_sample = 50000, info_print = False):
    '''
    Usage:
        To get the FID score of a final trained generator in a one-button wrapper
    
    Args:
        generator: (nn.Module) of a generator network
        load_inception_net: (bool) whether to load the inception network or not
        truncation: (float) >= 1 means ???
        mean_latent: (???) ???
        batch_size: (int) size of the minibatch to generate one image
        num_sample: (int) number of sample to be generated
        info_print: (bool) whether to print the process or not
    '''
    
    # Preload inception statistics
    if 'module' in list(generator.state_dict().keys())[0]:
            img_size = generator.module.size
    else:
            img_size = generator.size

    if img_size == 256:
        inception_ffhq_stats = pkl.load(open(INCEPTION_FFHQ_256_FILE,'rb'))
        print('Using prestored stats: ' +  INCEPTION_FFHQ_256_FILE)
    elif img_size == 1024:        
        inception_ffhq_stats = pkl.load(open(INCEPTION_FFHQ_1024_FILE,'rb'))
        print('Using prestored stats: ' +  INCEPTION_FFHQ_1024_FILE)
    else:
        raise ValueError('Image Size is Invalid!')
    
    # Preload inception module
    if load_inception_net:
        inception = load_patched_inception_v3().to(device)
        if gpu_device_ids is None:
            inception = nn.DataParallel(inception)
        else:
            inception = nn.DataParallel(inception, device_ids = gpu_device_ids)
        inception.eval()
    
    # Get the features
    start_time = time.time()
    features = extract_feature_from_samples(generator, inception, truncation, mean_latent, 
                                            batch_size, num_sample, device, info_print).numpy()
    end_time = time.time()
    if info_print:
        print('')
        print('Total time to get the features: ' + str(round(end_time - start_time, 2)))
        print('Feature shapes: ' + str(features.shape))
    
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)
    real_mean,real_cov = inception_ffhq_stats['mean'], inception_ffhq_stats['cov']
    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    return fid



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--inception', type=str, default=None, required=True)
    parser.add_argument('ckpt', metavar='CHECKPOINT')

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)

    g = Generator(args.size, 512, 8).to(device)
    g.load_state_dict(ckpt['g_ema'])
    g = nn.DataParallel(g)
    g.eval()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    features = extract_feature_from_samples(
        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    ).numpy()
    print(f'extracted {features.shape[0]} features')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, 'rb') as f:
        embeds = pkl.load(f)
        real_mean = embeds['mean']
        real_cov = embeds['cov']

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print('fid:', fid)
