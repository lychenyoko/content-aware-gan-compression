import argparse
import random
import os
import time
import datetime

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

from model import Generator, Discriminator
from dataset import FFHQ_Dataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from Util.network_util import Build_Generator_From_Dict

# Hyper-parameters for training!
import train_hyperparams

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default=train_hyperparams.data_folder)
parser.add_argument('--size', type=int, default=train_hyperparams.generated_img_size)
parser.add_argument('--ckpt', type=str, default=train_hyperparams.ckpt)
parser.add_argument('--channel_multiplier', type=int, default=train_hyperparams.channel_multiplier)
parser.add_argument('--load_train_state', type=bool, default=train_hyperparams.load_train_state)

parser.add_argument('--iter', type=int, default=train_hyperparams.training_iters)
parser.add_argument('--batch', type=int, default=train_hyperparams.batch_size)
parser.add_argument('--lr', type=float, default=train_hyperparams.init_lr)
parser.add_argument('--r1', type=float, default=train_hyperparams.discriminator_r1)
parser.add_argument('--path_regularize', type=float, default=train_hyperparams.generator_path_reg_weight)
parser.add_argument('--path_batch_shrink', type=int, default=train_hyperparams.path_reg_batch_shrink)
parser.add_argument('--d_reg_every', type=int, default=train_hyperparams.d_reg_freq)
parser.add_argument('--g_reg_every', type=int, default=train_hyperparams.g_reg_freq)
parser.add_argument('--mixing', type=float, default=train_hyperparams.noise_mixing)

parser.add_argument('--n_sample', type=int, default=train_hyperparams.val_sample_num)
parser.add_argument('--val_sample_freq', type=int, default=train_hyperparams.val_sample_freq)
parser.add_argument('--model_save_freq', type=int, default=train_hyperparams.model_save_freq)

parser.add_argument('--teacher_ckpt', type=str, default=train_hyperparams.teacher)
parser.add_argument('--kd_lambda', type=float, default=train_hyperparams.kd_lambda)
parser.add_argument('--kd_mode', type=str, default=train_hyperparams.kd_mode)


args = parser.parse_args()
n_gpu = len(train_hyperparams.gpu_device_ids)
device = train_hyperparams.primary_device
args.distributed = n_gpu > 1
args.latent = 512
args.n_mlp = 8


def Print_Experiment_Status():
    '''
    Usage:
        To print out all the relevant status of 
    '''
    print('\n' + '--------------- Training Start ---------------')
    print('Params: ' + '\n\n' + 
          '  Model and Data: ' + '\n' +
          '    Data Folder: ' + str(args.path) + '\n' + 
          '    Multi-Layer Perceptron Num Layers: ' + str(args.n_mlp) + '\n' +
          '    Latent Variable Dimension: ' + str(args.latent) + '\n' +
          '    Generated Image Size: ' + str(args.size) + '\n' +
          '    Channel Multiplier: ' + str(args.channel_multiplier) + '\n' +
          '    Initial Checkpoint: ' + str(args.ckpt) + '\n' +
          '    Load Training State: ' + str(args.load_train_state) + '\n\n' + 

          '  GPU Setup: ' + '\n' +
          '    Distributed Training: ' + str(args.distributed) + '\n' +
          '    Primiary GPU Device: ' + device + '\n' + 
          '    GPU Device IDs: ' + str(train_hyperparams.gpu_device_ids) + '\n' +
          '    Number of GPUs: ' + str(n_gpu) + '\n\n' +

          '  Training Params: ' + '\n' +
          '    Training Iterations: ' + str(args.iter) + '\n' +
          '    Batch Size: ' + str(args.batch) + '\n' +
          '    Learning Rate: ' + str(args.lr) + '\n' +
          '    Generator Path Regularization Frequency: ' + str(args.g_reg_every) + '\n' +
          '    Path Regularization Weight: ' + str(args.path_regularize) + '\n' +  
          '    Path Batch Shrink Ratio: ' + str(args.path_batch_shrink) + '\n' +
          '    Discriminator Regularization Frequency: ' + str(args.d_reg_every) + '\n' +
          '    Discriminator Regularization Weight: ' + str(args.r1) + '\n' +
          '    Noise Mixing: ' + str(args.mixing) + '\n\n' +

          '  Validation Params: ' + '\n' + 
          '    Number of Validated Samples: ' + str(args.n_sample) + '\n' + 
          '    Generate Sample Frequency: ' + str(args.val_sample_freq) + '\n' +
          '    Model Saving Frequency: ' + str(args.model_save_freq) + '\n' 
    )

    if args.teacher_ckpt is not None:
        print('  Knowledge Distillation Params: ' + '\n' + 
              '    Teacher Checkpoint: ' + str(args.teacher_ckpt) + '\n' +
              '    L1 Knowledge Distillation Weight: ' + str(args.kd_lambda) + '\n' +              
              '    L1 Knowledge Distillation Mode: ' + str(args.kd_mode) + '\n'
        )
    else:
        print('  No Knowledge Distillation')
    

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def Get_Readable_Cur_Time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def D_Loss_BackProp(generator, discriminator, real_img, args, device, loss_dict, d_optim):
    '''
    Usage:
        To update the discriminator based on the GAN loss
    '''

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    noise = mixing_noise(args.batch, args.latent, args.mixing, device)

    time1 = time.time()
    fake_img, _ = generator(noise)

    time2 = time.time()
    fake_pred = discriminator(fake_img)
    real_pred = discriminator(real_img)
    d_loss = d_logistic_loss(real_pred, fake_pred)

    time3 = time.time()
    discriminator.zero_grad()
    d_loss.backward()
    d_optim.step()

    time4 = time.time()

    time_train_D_g_forward = time2 - time1
    time_train_D_d_forward = time3 - time2
    time_train_D_d_backward = time4 - time3

    loss_dict['d'] = d_loss
    loss_dict['real_score'] = real_pred.mean()
    loss_dict['fake_score'] = fake_pred.mean()

    return time_train_D_g_forward, time_train_D_d_forward, time_train_D_d_backward


def D_Reg_BackProp(real_img, discriminator, args, d_optim):
    '''
    Usage:
        To update the discriminator based on the regularization
    '''
    time1 = time.time()
    real_img.requires_grad = True
    real_pred = discriminator(real_img)
    r1_loss = d_r1_loss(real_pred, real_img)

    discriminator.zero_grad()
    print('Inside D_Reg discriminator zero grad Done!')
    d_reg_loss = args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]
    print(d_reg_loss.item())
    d_reg_loss.backward()
    d_optim.step()

    time2 = time.time()

    reg_D_time = time2 - time1
    return r1_loss, reg_D_time

def G_Loss_BackProp(generator, discriminator, args, device, loss_dict, g_optim, teacher_g):
    '''
    Usage:
        To update the generator based on the GAN loss and KD loss
    '''

    requires_grad(generator, True)
    requires_grad(discriminator, False)

    # GAN Loss
    time1 = time.time()
    noise = mixing_noise(args.batch, args.latent, args.mixing, device)
    fake_img_list, _ = generator(noise, return_rgb_list=True)
    fake_img = fake_img_list[-1]

    time2 = time.time()
    fake_pred = discriminator(fake_img)
    g_loss = g_nonsaturating_loss(fake_pred)
    loss_dict['g'] = g_loss

    total_loss = g_loss

    # KD Loss
    if teacher_g is not None:
        if args.kd_mode == 'Output_Only':
            fake_img_teacher, _ = teacher_g(noise)
            fake_img_teacher.requires_grad = True
            kd_l1_loss = args.kd_lambda * torch.mean(torch.abs(fake_img_teacher - fake_img))
        elif args.kd_mode == 'Intermediate':
            fake_img_teacher_list, _ = teacher_g(noise, return_rgb_list=True)
            for fake_img_teacher in fake_img_teacher_list:
                fake_img_teacher.requires_grad = True
            loss_list = [torch.mean(torch.abs(fake_img_teacher - fake_img)) for (fake_img_teacher, fake_img) in zip(fake_img_teacher_list, fake_img_list)] 
            kd_l1_loss = args.kd_lambda * sum(loss_list)   

        loss_dict['kd_loss'] = kd_l1_loss
        total_loss = g_loss + kd_l1_loss

    time3 = time.time()
    generator.zero_grad()
    total_loss.backward()
    g_optim.step()
    time4 = time.time()

    time_train_G_g_forward = time2 - time1
    time_train_G_d_forward = time3 - time2
    time_train_G_g_backward = time4 - time3
    return time_train_G_g_forward, time_train_G_d_forward, time_train_G_g_backward

def G_Reg_BackProp(generator, args, mean_path_length, g_optim):
    '''
    Usage:
        To update the generator based on the regularization
    '''
    time1 = time.time()

    path_batch_size = max(1, args.batch // args.path_batch_shrink)
    noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
    
    fake_img, path_lengths = generator(noise, PPL_regularize=True)
    decay = 0.01
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_loss = (path_lengths - path_mean).pow(2).mean()
    mean_path_length = path_mean.detach()

    generator.zero_grad()
    weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

    if args.path_batch_shrink:
        weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

    weighted_path_loss.backward()

    g_optim.step()

    mean_path_length_avg = (
        reduce_sum(mean_path_length).item() / get_world_size()
    )
    time2 = time.time()

    reg_G_time = time2 - time1

    return path_loss, path_lengths, mean_path_length, mean_path_length_avg, reg_G_time


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, teacher_g):

    # Experiment Statistics Setup
    loader = sample_data(loader)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length = 0
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for iter_idx in range(args.start_iter, args.iter):
        iter_start_time = time.time()
        
        real_img = next(loader)
        real_img = real_img.to(device)

        # Use GAN loss to train the discriminator
        time_train_D_g_forward, time_train_D_d_forward, time_train_D_d_backward = D_Loss_BackProp(generator, discriminator, real_img, args, device, loss_dict, d_optim)
        print((time_train_D_g_forward, time_train_D_d_forward, time_train_D_d_backward))
        # Discriminator regularization
        if iter_idx % args.d_reg_every == 0:
            print('Entering D_Reg')
            r1_loss, reg_D_time = D_Reg_BackProp(real_img, discriminator, args, d_optim)
            print(reg_D_time)

        loss_dict['r1'] = r1_loss

        # Use GAN loss to train the generator 
        time_train_G_g_forward, time_train_G_d_forward, time_train_G_g_backward = G_Loss_BackProp(generator, discriminator, args, device, loss_dict, g_optim, teacher_g)

        # Generator regularization
        if iter_idx % args.g_reg_every == 0:
            path_loss, path_lengths, mean_path_length, mean_path_length_avg, reg_G_time = G_Reg_BackProp(generator, args, mean_path_length, g_optim)
            
        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()
        iter_end_time = time.time()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()
        if teacher_g is not None:
            kd_loss = loss_reduced['kd_loss'].mean().item()
        else:
            kd_loss = 0

        if iter_idx % 1 == 0:
            print('Iter #: ' + str(iter_idx) + ' Train Time: ' + str(round(iter_end_time - iter_start_time, 2)) +
                  ' train_D_g_forward: ' + str(round(time_train_D_g_forward, 3))  + ' train_D_d_forward: ' + str(round(time_train_D_d_forward, 3)) + ' train_D_d_backward: ' + str(round(time_train_D_d_backward, 3)) +' D_Reg: ' + str(round(reg_D_time, 3)) + 
                  ' train_G_g_forward: ' + str(round(time_train_G_g_forward, 3))  + ' train_G_d_forward: ' + str(round(time_train_G_d_forward, 3)) + ' train_G_g_backward: ' + str(round(time_train_G_g_backward, 3)) +' G_Reg: ' + str(round(reg_G_time, 3))
            )
            

if __name__ == '__main__':

    # ============================== Setting All Hyperparameters ==============================
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)


    # ============================== Building Dataset ==============================
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    
    train_dataset = FFHQ_Dataset(args.path, transform)
    loader = data.DataLoader(train_dataset,
                             batch_size = args.batch,
                             shuffle=True,
                             num_workers=8)


    # ============================== Building Network Model ==============================

    # Building target compressed model
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    if args.ckpt is not None:    
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        generator = Build_Generator_From_Dict(ckpt['g']).to(device)
        g_ema = Build_Generator_From_Dict(ckpt['g_ema']).to(device)
        discriminator.load_state_dict(ckpt['d'])
        
    else:
        generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
        g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
        accumulate(g_ema, generator, 0)

    g_ema.eval()


    # Building the teacher model
    if args.teacher_ckpt is not None:
        teacher = torch.load(args.teacher_ckpt, map_location=lambda storage, loc: storage)
        teacher_g = Build_Generator_From_Dict(teacher['g_ema']).to(device)
        teacher_g.eval()
        requires_grad(teacher_g, False)
    else:
        teacher_g = None

    if args.distributed:
        generator = nn.DataParallel(generator, device_ids=train_hyperparams.gpu_device_ids)
        discriminator = nn.DataParallel(discriminator, device_ids=train_hyperparams.gpu_device_ids)
        if teacher_g is not None:
            teacher_g = nn.DataParallel(teacher_g, device_ids=train_hyperparams.gpu_device_ids)

    # ============================== Initializing Optimizers ==============================
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    if args.load_train_state:
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        args.start_iter = int(args.ckpt[-9: -3]) + 1
    else:
        args.start_iter = 0

    # ============================== Training Start ==============================
    Print_Experiment_Status()

    train_start_time = time.time()
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, teacher_g)
    train_end_time = time.time()

    print('\n' + 'Total training time: ' + str(round(train_end_time - train_start_time, 3)))
