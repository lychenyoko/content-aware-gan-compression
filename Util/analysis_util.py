import torch
import numpy as np
from .network_util import Get_Layer_Output, Convert_Tensor_To_Image

from matplotlib import pyplot as plt
import os

def Get_Visual_Response(generator, num_img, layer_id):
    '''
    Usage:
        Get the response of a layer as well as the final generated image for visualization
    Args:
        generator: (nn.Module) a StyleGAN2 generator
        num_img:   (int) the number of visualized images
        layer_id:  (int) the index of the hidden layer to be visualized
    '''
    LATENT_DIM = 512
    noise_z = torch.randn(num_img, LATENT_DIM)
    
    if torch.cuda.is_available():
        noise_z = noise_z.to('cuda')
    layer_response = Get_Layer_Output(generator, noise_z, layer_id)
    
    img_tensor, _ = generator([noise_z])
    generated_image = []
    for i in range(num_img):
        pil_img = Convert_Tensor_To_Image(img_tensor[i])
        np_img = np.array(pil_img)
        generated_image.append(np_img)
    
    return layer_response, generated_image

def Pick_High_Low_Activations(layer_activation, channel_rank):
    '''
    Usage:
        Pick the responses of good channel and bad channel (good and bad are based on their ranks).
        Also return the min and max of the responses for colormap plotting
    
    Args:
        layer_activation: (np.array) of [N, C, H, W]
        channel_rank:     (np.array) indicating the rank of the channel, 
                                     channel_rank[0] refers to the worst channel
                                     channel_rank[-1] refers to the best channel 
    '''
    num_img = layer_activation.shape[0]
    good_channel = np.array([layer_activation[i][channel_rank[-1]] for i in range(num_img)])
    bad_channel = np.array([layer_activation[i][channel_rank[0]] for i in range(num_img)])
    value_min = np.min([np.min(good_channel), np.min(bad_channel)])
    value_max = np.max([np.max(good_channel), np.max(bad_channel)])
    return good_channel, bad_channel, value_max, value_min

def Channel_Image_Visualization(generated_image, good_channel, bad_channel, value_max, value_min):
    '''
    Usage:
        Visualize the generated image as well as good and bad channel for metric analysis.
        
    Args:
        generated_image: (list) of generated images
        good_channel:    (np.array) of activations from good channel
        bad_channel:     (np.array) of activations from bad channel
        value_max:       (float) of the maximum single pixel activation
        value_min:       (float) of the minimum single pixel activation
    '''
    
    dpi = 200
    wspace = -0.3
    hspace = 0.1
    num_img = len(generated_image)

    fig = plt.figure(dpi = dpi)
    for i in range(num_img):
        index = i + 1 
        plt.subplot(3,num_img,index)
        plt.imshow(generated_image[i])
        plt.axis('off');
        index+=num_img

        plt.subplot(3,num_img,index)
        plt.imshow(good_channel[i], cmap = 'jet', vmin=value_min, vmax=value_max)
        plt.axis('off');

        index+=num_img
        plt.subplot(3,num_img,index)
        im = plt.imshow(bad_channel[i], cmap = 'jet', vmin=value_min, vmax=value_max)
        plt.axis('off');

    plt.subplots_adjust(wspace=wspace,hspace=hspace)
    cbar_ax = fig.add_axes([0.865, 0.12, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax)


def Extract_Training_Log(exp_dir):
    '''
    Usage:
        To extract the FID & FLOPs information from a training log

    Args:
        exp_dir: (str) of the checkpoint directory
    '''
    
    # Find the Experiment Log
    for file in os.listdir(exp_dir):
        if '.out' in file:
            exp_log_file = os.path.join(exp_dir, file)
    
    FLOPS_STR = 'FLOPs %:'
    FID_STR = 'Evaluated FID:'

    FLOPs_list = []
    FID_list = []

    for line in open(exp_log_file, 'r').readlines():
        if FLOPS_STR in line:
            FLOPs = float(line[len(FLOPS_STR):])
            FLOPs_list.append(FLOPs)
        elif FID_STR in line:
            FID = float(line[len(FID_STR):])
            FID_list.append(FID)
    
    return FLOPs_list, FID_list

def Extract_Training_KD_Loss(exp_dir):
    '''
    Usage:
        To extract the FID & FLOPs information from a training log
    Args:
        exp_dir: (str) of the checkpoint directory
    '''
    
    # Find the Experiment Log
    for file in os.listdir(exp_dir):
        if '.out' in file:
            exp_log_file = os.path.join(exp_dir, file)
        
    KD_L1_STR = '''KD_L1_Loss:'''
    KD_LPIPS_STR = '''KD_LPIPS_Loss:'''
    END_STR = '''D_Reg:'''
    
    kd_l1_loss_list = []
    kd_lpips_loss_list = []
    for line in open(exp_log_file, 'r').readlines():
        if 'Iter #' in line:
            index_list = [line.find(_str) for _str in [KD_L1_STR,KD_LPIPS_STR,END_STR]]
            kd_l1_loss = float(line[index_list[0] + len(KD_L1_STR) :index_list[1]])
            kd_l1_loss_list.append(kd_l1_loss)

            kd_lpips_loss = float(line[index_list[1] + len(KD_LPIPS_STR) :index_list[2]])
            kd_lpips_loss_list.append(kd_lpips_loss)
    
    return kd_l1_loss_list, kd_lpips_loss_list
