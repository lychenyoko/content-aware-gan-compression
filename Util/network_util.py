from torchvision import utils
import torch
from torch import nn
from PIL import Image

import sys
sys.path.append('..')
from model import Generator

def Get_Conv_Kernel_Key(model_dict):
    '''
    Usage:
        Return a list of keys of the convolutional weights in main feedforwarding flow
    
    Args:
        model_dict: (dict) of a StyleGAN2 generator.
    '''
    CONV1_KEY = 'conv1.conv.weight'
    
    conv_key_list = [CONV1_KEY]
    for key in model_dict.keys():
        if ('convs' in key) and ('conv.weight' in key):
            conv_key_list.append(key)
    
    return conv_key_list

def Get_Network_Shape(model_dict):
    '''
    Usage:
        Return the shape of the network (number of channels in each layer) in a list  
    
    Args:
        model_dict: (dict) of a StyleGAN2 generator
    '''
    conv_key_list = Get_Conv_Kernel_Key(model_dict)
    num_channels = [model_dict[key].shape[2] for key in conv_key_list] # from start to end
    num_channels.append(model_dict[conv_key_list[-1]].shape[1])        # last layer
    return num_channels


def Convert_Tensor_To_Image(img_tensor):
    '''
    Usage:
        Convert a torch.Tensor output from the StyleGAN2 to a PIL image
    '''
    grid = utils.make_grid(img_tensor, nrow =1, padding=2, pad_value=0, 
                           normalize=True, range = (-1,1), scale_each=False)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def Get_Layer_Output(generator, sample_z, layer_id):
    '''
    Usage:
        Return the 4D output tensor of a layer in shape [N, C, H, W] in a noise = None manner 
    
    Args:
        generator: (nn.Module) a generator that can be either DataParalleld or not
        sample_z:  (torch.Tensor) a noise tensor with shape [N, LATENT_DIMENSION]
        layer_id:  (int) of the id of the layer, 0 corresponds to the constant input, 5 corresponds to the 5th layer, etc.
    '''
    
    with torch.no_grad():
        
        generator.eval()
        
        if 'module' in list(generator.state_dict().keys())[0]:
            g_module = generator.module
        else:
            g_module = generator

        # Get the latent input first
        latent_W = g_module.style(sample_z)

        constant_input = g_module.input(latent_W)
        if layer_id == 0: # constant input layer
            return constant_input.cpu().numpy()

        out = g_module.conv1(constant_input, latent_W)
        for i in range(layer_id - 1):
            out = g_module.convs[i](out, latent_W)

        numpy_out = out.cpu().numpy()
        del out    
        return numpy_out

def Build_Generator_From_Dict(model_dict, size=256, latent=512, n_mlp=8):
    '''
    Usage:
        To build a generator based on a model dict
    
    Args:
        model_dict: (dict) of the state of the generator
        size:       (int) size of the generated image
        latent:     (int) dimension of the latent noise
        n_mlp:      (int) number of layers in the mapping mlp network
    '''
    net_shape = Get_Network_Shape(model_dict)
    generator = Generator(size, latent, n_mlp, generator_net_shape=net_shape)
    generator.load_state_dict(model_dict, strict=False)
    return generator    


def Get_Generator_Effective_Weights(generator, noise_z):
    '''
    Usage:
        Return the kernels after modulation and demodulation in each layer as a list
    
    Args:
        generator: (nn.Module) a generator that can be either DataParalleld or not
        noise_z:  (torch.Tensor) a noise tensor with shape [N, LATENT_DIMENSION]
    '''
    
    with torch.no_grad():
        
        generator.eval()
        
        # Get the latent input first
        if 'module' in list(generator.state_dict().keys())[0]:
            style_DP = nn.DataParallel(generator.module.style, device_ids = generator.device_ids)
            latent_W = style_DP(noise_z)
            g_module = generator.module
            gpu_device_ids = generator.device_ids
        else:
            latent_W = generator.style(noise_z)
            g_module = generator.module
            gpu_device_ids = None

        # The styled conv's effective weights
        effective_weight_list = []
        styled_conv_list = [g_module.conv1] + list(g_module.convs) + [g_module.to_rgbs[-1]]
        for styled_conv in styled_conv_list:
            effective_weight = Get_Styled_Conv_Effective_Weights(styled_conv, latent_W, gpu_device_ids)
            effective_weight_list.append(effective_weight)
        
        return effective_weight_list

def Get_Styled_Conv_Effective_Weights(styled_conv, latent_W, gpu_device_ids=None):
    '''
    Usage:
        Return the kernel after modulation and demodulation in a styled convolution layer
        
    Args:
        styled_conv: (nn.Module) a styled convolution module
        latent_W:    (torch.Tensor) mapped latent_W for styled control
    '''
    
    # Modulation
    if gpu_device_ids is None:
        style = styled_conv.conv.modulation(latent_W)
    else:
        modulation_DP = nn.DataParallel(styled_conv.conv.modulation, device_ids=gpu_device_ids) 
        style = modulation_DP(latent_W)

    N, C = style.shape
    style = style.view(N,1,C,1,1)
    weight = styled_conv.conv.scale * styled_conv.conv.weight.cpu() * style.cpu() # Place the memory on cpu
    
    # Demodulation
    if styled_conv.conv.demodulate == True:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(N, styled_conv.conv.out_channel, 1, 1, 1)
    return weight.numpy()


def Get_Generator_Styles(generator, noise_z):
    '''
    Usage:
        Return the styles (affine transformation weights) for each channel in (np.array)
    
    Args:
        generator:       (nn.Module) a generator that can be either DataParalleld or not
        noise_z:         (torch.Tensor) a noise tensor with shape [N, LATENT_DIMENSION]
    '''
    
    with torch.no_grad():

        generator.eval()

        # Get the latent_W here
        if 'module' in list(generator.state_dict().keys())[0]:
            style_tran = nn.DataParallel(generator.module.style, device_ids=generator.device_ids)
            g_module = generator.module
        else:
            style_tran = generator.style
            g_module = generator            
        latent_W = style_tran(noise_z)

        # The styled conv's effective weights
        style_list = []
        styled_conv_list = [g_module.conv1] + list(g_module.convs) + [g_module.to_rgbs[-1]]
        for styled_conv in styled_conv_list:
            style = styled_conv.conv.modulation(latent_W).cpu().numpy()
            style_list.append(style)
                
        return style_list
