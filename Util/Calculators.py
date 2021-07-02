import numpy as np
from .network_util import Get_Network_Shape

# Constant Parameters
MAP_SIZE = []
for i in range(2, 11):
    map_size = 2 ** i
    MAP_SIZE += [map_size, map_size]

STYLE_CONV_KER_SIZE = 3
TORGB_CONV_KER_SIZE = 1
NUM_RGB_CHANNEL = 3
GENERATOR_FLOPS_256PX = 45124673536
GENERATOR_FLOPS_1024PX = 74266894336

def Styled_Conv_FLOPCal(model_dict, return_detail = True):
    '''
    Usage:
        To calculate the FLOPs of the styled convolution part in the StyleGAN2 generator
        
    Args:
        model_dict: (dict) of a StyleGAN2 generator
    '''
    
    net_shape = Get_Network_Shape(model_dict)
    lay_FLOPs_list = []
    for i in range(len(net_shape) - 1):
        N_in, N_out = net_shape[i], net_shape[i + 1]
        lay_FLOPs = N_in * N_out * (STYLE_CONV_KER_SIZE ** 2) * (MAP_SIZE[i] ** 2)
        lay_FLOPs_list.append(lay_FLOPs)
    
    all_FLOPs = sum(lay_FLOPs_list)
    
    if return_detail:
        return all_FLOPs, lay_FLOPs_list
    else:
        return all_FLOPs

def ToRGB_Conv_FLOPCal(model_dict, return_detail = True):
    '''
    Usage:
        To calculate the FLOPs of the toRGB convolution part in the StyleGAN2 generator
        
    Args:
        model_dict: (dict) of a StyleGAN2 generator
    '''
    
    net_shape = Get_Network_Shape(model_dict)

    toRGB_FLOPs_list = []
    for i in range(len(net_shape)//2):
        N_in = net_shape[2 * i + 1]
        toRGB_FLOPs = N_in * NUM_RGB_CHANNEL * (TORGB_CONV_KER_SIZE ** 2) * (MAP_SIZE[2 * i + 1] ** 2)
        toRGB_FLOPs_list.append(toRGB_FLOPs)
    
    all_FLOPs = sum(toRGB_FLOPs_list)
    
    if return_detail:
        return all_FLOPs, toRGB_FLOPs_list
    else:
        return all_FLOPs

def Mapping_Network_FLOPCal(model_dict):
    '''
    Usage:
        To calculate the FLOPs of the mapping network part in the StyleGAN2 generator
        
    Args:
        model_dict: (dict) of a StyleGAN2 generator
    '''
    mapping_network_FLOPs = []
    for key in model_dict.keys():
        if 'style' in key and 'weight' in key:
            FLOPs = np.prod(model_dict[key].shape)
            mapping_network_FLOPs.append(FLOPs)
    
    return sum(mapping_network_FLOPs)

def Style_Modulation_FLOPCal(model_dict):
    '''
    Usage:
        To calculate the FLOPs of the generating styles from latent code part in the StyleGAN2 generator
        
    Args:
        model_dict: (dict) of a StyleGAN2 generator
    '''
    style_mod_FLOPs = []
    for key in model_dict.keys():
        if 'modulation.weight' in key:
            FLOPs = np.prod(model_dict[key].shape)
            style_mod_FLOPs.append(FLOPs)
    
    return sum(style_mod_FLOPs)

def StyleGAN2_FLOPCal(generator_dict):
    '''
    Usage:
        Return the overall FLOPs of a StyleGAN2 generator
    '''
    styled_conv_FLOPs = Styled_Conv_FLOPCal(generator_dict, return_detail=False)
    toRGB_FLOPs = ToRGB_Conv_FLOPCal(generator_dict, False)
    mapping_network_FLOPs = Mapping_Network_FLOPCal(generator_dict)
    style_mod_FLOPs = Style_Modulation_FLOPCal(generator_dict)    
    all_FLOPs = sum([styled_conv_FLOPs, toRGB_FLOPs, mapping_network_FLOPs, style_mod_FLOPs])
    return all_FLOPs
