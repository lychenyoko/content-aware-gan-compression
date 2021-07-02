import random
from copy import deepcopy

from .network_util import Get_Conv_Kernel_Key


def get_random_mask(size, mask_ratio):
    return [random.random() > mask_ratio for i in range(size)]


def Mask_the_Generator(model_dict, net_mask_list):
    '''
    Usage:
        Produce a pruned generator dictionary based on a mask list
    
    Args:
        model_dict:    (dict) of the state of the generator
        net_mask_list: (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
    '''
    
    # Getting the styled_conv key to be masked
    styled_conv_key = Get_Conv_Kernel_Key(model_dict) 
    styled_mod_key = ['conv1.conv.modulation.weight', 'conv1.conv.modulation.bias']
    for key in model_dict.keys():
        if ('convs' in key) and ('modulation' in key):
            styled_mod_key.append(key)
    styled_act_key = []
    for key in model_dict.keys():
        if ('activate' in key):
            styled_act_key.append(key)
    
    # Getting the toRGB key to be masked
    toRGB_key = []
    for key in model_dict.keys():
        if ('to_rgb' in key) and (('conv.weight' in key) or ('modulation' in key)):
            toRGB_key.append(key)
    
    # The dictionary of the final pruned model
    pruned_dict = deepcopy(model_dict)
    
    # Masking operation
    pruned_dict['input.input'] = model_dict['input.input'].cpu()[:, net_mask_list[0], ...]
    Mask_Styled_Conv_Key(model_dict, pruned_dict, net_mask_list, styled_conv_key)
    Mask_Styled_Mod_Key(model_dict, pruned_dict, net_mask_list, styled_mod_key)
    Mask_Styled_Act_Key(model_dict, pruned_dict, net_mask_list, styled_act_key)
    Mask_toRGB_Key(model_dict, pruned_dict, net_mask_list, toRGB_key)
    
    return pruned_dict


def Mask_Styled_Conv_Key(model_dict, pruned_dict, net_mask_list, styled_conv_key):
    '''
    Usage:
        Update the conv weights of styled convolution
    
    Args:
        model_dict:      (dict) of the original model state
        pruned_dict:     (dict) of the pruned model state
        net_mask_list:   (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
        styled_conv_key: (list) of key for the styled conv kernel
    '''
    for idx,key in enumerate(styled_conv_key):
        input_mask, output_mask = net_mask_list[idx], net_mask_list[idx + 1]
        masked_weight = model_dict[key].cpu()[:, output_mask, ...][:, :, input_mask, ...]
        pruned_dict[key] = masked_weight


def Mask_Styled_Mod_Key(model_dict, pruned_dict, net_mask_list, styled_mod_key):
    '''
    Usage:
        Update the weights of the affine transformation in the styled convolution
    
    Args:
        model_dict:     (dict) of the original model state
        pruned_dict:    (dict) of the pruned model state
        net_mask_list:  (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
        styled_mod_key: (list) of key for the affine transformation to get styles 
    '''
    NUM_KEY_EACH_LAYER = 2
    for idx in range(len(styled_mod_key) // NUM_KEY_EACH_LAYER):
        layer_mask = net_mask_list[idx]
        weight_key, bias_key = styled_mod_key[idx * 2], styled_mod_key[idx * 2 + 1]
        pruned_dict[weight_key] = model_dict[weight_key].cpu()[layer_mask, ...]
        pruned_dict[bias_key]   = model_dict[bias_key].cpu()[layer_mask]


def Mask_Styled_Act_Key(model_dict, pruned_dict, net_mask_list, styled_act_key):
    '''
    Usage:
        Update the styled convolution's activation bias
    
    Args:
        model_dict:     (dict) of the original model state
        pruned_dict:    (dict) of the pruned model state
        net_mask_list:  (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
        styled_act_key: (list) of key for the styled conv activation bias
    '''
    for idx,key in enumerate(styled_act_key):
        output_mask = net_mask_list[idx + 1]
        masked_act_bias = model_dict[key].cpu()[output_mask]
        pruned_dict[key] = masked_act_bias


def Mask_toRGB_Key(model_dict, pruned_dict, net_mask_list, toRGB_key):
    '''
    Usage:
        Update the weights in the toRGB module
    
    Args:
        model_dict:    (dict) of the original model state
        pruned_dict:   (dict) of the pruned model state
        net_mask_list: (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
        toRGB_key:     (list) of key for the toRGB module
    '''
    NUM_KEY_EACH_LAYER = 3
    for idx in range(len(toRGB_key) // NUM_KEY_EACH_LAYER):
        layer_mask = net_mask_list[idx * 2 + 1] # the layer idx corresponding to the toRGB module
        conv_key, mod_weight_key, mod_bias_key = toRGB_key[idx * NUM_KEY_EACH_LAYER: (idx+1) * NUM_KEY_EACH_LAYER]
        pruned_dict[conv_key]       = model_dict[conv_key].cpu()[:, :, layer_mask, ...]
        pruned_dict[mod_weight_key] = model_dict[mod_weight_key].cpu()[layer_mask, ...]
        pruned_dict[mod_bias_key]   = model_dict[mod_bias_key].cpu()[layer_mask]
