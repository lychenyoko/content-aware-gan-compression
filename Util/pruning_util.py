import numpy as np
import torch

from .network_util import Get_Layer_Output, Get_Network_Shape, Get_Generator_Effective_Weights, Get_Generator_Styles
from .pruning_metrics import Get_ASV_Score, Get_Map_L1_Norm, Get_L1_Normalized_ASV, Get_L2_Normalized_ASV, Get_Outgoing_L1Norm, Get_Incoming_L1Norm

#---------------------Define the Network Scoring Method---------------------

def Get_Map_Based_Score(generator, noise_z, net_shape, metric, info_print = False):
    '''
    Usage:
        Return the ASV score of channels in a generator 
    Args:
        generator:  (nn.Module) a generator that can be either DataParalleld or not
        noise_z:    (torch.Tensor) a noise tensor with shape [N, LATENT_DIMENSION]
        net_shape:  (list) of the # of channels in each layer
    '''
    if metric == 'ASV':
        map_score_func = Get_ASV_Score
    elif metric == 'l1-map':
        map_score_func = Get_Map_L1_Norm
    elif metric == 'l1-norm-ASV':
        map_score_func = Get_L1_Normalized_ASV
    elif metric == 'l2-norm-ASV':
        map_score_func = Get_L2_Normalized_ASV

    BATCH_SIZE = 10
    
    num_batch = noise_z.shape[0] // BATCH_SIZE
    num_layer = len(net_shape)
    
    Map_Score_List = []
    for layer_id in range(num_layer):
        if info_print:
            print('Scoring Layer: ' + str(layer_id))
        
        lay_out_list = []
        for i in range(num_batch):
            layer_out = Get_Layer_Output(generator, noise_z[i*BATCH_SIZE:(i+1)*BATCH_SIZE], layer_id)
            lay_out_list.append(layer_out)
        concat_lay_out = np.concatenate(lay_out_list, axis = 0)
        map_score = map_score_func(concat_lay_out)
        del layer_out
        
        if info_print:
            print(map_score)
            print('')
            
        Map_Score_List.append(map_score)
    return Map_Score_List



def Get_Network_Random_Score(net_shape):
    '''
    Usage:
        Return a random channel scoring of a generator
    Args:
        net_shape: (list) of the # of channels in each layer
    '''
    
    Random_Score_List = []
    for shape in net_shape:
        random_score = np.random.random(shape)
        Random_Score_List.append(random_score)
    return Random_Score_List



def Get_Weight_Based_Score(generator, noise_z, metric, info_print = False):
    '''
    Usage:
        Return the scores of channels based on network weights 
    Args:
        generator: (nn.Module) a generator that can be either DataParalleld or not
        noise_z:   (torch.Tensor) a noise tensor with shape [N, LATENT_DIMENSION]
        metric:    (str) of whether getting l1 norm of incoming filters or out-going filters
    '''

    if metric == 'l1-in':
        weight_score_func = Get_Incoming_L1Norm
    elif metric == 'l1-out':
        weight_score_func = Get_Outgoing_L1Norm
    
    weight_list = Get_Generator_Effective_Weights(generator, noise_z)

    Weight_Score_List = []
    for layer_id in range(len(weight_list) - 1):
        if info_print:
            print('Scoring Layer: ' + str(layer_id))        
        
        effective_weight = np.mean(weight_list[layer_id], axis=0)
        weight_score = weight_score_func(effective_weight)
        
        if info_print:
            print(weight_score)
            print('')
            
        Weight_Score_List.append(weight_score)
        
    # Random Appending for unevaluated channel
    if metric == 'l1-in':
        input_shape = weight_list[0].shape[2]
        Weight_Score_List.insert(0, list(np.random.random(input_shape)))
    
    elif metric == 'l1-out':
        effective_weight = np.mean(weight_list[-1], axis=0) # the toRGB weights 
        weight_score = weight_score_func(effective_weight)
        Weight_Score_List.append(weight_score)
        
    del weight_list
    return Weight_Score_List



def Get_Style_Based_Score(generator, noise_z, metric, info_print = False):
    '''
    Usage:
        Return the scores of channels based on network channel styles 
    Args:
        generator: (nn.Module) a generator that can be either DataParalleld or not
        noise_z:   (torch.Tensor) a noise tensor with shape [N, LATENT_DIMENSION]
        metric:    (str) of getting l1 norm of styles
    '''
    if metric == 'l1-style':
        score_func = lambda s: np.abs(np.mean(s, axis=0))

    style_list = Get_Generator_Styles(generator, noise_z)

    Style_Score_List = []
    for layer_id in range(len(style_list)):
        if info_print:
            print('Scoring Layer: ' + str(layer_id))        
        
        style_score = score_func(style_list[layer_id])
        
        if info_print:
            print(style_score)
            print('')
            
        Style_Score_List.append(style_score)

    return Style_Score_List


def Get_Network_Score_List(generator, noise_z, metric, info_print = True):
    '''
    Usage:
        Return the score of channels in a generator instance as a list of list 
    Args:
        generator:  (nn.Module) a generator that can be either DataParalleld or not
        noise_z:    (torch.Tensor) a noise tensor with shape [N, LATENT_DIMENSION],
        metric:     (str) of the channel importance metric to evaluate the channel
        info_print: (bool) whether or not to print the scoring details
    '''
    METRIC_LIST = ['Random', 'ASV', 'l1-norm-ASV', 'l2-norm-ASV', 'l1-map', 'l1-in', 'l1-out', 'l1-style']
    assert metric in METRIC_LIST
    
    if 'module' in list(generator.state_dict().keys())[0]:
        g_dict = generator.module.state_dict()
    else:
        g_dict = generator.state_dict()
    net_shape = Get_Network_Shape(g_dict)
    
    print('\n' + '''-----------------------------Scoring Network Channels-----------------------------''')
    print('Scoring Metric: ' + metric)
    
    if metric == 'Random':
        Network_Score_List = Get_Network_Random_Score(net_shape)
    
    elif metric in ['ASV', 'l1-map', 'l1-norm-ASV', 'l2-norm-ASV']:
        Network_Score_List = Get_Map_Based_Score(generator, noise_z, net_shape, metric, info_print)
    
    elif metric in ['l1-in', 'l1-out']:
        Network_Score_List = Get_Weight_Based_Score(generator, noise_z, metric, info_print)

    elif metric in ['l1-style']:        
        Network_Score_List = Get_Style_Based_Score(generator, noise_z, metric, info_print)
    
    return Network_Score_List


#---------------------Define the Pruning Scheduling Method---------------------

def Get_Default_Mask_From_Shape(net_shape):
    '''
    Usage:
        Obtain the all [True] default mask list for a given net shape
    
    Args:
        net_shape: (list) of of number of channels in the layer
    '''
    net_default_mask = [np.array([True] * layer_shape) for layer_shape in net_shape]
    return net_default_mask


def Generate_Prune_Mask_List(Net_Score_List, net_shape, rmve_list, info_print = False):
    '''
    Usage:
        Get prune_mask_list by the channel score list and the removal list
    Args:
        Net_Score_List: (list) of layer (list) of scores of every channel 
        net_shape:      (list) of of number of channels in the layer
        rmve_list:      (list) containing number of removed channels of every layer
        info_print:     (bool) whether to print the information or not
    '''

    net_mask_list = Get_Default_Mask_From_Shape(net_shape)
    print('\n' + '-----------------------------Actual Pruning Happens-----------------------------')

    for lay_k in range(len(net_shape)):

        layer_mask = net_mask_list[lay_k]
        layer_rmv = rmve_list[lay_k]                                
        layer_score_list = Net_Score_List[lay_k]
        assert len(layer_mask) == len(layer_score_list)

        if info_print:
            print('\n' + 'Layer ID: ' + str(lay_k))
            print('Layer Remove: ' + str(layer_rmv))

        # Pruning maps
        if (sum(layer_mask) > layer_rmv and layer_rmv > 0):
            rmv_node = np.argsort(layer_score_list)[:layer_rmv]
            layer_mask[rmv_node] = False

            print('We have masked out  #' + str(rmv_node) + ' in layer ' + str(lay_k) + '. It will have ' 
            + str(sum(layer_mask)) +' maps.')

    return net_mask_list


def Get_Uniform_RmveList(net_shape, pruning_ratio):

    '''
    Usage:
        To get the uniform remove list of the whole neural net based on a certain ratio
    Args:
        net_shape:     (list) the shape of the unpruned network
        pruning_ratio: (float) the ratio of channels to be removed
    '''

    rmve_list = (np.array(net_shape) * pruning_ratio).astype(int)
    return rmve_list
