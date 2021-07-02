import numpy as np

def Get_ASV_Score(fea_maps):
    '''
    Usage:
        To compute the ASV scores for a whole layer's tensor and return them as a list with len = C
    
    Args:
        fea_maps: 4D (np.array) of shape [N, C, H, W]
    '''
    std_map = np.std(fea_maps, axis=0)
    channel_score_list = std_map.mean(axis = (1,2))
    return channel_score_list

def Get_Map_L1_Norm(fea_maps):
    '''
    Usage:
        To compute the l1_norm scores for a whole layer's tensor and return them as a list with len = C
    
    Args:
        fea_maps: 4D (np.array) of shape [N, C, H, W]
    '''
    channel_score_list = np.mean(np.abs(fea_maps), axis = (0,2,3))
    return channel_score_list

def Get_Map_L2_Norm(fea_maps):
    '''
    Usage:
        To compute the l2_norm scores of a layer activation and return them as a list with len = C (# of channels)
    
    Args:
        fea_maps: 4D (np.array) of shape [N, C, H, W]
    '''
    channel_score_list = np.sqrt(np.sum(np.square(fea_maps), axis = (0,2,3)))
    return channel_score_list


def Get_L1_Normalized_ASV(fea_maps):
    '''
    Usage:
        Normalize each channel tensor [N, H, W] to have the same l1-norm and then measure their ASV

    Args:
        fea_maps: 4D (np.array) of shape [N, C, H, W]
    '''
    asv_score = Get_ASV_Score(fea_maps)
    pixel_l1_norm_score = Get_Map_L1_Norm(fea_maps)
    l1_normalized_asv = asv_score / pixel_l1_norm_score
    return l1_normalized_asv


def Get_L2_Normalized_ASV(fea_maps):
    '''
    Usage:
        Normalize each channel tensor [N, H, W] to have the same l2-norm and then measure their ASV

    Args:
        fea_maps: 4D (np.array) of shape [N, C, H, W]
    '''
    asv_score = Get_ASV_Score(fea_maps)
    l2_norm_score = Get_Map_L2_Norm(fea_maps)
    l2_normalized_asv = asv_score / l2_norm_score
    return l2_normalized_asv


def Get_Outgoing_L1Norm(filter_4D):
    '''
    Usage: 
        To use the l1-norm of outgoing weights for channel importance evaluation
    Args:
        filter_4D: (np.array) of 4D filters with shape [Nout, Nin, H, W]
    '''
    channel_score = []
    for i in range(filter_4D.shape[1]):
        outgoing_kernel = filter_4D[:,i,...]
        l1norm = np.linalg.norm(outgoing_kernel.reshape(-1), ord=1)
        channel_score.append(l1norm)    
    return channel_score


def Get_Incoming_L1Norm(filter_4D):
    '''
    Usage: 
        To use the l1-norm of incoming weights for channel importance evaluation
    Args:
        filter_4D: (np.array) of 4D filters with shape [Nout, Nin, H, W]
    '''
    channel_score = []
    for i in range(filter_4D.shape[0]):
        outgoing_kernel = filter_4D[i,...]
        l1norm = np.linalg.norm(outgoing_kernel.reshape(-1), ord=1)
        channel_score.append(l1norm)    
    return channel_score
