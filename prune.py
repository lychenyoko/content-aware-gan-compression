# Content-aware pruning on a single GPU
# Author: Yuchen Liu

import torch
import numpy as np

import time
import datetime
import argparse

from Util.mask_util import Mask_the_Generator
from Util.content_aware_pruning import Get_Content_Aware_Pruning_Score
from Util.network_util import Build_Generator_From_Dict, Get_Network_Shape
from Util.pruning_util import Get_Uniform_RmveList, Generate_Prune_Mask_List

device = 'cuda:0'

# Parameter Parsing
parser = argparse.ArgumentParser()

parser.add_argument('--generated_img_size', type=int, default=256)
parser.add_argument('--ckpt', type=str, default='''./Model/full_size_model/256px_full_size.pt''')
parser.add_argument('--n_sample', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--noise_prob', type=float, default=0.05)
parser.add_argument('--remove_ratio', type=float, default=0.7)
parser.add_argument('--info_print', action='store_true', default=False)

args = parser.parse_args()


# Generator Loading

model_dict = torch.load(args.ckpt, map_location=device)
g_ema = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size).to(device)

# Generator Scoring
start_time = time.time()
grad_score_list = Get_Content_Aware_Pruning_Score(generator=g_ema, 
                                                  n_sample=args.n_sample, 
                                                  batch_size=args.batch_size, 
                                                  noise_prob=args.noise_prob, 
                                                  device=device, info_print=args.info_print)

grad_score_array = np.array([np.array(grad_score) for grad_score in grad_score_list])
content_aware_pruning_score = np.sum(grad_score_array, axis=0)

end_time = time.time()

print('The content-aware metric scoring takes: ' + str(round(end_time - start_time, 4)) + ' seconds')

# Generator Pruning

net_shape = Get_Network_Shape(g_ema.state_dict())
rmve_list = Get_Uniform_RmveList(net_shape, args.remove_ratio)
prune_net_mask = Generate_Prune_Mask_List(content_aware_pruning_score, net_shape, rmve_list, info_print=args.info_print)

pruned_generator_dict = Mask_the_Generator(g_ema.state_dict(), prune_net_mask)

pruned_ckpt = {'g': pruned_generator_dict, 'd': model_dict['d'], 'g_ema': pruned_generator_dict}
m_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
ckpt_file = './Model/pruned_model/content_aware_pruned_' + str(args.remove_ratio) + '_' + str(args.generated_img_size) + 'px_model_' + m_time + '.pth'

torch.save(pruned_ckpt, ckpt_file)
