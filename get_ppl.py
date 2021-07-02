import torch
import torch.nn as nn
import time
import argparse

from Util.network_util import Build_Generator_From_Dict
from Evaluation.ppl import Get_PPL_Score

device = 'cuda:0'
gpu_device_ids = [0,1]
latent_dim = 512

# Arg Parsing

parser = argparse.ArgumentParser()

parser.add_argument('--generated_img_size', type=int, default=256)
parser.add_argument('--ckpt', type=str, default='''./Model/full_size_model/256px_full_size.pt''')
parser.add_argument('--n_sample', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--eps', type=float, default=1e-4)
parser.add_argument('--info_print', action='store_true', default=False)

args = parser.parse_args()


model_dict = torch.load(args.ckpt, map_location=device)
g_ema = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size).to(device)
g_ema = nn.DataParallel(g_ema, device_ids=gpu_device_ids)
g_ema.eval();


print('Number of samples: ' + str(args.n_sample))
start_time = time.time()
ppl_score = Get_PPL_Score(g_ema, args.n_sample, args.batch_size, args.eps, latent_dim, device, gpu_device_ids, info_print = args.info_print)
end_time = time.time()
print('Total time is: ' + str(round(end_time - start_time, 4)))
print('PPL Scores: ' + str(ppl_score) + '\n')
