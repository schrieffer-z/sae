# interp_path = '/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/interpret/interp_checkpoint-175_token_Latent24576_Layer18_K64_100M-1.json'
# model_name = '/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae4rm/models/Llama-3.2-3B-Instruct_token_Latent24576_Layer18_K64_100M-SAE4RM/checkpoint-175'

# 0.04
# interp_path = '/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/interpret/interp_checkpoint-450_token_Latent131072_Layer16_K192_10M-1.json'
# model_name = '/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae4rm/models/Llama-3.1-8B-Instruct_token_Latent131072_Layer16_K192_50M-SAE4RM/checkpoint-50'


interp_path = '/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/interpret/interp_Llama-3.1-8B-Instruct_token_Latent131072_Layer16_K192_50M.json'
model_name = '/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae4rm/models/Llama-3.1-8B-Instruct_token_Latent131072_Layer16_K192_50M-SAE4RM/checkpoint-150'


import os
import json
from safetensors.torch import load_file

with open(interp_path, "r", encoding="utf-8") as f:
    interpret = json.load(f)


weight_map = None
safetensors_name = None
weight_map_path = os.path.join(model_name, 'model.safetensors.index.json')
if os.path.exists(weight_map_path):
    with open(weight_map_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)['weight_map']
    score_weights = load_file(os.path.join(model_name, weight_map['score.weight']))['score.weight'].view(-1)
else:
    score_weights = load_file(os.path.join(model_name,"model.safetensors"))['score.weight'].view(-1)


acc = 0 
total = 0
for f in interpret['results']:
    if interpret['results'][f]['score'] != 0:
        if interpret['results'][f]['score']==None:
            continue
        # if interpret['results'][f]['score']!=-2 and interpret['results'][f]['score']!=2:
        #     continue
        if interpret['results'][f]['score'] * score_weights[int(f)] > 0:
            print(f,interpret['results'][f]['score'], score_weights[int(f)])
            acc += 1
        # else:
        #     print()
        #     print('='*80)
        #     print(f, interpret['results'][f]['score'], score_weights[int(f)].item())
        #     for c in interpret['results'][f]['contexts']:
        #         print(c)
        total += 1

print(acc,total)
print(acc/total)