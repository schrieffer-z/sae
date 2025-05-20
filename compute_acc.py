model_name = '/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae4rm/models/Llama-3.1-8B-Instruct_token_Latent131072_Layer16_K192_50M-SAE4RM/checkpoint-225'

import os
import json
import torch
from safetensors.torch import load_file

weight_map = None
safetensors_name = None
weight_map_path = os.path.join(model_name, 'model.safetensors.index.json')
if os.path.exists(weight_map_path):
    with open(weight_map_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)['weight_map']
    score_weights = load_file(os.path.join(model_name, weight_map['score.weight']))['score.weight'].view(-1)
else:
    score_weights = load_file(os.path.join(model_name,"model.safetensors"))['score.weight'].view(-1)


print(score_weights[score_weights!=0])

print(torch.arange(len(score_weights))[score_weights!=0])
print(len(torch.arange(len(score_weights))[score_weights!=0]))


latents = torch.load('./latents.pt')
latents_of_rm = [str(i.item()) for i in latents.item()]

# torch.save(torch.arange(len(score_weights))[score_weights!=0], 'latents.pt')



# topk = torch.topk(score_weights.abs(), 500, dim=-1)
# print(topk.indices)
# print(topk.values)

