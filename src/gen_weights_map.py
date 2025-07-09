model_name = ''

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

topk = torch.topk(score_weights, 200, dim=-1)
pos = {str(latent.item()):act.item() for latent, act in zip(topk.indices, topk.values)}
topk = torch.topk(-score_weights, 200, dim=-1)
neg = {str(latent.item()):-act.item() for latent, act in zip(topk.indices, topk.values)}

pos.update(neg)


with open('a.json', 'w') as f:
    json.dump(pos, f, indent=4)
