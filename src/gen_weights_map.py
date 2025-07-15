model_name = ''
context_path = ''


import json
with open(context_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
latent_context_map = data.get('latent_context_map', {})
all_latents = set(latent_context_map.keys())

import os

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

latent_list = []
for latent in all_latents:
    latent_list.append( {latent:score_weights[int(latent)].item()} )

latent_list.sort(key=lambda x: list(x.values())[0], reverse=True)

selected_latent_dict = dict()
selected_latent_list = latent_list[:100] + latent_list[-100:]
for i in selected_latent_list:
    selected_latent_dict.update(i)


with open('a.json', 'w') as f:
    json.dump(selected_latent_dict, f, indent=4)
