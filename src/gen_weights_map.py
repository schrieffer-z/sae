import os
import json
import argparse
import torch
from safetensors.torch import load_file

def extract_top_latents(model_name: str,
                        context_path: str,
                        top_k_list: list[int],
                        output_dir: str = '.',
                        output_name: str = 'a'):
    # 1. 读取 latent_context_map
    with open(context_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    latent_context_map = data.get('latent_context_map', {})
    all_latents = set(latent_context_map.keys())

    # 2. 读取模型权重
    weight_map_path = os.path.join(model_name, 'model.safetensors.index.json')
    if os.path.exists(weight_map_path):
        with open(weight_map_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)['weight_map']
        score_weights = load_file(os.path.join(model_name, weight_map['score.weight']))['score.weight'].view(-1)
    else:
        score_weights = load_file(os.path.join(model_name, "model.safetensors"))['score.weight'].view(-1)

    # 3. 构建 latent-score 映射并排序
    latent_list = []
    for latent in all_latents:
        latent_list.append({latent: score_weights[int(latent)].item()})
    latent_list.sort(key=lambda x: list(x.values())[0], reverse=True)

    # 4. 提取 top-K 并写入 json
    os.makedirs(output_dir, exist_ok=True)
    for num in top_k_list:
        selected_latent_list = latent_list[:num] + latent_list[-num:]
        selected_latent_dict = {}
        for i in selected_latent_list:
            selected_latent_dict.update(i)

        output_path = os.path.join(output_dir, f'{output_name}-top{num*2}.json')
        with open(output_path, 'w') as f:
            json.dump(selected_latent_dict, f, indent=4)

    print(f"Saved top-{[n*2 for n in top_k_list]} latent score files to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract top latent scores from safetensors model.")
    parser.add_argument('--model_name', type=str, required=True,
                        help='Path to the model directory containing model.safetensors or index.json')
    parser.add_argument('--context_path', type=str, required=True,
                        help='Path to context JSON file containing latent_context_map')
    parser.add_argument('--top_k_list', type=int, nargs='+', default=[50, 100, 150, 200, 250, 300, 350, 400],
                        help='List of top-K values to extract (default: %(default)s)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save output JSON files (default: current directory)')
    parser.add_argument('--output_name', type=str, default='a',
                        help='Directory to save output JSON files (default: current directory)')

    args = parser.parse_args()
    extract_top_latents(
        model_name=args.model_name,
        context_path=args.context_path,
        top_k_list=args.top_k_list,
        output_dir=args.output_dir
        output_name=args.output_name
    )
