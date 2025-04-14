import os
import re
import time
import nltk
import json
import yaml
import torch
import heapq
import wandb
import datasets
import random
import tiktoken
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import *
from typing import List, Union
from tqdm import tqdm
from torch.optim import Adam
from openai import AzureOpenAI
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.hooks import RemovableHandle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Configuration on sparse autoencoders pipeline')

    parser.add_argument('--model_size', type=str, required=True, help='Language model name (e.g., "Llama-3.2-1B")')
    parser.add_argument('--model_path', type=str, required=True, help='Language model path')
    parser.add_argument('--sequence_or_token', type=str, required=True, help='Training SAE in sequence level or token level (e.g., "sequence" or "token")')
    parser.add_argument('--hidden_size', type=int, required=True, help='Dimensionality of the input residual stream activation')
    parser.add_argument('--latent_size', type=int, required=True, help='Size of the latent space')
    parser.add_argument('--k', type=int, required=True, help='Hyperparameter k for TopK')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--max_length', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--device', type=str, required=True, help='Device to run the model on (e.g., "cuda:0", "cpu")')
    parser.add_argument('--use_wandb', type=int, required=True, help='Whether to use wandb for logging')
    
    parser.add_argument('--dataset_name', type=str, required=False, help='dataset name')
    parser.add_argument('--data_path', type=str, required=False, help='Path to the dataset')
    parser.add_argument('--wandb_project', type=str, required=False, help='Wandb project name')
    parser.add_argument('--num_epochs', type=int, required=False, help='Number of training epochs')
    parser.add_argument('--lr', type=float, required=False, help='Learning rate for training')
    parser.add_argument('--betas', type=float, nargs=2, required=False, help='Beta values for the optimizer')
    parser.add_argument('--seed', type=int, required=False, help='Random seed for reproducibility')
    parser.add_argument('--layer', type=int, required=False, help='Target layer index, start with 1')
    parser.add_argument('--steps', type=int, required=False, help='Number of step batches for unit norm decoder')


    parser.add_argument('--resume_from', type=str, required=False, help='Path to a pretrained SAE state dict')
    parser.add_argument('--SAE_path', type=str, required=False, help='Path to the trained SAE model file')
    parser.add_argument('--metric', type=str, required=False, help='Evaluation metric (e.g., "NormMSE", "DeltaCE", "KLDiv")')

    parser.add_argument('--api_base', type=str, required=False, help='OpenAI api base')
    parser.add_argument('--api_key', type=str, required=False, help='OpenAI api key')
    parser.add_argument('--api_version', type=str, required=False, help='OpenAI api version')
    parser.add_argument('--engine', type=str, required=False, help='OpenAI api engine (e.g., "gpt-4o", "gpt-4o-mini")')

    parser.add_argument('--pipe_data_path', type=str, nargs='+', required=False, help='Path to the pipe dataset: train, eval and apply')
    parser.add_argument('--pipe_project', type=str, nargs='+', required=False, help='Wandb project name for pipe: train, eval and pipe')

    args = parser.parse_args()
    return args


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_config(config_path: str) -> Config:
    # load config from yaml file
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


def validate_english_text(text: str, allow_extended_ascii=False) -> bool:
    """
    综合验证是否为英语文本\n
    text: 输入字符串\n
    allow_extended_ascii: 是否允许扩展ASCII字符（如é, ñ）\n
    return: True表示全英语
    """
    # 定义ASCII及扩展ASCII范围
    max_code = 0xFF if allow_extended_ascii else 0x7F

    for char in text:
        code = ord(char)
        if code > max_code:
            return False
    return True



class OpenWebTextDataset(Dataset):
    def __init__(self, 
                 folder_path: str, 
                 tokenizer: AutoTokenizer, 
                 max_length: int,
                 keyword: str = 'text'):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keyword = keyword
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]

        self.data = self.load_data()

    
    def load_data(self):
        sent_token_num = []
        data = []
        for file_name in self.file_list:
            file_path = os.path.join(self.folder_path, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        raw_text = record['text']
                        spilited_sents = nltk.tokenize.sent_tokenize(raw_text)
                        for sent in spilited_sents:
                            if not validate_english_text(sent, allow_extended_ascii=True):
                                continue

                            inputs = self.tokenizer(
                                sent,
                                return_tensors='pt',
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True
                            )
                            seq_len = torch.sum(inputs['input_ids'][0]!=self.tokenizer.pad_token_id).item()
                            sent_token_num.append(seq_len)
                            
                            input_ids = inputs['input_ids'].squeeze(0)
                            attention_mask = inputs['attention_mask'].squeeze(0)
                        
                            if seq_len>9:
                                data.append((input_ids, attention_mask))

                    except json.JSONDecodeError:
                        print(f'Error decoding JSON in file: {file_path}')
                        continue

        a = np.array(sent_token_num)
        print('Sentence Lens Quantile'.center(80, '='))
        print(np.quantile(a,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SkyworkPreferenceDataset(Dataset):
    def __init__(self, 
                 pref_data_path : str,
                 tokenizer: AutoTokenizer, 
                 max_length: int,
        ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pref_data_path = pref_data_path
        self.data = self.load_data()
        ck=0

    def load_data(self):
        # for i in range(len(ds)):
        #     prompt1 = self.tokenizer.apply_chat_template(ds[i]['chosen'], tokenize=False, add_generation_prompt=False).replace(self.tokenizer.bos_token, "")
        #     prompt2 = self.tokenizer.apply_chat_template(ds[i]['rejected'], tokenize=False, add_generation_prompt=False).replace(self.tokenizer.bos_token, "")
        #     tokenized_p1 = self.tokenizer(prompt1, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        #     tokenized_p2 = self.tokenizer(prompt2, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        #     ret.append({
        #         'input_ids': tokenized_p1['input_ids'],
        #         'attention_mask': tokenized_p1['attention_mask'],
        #     })
        #     ret.append({
        #         'input_ids': tokenized_p1['input_ids'],
        #         'attention_mask': tokenized_p2['attention_mask'],
        #     })
        #     seq_len1 = torch.max(tokenized_p1['attention_mask']*torch.arange(self.max_length), dim=1).values
        #     seq_len2 = torch.max(tokenized_p2['attention_mask']*torch.arange(self.max_length), dim=1).values
        #     sent_token_num.append(seq_len1.item())
        #     sent_token_num.append(seq_len2.item())
        # a = np.array(sent_token_num)
        # print('Sentence Lens Quantile'.center(80, '='))
        # print(np.quantile(a,[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]))
        # 
        # for f in ds_spilit:
        #     prompt = self.tokenizer.apply_chat_template(
        #         f['text'], tokenize=True, add_generation_prompt=False)
        #     sent_token_num.append(len(prompt))
        
        ds = datasets.load_dataset('parquet', data_dir=self.pref_data_path)['train'].shuffle(seed=42)
        tokenizer = self.tokenizer
        def tokenize(sample):
            prompt = tokenizer.apply_chat_template(
                sample['text'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")

            tokenized_pos = tokenizer(
                prompt,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            return tokenized_pos
        return datasets.Dataset.from_dict({'text':ds['chosen']+ds['rejected']}).map(tokenize, num_proc=8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloader(
    dataset_name: str,
    folder_path: str, 
    tokenizer: AutoTokenizer, 
    batch_size: int, 
    max_length: int,
    keyword: str = 'text'
) -> DataLoader:
    if dataset_name=='OpenWebText':
        dataset = OpenWebTextDataset(folder_path, tokenizer, max_length, keyword)
    elif dataset_name=='Skywork-Reward-Preference-80K':
        dataset = SkyworkPreferenceDataset(folder_path, tokenizer, max_length)
    else:
        return None 

    
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item['input_ids']).squeeze() for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']).squeeze() for item in batch])
        return input_ids, attention_mask

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    return dataloader


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wandb_init(project: str, config: dict, name: str) -> None:
    wandb.init(
        project=project,
        config=config,
        name=name
    )


def get_language_model(model_path: str, device: torch.device) -> tuple:
    '''
    Loads and returns a tokenizer and a language model from the specified model path.
    '''
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, return_dict_in_generate=True, output_hidden_states=True
    ).to(device)
    return tokenizer, language_model


def get_outputs(
    cfg, batch: tuple, language_model: nn.Module, device: torch.device
) -> tuple:
    '''
    Extracts model outputs and hidden states from a given batch and language model.
    '''
    input_ids, attention_mask = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[cfg.layer]

    return input_ids, attention_mask, outputs, hidden_states


def pre_process(hidden_stats: torch.Tensor, eps: float = 1e-6) -> tuple:
    '''
    :param hidden_stats: Hidden states (shape: [batch, max_length, hidden_size]).
    :param eps: Epsilon value for numerical stability.
    '''
    mean = hidden_stats.mean(dim=-1, keepdim=True)
    std = hidden_stats.std(dim=-1, keepdim=True)
    x = (hidden_stats - mean) / (std + eps)
    return x, mean, std


def Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()


@torch.no_grad()
def unit_norm_decoder(model: TopkSAE) -> None:
    '''
    Normalize the decoder weights to unit norm
    '''
    model.decoder.weight.data /= model.decoder.weight.data.norm(dim=0)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Saved data to {path}')


def hook_SAE(
    cfg,
    model: TopkSAE,
    hooked_module: nn.Module,
) -> List[RemovableHandle]:

    def hook(module: torch.nn.Module, _, outputs):
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = [outputs]
        
        x, mu, std = pre_process(unpack_outputs[0])
        latents = model.encode(x)
        x_hat = model.decode(latents)
        unpack_outputs[0] = x_hat * std + mu

        if isinstance(outputs, tuple):
            return tuple(unpack_outputs)
        else:
            return unpack_outputs[0]

    handles = [hooked_module.register_forward_hook(hook)]
    return handles


class LinearWarmupLR(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, max_lr):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.max_lr = max_lr
        lr_lambda = self.lr_lambda
        super().__init__(optimizer, lr_lambda)

    def lr_lambda(self, step: int):
        if step < self.num_warmup_steps:
            return float(step) / float(max(1, self.num_warmup_steps))
        elif step < self.num_training_steps - self.num_training_steps // 5:
            return 1.0
        else:
            decay_steps = self.num_training_steps - self.num_warmup_steps - self.num_training_steps // 5
            return max(0.0, float(self.num_training_steps - step - self.num_warmup_steps) / float(decay_steps))


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.dataset_name ,cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)

        self.title = f'{cfg.sequence_or_token}_Latent{cfg.latent_size}_Layer{cfg.layer}_K{cfg.k}_{cfg.model_size}_{cfg.pipe_data_path[0].split('/')[-1]}'
        if self.cfg.resume_from is not None:
            self.title = f'{'token' if 'token' in self.cfg.resume_from else 'sequence'}_Adapted_{cfg.sequence_or_token}_Latent{cfg.latent_size}_Layer{cfg.layer}_K{cfg.k}_{cfg.model_size}_{cfg.pipe_data_path[0].split("/")[-1]}'
        self.config_dict = {
            'batch_size': self.cfg.batch_size,
            'num_epochs': self.cfg.num_epochs,
            'lr': self.cfg.lr,
            'steps': self.cfg.steps
        }
        self.model = TopkSAE(cfg.hidden_size, cfg.latent_size, cfg.k)
        if self.cfg.resume_from is not None:
            
            self.model.load_state_dict(torch.load(cfg.resume_from, weights_only=False))
            print(f'weights loaded from {cfg.resume_from}')
        self.model.to(self.device)
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, betas=cfg.betas)
        
        num_training_steps = cfg.num_epochs * len(self.dataloader)
        num_warmup_steps = int(num_training_steps * 0.05)
        self.scheduler = LinearWarmupLR(self.optimizer, num_warmup_steps, num_training_steps, cfg.lr)

    def run(self):
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        curr_loss = 0.0
        unit_norm_decoder(self.model)
        for epoch in range(self.cfg.num_epochs):
            for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SAE training"):
                _, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
                # hidden_states=(bz, seq_len, d_model)

                if self.cfg.sequence_or_token=='sequence':
                    position_ids = torch.arange(self.cfg.max_length, dtype=torch.long)
                    position_ids = position_ids * batch[1]
                    seq_len = torch.max(position_ids, dim=1).values

                    # hidden_states=(bz, seq_len, d_model) -> (bz, d_model) 选出每个sequence的last token's hidden state
                    hidden_states = torch.concat([hidden_states[i,pos,:].unsqueeze(0) for i,pos in enumerate(seq_len)], dim=0)
                    
                x, _, _ = pre_process(hidden_states)
                _, x_hat = self.model(x)
                loss = Normalized_MSE_loss(x, x_hat)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                curr_loss = loss.item()
                if batch_idx % self.cfg.steps == 0:
                    unit_norm_decoder(self.model)
                if self.cfg.use_wandb:
                    wandb.log({'Normalized_MSE': curr_loss})
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {curr_loss}')
        
        if self.cfg.use_wandb:
            wandb.finish()
        unit_norm_decoder(self.model)
        torch.save(self.model.state_dict(), f'../SAE_models/{self.title}.pt')
        return curr_loss


class Evaluater:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.sequence_or_token, cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        self.title = f'{os.path.splitext(os.path.basename(cfg.SAE_path))[0]}_{os.path.basename(cfg.data_path)}_{cfg.metric}'
        self.config_dict = {
            'batch_size': self.cfg.batch_size,
        }
        self.model = TopkSAE(cfg.hidden_size, cfg.latent_size, cfg.k)
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        if cfg.metric == 'KLDiv':
            self.hooked_module = self.language_model.get_submodule(f'model.layers.{cfg.layer-1}')
        
        self.num_batches = 0
        self.total_loss = 0.0
        self.total_counts = 0.0

    def KLDiv(
        self, logits_original: torch.Tensor, logits_reconstruct: torch.Tensor
    ) -> torch.Tensor:
        probs_original = F.softmax(logits_original, dim=-1)
        log_probs_reconstruct = F.log_softmax(logits_reconstruct, dim=-1)
        loss = F.kl_div(log_probs_reconstruct, probs_original, reduction='batchmean')
        return loss
    
    @torch.no_grad()
    def run(self):
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SAE evaluating"):
            input_ids, attention_mask, outputs, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)

            if self.cfg.sequence_or_token=='sequence':
                _, _, max_pos = batch
                max_pos=max_pos.to(self.device)
                # hidden_states=(bz, seq_len, d_model) -> (bz, d_model) 选出每个sequence的last token's hidden state
                hidden_states = torch.concat([hidden_states[i,pos,:].unsqueeze(0) for i,pos in enumerate(max_pos)], dim=0)

            x, _, _ = pre_process(hidden_states)
            _, x_hat = self.model(x)

            if self.cfg.metric == 'NormMSE': 
                loss = Normalized_MSE_loss(x, x_hat)
            
            elif self.cfg.metric == 'KLDiv':
                logits_original = outputs.logits
                handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                logits_reconstruct = self.language_model(input_ids=input_ids, attention_mask=attention_mask).logits
                for handle in handles:
                    handle.remove()
                del input_ids, attention_mask
                torch.cuda.empty_cache()
                loss = self.KLDiv(logits_original, logits_reconstruct)
                
            else:
                raise ValueError(f"Invalid metric: {self.cfg.metric}. Expected one of ['NormMSE', 'DeltaCE', 'KLDiv']")

            self.num_batches += 1
            self.total_loss += loss.item()

            if self.cfg.use_wandb:
                wandb.log({'Batch_loss': loss.item()})
            else:
                print(f'Batch: {batch_idx+1}, Loss: {loss.item()}')
        
        if self.cfg.use_wandb:
            wandb.log({'Avg_loss': self.total_loss / self.num_batches})
            wandb.finish()
        return self.total_loss / self.num_batches



class SeqApplier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = TopkSAE(cfg.hidden_size, cfg.latent_size, cfg.k)
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def validate_triples(self, triple_list: List[tuple[int, float, int]], name: str) -> None:
        if triple_list is not None:
            for i, (a, b, c) in enumerate(triple_list):
                if not (0 <= a < self.cfg.latent_size):
                    raise ValueError(f'Element {a} in {name} at index {i} is out of latent size range [0, {self.cfg.latent_size}).')
                if b <= 0:
                    raise ValueError(f'Value {b} in {name} at index {i} is not bigger than zero')
                if c not in [0, 1]:
                    raise ValueError(f'Mode {c} in {name} at index {i} must be 0 or 1.')
                
    @torch.no_grad()
    def get_context(
        self, 
        threshold: float = 10.0, 
        max_length: int = 64, 
        max_per_token: int = 2, 
        lines: int = 4,  
        output_path=None
    ):
    # get_context 需要修改，仅针对特定的latents进行context提取。
        if output_path is None:
            output_path = f'../contexts/{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}_{threshold}.json'

        sentence_enders = {'.', '!', '?', '<|end_of_text|>', '"'}
        half_length = max_length // 2

        latent_context_map = defaultdict(lambda: defaultdict(list))

        def find_sentence_bounds(seq_pos: int, tokens: List[str]):
            start_pos = seq_pos
            while start_pos > 0 and tokens[start_pos - 1] not in sentence_enders:
                start_pos -= 1
            end_pos = seq_pos
            while end_pos < len(tokens) - 1 and tokens[end_pos] not in sentence_enders:
                end_pos += 1
            if end_pos < len(tokens):
                end_pos += 1  
            return start_pos, end_pos

        def process_and_store_context(
            latent_dim: int, seq_pos: int, activation_value: float, tokens: List[str]
        ):
            start_pos, end_pos = find_sentence_bounds(seq_pos, tokens)
            sentence_tokens = tokens[start_pos:end_pos]
            sentence_length = len(sentence_tokens)

            if sentence_length > max_length:
                activated_token_idx = seq_pos - start_pos
                left_context_start = max(0, activated_token_idx - half_length)
                right_context_end = min(sentence_length, activated_token_idx + half_length + 1)
                context_tokens = sentence_tokens[left_context_start:right_context_end]
                activated_token_idx -= left_context_start
            else:
                context_tokens = sentence_tokens
                activated_token_idx = seq_pos - start_pos

            if not (0 <= activated_token_idx < len(context_tokens)):
                return

            context_tokens = context_tokens.copy()
            raw_token = context_tokens[activated_token_idx]
            context_tokens[activated_token_idx] = f'<ACTIVATED>{raw_token}</ACTIVATED>'

            while context_tokens and context_tokens[0] in ['<|end_of_text|>', ' ', '']:
                context_tokens.pop(0)
                activated_token_idx -= 1
            while context_tokens and context_tokens[-1] in ['<|end_of_text|>', ' ', '']:
                context_tokens.pop()

            if not context_tokens or not (0 <= activated_token_idx < len(context_tokens)):
                return

            context_text = self.tokenizer.convert_tokens_to_string(context_tokens).strip().strip('"')
            if not context_text:
                return

            activated_token_str = context_tokens[activated_token_idx]
            if activated_token_str.startswith('<ACTIVATED>') and activated_token_str.endswith('</ACTIVATED>'):
                raw_token = activated_token_str[len('<ACTIVATED>'):-len('</ACTIVATED>')].strip()
            else:
                raw_token = activated_token_str.strip()

            token_class = raw_token.lower()

            heap = latent_context_map[latent_dim][token_class]
            heapq.heappush(heap, (activation_value, context_text))
            if len(heap) > max_per_token:
                heapq.heappop(heap)

        self.tokenizer, self.language_model = get_language_model(self.cfg.model_path, self.device)
        self.dataloader = create_dataloader(self.cfg.sequence_or_token, self.cfg.data_path, self.tokenizer, self.cfg.batch_size, self.cfg.max_length)
        
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            input_ids, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
            x, _, _ = pre_process(hidden_states)

            latents, _ = self.model(x)
            batch_size, seq_len, _ = latents.shape
            positions = (latents > threshold)

            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                latent_indices = torch.nonzero(positions[i], as_tuple=False)

                for activation in latent_indices:
                    seq_pos, latent_dim = activation.tolist()
                    activation_value = latents[i, seq_pos, latent_dim].item()
                    process_and_store_context(latent_dim, seq_pos, activation_value, tokens)

        filtered_latent_context = {}
        for latent_dim, token_dict in latent_context_map.items():
            # # Skip latent token categories exceeding 32
            if len(token_dict) > 32:
                continue    
            total_contexts = sum(len(contexts) for contexts in token_dict.values())
            if total_contexts > lines:
                sorted_token_dict = {}
                for t_class, heap in token_dict.items():
                    contexts_list = list(heap)
                    contexts_list.sort(key=lambda x: x[0], reverse=True)
                    sorted_token_dict[t_class] = [
                        {'context': ctx, 'activation': act} for act, ctx in contexts_list
                    ]
                filtered_latent_context[latent_dim] = dict(sorted(sorted_token_dict.items()))

        total_latents = len(filtered_latent_context)
        sorted_latent_context = dict(sorted(filtered_latent_context.items()))

        output_data = {
            'total_latents': total_latents,
            'threshold': threshold,
            'max_length': max_length,
            'max_per_token': max_per_token,
            'lines': lines,
            'latent_context_map': sorted_latent_context,
        }
        save_json(output_data, output_path)
        return total_latents, output_path


class Applier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = TopkSAE(cfg.hidden_size, cfg.latent_size, cfg.k)
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def validate_triples(self, triple_list: List[tuple[int, float, int]], name: str) -> None:
        if triple_list is not None:
            for i, (a, b, c) in enumerate(triple_list):
                if not (0 <= a < self.cfg.latent_size):
                    raise ValueError(f'Element {a} in {name} at index {i} is out of latent size range [0, {self.cfg.latent_size}).')
                if b <= 0:
                    raise ValueError(f'Value {b} in {name} at index {i} is not bigger than zero')
                if c not in [0, 1]:
                    raise ValueError(f'Mode {c} in {name} at index {i} must be 0 or 1.')
                
    @torch.no_grad()
    def get_context(
        self, 
        threshold: float = 10.0, 
        max_length: int = 64, 
        max_per_token: int = 2, 
        lines: int = 4,  
        output_path=None
    ):
    # get_context 需要修改，仅针对特定的latents进行context提取。
        if output_path is None:
            output_path = f'../contexts/{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}_{threshold}.json'

        sentence_enders = {'.', '!', '?', '<|end_of_text|>', '"'}
        half_length = max_length // 2

        latent_context_map = defaultdict(lambda: defaultdict(list))

        def find_sentence_bounds(seq_pos: int, tokens: List[str]):
            start_pos = seq_pos
            while start_pos > 0 and tokens[start_pos - 1] not in sentence_enders:
                start_pos -= 1
            end_pos = seq_pos
            while end_pos < len(tokens) - 1 and tokens[end_pos] not in sentence_enders:
                end_pos += 1
            if end_pos < len(tokens):
                end_pos += 1  
            return start_pos, end_pos

        def process_and_store_context(
            latent_dim: int, seq_pos: int, activation_value: float, tokens: List[str]
        ):
            start_pos, end_pos = find_sentence_bounds(seq_pos, tokens)
            sentence_tokens = tokens[start_pos:end_pos]
            sentence_length = len(sentence_tokens)

            if sentence_length > max_length:
                activated_token_idx = seq_pos - start_pos
                left_context_start = max(0, activated_token_idx - half_length)
                right_context_end = min(sentence_length, activated_token_idx + half_length + 1)
                context_tokens = sentence_tokens[left_context_start:right_context_end]
                activated_token_idx -= left_context_start
            else:
                context_tokens = sentence_tokens
                activated_token_idx = seq_pos - start_pos

            if not (0 <= activated_token_idx < len(context_tokens)):
                return

            context_tokens = context_tokens.copy()
            raw_token = context_tokens[activated_token_idx]
            context_tokens[activated_token_idx] = f'<ACTIVATED>{raw_token}</ACTIVATED>'

            while context_tokens and context_tokens[0] in ['<|end_of_text|>', ' ', '']:
                context_tokens.pop(0)
                activated_token_idx -= 1
            while context_tokens and context_tokens[-1] in ['<|end_of_text|>', ' ', '']:
                context_tokens.pop()

            if not context_tokens or not (0 <= activated_token_idx < len(context_tokens)):
                return

            context_text = self.tokenizer.convert_tokens_to_string(context_tokens).strip().strip('"')
            if not context_text:
                return

            activated_token_str = context_tokens[activated_token_idx]
            if activated_token_str.startswith('<ACTIVATED>') and activated_token_str.endswith('</ACTIVATED>'):
                raw_token = activated_token_str[len('<ACTIVATED>'):-len('</ACTIVATED>')].strip()
            else:
                raw_token = activated_token_str.strip()

            token_class = raw_token.lower()

            heap = latent_context_map[latent_dim][token_class]
            heapq.heappush(heap, (activation_value, context_text))
            if len(heap) > max_per_token:
                heapq.heappop(heap)

        self.tokenizer, self.language_model = get_language_model(self.cfg.model_path, self.device)
        self.dataloader = create_dataloader(self.cfg.sequence_or_token, self.cfg.data_path, self.tokenizer, self.cfg.batch_size, self.cfg.max_length)
        
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SAE applying"):
            input_ids, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
            x, _, _ = pre_process(hidden_states)

            latents, _ = self.model(x)
            batch_size, seq_len, _ = latents.shape
            positions = (latents > threshold)

            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                latent_indices = torch.nonzero(positions[i], as_tuple=False)

                for activation in latent_indices:
                    seq_pos, latent_dim = activation.tolist()
                    activation_value = latents[i, seq_pos, latent_dim].item()
                    process_and_store_context(latent_dim, seq_pos, activation_value, tokens)

        filtered_latent_context = {}
        for latent_dim, token_dict in latent_context_map.items():
            # # Skip latent token categories exceeding 32
            if len(token_dict) > 32:
                continue    
            total_contexts = sum(len(contexts) for contexts in token_dict.values())
            if total_contexts > lines:
                sorted_token_dict = {}
                for t_class, heap in token_dict.items():
                    contexts_list = list(heap)
                    contexts_list.sort(key=lambda x: x[0], reverse=True)
                    sorted_token_dict[t_class] = [
                        {'context': ctx, 'activation': act} for act, ctx in contexts_list
                    ]
                filtered_latent_context[latent_dim] = dict(sorted(sorted_token_dict.items()))

        total_latents = len(filtered_latent_context)
        sorted_latent_context = dict(sorted(filtered_latent_context.items()))

        output_data = {
            'total_latents': total_latents,
            'threshold': threshold,
            'max_length': max_length,
            'max_per_token': max_per_token,
            'lines': lines,
            'latent_context_map': sorted_latent_context,
        }
        save_json(output_data, output_path)
        return total_latents, output_path


class Interpreter:
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate_cost(self, input_text: str, output_text: str) -> float:
        encoding = tiktoken.encoding_for_model(self.cfg.engine)
        num_input_tokens = len(encoding.encode(input_text))
        num_output_tokens = len(encoding.encode(output_text))
        if self.cfg.engine == 'gpt-4o':
            return num_input_tokens * 2.5 / 1_000_000 + num_output_tokens * 10 / 1_000_000
        elif self.cfg.engine == 'gpt-4o-mini':
            return num_input_tokens * 0.15 / 1_000_000 + num_output_tokens * 0.6 / 1_000_000
        else:
            return 0.0
    
    def construct_prompt(self, tokens_info: dict) -> str:
        prompt = (
            'We are analyzing the activation levels of features in a language model, where each feature activates certain tokens in a text.\n'
            'Each token\'s activation value indicates its relevance to the feature, with higher values showing stronger association.\n'
            'Your task is to give this feature a monosemanticity score based on the following scoring rubric:\n'
            'Activation Consistency\n'
            '5: Clear pattern with no deviating examples\n'
            '4: Clear pattern with one or two deviating examples\n'
            '3: Clear overall pattern but quite a few examples not fitting that pattern\n'
            '2: Broad consistent theme but lacking structure\n'
            '1: No discernible pattern\n'
            'Consider the following activations for a feature in the language model.\n\n'
        )
        for info in tokens_info:
            prompt += f"Token: {info['token']} | Activation: {info['activation']} | Context: {info['context']}\n\n"
        prompt += (
            'Provide your response in the following fixed format:\n'
            'Score: [5/4/3/2/1]\n'
            'Explanation: [Your brief explanation]\n'
        )
        return prompt

    def chat_completion(
        self, client: AzureOpenAI, prompt: str, max_retry: int=3
    ) -> str:
        assert client is not None, 'Client is not set'
        for attempt in range(1, max_retry + 1):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are an assistant that helps explain the latent semantics of language models.',
                        },
                        {'role': 'user', 'content': prompt},
                    ],
                    model=self.cfg.engine,
                    max_tokens=128,  
                    temperature=0.1,
                )
                response_content = chat_completion.choices[0].message.content
                assert response_content is not None, 'Response is None'
                return response_content.strip()
            except Exception as e:
                if attempt == max_retry:
                    print('Failed to get a response from the OpenAI API after multiple attempts.')
                    raise e  
        raise Exception('Failed to get a response from the OpenAI API')
    
    def run(
        self, data_path: str=None, sample_latents: int=100, output_path: str=None
    ) -> float:
        if data_path is None:
            data_path = self.cfg.data_path

        if output_path is None:
            output_path = f'../interpret/interp_{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        latent_context_map = data.get('latent_context_map', {})
        all_latents = list(latent_context_map.keys())
        sample_size = min(sample_latents, len(all_latents))
        sampled_indices = random.sample(range(len(all_latents)), sample_size)
        sampled_latents = [all_latents[i] for i in sorted(sampled_indices)]

        client = AzureOpenAI(
            azure_endpoint=self.cfg.api_base,
            api_version=self.cfg.api_version,
            api_key=self.cfg.api_key,
        )

        cost = 0.0
        results = {}
        total_score = 0.0
        scored_features = 0

        pattern = re.compile(
            r"Score:\s*(?P<score>[1-5])\s*\n"
            r"Explanation:\s*(?P<explanation>.+)",
            re.IGNORECASE | re.DOTALL,
        )

        for latent in sampled_latents:
            try:
                latent_id = int(latent)
            except ValueError:
                print(f"Invalid latent ID {latent}. Skipping.")
                results[latent] = {
                    'score': None,
                    'explanation': "Invalid latent ID.",
                }
                continue
            token_contexts = latent_context_map[latent]
            tokens_info = []
            for token_class, contexts in token_contexts.items():
                for context in contexts:
                    token = token_class
                    if token.startswith('ġ'):
                        token = ' ' + token[1:]
                    tokens_info.append({
                        'token': token,
                        'context': context['context'],
                        'activation': context['activation'],
                    })

            prompt = self.construct_prompt(tokens_info)
            try:
                response = self.chat_completion(client, prompt)
                cost += self.calculate_cost(prompt, response)
                match = pattern.search(response)
                if match:
                    score = int(match.group('score'))
                    explanation = match.group('explanation').strip()
                    if 1 <= score <= 5:
                        results[latent_id] = {
                            'score': score,
                            'explanation': explanation,
                        }
                        total_score += score
                        scored_features += 1
                    else:
                        print(f"Invalid score '{score}' for latent {latent_id}. Skipping.")
                        results[latent_id] = {
                            'score': None,
                            'explanation': "Invalid score provided.",
                        }
                else:
                    print(f"Failed to parse response for latent {latent_id}. Response: {response}")
                    results[latent_id] = {
                        'score': None,
                        'explanation': "Failed to parse response.",
                    }
            except Exception as e:
                print(f"Error processing latent {latent_id}: {e}")
                results[latent_id] = {
                    'score': None,
                    'explanation': "Error during processing.",
                }
                continue
        avg_score = total_score / scored_features if scored_features > 0 else 0.0
        output_data = {
            'cost': cost,
            'engine': self.cfg.engine,
            'features_scored': scored_features,
            'average_score': avg_score,
            'results': results,
        }
        save_json(output_data, output_path)
        return avg_score


class SAE_pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.title = f'{cfg.sequence_or_token}_Latent{cfg.latent_size}_Layer{cfg.layer}_K{cfg.k}_{cfg.model_size}_{cfg.pipe_data_path[0].split('/')[-1]}'
        self.cfg.SAE_path = f'../SAE_models/{self.title}.pt'
        self.result_dict = {}
    
    def train(self):
        set_seed(self.cfg.seed)
        self.cfg.data_path = self.cfg.pipe_data_path[0]
        self.cfg.wandb_project = self.cfg.pipe_project[0]
        trainer = Trainer(self.cfg)
        self.result_dict['Train_Loss'] = trainer.run()
        del trainer
        torch.cuda.empty_cache()
    
    def evaluate(self):
        self.cfg.data_path = self.cfg.pipe_data_path[1]
        self.cfg.wandb_project = self.cfg.pipe_project[1]
        self.cfg.batch_size = self.cfg.batch_size // 2
        for metric in ['NormMSE','KLDiv']:
            self.cfg.metric = metric
            evaluater = Evaluater(self.cfg)
            self.result_dict[f'{metric}'] = evaluater.run()
            del evaluater
            torch.cuda.empty_cache()

    def apply(self):
        self.cfg.data_path = self.cfg.pipe_data_path[2]
        applier = Applier(self.cfg)
        self.result_dict[f'Features'], self.context_path = applier.get_context(
            threshold=15, max_length=64, max_per_token=2, lines=4
        )
        del applier
        torch.cuda.empty_cache

    def interpret(self):
        self.cfg.data_path = self.context_path
        interpreter = Interpreter(self.cfg)
        score = interpreter.run(sample_latents=100)
        self.result_dict[f'Score'] = score
        del interpreter

    def run(self):
        start_time = time.time()

        self.train()
        # self.evaluate()
        # self.apply()
        # self.interpret()

        end_time = time.time()
        self.result_dict['Runtime'] = (end_time - start_time) / 3600

        if self.cfg.use_wandb:
            wandb_init(self.cfg.pipe_project[2], self.result_dict, self.title)
            wandb.finish()
            

