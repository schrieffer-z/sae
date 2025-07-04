import os
import re
import time
import nltk
import json
import yaml
import math
import numpy as np
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
from typing import List, Union
from tqdm import tqdm
from torch.optim import Adam
from openai import AzureOpenAI, OpenAI
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.hooks import RemovableHandle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# local
from model import *


def parse_args():
    parser = argparse.ArgumentParser(description='Configuration on sparse autoencoders pipeline')

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


    parser.add_argument('--apply_threshold', type=float, required=False)
    parser.add_argument('--resume_from', type=str, required=False, help='Path to a pretrained SAE state dict')
    parser.add_argument('--SAE_path', type=str, required=False, help='Path to the trained SAE model file')
    parser.add_argument('--metric', type=str, required=False, help='Evaluation metric (e.g., "NormMSE", "DeltaCE", "KLDiv")')

    parser.add_argument('--api_base', type=str, required=False, help='OpenAI api base')
    parser.add_argument('--api_key', type=str, required=False, help='OpenAI api key')
    parser.add_argument('--api_version', type=str, required=False, help='OpenAI api version')
    parser.add_argument('--engine', type=str, required=False, help='OpenAI api engine (e.g., "gpt-4o", "gpt-4o-mini")')

    parser.add_argument('--pipe_run', type=str, required=True, help='0000:nothing, train, evaluate, apply, interpret')
    parser.add_argument('--output_path', type=str, default="../SAE_models", help='0000:nothing, train, evaluate, apply, interpret')
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



class OpenWebText(Dataset):
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
        print(np.quantile(a,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Preference(Dataset):
    def __init__(self, 
                 pref_data_path : str,
                 tokenizer: AutoTokenizer, 
                 max_length: int,
        ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pref_data_path = pref_data_path
        self.data = self.load_data()

    def load_data(self):
        ds = datasets.load_dataset('parquet', data_dir=self.pref_data_path)['train'].shuffle(seed=42)
        tokenizer = self.tokenizer
        def tokenize(sample):
            formated_text = tokenizer.apply_chat_template(sample['text'], tokenize=False, add_generation_prompt=False)
            if tokenizer.bos_token!=None:
                formated_text=formated_text.replace(tokenizer.bos_token, "")
            return tokenizer(
                formated_text, 
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length'
            )
        return datasets.Dataset.from_dict({'text':ds['chosen']+ds['rejected']}).map(tokenize, num_proc=1)

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
    if dataset_name=='corpus':
        dataset = OpenWebText(folder_path, tokenizer, max_length, keyword)
    elif 'preference' in dataset_name.lower():
        dataset = Preference(folder_path, tokenizer, max_length)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


    def collate_fn(batch):
        input_ids = torch.stack([
                item[0] if dataset_name=='corpus'  
                else torch.tensor(item['input_ids']).squeeze()
                for item in batch
            ]
        )
        attention_mask = torch.stack([
                item[1] if dataset_name=='corpus'  
                else torch.tensor(item['attention_mask']).squeeze()
                for item in batch
            ]
        )
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


def get_language_model(cfg, model_path: str, device: torch.device) -> tuple:
    '''
    Loads and returns a tokenizer and a language model from the specified model path.
    '''
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model = MyLlamaModel.from_pretrained(
        model_path, hidden_state_source_layer=cfg.layer,trust_remote_code=True, return_dict_in_generate=True, output_hidden_states=True, torch_dtype=torch.bfloat16
    ).to(device)
    return tokenizer, language_model


def get_outputs(
    cfg, batch: tuple, language_model: nn.Module, device: torch.device
) -> tuple:
    '''
    Extracts model outputs and hidden states from a given batch and language model.
    '''
    input_ids, attention_mask = batch[0], batch[1]
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
    
def Masked_Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.device, x.dtype)
    loss = ((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)
    assert loss.shape==mask.shape
    seq_loss = (mask * loss).sum(-1) / (mask.sum(-1))
    return seq_loss.mean()


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
        self.tokenizer, self.language_model = get_language_model(cfg, cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.dataset_name ,cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        self.title = f'{cfg.sequence_or_token}_Latent{cfg.latent_size}_Layer{cfg.layer}_K{cfg.k}_{cfg.pipe_data_path[0].split('/')[-1]}'
        
        mp=self.cfg.model_path
        if 'Llama' in self.cfg.model_path:
            self.title = mp[mp.find('Llama'):]+'_'+self.title
        elif 'Qwen2.5' in self.cfg.model_path:
            self.title = mp[mp.find('Qwen2.5'):]+'_'+self.title
        elif 'gemma' in self.cfg.model_path:
            self.title = mp[mp.find('gemma'):]+'_'+self.title
        else:
            raise ValueError(f'Unsupport base model type from path {self.cfg.model_path}')

        if self.cfg.resume_from is not None:
            self.title = f'{'token' if 'token' in self.cfg.resume_from else 'sequence'}_Adapted_{cfg.sequence_or_token}_Latent{cfg.latent_size}_Layer{cfg.layer}_K{cfg.k}_{cfg.pipe_data_path[0].split("/")[-1]}'
        
        assert os.path.exists(self.cfg.output_path)
        
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

        self.model.to(torch.bfloat16).to(self.device)
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, betas=cfg.betas)
        
        num_training_steps = cfg.num_epochs * len(self.dataloader)
        num_warmup_steps = int(num_training_steps * 0.05)
        self.scheduler = LinearWarmupLR(self.optimizer, num_warmup_steps, num_training_steps, cfg.lr)

    def run(self):
        print(self.cfg.use_wandb)
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        curr_loss = 0.0
        global_step_idx = 0
        num_trained_tokens = 0
        unit_norm_decoder(self.model)
        for epoch in range(self.cfg.num_epochs):
            for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SAE training"):
                _, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
                # hidden_states=(bz, seq_len, d_model)

                if self.cfg.sequence_or_token=='sequence':
                    position_ids = torch.arange(self.cfg.max_length, dtype=torch.long)
                    position_ids = position_ids * batch[1]
                    seq_len = torch.max(position_ids, dim=1).values

                    batch_size = hidden_states.shape[0]
                    split_sent = [',', '?', '--', ';', '!', '\"', ' ,', ' ?', ' --', ' ;', ' !', ' \"']
                    split_sent_tokens = self.tokenizer.batch_encode_plus(split_sent, return_tensors='pt')['input_ids'][:,1].tolist()

                    h=[]
                    for i in range(batch_size):
                        for j in range(seq_len[i]+1):
                            if batch[0][i][j] in split_sent_tokens or j == seq_len[i]:
                                h.append(hidden_states[i,j,:].unsqueeze(0))
                            
                    # hidden_states=(bz, seq_len, d_model) -> (bz', d_model) 选出每个sequence包含在split_sent中的token's hidden state（包含",","--"作为加速手段）
                    hidden_states = torch.concat(h, dim=0)
                    
                num_trained_tokens += hidden_states.view(-1, hidden_states.shape[-1]).shape[0]
                x, _, _ = pre_process(hidden_states)
                _, x_hat = self.model(x)
                if self.cfg.sequence_or_token=='sequence':
                    loss = Normalized_MSE_loss(x, x_hat)
                elif self.cfg.sequence_or_token=='token':
                    loss = Masked_Normalized_MSE_loss(x, x_hat, batch[1])
                else:
                    raise ValueError(f'unsupported train level {self.cfg.sequence_or_token}')
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                global_step_idx+=1
                
                curr_loss = loss.item()
                if batch_idx % self.cfg.steps == 0:
                    unit_norm_decoder(self.model)
                if self.cfg.use_wandb:
                    wandb.log({'Normalized_MSE': curr_loss})
                
                save_ats = np.round(len(self.dataloader)*np.linspace(0,1,10))[1:-1].astype(np.int64)
                if (global_step_idx in save_ats):
                    title = self.title + f'@step{global_step_idx}'
                    mp=self.cfg.model_path
                    if 'Llama' in self.cfg.model_path:
                        title = mp[mp.find('Llama'):]+'_'+title
                    elif 'Qwen2.5' in self.cfg.model_path:
                        title = mp[mp.find('Qwen2.5'):]+'_'+title
                    elif 'gemma' in self.cfg.model_path:
                        title = mp[mp.find('gemma'):]+'_'+title
                    
                    os.makedirs(os.path.join(self.cfg.output_path, 'tmp'), exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(self.cfg.output_path, 'tmp', f'{title}.pt'))
        
        if self.cfg.use_wandb:
            wandb.finish()
        unit_norm_decoder(self.model)
        torch.save(self.model.state_dict(), os.path.join(self.cfg.output_path, f'{self.title}.pt'))
        print(f"trained on {num_trained_tokens/1_000}K tokens")
        return curr_loss


class Evaluater:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer, self.language_model = get_language_model(cfg, cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.dataset_name, cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
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
        threshold: float = 3.0, 
        max_length: int = 96, 
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

        self.tokenizer, self.language_model = get_language_model(self.cfg, self.cfg.model_path, self.device)
        self.dataloader = create_dataloader(self.cfg.dataset_name, self.cfg.data_path, self.tokenizer, self.cfg.batch_size, self.cfg.max_length)
        self.pos_latents = torch.load('../pos1.pt', weights_only=True).tolist()
        self.neg_latents = torch.load('../neg1.pt', weights_only=True).tolist()
        
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SAE applying"):
            input_ids, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
            x, _, _ = pre_process(hidden_states)

            latents, _ = self.model(x)
            batch_size, seq_len, _ = latents.shape
            positions = (latents > threshold)

            position_ids = torch.arange(self.cfg.max_length, dtype=torch.long)
            position_ids = position_ids * batch[1]
            seq_len = torch.max(position_ids, dim=1).values

            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                latent_indices = torch.nonzero(positions[i], as_tuple=False)

                for activation in latent_indices:
                    seq_pos, latent_dim = activation.tolist()
                    if seq_pos==0 or seq_pos>seq_len[i]:
                        continue
                    if (latent_dim not in self.pos_latents) and (latent_dim not in self.neg_latents):
                        continue
                    activation_value = latents[i, seq_pos, latent_dim].item()
                    process_and_store_context(latent_dim, seq_pos, activation_value, tokens)

        filtered_latent_context = {}
        for latent_dim, token_dict in latent_context_map.items():
            # # Skip latent token categories exceeding 32
            if len(token_dict) > 100:
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


def save_latent_dict(latent_context_map, output_path, threshold, max_length, max_per_token, lines):
    filtered_latent_context = {}
    for latent_dim, token_dict in latent_context_map.items():
        # # Skip latent token categories exceeding 32
        if len(token_dict) > 100:
            continue    
        total_contexts = sum(len(contexts) for contexts in token_dict.values())
        if total_contexts > 4:
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
    return


class SequenceApplier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = TopkSAE(cfg.hidden_size, cfg.latent_size, cfg.k)
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device).to(torch.bfloat16)
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
        threshold: float = 3.0, 
        max_length: int = 96, 
        max_per_token: int = 256, 
        lines: int = 4,  
        output_path=None
    ):
    # get_context 需要修改，仅针对特定的latents进行context提取。
        title = f'{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}_{threshold}.json'
        if self.cfg.output_path is None:
            output_path = os.path.join("../contexts/", title)
        else:
            output_path = os.path.join(self.cfg.output_path, title)

        self.tokenizer, self.language_model = get_language_model(self.cfg, self.cfg.model_path, self.device)
        self.dataloader = create_dataloader(self.cfg.dataset_name, self.cfg.data_path, self.tokenizer, self.cfg.batch_size, self.cfg.max_length)
        
        sentence_enders = [
            '?',    '.',   ';',    '!',     '"',
            ' ?',   ' .',  ' ;',   ' !',    ' "',
            '<|end_of_text|>',
            '<|eot_id|>' 
        ]
        sentence_enders_tokens = self.tokenizer.batch_encode_plus(sentence_enders, return_tensors='pt')['input_ids'][:,1].tolist()

        global_step_idx=0
        ckmap = defaultdict(lambda: defaultdict(list))
        latent_context_map = defaultdict(lambda: defaultdict(list))
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SAE applying"):
            input_ids, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
            x, _, _ = pre_process(hidden_states)

            latents, _ = self.model(x)
            batch_size, seq_len, _ = latents.shape
            positions = (latents > threshold)

            position_ids = torch.arange(self.cfg.max_length, dtype=torch.long)
            position_ids = position_ids * batch[1]
            seq_len = torch.max(position_ids, dim=1).values

            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                assert tokens[25] == '<|eot_id|>' # system message长度是固定值

                prev_pos = 26
                latent_indices = torch.nonzero(positions[i], as_tuple=False)
                for pos in range(26, seq_len[i]+1):
                    if input_ids[i][pos] not in sentence_enders_tokens:
                        continue
                    
                    mask = (latent_indices[:, 0] == pos)
                    latents_at_pos = latent_indices[mask]
                    for activation in latents_at_pos:
                        _pos, latent_dim = activation.tolist()
                        assert _pos==pos

                        activation_value = latents[i, pos, latent_dim].item()
                        raw = self.tokenizer.convert_tokens_to_ids(tokens[prev_pos:pos+1])
                        context_text = self.tokenizer.decode(raw).strip()
                        
                        assert len(ckmap)==len(latent_context_map)
                        heap = latent_context_map[latent_dim][tokens[pos]]
                        cmap = ckmap[latent_dim][tokens[pos]]
                        if context_text in cmap:
                            continue
                        heapq.heappush(cmap, context_text)
                        heapq.heappush(heap, (activation_value, context_text))
                        if len(heap) > max_per_token:
                            heapq.heappop(heap)
                        
                    prev_pos = pos + 1

            save_ats = np.round(len(self.dataloader)*np.linspace(0,1,11))[1:-1].astype(np.int64)
            if (global_step_idx in save_ats):
                base_t = os.path.join(self.cfg.output_path if self.cfg.output_path is not None else '../contexts/', 'tmp')
                title_t = f'{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}_{threshold}@step{global_step_idx}.json'
                
                os.makedirs(base_t, exist_ok=True)
                output_path_tmp = os.path.join(base_t, title_t)
                save_latent_dict(latent_context_map, output_path_tmp, threshold, max_length, max_per_token, lines)
            global_step_idx += 1


        filtered_latent_context = {}
        for latent_dim, token_dict in latent_context_map.items():
            # # Skip latent token categories exceeding 32
            if len(token_dict) > 100:
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
            "A reward model outputs a scalar representing the quality of a model-generated response. "
            "Responses with higher scores are more likely to align with human preferences given a particular question.\n\n"
            
            "A Sparse Autoencoder (SAE) extracts human-interpretable features from the hidden states of a language model when provided with the concatenation of a question-response pair. "
            "Ideally, each SAE feature activates only in response to a specific context type.\n\n"
            
            "Your task is to analyze whether the activation of a particular SAE feature (indicated by the presence of its corresponding context type) affects the likelihood that humans would prefer the response.\n"
            
            "Use the following scoring criteria:\n"
            "'-2': Activation of this feature (presence of this context) strongly decreases the likelihood of human preference.\n"
            "'-1': Activation of this feature (presence of this context) moderately decreases the likelihood of human preference.\n"
            "'0': Activation of this feature (presence of this context) has a neutral effect on human preference.\n"
            "'1': Activation of this feature (presence of this context) moderately increases the likelihood of human preference.\n"
            "'2': Activation of this feature (presence of this context) strongly increases the likelihood of human preference.\n\n"
            
            "Important Notes:\n"
            "- SAE feature activations occur when certain context types appear. Therefore, closely analyze the local semantics around the activated token, which is indicated between '<ACTIVATED>' and '</ACTIVATED>' tags.  \n"
            "- Context before the activated token is particularly significant and should weigh more in your assessment than the context after.\n\n"
            
            "Provide your analysis strictly in the format below:\n\n"
            "Consider the following context in which a particular SAE feature activates:\n\n"
        )
        for info in tokens_info:
            if "|>" in info['token'] or "<|" in info['token'] :
                continue
            ck = "|>\n\n" 
            if ck in info['context']:
                s = info['context']
                info['context'] = s[s.rfind(ck)+len(ck):]
            prompt += f"Token: {info['token']} | Context: {info['context']}\n\n"
        prompt += (
            'Provide your response in the following fixed format:\n'
            "Score: [-2/-1/0/1/2]"
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

        sampled_latents=[]
        pos_latents = torch.load('../pos_llama8b_sequence_Latent65536_Layer18_K192_1B.pt', weights_only=True).tolist()
        neg_latents = torch.load('../neg_llama8b_sequence_Latent65536_Layer18_K192_1B.pt', weights_only=True).tolist()
        latents = pos_latents + neg_latents
        latents_of_rm = [str(i) for i in latents]

        for f in latents_of_rm:
            if f in all_latents:
                sampled_latents.append(f)
                


        client = OpenAI(
            api_key=self.cfg.api_key,  
            base_url=self.cfg.api_base,
            timeout=60,
            max_retries=2
        )

        cost = 0.0
        results = {}
        total_score = 0.0
        scored_features = 0

        for latent in tqdm(sampled_latents):
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
                match = re.search(r"-?\d+", response)
                if match:
                    score = int(match.group(0))
                    if -2 <= score <= 2:
                        results[latent_id] = {
                            'score': score,
                            'contexts': [tokens_info[i]['context'] for i in range(len(tokens_info))]
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
        self.title = f'{cfg.sequence_or_token}_Latent{cfg.latent_size}_Layer{cfg.layer}_K{cfg.k}_{cfg.pipe_data_path[0].split('/')[-1]}'
        
        mp=self.cfg.model_path
        if 'Llama' in self.cfg.model_path:
            self.title = mp[mp.find('Llama'):]+'_'+self.title
        elif 'Qwen' in self.cfg.model_path:
            self.title = mp[mp.find('Qwen'):]+'_'+self.title
        elif 'gemma' in self.cfg.model_path:
            self.title = mp[mp.find('gemma'):]+'_'+self.title
        else:
            raise ValueError(f'Unsupport base model type from path {self.cfg.model_path}')
        if self.cfg.SAE_path is None:
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
        if self.cfg.sequence_or_token == 'token':
            applier = Applier(self.cfg)
        elif self.cfg.sequence_or_token == 'sequence':
            applier = SequenceApplier(self.cfg)
        else:
            raise ValueError(f'Unsupport train level--{self.cfg.sequence_or_token}')

        self.result_dict[f'Features'], self.context_path = applier.get_context(
            threshold=self.cfg.apply_threshold, max_length=96
        )
        del applier
        torch.cuda.empty_cache()

    def interpret(self):
        if self.cfg.pipe_run[2]=='0':
            self.context_path = f'../contexts/{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}_{self.cfg.apply_threshold}.json'
        self.cfg.data_path = self.context_path
        interpreter = Interpreter(self.cfg)
        score = interpreter.run(sample_latents=500)
        self.result_dict[f'Score'] = score
        del interpreter

    def run(self):
        start_time = time.time()

        if self.cfg.pipe_run[0]=='1':
            self.train()
        if self.cfg.pipe_run[1]=='1':
            self.evaluate()
        if self.cfg.pipe_run[2]=='1':
            self.apply()
        if self.cfg.pipe_run[3]=='1':
            self.interpret()

        end_time = time.time()
        self.result_dict['Runtime'] = (end_time - start_time) / 3600

        if self.cfg.use_wandb:
            wandb_init(self.cfg.pipe_project[2], self.result_dict, self.title)
            wandb.finish()
            

