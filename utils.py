import pandas as pd
import json
import os
import numpy as np
import random
from tqdm import tqdm
import re
from sample import Sampler

import ray
from ray.util import ActorPool
from tqdm import tqdm
import torch

def init_ray(num_cpus=100, num_gpus=None):
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
        )

@ray.remote(num_cpus=1)
def remote_sample_processing_bm25(train_df, test_data_chunk, max_sample_num):
    sample = Sampler(train_df)
    retriever, corpus = sample.init_bm25()
    all_samples_chunk = {}
    for test_item in test_data_chunk:
        clean_text = test_item['clean_text']
        samples = sample.top_bm25(retriever=retriever, query=test_item, corpus=corpus, n_samples=max_sample_num, random_state=42)
        all_samples_chunk[clean_text] = samples
    return all_samples_chunk

@ray.remote(num_cpus=1, num_gpus=1)
def remote_sample_processing_gte(train_df, test_data_chunk, max_sample_num):
    sample = Sampler(train_df)
    retriever, corpus = sample.init_gte()
    all_samples_chunk = {}
    for test_item in test_data_chunk:
        clean_text = test_item['clean_text']
        samples = sample.top_gte(retriever=retriever, query=test_item, corpus=corpus, n_samples=max_sample_num, random_state=42)
        all_samples_chunk[clean_text] = samples
    return all_samples_chunk

@ray.remote(num_cpus=1)
def remote_sample_processing_random(train_df, test_data_chunk, max_sample_num):
    sample = Sampler(train_df)
    all_samples_chunk = {}
    for test_item in test_data_chunk:
        clean_text = test_item['clean_text']
        samples = sample.random_samples_all(n_samples=max_sample_num, random_state=random.randint(0, 1000))
        all_samples_chunk[clean_text] = samples
    return all_samples_chunk

@ray.remote(num_cpus=1)
def remote_sample_processing_fixed(train_df, test_data_chunk, max_sample_num):
    sample = Sampler(train_df)
    all_samples_chunk = {}
    for test_item in test_data_chunk:
        clean_text = test_item['clean_text']
        samples = sample.fixed_samples_all(n_samples=max_sample_num, random_state=42)
        all_samples_chunk[clean_text] = samples
    return all_samples_chunk

def preprocess_samples(train_df, test_data, max_sample_num=80, config=None):
    if config.sample_strategy == 'top_gte':
        chunks_num = max(1, torch.cuda.device_count())
        tasks = [
            remote_sample_processing_gte.remote(train_df, chunk, max_sample_num) 
            for chunk in np.array_split(test_data, chunks_num)
        ]
    elif config.sample_strategy == 'top_bm25':
        chunks_num = 100
        tasks = [
            remote_sample_processing_bm25.remote(train_df, chunk, max_sample_num) 
            for chunk in np.array_split(test_data, chunks_num)
        ]
    elif config.sample_strategy == 'random':
        chunks_num = max(1, torch.cuda.device_count())
        tasks = [
            remote_sample_processing_random.remote(train_df, chunk, max_sample_num) 
            for chunk in np.array_split(test_data, chunks_num)
        ]
    elif config.sample_strategy == 'fixed':
        chunks_num = max(1, torch.cuda.device_count())
        tasks = [
            remote_sample_processing_fixed.remote(train_df, chunk, max_sample_num) 
            for chunk in np.array_split(test_data, chunks_num)
        ]
    else:
        raise ValueError(f"Unknown sample strategy: {config.sample_strategy}")
    
    all_samples = {}
    for result in tqdm(ray.get(tasks), desc="Processing samples"):
        all_samples.update(result)
    
    return all_samples



def read_json(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    return data



def truncate_text(text, max_length=2048):
    if len(text) <= max_length:
        return text
    
    # 截断到最大长度
    truncated = text[:max_length]
    
    # 找到最后一个句号
    last_period_index = truncated.rfind('.')
    if last_period_index == -1:
        # 如果找不到句号，则直接返回尽量短的文本
        return truncated
    
    # 返回以句号结尾的完整文本
    return truncated[:last_period_index + 1]


# prompt生成器   
def prompt_generartor(samples,query):
    prompt = ''
    #prompt = "if anything , see it for karen black , who camps up a storm as a fringe feminist conspiracy theorist named dirty dick .  >positive\nthe entire point of a shaggy dog story , of course , is that it goes nowhere , and this is classic nowheresville in every sense .  >negative\nfilm to affirm love 's power to help people endure almost unimaginable horror . >positive\n"
    label_distribution = []
    query['clean_text'] = query['clean_text'].replace('>','')
    query['clean_text'] = truncate_text(query['clean_text'])
    for sample in samples:
        # sample['label'] = 'negative' if sample['label'] == 0 else 'positive'
        # 去掉文本中的特殊字符'>'
        sample['clean_text'] = sample['clean_text'].replace('>','')
        sample['clean_text'] = truncate_text(sample['clean_text']) # 截断文本最大长度2048
        prompt += f'{sample["clean_text"]} >{sample["label"]}'+ '\n'
        label_distribution.append(sample['label']) 
    prompt = prompt + f'{query["clean_text"]} >'
    return prompt,label_distribution

def prior_prompts_generartor(samples,prior_list,query):
    prompt_pre = ''
    #prompt_pre = "if anything , see it for karen black , who camps up a storm as a fringe feminist conspiracy theorist named dirty dick .  >positive\nthe entire point of a shaggy dog story , of course , is that it goes nowhere , and this is classic nowheresville in every sense .  >negative\nfilm to affirm love 's power to help people endure almost unimaginable horror . >positive\n"
    prompts = []
    query['clean_text'] = query['clean_text'].replace('>','')
    query['clean_text'] = truncate_text(query['clean_text'])
    for prior in prior_list:
        prior = prior.replace('>','')
        prior = truncate_text(prior)
    for sample in samples:
        sample['clean_text'] = sample['clean_text'].replace('>','')
        sample['clean_text'] = truncate_text(sample['clean_text'])
        prompt_pre += f'{sample["clean_text"]} >{sample["label"]}'+ '\n'
    #prompt_pre += f'{query["clean_text"]} >{query["label"]}'+ '\n' #在进行惊奇影响先验探测实验时去掉这一行的注释
    #prompt_pre += f'N/A >positive'+'\n'
    for prior in prior_list:
        prompt = prompt_pre + f'{prior} >'
        prompts.append(prompt)
    return prompts
    
# 计算准确率
def calculate_accuracy(labels, predictions):
    if len(labels) == 0 or len(predictions) == 0:
        return 0.0
    correct = sum(1 for l, p in zip(labels, predictions) if l == p)
    return correct / len(labels)

# 序列化结果
def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd._libs.tslibs.nattype.NaTType):
        return None  # or return a string representation like 'NaT'
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # or any other string representation you prefer
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj

def convert_all_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_all_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_all_to_serializable(i) for i in obj]
    else:
        return convert_to_serializable(obj)

def turn_response_to_label(response, label_list,config):
    response_mapping = config.response_mapping
    try:
        response = response_mapping[response]
    except KeyError:
        response = random.choice(label_list)
    return response