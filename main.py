import os
import json
import argparse
import logging
import argparse
import pandas as pd
import torch
#设置固定的随机种子
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from config import base_config

import numpy as np
np.random.seed(3)

MAX_SAMPLE_NUM = 100

# command line args
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default='0,1,2,3,4,5,6,7,8,9', help='set cuda device', type=str)
parser.add_argument('--dataset', default='op_spam_full', help='Dataset name')
parser.add_argument('--model', type=str, default='qwen_7b', help='Model name')
parser.add_argument('--num_sample', type=int, default=60, help='Number of samples to use for prompt generation')
parser.add_argument('--sample_strategy', type=str, default='top_bm25', help='Sampling strategy for selecting samples')
parser.add_argument('--rank_strategy', type=str, default='increase', help='Ranking strategy for selecting samples')
parser.add_argument('--do_probe', action='store_true',help='Whether to do probe')
parser.add_argument('--do_predict', action='store_true',help='Whether to do predict')
args = parser.parse_args()

# update base_config with command line args
config = base_config()
config.__update__(vars(args))

# setup verblizer
verblizer_list = config.verblizer_mapping[config.dataset]
print('verblizer_list:',verblizer_list)

# setup cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda:',config.cuda)

from model import QwenModel
from utils import read_json, calculate_accuracy, convert_to_serializable, prompt_generartor, turn_response_to_label, preprocess_samples, convert_all_to_serializable, init_ray,prior_prompts_generartor
def setup_logging(log_path):
    # 确保日志目录存在
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # Append mode
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
setup_logging(config.log_path)
print('log_path:',config.log_path)

logging.info('Dataset:{};Model:{}'.format(config.dataset, config.model))
logging.info(config.sample_strategy)
logging.info(config.sample_data_path)

print('train_data_path:',config.train_data_path)
# 检查路径是否存在
import json

with open(config.train_data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)  # 尝试解析每一行
        except json.JSONDecodeError as e:
            print(f"第 {i} 行 JSON 格式错误: {e}")
# read data
train_df = pd.read_json(config.train_data_path, lines=True)
#train_data = read_json(config.train_data_path)
test_data = read_json(config.test_data_path)
probe_data = read_json(config.probe_data_path)
prior_data = read_json(config.prior_data_path)
# turn clean_text in prior_data to a list
prior_text = [item['clean_text'] for item in prior_data]
prior_label = [str(item['label']) for item in prior_data]
print('prior_label:',prior_label)

if len(verblizer_list) == 2 or len(verblizer_list) == 4:
    prior_label = [config.binary_list_mapping[label] for label in prior_label]
elif len(verblizer_list) == 3:
    prior_label = [config.trinary_list_mapping[label] for label in prior_label]
cc_prior_data = ['N/A','','[MASK]']

# precompute samples
if config.do_predict:
    if not os.path.exists(config.sample_data_path):
        logging.info('Precomputing samples...')
        init_ray(num_gpus=len(config.cuda.split(',')))
        all_samples = preprocess_samples(train_df, test_data, max_sample_num=30,config=config)
        with open(config.sample_data_path, 'w') as f:
            json.dump(convert_all_to_serializable(all_samples), f, indent=4)
        logging.info('Precomputing samples done!')
        logging.info('Loading precomputed samples from {}'.format(config.sample_data_path))
        with open(config.sample_data_path, 'r') as f:
            precomputed_samples = json.load(f)
    else:
        logging.info('Loading precomputed samples from {}'.format(config.sample_data_path))
    with open(config.sample_data_path, 'r') as f:
        precomputed_samples = json.load(f)

if config.do_probe:
    if not os.path.exists(config.probe_sample_path):
        logging.info('Precomputing probe samples...')
        init_ray(num_gpus=len(config.cuda.split(',')))
        all_samples = preprocess_samples(train_df, probe_data, max_sample_num=30,config=config)
        with open(config.probe_sample_path, 'w') as f:
            json.dump(convert_all_to_serializable(all_samples), f, indent=4)
        logging.info('Precomputing probe samples done!')
        logging.info('Loading precomputed probe samples from {}'.format(config.probe_sample_path))
    with open(config.probe_sample_path, 'r') as f:
        precomputed_probe_samples = json.load(f)

def calculate_prior(prior_prompts, model, special_tokens):
    confidence_list = []
    predict_list = []
    for prior_prompt in prior_prompts: 
        response, top_k_candidates,label_token_confidences = model.generate(prior_prompt,special_tokens)
        # 提取label_token_confiences最后一个键值对的值
        # print('label_token_confidences:',label_token_confidences)
        label_token_confidences = list(label_token_confidences.values())[-1] #{'positive': 0.9, 'negative': 0.1}
        # 归一化
        #label_token_confidences = {key: value / sum(label_token_confidences.values()) for key, value in label_token_confidences.items()}
        predict_confidences = list(label_token_confidences.values())
        confidence_list.append(label_token_confidences)
        predict_list.append(predict_confidences)
    # 计算每个标签置信度的均值,confidence_list[{'positive':0.9,'negative':0.1},{'positive':0.8,'negative':0.2}]
    if len(confidence_list) > 0:
        label_token_confidences = {key: np.mean([d.get(key, 0) for d in confidence_list]) for key in confidence_list[0]}
    else:
        label_token_confidences = {}
    return predict_list,confidence_list,label_token_confidences

import torch

def train_LinC(predict_list, label_list, epochs=300, lr=0.1):
    """
    使用PyTorch实现的分类任务线性参数调整
    
    Args:
        predict_list: 模型预测的置信度列表，形状为 data_size x num_labels
        label_list: 真实标签列表（one-hot编码），形状为 data_size x num_labels
        epochs: 训练轮次（默认300）
        lr: 学习率（默认0.1）
        
    Returns:
        list: 包含权重和偏置的列表，前num_labels个为权重，最后一个为偏置
    """
    if len(predict_list) == 0 or len(label_list) == 0:
        return [1.0] * len(verblizer_list), [0.0]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 转换数据格式并移动到设备
    X = torch.tensor(predict_list, dtype=torch.float32, device=device)
    y = torch.tensor([torch.argmax(torch.tensor(l)) for l in label_list], 
                    dtype=torch.long, device=device)
    
    num_labels = X.shape[1]
    
    # 直接在目标设备上创建参数（关键修复点）
    weights = torch.nn.Parameter(
        torch.ones(num_labels, device=device),  # 创建时直接指定设备
        requires_grad=True
    )
    bias = torch.nn.Parameter(
        torch.zeros(1, device=device),  # 创建时直接指定设备
        requires_grad=True
    )
    
    optimizer = torch.optim.SGD([weights, bias], lr=lr)
    #optimizer = torch.optim.Adam([weights, bias], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        adjusted = X * weights + bias
        loss = torch.nn.functional.cross_entropy(adjusted, y)
        loss.backward()
        optimizer.step()
        
    return weights.detach().cpu().tolist(), [bias.detach().cpu().item()]

def predict(test_data, samples,sorted_indices, result_file,no_log=False,do_prior=False):
    """
    test_data: 测试数据
    samples: 包含样本相似度的列表
    sorted_indices: 排序策略 
    """
    sample_num = config.num_sample
    
    label_list = []
    predicted_list = []
    result_list = []

    for test_index, test_item in enumerate(test_data):
        clean_text = test_item['clean_text']
    
        samples_text = samples[clean_text][-sample_num:]

        reordered_samples = [samples_text[i] for i in sorted_indices]
        prompt,label_distribution = prompt_generartor(reordered_samples, test_item)

        label = str(test_item['label'])
        
        label_distribution = [str(label) for label in label_distribution]
        label_distribution = label_distribution + [label]
        
        special_tokens = config.special_tokens_mapping[config.dataset]

        
        response, top_k_candidates,label_token_confidences = model.generate(prompt,special_tokens)
        #print(label,response, top_k_candidates,label_token_confidences)

        prior_list = None
        prior_confidences = None
        cc_prior_list = None
        cc_prior_confidences = None
        LinC_pram = None

        if do_prior:
            piror_prompts = prior_prompts_generartor(reordered_samples, prior_text, test_item)
            predict_list,prior_list,prior_confidences = calculate_prior(piror_prompts, model, special_tokens)
            if len(predict_list) > 0 and len(prior_label) > 0:
                weight,bias = train_LinC(predict_list, prior_label)
                LinC_pram = {'weight':weight,'bias':bias}

            cc_prior_prompts = prior_prompts_generartor(reordered_samples, cc_prior_data, test_item)
            predict_list,cc_prior_list,cc_prior_confidences = calculate_prior(cc_prior_prompts, model, special_tokens)

        predicted_label = turn_response_to_label(response,verblizer_list,config)
        

        label_list.append(label)
        predicted_list.append(predicted_label)

        result = {
            'sample_num': sample_num,
            'test_index': test_index,
            'label': label,
            'predicted_label': predicted_label,
            'top_k_candidates': top_k_candidates,
            'label_distribution': label_distribution,
            'label_token_confidences': label_token_confidences,
            'prior_confidences': prior_confidences,
            'prior_list': prior_list,
            'cc_prior_confidences':  cc_prior_confidences,
            'cc_prior_list': cc_prior_list,
            'LinC_pram': LinC_pram,
        }

        # 序列化结果
        result= json.loads(json.dumps(result, default=convert_to_serializable))
        result_list.append(result)

        # 追加结果到文件中
        if not no_log:
            # 确保输出目录存在
            result_dir = os.path.dirname(result_file)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                
            with open(result_file, 'a') as f:
                f.write(json.dumps(result) + "\n")
        # 每预测50个样本记录一次准确率
            if (test_index + 1) % 50 == 0 or test_index == len(test_data) - 1:
                accuracy = calculate_accuracy(label_list, predicted_list)
                logging.info(f'Iteration {test_index + 1}: Accuracy = {accuracy:.4f}')
                print(f'Iteration {test_index + 1}: Accuracy = {accuracy:.4f}')
                logging.info("\n")
        else:
            with open(result_file, 'w') as f:
                json.dump(result_list, f, indent=4)


    return label_list, predicted_list, result_list


sorted_indices = config.rank_strategys_mapping[config.rank_strategy]

logging.info('Sorted indices: {}'.format(sorted_indices))


if __name__ == '__main__':
    model_id = config.model_mapping[config.model]
    model = QwenModel(model_id)
    result_file = config.result_path
    prone_result_file = config.probe_result_path
    if config.do_probe:
        predict(probe_data, precomputed_probe_samples,sorted_indices, prone_result_file,no_log=False)
    if config.do_predict:
        predict(test_data, precomputed_samples,sorted_indices, result_file,no_log=False,do_prior=True)