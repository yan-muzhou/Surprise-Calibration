import os
import json
import random
#from model import *

class base_config():

    def __init__(self):
        self.root_path = os.getcwd()
        self.data_path = os.path.join(self.root_path, "dataset")


        self.model_mapping = {
            "qwen_70b": "qwen/Qwen2-72B",
            "qwen_14b":"Qwen/Qwen2.5-14B",
            "qwen_7b": "Qwen/Qwen2.5-7B",
            "qwen_3b": "Qwen/Qwen2.5-3B",
            "qwen_1.5b": "Qwen/Qwen2.5-1.5B",
            'qwen_0.5b': "Qwen/Qwen2.5-0.5B",
            "llama_8b":"LLM-Research/Meta-Llama-3-8B"
        }

        self.verblizer_mapping = {
            "YouTube": ['truthful', 'deceptive'],
            "YouTube_ub": ['truthful', 'deceptive'],
            "AI-GA_1-1": ['0', '1'],
            "SST-2": ['negative', 'positive'],
            "SST-2_mini": ['negative', 'positive'],
            "MRPC": ['no', 'yes'],
            "RTE": ['no', 'yes'],
            "QNLI": ['no', 'yes'],
            "WiC":['false', 'true'],
            "MNLI":['no','maybe','yes'],
            "AG-news":["world","sports","business","tech"],
            "RTE_incorrectcorrect":['incorrect','correct'],
            "RTE_ab":['a','b'],
            "RTE_badgood":['bad','good'],
            "RTE_01":['0','1'],
            "RTE_falsetrue":['false','true'],
        }

        self.special_tokens_mapping = {
            "AI-GA_1-1": ['0', '1'],
            "SST-2": ['negative', 'positive'],
            "SST-2_mini": ['negative', 'positive'],
            "MRPC": ['no', 'yes'],
            "YouTube": ['truth', 'de'],
            "RTE": ['no', 'yes'],
            "QNLI": ['no', 'yes'],
            "WiC":['false', 'true'],
            "MNLI":['no','maybe','yes'],
            "AG-news":["world","sports","business","tech"],
            "RTE_incorrectcorrect":['incorrect','correct'],
            "RTE_ab":['a','b'],
            "RTE_badgood":['bad','good'],
            "RTE_01":['0','1'],
            "RTE_falsetrue":['false','true'],
        }

        self.response_mapping = {
            '0': '0',
            '1': '1',
            'negative': 'negative',
            'positive': 'positive',
            'truth': 'truthful',
            'de': 'deceptive',
            "no": 'no',
            "yes": 'yes',
            "false": 'false',
            "true": 'true',
            "maybe": 'maybe',
            "world":"world",
            "sports":"sports",
            "business":"business",
            "tech":"tech",
            "a":"a",
            "b":"b",
            "bad":"bad",
            "good":"good",
            "incorrect":"incorrect",
            "correct":"correct",
        }

        self.binary_list_mapping = {
            "negative": [1,0],
            "positive": [0,1],
            "0": [1,0],
            "1": [0,1],
            "truthful": [1,0],
            "deceptive": [0,1],
            'no': [1,0],
            'yes': [0,1],
            'false': [1,0],
            'true': [0,1],
            'world': [1,0,0,0],
            'sports': [0,1,0,0],
            'business': [0,0,1,0],
            'tech': [0,0,0,1],
            "a":[1,0],
            "b":[0,1],
            "bad":[1,0],
            "good":[0,1],
            "incorrect":[1,0],
            "correct":[0,1],
        }

        self.trinary_list_mapping = {
            "no": [1,0,0],
            "maybe": [0,1,0],
            "yes": [0,0,1],
        }


    def __update__(self, kwargs):

        def generate_ucurve(num_sample):
            """
            生成一个U型排序的索引列表，适用于任何大于1的正整数num_sample。
            
            参数:
                num_sample (int): 样本数量，必须大于1。
            
            返回:
                list: 按U型排序的索引列表。
            """
            if num_sample <= 2:
                if num_sample == 1:
                    return [0]
                elif num_sample == 2:
                    return [0, 1]
                else:
                    return []
            
            # 根据num_sample的奇偶性生成不同的lower部分
            if num_sample % 2 == 0:
                # 偶数：从num_sample-2开始，步长为-2
                lower = list(range(num_sample - 2, -1, -2))
            else:
                # 奇数：从num_sample-1开始，步长为-2
                lower = list(range(num_sample - 1, -1, -2))
            
            # 生成upper部分，所有奇数索引
            upper = list(range(1, num_sample, 2))
            
            # 合并lower和upper部分，形成U型排序
            ucurve = lower + upper
            
            return ucurve
        for k, v in kwargs.items():
            setattr(self, k, v) 
        self.train_data_path = os.path.join(self.data_path, self.dataset, "train.jsonl")
        self.test_data_path = os.path.join(self.data_path, self.dataset, "test.jsonl")
        self.probe_data_path = os.path.join(self.data_path, self.dataset, "probe.jsonl")
        self.prior_data_path = os.path.join(self.data_path, self.dataset, "prior.jsonl")
        self.output_path = os.path.join(self.root_path, "output",self.dataset)
        self.cache_path = os.path.join(self.root_path, "cache",self.dataset)
        # 如果output_path不存在，则创建
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.log_path = os.path.join(self.output_path, self.model+ '-' +self.sample_strategy+'-'+str(self.num_sample)+self.rank_strategy+".log") #日志文件
        self.result_path = os.path.join(self.output_path, self.model+ '-' +self.sample_strategy+'-'+str(self.num_sample)+self.rank_strategy+".jsonl") #测试集预测结果
        self.sample_data_path = os.path.join(self.cache_path, self.sample_strategy+'-sample'+'.jsonl') #测试集相似度计算结果

        self.probe_result_path = os.path.join(self.cache_path, self.model+ '-' +self.sample_strategy+'-'+str(self.num_sample)+self.rank_strategy+'-probe'+'.jsonl') #探测集预测结果
        self.probe_sample_path = os.path.join(self.cache_path, self.sample_strategy+'-probe'+'.jsonl') #探测集相似度计算结果

        self.rank_strategys_mapping = {
            "Ucurve": generate_ucurve(self.num_sample),
            "increase":list(range(self.num_sample)),
            "decrease":list(range(self.num_sample - 1, -1, -1)),
            #0-59的随机排列
            "random": random.sample(range(self.num_sample), self.num_sample)
        }
    