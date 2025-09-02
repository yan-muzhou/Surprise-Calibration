import pandas as pd
from rank_bm25 import BM25Okapi
import spacy
import tqdm
import random
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import bm25s

def df_to_list(df):
    return df.to_dict(orient='records')

class Sampler:
    '''
    Sampler类用于从数据集中选择样本,以便用于生成prompt。
    '''
    def __init__(self,df):
        
        self.model_path = '/workspace/ICL_fake_reviews/Models/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1'
        # 加载spaCy的模型
        self.nlp = spacy.load(self.model_path)
        self.pipeline_se = pipeline(Tasks.sentence_embedding, model="iic/nlp_gte_sentence-embedding_english-small", sequence_length=512)
        self.batch_size = 10000
        self.df = df
    def tokenize(self, text):
        # 使用spaCy进行分词、去停用词和词干提取
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def random_samples_all(self, df, query, n_samples, random_state=42):
        return df_to_list(df.sample(n_samples, random_state=random_state))
    

    def init_bm25(self):
        df = self.df
        corpus = df['clean_text'].tolist()
        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        return retriever,corpus


    def top_bm25(self, retriever, query, corpus ,n_samples, random_state=42):
        df = self.df
        query_text = query['clean_text']  # 假设查询包含一个'clean_text'字段
        query_tokens = bm25s.tokenize(query_text)

        results, scores = retriever.retrieve(query_tokens, corpus=corpus, k = len(corpus))

        # 计算BM25分数
        retrieve_df = pd.DataFrame({'clean_text': results[0].tolist(), 'bm25_score': scores[0].tolist()})
        
        # 按照clean_text合并dataframe
        df = pd.merge(df, retrieve_df, on='clean_text')

        # 删除query所在的行
        df = df[df['clean_text'] != query_text]
        
        # 按相关性得分升序排序，并选择后n个样本
        top_samples = df.sort_values(by='bm25_score', ascending=True).tail(n_samples+1)

        # 返回按相关性排列的样本
        return df_to_list(top_samples)

    def init_gte(self, chunk_size=10):
        df = self.df
        corpus = df['clean_text'].tolist()

        # 初始化一个空列表来存储所有块的嵌入向量
        all_embeddings = []

        # 将corpus分块处理
        for i in range(0, len(corpus), chunk_size):
            corpus_chunk = corpus[i:i + chunk_size]  # 获取当前块
            inputs = {
                "source_sentence": corpus_chunk
            }
            result = self.pipeline_se(input=inputs)  # 生成当前块的嵌入向量
            text_embeddings = result['text_embedding']
            text_embeddings = torch.tensor(text_embeddings, device='cpu')  # 将嵌入向量移动到CPU
            all_embeddings.append(text_embeddings)

        # 合并所有块的嵌入向量
        text_embeddings = torch.cat(all_embeddings, dim=0)

        return None, text_embeddings



    def top_gte(self, retriever, query, corpus, n_samples, random_state=42, chunk_size=10):
        df = self.df
        query_text = query['clean_text']  # 假设查询包含一个'clean_text'字段
        inputs = {
            "source_sentence": [query_text]
        }
        result = self.pipeline_se(input=inputs)
        query_embedding = result['text_embedding']
        query_embedding = torch.tensor(query_embedding, device='cuda')

        # 初始化一个空列表来存储所有块的相似度
        all_similarities = []

        # 将corpus分块处理
        for i in range(0, len(corpus), chunk_size):
            corpus_chunk = corpus[i:i + chunk_size].to('cuda')  # 将当前块移动到GPU
            similarities = torch.cosine_similarity(query_embedding, corpus_chunk, dim=1).cpu().numpy().tolist()
            all_similarities.extend(similarities)

        # 将相似度添加到数据框
        df['similarity_score'] = all_similarities

        df = df[df['clean_text'] != query_text]
        
        # 按相似度得分降序排序，并选择后n个样本
        top_samples = df.sort_values(by='similarity_score', ascending=True).tail(n_samples)
        
        # 返回按相似度降序排列的样本
        return df_to_list(top_samples)

    def init_random(self):
        return None,None
    def init_fixed(self):
        return None,None
    def fixed_samples_all(self,  n_samples, random_state=42):
        return df_to_list(self.df.sample(n_samples, random_state=42))
def df_to_list(df):
    return df.to_dict('records')
    


