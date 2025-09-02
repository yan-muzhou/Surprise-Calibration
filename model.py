import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer,snapshot_download
import torch.nn.functional as F

PYTORCH_CUDA_ALLOC_CONF = {"expandable_segments": True}



class QwenModel():
    def __init__(self, model_name, max_length=2):
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype= torch.float16,
            #fp16
            #torch_dtype=torch.float16,
            device_map='auto',
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


    def _clear_cache(self, *args):
        for arg in args:
            del arg
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # 强制调用垃圾回收

    def get_top_k_candidates(self, logits, k=5):
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        # 获取前几个候选词及其概率
        top_k_probs, top_k_indices = torch.topk(probs, k)
        
        top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices[0])
        top_k_probs = top_k_probs[0].to(torch.float32).cpu().numpy()

        top_k_candidates = [(token, prob) for token, prob in zip(top_k_tokens, top_k_probs)]
        return top_k_candidates
    

    def compute_confidence(self, logits, special_tokens):
        """
        计算特殊令牌的置信度，仅在这些令牌上计算softmax概率。

        :param logits: 模型的logits，形状可为 (vocab_size,) 或 (1, vocab_size)
        :param special_tokens: 特殊令牌列表
        :return: 特殊令牌的置信度字典 {special_token: confidence}
        """
        # 确保每个特殊令牌编码为单个token ID
        special_token_ids = []
        for token in special_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_id) != 1:
                raise ValueError(f"特殊令牌 '{token}' 编码为 {len(token_id)} 个ID，应为1个。")
            special_token_ids.append(token_id[0])
        
        # 提取目标logits并计算概率
        if logits.dim() == 2:  # 处理二维logits (e.g., 批处理场景)
            selected_logits = logits[:, special_token_ids].squeeze(0)  # 变为一维
        else:                  # 一维logits直接索引
            selected_logits = logits[special_token_ids]
        
        probabilities = torch.softmax(selected_logits, dim=-1)
        
        # 生成置信度字典
        return {
            token: prob.item()
            for token, prob in zip(special_tokens, probabilities)
        }

    def generate(self, prompt,special_tokens, max_new_tokens=None):
        """
        增强版 generate 函数，获取分隔符 ' >' 和特殊 token 的置信度。

        :param prompt: 输入的文本
        :param special_tokens: 特殊 token 的列表，例如 ["<special1>", "<special2>"]
        :param max_new_tokens: 生成的最大新 token 数，默认为 None（使用 self.max_length）
        :return: 最后一个分割符最大概率候选词，最后一个分隔符的 top-k 候选项，每个分隔符对每个特殊 token的置信度
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_length

        # Tokenize prompt
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Token IDs of the input
        input_ids = model_inputs['input_ids'][0]

        # Get the token ID of the separator '>' and special tokens
        #separator_token_id = self.tokenizer.encode(" >")[0] #qwen_setting
        separator_token_id = self.tokenizer.encode(" >")[-1] #llama_setting
        

        separator_logits = []  # Store logits for each '>'
        special_token_confidences = {}  # Store confidence for each special token
        with torch.no_grad():
            # Generate outputs
            outputs = self.model(**model_inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

            # Loop over the sequence to track logits for each '>'
            k = 0
            for i, token_id in enumerate(input_ids):
                if token_id == separator_token_id:
                    k += 1
                    current_logits = logits[:, i, :]
                    separator_logits.append(current_logits)  # Save logits for this '>'
                    # compute confidence for special tokens
                    confidences_dict = self.compute_confidence(current_logits, special_tokens)
                    special_token_confidences[k] = confidences_dict
            
            top_k_candidates = self.get_top_k_candidates(separator_logits[-1])
            top_candidate = top_k_candidates[0][0]
            self._clear_cache(model_inputs,outputs,logits)
        return top_candidate, top_k_candidates, special_token_confidences