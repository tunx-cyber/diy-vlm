import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import json
import numpy as np
from model.template import apply_chat_template

class PretrainDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_len: int,
        ):
        '''预训练数据集
        Args:
            data_path: jsonl 格式
            tokenizer: 处理文本的分词器
            max_len: 训练的最大长度
      
        其中数据格式
        {
            "messages": [{"role": "", "content": ""}],视觉信息以part形式嵌入到messages里面
            "tools": []
        }
        '''
        super().__init__()
        self.data = []
        with open(data_path,"r",encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    # 可以加入检查机制，本项目简化这里
                    self.data.append(raw)
                except json.JSONDecodeError:
                    continue
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_one_sample(self, sample:dict):
        if "messages" not in sample and "text" in sample:
            chat_str = sample["text"]
            pixel_vals=None
        res = apply_chat_template(sample["messages"], tokenizer=self.tokenizer)
        chat_str, pixel_vals = res["inputs"], res["pixel_vals"]
        tokens = self.tokenizer(chat_str, add_special_tokens=False, max_length=self.max_len - 2, truncation=True).input_ids
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":input_ids,
            "labels": labels,# model 内部会自动往后shift一个token，这里就不用shift了
            "pixel_vals": pixel_vals
        }

    def __getitem__(self, index):
        sample = self.data[index]
        return self.process_one_sample(sample)

    def __len__(self):
        return len(self.data)

class SFTDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_len: int,
        ):
        '''预训练数据集
        Args:
            data_path: jsonl 格式
            tokenizer: 处理文本的分词器
            max_len: 训练的最大长度
        
        其中数据格式
        {
            "messages": [{"role": "", "content": ""}],
            "tools":[]
        }
        '''
        super().__init__()
        self.data = []
        with open(data_path,"r",encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    # 可以加入检查机制，本项目简化这里
                    self.data.append(raw)
                except json.JSONDecodeError:
                    continue
                
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids

    def generate_labels(self, input_ids):
            labels = [-100] * len(input_ids)
            i = 0
            while i < len(input_ids):
                if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                    start = i + len(self.bos_id)
                    end = start
                    while end < len(input_ids):
                        if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                            break
                        end += 1
                    for j in range(start, min(end + len(self.eos_id), self.max_len)):
                        labels[j] = input_ids[j]
                    i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
                else:
                    i += 1
            return labels
    def process_one_sample(self, sample:dict):              
        res = apply_chat_template(sample["messages"], sample.get("tools"),
                                    tokenizer=self.tokenizer)
        chat_str, pixel_vals = res["inputs"], res["pixel_vals"]
        input_ids = self.tokenizer(chat_str).input_ids[:self.max_len]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
        labels = self.generate_labels(input_ids)
        return {
                "input_ids":input_ids,
                "labels": labels,# model 内部会自动往后shift一个token，这里就不用shift了
                "pixel_vals": pixel_vals
            }

    def __getitem__(self, index):
        sample = self.data[index]
        return self.process_one_sample(sample)

    def __len__(self):
        return len(self.data)

