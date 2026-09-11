import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import json
import numpy as np

class PretrainDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_len: int
        ):
        super().__init__()
        self.data = []
        with open(data_path,"r") as f:
            for line in tqdm(f):
                try:
                    raw = json.loads(line)
                except:
                    continue
                if raw:
                    self.data.append(raw["text"])
        self.tokenizer = tokenizer
        self.max_len = max_len

        # self.data = tokenizer(self.data,truncation=True,man_length=512)["input_ids"]
        lens = np.array([len(item) for item in self.data])
        print("均值: ", lens.mean())
        print("中位数: ", np.median(lens))
        print("分位数: ", np.percentile(lens,[25,50,75,90,99]))

    def __getitem__(self, index):
        sample = self.data[index]
        tokens = self.tokenizer(sample, add_special_tokens=False, max_length=self.max_len - 2, truncation=True).input_ids
        tokens = tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

    def __len__(self):
        return len(self.data)

class SFTDataset(Dataset):
    def __init__(self, data_path):
        pass

    def __getitem__(self, index):
        pass

