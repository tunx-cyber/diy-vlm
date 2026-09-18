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
            max_len: int,
            stats_sample: int = 2000
        ):
        super().__init__()
        self.data = []
        with open(data_path,"r",encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    # 只跳过坏 JSON；以前是裸 except，连 KeyboardInterrupt 都会被吞掉
                    continue
                # 缺 "text" 字段时以前会直接 KeyError
                text = raw.get("text") if isinstance(raw, dict) else None
                if text:
                    self.data.append(text)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self._print_stats(stats_sample)

    def _print_stats(self, stats_sample: int):
        """打印长度分布。

        注意：self.data 里存的是原始文本，len(item) 是「字符数」不是 token 数，
        以前直接把字符数当 token 统计打出来了。这里两个都打，token 统计只在
        抽样的前 stats_sample 条上做，避免大数据集启动时卡住。
        """
        if not self.data:
            print("[data] 没有读到任何样本，检查 data_path 和 json 的 text 字段")
            return
        char_lens = np.array([len(item) for item in self.data])
        print(f"[data] 样本数: {len(self.data)}")
        print(f"[data] 字符长度 mean={char_lens.mean():.1f} median={np.median(char_lens):.0f} "
              f"p25/p50/p75/p90/p99={np.percentile(char_lens,[25,50,75,90,99]).round(1).tolist()}")

        sample = self.data[:stats_sample]
        tok_lens = np.array([
            len(self.tokenizer(t, add_special_tokens=False).input_ids) for t in sample
        ])
        over = int((tok_lens > self.max_len - 2).sum())
        print(f"[data] token 长度(前 {len(sample)} 条) mean={tok_lens.mean():.1f} "
              f"median={np.median(tok_lens):.0f} max={tok_lens.max()} "
              f"超长(>{self.max_len - 2}) {over}/{len(sample)}")

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

