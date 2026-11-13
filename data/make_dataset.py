import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from model.template import apply_chat_template, IMAGE_PAD_TOKEN, IMAGE_TOKEN_LEN
from model.vision import preprocess_images


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"load {path}"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 半行/坏行直接跳过，不让一条脏数据毁掉整个训练
                continue
    if not rows:
        raise ValueError(f"{path} 里没有解析出任何样本")
    return rows


class ChatDatasetBase(Dataset):
    """Pretrain / SFT 的公共部分：读 jsonl -> 渲染 -> tokenize -> 图片预处理。

    image_config 传 model.vision.SiglipImageConfig；没有图片时可以是 None。
    样本里有图却没传 image_config 会直接报错，避免静默丢掉视觉信息。
    """

    def __init__(self, data_path: str, tokenizer, max_len: int, image_config=None):
        self.data = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_config = image_config
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)

    # ---------------- 内部工具 ----------------
    def truncate_keep_images(self, ids: list) -> list:
        if len(ids) <= self.max_len:
            return ids
        ids = ids[: self.max_len]
        # 截断可能正好切在一张图的 IMAGE_TOKEN_LEN 个 <|image_pad|> 中间。
        # 留着缺口会让视觉特征数和 pad 数对不上，forward 里必然报错；
        # 这里把这截不完整的尾巴整段丢掉，样本仍是一条合法的短样本。
        tail = len(ids)
        while tail > 0 and ids[tail - 1] == self.image_token_id:
            tail -= 1
        if (len(ids) - tail) % IMAGE_TOKEN_LEN:
            ids = ids[:tail]
        return ids

    def encode_text(self, text: str) -> list:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return self.truncate_keep_images(ids)

    def to_pixel_values(self, images):
        if not images:
            return None
        if self.image_config is None:
            raise ValueError(
                "样本里包含图片，但构造 Dataset 时没有传 image_config，"
                "无法做 resize/normalize")
        cfg = self.image_config
        return preprocess_images(
            images, image_size=cfg.image_size, mean=cfg.mean, std=cfg.std,
            resample=cfg.resample)

    def build_sample(self, ids: list, labels: list, pixel_values):
        if not ids:
            # 整行都是空的（例如 content 为空）：attention_mask 会全 0，
            # softmax 全是 -inf -> NaN，直接把整个 loss 污染掉。用单个 eos 兜底。
            ids = [self.tokenizer.eos_token_id]
            labels = list(ids)
        if len(ids) > self.max_len:
            raise ValueError(
                f"样本长度({len(ids)})超过 max_len({self.max_len})：encode_text 应该已经截断过")
        n_pad = self.max_len - len(ids)
        input_ids = torch.tensor(ids + [self.pad_token_id] * n_pad, dtype=torch.long)
        labels = torch.tensor(labels + [-100] * n_pad, dtype=torch.long)
        attention_mask = torch.tensor(
            [1] * len(ids) + [0] * n_pad, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,      # model 内部会自己往后 shift 一个 token
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    def __getitem__(self, index):
        return self.process_one_sample(self.data[index])

    def __len__(self):
        return len(self.data)


class PretrainDataset(ChatDatasetBase):
    """预训练数据集。

    jsonl 每行两种形式：
        {"messages": [{"role": "", "content": ""}], "tools": []}   视觉信息以 part 形式嵌在 messages 里
        {"text": "..."}                                            纯文本
    """

    def process_one_sample(self, sample: dict):
        if "messages" in sample:
            res = apply_chat_template(sample["messages"], sample.get("tools"),
                                      tokenizer=self.tokenizer)
            text, images = res["inputs"], res["pixel_vals"]
        elif "text" in sample:
            text, images = sample["text"], []
        else:
            raise KeyError(f"样本既没有 messages 也没有 text: {list(sample)[:5]}")

        ids = self.encode_text(text)
        # 只把补齐出来的位置设成 -100；真实文本里的 <|endoftext|> 不该被误屏蔽
        labels = list(ids)
        return self.build_sample(ids, labels, self.to_pixel_values(images))


class SFTDataset(ChatDatasetBase):
    """SFT：只在 assistant 回复段（<|im_start|>assistant\\n ... <|im_end|>）上算 loss。"""

    def __init__(self, data_path: str, tokenizer, max_len: int, image_config=None):
        super().__init__(data_path, tokenizer, max_len, image_config)
        self.bos_id = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
        if not self.bos_id or not self.eos_id:
            raise ValueError("assistant 的起始/结束标记分词为空，检查 tokenizer")

    def generate_labels(self, ids: list) -> list:
        labels = [-100] * len(ids)
        n_bos, n_eos = len(self.bos_id), len(self.eos_id)
        i = 0
        while i <= len(ids) - n_bos:
            if ids[i:i + n_bos] != self.bos_id:
                i += 1
                continue
            start = i + n_bos
            end = start
            while end <= len(ids) - n_eos and ids[end:end + n_eos] != self.eos_id:
                end += 1
            stop = min(end + n_eos, len(ids))   # 把 <|im_end|> 也算进 loss
            for j in range(start, stop):
                labels[j] = ids[j]
            i = stop
        return labels

    def process_one_sample(self, sample: dict):
        res = apply_chat_template(sample["messages"], sample.get("tools"),
                                  tokenizer=self.tokenizer)
        ids = self.encode_text(res["inputs"])
        labels = self.generate_labels(ids)
        return self.build_sample(ids, labels, self.to_pixel_values(res["pixel_vals"]))


def collate_fn(batch: list) -> dict:
    """把 batch 里的图片拼成 [sum_images, 3, H, W]。

    拼的顺序 == 样本顺序，和 input_ids 里 <|image_pad|> 的展平顺序一致，
    所以模型侧一次索引赋值就能全部填上，不用逐样本循环。
    """
    pixel_values = [b["pixel_values"] for b in batch if b["pixel_values"] is not None]
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "pixel_values": torch.cat(pixel_values, dim=0) if pixel_values else None,
    }
