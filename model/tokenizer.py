"""不依赖 transformers 的 tokenizer。

直接读 tokenizer.json（Rust 的 `tokenizers` 库，不是 transformers），
特殊 token 的 id 从 added_tokens / tokenizer_config.json 里取。
对外接口刻意做成 AutoTokenizer 的一个子集，方便替换：

    ids = tokenizer(text, add_special_tokens=False).input_ids
    tokenizer.decode(ids, skip_special_tokens=True)
    tokenizer.pad_token_id / eos_token_id / bos_token_id
    len(tokenizer)
"""

import json
import os
from types import SimpleNamespace

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# 项目统一的 tokenizer 来源；可用环境变量覆盖成本地目录
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "jingyaogong/minimind-3")

NEEDED_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def download_files(repo_id: str) -> dict:
    if os.path.isdir(repo_id):
        paths = {name: os.path.join(repo_id, name) for name in NEEDED_FILES}
        if not os.path.isfile(paths["tokenizer.json"]):
            raise FileNotFoundError(f"{repo_id} 下没有 tokenizer.json")
        return paths
    out = {}
    for name in NEEDED_FILES:
        try:
            out[name] = hf_hub_download(repo_id, name)
        except Exception:
            # special_tokens_map.json 有些仓库没有，缺了不影响
            if name == "tokenizer.json":
                raise
    return out


def as_text(value):
    """tokenizer_config 里的 token 可能是 str，也可能是 {"content": ...}。"""
    if isinstance(value, dict):
        return value.get("content")
    return value


class HFTokenizer:
    def __init__(self, repo_id: str = TOKENIZER_PATH):
        paths = download_files(repo_id)
        with open(paths["tokenizer.json"], "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.tok = Tokenizer.from_file(paths["tokenizer.json"])

        # added_tokens 里 id 可能和基础词表重叠，vocab_len 要取两者的并集上界
        self.base_vocab = len(raw["model"]["vocab"])
        added = raw.get("added_tokens", [])
        self.added_tokens = {a["content"]: int(a["id"]) for a in added}
        max_added = max((int(a["id"]) for a in added), default=-1)
        self.vocab_len = max(self.base_vocab, max_added + 1)

        cfg = {}
        if "tokenizer_config.json" in paths:
            with open(paths["tokenizer_config.json"], "r", encoding="utf-8") as f:
                cfg = json.load(f)
        self.pad_token = as_text(cfg.get("pad_token")) or "<|endoftext|>"
        self.eos_token = as_text(cfg.get("eos_token")) or "<|im_end|>"
        self.bos_token = as_text(cfg.get("bos_token")) or "<|im_start|>"
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        self.eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        self.bos_token_id = self.convert_tokens_to_ids(self.bos_token)

    # ---------------- 兼容 AutoTokenizer 的最小子集 ----------------
    def __len__(self) -> int:
        return self.vocab_len

    @property
    def vocab_size(self) -> int:
        """基础 BPE 词表大小（不含 added tokens），和 HF 的语义保持一致。

        embedding 必须用 len(tokenizer)：用 vocab_size 会让 <|im_end|> 这类
        added token 越界，在 CUDA 上表现为 device-side assert，看不出原因。
        """
        return self.base_vocab

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tid = self.added_tokens.get(tokens)
            if tid is None:
                tid = self.tok.token_to_id(tokens)
            if tid is None:
                raise KeyError(f"词表里没有 token: {tokens!r}")
            return tid
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(
        self,
        text,
        add_special_tokens: bool = True,
        max_length=None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors=None,
        return_attention_mask=None,
        **kwargs,
    ):
        if not isinstance(text, str):
            raise TypeError(
                f"这个 tokenizer 只接受单条 str，收到 {type(text).__name__}；"
                "批量请自己在 dataset 里循环")
        input_ids = self.tok.encode(text, add_special_tokens=add_special_tokens).ids
        if truncation and max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        if padding == "max_length" and max_length is not None:
            input_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))
        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = [1] * len(input_ids) if return_attention_mask else None
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        token_ids = [int(t) for t in token_ids]
        return self.tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]


tokenizer = HFTokenizer(TOKENIZER_PATH)
# 词表真实大小必须用 len()，不能用 tokenizer.vocab_size：
# HF 的 vocab_size 不含 added special tokens，按它建 embedding 会越界。
vocab_size = len(tokenizer)
