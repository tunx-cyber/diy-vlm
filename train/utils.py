import random
import math
import torch
import os
import numpy as np
import torch.distributed as dist

from model.checkpoint import save_checkpoint

def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def make_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """右 padding 的 0/1 attention mask。
    训练时必须把它传给模型：data/make_dataset.py 用 pad_token_id（=<|endoftext|>）
    把每条样本补齐到 max_len，不传 mask 的话真实 token 会 attend 一长串 padding，
    hidden state 会被污染。
    """
    return (input_ids != pad_token_id).long()

def save_model(model, path, config=None):
    """保存权重（bf16）。

    统一走 model.checkpoint 的新格式，checkpoint 里会带上 config，
    避免加载时架构对不上。
    """
    save_checkpoint(model, path, config=config, dtype="bf16")

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False