import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def get_lr(current_step, total_steps, lr, warmup_steps=0, min_lr_ratio=0.1):
    """线性 warmup + 余弦退火，返回当前步的 lr。

    以前的版本没有 warmup，且 (1 + cos) 在 step=0 时是 2.0，
    实际起步学习率是 2*lr 而不是 lr，容易一开始就训崩。
    """
    if total_steps <= 0:
        return lr
    if warmup_steps > 0 and current_step < warmup_steps:
        return lr * (current_step + 1) / warmup_steps
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * decay)


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def make_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """右 padding 的 0/1 attention mask。

    训练时必须把它传给模型：dataset 用 pad_token_id 把每条样本补齐到 max_len，
    不传 mask 的话真实 token 会 attend 一长串 padding，hidden state 会被污染。
    """
    return (input_ids != pad_token_id).long()


def setup_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_autocast(args, device: torch.device):
    """返回 (是否启用 autocast, autocast dtype)。

    bf16 不需要 loss scaling；fp16 必须配 GradScaler，否则梯度下溢成 0。
    """
    name = args.dtype
    if name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            name = "bf16"
        elif device.type == "cuda":
            name = "fp16"
        else:
            # CPU / MPS 上 bf16 autocast 支持不全，默认退回 fp32
            name = "fp32"
    if name == "fp32":
        return False, torch.float32
    return True, torch.bfloat16 if name == "bf16" else torch.float16


def filter_state_dict(model, exclude_prefixes):
    excluded = tuple(exclude_prefixes)
    if not excluded:
        return model.state_dict()
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith(excluded)}


def save_checkpoint(path, model, optimizer=None, scaler=None,
                    step=0, epoch=0, extra=None, exclude_prefixes=("vision_encoder.",)):
    """保存检查点。

    冻结的视觉塔不进 checkpoint（默认排除 vision_encoder.）：
    它只读不训，存进去只会让每个 ckpt 白胖几百 MB。
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    state = {
        "model": filter_state_dict(model, exclude_prefixes),
        "step": step,
        "epoch": epoch,
        "exclude_prefixes": tuple(exclude_prefixes),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra:
        state["extra"] = extra
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # 先写临时文件再原子替换，防止 ckpt 写一半被中断
    return path


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    """加载检查点，返回 (step, epoch, extra)。"""
    state = torch.load(path, map_location=map_location, weights_only=False)
    exclude = tuple(state.get("exclude_prefixes", ()))
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # 缺的键只能是被刻意排除的冻结权重，其余的说明架构对不上
    bad = [k for k in missing if not k.startswith(exclude)]
    if bad:
        raise RuntimeError(
            f"加载 {path} 时缺少权重: {bad[:5]}（共 {len(bad)} 个）；"
            "检查模型结构和 checkpoint 是否同源")
    if unexpected:
        raise RuntimeError(
            f"加载 {path} 时有多余权重: {unexpected[:5]}（共 {len(unexpected)} 个）")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("step", 0)), int(state.get("epoch", 0)), state.get("extra")


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
