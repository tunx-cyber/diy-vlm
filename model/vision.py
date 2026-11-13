"""纯 PyTorch 实现的 SigLIP 视觉塔（不依赖 transformers）。

关于版本：google/siglip2-base-patch32-256 的权重文件里
    vision_model.embeddings.patch_embedding.weight  -> [768, 3, 32, 32]
    vision_model.embeddings.position_embedding.weight -> [64, 768]
是 Conv2d + 固定 64 个位置编码的 **SigLIP v1** 结构，不是 transformers 里
Siglip2VisionModel 那套 nn.Linear + 位置编码插值（naflex）实现。
用 Siglip2VisionModel 加载会报 shape mismatch，这里按 v1 实现。

结构（对应 modeling_siglip.py）：
    Conv2d(p, p) -> flatten -> + pos_embed -> 12 x [pre-norm attn + pre-norm mlp]
    -> post_layernorm
注意力是**非因果**的，且没有 CLS token，输出 patch 数 = (image/patch)**2。
"""

import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def act_fn(name: str):
    if name in ("gelu_pytorch_tanh", "gelu_new"):
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"不支持的激活函数: {name}")


class SiglipAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size({hidden_size}) 必须能被 num_heads({num_heads}) 整除")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        # 视觉塔是非因果的：每个 patch 都能看到全部 patch，且没有 padding
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.out_proj(out)


class SiglipMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = act_fn(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class SiglipEncoderLayer(nn.Module):
    """pre-norm 残差：x = x + attn(norm1(x)); x = x + mlp(norm2(x))"""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 act: str, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = SiglipAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = SiglipMLP(hidden_size, intermediate_size, act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, image_size: int, patch_size: int,
                 num_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size,
            padding="valid")
        self.position_embedding = nn.Embedding(self.num_patches, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = pixel_values.shape
        expected = self.patch_size * int(self.num_patches ** 0.5)
        if height != expected or width != expected:
            # 位置编码是按固定分辨率预计算的，喂别的尺寸会静默错位
            raise ValueError(
                f"图片尺寸必须是 {expected}x{expected}，收到 {height}x{width}："
                "SigLIP v1 的位置编码不支持插值，请检查预处理里的 resize")
        patch_embeds = self.patch_embedding(pixel_values)      # [B, D, h, w]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)   # [B, P, D]
        return embeddings + self.position_embedding.weight.to(embeddings.dtype)


class SiglipVisionTower(nn.Module):
    """视觉塔本体：输入 [N, 3, H, W]，输出 [N, P, hidden]。"""

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 256,
        patch_size: int = 32,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size({image_size}) 必须是 patch_size({patch_size}) 的整数倍")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embeddings = SiglipVisionEmbeddings(
            hidden_size, image_size, patch_size, num_channels)
        self.encoder = nn.ModuleList(
            SiglipEncoderLayer(hidden_size, intermediate_size, num_attention_heads,
                               hidden_act, layer_norm_eps)
            for _ in range(num_hidden_layers)
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)


@dataclass
class SiglipImageConfig:
    image_size: int = 256
    mean: Sequence[float] = (0.5, 0.5, 0.5)
    std: Sequence[float] = (0.5, 0.5, 0.5)
    # PIL 的重采样枚举：0=NEAREST 1=LANCZOS 2=BILINEAR 3=BICUBIC 4=BOX 5=HAMMING
    resample: int = 2


def preprocess_image(image, image_size: int = 256,
                     mean: Sequence[float] = (0.5, 0.5, 0.5),
                     std: Sequence[float] = (0.5, 0.5, 0.5),
                     resample: int = 2) -> torch.Tensor:
    """PIL.Image -> [3, H, W] float32。

    对齐 SiglipImageProcessor：resize(resample) -> /255 -> (x-mean)/std。
    resample 默认 2=BILINEAR，和 checkpoint 里 preprocessor_config.json 一致。
    """
    from PIL import Image as PILImage

    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size),
                             resample=PILImage.Resampling(resample))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def preprocess_images(images, **kwargs) -> torch.Tensor:
    """list[PIL.Image] -> [N, 3, H, W]，N 是图片张数。"""
    if not images:
        raise ValueError("preprocess_images 收到了空列表")
    return torch.stack([preprocess_image(img, **kwargs) for img in images])


def resolve_weights(path: str) -> tuple[str, str]:
    """返回 (权重文件, 同目录的 config.json 或 "")。"""
    for filename in ("model.safetensors", "pytorch_model.bin"):
        local = os.path.join(path, filename)
        if os.path.isfile(local):
            cfg = os.path.join(path, "config.json")
            return local, cfg if os.path.isfile(cfg) else ""
    if os.path.isfile(path):
        return path, ""
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(path, "model.safetensors")
    try:
        cfg = hf_hub_download(path, "config.json")
    except Exception:
        cfg = ""
    return weights, cfg


def load_state_dict(weights_path: str) -> dict:
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def tower_kwargs_from_config(config_path: str, state_dict: dict) -> dict:
    cfg = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # SiglipConfig 把视觉参数放在 vision_config 下；也可能是扁平结构
        cfg = raw.get("vision_config", raw)

    patch_weight = state_dict["vision_model.embeddings.patch_embedding.weight"]
    hidden_size = int(patch_weight.shape[0])
    num_channels = int(patch_weight.shape[1])
    patch_size = int(patch_weight.shape[2])
    num_layers = max(int(k.split(".")[3]) for k in state_dict
                     if k.startswith("vision_model.encoder.layers.")) + 1
    intermediate_size = int(state_dict[
        "vision_model.encoder.layers.0.mlp.fc1.weight"].shape[0])
    num_patches = int(state_dict[
        "vision_model.embeddings.position_embedding.weight"].shape[0])
    grid = int(round(num_patches ** 0.5))

    kwargs = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        # config.json 里这个模型没写 num_attention_heads，默认 12 与权重相符
        "num_attention_heads": int(cfg.get("num_attention_heads", 12)),
        "num_channels": num_channels,
        "image_size": int(cfg.get("image_size", grid * patch_size)),
        "patch_size": patch_size,
        "hidden_act": cfg.get("hidden_act", "gelu_pytorch_tanh"),
        "layer_norm_eps": float(cfg.get("layer_norm_eps", 1e-6)),
    }
    if (kwargs["image_size"] // patch_size) ** 2 != num_patches:
        raise ValueError(
            f"config 的 image_size({kwargs['image_size']}) 和权重里的 "
            f"{num_patches} 个位置编码对不上")
    return kwargs


def load_image_config(path: str) -> SiglipImageConfig:
    """读取 preprocessor_config.json；文件不存在时用 SigLIP 默认值。"""
    try:
        _, cfg_path = resolve_weights(path)
        base = os.path.dirname(cfg_path) if cfg_path else path
        pre = os.path.join(base, "preprocessor_config.json")
        if not os.path.isfile(pre):
            from huggingface_hub import hf_hub_download
            pre = hf_hub_download(path, "preprocessor_config.json")
        with open(pre, "r", encoding="utf-8") as f:
            raw = json.load(f)
        size = raw.get("size") or {}
        return SiglipImageConfig(
            image_size=int(size.get("height") or size.get("shortest_edge") or 256),
            mean=tuple(raw.get("image_mean", (0.5, 0.5, 0.5))),
            std=tuple(raw.get("image_std", (0.5, 0.5, 0.5))),
            resample=int(raw.get("resample", 2)),
        )
    except Exception:
        return SiglipImageConfig()


def load_siglip_vision_tower(path: str, dtype: torch.dtype = torch.float32):
    """从本地目录或 HF repo id 加载视觉塔。

    返回 (tower, image_config)。tower 处于 eval + 冻结状态。
    """
    weights_path, config_path = resolve_weights(path)
    state = load_state_dict(weights_path)
    kwargs = tower_kwargs_from_config(config_path, state)

    tower = SiglipVisionTower(**kwargs)
    converted = {}
    for key, value in state.items():
        if not key.startswith("vision_model."):
            continue  # text_model.* / head.* / logit_scale 都不需要
        new_key = key[len("vision_model."):].replace("encoder.layers.", "encoder.")
        if new_key.startswith("head."):
            continue  # attention-pooling 头只产出 pooled 特征，这里不需要
        converted[new_key] = value
    missing, unexpected = tower.load_state_dict(converted, strict=False)
    if missing:
        raise RuntimeError(f"视觉塔缺少权重: {missing[:5]}（共 {len(missing)} 个）")
    if unexpected:
        raise RuntimeError(f"视觉塔有多余权重: {unexpected[:5]}（共 {len(unexpected)} 个）")

    tower = tower.to(dtype=dtype)
    tower.requires_grad_(False)
    tower.eval()
    return tower, load_image_config(path)
