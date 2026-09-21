'''
支持传统的pt格式
支持分布式训练的checkpoint比如DDP，fsdp，TP
支持safetensors

------------------------------------------------------------------
三种格式

  pt            torch.save 单文件：{"model": 权重, "config": 结构, "optimizer": ...}
  safetensors   单个 .safetensors，config 写在 metadata 里；优化器状态放 <name>.optim.pt
  dcp           torch.distributed.checkpoint 目录，各 rank 按 shard 写盘，
                FSDP2 / TP 的 DTensor 直接按 shard 存取，权重 + 优化器状态一起存

调用约定
--------
* 所有 rank 都必须调用 save_checkpoint / load_into，不要在调用点外面套 `if rank == 0:`。
  - pt / safetensors：FSDP 时内部做一次 gather 的集合通信，DDP / 单卡不做集合通信，
    只有 rank0 真正写盘。
  - dcp：DCP 自己就是集合通信，必须所有 rank 一起调。
* save_checkpoint 里没有“无条件”的 dist.barrier()：只有 dcp 覆盖写盘前清理旧目录时会
  barrier，而这条路径本来就要求所有 rank 参与。
* 加载时按路径自动识别格式，传的就是 save_checkpoint 返回的路径。
'''

from __future__ import annotations

import json
import os
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .config import ModelConfig

try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import DefaultLoadPlanner
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - 老版本 torch 或缺分布式支持
    dcp = DefaultLoadPlanner = StateDictOptions = DTensor = FSDPModule = None

__all__ = ["FORMATS", "LoadResult", "detect_format", "load_into", "read_config", "save_checkpoint"]

FORMATS = ("pt", "safetensors", "dcp")

_PT_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt")
# 这些字段决定权重 shape，对不上就没法加载，直接报错；
# 其余字段（max_position_embeddings / rope_base / dropout ...）允许不一致，只 warning
_SHAPE_KEYS = ("layers", "vocab_size", "kv_heads", "attn_heads", "hidden_dim", "intermediate_dim")

_DCP_MODEL_KEY = "model"
_DCP_OPTIM_KEY = "optimizer"
_DCP_META_FILE = "config.json"

_DTYPE_TABLE = {
    "fp32": torch.float32, "float32": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0") or 0)


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1") or 1)


def _is_main_process() -> bool:
    return _rank() == 0


def _normalize_fmt(fmt: str) -> str:
    key = str(fmt).lower().lstrip(".")
    if key not in FORMATS:
        raise ValueError(f"不支持的 checkpoint 格式 {fmt!r}，可选: {list(FORMATS)}")
    return key


def _config_to_dict(config) -> Optional[dict]:
    """ModelConfig / dict / 有 to_dict() 的对象 -> 纯 dict。"""
    if config is None:
        return None
    if isinstance(config, ModelConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError(f"config 需要是 ModelConfig / dict，收到 {type(config)!r}")


def _resolve_dtype(dtype) -> Optional[torch.dtype]:
    """dtype -> torch.dtype；None 表示保持原精度。"""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key not in _DTYPE_TABLE:
        raise ValueError(
            f"不支持的保存精度 {dtype!r}，可选: 'bf16' | 'fp16' | 'fp32' | None，或直接传 torch.dtype")
    return _DTYPE_TABLE[key]


def _is_sharded(model: nn.Module) -> bool:
    """权重是不是被切过（FSDP2 / TP 的 DTensor，或 FSDP1 包装）。"""
    if DTensor is not None and any(isinstance(p, DTensor) for p in model.parameters()):
        return True
    return isinstance(FSDPModule, type) and isinstance(model, FSDPModule)


def _offload(obj):
    """递归把非 cpu/cuda 的张量（mps 之类）搬到 cpu；cuda 保持原样交给 DCP。"""
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        return t.to("cpu") if t.device.type not in ("cpu", "cuda") else t
    if isinstance(obj, dict):
        return {k: _offload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_offload(v) for v in obj)
    return obj


def _to_save_tensors(sd: dict, dtype: Optional[torch.dtype]) -> dict:
    """单文件落盘前的整理：detach -> contiguous -> cpu -> 可选转 dtype。"""
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if DTensor is not None and isinstance(v, DTensor):
            raise RuntimeError(
                f"权重 {k!r} 还是 DTensor 分片，不能写进单文件格式。"
                "FSDP/TP 请用 fmt='dcp'（内部会先做 full_tensor 聚合）"
            )
        t = v.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        t = t.to("cpu")
        # 只转浮点权重：整数 buffer 转了会坏掉；优化器状态也不在这里转（exp_avg 要留 fp32）
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        out[k] = t
    return out


def _cast_floating(sd: dict, dtype: Optional[torch.dtype]) -> dict:
    """dcp 用的按 dtype 转换（值可能是 DTensor 分片，不做 cpu 搬移）。"""
    if dtype is None:
        return sd
    return {k: (v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in sd.items()}


# ---------------------------------------------------------------------------
# 路径与格式识别
# ---------------------------------------------------------------------------
def _resolve_save_path(path, fmt: str) -> str:
    """pt 用原路径；safetensors 换成/补上 .safetensors；dcp 是目录（0_1999.pth -> 0_1999）。"""
    p = os.fspath(path).rstrip("/\\")
    stem, ext = os.path.splitext(p)
    if fmt == "safetensors":
        return p if ext.lower() == ".safetensors" else stem + ".safetensors"
    if fmt == "dcp":
        return stem if ext.lower() in _PT_SUFFIXES + (".safetensors",) else p
    return p


def detect_format(path) -> tuple:
    """按路径识别格式，返回 (fmt, 实际文件/目录)。传 save_checkpoint 的返回值即可。"""
    p = os.fspath(path)
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, ".metadata")):
            return "dcp", p
        for name in sorted(os.listdir(p)):
            if name.endswith(".safetensors"):
                return "safetensors", os.path.join(p, name)
        raise FileNotFoundError(f"{p} 是目录，但里面既没有 .metadata(dcp) 也没有 .safetensors")
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到 checkpoint: {p}")
    return ("safetensors" if p.endswith(".safetensors") else "pt"), p


# ---------------------------------------------------------------------------
# 读文件
# ---------------------------------------------------------------------------
def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        warnings.warn(
            f"weights_only=True 读取 {path} 失败（{exc}），回退到 weights_only=False；"
            "请只加载自己可信的 checkpoint。"
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _read_pt(path: str):
    """-> (state_dict, config, step, optimizer_state)。兼容老格式的裸 state_dict。"""
    obj = _torch_load(path)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} 里不是 dict，无法解析")
    if isinstance(obj.get(_DCP_MODEL_KEY), dict):
        return obj[_DCP_MODEL_KEY], obj.get("config"), obj.get("step"), obj.get(_DCP_OPTIM_KEY)
    return obj, None, None, None


def _read_safetensors(path: str, with_tensors: bool = True):
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise ImportError("读取 safetensors 需要 `pip install safetensors`") from exc

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()} if with_tensors else {}

    try:
        config = json.loads(meta["config"]) if meta.get("config") else None
    except json.JSONDecodeError as exc:
        warnings.warn(f"safetensors metadata 里的 config 解析失败：{exc}")
        config = None
    step = int(meta["step"]) if str(meta.get("step", "")).isdigit() else None

    optim_sd = None
    sidecar = os.path.splitext(path)[0] + ".optim.pt"
    if with_tensors and os.path.exists(sidecar):
        payload = _torch_load(sidecar)
        optim_sd = payload.get(_DCP_OPTIM_KEY, payload) if isinstance(payload, dict) else payload
    return sd, config, step, optim_sd


def _read_dcp_meta(ckpt_dir: str) -> dict:
    meta_path = os.path.join(ckpt_dir, _DCP_META_FILE)
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"{meta_path} 解析失败：{exc}")
        return {}


def read_config(path) -> Optional[ModelConfig]:
    """只读 config（不加载权重），用来在推理/断点续训时先建模型。"""
    fmt, art = detect_format(path)
    if fmt == "dcp":
        raw = _read_dcp_meta(art).get("config")
    elif fmt == "safetensors":
        raw = _read_safetensors(art, with_tensors=False)[1]
    else:
        raw = _read_pt(art)[1]
    return ModelConfig.from_dict(raw) if raw else None


# ---------------------------------------------------------------------------
# config 校验
# ---------------------------------------------------------------------------
def _verify_config(saved: Optional[dict], current, where: str) -> None:
    """saved 来自 checkpoint，current 来自本次训练/推理：shape 字段不一致直接报错。"""
    if saved is None or current is None:
        return
    current = _config_to_dict(current)

    fatal = {k: (saved[k], current[k]) for k in _SHAPE_KEYS
             if k in saved and k in current and saved[k] != current[k]}
    if fatal:
        detail = "，".join(f"{k}: ckpt={v[0]} vs 当前={v[1]}" for k, v in fatal.items())
        raise ValueError(
            f"checkpoint 的模型结构和当前 config 不一致（{where}）：{detail}。"
            "权重 shape 对不上，请用同一份 config，或删掉 resume 从头训。"
        )

    soft = {k: (saved.get(k), current.get(k)) for k in set(saved) | set(current)
            if k not in _SHAPE_KEYS and saved.get(k) != current.get(k)}
    if soft:
        detail = "，".join(f"{k}: ckpt={v[0]} vs 当前={v[1]}" for k, v in soft.items())
        warnings.warn(f"checkpoint 和当前 config 的非结构性字段不同（{where}）：{detail}")


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def _model_state_dict_for_save(model: nn.Module, gather: bool) -> dict:
    """gather=True 时 FSDP 会 all_gather 成完整权重（集合通信，所有 rank 都要调）。"""
    if _is_sharded(model):
        return get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=gather, cpu_offload=gather))
    return {k: v.detach() for k, v in model.state_dict().items()}


def _optimizer_state_dict_for_save(model: nn.Module, optimizer, sharded: bool):
    if optimizer is None:
        return None
    if sharded and StateDictOptions is not None:
        # FSDP2 的优化器状态也是 DTensor，交给官方接口
        return get_optimizer_state_dict(model, optimizer)
    return _offload(optimizer.state_dict())


def _save_single_file(model, path, fmt, cfg, optimizer, step, dtype, main) -> str:
    """pt / safetensors：rank0 写盘。gather 在所有 rank 上都会执行（FSDP 时是集合通信）。"""
    sd = _model_state_dict_for_save(model, gather=True)
    optim_sd = _optimizer_state_dict_for_save(model, optimizer, _is_sharded(model))
    if not main:
        return path

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("保存 safetensors 需要 `pip install safetensors`") from exc
        metadata = {"format": "pt", "format_version": "1"}
        if cfg is not None:
            metadata["config"] = json.dumps(cfg, ensure_ascii=False)
        if step is not None:
            metadata["step"] = str(int(step))
        save_file(_to_save_tensors(sd, dtype), path, metadata=metadata)
        if optim_sd is not None:
            # safetensors 只能放张量，优化器状态走旁路文件；始终 fp32
            torch.save({_DCP_OPTIM_KEY: _offload(optim_sd)},
                       os.path.splitext(path)[0] + ".optim.pt")
        return path

    payload = {"format_version": 1, "model": _to_save_tensors(sd, dtype)}
    if cfg is not None:
        payload["config"] = cfg
    if optim_sd is not None:
        payload[_DCP_OPTIM_KEY] = _offload(optim_sd)   # 始终 fp32：exp_avg 转 bf16 会掉精度
    if step is not None:
        payload["step"] = int(step)
    torch.save(payload, path)
    return path


def _save_dcp(model, ckpt_dir, cfg, optimizer, step, dtype, main) -> str:
    """DCP 目录：所有 rank 都必须调用（内部是集合通信）。"""
    if dcp is None:
        raise RuntimeError("当前 torch 不支持 torch.distributed.checkpoint，请升级 torch")

    if main and os.path.isdir(ckpt_dir):
        # 旧的 shard 文件名和当前 world_size 无关，不清理会残留脏数据
        shutil.rmtree(ckpt_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # dcp 本来就是集合通信，这里 barrier 不会造成“单边等待”

    # 只有 model 子树按 dtype 转；优化器状态原样保留，否则续训掉精度
    payload = {_DCP_MODEL_KEY: _cast_floating(_model_state_dict_for_save(model, False), dtype)}
    if optimizer is not None:
        payload[_DCP_OPTIM_KEY] = _optimizer_state_dict_for_save(
            model, optimizer, _is_sharded(model))

    dcp.save(payload, checkpoint_id=ckpt_dir)

    if main:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, _DCP_META_FILE), "w", encoding="utf-8") as f:
            json.dump({
                "format": "dcp", "format_version": 1, "config": cfg,
                "step": int(step) if step is not None else None,
                "world_size": _world_size(),
                "has_optimizer": optimizer is not None,
            }, f, ensure_ascii=False, indent=2)
    return ckpt_dir


def save_checkpoint(model: nn.Module, path, config=None, fmt: str = "pt", dtype=None,
                    optimizer=None, step: Optional[int] = None) -> str:
    """保存 checkpoint，返回真正落盘的路径（所有 rank 返回值一致）。

    model     : 普通 nn.Module / DDP / FSDP2(fully_shard) 都可以
    path      : pt 用文件；safetensors 会把后缀换成 .safetensors；dcp 是目录
    config    : ModelConfig / dict，一起存下来，加载时用来校验结构
    fmt       : "pt" | "safetensors" | "dcp"
    dtype     : "bf16" / "fp16" / "fp32" / None(保持原精度) 或 torch.dtype。
                只影响浮点权重；整数 buffer 和优化器状态都不转
    optimizer : 传了就把优化器状态也存下来（续训用）
    step      : 全局步数，续训时用它接着算 LR 调度
    """
    fmt = _normalize_fmt(fmt)
    save_dtype = _resolve_dtype(dtype)
    path = _resolve_save_path(path, fmt)
    main = _is_main_process()

    if fmt == "dcp":
        return _save_dcp(model, path, _config_to_dict(config), optimizer, step, save_dtype, main)
    return _save_single_file(model, path, fmt, _config_to_dict(config), optimizer, step,
                             save_dtype, main)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
@dataclass
class LoadResult:
    path: str
    fmt: str
    config: Optional[ModelConfig] = None
    step: Optional[int] = None
    optimizer_loaded: bool = False
    missing_keys: list = field(default_factory=list)
    unexpected_keys: list = field(default_factory=list)


def _apply_state_dict(model: nn.Module, sd: dict, strict: bool):
    """把 state_dict 灌进 model；FSDP/TP 的 DTensor 参数走官方的 set_model_state_dict。"""
    if _is_sharded(model):
        if StateDictOptions is None:  # pragma: no cover
            raise RuntimeError("当前 torch 不支持 set_model_state_dict，请升级 torch")
        # 传进来的是完整权重，由 FSDP2 自己切回本地 shard，不需要额外集合通信
        res = set_model_state_dict(
            model, sd, options=StateDictOptions(full_state_dict=True, strict=strict))
    else:
        res = model.load_state_dict(sd, strict=strict)
    return (list(getattr(res, "missing_keys", None) or []),
            list(getattr(res, "unexpected_keys", None) or []))


def _load_single_file(model, art, fmt, config, strict, optimizer) -> LoadResult:
    sd, saved_cfg, step, optim_sd = (
        _read_safetensors(art) if fmt == "safetensors" else _read_pt(art))

    _verify_config(saved_cfg, config, where=art)
    missing, unexpected = _apply_state_dict(model, sd, strict)
    if missing and not strict:
        warnings.warn(f"{art}: 缺少 {len(missing)} 个权重（strict=False 已忽略），例如 {missing[:3]}")

    optim_loaded = optimizer is not None and optim_sd is not None
    if optim_loaded:
        optimizer.load_state_dict(optim_sd)

    return LoadResult(art, fmt, ModelConfig.from_dict(saved_cfg) if saved_cfg else None,
                      step, optim_loaded, missing, unexpected)


def _dcp_optim_skeleton(optimizer, metadata) -> Optional[dict]:
    """给全新的 optimizer 造一个能被 dcp.load 填进去的骨架。

    DCP 默认的（公开）load 只处理“目标 state_dict 里已经存在的 key”：刚 new 出来的
    AdamW 的 state_dict()["state"] 是空 dict，直接 dcp.load 会一个参数都不加载并且
    **不报错**（静默失败，最难查）。这里按 metadata 里的 shape/dtype 先把结构建出来。
    """
    sd = optimizer.state_dict()
    if sd.get("state"):
        return None  # 已经在训的 optimizer，按正常路径 load 即可

    prefix = _DCP_OPTIM_KEY + ".state."
    state: dict = {}
    for fqn, md in metadata.state_dict_metadata.items():
        if not fqn.startswith(prefix):
            continue
        idx, _, field = fqn[len(prefix):].partition(".")
        if not field or not idx.isdigit():
            continue  # 非 "state.<int>.<field>" 的结构（如 FSDP 的 FQN key）不在这里处理
        entry = state.setdefault(int(idx), {})
        entry[field] = (torch.empty(md.size, dtype=md.properties.dtype)
                        if isinstance(md, dcp.TensorStorageMetadata) else None)
    if not state:
        return None
    sd["state"] = state
    return sd


def _load_dcp(model, ckpt_dir, config, strict, optimizer) -> LoadResult:
    if dcp is None:
        raise RuntimeError("当前 torch 不支持 torch.distributed.checkpoint，请升级 torch")

    meta = _read_dcp_meta(ckpt_dir)
    _verify_config(meta.get("config"), config, where=ckpt_dir)

    metadata = dcp.FileSystemReader(ckpt_dir).read_metadata()
    ckpt_keys = set(metadata.state_dict_metadata.keys())
    if not any(k == _DCP_MODEL_KEY or k.startswith(_DCP_MODEL_KEY + ".") for k in ckpt_keys):
        raise ValueError(f"{ckpt_dir} 里没有 'model' 权重，这个 dcp 目录不是 save_checkpoint 写的？")
    has_optim = any(k == _DCP_OPTIM_KEY or k.startswith(_DCP_OPTIM_KEY + ".") for k in ckpt_keys)
    want_optim = optimizer is not None and has_optim
    plan = DefaultLoadPlanner(allow_partial_load=not strict)
    optim_loaded = False

    if _is_sharded(model):
        # 官方推荐路径：本地 shard 的 state_dict -> dcp.load（自动 reshard）-> set_model_state_dict
        payload = {_DCP_MODEL_KEY: get_model_state_dict(model)}
        if want_optim:
            # FSDP2 对全新 optimizer 也会先把每个参数的 state 条目建出来（空 DTensor），可直接 load
            payload[_DCP_OPTIM_KEY] = get_optimizer_state_dict(model, optimizer)
        dcp.load(payload, checkpoint_id=ckpt_dir, planner=plan)
        res = set_model_state_dict(
            model, payload[_DCP_MODEL_KEY], options=StateDictOptions(strict=strict))
        if want_optim:
            set_optimizer_state_dict(model, optimizer, payload[_DCP_OPTIM_KEY],
                                     options=StateDictOptions(full_state_dict=True, strict=strict))
            optim_loaded = True
    else:
        # 单卡 / DDP：本地就是完整权重，DCP 会按 metadata 把 shard 拼回来
        payload = {_DCP_MODEL_KEY: model.state_dict()}
        if want_optim:
            optim_sd = _dcp_optim_skeleton(optimizer, metadata)
            if optim_sd is not None:
                payload[_DCP_OPTIM_KEY] = optim_sd
            elif not optimizer.state:
                warnings.warn(f"{ckpt_dir} 里的优化器状态没法恢复到一个全新的 optimizer，已跳过")
            else:
                payload[_DCP_OPTIM_KEY] = optimizer.state_dict()
        dcp.load(payload, checkpoint_id=ckpt_dir, planner=plan)
        res = model.load_state_dict(payload[_DCP_MODEL_KEY], strict=strict)
        if want_optim and _DCP_OPTIM_KEY in payload:
            optimizer.load_state_dict(payload[_DCP_OPTIM_KEY])
            optim_loaded = True

    return LoadResult(
        path=ckpt_dir, fmt="dcp",
        config=ModelConfig.from_dict(meta["config"]) if meta.get("config") else None,
        step=meta.get("step"), optimizer_loaded=optim_loaded,
        missing_keys=list(getattr(res, "missing_keys", None) or []),
        unexpected_keys=list(getattr(res, "unexpected_keys", None) or []),
    )


def load_into(model: nn.Module, path, config=None, strict: bool = True,
              optimizer=None) -> LoadResult:
    """把 checkpoint 读进 model（格式自动识别）。

    - 所有 rank 都要调用；DDP 时各 rank 各读一份文件，FSDP/dcp 走集合通信。
    - config 传当前训练的 config，会和 checkpoint 里存的做校验（shape 不一致直接报错）。
    - optimizer 传了就顺带恢复优化器状态（存过才行），看 result.optimizer_loaded。
    """
    fmt, art = detect_format(path)
    if fmt == "dcp":
        return _load_dcp(model, art, config, strict, optimizer)
    return _load_single_file(model, art, fmt, config, strict, optimizer)
