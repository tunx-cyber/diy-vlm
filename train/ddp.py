import os
import contextlib

import torch
import torch.distributed as dist
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .utils import get_lr, make_attention_mask
from data.make_dataset import PretrainDataset
from model.tokenizer import tokenizer
from model.checkpoint import load_into, save_checkpoint

def cleanup():
    dist.destroy_process_group()

def load_model(model, path, device=None, config=None, optimizer=None):
    """断点续训：把 checkpoint 权重读进 model（所有 rank 都调用）。

    统一走 model/checkpoint.py 的 load_into：格式（pt / safetensors / dcp）自动识别，
    并校验 checkpoint 里的 config 和当前 config 是否一致。
    传了 optimizer 就顺带恢复优化器状态（AdamW 的 exp_avg / exp_avg_sq / step），
    是否真的恢复看返回值的 optimizer_loaded。
    """
    return load_into(model, path, config=config, optimizer=optimizer)

def save_model(model, path, config=None, fmt="pt", optimizer=None, step=None,
               dtype="bf16"):
    """保存权重，所有 rank 都调用，只有 rank0 实际写盘（fmt="dcp" 时所有 rank 都写）。

    这里绝对不能加 dist.barrier() 之类的集合通信：调用点一旦写成
    `if ... and rank == 0: save_model(...)`，其他 rank 就永远到不了 barrier，
    整个训练会卡死（要等 NCCL watchdog 超时才报错，很难定位）。
    """
    return save_checkpoint(model, path, config=config, fmt=fmt, optimizer=optimizer,
                           step=step, dtype=dtype)

def train(
        data_path: str,
        model: torch.nn.Module,
        config=None,
        epochs: int = 1,
        lr: float = 5e-5,
        batch_size = 16,
        max_len = 512,
        accumulation_steps = 8,
        log_steps = 1,
        save_steps= 2000,
        max_norm = 1.0,
        resume_from: str = None,
        num_workers: int = 0,
        save_format: str = "pt"
    ):
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # 多机时 rank 是全局序号，DEVICE 要用 LOCAL_RANK；不 set_device 会挂错卡
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    model = model.to(local_rank)
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print("训练总参数：",num_params/(1024*1024))

    # 优化器必须在 resume 之前创建：断点续训要把优化器状态一起恢复。
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    start_step = 0   # 续训时已经跑完的步数（checkpoint 里存的 step），用来接着算 LR
    if resume_from:
        result = load_model(model, resume_from, device, config, optimizer=optimizer)
        start_step = result.step or 0
        if rank == 0:
            print(f"已从 {resume_from} 恢复权重"
                  f"（优化器状态: {'已恢复' if result.optimizer_loaded else '未恢复'}，"
                  f"已训练 {start_step} 步）")

    pt_ds = PretrainDataset(data_path=data_path,tokenizer=tokenizer,max_len=max_len)
    sampler = DistributedSampler(pt_ds,num_replicas=world_size,rank=rank,shuffle=True)
    dataloader = DataLoader(pt_ds,batch_size=batch_size,sampler=sampler,num_workers=num_workers)

    ddp_model = DDP(model,device_ids=[local_rank])
    dl_len = len(dataloader)
    lr_total_steps = start_step + epochs * dl_len
    total_steps = start_step

    for epoch in range(epochs):
        # 不加这句的话每个 epoch 的 shuffle 顺序完全一样
        sampler.set_epoch(epoch)
        for step,(input_ids, label_ids) in enumerate(tqdm(dataloader)):
            total_steps += 1
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            # 训练也必须把 padding mask 传进去，否则会 attend 到 pad 位置
            attention_mask = make_attention_mask(input_ids, tokenizer.pad_token_id)

            cur_lr = get_lr(total_steps - 1, lr_total_steps, lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            sync_ctx = ddp_model.no_sync() if total_steps % accumulation_steps != 0 \
                else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    loss = ddp_model(input_ids,attention_mask=attention_mask,labels=label_ids).loss

                # bf16 的指数范围和 fp32 一样，不存在下溢，不需要 GradScaler
                (loss/accumulation_steps).backward()
            if total_steps % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_norm)
                optimizer.step()
                optimizer.zero_grad()

            if total_steps % log_steps == 0 and rank == 0:
                print(loss.item()*accumulation_steps)

            if total_steps % save_steps == 0 or step == dl_len - 1:
                save_model(model, f"{epoch}_{step}.pth", config,
                           fmt=save_format, optimizer=optimizer, step=total_steps)

    cleanup()
