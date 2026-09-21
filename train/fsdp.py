# torchrun --nproc_per_node 2 fsdp.py
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

from model.tokenizer import tokenizer
from data.make_dataset import PretrainDataset
from model.checkpoint import load_into, save_checkpoint
from .utils import get_lr, make_attention_mask

# FSDP2 下参数是切过的 DTensor，自己写 full_tensor / distribute_tensor 很容易写错，
# 统一交给 model/checkpoint.py（内部用官方的 get_model_state_dict / set_model_state_dict）。
def load_model(model, path, config=None, optimizer=None):
    """断点续训：传了 optimizer 就顺带恢复优化器状态（存过才行）。"""
    return load_into(model, path, config=config, optimizer=optimizer)

def save_model(model, path, config=None, fmt="pt", optimizer=None, step=None,
               dtype="bf16"):
    """所有 rank 都必须调用（FSDP2 的 gather 是集合通信），只有 rank0 落盘。

    权重按 bf16 存（FSDP2 的 param_dtype 本来就是 bf16，dcp 分片里也是 bf16）；
    优化器状态保持 fp32。
    """
    return save_checkpoint(model, path, config=config, fmt=fmt,
                           optimizer=optimizer, step=step, dtype=dtype)

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
        save_format: str = "pt",
        resume_from: str = None
    ):
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    pt_ds = PretrainDataset(data_path=data_path,tokenizer=tokenizer,max_len=max_len)
    sampler = DistributedSampler(pt_ds,num_replicas=world_size,rank=rank,shuffle=True)
    dataloader = DataLoader(pt_ds,batch_size=batch_size,sampler=sampler)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer,mp_policy=mp_policy)
    fully_shard(model,mp_policy=mp_policy)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print("Params: ", num_params/(1024*1024))

    optimizer = AdamW(model.parameters(),lr=lr)
    start_step = 0   # 续训时已经跑完的步数（checkpoint 里存的 step），用来接着算 LR
    if resume_from:
        # 优化器已经建好了，传进去才能把 AdamW 的 exp_avg / exp_avg_sq / step 一起恢复
        result = load_model(model, resume_from, config, optimizer=optimizer)
        start_step = result.step or 0
        if rank == 0:
            print(f"已从 {resume_from} 恢复权重"
                  f"（优化器状态: {'已恢复' if result.optimizer_loaded else '未恢复'}，"
                  f"已训练 {start_step} 步）")
    dl_len = len(dataloader)
    lr_total_steps = start_step + epochs * dl_len
    totol_steps = start_step
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for step, (input_ids, label_ids) in enumerate(tqdm(dataloader)):
            totol_steps += 1
            input_ids, label_ids = input_ids.to(device), label_ids.to(device)
            # 训练也要传 padding mask，否则会 attend 到 pad 位置
            attention_mask = make_attention_mask(input_ids, tokenizer.pad_token_id)

            cur_lr = get_lr(totol_steps - 1, lr_total_steps, lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            loss = model(input_ids,attention_mask=attention_mask,labels = label_ids).loss / accumulation_steps
            loss.backward()
            if totol_steps % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_norm)
                optimizer.step()
                optimizer.zero_grad()
            if totol_steps % log_steps == 0 and rank == 0:
                print(loss.item()*accumulation_steps)
                
            if totol_steps % save_steps == 0 or step == dl_len-1:
                save_model(model, f"{epoch}_{step}.pth", config,
                           fmt=save_format, optimizer=optimizer, step=totol_steps)
    dist.destroy_process_group()