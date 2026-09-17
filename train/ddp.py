import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .utils import get_lr
from data.make_dataset import PretrainDataset
from model.tokenizer import tokenizer

def cleanup():
    dist.destroy_process_group()

def load_model(ddp_model,path):
    rank = dist.get_rank()
    acc = torch.accelerator.current_accelerator()
    # configure map_location properly
    map_location = {f'{acc}:0': f'{acc}:{rank}'}
    ddp_model.load_state_dict(
    torch.load(path, map_location=map_location, weights_only=True))

def save_model(ddp_model,path):
    rank = dist.get_rank()
    if rank == 0:
        torch.save(ddp_model.state_dict(), path)
    dist.barrier()

def train(
        data_path: str,
        model: torch.nn.Module,
        epochs: int = 1,
        lr: float = 5e-5,
        batch_size = 16,
        max_len = 512,
        accumulation_steps = 8,
        log_steps = 1,
        save_steps= 2000,
        max_norm = 1.0
    ):
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = f"cuda:{rank}"
    model = model.to(rank)
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print("训练总参数：",num_params/(1024*1024))

    pt_ds = PretrainDataset(data_path=data_path,tokenizer=tokenizer,max_len=max_len)
    sampler = DistributedSampler(pt_ds,num_replicas=world_size,rank=rank,shuffle=True)
    dataloader = DataLoader(pt_ds,batch_size=batch_size,sampler=sampler)

    ddp_model = DDP(model,device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr)
    scaler = torch.amp.grad_scaler.GradScaler(device)
    dl_len = len(dataloader)
    totol_steps = 0

    for epoch in range(epochs):
        for step,(input_ids, label_ids) in enumerate(tqdm(dataloader)):
            totol_steps += 1
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            
            cur_lr = get_lr(epoch * dl_len + step,epochs*dl_len,lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                loss = model(input_ids,labels=label_ids)

            scaler.scale(loss/accumulation_steps).backward()
            if totol_steps % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if totol_steps % log_steps == 0:
                print(loss.item()*accumulation_steps)

            if totol_steps % save_steps == 0:
                save_model(model,f"{step}.pth")

    cleanup()

