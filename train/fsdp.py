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
from .utils import get_lr

def load_model(model, path):
    # mmap=True reduces CPU memory usage
    full_sd = torch.load(
        path,
        mmap=True,
        weights_only=True,
        map_location='cpu',
    )
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
    # `assign=True` since we cannot call `copy_` on meta tensor
    model.load_state_dict(sharded_sd, assign=True)

def save_model(model,path):
    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        if torch.distributed.get_rank() == 0:
            cpu_state_dict[param_name] = full_param.half().cpu()
        else:
            del full_param
    torch.save(cpu_state_dict, path)

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
    totol_steps = 0
    dl_len = len(dataloader)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for step, (input_ids, label_ids) in enumerate(tqdm(dataloader)):
            totol_steps += 1
            input_ids, label_ids = input_ids.to(device), label_ids.to(device)

            cur_lr = get_lr(epoch * dl_len + step,epochs*dl_len,lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            loss = model(input_ids,labels = label_ids) / accumulation_steps
            loss.backward()
            if totol_steps % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_norm)
                optimizer.step()
                optimizer.zero_grad()
            if totol_steps % log_steps == 0 and rank == 0:
                print(loss.item()*accumulation_steps)
            if totol_steps % save_steps == 0 or step == dl_len-1:
                save_model(model,f"{epoch}_{step}.pth")
    dist.destroy_process_group()