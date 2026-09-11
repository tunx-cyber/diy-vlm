import torch
from torch.optim import AdamW
from model.model import CausalModel
from model.tokenizer import tokenizer, vocab_size
from data.make_dataset import PretrainDataset
from torch.utils.data import DataLoader
import json
from .utils import get_lr,save_model
from tqdm import tqdm

def train(data_path: str, epochs=1, device="mps"):
    pt_ds = PretrainDataset(
        data_path,
        tokenizer=tokenizer,
        max_len = 512
    )
    dataloader = DataLoader(pt_ds,batch_size=32)
    model = CausalModel(
        layers=8,
        vocab_size=vocab_size,
        kv_heads=8,
        attn_heads=32,
        hidden_dim=512,
        intermediate_dim=int(512*8/3),
        max_position_embeddings=16*1024,
        rope_base=1e6,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print("训练总参数：",num_params/(1024*1024))
    optimizer = AdamW(model.parameters(),lr=5e-5)
    accumulation_steps = 8
    log_steps = 8
    save_steps = 200

    scaler = torch.amp.grad_scaler.GradScaler(device)
    for epoch in range(epochs):
        for step,(input_ids, label_ids) in enumerate(tqdm(dataloader)):
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            
            lr = get_lr(epoch * len(dataloader)  + step, epochs * len(dataloader), 5e-5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                loss = model(input_ids,labels=label_ids)

            scaler.scale(loss/accumulation_steps).backward()
            if step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % log_steps == 0:
                print(loss.item()*accumulation_steps)

            if step % save_steps == 0:
                save_model(model,f"{step}.pth")
