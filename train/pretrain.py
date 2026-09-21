'''
单机单卡训练的sample
'''

import torch
from torch.optim import AdamW
from model.config import default_config
from model.model import CausalModel
from model.tokenizer import tokenizer, vocab_size
from model.checkpoint import load_into, save_checkpoint
from data.make_dataset import PretrainDataset
from torch.utils.data import DataLoader
from .utils import get_lr,make_attention_mask
from tqdm import tqdm

def train(
        data_path: str,
        epochs=1,
        device="mps",
        config=None,
        max_len=512,
        batch_size=32,
        lr=5e-5,
        accumulation_steps=8,
        log_steps=8,
        save_steps=200,
        save_format="pt",
        resume_from=None
    ):
    # 模型结构统一从 model/config.py 取，和 main.py / simple_infer.py 保持一致
    config = config or default_config(vocab_size=vocab_size)
    pt_ds = PretrainDataset(
        data_path,
        tokenizer=tokenizer,
        max_len = max_len
    )
    dataloader = DataLoader(pt_ds,batch_size=batch_size)
    model = CausalModel(**config.model_kwargs()).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print("训练总参数：",num_params/(1024*1024))
    optimizer = AdamW(model.parameters(),lr=lr)
    device_type = torch.device(device).type

    start_step = 0   # 续训时已经跑完的步数（checkpoint 里存的 step），用来接着算 LR
    if resume_from:
        # 优化器先建好再 load：断点续训把权重和优化器状态（AdamW 的
        # exp_avg / exp_avg_sq / step）一起恢复，否则续训初期会有明显的 loss 抖动
        result = load_into(model, resume_from, config=config, optimizer=optimizer)
        start_step = result.step or 0
        print(f"已从 {resume_from} 恢复权重"
              f"（优化器状态: {'已恢复' if result.optimizer_loaded else '未恢复'}，"
              f"已训练 {start_step} 步）")

    dl_len = len(dataloader)
    # LR 调度接着上次的位置往下走：分子是全局步数，分母 = 已完成步数 + 本次要跑的步数。
    # 老 checkpoint 没存 step 时 start_step=0，和原来的 get_lr(epoch*dl_len+step, ...) 完全一致。
    # 想复现原始学习率曲线，就让「已完成 + 本次」= 原计划总步数。
    lr_total_steps = start_step + epochs * dl_len
    total_steps = start_step
    for epoch in range(epochs):
        for step,(input_ids, label_ids) in enumerate(tqdm(dataloader)):
            total_steps += 1
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            # 训练也要传 padding mask，否则会 attend 到 pad 位置
            attention_mask = make_attention_mask(input_ids, tokenizer.pad_token_id)

            cur_lr = get_lr(total_steps - 1, lr_total_steps, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

            with torch.autocast(device_type=device_type,dtype=torch.bfloat16):
                loss = model(input_ids,attention_mask=attention_mask,labels=label_ids).loss

            # bf16 的指数范围和 fp32 一样，不存在下溢，不需要 GradScaler；
            # 以前套了 GradScaler 只会多一层没用的缩放（换 fp16 时再加回来）
            (loss/accumulation_steps).backward()
            # 用 total_steps（先自增再取模）：原来是 step % accumulation_steps == 0，
            # step 从 0 开始，第一次更新只用了 1/accumulation_steps 个 batch
            if total_steps % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if total_steps % log_steps == 0:
                print(loss.item()*accumulation_steps)

            if total_steps % save_steps == 0 or step == dl_len - 1:
                # 权重存 bf16（和训练精度一致）；优化器状态始终 fp32。
                # step 记的是全局步数，续训时用它接着算 LR 调度
                save_checkpoint(model, f"{epoch}_{step}.pth", config=config,
                                fmt=save_format, dtype="bf16",
                                optimizer=optimizer, step=total_steps)
