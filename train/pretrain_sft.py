"""预训练 / SFT：纯 PyTorch 训练循环（不依赖 transformers / Trainer）。

要点：
- 混合精度：bf16 不需要 loss scaling；fp16 自动配 GradScaler；--dtype auto 自动挑
- 梯度累积：--accumulation-steps N，每 N 个 micro-batch 才更新一次参数
- 梯度裁剪 / warmup + 余弦学习率 / 定期存 ckpt / --resume-from 续训

用法:
    python train/pretrain_sft.py --train-type sft --data-path sft_process.jsonl \
        --max-len 1024 --batch-size 8 --accumulation-steps 4 --dtype bf16
"""

import argparse
import math
import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 直接 `python train/pretrain_sft.py` 运行时，把项目根目录加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.make_dataset import PretrainDataset, SFTDataset, collate_fn
from model.model import VisionModel
from model.template import IMAGE_PAD_TOKEN
from model.tokenizer import tokenizer
from train.utils import (count_parameters, get_lr, load_checkpoint, pick_device,
                         resolve_autocast, save_checkpoint, setup_seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="预训练 / SFT（纯 PyTorch 训练循环）")
    # 数据
    parser.add_argument("--data-path", default="train.jsonl", help="jsonl 数据集路径")
    parser.add_argument("--train-type", default="pretrain", choices=("pretrain", "sft"))
    parser.add_argument("--max-len", type=int, default=512, help="单条样本最大 token 数")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数")
    # batch / 训练
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32, help="每张卡的 micro-batch")
    parser.add_argument("--accumulation-steps", type=int, default=8,
                        help="梯度累积的 micro-batch 数；等效 batch = batch_size * 该值")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="余弦退火的下限比例")
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu/mps")
    parser.add_argument("--seed", type=int, default=42)
    # 日志 / 存档
    parser.add_argument("--log-steps", type=int, default=8, help="每多少个优化步打一次日志")
    parser.add_argument("--save-steps", type=int, default=200,
                        help="每多少个优化步存一次；<=0 只在训练结束存")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--resume-from", default=None, help="从哪个 ckpt 续训")
    # 模型结构
    parser.add_argument("--vision-model-path", default="google/siglip2-base-patch32-256")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--attn-heads", type=int, default=32)
    parser.add_argument("--intermediate-dim", type=int, default=512 * 3 // 8)
    parser.add_argument("--max-position-embeddings", type=int, default=16 * 1024)
    parser.add_argument("--rope-base", type=float, default=1e6)
    return parser.parse_args(argv)


def build_model(args) -> VisionModel:
    return VisionModel(
        vision_model_path=args.vision_model_path,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN),
        layers=args.layers,
        vocab_size=len(tokenizer),      # 必须用 len()，见 model/tokenizer.py
        kv_heads=args.kv_heads,
        attn_heads=args.attn_heads,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        max_position_embeddings=args.max_position_embeddings,
        rope_base=args.rope_base,
    )


def build_dataset(args, image_config):
    if args.train_type == "pretrain":
        cls = PretrainDataset
    else:
        cls = SFTDataset
    return cls(args.data_path, tokenizer=tokenizer, max_len=args.max_len,
               image_config=image_config)


def train(args):
    setup_seed(args.seed)
    device = pick_device(args.device)
    use_amp, amp_dtype = resolve_autocast(args, device)

    model = build_model(args).to(device)
    model.train()   # VisionModel.train() 会把冻结的视觉塔重新压回 eval

    dataset = build_dataset(args, model.image_config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"[info] device={device} amp={'off' if not use_amp else str(amp_dtype).replace('torch.', '')} "
          f"参数总量={total_params / 1e6:.2f}M 可训练={trainable_params / 1e6:.2f}M")
    print(f"[info] 样本 {len(dataset)} 条 | micro-batch {args.batch_size} | "
          f"累积 {args.accumulation_steps} -> 等效 batch {args.batch_size * args.accumulation_steps}")

    # 冻结的视觉塔不参与优化器，省掉一次无用的 grad 检查
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                      weight_decay=args.weight_decay)
    # fp16 必须靠 GradScaler 防止梯度下溢；bf16 的指数范围和 fp32 一样，不需要
    scaler = torch.amp.GradScaler(
        device.type, enabled=(use_amp and amp_dtype == torch.float16))

    steps_per_epoch = max(1, math.ceil(len(loader) / args.accumulation_steps))
    total_steps = steps_per_epoch * args.epochs

    global_step, start_epoch = 0, 0
    if args.resume_from:
        global_step, start_epoch, _ = load_checkpoint(
            args.resume_from, model, optimizer, scaler, map_location=device)
        print(f"[info] 从 {args.resume_from} 续训：step={global_step} epoch={start_epoch}")

    def optimizer_step(micro_batches: int):
        """梯度裁剪 + 更新参数，返回 (grad_norm, lr)。

        梯度已经在 backward 前除以过 accumulation_steps，这里直接 step。
        micro_batches 只用于断言，防止窗口边界算错。
        """
        if micro_batches != args.accumulation_steps:
            # 末尾不完整的窗口只在收尾时出现，属于预期
            pass
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_norm)
        lr = get_lr(global_step, total_steps, args.lr,
                    warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return float(grad_norm), lr

    optimizer.zero_grad(set_to_none=True)
    micro_in_window = 0        # 当前累积窗口里已经做了几个 micro-batch
    loss_window = 0.0
    last_mean_loss = float("nan")
    last_grad_norm, last_lr = 0.0, args.lr
    started = time.perf_counter()
    last_log = started

    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("模型没有返回 loss：batch 里缺少 labels")
                if outputs.aux_loss is not None:
                    loss = loss + outputs.aux_loss
                # 先除以累积步数，N 个 micro-batch 的梯度加起来正好是均值
                loss_to_backward = loss / args.accumulation_steps

            scaler.scale(loss_to_backward).backward()
            micro_in_window += 1
            loss_window += float(loss.detach())

            if micro_in_window < args.accumulation_steps:
                continue

            last_grad_norm, last_lr = optimizer_step(micro_in_window)
            global_step += 1
            micro_in_window = 0
            mean_loss = loss_window / args.accumulation_steps
            last_mean_loss = mean_loss
            loss_window = 0.0

            if args.log_steps > 0 and global_step % args.log_steps == 0:
                now = time.perf_counter()
                window = now - last_log
                tokens = args.batch_size * args.accumulation_steps * args.max_len
                print(f"epoch {epoch} | step {global_step} | "
                      f"loss {mean_loss:.4f} | lr {last_lr:.2e} | "
                      f"grad_norm {last_grad_norm:.3f} | "
                      f"{tokens / max(window, 1e-6):.0f} tok/s", flush=True)
                last_log = now

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                path = os.path.join(args.save_dir, f"step_{global_step}.pt")
                save_checkpoint(path, model, optimizer, scaler,
                                step=global_step, epoch=epoch,
                                extra={"train_args": vars(args)})
                print(f"[info] 已保存 {path}")

    # 最后一个不完整的累积窗口也要更新，否则末尾几个 micro-batch 白算
    if micro_in_window > 0:
        last_grad_norm, last_lr = optimizer_step(micro_in_window)
        global_step += 1

    final_path = os.path.join(args.save_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, scaler,
                    step=global_step, epoch=args.epochs,
                    extra={"train_args": vars(args)})
    elapsed = time.perf_counter() - started
    print(f"[done] {global_step} 个优化步，耗时 {elapsed / 60:.1f} min，"
          f"最后 loss {last_mean_loss:.4f}")
    print(f"[done] 权重已保存到 {final_path}")
    return model


def main(argv=None):
    args = parse_args(argv)
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps 必须 >= 1")
    train(args)


if __name__ == "__main__":
    main()
