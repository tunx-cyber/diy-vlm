from data.download_data import dl_ds
import os
if not os.path.exists("train.jsonl"):
    dl_ds()

from train.pretrain import train
import torch
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
train("./train.jsonl",device=device)