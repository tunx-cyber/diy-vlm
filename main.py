from data.download_data import dl_ds

dl_ds()

from train.pretrain import train
device = "mps"
train("./train.jsonl",device=device)