# torchrun --nproc_per_node 2 main.py
from model.model import CausalModel
from model.tokenizer import tokenizer,vocab_size
from train.ddp import train

if __name__ == '__main__':
    model = CausalModel(
        layers=8,
        vocab_size=vocab_size,
        kv_heads=4,
        attn_heads=16,
        hidden_dim=512,
        intermediate_dim=(512*8//3),
        max_position_embeddings=32*1024,
        rope_base=1e6,
    )
    train(
        data_path="train.jsonl",
        model=model,
        epochs=1
    )