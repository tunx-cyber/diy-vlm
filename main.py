# torchrun --nproc_per_node 2 main.py
from model.config import default_config
from model.model import CausalModel
from model.tokenizer import vocab_size
from train.ddp import train

if __name__ == '__main__':
    # 配置只在 model/config.py 里定义一份，main.py / pretrain.py / simple_infer.py
    # 都从它取，避免三个文件各写一套超参（以前 head 数量不一致，权重直接加载不了）
    config = default_config(vocab_size=vocab_size)
    model = CausalModel(**config.model_kwargs())
    train(
        data_path="train.jsonl",
        model=model,
        config=config,
        epochs=1
    )
