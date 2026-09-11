from transformers import AutoTokenizer

# TODO: 不依赖transformers训练一个BBPE和BPE

# 实际上我们用别人训练的更好的
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
