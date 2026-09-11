from transformers import AutoTokenizer

# TODO: 不依赖transformers训练一个BBPE和BPE

# 实际上我们用别人训练的更好的
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

# 词表真实大小必须用 len()，不能用 tokenizer.vocab_size：
# 后者只是基础 BPE 词表（248044），不含 added special tokens，
# 而 pad=248044 / eos=248046 这些 id 同样会进 embedding，
# vocab_size 小于 len(tokenizer) 会让 embedding 越界 -> CUDA device-side assert。
vocab_size = len(tokenizer)
