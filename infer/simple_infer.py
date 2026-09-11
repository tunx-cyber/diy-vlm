# 实现有kv cache的生成，不依赖transformers
"""流式推理：KV Cache 逐 token 生成 + 边生成边输出文本。

生成跑在后台线程里，主线程从 streamer 里取已经解码好的文本片段并 flush 到终端，
所以是真正意义上的流式输出（首 token 出来就能看到），而不是等生成结束再一次性打印。

用法:
    python infer/simple_infer.py --prompt "once upon a time" --max-new-tokens 256
    python infer/simple_infer.py --ckpt 200.pth --greedy --device cuda
"""
import argparse
import os
import queue
import sys
import threading
import time

import torch

# 直接 `python infer/simple_infer.py` 时把项目根目录加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import CausalModel
from model.tokenizer import tokenizer


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextStreamer:
    """把 generate 产出的 token 增量解码成文本，供主线程迭代消费。

    用法和 transformers 的 TextStreamer 一致：
        streamer = TextStreamer(tokenizer)
        model.generate(..., streamer=streamer)
    生成线程调用 put()/end()，主线程 `for chunk in streamer` 拿文本。
    """

    _END = object()

    def __init__(self, tokenizer, skip_prompt: bool = True, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.num_tokens = 0          # 已生成的 token 数（不含 prompt）
        self.exception = None        # 生成线程里的异常，回传给主线程抛出
        self._queue = queue.Queue()
        self._ids = []
        self._decoded_len = 0
        self._first_put = True

    def put(self, token_ids):
        """generate 每步调用一次；第一次是整段 prompt，之后是单个新 token。"""
        if torch.is_tensor(token_ids):
            token_ids = token_ids.reshape(-1).tolist()
        elif not isinstance(token_ids, (list, tuple)):
            token_ids = [token_ids]

        if self._first_put:
            self._first_put = False
            if self.skip_prompt:
                return

        self._ids.extend(int(t) for t in token_ids)
        self.num_tokens = len(self._ids)

        text = self.tokenizer.decode(self._ids, skip_special_tokens=self.skip_special_tokens)
        # byte-level BPE 可能把多字节字符切在两个 token 里，结尾的半个字符先压住，
        # 等下一个 token 补齐再输出，避免吐出乱码（）
        text = text.rstrip("\ufffd")
        if len(text) > self._decoded_len:
            self._queue.put(text[self._decoded_len:])
            self._decoded_len = len(text)

    def end(self):
        self._queue.put(self._END)

    def __iter__(self):
        while True:
            chunk = self._queue.get()
            if chunk is self._END:
                return
            yield chunk


def load_model(ckpt: str, device: str) -> CausalModel:
    model = CausalModel(
        layers=8,
        vocab_size=248044,
        kv_heads=8,
        attn_heads=32,
        hidden_dim=512,
        intermediate_dim=int(512 * 8 / 3),
        max_position_embeddings=16 * 1024,
        rope_base=1e6,
    )
    if ckpt and os.path.exists(ckpt):
        state_dict = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[info] 已加载权重 {ckpt}", file=sys.stderr)
    else:
        print(
            f"[warn] 没找到权重文件 {ckpt}，用随机初始化权重跑（只用于验证流式输出链路）",
            file=sys.stderr,
        )
    return model.to(device).eval()


def stream_generate(
    model: CausalModel,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = 256,
    temperature: float = 0.85,
    top_p: float = 0.85,
    top_k: int = 50,
    do_sample: bool = True,
    eos_token_id: int = None,
    use_cache: bool = True,
):
    """流式生成并直接打印，返回 (prompt + 生成文本, 生成 token 数)。"""
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = input_ids.to(device)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    gen_kwargs = dict(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
        use_cache=use_cache,
        streamer=streamer,
    )

    def worker():
        try:
            model.generate(**gen_kwargs)
        except BaseException as exc:  # 保证主线程不会卡在 queue.get() 上
            streamer.exception = exc
        finally:
            streamer.end()

    thread = threading.Thread(target=worker, daemon=True)
    start = time.perf_counter()
    thread.start()

    print(prompt, end="", flush=True)
    first_chunk_at = None
    text = prompt
    for chunk in streamer:            # 主线程边收边打印
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
        print(chunk, end="", flush=True)
        text += chunk
    print()
    thread.join()

    if streamer.exception is not None:
        raise streamer.exception

    elapsed = time.perf_counter() - start
    ttft = (first_chunk_at - start) if first_chunk_at else elapsed
    speed = streamer.num_tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[info] 生成 {streamer.num_tokens} tokens，耗时 {elapsed:.2f}s，"
        f"首 token {ttft * 1000:.0f}ms，速度 {speed:.1f} tok/s",
        file=sys.stderr,
    )
    return text, streamer.num_tokens


def main():
    parser = argparse.ArgumentParser(description="KV Cache 流式推理")
    parser.add_argument("--ckpt", default="200.pth", help="权重路径")
    parser.add_argument("--prompt", default="once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true", help="贪心解码，关闭采样")
    parser.add_argument("--no-cache", action="store_true", help="关闭 KV Cache（仅用于对比）")
    parser.add_argument("--device", default="auto", help="auto/cuda/mps/cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device(args.device)
    model = load_model(args.ckpt, device)
    stream_generate(
        model,
        args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.greedy,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
