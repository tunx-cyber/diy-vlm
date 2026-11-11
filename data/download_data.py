import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

import sys
from datasets import load_dataset

def dl_ds(
        text_output_path="pretrain_text.jsonl",
        vision_output_path="sft_vision.jsonl"
    ):

    REPO = "openbmb/Ultra-FineWeb-L3"
    CONFIGS = [
        "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
    ]
    CACHE_DIR = "hf_data/"
    limit = 100000
    for config in CONFIGS:
        ds = load_dataset(REPO, config, streaming=True, cache_dir=CACHE_DIR, split="train")
        with open(text_output_path, "a", encoding="utf-8") as f:
            for i, item in enumerate(ds):
                if i > limit:
                    break
                if (i+1) % 500==0:
                    print(f"{config}: {i+1}/{limit}")
                f.write(
                        json.dumps(
                            {"messages": [{"role": "assistant", "content": f"{item['content']}"}]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    print("openbmb/Ultra-FineWeb-L3 downloading finish")

    ds = load_dataset("InnovatorLab/Innovator-VL-Instruct-46M",streaming=True, cache_dir=CACHE_DIR, split="train")
    limit = 10000
    with open(vision_output_path, "a", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            if i > limit:
                break
            if (i+1) % 500 == 0:
                print(f"InnovatorLab/Innovator-VL-Instruct-46M: {i+1}/{limit}")
            os.mkdir(f"imgs/{i}")
            for j, img in enumerate(item["images"]):
                img.save(f"imgs/{i}/{j}.png")
            f.write(
                    json.dumps(
                        {"messages": item["conversations"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

def _list_images(img_dir):
    """按文件名里的序号列出图片，保证和下载时 item["images"] 的顺序一致。"""
    if not os.path.isdir(img_dir):
        return []

    def sort_key(name):
        stem = os.path.splitext(name)[0]
        return (0, int(stem)) if stem.isdigit() else (1, name)

    return [os.path.join(img_dir, name) for name in sorted(os.listdir(img_dir), key=sort_key)]


def post_process(sft_path="sft_vision.jsonl", out_path="sft_process.jsonl", img_root="imgs"):
    """
    human 的 value 按 <image> 切成 parts 数组：
      * 文本段 -> {"type": "text", "text": ...}
      * 每个 <image> -> {"type": "image", "image": "imgs/<idx>/<序号>.png"}
    图片按下载顺序（0.png, 1.png, ...）取，对应 item["images"] 里的地址顺序；
    model/template.py::render_content 会把每张图展开成 IMAGE_TOKEN_LEN 个
    <|image_pad|>，并把地址按同样顺序放进 pixel_vals。
    """
    total = 0
    with open(sft_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as wf:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            img_paths = _list_images(os.path.join(img_root, str(idx)))
            messages = []
            img_cnt = 0  # 当前对话已经用掉的图片数
            for msg in data["messages"]:
                role = msg.get("from")
                value = msg.get("value") or ""
                if role == "human":
                    parts = []
                    # "<image>\n问题" -> ["", "\n问题"]，按分隔位置插图片
                    chunks = value.split("<image>")
                    for k, chunk in enumerate(chunks):
                        if chunk.strip():
                            parts.append({"type": "text", "text": chunk.strip()})
                        if k == len(chunks) - 1:
                            continue
                        if img_cnt >= len(img_paths):
                            raise ValueError(
                                f"{sft_path} 第 {idx} 行的 <image> 比图片文件多: "
                                f"{value[:80]!r} / {img_paths}"
                            )
                        parts.append({"type": "image", "image": img_paths[img_cnt]})
                        img_cnt += 1
                    messages.append({"role": "user", "content": parts})
                elif role == "gpt":
                    messages.append({"role": "assistant", "content": value})
                else:
                    raise ValueError(f"{sft_path} 第 {idx} 行出现未知 role: {role!r}")

            if img_cnt != len(img_paths):
                print(
                    f"[warn] 第 {idx} 行只用了 {img_cnt}/{len(img_paths)} 张图: {img_paths}",
                    file=sys.stderr,
                )
            wf.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            total += 1
    print(f"post_process 完成: {sft_path} -> {out_path} 共 {total} 条")
