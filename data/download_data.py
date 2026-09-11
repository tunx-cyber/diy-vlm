from datasets import load_dataset
import json
def dl_ds():
    ds = load_dataset("karpathy/tinystories-gpt4-clean",cache_dir="hf_data/")
    for item in ds["train"]:
        with open("train.jsonl","a") as f:
            f.write(json.dumps({
                    "text":item["text"]
                },ensure_ascii=False)+'\n'
            )
