export HF_ENDPOINT=https://hf-mirror.com

pip install -r requirements.txt

mkdir hf_data

python3 main.py

python infer/simple_infer.py --prompt "once upon a time" --max-new-tokens 256 --ckpt 200.pth --greedy --device cuda