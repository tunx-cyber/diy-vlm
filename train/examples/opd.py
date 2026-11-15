import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from dataclasses import dataclass
from typing import Optional

from peft import LoraConfig, get_peft_model, PeftModel

# ============================================================
# 0. 配置（新增 LoRA 字段）
# ============================================================
@dataclass
class OPDConfig:
    teacher_name: str = "Qwen/Qwen3.5-9B"
    student_name: str = "Qwen/Qwen3.5-0.8B"
    dtype: str = "bfloat16"

    teacher_backend: str = "vllm"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen3.5-9B"
    top_k_teacher: int = 0

    dataset_name: str = "open-r1/DAPO-Math-17k-Processed"
    prompt_column: str = "prompt"

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    enable_thinking: bool = False

    lr: float = 1e-4                      # LoRA 通常比全量微调 lr 大 10x 左右
    grad_accum_steps: int = 4
    epochs: int = 1
    save_steps: int = 500
    max_grad_norm: float = 1.0
    output_dir: str = "./qwen35-0.8b-opd-lora"
    max_steps: Optional[int] = None

    # ---- LoRA 配置 ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_use_rslora: bool = True          # rsLoRA 在 Qwen3.5 上常见，缩放更稳定
    # 默认只挂 attention + MLP；若要线性注意力层，追加 linear_attn.* 三项
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


# ============================================================
# 1. 全局句柄
# ============================================================
teacher_model = None
teacher_client = None
student_model = None          # 这是 PEFT 包装后的模型
tokenizer = None


def _torch_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _build_lora_config(cfg: OPDConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=cfg.lora_use_rslora,
    )


def init_teacher_student(cfg: OPDConfig):
    global teacher_model, teacher_client, student_model, tokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.student_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ---- 学生：先加载基座，再用 LoRA 包装 ----
    if student_model is None:
        base = AutoModelForCausalLM.from_pretrained(
            cfg.student_name,
            torch_dtype=_torch_dtype(cfg.dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        lora_cfg = _build_lora_config(cfg)
        student_model = get_peft_model(base, lora_cfg)
        student_model.print_trainable_parameters()
        student_model.train()

    # ---- 教师 ----
    if cfg.teacher_backend == "transformers":
        if teacher_model is None:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                cfg.teacher_name,
                torch_dtype=_torch_dtype(cfg.dtype),
                device_map="auto",
                trust_remote_code=True,
            )
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            print("编译教师模型计算图")
            torch.compile(teacher_model)
            print("编译完成")
    elif cfg.teacher_backend == "vllm":
        if teacher_client is None:
            from openai import OpenAI
            teacher_client = OpenAI(base_url=cfg.vllm_base_url, api_key="EMPTY")
    else:
        raise ValueError(cfg.teacher_backend)

    # assert tokenizer.vocab_size == student_model.config.vocab_size


# ============================================================
# 2. 数据
# ============================================================
def load_dataset(cfg: OPDConfig):
    ds = datasets.load_dataset(cfg.dataset_name)["train"]
    if cfg.prompt_column not in ds.column_names:
        raise KeyError(f"{cfg.prompt_column} not in {ds.column_names}")
    return ds


def build_inputs(prompt_text: str, cfg: OPDConfig):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=cfg.enable_thinking,
    )
    return tokenizer(text, return_tensors="pt").to(student_model.device)


# ============================================================
# 3. Rollout
# ============================================================
@torch.no_grad()
def generate_rollout(cfg: OPDConfig, prompt_text: str):
    inputs = build_inputs(prompt_text, cfg)
    prompt_len = inputs.input_ids.shape[1]

    was_training = student_model.training
    student_model.eval()
    out = student_model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if was_training:
        student_model.train()

    full_ids = out[0]
    resp_ids = full_ids[prompt_len:]
    return inputs.input_ids[0], resp_ids, prompt_len


# ============================================================
# 4. 教师 logprob
# ============================================================
@torch.no_grad()
def _teacher_logprobs_transformers(full_ids: torch.Tensor, prompt_len: int):
    t_device = teacher_model.device
    ids = full_ids.unsqueeze(0).to(t_device)
    logits = teacher_model(input_ids=ids).logits
    resp_logits = logits[:, prompt_len - 1: -1, :]
    return F.log_softmax(resp_logits.float(), dim=-1).squeeze(0)


def _teacher_logprobs_vllm(cfg: OPDConfig, full_ids: torch.Tensor, prompt_len: int, resp_ids: torch.Tensor):
    ids = full_ids.tolist()
    resp = teacher_client.completions.create(
        model=cfg.vllm_model_name,
        prompt=ids,
        max_tokens=1,
        temperature=0.0,
        echo=True,
        logprobs=(-1 if cfg.top_k_teacher <= 0 else cfg.top_k_teacher),
    )
    prompt_logprobs = resp.choices[0].prompt_logprobs
    out = []
    for t in range(prompt_len, len(ids)):
        tok = str(ids[t])
        entry = prompt_logprobs[t]
        if entry is None:
            out.append(0.0); continue
        if tok in entry:
            out.append(entry[tok].logprob)
        else:
            out.append(min(v.logprob for v in entry.values()) - 1.0)
    return torch.tensor(out, dtype=torch.float32)


def get_teacher_log_prob(cfg, full_ids, resp_ids, prompt_len):
    if cfg.teacher_backend == "transformers":
        log_probs = _teacher_logprobs_transformers(full_ids, prompt_len).to(student_model.device)
        return log_probs.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)
    else:
        return _teacher_logprobs_vllm(cfg, full_ids, prompt_len, resp_ids).to(student_model.device)


# ============================================================
# 5. OPD Loss
# ============================================================
def get_opd_loss(cfg: OPDConfig, prompt_ids: torch.Tensor, resp_ids: torch.Tensor):
    prompt_len = prompt_ids.shape[0]
    full_ids = torch.cat([prompt_ids, resp_ids], dim=0)

    s_logits = student_model(input_ids=full_ids.unsqueeze(0)).logits
    s_resp_logits = s_logits[:, prompt_len - 1: -1, :]
    s_log_probs = F.log_softmax(s_resp_logits.float(), dim=-1)
    s_token_logp = s_log_probs.gather(-1, resp_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)

    t_token_logp = get_teacher_log_prob(cfg, full_ids, resp_ids, prompt_len)
    return (s_token_logp - t_token_logp).mean()


# ============================================================
# 6. 保存：只存 LoRA 适配器
# ============================================================
def save_student_lora(cfg: OPDConfig):
    student_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[save] LoRA adapter -> {cfg.output_dir}")


# ============================================================
# 7. 训练（优化器只取 LoRA 参数）
# ============================================================
def train(cfg: Optional[OPDConfig] = None):
    cfg = cfg or OPDConfig()
    init_teacher_student(cfg)
    ds = load_dataset(cfg)

    # ---- 关键：只优化 requires_grad=True 的参数 ----
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    print(f"trainable tensors: {len(trainable_params)}")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    student_model.train()
    optimizer.zero_grad()

    global_step, micro_step = 0, 0
    running_kl, running_n = 0.0, 0

    for epoch in range(cfg.epochs):
        for ex in ds:
            prompt_ids, resp_ids, _ = generate_rollout(cfg, ex[cfg.prompt_column])
            if resp_ids.numel() == 0:
                continue

            loss = get_opd_loss(cfg, prompt_ids, resp_ids)
            (loss / cfg.grad_accum_steps).backward()

            running_kl += loss.item(); running_n += 1
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                print(f"[epoch {epoch}] step {global_step} | reverse_kl {running_kl/running_n:.4f} "
                      f"| resp_len {resp_ids.numel()}")
                running_kl, running_n = 0.0, 0

                if cfg.save_steps and global_step % cfg.save_steps == 0:
                    save_student_lora(cfg)
                if cfg.max_steps and global_step >= cfg.max_steps:
                    save_student_lora(cfg); return

    if micro_step % cfg.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
        optimizer.step(); optimizer.zero_grad()

    save_student_lora(cfg)


if __name__ == "__main__":
    cfg = OPDConfig(
        teacher_backend="transformers",
        dataset_name="open-r1/DAPO-Math-17k-Processed",
        lr=1e-4,                  # LoRA 从 1e-4 起步，比全量微调的 1e-5 大
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_use_rslora=True,
        output_dir="./qwen35-0.8b-opd-lora",
    )
    train(cfg)