from dataclasses import dataclass, asdict, fields, replace

@dataclass
class ModelConfig:
    layers: int = 8
    vocab_size: int = 0                      # 由 tokenizer 决定，用 default_config(vocab_size=...) 填
    kv_heads: int = 4
    attn_heads: int = 16
    hidden_dim: int = 512
    intermediate_dim: int = 512 * 8 // 3
    max_position_embeddings: int = 32 * 1024
    rope_base: float = 1e6
    rope_scaling: dict | None = None         # 还没实现，非 None 会直接报错（见 precompute_freqs_cis）
    norm_eps: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self):
        if self.attn_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("attn_heads / kv_heads 必须为正数")
        if self.hidden_dim % self.attn_heads != 0:
            raise ValueError(
                f"hidden_dim({self.hidden_dim}) 必须能被 attn_heads({self.attn_heads}) 整除")
        if self.attn_heads % self.kv_heads != 0:
            raise ValueError(
                f"attn_heads({self.attn_heads}) 必须是 kv_heads({self.kv_heads}) 的整数倍（GQA）")
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim({self.head_dim}) 必须是偶数：RoPE 按前半/后半切分")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.attn_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, strict: bool = True) -> "ModelConfig":
        """从 dict 建配置，忽略不认识的历史遗留键。

        strict=True 时会检查关键字段是否存在：防止把 **HF 风格的 config.json**
        （字段名是 hidden_size / num_hidden_layers / num_attention_heads ...）
        误当成本项目 config 传进来 —— 那样所有键都会被忽略，最后静默用默认架构
        建模型，等到加载权重才报一堆 shape mismatch，非常难查。
        """
        known = {f.name for f in fields(cls)}
        if strict:
            required = {"layers", "hidden_dim", "attn_heads", "vocab_size"}
            missing = required - set(d)
            if missing:
                raise ValueError(
                    f"这不像本项目的 model config（缺少 {sorted(missing)}）。"
                    f"HF 的 config.json 字段名不同（hidden_size / num_hidden_layers），"
                    f"不要混用。收到的键: {sorted(d)[:8]}..."
                )
        return cls(**{k: v for k, v in d.items() if k in known})

    def model_kwargs(self) -> dict:
        """展开成 CausalModel(**kwargs)。"""
        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size 还没设置：请用 default_config(vocab_size=len(tokenizer))")
        return asdict(self)


DEFAULT_CONFIG = ModelConfig()


def default_config(vocab_size: int | None = None, **overrides) -> ModelConfig:
    """DEFAULT_CONFIG 加覆盖项；vocab_size 一般传 len(tokenizer)。"""
    cfg = replace(DEFAULT_CONFIG, **overrides)
    if vocab_size is not None:
        cfg = replace(cfg, vocab_size=vocab_size)
    return cfg
