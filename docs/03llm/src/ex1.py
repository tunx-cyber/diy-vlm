import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """把后半维取负放到前面：[x1, x2] -> [-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RoPE(nn.Module):
    """
    旋转位置编码。
    dim 是 head_dim（不是 d_model）。
    inv_freq[j] = 1 / base^(2j/dim)，j 越小频率越高。
    波长 lambda_j = 2*pi / inv_freq[j]。
    """
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # j = 0, 1, ..., dim/2-1
        # j=0 是最高频（波长最短），j 越大频率越低
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)          # [L, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [L, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(self, x):
        # x: [B, H, L, D]
        L = x.shape[-2]
        if L > self.max_seq_len:
            self._build_cache(L)

        cos = self.cos_cached[:L].unsqueeze(0).unsqueeze(0)  # [1,1,L,D]
        sin = self.sin_cached[:L].unsqueeze(0).unsqueeze(0)
        return x * cos + rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    """
    多头注意力 + RoPE，支持 GQA。
    n_kv_heads == n_heads 时是标准 MHA；
    n_kv_heads < n_heads 时是 GQA（如 Llama 3）。
    """
    def __init__(self, d_model, n_heads, n_kv_heads=None,
                 max_seq_len=8192, rope_base=10000.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = n_heads // self.n_kv_heads

        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RoPE(self.head_dim, max_seq_len, rope_base)
        self.dropout = dropout

    def forward(self, x, attention_mask=None):
        # x: [B, L, d_model]
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 作用在 q/k 的每个 head 上（v 不旋转）
        q = self.rope(q)
        k = self.rope(k)

        # GQA: 把 kv heads 复制到 n_heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0

        if attention_mask is None:
            # 纯因果注意力，SDPA 内部高效实现
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )
        else:
            # attention_mask: [B, L] bool，True 表示有效 token
            causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            pad = attention_mask[:, None, None, :].bool()   # [B,1,1,L]
            mask = causal & pad                              # [B,1,L,L]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p
            )

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)