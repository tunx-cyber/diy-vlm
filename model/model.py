import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)

    def forward(self, x: torch.Tensor):
        return (self.weight*self.norm(x.float())).type_as(x)

def precompute_freqs_cis(
        dim: int, 
        end: int = int(32*1024), 
        rope_base: float = 1e6, 
        rope_scaling: dict = None):
    if dim % 2 != 0:
        # 下面 cat([cos, cos]) 会得到 dim-1 列，和 head_dim 对不上
        raise ValueError(f"RoPE 的 dim 必须是偶数，收到 {dim}")
    if rope_scaling:
        # 以前这个参数收了就丢掉，静默失效比直接报错更难查
        raise NotImplementedError("rope_scaling 还没有实现，请保持 None")
    freqs = 1.0 / (rope_base**(torch.arange(0,dim,2).float()/dim))
    t = torch.arange(end,device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)
    return freqs_cos, freqs_sin

def apply_rope_emb(q, k, cos, sin, unzqueeze_dim=1):
    def rotate_half(x: torch.Tensor):
        return torch.cat((-x[..., x.shape[-1] // 2:],x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unzqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unzqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unzqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unzqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :].expand(batch_size,seq_len,num_kv_heads,n_rep,head_dim).\
        reshape(batch_size,seq_len,num_kv_heads*n_rep,head_dim))

class Attention(nn.Module):
    def __init__(
            self,
            kv_heads: int,
            attn_heads: int,
            hidden_dim: int,
            norm_eps: float = 1e-6,
            dropout: float = 0,
            is_causal: bool = True
        ):
        super().__init__()
        if attn_heads <= 0 or kv_heads <= 0:
            raise ValueError(f"attn_heads/kv_heads 必须为正数，收到 {attn_heads}/{kv_heads}")
        if hidden_dim % attn_heads != 0:
            raise ValueError(
                f"hidden_dim({hidden_dim}) 必须能被 attn_heads({attn_heads}) 整除，"
                "否则 head_dim 被截断，q_proj 的输出和 o_proj 的输入对不上")
        if attn_heads % kv_heads != 0:
            raise ValueError(
                f"attn_heads({attn_heads}) 必须是 kv_heads({kv_heads}) 的整数倍（GQA）")
        head_dim = hidden_dim // attn_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim({head_dim}) 必须是偶数：RoPE 按前半/后半切分")
        self.local_kv_heads = kv_heads
        self.local_heads = attn_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_rep = attn_heads // kv_heads
        self.q_proj = nn.Linear(hidden_dim, self.head_dim * self.local_heads,bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * self.local_kv_heads,bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * self.local_kv_heads,bias=False)
        self.o_proj = nn.Linear(self.hidden_dim,self.hidden_dim,bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim,eps=norm_eps)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        # 配置放在 __init__ 里，不要在 forward 里改模块状态（对 torch.compile 不友好）
        self.is_causal = is_causal
        self.flash_attn = hasattr(F,'scaled_dot_product_attention')

    def build_attn_mask(self, bsz: int, seq_len: int, kv_len: int, attention_mask, device, force: bool = False):
        if attention_mask is not None and attention_mask.shape[-1] != kv_len:
            # 否则下面 reshape 会抛 "shape '[1,1,1,12]' is invalid for input of size 11"，
            # 完全看不出是 mask 长度错了
            raise ValueError(
                f"attention_mask 长度({attention_mask.shape[-1]})和 kv_len({kv_len})不一致："
                "增量解码时要保证 mask 覆盖 [0, past_len + 新 token)")
        if seq_len > kv_len:
            # query 比 key 还长时，右下角对齐的因果 mask 会出现整行全 False，
            # softmax 全 -inf 直接产生 NaN（不报错，最难查）
            raise ValueError(
                f"query 长度({seq_len})不能大于 key 长度({kv_len})："
                "请检查 past_key_value / attention_mask 的长度是否和输入对得上")
        if attention_mask is None and (seq_len == 1 or (kv_len == seq_len and not force)):
            # 没有 padding、也不需要按 cache 对齐时，交给 SDPA 的 is_causal 处理，
            # 省掉一次 mask 分配（q_len == kv_len 时 is_causal 就是普通因果 mask）
            return None
        if self.is_causal:
            # j <= i + (kv_len - seq_len) 即右下角对齐的因果 mask
            mask = torch.ones(seq_len, kv_len, dtype=torch.bool, device=device).tril(kv_len - seq_len)
        else:
            # 非因果注意力（例如 vision）只屏蔽 padding，不能套因果 mask
            mask = torch.ones(seq_len, kv_len, dtype=torch.bool, device=device)
        mask = mask[None, None].expand(bsz, 1, seq_len, kv_len)
        if attention_mask is not None:
            # attention_mask 既可能是 0/1 的 int，也可能是 bool，统一按 bool 处理
            mask = mask & attention_mask.reshape(bsz, 1, 1, kv_len).bool()
        return mask.contiguous()

    def forward(
            self,
            x: torch.Tensor,
            pos_embedding,
            past_key_value=None,
            use_cache=False,
            attention_mask=None
        ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        cos, sin = pos_embedding
        xq, xk = apply_rope_emb(xq,xk,cos,sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0],xk],dim=1)# 在seq_len维度对齐
            xv = torch.cat([past_key_value[1],xv],dim=1)

        past_kv = (xk,xv) if use_cache else None
        kv_len = xk.shape[1]

        xq = xq.transpose(1,2)
        xk = repeat_kv(xk, self.n_rep).transpose(1,2)
        xv = repeat_kv(xv, self.n_rep).transpose(1,2)

        attn_mask = self.build_attn_mask(bsz, seq_len, kv_len, attention_mask, xq.device, force=not self.flash_attn)
        if self.flash_attn:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                # 只有「没有 mask 且 q_len == kv_len」才能让 SDPA 自己造因果 mask
                is_causal=self.is_causal and attn_mask is None and kv_len == seq_len,
            )
        else:
            scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            output = self.attn_dropout(F.softmax(scores.float(),dim=-1).type_as(xq)) @ xv
        output = output.transpose(1,2).reshape(bsz,seq_len,-1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
        
class FeedForward(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int
        ):
        super().__init__()
        self.gate = nn.Linear(hidden_size,intermediate_size,bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size,bias=False)
        self.up   = nn.Linear(hidden_size, intermediate_size,bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor):
        x = self.act_fn(self.gate(x)) * self.up(x)
        return self.down(x)

class MOEFeedForward(nn.Module):
    """MoE FFN 占位（路由逻辑还没接上）。

    子类请把路由的 auxiliary loss 写到 self.aux_loss；
    """

    def __init__(self):
        super().__init__()
        self.aux_loss = 0.0

class DecoderBlock(nn.Module):
    def __init__(
            self,
            layer_id:int, 
            kv_heads: int,
            attn_heads: int,
            hidden_dim: int,
            intermediate_dim:int,
            norm_eps: float = 1e-6,
            dropout: float = 0,

        ):
        super().__init__()
        self.attention = Attention(
            kv_heads=kv_heads,
            attn_heads=attn_heads,
            hidden_dim=hidden_dim,
            norm_eps=norm_eps,
            dropout=dropout,
        )
        self.pre_norm = RMSNorm(hidden_dim,eps=norm_eps)
        self.post_attention_norm = RMSNorm(hidden_dim,eps=norm_eps)
        self.mlp = FeedForward(hidden_size=hidden_dim,intermediate_size=intermediate_dim)


    def forward(
            self, 
            hidden_states, 
            pos_embedding, 
            past_key_value=None, 
            use_cache=False, 
            attention_mask=None
        ):
        residual = hidden_states
        hidden_states, present_kv = self.attention(
            self.pre_norm(hidden_states),
            pos_embedding,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_norm(hidden_states)
        )
        return hidden_states, present_kv

class Model(nn.Module):
    def __init__(
            self, 
            layers:int ,
            vocab_size:int, 
            kv_heads: int,
            attn_heads: int,
            hidden_dim: int,
            intermediate_dim:int,
            max_position_embeddings: int,
            rope_base: int,
            rope_scaling = None,
            norm_eps: float = 1e-6,
            dropout: float = 0,
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            DecoderBlock(
                l,
                kv_heads=kv_heads,
                attn_heads=attn_heads,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                norm_eps=norm_eps,
                dropout=dropout
            )
            for l in range(layers)
        )
        self.norm = RMSNorm(hidden_dim,norm_eps)
        self.rope_dim = hidden_dim // attn_heads
        self.rope_end = max_position_embeddings
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.register_buffer("freqs_cos", None, persistent=False)
        self.register_buffer("freqs_sin", None, persistent=False)
        self.rope_ready = False
        self.init_rope_buffers()

    def init_rope_buffers(self, device=None, dtype=None):
        if device is None:
            device = self.embedding.weight.device
        if dtype is None:
            dtype = self.embedding.weight.dtype
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.rope_dim,
            end=self.rope_end,
            rope_base=self.rope_base,
            rope_scaling=self.rope_scaling,
        )
        self.freqs_cos = freqs_cos.to(device=device, dtype=dtype)
        self.freqs_sin = freqs_sin.to(device=device, dtype=dtype)
        self.rope_ready = True

    def forward(
            self,
            input_ids, 
            attention_mask=None, 
            past_key_values=None, 
            use_cache=False, 
            inputs_embeds=None,
            **kwargs
        ):
        # 方便vision model透传
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("input_ids 和 inputs_embeds 必须且只能传一个")
        if inputs_embeds is not None:
            hidden_states = self.dropout(inputs_embeds)
            seq_len = hidden_states.shape[1]
        else:
            hidden_states = self.dropout(self.embedding(input_ids))
            seq_len = input_ids.shape[1]

        # 处理past_kv
        if past_key_values is not None and not isinstance(past_key_values, (list, tuple)):
            raise TypeError(
                f"past_key_values 需要 list[(k, v)]，收到 {type(past_key_values).__name__}；"
                "HF 的 DynamicCache 请先用 to_legacy_cache() 转换")
        if past_key_values is None:
            past_key_values = (None,) * len(self.layers)   # 元组，不额外占用/可共享
        elif len(past_key_values) != len(self.layers):
            raise ValueError(
                f"past_key_values 长度({len(past_key_values)})和层数({len(self.layers)})不一致，"
                "请确认传回来的是同一个模型")
        start_pos = 0
        for past in past_key_values:
            if past is not None:
                start_pos = past[0].shape[1]
                break
        # RoPE 表按需重建（正常训练/推理只在第一次 forward 触发一次）
        if (not self.rope_ready or self.freqs_cos is None
                or self.freqs_cos.is_meta
                or self.freqs_cos.device != hidden_states.device):
            self.init_rope_buffers(hidden_states.device, self.embedding.weight.dtype)
        if start_pos + seq_len > self.freqs_cos.shape[0]:
            raise ValueError(
                f"位置超出 RoPE 预计算范围：需要 [{start_pos}, {start_pos + seq_len})，"
                f"但只预计算了 {self.freqs_cos.shape[0]} 个位置（max_position_embeddings）。"
                "请增大 max_position_embeddings 或缩短输入/生成长度")
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_len], self.freqs_sin[start_pos:start_pos + seq_len])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        # 累加 MoE 的 aux loss；用 getattr 兜底，子类忘了定义 aux_loss 也不会崩
        aux_loss = hidden_states.new_zeros((), dtype=torch.float32)
        for layer in self.layers:
            if isinstance(layer.mlp, MOEFeedForward):
                aux_loss = aux_loss + getattr(layer.mlp, "aux_loss", 0.0)
        return hidden_states, presents, aux_loss

class CausalLMOutput:
    def __init__(self, logits=None, past_key_values=None, loss=None, aux_loss=None):
        self.logits = logits
        self.past_key_values = past_key_values
        self.loss = loss
        self.aux_loss = aux_loss

    def __getitem__(self, item):
        return (self.logits, self.past_key_values, self.loss)[item]


class TextModel(nn.Module):
    def __init__(
            self,
            layers:int ,
            vocab_size:int, 
            kv_heads: int,
            attn_heads: int,
            hidden_dim: int,
            intermediate_dim:int,
            max_position_embeddings: int,
            rope_base: int,
            rope_scaling = None,
            norm_eps: float = 1e-6,
            dropout: float = 0,
        ):
        super().__init__()
        self.model = Model(
            layers=layers,
            vocab_size=vocab_size,
            kv_heads=kv_heads,
            attn_heads=attn_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            norm_eps=norm_eps,
            dropout=dropout
        )
        self.model_head = nn.Linear(hidden_dim,vocab_size,bias=False)

    def forward(
            self, 
            input_ids, 
            attention_mask=None, 
            past_key_values=None, 
            use_cache=False, 
            logits_to_keep=0, 
            labels=None, 
            inputs_embeds=None,
            **kwargs
        ):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, inputs_embeds=inputs_embeds, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.model_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            if logits.size(1) != labels.size(1):
                raise ValueError(
                    f"labels 长度({labels.size(1)})和 logits 长度({logits.size(1)})不一致："
                    "训练时 logits_to_keep 必须为 0，不能只保留最后几个 token"
                )
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return CausalLMOutput(
            logits=logits, past_key_values=past_key_values, loss=loss, aux_loss=aux_loss)

    @torch.inference_mode()
    def generate(
            self, 
            inputs=None, 
            attention_mask=None, 
            max_new_tokens=8192, 
            temperature=0.85, 
            top_p=0.85, 
            top_k=50, 
            eos_token_id=None, 
            streamer=None, 
            use_cache=True, 
            do_sample=True, 
            repetition_penalty=1.0, 
            **kwargs
        ):

        input_ids = kwargs.pop("input_ids", inputs)
        if input_ids is None:
            raise ValueError("generate 需要传入 inputs（或 input_ids）")
        if not torch.is_tensor(input_ids):
            # 兼容 HF 的用法：list / tuple 也接受
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and not torch.is_tensor(attention_mask):
            attention_mask = torch.as_tensor(attention_mask)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        past_key_values = kwargs.pop("past_key_values", None)
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0；要贪心解码请用 do_sample=False")
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # past_key_values 可能是 [None]*layers（例如 use_cache=False 时 forward 的
            # 返回值原样传回来），这时 [0] 是 None，不能直接取 [0][0]
            past_len = 0
            if past_key_values:
                first_past = past_key_values[0]
                if first_past is not None:
                    past_len = first_past[0].shape[1]
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                # top_k 不能超过词表大小，否则 torch.topk 直接报 index out of range
                k = min(top_k, logits.size(-1))
                logits[logits < torch.topk(logits, k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


class VisionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,out_dim),
            nn.GELU(),
            nn.Linear(out_dim,out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

try:  # 既支持 `import model.model`，也支持直接运行本文件
    from .vision import load_siglip_vision_tower
except ImportError:  # pragma: no cover
    from vision import load_siglip_vision_tower


class VisionModel(TextModel):
    def __init__(
            self,
            vision_model_path: str,# google/siglip2-base-patch32-256
            image_token_id: int,
            layers:int ,
            vocab_size:int, 
            kv_heads: int,
            attn_heads: int,
            hidden_dim: int,
            intermediate_dim:int,
            max_position_embeddings: int,
            rope_base: int,
            rope_scaling = None,
            norm_eps: float = 1e-6,
            dropout: float = 0,
        ):
        super().__init__(
            layers=layers, vocab_size=vocab_size, kv_heads=kv_heads, attn_heads=attn_heads,
            hidden_dim=hidden_dim, intermediate_dim=intermediate_dim,
            max_position_embeddings=max_position_embeddings, rope_base=rope_base,
            rope_scaling=rope_scaling, norm_eps=norm_eps, dropout=dropout,
        )
        self.image_token_id = image_token_id
        # 纯 PyTorch 的视觉塔，权重直接从 safetensors 读，不经过 transformers
        self.vision_encoder, self.image_config = load_siglip_vision_tower(vision_model_path)
        self.vision_hidden_size = self.vision_encoder.hidden_size
        self.num_image_tokens = self.vision_encoder.num_patches
        self.vision_proj = VisionProjector(
            in_dim=self.vision_hidden_size, out_dim=hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()   # 冻结塔永远 eval，否则父类 .train() 会把它一起打开
        return self

    @torch.no_grad()
    def encode_frozen(self, pixel_values):
        """只跑冻结的视觉塔，返回 [N * P, C] 的特征（不带梯度）。"""
        vp = self.vision_encoder
        param = next(vp.parameters())
        feats = vp(pixel_values.to(device=param.device, dtype=param.dtype))  # [N, P, C]
        n, p, c = feats.shape
        if p != self.num_image_tokens:
            raise ValueError(
                f"视觉塔输出 {p} 个 token，但模板里每张图是 {self.num_image_tokens} 个 "
                "IMAGE_TOKEN_LEN：两者必须一致")
        return feats.reshape(n * p, c)

    def encode_images(self, pixel_values, dtype):
        """pixel_values: [N, 3, 256, 256] -> [N * P, out_dim]

        投影层必须留在 no_grad 外面，否则 vision_proj 永远拿不到梯度，
        训练时 loss 照降（文本塔在学），但视觉对齐完全没生效。
        """
        if pixel_values is None:
            return None
        feats = self.encode_frozen(pixel_values)
        return self.vision_proj(feats.to(dtype))

    def replace_vision_marker_embedding(self, input_ids, vision_embeds,
                                        inputs_embeds=None, image_token_id=None):
        token_id = self.image_token_id if image_token_id is None else image_token_id
        if inputs_embeds is None:
            inputs_embeds = self.model.embedding(input_ids)
        if vision_embeds is None:
            return inputs_embeds
        marker = input_ids.eq(token_id)
        if int(marker.sum()) != vision_embeds.shape[0]:
            raise ValueError(
                f"图像特征数({vision_embeds.shape[0]})和 <|image_pad|> 数({int(marker.sum())})不一致")
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[marker] = vision_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            labels: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            vision_embeds: Optional[torch.Tensor] = None,
            image_token_id: Optional[int] = None,
            **kwargs
        ):
        if input_ids is None:
            raise ValueError("VisionModel.forward 需要 input_ids：要用它定位 <|image_pad|>")
        token_id = self.image_token_id if image_token_id is None else image_token_id

        inputs_embeds = self.model.embedding(input_ids)
        marker = input_ids.eq(token_id)

        # 只有 prefill 才可能有 image_pad。用 past 是否为空来判断，而不是
        # marker.any()：后者会每步触发一次 GPU->CPU 同步，而且在 use_cache=False
        # 的生成里会让每个 token 都重跑一遍视觉塔。
        has_past = past_key_values is not None and any(p is not None for p in past_key_values)
        if not has_past and bool(marker.any()):
            if vision_embeds is None:
                if pixel_values is None:
                    raise ValueError("序列里有 <|image_pad|> 但没传 pixel_values")
                vision_embeds = self.encode_images(pixel_values, dtype=inputs_embeds.dtype)
            inputs_embeds = self.replace_vision_marker_embedding(
                input_ids, vision_embeds, inputs_embeds, token_id)

        if labels is not None:
            labels = labels.clone()
            labels[marker] = -100   # 图像 token 不参与 loss

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
            self,
            inputs=None,
            attention_mask=None,
            *,
            pixel_values: Optional[torch.FloatTensor] = None,
            **kwargs
        ):
        """带图生成。

        父类的 generate 每一步都会调 self.forward(...) 并原样透传 kwargs，
        所以这里先把图片编码一次，再把 vision_embeds 透传下去。

        pixel_values 是 keyword-only：父类签名里第 2 个位置参数是 attention_mask，
        写成 model.generate(prompt, pv) 会把图片当成 mask 传进去，报错完全看不出原因。
        """
        # 视觉塔只跑一次，把特征直接透传给每一步的 forward。
        # 否则 use_cache=False 时每个 token 都要重跑一遍 12 层视觉塔。
        vision_embeds = None
        if pixel_values is not None:
            dtype = next(self.parameters()).dtype
            vision_embeds = self.encode_images(pixel_values, dtype=dtype)

        return super().generate(
            inputs=inputs,
            attention_mask=attention_mask,
            vision_embeds=vision_embeds,
            **kwargs,
        )

