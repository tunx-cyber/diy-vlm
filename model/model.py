import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    freqs= 1.0 / (rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))
    attn_factor = 1.0
    t = torch.arange(end,device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1) * attn_factor
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
            dropout: float = 0
        ):
        super().__init__()
        self.local_kv_heads = kv_heads
        self.local_heads = attn_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // attn_heads
        self.n_rep = attn_heads // kv_heads
        self.q_proj = nn.Linear(hidden_dim, self.head_dim * self.local_heads,bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * self.local_kv_heads,bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * self.local_kv_heads,bias=False)
        self.o_proj = nn.Linear(self.hidden_dim,self.hidden_dim)

        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim,eps=norm_eps)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        self.flash_attn = hasattr(F,'scaled_dot_product_attention')

    def forward(
            self,
            x: torch.Tensor,
            pos_embedding,
            past_key_value=None,
            use_cache=False,
            attention_mask=None
        ):
        bsz, seq_len, _ = x.shape
        self.is_causal = True
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

        xq = xq.transpose(1,2)
        xk = repeat_kv(xk, self.n_rep).transpose(1,2)
        xv = repeat_kv(xv, self.n_rep).transpose(1,2)

        if self.flash_attn and seq_len > 1:
            output = F.scaled_dot_product_attention(xq,xv,xv,attention_mask,self.dropout if self.training else 0.0,is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:,:,:,-seq_len:] += torch.full((seq_len,seq_len),float('-inf'),device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
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
    pass

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
            ropo_scaling = None,
            norm_eps: float = 1e-6,
            dropout: float = 0,
        ):
        super().__init__()
        self.embeddig = nn.Embedding(vocab_size,hidden_dim)
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
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=hidden_dim//attn_heads,
            end=max_position_embeddings,
            rope_base=rope_base,
            rope_scaling=ropo_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            input_ids, 
            attention_mask=None, 
            past_key_values=None, 
            use_cache=False, 
            **kwargs
        ):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embeddig(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # if self.freqs_cos[0, 0] == 0:
        #     freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        #     self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
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
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class CausalModel(nn.Module):
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
            ropo_scaling = None,
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
            ropo_scaling=ropo_scaling,
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
            **kwargs
        ):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.model_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return loss

    @torch.inference_mode()
    def generate(
            self, 
            inputs=None, 
            attention_mask=None, 
            max_new_tokens=8192, 
            temperature=0.85, 
            top_p=0.85, 
            top_k=50, 
            eos_token_id=2, 
            streamer=None, 
            use_cache=True, 
            num_return_sequences=1, 
            do_sample=True, 
            repetition_penalty=1.0, 
            **kwargs
        ):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
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

class VisionModel(CausalModel):
    def __init__(
            self,
            vision_model_path: str,# google/siglip2-base-patch32-256
            layers:int ,
            vocab_size:int, 
            kv_heads: int,
            attn_heads: int,
            hidden_dim: int,
            intermediate_dim:int,
            max_position_embeddings: int,
            rope_base: int,
            ropo_scaling = None,
            norm_eps: float = 1e-6,
            dropout: float = 0,
        ):
        super().__init__(
            layers=layers,
            vocab_size=vocab_size,
            kv_heads=kv_heads,
            attn_heads=attn_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            ropo_scaling=ropo_scaling,
            norm_eps=norm_eps,
            dropout=dropout
        )
        from transformers import AutoModel, AutoProcessor
        self.vision_encoder = AutoModel.from_pretrained(vision_model_path).eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(vision_model_path)

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs
    
    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)


def main():
    model = CausalModel(
        layers=8,
        vocab_size=6400,
        kv_heads=8,
        attn_heads=32,
        hidden_dim=512,
        intermediate_dim=int(512*8/3),
        max_position_embeddings=16*1024,
        rope_base=1e6,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/(1024*1024))
    input_ids = torch.randint(1, 127, (8, 128))
    labels_ids = torch.randint(1, 127, (8, 128))
    print(model(input_ids,labels=labels_ids))
