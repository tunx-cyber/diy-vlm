# 简化版LLM + MoE层示例，支持多批次token输入
# 依赖：torch >=1.8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# 字节级分词器
class ByteTokenizer:
    """
    简单字节级分词器：
    - 将每个字节 0..255映射为token id 0..255
    - 提供特殊 token: <bos>=256, <eos>=257, <pad>=258
    """
    def __init__(self):
        self.vocab_size = 259
        self.bos = 256
        self.eos = 257
        self.pad = 258

    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        b = text.encode('utf-8', errors='surrogatepass')
        ids = list(b)
        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]
        return ids

    def batch_encode(self, texts: List[str], pad_to=None) -> Tuple[torch.LongTensor, torch.LongTensor]:
        encs = [self.encode(t) for t in texts]
        maxlen = max(len(x) for x in encs) if pad_to is None else pad_to
        pad = self.pad
        arr = [x + [pad] * (maxlen - len(x)) for x in encs]
        lengths = torch.LongTensor([len(x) for x in encs])
        return torch.LongTensor(arr), lengths

# 简单自注意力
class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_mask = (~(mask.bool().unsqueeze(1).unsqueeze(2))) * -1e9
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)

# MoE层
class MoELayer(nn.Module):
    """
    简化 MoE 层
    - d_model: 输入输出维度
    - d_ff: 专家内部隐藏维度
    - n_experts: 专家数量
    - k: top-k激活专家数
    - capacity_factor: 每个专家容量系数
    """
    def __init__(self, d_model, d_ff, n_experts=4, k=1, capacity_factor=1.25, noisy_gating=True):
        super().__init__()
        assert k in (1,2)
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating

        # 门控网络
        self.w_gating = nn.Linear(d_model, n_experts, bias=False)
        if noisy_gating:
            self.w_noise = nn.Linear(d_model, n_experts, bias=False)

        # 专家网络，每个是两层FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])

    def _noisy_logits(self, x):
        logits = self.w_gating(x)
        if self.noisy_gating and self.training:
            noise_std = torch.sigmoid(self.w_noise(x))
            logits = logits + torch.randn_like(logits) * noise_std
        return logits

    def forward(self, x, mask=None):
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        logits = self._noisy_logits(x_flat)
        scores = F.softmax(logits, dim=-1)

        if self.k == 1:
            top1 = torch.argmax(scores, dim=-1)
            dispatch_mask = F.one_hot(top1, num_classes=self.n_experts).to(x.dtype)
            combine_weights = torch.gather(scores, 1, top1.unsqueeze(1)).squeeze(1)
            capacity = int((N/self.n_experts)*self.capacity_factor)+1

            expert_inputs = []
            expert_indices = []
            for e in range(self.n_experts):
                idx = torch.nonzero(dispatch_mask[:, e], as_tuple=False).squeeze(-1)
                if idx.numel() > capacity:
                    idx = idx[:capacity]
                expert_inputs.append(x_flat[idx])
                expert_indices.append(idx)

            out_flat = torch.zeros_like(x_flat)
            for e in range(self.n_experts):
                if expert_inputs[e].size(0)==0:
                    continue
                y = self.experts[e](expert_inputs[e])
                out_flat[expert_indices[e]] = y
            out_flat = out_flat * combine_weights.unsqueeze(1)
            return out_flat.view(B, T, D)
        else:
            # Top-2简化实现
            topk_vals, topk_idx = torch.topk(scores, k=2, dim=-1)
            capacity = int((N/self.n_experts)*self.capacity_factor)+1
            expert_buckets = [[] for _ in range(self.n_experts)]
            for i in range(N):
                for j in range(2):
                    e = int(topk_idx[i,j].item())
                    w = float(topk_vals[i,j].item())
                    expert_buckets[e].append((i,w))

            out_flat = torch.zeros_like(x_flat)
            for e in range(self.n_experts):
                bucket = expert_buckets[e]
                if len(bucket)==0: continue
                if len(bucket) > capacity:
                    bucket = bucket[:capacity]
                idxs = torch.tensor([i for i,_ in bucket], device=x.device, dtype=torch.long)
                weights = torch.tensor([w for _,w in bucket], device=x.device, dtype=x.dtype)
                inp = x_flat[idxs]
                y = self.experts[e](inp)
                out_flat[idxs] += y * weights.unsqueeze(1)
            return out_flat.view(B,T,D)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, use_moe=False, moe_params=None, dropout=0.1):
        super().__init__()
        self.attn = SimpleSelfAttention(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_moe = use_moe

        if use_moe:
            assert moe_params is not None
            self.moe = MoELayer(**moe_params)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask=mask))
        if self.use_moe:
            x = x + self.dropout(self.moe(self.ln2(x), mask=mask))
        else:
            x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


# Mini LLM + MoE模型
class MiniMoELLModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, n_layers=4, d_ff=1024,
                 use_moe_layer_index=None, moe_params=None):
        """
        use_moe_layer_index: 哪些层使用MoE，例如[1,3]
        moe_params: MoE参数字典，会自动注入 d_model和d_ff
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token+位置编码
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(4096, d_model)

        # Transformer层
        self.layers = nn.ModuleList()
        if use_moe_layer_index is None:
            use_moe_layer_index = set()
        else:
            use_moe_layer_index = set(use_moe_layer_index)

        if moe_params is not None:
            moe_params = moe_params.copy()
            moe_params.setdefault("d_model", d_model)
            moe_params.setdefault("d_ff", d_ff)

        for i in range(n_layers):
            use_moe = (i in use_moe_layer_index)
            self.layers.append(
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    d_ff=d_ff,
                    use_moe=use_moe,
                    moe_params=moe_params
                )
            )

        # 输出层（共享embedding权重）
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.layers:
            x = blk(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# 测试示例 + LLM批量性能评估
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()

    texts = [
        "Hello MoE!",
        "Mixture-of-Experts in LLMs.",
        "MoE是一个很重要的架构思路",
        "你好！😆"
    ]
    token_ids, lengths = tokenizer.batch_encode(texts)
    token_ids = token_ids.to(device)

    moe_params = dict(
        n_experts=4,
        k=1,
        capacity_factor=1.25,
        noisy_gating=True
    )

    model = MiniMoELLModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=4,
        n_layers=4,
        d_ff=1024,
        use_moe_layer_index=[1,3],
        moe_params=moe_params
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(token_ids)  # [B, T, V]
        print("Logits shape:", logits.shape)

        # Top-5示例
        probs_last = F.softmax(logits[0,lengths[0]-1], dim=-1)
        top5 = torch.topk(probs_last, 5)
        print("第一个样本最后位置Top-5 token id:", top5.indices.cpu().tolist())

        # 批量 LLM 性能评估
        total_loss = 0.0
        total_tokens = 0
        top1_acc = 0
        top5_acc = 0

        pad_id = tokenizer.pad

        for b, length in enumerate(lengths):
            # 去掉BOS，并对真实长度裁剪
            input_ids = token_ids[b, :length-1]   # [T-1]
            target_ids = token_ids[b, 1:length]   # 预测下一个 token

            out_logits = model(input_ids.unsqueeze(0))  # [1, T-1, V]
            probs = F.softmax(out_logits, dim=-1)

            # 忽略pad
            mask = (target_ids != pad_id)
            valid_len = mask.sum().item()
            total_tokens += valid_len

            ce_loss = -torch.log(probs[0, torch.arange(length-1), target_ids] + 1e-9)
            ce_loss = ce_loss * mask
            total_loss += ce_loss.sum().item()

            # Top-1、Top-5
            topk_vals, topk_idx = torch.topk(probs[0], 5, dim=-1)
            top1_acc += ((topk_idx[:,0] == target_ids) * mask).sum().item()
            top5_acc += sum([(target_ids[i].item() in topk_idx[i].tolist()) * mask[i].item()
                             for i in range(length-1)])

            # 每条文本的PPL
            ppl_text = math.exp(ce_loss.sum().item() / max(valid_len,1))
            print(f"样本{b} PPL: {ppl_text:.2f}")

        ppl = math.exp(total_loss / total_tokens)
        print(f"整体困惑度(Perplexity, PPL): {ppl:.2f}")
        print(f"整体Top-1准确率: {top1_acc/total_tokens*100:.2f}%")
        print(f"整体Top-5准确率: {top5_acc/total_tokens*100:.2f}%")