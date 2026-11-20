# 知识蒸馏与 On-Policy Distillation（OPD）笔记

> 本文整理 LLM 场景下的知识蒸馏（Knowledge Distillation, KD）与 On-Policy Distillation（OPD），重点说明训练上下文分布、Forward/Reverse KL、策略梯度近似，以及不同 Tokenizer 之间的文本片段对齐。

> **公式格式说明：** 本文使用 VS Code 内置 Markdown Preview 支持的数学定界符：行间公式使用 `$$...$$`，行内公式使用 `$...$`。

## 1. 基本符号

设：

- 教师模型为 $\pi_\phi$，参数 $\phi$ 固定，不参与训练。
- 学生模型为 $\pi_\theta$，参数 $\theta$ 需要更新。
- 输入 Prompt 为 $x$。
- 学生生成的回答为 $y=(y_1,\ldots,y_T)$。
- 第 $t$ 步的上下文为

  $$
  s_t=(x,y_{<t})=(x,y_1,\ldots,y_{t-1}).
  $$

- 教师和学生在上下文 $s_t$ 上的下一个 Token 分布分别为

  $$
  q_\phi(v\mid s_t)=\pi_\phi(v\mid s_t),
  \qquad
  p_\theta(v\mid s_t)=\pi_\theta(v\mid s_t).
  $$

本文用 $p_\theta$ 表示学生分布，用 $q_\phi$ 表示教师分布。

## 2. 普通知识蒸馏（KD）

### 2.1 核心思想

知识蒸馏让学生拟合教师的输出分布。与只学习硬标签不同，教师输出的概率分布包含了不同候选 Token 之间的相对偏好，因此通常称为软标签（soft targets）。

给定一个上下文 $s$，经典的 Token 级 Forward KL 蒸馏目标为：

$$
\mathcal L_{\mathrm{KD}}(s)
=D_{\mathrm{KL}}\left(q_\phi(\cdot\mid s)\Vert p_\theta(\cdot\mid s)\right).
$$

展开：

$$
\mathcal L_{\mathrm{KD}}(s)
=\sum_{v\in\mathcal V}q_\phi(v\mid s)
\log\frac{q_\phi(v\mid s)}{p_\theta(v\mid s)}.
$$

教师被冻结，因此对学生参数优化时，教师熵项是常数：

$$
\mathcal L_{\mathrm{KD}}(s)
\equiv -\sum_{v\in\mathcal V}q_\phi(v\mid s)
\log p_\theta(v\mid s)+C.
$$

所以，Forward KL KD 等价于使用教师概率分布作为软标签的交叉熵训练。

### 2.2 温度系数

实践中可以对 Logits 使用温度 $\tau$：

$$
q_\phi^\tau(v\mid s)
=\operatorname{softmax}\left(\frac{z_\phi(s)}{\tau}\right)_v,
\qquad
p_\theta^\tau(v\mid s)
=\operatorname{softmax}\left(\frac{z_\theta(s)}{\tau}\right)_v.
$$

常见蒸馏目标为：

$$
\mathcal L_{\mathrm{KD}}^\tau
=\tau^2D_{\mathrm{KL}}(q_\phi^\tau\Vert p_\theta^\tau).
$$

其中 $\tau^2$ 是常见的梯度尺度补偿项。温度越高，分布通常越平滑，低概率候选 Token 的相对信息也更明显。

### 2.3 KD 的关键不只是损失，还包括上下文来源

自回归 LLM 的蒸馏通常需要先确定教师和学生在哪些前缀上下文上计算分布。常见来源包括：

- 人工数据或固定语料提供的前缀；
- 教师模型生成的轨迹；
- 学生模型生成的轨迹。

因此，KD 并不必然意味着“教师生成答案、学生模仿答案”。只要教师在给定上下文上提供软分布，便可以构成分布蒸馏。上下文来自谁，是区分 On-Policy 与 Off-Policy 蒸馏的重要维度。

## 3. On-Policy Distillation（OPD）

### 3.1 核心思想

OPD 的关键不是某一种特定 KL 方向，而是**使用学生当前策略自己生成的轨迹作为训练上下文，再让教师在这些上下文上提供监督**。

学生先生成：

$$
y\sim\pi_\theta(\cdot\mid x).
$$

对轨迹中的每个位置，构造学生访问到的上下文：

$$
s_t=(x,y_{<t}).
$$

然后让教师和学生都在同一个文本前缀上预测下一个 Token，并计算选定的蒸馏损失。

一个通用的 OPD 目标可以写为：

$$
\boxed{
\mathcal L_{\mathrm{OPD}}(\theta)
=\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[\sum_{t=1}^{T}w_t\,
D\left(q_\phi(\cdot\mid s_t),p_\theta(\cdot\mid s_t)\right)\right]
}
$$

其中：

- $D$ 可以是 Forward KL、Reverse KL、Jensen–Shannon Divergence（JSD）等；
- $w_t$ 是 Token 权重或有效位置 Mask；
- 外层上下文由学生策略诱导，因此学生可能访问自己生成的错误前缀、偏离教师轨迹的前缀。

### 3.2 为什么学生生成的上下文重要？

例如，学生在数学任务中生成了错误中间步骤：

```text
问题：123 × 456 等于多少？
学生前缀：123 × 456 = 123 × 400 = 49200
```

教师在这个学生实际生成的前缀上继续预测，可以对学生已经偏离正确推理的状态提供监督。相比只在教师正确轨迹上蒸馏，这种训练能直接覆盖学生自身生成时可能遇到的上下文。

这也是 OPD 与 Exposure Bias 相关的原因之一：训练监督不只覆盖理想答案前缀，还覆盖学生策略实际访问的前缀。

### 3.3 KD 与 OPD 的关系

OPD 可以视为一种按学生策略状态分布进行的在线/在策略蒸馏设置，而不是与 KD 完全并列、互斥的损失类别。

| 维度 | 常见离线 KD | OPD |
|---|---|---|
| 监督目标 | 教师概率分布 | 教师概率分布或其变体 |
| 上下文来源 | 固定语料、标注前缀、教师轨迹等 | 学生当前策略生成的轨迹 |
| 是否覆盖学生偏离后的前缀 | 不一定 | 会主动覆盖 |
| KL 方向 | 常见 Forward KL，也可选其他目标 | Forward/Reverse KL、JSD 等均可 |
| 训练状态分布 | 通常固定或预先构造 | 随学生策略更新而变化 |

需要避免一个过度简化的说法：普通 KD 并非绝对不能使用学生生成的数据。如果教师在学生轨迹上提供分布监督，这类方法本身就具有 On-Policy 蒸馏属性。更准确的区别是训练上下文的来源和更新方式。

## 4. Forward KL 与 Reverse KL

KL 的方向决定了蒸馏时由哪个分布作为期望分布。它与 On-Policy / Off-Policy 是两个不同维度。

### 4.1 Forward KL：教师分布到学生分布

定义：

$$
\boxed{
D_{\mathrm{KL}}(q_\phi\Vert p_\theta)
=\sum_{v\in\mathcal V}q_\phi(v\mid s)
\log\frac{q_\phi(v\mid s)}{p_\theta(v\mid s)}
}
$$

去掉与学生参数无关的教师熵项后：

$$
\mathcal L_F(s)
=-\sum_v q_\phi(v\mid s)\log p_\theta(v\mid s)+C.
$$

特点：

- 由教师分布对词表中的候选 Token 加权；
- 教师认为有概率质量的区域会得到监督；
- 通常有 mode-covering 倾向，即学生需要覆盖教师分布中较重要的多个模式；
- 在固定上下文上的损失形式就是经典软标签 KD。

如果这些固定上下文是学生生成的前缀，那么它就是 Forward-KL OPD。它与普通 KD 的局部损失形式相同，区别在于上下文分布。

### 4.2 Reverse KL：学生分布到教师分布

定义：

$$
\boxed{
D_{\mathrm{KL}}(p_\theta\Vert q_\phi)
=\sum_{v\in\mathcal V}p_\theta(v\mid s)
\log\frac{p_\theta(v\mid s)}{q_\phi(v\mid s)}
}
$$

特点：

- 由学生当前分布对候选 Token 加权；
- 更关注学生当前有概率质量、实际可能生成的 Token；
- 通常具有 mode-seeking 倾向，可能偏向教师认可的某些高概率模式，而不充分覆盖教师的所有模式；
- 如果教师对学生可能生成的 Token 给出极低概率，Reverse KL 会对这种概率质量施加较强惩罚。

Forward KL 的 mode-covering、Reverse KL 的 mode-seeking 是常见倾向，不是对所有模型、数据和优化设置都成立的绝对结论。

### 4.3 两种 KL 的区别

| 比较 | Forward KL | Reverse KL |
|---|---|---|
| 形式 | $D_{\mathrm{KL}}(q_\phi\Vert p_\theta)$ | $D_{\mathrm{KL}}(p_\theta\Vert q_\phi)$ |
| 期望分布 | 教师 | 学生 |
| 主要关注 | 教师分布覆盖的候选 | 学生当前会生成的候选 |
| 常见倾向 | Mode-covering | Mode-seeking |
| 固定上下文下是否等价于软标签 CE | 是 | 否 |
| 是否可以用于 OPD | 可以 | 可以 |

**关键结论：Forward KL 看起来就是普通 KD，这个理解是正确的。** 在相同上下文上，Forward KL OPD 的局部损失就是普通的软标签 KD。OPD 的额外特点来自上下文由学生自己的轨迹提供，并且随着学生策略更新而变化。

## 5. OPD 的两种计算方式

需要区分“完整词表分布 KL”与“采样式策略梯度近似”。两者优化目标相关，但计算方式和梯度性质不同。

### 5.1 完整词表 Reverse KL

对于固定上下文 $s_t$，直接计算：

$$
\mathcal L_{R,t}
=\sum_{v\in\mathcal V}p_\theta(v\mid s_t)
\left[\log p_\theta(v\mid s_t)-\log q_\phi(v\mid s_t)\right].
$$

对学生分布直接求梯度，教师分布停止梯度。这种方法需要计算词表维度上的概率，显存和计算开销较大，但没有只采样一个 Token 带来的 Monte Carlo 估计方差。

### 5.2 单样本 Monte Carlo Reverse-KL 策略梯度

Reverse KL 可以改写为：

$$
D_{\mathrm{KL}}(p_\theta\Vert q_\phi)
=\mathbb E_{a\sim p_\theta(\cdot\mid s)}
\left[\log p_\theta(a\mid s)-\log q_\phi(a\mid s)\right].
$$

因此可以从学生分布采样一个动作（在 LLM 中就是学生实际采样的 Token）：

$$
a\sim p_\theta(\cdot\mid s).
$$

记：

$$
r(a,s)=\log q_\phi(a\mid s)-\log p_\theta(a\mid s).
$$

对固定上下文，Reverse KL 的策略梯度可写为：

$$
\nabla_\theta D_{\mathrm{KL}}(p_\theta\Vert q_\phi)
=\mathbb E_{a\sim p_\theta}
\left[(\log p_\theta(a\mid s)-\log q_\phi(a\mid s))
\nabla_\theta\log p_\theta(a\mid s)\right].
$$

于是可以使用如下策略梯度代理损失：

$$
\boxed{
\mathcal L_{\mathrm{PG}}(s)
=-\operatorname{stopgrad}(r(a,s))\,
\log p_\theta(a\mid s)
}
$$

其中 $\operatorname{stopgrad}$ 表示停止 Advantage 的梯度。对该代理损失求梯度，可以得到固定上下文 Reverse KL 梯度的单样本无偏估计（忽略可选的方差缩减基线）。

直观解释：

- 若教师给实际采样 Token 的概率高于学生，即 $r>0$，则提高该 Token 的学生概率；
- 若教师给该 Token 的概率低于学生，即 $r<0$，则降低该 Token 的学生概率。

注意：这里的 $\mathcal L_{\mathrm{PG}}$ 是梯度代理损失，它的数值本身不等于完整 Reverse KL。训练日志中的代理 Loss 不能直接解释为真实 KL 值。

### 5.3 Forward KL 的采样估计不能照搬 Reverse KL 采样方式

Forward KL 满足：

$$
D_{\mathrm{KL}}(q_\phi\Vert p_\theta)
=\mathbb E_{a\sim q_\phi}
\left[\log q_\phi(a\mid s)-\log p_\theta(a\mid s)\right].
$$

因此，若要用单样本 Monte Carlo 估计 Forward KL 的期望，应从教师分布 $q_\phi$ 采样，而不是直接使用学生采样的 Token。若希望在学生轨迹上下文上进行 Forward KL OPD，可以这样做：

1. 学生生成前缀轨迹，确定待监督的上下文 $s_t$；
2. 教师在每个 $s_t$ 上产生完整概率分布，或从教师分布采样候选 Token；
3. 使用完整 Forward KL 或教师采样估计更新学生。

这与 Reverse-KL OPD 中“学生采样 Token + 教师/学生该 Token 的 Log Probability 差”不是同一种估计器。

### 5.4 JSD

Jensen–Shannon Divergence 可以作为介于两种分布监督方式之间的另一种选择。对称 JSD 定义为：

$$
m(v\mid s)=\frac{1}{2}q_\phi(v\mid s)+\frac{1}{2}p_\theta(v\mid s),
$$

$$
\operatorname{JSD}(q_\phi,p_\theta)
=\frac{1}{2}D_{\mathrm{KL}}(q_\phi\Vert m)
+\frac{1}{2}D_{\mathrm{KL}}(p_\theta\Vert m).
$$

JSD 是有界、对称的分布差异度量。广义 JSD 可以改变混合权重，但不要把其权重端点简单理解为 Forward KL 和 Reverse KL：标准加权 JSD 在端点会退化为零，而不是直接变成某一个 KL。

## 6. 一个重要的梯度细节：轨迹采样与固定上下文

OPD 目标包含外层学生轨迹采样：

$$
\tau\sim\pi_\theta.
$$

实际训练中通常先在无梯度模式下生成 Rollout，再把生成的 Token 序列作为固定训练样本，重新 Forward 计算学生 Log Probability 和蒸馏损失。离散 Token 采样本身不通过普通反向传播求导。

因此需要区分：

- **固定上下文上的分布蒸馏梯度**：对采样到的前缀 $s_t$ 计算完整 KL，并对学生当前输出分布求导；
- **固定上下文上的 Reverse-KL 策略梯度估计**：使用学生采样 Token 和 Log-probability 差构造 REINFORCE 代理损失；
- **外层状态分布变化**：学生参数更新后，学生生成的前缀分布也会改变。实践中通常通过周期性重新生成 Rollout 来跟踪当前学生分布。

不能把“在学生轨迹上计算固定上下文 KL”与“对完整轨迹期望目标的所有参数依赖求导”混为一谈。若复用旧策略产生的 Rollout 进行多轮更新，则样本不再严格来自当前策略，需要考虑重要性采样、PPO-style ratio/限制更新幅度，或重新生成 Rollout。

## 7. 同 Tokenizer 下的基础 PyTorch 实现

下面给出 Reverse-KL OPD 的单样本策略梯度核心模板。假设：

- 教师和学生使用完全相同且兼容的 Tokenizer；
- 学生轨迹是在当前学生策略下采样的；
- 生成时使用原始模型分布，例如 `temperature=1.0`、`top_p=1.0`，没有额外改变采样分布的 Logits Processor；
- 只在 Response Token 位置计算损失，Prompt 位置 Mask 为 0。

### 7.1 计算指定 Token 的 Log Probability

```python
import torch
import torch.nn.functional as F


def get_token_logprobs(logits, labels):
    """Return log P(label_t | prefix_t) for each position.

    logits: [B, T, V]
    labels: [B, T]
    return: [B, T]
    """
    vocab_size = logits.size(-1)
    log_probs = -F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    )
    return log_probs.view_as(labels)
```

### 7.2 构造 Response Mask

对于完整序列：

```text
Prompt:    A B C
Response:  D E F
Full:      A B C D E F
```

Causal LM 的 `logits[:, t]` 预测 `input_ids[:, t+1]`，因此第一个 Response Token `D` 对应的 Logit 位置是 `C` 的位置。

```python
def build_batch(prompt_ids_list, response_ids_list, pad_token_id, device):
    sequences = [
        torch.cat([p, r], dim=0)
        for p, r in zip(prompt_ids_list, response_ids_list)
    ]

    batch_size = len(sequences)
    max_len = max(seq.numel() for seq in sequences)

    input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )
    response_mask = torch.zeros(
        (batch_size, max_len - 1),
        dtype=torch.float32,
        device=device,
    )

    for i, (prompt_ids, response_ids, seq) in enumerate(
        zip(prompt_ids_list, response_ids_list, sequences)
    ):
        seq_len = seq.numel()
        prompt_len = prompt_ids.numel()

        input_ids[i, :seq_len] = seq.to(device)
        attention_mask[i, :seq_len] = 1

        # logits position prompt_len - 1 predicts the first response token
        response_mask[i, prompt_len - 1 : seq_len - 1] = 1.0

    return input_ids, attention_mask, response_mask
```

### 7.3 Reverse-KL 单样本策略梯度损失

```python
def compute_reverse_kl_pg_loss(
    student,
    teacher,
    input_ids,
    attention_mask,
    response_mask,
):
    """Single-sample reverse-KL policy-gradient surrogate.

    The input response tokens must have been sampled from the same
    student policy distribution whose log-probabilities are recomputed here.
    """
    student.eval()  # disables dropout but keeps autograd enabled
    teacher.eval()

    labels = input_ids[:, 1:]

    # Student forward: gradients enabled
    student_out = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    student_logits = student_out.logits[:, :-1, :]
    student_logp = get_token_logprobs(student_logits, labels)

    # Teacher forward: frozen, no gradients
    with torch.no_grad():
        teacher_out = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        teacher_logits = teacher_out.logits[:, :-1, :]
        teacher_logp = get_token_logprobs(teacher_logits, labels)

    # r_t = log q_teacher(y_t | s_t) - log p_student(y_t | s_t)
    advantage = (teacher_logp - student_logp.detach())

    # Policy-gradient surrogate; its scalar value is not the KL value itself.
    token_loss = -advantage * student_logp

    denom = response_mask.sum().clamp_min(1.0)
    loss = (token_loss * response_mask).sum() / denom
    return loss
```

### 7.4 训练步骤

```python
optimizer.zero_grad(set_to_none=True)

loss = compute_reverse_kl_pg_loss(
    student=student,
    teacher=teacher,
    input_ids=input_ids,
    attention_mask=attention_mask,
    response_mask=response_mask,
)

loss.backward()
torch.nn.utils.clip_grad_norm_(
    [p for p in student.parameters() if p.requires_grad],
    max_norm=1.0,
)
optimizer.step()
```

### 7.5 生成与训练时需要保持一致的事项

1. 学生生成轨迹时，采样分布必须与重新计算 Log Probability 时使用的分布一致。若生成时使用 Temperature、Top-p、Top-k、重复惩罚或其他 Logits Processor，严格来说就需要在 Log Probability 中使用同样变换后的策略分布。
2. 生成时通常使用 `torch.no_grad()`；训练时重新 Forward 学生模型并保留梯度。
3. 教师使用 `eval()` 和 `torch.no_grad()`，不更新教师参数。
4. Prompt Token 不应计入 Response Loss；Padding、被截断位置和无效 Token 都应 Mask。
5. 上面的代码为了易读仍会产生 `[B,T,V]` 的 Logits 张量。大模型或长上下文训练时，需要结合 Flash/分块交叉熵、梯度检查点、LoRA、序列分块等方式控制显存。
6. 该模板是固定 Rollout 上的单样本 Reverse-KL 策略梯度示例，不包含 PPO clipping、旧策略多轮复用、KL 控制系数或分布式训练。

## 8. 完整词表 KL 的 PyTorch 计算

如果教师和学生词表一致，并且可以承担完整词表计算开销，可以直接在学生 Rollout 的每个有效上下文上计算完整 KL。

```python
def compute_full_vocab_kl(
    student_logits,
    teacher_logits,
    response_mask,
    direction="forward",
):
    """Compute full-vocabulary KL on fixed student-visited prefixes.

    logits: [B, T, V]
    response_mask: [B, T]
    """
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    student_p = student_logp.exp()

    with torch.no_grad():
        teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        teacher_p = teacher_logp.exp()

    if direction == "forward":
        # KL(teacher || student)
        token_kl = (
            teacher_p * (teacher_logp - student_logp)
        ).sum(dim=-1)
    elif direction == "reverse":
        # KL(student || teacher)
        token_kl = (
            student_p * (student_logp - teacher_logp)
        ).sum(dim=-1)
    else:
        raise ValueError("direction must be 'forward' or 'reverse'")

    mask = response_mask.float()
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
```

说明：这个函数计算的是学生轨迹上下文上的完整分布 KL。它不是前一节的单样本策略梯度估计。完整 KL 需要在词表维度上计算分布，显存成本通常更高。

## 9. 教师与学生 Tokenizer 不同：文本片段级对齐

### 9.1 问题

如果教师和学生的 Tokenizer 不同，它们的 Token ID 和词表维度不同，不能直接计算：

$$
D_{\mathrm{KL}}(q_\phi(\cdot\mid s)\Vert p_\theta(\cdot\mid s)).
$$

例如同一个字符串：

```text
文本：international

Teacher Tokenizer： ["inter", "national"]
Student Tokenizer： ["in", "ter", "national"]
```

二者的 Token 数量和 Token ID 都不一致，因此不能逐 Token 对应。

### 9.2 基于共同文本片段的概率对齐

考虑一个共同文本片段 $z$。教师把它切分为 $n$ 个 Token：

$$
z=v_1\oplus v_2\oplus\cdots\oplus v_n.
$$

学生把它切分为 $m$ 个 Token：

$$
z=u_1\oplus u_2\oplus\cdots\oplus u_m.
$$

在各自模型中，使用表示同一文本前缀的上下文计算：

$$
\log P_T(z\mid c_T)
=\sum_{j=1}^{n}\log q_\phi(v_j\mid c_T,v_{<j}),
$$

$$
\log P_S(z\mid c_S)
=\sum_{i=1}^{m}\log p_\theta(u_i\mid c_S,u_{<i}).
$$

其中 $c_T$ 和 $c_S$ 分别是同一个文本前缀在教师、学生 Tokenizer 下的表示。

由此构造片段级概率差：

$$
\boxed{
A_z=\log P_T(z\mid c_T)-\log P_S(z\mid c_S)
}
$$

如果使用学生采样的片段作为动作，则可构造 Reverse-KL 风格的策略梯度代理损失：

$$
\boxed{
\mathcal L_{\mathrm{span}}
=-\operatorname{stopgrad}(A_z)\log P_S(z\mid c_S)
}
$$

并且：

$$
\log P_S(z\mid c_S)
=\sum_{i=1}^{m}\log p_\theta(u_i\mid c_S,u_{<i}).
$$

### 9.3 对齐流程

1. 学生使用自己的 Tokenizer 和当前策略生成回答。
2. 将学生回答还原为文本，并保留准确的字符/字节偏移信息。
3. 教师使用自己的 Tokenizer 对同一文本和前缀重新编码。
4. 使用 Tokenizer 的 `offset_mapping` 获取 Token 与文本区间的对应关系（要求相关 Fast Tokenizer 支持偏移映射）。
5. 寻找教师、学生都能确认的共同文本边界；将覆盖多个 Token 的区间合并成共同片段。
6. 在教师和学生各自的 Token 序列中累加该片段所覆盖 Token 的 Log Probability。
7. 计算片段级概率差，再构造训练目标。

### 9.4 需要注意的数学和工程问题

- **共同边界问题**：一个 Token 可能跨越候选片段边界。不能把该 Token 的 Log Probability 随意切分给两侧片段，应合并相邻片段，直到边界同时落在两边 Token 边界上。
- **上下文一致性**：教师与学生必须条件于同一个可见文本前缀，只是分别通过各自 Tokenizer 编码。不能让教师条件于教师自己的另一条回答。
- **空格和特殊字符**：很多 Tokenizer 的 Token 解码包含前导空格、换行或特殊符号。对齐时要统一文本规范化方式，避免字符偏移错位。
- **Unicode 与字节偏移**：部分 Tokenizer 基于字节或特殊预处理，字符索引与字节索引可能不一致，需要确认 `offset_mapping` 的单位和解码行为。
- **概率含义**：将某个字符串按各 Tokenizer 的规范切分后累加 Log Probability，得到的是该规范 Token 路径的概率。严格的字符串事件概率可能还需要对所有能够解码为同一字符串的 Token 路径求和，通常更难计算。因此片段级方案是一种可操作的路径概率对齐方法，不应不加限定地称为所有可能 Token 化路径上的精确字符串概率。
- **采样分布一致性**：如果学生用经过 Temperature 或 Top-p 变换的分布采样，学生片段概率也应与实际采样策略对应。

## 10. 推荐的对照实验设计

为了分离“学生 On-Policy 上下文”与“KL 方向”的影响，可以设计一个 $2\times2$ 实验：

| 实验 | 上下文/轨迹来源 | 蒸馏方向 |
|---|---|---|
| A | 教师生成轨迹或固定数据轨迹 | Forward KL |
| B | 教师生成轨迹或固定数据轨迹 | Reverse KL |
| C | 学生生成轨迹 | Forward KL |
| D | 学生生成轨迹 | Reverse KL |

其中 D 是典型的 Reverse-KL OPD 设定，C 是 Forward-KL OPD。A、B 是非学生 On-Policy 上下文上的蒸馏对照。为保证实验公平，应尽量控制 Prompt、训练 Token 数、模型初始化、更新步数和计算预算。

建议评估：

- 任务指标：Accuracy、Exact Match、Pass@1 等；
- 生成质量：回答正确性、完整性、格式合规性；
- 分布指标：教师-学生 KL、Token Log-probability Gap、学生 Entropy；
- 行为指标：Response Length、重复率、拒答率；
- 效率指标：生成吞吐、训练吞吐、GPU 显存和总训练 FLOPs。

不要只看蒸馏训练 Loss。尤其对于采样式策略梯度代理损失，其标量值本身不是完整 KL，最好同时在固定评估前缀上计算独立的 Forward/Reverse KL 诊断指标。

## 11. 总结

可以用下面三句话概括全文：

1. **普通 KD**：在指定上下文上，让学生拟合教师的软概率分布；经典 Forward KL 等价于软标签交叉熵。
2. **OPD**：在学生当前策略自己生成的轨迹和前缀上，让教师提供监督。OPD 的关键在于学生诱导的上下文分布，而不是 KL 必须采用某个特定方向。
3. **Forward / Reverse KL**：Forward KL 以教师分布作为期望分布，Reverse KL 以学生分布作为期望分布；两者都可以用于 OPD，但计算方法、梯度估计和行为倾向不同。

一个简洁的概念表达是：

$$
\boxed{
\text{OPD}
=\text{Student On-Policy Contexts}
+\text{Teacher Supervision}
+\text{Distillation Objective}
}
$$



