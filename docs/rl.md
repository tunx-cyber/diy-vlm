# 强化学习中的策略梯度、异步策略梯度与 PPO：完整数学推导

## 目录

1. 强化学习基础与优化目标
2. 策略梯度上升算法（Policy Gradient）
   - 2.1 策略与轨迹概率分布
   - 2.2 目标函数的定义
   - 2.3 策略梯度定理推导
   - 2.4 REINFORCE 算法
   - 2.5 Baseline 与方差降低
   - 2.6 Advantage 函数
   - 2.7 Actor-Critic 算法
3. 异步策略梯度算法（A3C）
   - 3.1 为什么需要异步训练
   - 3.2 多 Worker 并行采样
   - 3.3 异步梯度更新
   - 3.4 A3C 的目标函数
   - 3.5 A3C 的完整训练流程
4. PPO 算法
   - 4.1 从策略梯度到策略更新约束
   - 4.2 Importance Sampling 推导
   - 4.3 TRPO 与 KL 约束
   - 4.4 PPO-Clip 目标函数推导
   - 4.5 GAE 优势估计
   - 4.6 PPO 完整损失函数
   - 4.7 PPO 完整训练流程
5. REINFORCE、A3C 与 PPO 的联系和区别
6. 总结：核心公式速查

---

# 1. 强化学习基础与优化目标

## 1.1 马尔可夫决策过程（MDP）

强化学习通常使用马尔可夫决策过程（Markov Decision Process，MDP）进行建模。

一个 MDP 可以表示为：

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)
$$

其中：

- $\mathcal{S}$：状态空间。
- $\mathcal{A}$：动作空间。
- $P(s'|s,a)$：状态转移概率。
- $r(s,a)$：奖励函数。
- $\gamma\in[0,1)$：折扣因子。

智能体在时刻 $t$ 处于状态 $s_t$，根据策略选择动作：

$$
a_t\sim\pi_\theta(a_t|s_t)
$$

然后环境根据状态转移概率产生下一个状态：

$$
s_{t+1}\sim P(s_{t+1}|s_t,a_t)
$$

环境返回奖励：

$$
r_t=r(s_t,a_t)
$$

因此，一个完整的交互过程可以表示为轨迹（Trajectory）：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots)
$$

这里的 $\pi_\theta$ 是参数为 $\theta$ 的随机策略，策略网络的作用就是学习状态到动作概率分布的映射。

## 1.2 强化学习的优化目标

强化学习的目标不是简单地预测正确动作，而是最大化长期累计奖励。

定义从时刻 $t$ 开始的折扣累计回报：

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}
$$

其中 $T$ 是轨迹的终止时刻。

对于整个轨迹，累计折扣回报为：

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t
$$

我们的优化目标是：

$$
\boxed{
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}
[R(\tau)]
}
$$

其中：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
$$

这里：

- $\rho_0(s_0)$：初始状态分布。
- $\pi_\theta(a_t|s_t)$：策略产生动作的概率。
- $P(s_{t+1}|s_t,a_t)$：环境转移概率。

**核心问题：**

如何调整策略网络参数 $\theta$，使得期望累计回报 $J(\theta)$ 最大？

这就引出了策略梯度算法。

---

# 2. 策略梯度上升算法（Policy Gradient）

## 2.1 为什么使用梯度上升？

我们希望最大化目标函数：

$$
\max_\theta J(\theta)
$$

根据梯度上升（Gradient Ascent），参数更新为：

$$
\boxed{
\theta_{t+1}
=
\theta_t+\alpha\nabla_\theta J(\theta_t)
}
$$

其中 $\alpha$ 是学习率。

注意，这里是梯度上升，而普通监督学习通常使用梯度下降：

$$
\theta_{t+1}
=
\theta_t-\alpha\nabla_\theta L(\theta_t)
$$

两者的区别只是优化目标不同：

- 强化学习策略优化：最大化期望奖励。
- 监督学习：最小化预测误差。

接下来的核心问题是：

如何计算 $\nabla_\theta J(\theta)$？

## 2.2 策略梯度定理的推导

从目标函数开始：

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta(\tau)}
[R(\tau)]
$$

根据期望的定义，可以将其写为积分形式：

$$
J(\theta)
=
\int p_\theta(\tau)R(\tau)d\tau
$$

对参数 $\theta$ 求梯度：

$$
\nabla_\theta J(\theta)
=
\nabla_\theta
\int p_\theta(\tau)R(\tau)d\tau
$$

交换积分与求导：

$$
=
\int
\nabla_\theta p_\theta(\tau)
R(\tau)d\tau
$$

这里使用一个非常重要的数学技巧：**Likelihood Ratio Trick（似然比技巧）**。

根据对数求导公式：

$$
\nabla_\theta\log p_\theta(\tau)
=
\frac{\nabla_\theta p_\theta(\tau)}
{p_\theta(\tau)}
$$

因此：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
$$

代入原式：

$$
\nabla_\theta J(\theta)
=
\int
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
R(\tau)d\tau
$$

重新写成期望形式：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
R(\tau)
\nabla_\theta\log p_\theta(\tau)
\right]
}
$$

这就是策略梯度推导中最关键的一步。

### 进一步展开轨迹概率

轨迹概率为：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
$$

两边取对数：

$$
\begin{aligned}
\log p_\theta(\tau)
=&\log\rho_0(s_0)\\
&+\sum_{t=0}^{T-1}
\log\pi_\theta(a_t|s_t)\\
&+\sum_{t=0}^{T-1}
\log P(s_{t+1}|s_t,a_t)
\end{aligned}
$$

注意：

初始状态分布 $\rho_0$ 和环境转移概率 $P$ 都不依赖策略参数 $\theta$。

所以求导后：

$$
\boxed{
\nabla_\theta\log p_\theta(\tau)
=
\sum_{t=0}^{T-1}
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

代回策略梯度公式：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
R(\tau)
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
}
$$

这就是最基础的策略梯度公式。

它的直观含义是：

- 如果一条轨迹获得了较高奖励，就提高其中动作的发生概率。
- 如果一条轨迹获得了较低奖励，就降低其中动作的发生概率。

也就是说，奖励决定更新方向的权重，而策略的对数概率梯度决定参数更新的方向。

## 2.3 从整条轨迹回报到 Reward-to-Go

刚才得到的公式是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
R(\tau)
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
$$

但是这里存在一个问题：

在时刻 $t$ 执行的动作，不应该受到时刻 $t$ 之前奖励的影响。

例如：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1)
$$

对于动作 $a_1$ 来说，$r_0$ 已经在执行 $a_1$ 之前产生，因此 $a_1$ 不可能影响 $r_0$。

因此可以证明：

$$
\mathbb{E}
\left[
r_k\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]=0,
\quad k<t
$$

推导原因是，在给定状态 $s_t$ 后：

$$
\begin{aligned}
&\mathbb{E}_{a_t\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]\\
=&\sum_{a_t}
\pi_\theta(a_t|s_t)
\nabla_\theta\log\pi_\theta(a_t|s_t)\\
=&\sum_{a_t}
\nabla_\theta\pi_\theta(a_t|s_t)\\
=&\nabla_\theta
\sum_{a_t}\pi_\theta(a_t|s_t)\\
=&\nabla_\theta 1\\
=&0
\end{aligned}
$$

因此，可以将整条轨迹的累计回报替换为从当前时刻开始的累计回报：

$$
G_t=\sum_{k=t}^{T-1}
\gamma^{k-t}r_k
$$

最终得到：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_{t=0}^{T-1}
G_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
}
$$

这就是 REINFORCE 算法的核心公式。

## 2.4 REINFORCE 算法

REINFORCE 是最经典的 Monte Carlo 策略梯度算法。

它的基本思想：

1. 使用当前策略与环境交互。
2. 收集完整轨迹。
3. 计算每个时间步的累计回报 $G_t$。
4. 根据策略梯度公式计算梯度。
5. 使用梯度上升更新策略参数。

单条轨迹的梯度估计为：

$$
\hat{g}
=
\sum_{t=0}^{T-1}
G_t\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

参数更新：

$$
\boxed{
\theta
\leftarrow
\theta+
\alpha
\sum_{t=0}^{T-1}
G_t\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

如果采用损失函数的形式进行梯度下降，可以定义：

$$
L_{\text{PG}}(\theta)
=
-\sum_{t=0}^{T-1}
G_t\log\pi_\theta(a_t|s_t)
$$

然后：

$$
\theta\leftarrow
\theta-\alpha\nabla_\theta L_{\text{PG}}
$$

两种写法本质等价。

### REINFORCE 的问题

REINFORCE 有一个比较明显的问题：

**梯度估计的方差比较大。**

原因在于 $G_t$ 受到后续整个轨迹的影响。

即使两个状态非常相似，它们获得的回报也可能因为随机性而有很大差异。

例如：

- 第一次采样：$G_t=100$。
- 第二次采样：$G_t=-20$。

这会导致策略更新不稳定。

因此，我们希望降低策略梯度的方差。

## 2.5 Baseline 与方差降低

一个非常重要的数学性质：

对于任意只依赖状态、不依赖当前动作的函数 $b(s_t)$，都有：

$$
\boxed{
\mathbb{E}
\left[
b(s_t)
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]=0
}
$$

证明如下：

$$
\begin{aligned}
&\mathbb{E}_{a_t\sim\pi_\theta}
\left[
b(s_t)\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]\\
=&b(s_t)
\sum_{a_t}
\pi_\theta(a_t|s_t)
\nabla_\theta\log\pi_\theta(a_t|s_t)\\
=&b(s_t)
\nabla_\theta
\sum_{a_t}\pi_\theta(a_t|s_t)\\
=&b(s_t)\nabla_\theta 1\\
=&0
\end{aligned}
$$

因此，我们可以在原来的梯度公式中减去一个 baseline：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
(G_t-b(s_t))
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
}
$$

为什么这样可以降低方差？

考虑：

$$
G_t-b(s_t)
$$

它不再表示绝对回报，而是表示当前回报相对于某个参考水平的差异。

例如：

| 状态 | 动作 | 实际回报 | Baseline | 差值 |
|---|---|---:|---:|---:|
| $s_1$ | $a_1$ | 100 | 80 | +20 |
| $s_1$ | $a_2$ | 60 | 80 | -20 |
| $s_2$ | $a_1$ | 50 | 40 | +10 |
| $s_2$ | $a_2$ | 30 | 40 | -10 |

这样就可以减少不同状态本身回报水平不同造成的影响。

需要强调：

Baseline 的作用是改变梯度估计的方差，而不改变期望梯度。

### 最优 Baseline

对于单个状态，如果考虑梯度估计：

$$
g=(G_t-b(s_t))z_t
$$

其中：

$$
z_t=\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

使梯度估计二阶矩最小的 baseline 为：

$$
b^*(s_t)
=
\frac{
\mathbb{E}[G_t\|z_t\|^2|s_t]
}{
\mathbb{E}[\|z_t\|^2|s_t]
}
$$

在实际强化学习中，通常使用状态价值函数 $V^\pi(s_t)$ 作为 baseline。

## 2.6 Advantage 函数

首先定义状态价值函数：

$$
\boxed{
V^\pi(s)
=
\mathbb{E}_\pi[G_t|s_t=s]
}
$$

它表示：

从状态 $s$ 出发，按照当前策略继续执行，能够获得的期望累计回报。

然后定义动作价值函数：

$$
\boxed{
Q^\pi(s,a)
=
\mathbb{E}_\pi[G_t|s_t=s,a_t=a]
}
$$

它表示：

在状态 $s$ 下先执行动作 $a$，然后继续按照当前策略执行，能够获得的期望累计回报。

两者之间的关系：

$$
V^\pi(s)
=
\mathbb{E}_{a\sim\pi}
[Q^\pi(s,a)]
$$

因此，定义优势函数（Advantage Function）：

$$
\boxed{
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
}
$$

其含义是：

**在当前状态下，执行某个动作，相比于当前策略的平均表现，好多少或者差多少。**

当：

$$
A^\pi(s,a)>0
$$

说明当前动作比平均水平好，应该提高其概率。

当：

$$
A^\pi(s,a)<0
$$

说明当前动作比平均水平差，应该降低其概率。

当：

$$
A^\pi(s,a)=0
$$

说明当前动作的表现与当前策略的平均水平一致。

因此，策略梯度可以写为：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
A^\pi(s_t,a_t)
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
}
$$

这里的 Advantage 实际上就是一种特殊的 Baseline 处理。

### 一个直观例子

假设：

$$
Q(s,a_1)=100
$$

$$
Q(s,a_2)=60
$$

$$
V(s)=80
$$

那么：

$$
A(s,a_1)=100-80=20
$$

$$
A(s,a_2)=60-80=-20
$$

于是：

- $a_1$ 的概率应该提高。
- $a_2$ 的概率应该降低。

注意，Advantage 并不直接决定更新幅度的全部因素，最终梯度还与策略概率的对数梯度有关。

## 2.7 Actor-Critic 算法

REINFORCE 使用完整轨迹计算 $G_t$，这属于 Monte Carlo 方法。

但是完整轨迹可能很长，而且必须等待轨迹结束才能获得准确的 Monte Carlo 回报。

因此，可以训练一个价值网络来估计：

$$
V_\phi(s_t)
$$

其中：

- Actor：策略网络，负责选择动作。
- Critic：价值网络，负责评估状态。

Critic 的目标是逼近真实状态价值：

$$
V_\phi(s)\approx V^\pi(s)
$$

### 1. TD Target

根据 Bellman 方程：

$$
V^\pi(s_t)
=
\mathbb{E}
\left[
r_t+\gamma V^\pi(s_{t+1})
\right]
$$

因此可以使用下一状态的价值估计构造 TD Target：

$$
\boxed{
y_t=r_t+\gamma V_\phi(s_{t+1})
}
$$

对于终止状态，需要令后续价值为零。

### 2. TD Error

定义时序差分误差：

$$
\boxed{
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})
-
V_\phi(s_t)
}
$$

可以将它理解为：

实际得到的即时奖励与下一状态价值，加起来比原来预期的状态价值多了多少。

在适当条件下：

$$
\mathbb{E}[\delta_t|s_t,a_t]
\approx A^\pi(s_t,a_t)
$$

因此，TD Error 可以作为 Advantage 的近似估计。

### 3. Actor 的损失函数

使用 TD Error 作为优势估计：

$$
\boxed{
L_{\text{actor}}
=
-\log\pi_\theta(a_t|s_t)
\operatorname{stopgrad}(\delta_t)
}
$$

这里的 `stopgrad` 表示不通过 Actor 损失反向传播到 Critic 的 TD Error 计算路径。

### 4. Critic 的损失函数

使用均方误差：

$$
\boxed{
L_{\text{critic}}
=
\frac{1}{2}
\left(
r_t+\gamma V_\phi(s_{t+1})
-
V_\phi(s_t)
\right)^2
}
$$

实际训练时，通常将目标值从计算图中分离，避免通过 TD Target 产生不需要的梯度路径。

### 5. Actor-Critic 的整体优化

Actor：

$$
\theta
\leftarrow
\theta-\alpha_\theta
\nabla_\theta L_{\text{actor}}
$$

Critic：

$$
\phi
\leftarrow
\phi-\alpha_\phi
\nabla_\phi L_{\text{critic}}
$$

这样就不再需要等待完整轨迹结束，可以使用 TD 方式逐步更新。

---

# 3. 异步策略梯度算法（A3C）

## 3.1 为什么需要异步训练？

普通的策略梯度和 Actor-Critic 通常采用以下流程：

1. 当前策略与环境交互。
2. 收集一批经验。
3. 计算梯度。
4. 更新策略。
5. 使用更新后的策略继续采样。

这种方式存在两个问题：

**问题一：采样效率。**

如果只有一个 Worker 与环境交互，那么环境采样速度可能成为训练瓶颈。

**问题二：样本相关性。**

连续时间步产生的状态通常具有较强的相关性：

$$
s_t\rightarrow s_{t+1}\rightarrow s_{t+2}
$$

如果一直从同一个环境中连续采样，训练数据可能缺乏多样性。

A3C（Asynchronous Advantage Actor-Critic）的核心思想是：

**使用多个独立 Worker 并行与环境交互，每个 Worker 独立计算梯度，并异步更新全局网络参数。**

## 3.2 A3C 的整体架构

A3C 包含：

- 一个 Global Actor-Critic 网络。
- 多个独立的 Worker。
- 每个 Worker 拥有自己的环境副本。
- 每个 Worker 拥有局部网络参数。

假设有 $N$ 个 Worker：

$$
W_1,W_2,\ldots,W_N
$$

全局参数：

$$
\theta_{\text{global}},\phi_{\text{global}}
$$

第 $i$ 个 Worker 的局部参数：

$$
\theta_i,\phi_i
$$

Worker 从全局网络同步参数：

$$
\theta_i\leftarrow\theta_{\text{global}}
$$

$$
\phi_i\leftarrow\phi_{\text{global}}
$$

然后使用自己的环境进行采样。

## 3.3 异步梯度更新

假设 Worker $i$ 收集了一段长度为 $n$ 的轨迹：

$$
(s_t,a_t,r_t,\ldots,s_{t+n})
$$

首先计算 n-step Return：

$$
\boxed{
R_t^{(n)}
=
\sum_{k=0}^{n-1}
\gamma^k r_{t+k}
+
\gamma^n V_{\phi_i}(s_{t+n})
}
$$

如果轨迹在 $n$ 步内已经终止，那么：

$$
R_t^{(n)}
=
\sum_{k=0}^{n-1}
\gamma^k r_{t+k}
$$

然后计算优势估计：

$$
\boxed{
A_t^{(n)}
=
R_t^{(n)}-V_{\phi_i}(s_t)
}
$$

Worker $i$ 的 Actor 损失：

$$
L_{\text{actor}}^{(i)}
=
-\sum_{t}
\log\pi_{\theta_i}(a_t|s_t)
A_t^{(n)}
$$

Critic 损失：

$$
L_{\text{critic}}^{(i)}
=
\frac{1}{2}
\sum_t
\left(
R_t^{(n)}-V_{\phi_i}(s_t)
\right)^2
$$

每个 Worker 分别计算梯度：

$$
g_{\theta_i}
=
\nabla_{\theta_i}L_{\text{actor}}^{(i)}
$$

$$
g_{\phi_i}
=
\nabla_{\phi_i}L_{\text{critic}}^{(i)}
$$

接下来，每个 Worker 将梯度应用到全局参数：

$$
\boxed{
\theta_{\text{global}}
\leftarrow
\theta_{\text{global}}
-
\alpha g_{\theta_i}
}
$$

$$
\boxed{
\phi_{\text{global}}
\leftarrow
\phi_{\text{global}}
-
\beta g_{\phi_i}
}
$$

注意：

这里使用的是梯度下降形式的损失函数，因此参数更新使用减号。

如果直接使用策略梯度上升形式，则对应加号。

### 为什么称为异步？

假设有三个 Worker：

- Worker 1 在采样。
- Worker 2 在计算梯度。
- Worker 3 正在更新全局参数。

它们不需要等待其他 Worker 完成当前任务。

因此，多个 Worker 可以独立运行。

这就是 A3C 中 Asynchronous 的含义。

### 一个需要注意的问题：参数滞后

由于各 Worker 的更新并不同步，可能出现：

1. Worker 1 从全局网络读取参数 $\theta_1$。
2. Worker 1 开始采样。
3. Worker 2 更新全局参数。
4. Worker 1 使用旧参数计算梯度。
5. Worker 1 将旧参数产生的梯度应用到当前全局网络。

因此：

$$
\theta_i\neq\theta_{\text{global}}
$$

这意味着 A3C 中存在一定程度的梯度滞后（Stale Gradient）。

在实际实现中，需要合理控制每个 Worker 的采样长度、更新频率以及参数同步频率。

## 3.4 A3C 的完整目标函数

除了 Actor 和 Critic，A3C 通常还会加入熵正则项（Entropy Regularization）。

策略熵定义为：

$$
H(\pi_\theta(\cdot|s))
=
-\sum_a
\pi_\theta(a|s)
\log\pi_\theta(a|s)
$$

熵越大，表示策略的动作分布越分散。

加入熵奖励可以鼓励探索，避免策略过早收敛到某一个动作。

A3C 的总损失可以写成：

$$
\boxed{
L_{\text{A3C}}
=
L_{\text{actor}}
+
c_vL_{\text{critic}}
-
c_eH(\pi_\theta)
}
$$

其中：

- $c_v$：价值损失权重。
- $c_e$：熵正则项权重。

注意，严格来说，Actor、Critic 和熵项通常分别按各自的计算图进行优化。

## 3.5 A3C 的训练流程

**Algorithm: A3C**

输入：

- Worker 数量 $N$。
- 折扣因子 $\gamma$。
- n-step 长度 $n$。
- Actor 学习率 $\alpha$。
- Critic 学习率 $\beta$。

流程：

1. 初始化全局 Actor-Critic 网络。
2. 创建 $N$ 个 Worker。
3. 每个 Worker 创建独立环境。
4. Worker 从全局网络同步参数。
5. Worker 使用局部策略进行 n 步采样。
6. 计算 n-step Return。
7. 计算 Advantage。
8. 计算 Actor 和 Critic 的梯度。
9. 将梯度异步应用到全局网络。
10. Worker 重新同步参数，继续采样。
11. 直到达到训练终止条件。

### A3C 与 A2C 的区别

A3C 和 A2C 都使用多个 Worker。

但两者更新方式不同：

| 对比项 | A3C | A2C |
|---|---|---|
| 全称 | Asynchronous Advantage Actor-Critic | Advantage Actor-Critic |
| 采样方式 | 多 Worker 并行 | 多 Worker 并行 |
| 更新方式 | 异步更新 | 同步更新 |
| 参数同步 | 不要求统一等待 | 等待所有 Worker |
| 梯度滞后 | 可能存在 | 通常不存在 |
| 实现特点 | 异步并发 | Batch 化更新 |

A2C 可以看作 A3C 的同步版本。

---

# 4. PPO 算法（Proximal Policy Optimization）

PPO 是一种非常重要的策略优化算法。

它仍然属于 Policy Gradient / Actor-Critic 这一类方法，但是它解决了一个关键问题：

**如何避免策略在一次更新中变化得过大？**

## 4.1 从策略梯度到策略更新约束

首先回顾普通策略梯度：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
A^\pi(s_t,a_t)
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
$$

普通策略梯度直接沿着梯度方向更新：

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
+
\alpha\nabla_\theta J(\theta)
$$

但是，策略网络通常是非线性的。

一次较大的参数更新可能导致：

$$
\pi_{\theta_{\text{new}}}
\gg
\pi_{\theta_{\text{old}}}
$$

即新旧策略之间的差异过大。

这样可能出现：

- 已经学习到的策略突然退化。
- 训练过程不稳定。
- 梯度估计与实际新策略不匹配。
- 旧数据对新策略的参考价值下降。

因此，我们希望：

**每次策略更新只允许发生适度的变化。**

TRPO 和 PPO 都是为了解决这个问题而提出的。

## 4.2 Importance Sampling 推导

这是 PPO 中最重要的数学基础之一。

假设我们使用旧策略采样：

$$
a_t\sim\pi_{\theta_{\text{old}}}(a_t|s_t)
$$

但是我们希望优化新策略：

$$
\pi_\theta(a_t|s_t)
$$

问题在于：

我们手里只有旧策略采样的数据，如何计算新策略的目标函数？

这时可以使用重要性采样（Importance Sampling）。

### 1. 基本原理

对于两个概率分布 $p(x)$ 和 $q(x)$：

$$
\mathbb{E}_{x\sim p}[f(x)]
=
\int p(x)f(x)dx
$$

乘除 $q(x)$：

$$
=
\int q(x)
\frac{p(x)}{q(x)}
f(x)dx
$$

因此：

$$
\boxed{
\mathbb{E}_{x\sim p}[f(x)]
=
\mathbb{E}_{x\sim q}
\left[
\frac{p(x)}{q(x)}f(x)
\right]
}
$$

这就是重要性采样。

### 2. 应用到强化学习

令：

$$
p(x)=\pi_\theta(a_t|s_t)
$$

$$
q(x)=\pi_{\theta_{\text{old}}}(a_t|s_t)
$$

那么：

$$
\mathbb{E}_{a_t\sim\pi_\theta}
[f(a_t)]
=
\mathbb{E}_{a_t\sim\pi_{\theta_{\text{old}}}}
\left[
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
f(a_t)
\right]
$$

定义重要性采样比率：

$$
\boxed{
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
}
$$

它表示：

新策略选择当前动作的概率，是旧策略选择当前动作概率的多少倍。

当：

$$
r_t(\theta)>1
$$

说明新策略比旧策略更倾向于选择这个动作。

当：

$$
r_t(\theta)<1
$$

说明新策略比旧策略更不倾向于选择这个动作。

当：

$$
r_t(\theta)=1
$$

说明新旧策略对当前动作的概率一致。

### 3. 推导 PPO 的基本 Surrogate Objective

原始策略梯度可以写为：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\pi_\theta}
\left[
A^\pi(s_t,a_t)
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
$$

利用重要性采样，将期望转换为旧策略分布下的期望：

$$
J_{\text{sur}}(\theta)
=
\mathbb{E}_{(s_t,a_t)\sim\pi_{\theta_{\text{old}}}}
\left[
r_t(\theta)A_t
\right]
$$

这里的 $A_t$ 是基于旧策略采样数据计算得到的优势估计，通常在一次 PPO 更新期间将它视为固定量。

于是得到：

$$
\boxed{
L^{\text{PG}}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta)A_t
\right]
}
$$

注意：

这里的期望是从旧策略采样的数据上计算的，而不是重新使用新策略采样。

### 4. 为什么使用概率比率？

对 $r_t(\theta)$ 求导：

$$
\nabla_\theta r_t(\theta)
=
\frac{
\nabla_\theta\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

进一步：

$$
=
r_t(\theta)
\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

因此：

$$
\nabla_\theta
[r_t(\theta)A_t]
=
r_t(\theta)A_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

当 $\theta=\theta_{\text{old}}$ 时：

$$
r_t(\theta)=1
$$

于是：

$$
\boxed{
\nabla_\theta L^{\text{PG}}
\big|_{\theta=\theta_{\text{old}}}
=
\mathbb{E}
\left[
A_t\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
}
$$

这说明 Importance Sampling 得到的 Surrogate Objective 在旧策略处具有正确的策略梯度。

但是，如果我们连续对这个目标进行多次优化，新策略可能与旧策略相差越来越大。

这就引出了 TRPO 和 PPO。

## 4.3 TRPO 与 KL 约束

TRPO（Trust Region Policy Optimization）的核心思想是：

在提高策略期望回报的同时，限制新旧策略之间的差异。

它使用一个带约束的优化目标：

$$
\boxed{
\begin{aligned}
\max_\theta\quad&
\mathbb{E}_t
\left[
r_t(\theta)A_t
\right]\\
\text{subject to}\quad&
\mathbb{E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}
\|
\pi_\theta
\right)
\right]
\leq\delta
\end{aligned}
}
$$

其中：

$$
D_{\mathrm{KL}}(p\|q)
=
\sum_a
p(a)\log\frac{p(a)}{q(a)}
$$

KL 散度衡量两个概率分布之间的差异。

这里约束：

$$
D_{\mathrm{KL}}
(\pi_{\theta_{\text{old}}}\|\pi_\theta)
\leq\delta
$$

意味着：

新策略不能偏离旧策略太远。

TRPO 通过信赖域约束控制策略更新幅度。

但 TRPO 的实现涉及二阶优化、Fisher 信息矩阵以及共轭梯度等操作，实现相对复杂。

PPO 希望用更加简单的方式达到类似目的。

## 4.4 PPO-Clip 目标函数推导

PPO 的核心创新是 Clipping。

先回顾 Surrogate Objective：

$$
L^{\text{PG}}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta)A_t
\right]
$$

假设某个动作的优势为正：

$$
A_t>0
$$

这说明该动作比当前策略的平均表现更好。

我们希望提高这个动作的概率：

$$
r_t(\theta)>1
$$

但是，如果概率提高得太多，就可能导致策略更新过大。

例如：

$$
r_t(\theta)=5
$$

意味着新策略选择这个动作的概率是旧策略的五倍。

这可能造成不稳定。

因此，我们希望限制概率比率的变化范围。

定义：

$$
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)
$$

其含义是：

$$
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)
=
\begin{cases}
1-\epsilon,&r_t<1-\epsilon\\
r_t,&1-\epsilon\leq r_t\leq1+\epsilon\\
1+\epsilon,&r_t>1+\epsilon
\end{cases}
$$

然后构造 PPO 的目标函数：

$$
\boxed{
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
}
$$

为什么要使用 `min`？

需要分两种情况讨论。

### 情况一：优势为正

$$
A_t>0
$$

例如：

$$
A_t=2,\quad \epsilon=0.2
$$

那么：

$$
1-\epsilon=0.8
$$

$$
1+\epsilon=1.2
$$

假设：

$$
r_t=1.5
$$

原始目标：

$$
r_tA_t=1.5\times2=3
$$

Clipped 目标：

$$
1.2\times2=2.4
$$

取两者最小值：

$$
\min(3,2.4)=2.4
$$

也就是说：

虽然继续增加动作概率可以提高原始目标，但是 PPO 不再给予超出裁剪范围的额外收益。

对于正优势动作：

$$
\boxed{
L_t^{\text{CLIP}}
=
\min(r_t,1+\epsilon)A_t
}
$$

这里省略了下界裁剪，因为当 $A_t>0$ 时，下界裁剪不会改变 `min` 的最终结果。

### 情况二：优势为负

$$
A_t<0
$$

例如：

$$
A_t=-2,\quad r_t=0.5
$$

原始目标：

$$
r_tA_t=0.5\times(-2)=-1
$$

Clipped 目标：

$$
0.8\times(-2)=-1.6
$$

取两者最小值：

$$
\min(-1,-1.6)=-1.6
$$

为什么这样设计？

因为负优势说明当前动作不理想，我们希望降低它的概率。

但是，如果降低得太多，也会造成策略更新过大。

PPO 对这种情况同样进行限制。

对于负优势动作：

$$
\boxed{
L_t^{\text{CLIP}}
=
\max(r_t,1-\epsilon)A_t
}
$$

注意，这里使用的是 `max`，而不是 `min`，因为 $A_t$ 是负数。

### PPO Clipping 的本质

可以总结为：

| Advantage | 更新方向 | 限制目的 |
|---|---|---|
| $A_t>0$ | 提高动作概率 | 不允许提高得过多 |
| $A_t<0$ | 降低动作概率 | 不允许降低得过多 |

PPO 的核心是：

**当策略更新已经足够大时，不再允许目标函数因为继续扩大策略变化而获得额外收益。**

需要特别注意：

PPO Clip 并不是严格的 KL 约束，也不能保证所有状态下的新旧策略概率比率都在裁剪区间内。

它是一种简单有效的保守策略更新机制。

## 4.5 GAE 优势估计

PPO 需要一个重要输入：

$$
A_t
$$

那么 Advantage 如何计算？

前面介绍过：

$$
A_t=Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

但是实际环境中，真实的 $Q^\pi$ 和 $V^\pi$ 通常都不知道。

因此需要估计 Advantage。

### 1. TD Error

首先定义：

$$
\boxed{
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
}
$$

TD Error 可以近似 Advantage：

$$
A_t\approx\delta_t
$$

这是最简单的 TD(0) 优势估计。

但它只考虑了一步未来奖励，存在较大的估计偏差。

### 2. n-step Advantage

考虑未来 n 步：

$$
A_t^{(n)}
=
\sum_{k=0}^{n-1}
\gamma^k r_{t+k}
+
\gamma^nV(s_{t+n})
-
V(s_t)
$$

可以将其展开为：

$$
\boxed{
A_t^{(n)}
=
\sum_{l=0}^{n-1}
\gamma^l\delta_{t+l}
}
$$

验证：

$$
\begin{aligned}
A_t^{(n)}
=&\delta_t+\gamma\delta_{t+1}
+\cdots+\gamma^{n-1}\delta_{t+n-1}\\
=&r_t+\gamma V(s_{t+1})-V(s_t)\\
&+\gamma r_{t+1}
+\gamma^2V(s_{t+2})
-\gamma V(s_{t+1})\\
&+\cdots\\
=&\sum_{l=0}^{n-1}\gamma^l r_{t+l}
+\gamma^nV(s_{t+n})-V(s_t)
\end{aligned}
$$

中间的价值函数项发生了抵消。

### 3. GAE 的核心思想

GAE（Generalized Advantage Estimation）的思想是：

将不同步数的 Advantage 估计进行加权组合。

定义：

$$
\boxed{
A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}
}
$$

其中：

$$
\lambda\in[0,1]
$$

$\lambda$ 控制不同步数 TD Error 的权重。

当：

$$
\lambda=0
$$

得到：

$$
A_t^{\mathrm{GAE}(\gamma,0)}
=
\delta_t
$$

这就是 TD(0)。

当：

$$
\lambda=1
$$

得到：

$$
A_t^{\mathrm{GAE}(\gamma,1)}
=
\sum_{l=0}^{\infty}
\gamma^l\delta_{t+l}
$$

对于完整轨迹，这等价于 Monte Carlo Return 减去状态价值：

$$
A_t=G_t-V(s_t)
$$

因此：

- $\lambda=0$：偏向 TD 估计，方差较低，但依赖 Critic 的准确性。
- $\lambda=1$：偏向 Monte Carlo 估计，偏差较低，但方差可能较大。
- $0<\lambda<1$：在偏差和方差之间进行折中。

实际 PPO 通常使用：

$$
\lambda=0.95
$$

### 4. GAE 的递归计算

直接计算无限求和不现实。

实际中可以使用递归形式：

$$
\boxed{
A_t
=
\delta_t+\gamma\lambda A_{t+1}
}
$$

从最后一个时间步开始反向计算：

$$
A_T=0
$$

然后依次计算：

$$
A_{T-1},A_{T-2},\ldots,A_0
$$

对于终止状态，需要将后续价值 bootstrap 关闭。

实际实现中需要区分：

- `terminated`：环境真正终止，不再 bootstrap。
- `truncated`：时间限制等原因导致截断，通常仍然可以 bootstrap。

因此，GAE 实现时应使用正确的终止掩码。

### 5. Value Target

得到 GAE Advantage 后，可以构造 Critic 的训练目标：

$$
\boxed{
V_t^{\text{target}}
=
A_t+V(s_t)
}
$$

这个值就是 PPO 中 Critic 的回归目标。

---

## 4.6 PPO 的完整损失函数

到目前为止，已经得到了：

1. Actor 的 PPO Clip 目标。
2. Critic 的 Value Loss。
3. Entropy Bonus。

将它们组合起来。

### 1. Actor Loss

$$
\boxed{
L_{\text{actor}}
=
-\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
}
$$

注意：

这里使用负号，是因为实际优化通常使用梯度下降。

### 2. Critic Loss

$$
\boxed{
L_{\text{critic}}
=
\frac{1}{2}
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-V_t^{\text{target}}
\right)^2
\right]
}
$$

### 3. Entropy Bonus

$$
\boxed{
L_{\text{entropy}}
=
-\mathbb{E}_t
\left[
H(\pi_\theta(\cdot|s_t))
\right]
}
$$

### 4. PPO 总损失

最终：

$$
\boxed{
\begin{aligned}
L_{\text{PPO}}(\theta,\phi)
=&L_{\text{actor}}\\
&+c_vL_{\text{critic}}\\
&+c_eL_{\text{entropy}}
\end{aligned}
}
$$

其中：

- $c_v$：价值损失系数。
- $c_e$：熵损失系数。

注意：由于这里将熵损失定义成负熵，所以总损失中是加号。

也可以把它写成：

$$
\boxed{
\begin{aligned}
L_{\text{PPO}}
=&-\mathbb{E}_t
\left[
\min
\left(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]\\
&+\frac{c_v}{2}
\mathbb{E}_t
\left[
(V_\phi(s_t)-V_t^{\text{target}})^2
\right]\\
&-c_e\mathbb{E}_t
\left[
H(\pi_\theta(\cdot|s_t))
\right]
\end{aligned}
}
$$

这就是 PPO 最常见的训练目标形式。

## 4.7 PPO 完整训练流程

PPO 通常采用以下流程：

### 第一步：初始化网络

初始化：

- Actor：$\pi_\theta(a|s)$。
- Critic：$V_\phi(s)$。

### 第二步：使用旧策略采样

保存当前策略参数：

$$
\theta_{\text{old}}\leftarrow\theta
$$

使用旧策略与环境交互：

$$
a_t\sim\pi_{\theta_{\text{old}}}(a_t|s_t)
$$

收集：

$$
\mathcal{D}
=
\{s_t,a_t,r_t,
\log\pi_{\theta_{\text{old}}}(a_t|s_t)\}
$$

还需要保存终止标记、价值预测等训练所需信息。

### 第三步：计算 Advantage

使用 Critic 估计：

$$
V_\phi(s_t)
$$

计算 TD Error：

$$
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

使用 GAE：

$$
A_t
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l\delta_{t+l}
$$

计算 Value Target：

$$
V_t^{\text{target}}
=
A_t+V_\phi(s_t)
$$

### 第四步：计算概率比率

使用当前策略重新计算采样动作的概率：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

注意：

旧策略的 log probability 应当在采样时保存，并且在当前 PPO 更新期间保持固定。

### 第五步：计算 PPO Clip Loss

$$
L_{\text{actor}}
=
-\mathbb{E}_t
\left[
\min
\left(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

### 第六步：计算 Critic Loss

$$
L_{\text{critic}}
=
\frac{1}{2}
\mathbb{E}_t
\left[
(V_\phi(s_t)-V_t^{\text{target}})^2
\right]
$$

### 第七步：更新网络

构造总损失：

$$
L=L_{\text{actor}}+c_vL_{\text{critic}}-c_eH
$$

通过梯度下降更新：

$$
\theta\leftarrow\theta-\alpha_\theta\nabla_\theta L
$$

$$
\phi\leftarrow\phi-\alpha_\phi\nabla_\phi L
$$

### 第八步：重复优化多个 Epoch

PPO 与 REINFORCE 的一个重要区别：

PPO 可以使用同一批采样数据进行多个 Epoch 的优化。

例如：

- 收集 2048 条时间步经验。
- 将经验划分为多个 Mini-batch。
- 使用同一批经验训练 10 个 Epoch。
- 完成后重新使用当前策略采样。

这里需要注意：

每次 PPO 更新都使用固定的旧策略概率作为分母，但分子会随着当前策略参数更新而变化。

因此：

$$
\pi_{\theta_{\text{old}}}
\quad\text{固定}
$$

$$
\pi_\theta
\quad\text{持续更新}
$$

经过多个 Epoch 后，旧数据与新策略之间可能产生更大的分布差异。因此实际训练中通常还会使用 KL 散度监控，并在 KL 过大时提前停止当前更新。

---

# 5. REINFORCE、A3C 与 PPO 的联系和区别

## 5.1 算法之间的演进关系

可以将这些算法的关系理解为：

**REINFORCE：**

通过完整轨迹的累计回报估计策略梯度。

$$
\nabla_\theta J
=
\mathbb{E}
[G_t\nabla_\theta\log\pi_\theta]
$$

**Actor-Critic：**

使用 Critic 估计状态价值，通过 Advantage 降低梯度估计方差。

$$
\nabla_\theta J
=
\mathbb{E}
[A_t\nabla_\theta\log\pi_\theta]
$$

**A3C：**

在 Actor-Critic 的基础上，引入多个 Worker 异步采样和更新。

$$
\theta_{\text{global}}
\leftarrow
\theta_{\text{global}}-\alpha g_i
$$

**PPO：**

在 Actor-Critic 的基础上，引入 Importance Sampling 和 Clipping，限制策略更新幅度。

$$
L^{\text{CLIP}}
=
\mathbb{E}
\left[
\min(r_tA_t,\operatorname{clip}(r_t)A_t)
\right]
$$

## 5.2 核心区别

| 算法 | 核心思想 | Advantage | 更新方式 |
|---|---|---|---|
| REINFORCE | Monte Carlo 策略梯度 | Return 或 Return-Baseline | 普通梯度上升 |
| Actor-Critic | Actor + Critic | TD Error 等 | 策略梯度 |
| A3C | 多 Worker 异步 Actor-Critic | n-step Advantage | 异步梯度更新 |
| A2C | 多 Worker 同步 Actor-Critic | n-step Advantage | 同步梯度更新 |
| PPO | 限制策略更新幅度 | 通常 GAE | Clipped Surrogate Objective |

## 5.3 PPO 与 A3C 的区别

需要特别注意：

PPO 和 A3C 并不是简单的前后代替关系。

A3C 主要解决的是：

**如何利用多个 Worker 提高并行采样效率。**

PPO 主要解决的是：

**如何让策略更新更加稳定，避免单次更新幅度过大。**

它们关注的问题不同。

理论上，可以将并行采样和 PPO 策略优化结合起来，但要正确处理数据采样、策略版本和更新同步。

标准 PPO 通常采用同步的 On-policy 数据采集和优化流程。

---

# 6. 总结：核心公式速查

## 6.1 策略梯度

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
R(\tau)
\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
}
$$

## 6.2 REINFORCE

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
G_t\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
}
$$

## 6.3 Baseline

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
(G_t-b(s_t))
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
}
$$

## 6.4 Advantage

$$
\boxed{
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
}
$$

## 6.5 TD Error

$$
\boxed{
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
}
$$

## 6.6 A3C n-step Return

$$
\boxed{
R_t^{(n)}
=
\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+
\gamma^nV(s_{t+n})
}
$$

## 6.7 Importance Sampling Ratio

$$
\boxed{
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
}
$$

## 6.8 PPO Clip Objective

$$
\boxed{
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
}
$$

## 6.9 GAE

$$
\boxed{
A_t^{\mathrm{GAE}}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}
}
$$

## 6.10 PPO 总损失

$$
\boxed{
L_{\text{PPO}}
=
L_{\text{actor}}
+
c_vL_{\text{critic}}
-
c_eH(\pi_\theta)
}
$$

---

# 7. 最后需要理解的几个关键问题

### 问题一：为什么策略梯度需要对数概率？

因为使用似然比技巧：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
$$

它让我们不需要直接对整个轨迹的概率分布求导，就能够得到策略梯度的无偏估计。

### 问题二：为什么 Advantage 比 Return 更适合更新策略？

Return 衡量绝对回报。

Advantage 衡量动作相对于当前状态平均水平的表现。

使用状态价值作为 Baseline，可以降低梯度估计方差，而不改变期望梯度。

### 问题三：A3C 的异步究竟是什么？

异步的是多个 Worker 对全局网络的梯度更新。

每个 Worker 独立采样、计算梯度，不需要等待其他 Worker 完成。

它解决的是并行采样与训练效率问题，而不是直接限制策略变化。

### 问题四：PPO 为什么需要 Importance Sampling？

因为 PPO 使用旧策略采集的数据，但希望优化新策略。

通过：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

可以将旧策略分布下的采样期望转换成新策略相关的 Surrogate Objective。

### 问题五：PPO 的 Clipping 到底限制了什么？

它限制的是目标函数中概率比率偏离 1 后所能带来的收益，而不是严格地将新策略的所有动作概率比率限制在 $[1-\epsilon,1+\epsilon]$。

这也是 PPO 与 TRPO 的重要区别。

### 问题六：为什么 PPO 还需要 Critic？

因为 PPO 通常采用 Actor-Critic 架构。

Critic 负责估计状态价值：

$$
V_\phi(s_t)
$$

通过 TD Error 和 GAE 得到 Advantage：

$$
A_t^{\mathrm{GAE}}
=
\sum_l(\gamma\lambda)^l\delta_{t+l}
$$

Actor 则利用 Advantage 更新策略。

Critic 并不直接决定动作，而是为 Actor 提供更稳定的学习信号。

---

**核心理解：**

策略梯度解决的是如何优化策略；Actor-Critic 解决的是如何更有效地估计策略梯度；A3C 解决的是如何并行、异步地采样和更新；PPO 解决的是如何在策略优化过程中限制更新幅度，提高训练稳定性。

---

OPD:学生模型自己生成回答，教师模型负责评价学生生成的每一个 Token，并提供细粒度的概率指导。在学生实际访问到的状态上进行蒸馏，而不是只在教师生成的状态上进行蒸馏。