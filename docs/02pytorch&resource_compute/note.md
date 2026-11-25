## 资源计算

### 时间估算
总计算量 $\approx 6 \times 参数量 \times token数$

为什么还是六倍
- 前向传播 (Forward Pass): 计算一次约为2倍参数量（乘法+加法）。
- 反向传播 (Backward Pass): 计算梯度约为前向传播的2 倍工作量，即4倍参数量

显卡的计算速度要看是否有结构化稀疏，普通的绸密模型，性能峰值只有一半，比如H100的990TFLOPS才是我们正常训练模型的理论峰值，实际上由于软硬件开销，只能30%到60%的利用率。

### 内存估算
训练的时候，显存里面有
- 模型参数
- 梯度，和参数一样大
- 优化器状态
- 激活值

假设fp32
- 参数：4byte
- 梯度：4byte
- 优化器： 2*4byte

激活值=前向传播中每一层的输出，每经过一个矩阵乘法或非线形函数就会产生中间结果

## 张量
看代码[示例代码与讲解](./src/ex1.py)

## 内存
张量的内存占用由两个因素决定：
- 元素数量：张量的形状（如4×8矩阵有32个元素）
- 数据类型：每个元素占用的字节数

### 浮点类型
fp32
- 规格：占用 4 字节 (32 bits)。结构为：1位符号位 + 8位指数位 + 23位尾数位。
- 地位：PyTorch 的默认数据类型，也是科学计算领域的“黄金标准”。
- 优点：数值精度高，动态范围大，训练最稳定，几乎不会出现数值溢出问题。
- 缺点：对于大模型而言太“奢侈”。它占用的显存是 16位格式的两倍，且在现代 GPU（如 H100）上的计算吞吐量远低于低精度格式。
- 炼丹用途：通常用于存储参数的主副本 (Master Weights) 和 优化器状态 (Optimizer States)，以确保在梯度累积和参数更新时不会因为精度丢失而导致模型无法收敛。

fp16
- 规格：占用 2 字节 (16 bits)。结构为：1位符号位 + 5位指数位 + 10位尾数位。
- 优点：显存占用比 FP32 减少一半，计算速度快。
- 致命缺陷：动态范围（Dynamic Range）太窄。
- 由于指数位只有 5 位，它无法表示非常小的数（会发生下溢 Underflow，直接变成 0）或非常大的数（会发生上溢 Overflow，变成 Infinity）。
- 例如：在 FP16 中，1e-8 这样的小数会被直接当作 0 处理，导致梯度消失。
- 用途：这是上一代 GPU（如 V100）混合精度训练的主流。为了解决溢出问题，必须使用复杂的损失缩放 (Loss Scaling) 技术。目前在 LLM 训练中正逐渐被 BF16 取代。

bf16
- 规格：占用 2 字节 (16 bits)。结构为：1位符号位 + 8位指数位 + 7位尾数位。
- 设计逻辑：“要范围，不要精度”。深度学习模型（尤其是神经网络）对小数点的后几位精度不敏感，但对数值的范围非常敏感。
- BF16 直接截断了 FP32 的尾数，但保留了和 FP32 相同的 8 位指数位。
优点：
- 拥有和 FP32 一样宽广的动态范围，不需要 Loss Scaling 也能稳定训练。
- 显存占用和 FP16 一样少。
- 在 A100/H100 等新硬件上计算速度极快。
- 用途：当前 LLM 训练的绝对主流选择。通常用于存储激活值 (Activations) 以及进行前向和反向传播的矩阵乘法计算。

在深度学习中，我们通常不关心小数点后第10位的精确度（精度），但非常关心能否表示非常大或非常小的数（动态范围）。FP16 的指数位太少，导致训练中经常出现 NaN 或 0（下溢）。BF16 截断了 FP32 的尾数，保留了指数位，因此它能表示的数值范围与 FP32 一样大，极大地提升了训练稳定性。


fp8
- 规格：占用 1 字节 (8 bits)。
- 变体：
    - E4M3：4位指数，3位尾数（精度稍高，范围稍小）。
    - E5M2：5位指数，2位尾数（范围稍大，精度更低）。

优点：极致的显存压缩和计算吞吐量。
- 硬件限制：仅在 NVIDIA H100 及更新的架构（配合 Transformer Engine）上才被原生支持。
- 炼丹用途：目前主要用于推理阶段的量化 (Quantization)。虽然理论上可以用于训练（H100 支持），但由于精度极低，训练极不稳定，属于比较前沿的研究领域。

nvfp4
NVFP4 不是单纯每个值都只有 4 比特的动态范围。"如果你天真地只用这 16 个值，是无法有效训练的。实际上每个值拥有 4 比特的自由度，但数据被划分成 block（块），每个块有一个共享的缩放因子（scale factor），可以整体放大或缩小。这样单个值实际拥有超过 4 比特的动态范围，但相邻值之间的比值仍然受限于 4 比特。"
用一个类比：每个 block 内你可以在 4 比特的精度范围内自由变化，而 block 的缩放因子让整个 block 的值可以移到不同的量级。代价是你不能让一个值非常大而它旁边的值非常小，因为它俩属于同一个 block，共享同一个缩放因子。
训练 vs 推理：Percy 特别区分了两者。

推理时：可以在 bf16 训练好模型后，将模型权重量化到 1 或 2 比特
训练时：用 1 比特训练语言模型要困难得多——Percy 直言"不认为有任何人做过可信的 1-bit 训练"

### 张量在内存中的存储机制
在早期的 pytorch 版本中，张量在一台机器上的存储，占用了两部分内存，一个内存储存了这个张量的形状（size）、步长（stride）、数据类型（dtype）等元数据信息，我们把这一部分称之为头信息区（Tensor）；另一个内存储存的就是真正的数据，我们称为存储区 （Storage）。但是将“存储区”作为一个独立的、可公开访问的对象，其带来的复杂性和潜在风险，远大于它为用户提供的一些灵活性。

在现在的 pytorch 版本中，PyTorch 的设计者们选择将这一层实现细节进行封装，让 Tensor 对象本身成为一个集成了“指针+元数据”的单一实体。这意味着一个张量本身并不直接“包含”数据，而是指向一块连续的内存区域，并附带一套规则（元数据），告诉程序如何根据你请求的索引去这块内存里找到对应的数据。

比如
```python
x = torch.tensor([
    [0., 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
])
```

张量在底层可以是按行存储也可以是按列存储。Numpy 和 Pytorch 都采用了按行存储的方式，任何维度的张量在底层存储都占据着内存中连续的空间，那么问题来了，我们如何访问到我们想要的位置的数据？
答案就是步长（Strides）！步长是一个元组，定义了在每个维度上移动一个单位时，在底层存储中需要“跳过”的元素数量。

```python
# 步长（stride）：在每个维度上移动时跳过的元素数
assert x.stride(0) == 4  # 行维度：移动到下一行需跳过4个元素
assert x.stride(1) == 1  # 列维度：移动到下一列需跳过1个元素

# 计算元素位置：行索引r，列索引c
r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1)  # 位置6
```

用一句话来说，PyTorch通过“步长”这一元数据，巧妙地将多维张量的逻辑结构映射到一维的物理内存上。
关键点： 很多操作（如 view, transpose, slice）不会复制数据，只是修改了 stride。这叫 Zero-copy，非常高效。

### 将张量从cpu移动到gpu
默认情况下，张量是存储在 CPU 内存上的。
检查是否有gpu
```python
torch.cuda.is_available():
```

获取GPU信息:
```python
num_gpus = torch.cuda.device_count() # 获取GPU数量
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i) # 获取第i块GPU的属性（如型号、显存大小等）
memory_allocated = torch.cuda.memory_allocated() # 获取当前已分配给PyTorch的显存总量
```

移动张量到GPU上有两种方法
```python
y = x.to("cuda:0") # 将张量x移动到编号为0的GPU上
assert y.device == torch.device("cuda", 0) # 验证移动成功

z = torch.zeros(32, 32, device="cuda:0") # 在GPU上直接创建一个32x32的零矩阵
```

验证内存分配
```python
new_memory_allocated = torch.cuda.memory_allocated() # 再次查询当前显存占用
memory_used = new_memory_allocated - memory_allocated # 计算新增的显存占用
assert memory_used == 2 * (32 * 32 * 4) # 验证：新增了2个32x32的矩阵，每个元素是4字节的float
```

## 计算效率

### 浮点运算总数
一个浮点运算（FLOP, Floating Point Operation）是一个基本的算术操作，比如加法 x + y 或乘法 x * y，理解 FLOPs 对性能分析至关重要。

首先，我们先辨析几个常见概念：

FLOPs（小写s）：执行的浮点运算总数，这是一个衡量计算量的指标，
FLOP/s（每秒浮点运算）：指硬件每秒能执行的浮点运算次数，也写作 FLOPS（大写S）。这是一个衡量“速度”的指标。例如，H100 GPU峰值性能为 990 TFLOP/s。


线性模型的计算量
```python
if torch.cuda.is_available():
    B = 16384  # Number of points
    D = 32768  # Dimension
    K = 8192   # Number of outputs
else:
    B = 1024
    D = 256
    K = 64
device = get_device()
x = torch.ones(B, D, device=device)
w = torch.randn(D, K, device=device)
y = x @ w
```
接下来让我们进行分析：

- x 是输入数据，形状为 (B, D)
- B：batch size（数据点数量）
- D：每个数据点的维度（输入特征数）
- w 是权重矩阵，形状为 (D, K)
- K：输出维度（比如分类任务的类别数）
- y 是输出，形状为 (B, K)

考虑 y = x @ w 中的一个输出元素 y[i, k]：
```python
y[i, k] = x[i, 0] * w[0, k] +
           x[i, 1] * w[1, k] +
           ...
           x[i, D-1] * w[D-1, k]
```
D次乘法和D-1次加法，总共就是约等于2D次FLOPs，y有B*K个，所以总的为 $2 \times B \times k \times D$，其中DK刚好是参数量。

虽然相较于线性模型， Transformer 还有注意力、LayerNorm 等额外操作，但矩阵乘法占主导，所以这个公式对 Transformer 也基本成立

benchmarking
```python
def benchmark(func, num_trials=5):
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # 确保之前的 CUDA 操作已完成
    # 运行 num_trials 次并取平均
    ...
    torch.cuda.synchronize()   # 操作后的同步点
```

### pytorch的梯度计算

前向
```python
x = torch.tensor([1., 2, 3]) # 输入数据，不需要计算梯度
w = torch.tensor([1., 1, 1], requires_grad=True) # 模型参数，需要计算梯度
pred_y = x @ w # 预测值（矩阵乘法）
loss = 0.5 * (pred_y - 5).pow(2) # 计算损失值
```
requires_grad=True 这个参数它告诉 PyTorch “请为这个张量 w 构建计算图，并在反向传播时计算它的梯度。” 对于输入数据 x，我们通常不需要它的梯度，所以不设置此标志。

方向传播
```python
loss.backward() # 触发反向传播
assert loss.grad is None # 损失值 loss 本身是一个标量，它没有“上游”的梯度，所以它的 .grad 属性是 None
assert pred_y.grad is None # 中间变量 pred_y 默认情况下也不会保存梯度，除非你显式地调用 pred_y.retain_grad()
assert x.grad is None # 因为 x 在创建时没有设置 requires_grad=True，所以它不会被追踪，其 .grad 也为 None
assert torch.equal(w.grad, torch.tensor([1, 2, 3])) # 验证模型参数 w 的梯度是否计算正确
```

调用 loss.backward() 后，PyTorch 会从 loss 开始，沿着计算图回溯，自动计算出每个需要梯度的张量的梯度。

计算反向传播（梯度）所需的浮点运算次数

```python
if torch.cuda.is_available():
    B = 16384  # Number of points
    D = 32768  # Dimension
    K = 8192   # Number of outputs
else:
    B = 1024
    D = 256
    K = 64
device = get_device()
x = torch.ones(B, D, device=device)
w1 = torch.randn(D, D, device=device, requires_grad=True)
w2 = torch.randn(D, K, device=device, requires_grad=True)

# Model: x --w1--> h1 --w2--> h2 -> loss
h1 = x @ w1 # 第一层：输入x乘以权重w1得到隐状态h1
h2 = h1 @ w2 # 第二层：h1乘以权重w2得到输出h2
loss = h2.pow(2).mean()  # 计算损失（均方误差）
```

让我们重点分析w2.grad的计算，根据链式法则（$h_2=h_1W_2 -> \frac{\partial L}{\partial W_2} = h_1^T \frac{\partial L}{\partial h_2}$）,

$$\frac{\partial L}{\partial W_2[j,k]} = \sum_i h_1[i,j] \frac{\partial L}{\partial h_2[i,k]}$$

其中i是batch维的下标。

这本质上是一个矩阵乘法：$h_1$的形状是(B,D)，$h_2.grad$是(B,K)，$w_2.grad$是（D，K）,所以计算 $w_2.grad$ 的FLOPs是 $2 \times B \times D \times K$。

为了将梯度继续传到第一层，需要先求 $\partial L/ \partial h_1$。由 $h_2 = h_1 W_2$得到 $\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_2} W_2^T$:

$$\frac{\partial L}{\partial h_1[i,j]} = \sum_k\frac{\partial L}{\partial h_2[i,k]} W_2[j,k]$$

计算 $h_1$ 的FLOPs是 $2 \times B \times D \times K$。

同理，第一层 $h_1 = x W_1$的梯度是 $\frac{\partial L}{\partial W_1} = x^T\frac{\partial L}{\partial h_1}$

$$\frac{\partial L}{\partial W_1[j,k]} = \sum_i x[i,j] \frac{\partial L}{\partial h_1[i,k]}$$

其中的FLOPs是 $2 \times B \times D \times D$。

总的为 $2 \times B \times D \times D + 4 \times B \times D \times K$。

这并不影响 6× 规则的成立。关键在于：一层完整的反向传播本应包含两个矩阵乘法，分别计算“权重梯度”和“输入梯度”，二者的计算量都等于该层前向；只要该层输入要求梯度，这两个矩阵乘法都会执行，该层反向就是 2× 前向。

第一层因为输入是数据、不求梯度，省掉了“输入梯度”这一步，故只算 1× 前向；但在真实的多层网络中，首层只占总参数的 $O(1/L)$，可以忽略


## 模型构建与训练基础

### 参数初始化

模型参数的存储
在 PyTorch 中，可训练的模型参数被封装为 nn.Parameter 对象：
```python
w = nn.Parameter(torch.randn(input_dim, output_dim))
```
nn.Parameter 是 torch.Tensor 的子类，因此它“表现得像一个张量”，可以进行所有张量操作。它有一个 .data 属性，用于访问其底层的 torch.Tensor 数据。

如果你直接用 torch.randn 进行标准高斯分布初始化参数，随着层数变深，数值会变得非常大（爆炸）或非常小（消失）。

为了克服这个问题，我们需要一种对输入维度 input_dim 不敏感的初始化方法。解决方法是使用 Xavier初始化，使用除以输入维度的根号来缩放权重
```python
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
```
经过缩放后，输出 output 的每个元素的值变得稳定在一个较小的范围内，不再随 input_dim 增长。

即使使用了 Xavier 初始化，由于正态分布的尾部是无界的，仍然存在产生极端值（outliers）的可能性。解决方案是使用截断正态分布（truncated normal distribution），将生成的随机数限制在一个合理的范围内
```python
w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), std=1 / np.sqrt(input_dim), a=-3, b=3))
```

### 自定义模型
[示例代码](./src/ex2.py)

### 管理随机性
随机性在深度学习的许多地方都会出现：

- 参数初始化：模型权重通常从随机分布（如正态分布）中采样。
- Dropout：训练时随机“关闭”一部分神经元。
- 数据排序：数据加载器通常会打乱数据顺序。
- 其他：如数据增强、优化器中的动量等。

设置随机种子
```python
seed = 0
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
```

### 数据加载
```python
orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
orig_data.tofile("data.npy") # 将数组保存到文件
```

对于超大数据集，使用memmap创建内存映射对象

```python
data = np.memmap("data.npy", dtype=np.int32) # 创建内存映射
assert np.array_equal(data, orig_data) # 验证加载的数据是否正确
```
data 对象本身不包含数据，而是一个指向磁盘上文件的“指针”。当你访问 data[0] 或 data[1:10] 时，系统才会按需将这部分数据从磁盘加载到内存中。这种方式极大地节省了内存占用，允许你在内存有限的情况下处理远超内存大小的数据集。

在实际训练中，我们不会直接操作整个数据集，而是通过一个“数据加载器”来生成训练所需的批次。

```python
def get_batch(data: np.array, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    # 随机采样起始位置
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,)) # 使用 torch.randint 在 [0, len(data) - sequence_length] 范围内随机生成 batch_size 个起始索引。这样可以确保每个序列都能完整地从数据中截取出来，不会越界
    assert start_indices.size() == torch.Size([batch_size]) # 断言验证了生成的索引数量正确

    # 根据起始索引提取数据
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices]) # 遍历 start_indices 中的每一个起始位置 start，从 data 中切片出长度为 sequence_length 的子序列。最后将这些子序列转换成一个 PyTorch 张量 x，其形状为 (batch_size, sequence_length)
    assert x.size() == torch.Size([batch_size, sequence_length])

    # 固定内存（Pinned Memory）优化
    if torch.cuda.is_available():
        x = x.pin_memory()

    # 异步数据传输
    x = x.to(device, non_blocking=True)

B = 2  # 批次大小，即一次训练处理多少个样本
L = 4  # 序列长度，即每个样本包含多少个词元（token）
x = get_batch(data, batch_size=B, sequence_length=L, device=get_device())
assert x.size() == torch.Size([B, L]) # 验证输出张量的形状
```
- 固定内存（Pinned Memory）优化：默认情况下 CPU 上的张量存储在“分页内存”（paged memory）中。当需要将数据从 CPU 传输到 GPU 时，操作系统必须先将数据复制到一个“非分页”的内存区域，然后再通过 PCIe 总线发送给 GPU。这个过程是同步的，会阻塞当前进程。通过调用 .pin_memory()，我们可以将 CPU 张量显式地标记为“固定”，这意味着它的物理内存地址是固定的，不会被操作系统换出到磁盘。这使得 GPU 驱动程序可以直接访问这块内存，无需额外的拷贝步骤。

- 异步传输：设置 non_blocking=True 参数，这告诉 PyTorch 数据传输可以在后台异步进行，不会阻塞当前的 Python 线程

通过结合“固定内存”和“异步传输”，我们可以实现以下高效的流水线操作：

- 在 GPU 上处理当前批次的数据。
- 在 CPU 上同时加载下一个批次的数据（例如，从硬盘读取或从内存映射文件中加载）。

这种并行化可以显著减少 GPU 的空闲等待时间，从而大幅提升整体训练吞吐量

### 优化器

常用优化器介绍

- SGD (随机梯度下降)：最基础的优化器，直接用学习率乘以梯度更新参数。
- Momentum (动量法)：在 SGD 基础上增加了一个“动量”项，即梯度的指数移动平均值，有助于加速收敛并减少震荡。
- AdaGrad：根据历史梯度的平方值来调整每个参数的学习率，对稀疏特征更友好。
- RMSProp：对 AdaGrad 的改进，使用梯度平方的指数加权平均代替简单累加，避免学习率过早衰减。
- Adam：融合了 RMSProp 和 Momentum 的思想，是目前最流行的优化器。

```python
class AdaGrad(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                # 获取状态（梯度平方和）
                state = self.state[p]
                if 'sum_squared_grad' not in state:
                    state['sum_squared_grad'] = torch.zeros_like(p.data)
                
                # 更新状态：累加梯度的平方
                state['sum_squared_grad'] += grad ** 2
                
                # 更新参数：除以 根号下(状态)
                std = state['sum_squared_grad'].sqrt() + 1e-10
                p.data -= group['lr'] * grad / std
```

优化器的使用

```python
# 实例化优化器
optimizer = AdaGrad(model.parameters(), lr=0.01) # model.parameters()：将模型中所有可学习的参数传递给优化器

# 计算梯度
loss.backward() # 计算损失函数对所有参数的梯度

# 执行一步更新
optimizer.step() # 根据梯度和优化器内部状态，更新模型参数

optimizer.zero_grad(set_to_none=True) # 优化器在调用 zero_grad 函数清空梯度时，设置 set_to_none=True 是一种更高效的内存管理方式，它会将梯度指针设为 None，而不是将其置零，可以节省内存。
```

优化器底层实现
```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(SGD, self).__init__(params, dict(lr=lr)) # 调用父类 Optimizer 的初始化方法，并将学习率 lr 存储在参数组的字典中

    def step(self):
        for group in self.param_groups: # 遍历所有的参数组（通常只有一个组）
            lr = group["lr"]
            for p in group["params"]: # 遍历该组中的每一个参数 p
                grad = p.grad.data  # 获取该参数的梯度 grad
                p.data -= lr * grad # 更新参数

class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p] # 获取优化器状态
                grad = p.grad.data

                g2 = state.get("g2", torch.zeros_like(grad)) # 获取或初始化梯度平方和
                g2 += torch.square(grad) # 累加当前梯度的平方
                state["g2"] = g2 # 更新状态

                p.data -= lr * grad / torch.sqrt(g2 + 1e-5) # 更新参数
```

### 训练循环
```python
def train_loop():
    # 数据生成
    D = 16
    true_w = torch.arange(D, dtype=torch.float32, device=get_device()) # 创建一个真实的权重向量 [0, 1, 2, ..., 15]

    # 数据加载器，用于生成训练批次
    def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(get_device()) # 从标准正态分布中随机采样 B 个 D 维的输入样本
        true_y = x @ true_w # 根据真实的权重 true_w 计算出对应的“真实标签”，即 y = x @ w_true
        return (x, true_y) # 作为模型的输入和目标输出

    # 执行训练
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.01)
    # 超参数调优（hyperparameter tuning）
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.1) # 通过改变学习率 lr 从 0.01 到 0.1，观察不同学习率对训练效果的影响

def train(name: str, get_batch,
          D: int, num_layers: int,
          B: int, num_train_steps: int, lr: float):

    # 初始化模型和优化器
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = SGD(model.parameters(), lr=lr)

    # 主训练循环
    for t in range(num_train_steps): # 循环执行 num_train_steps 次，每次迭代称为一个“训练步”
        # 获取数据
        x, y = get_batch(B=B) # 获取当前批次的输入 x 和目标输出 y

        # 前向传播（计算损失）
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y) # 计算预测值 pred_y 和真实值 y 之间的均方误差（Mean Squared Error），作为损失值 loss

        # 反向传播（计算梯度）
        loss.backward() # 调用 loss.backward()，触发自动微分机制，计算损失函数对所有模型参数的梯度，并将这些梯度存储在 .grad 属性中
        
        # 更新参数
        optimizer.step() # 根据计算出的梯度和优化器的更新规则（这里是 SGD），更新模型参数
        optimizer.zero_grad(set_to_none=True) # 清空上一步计算得到的梯度，为下一次迭代做准备
```

### 检查点

```python
# 1. 创建一个字典，包含模型和优化器的状态
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

# 2. 使用 torch.save 将字典序列化并保存到磁盘文件
torch.save(checkpoint, "model_checkpoint.pt") # 文件名通常使用 .pt 或 .pth 后缀


# 1. 从磁盘文件加载保存的字典
loaded_checkpoint = torch.load("model_checkpoint.pt")

# 2. （后续步骤）将状态加载回模型和优化器
# model.load_state_dict(loaded_checkpoint["model"])
# optimizer.load_state_dict(loaded_checkpoint["optimizer"])
```

### 混合精度训练

默认使用 float32，确保关键部分的计算精度。在可能的情况下使用 {bfloat16, fp8}，利用其高效的内存和计算特性。如下给出一个经典的混合精度训练方案：

- 前向传播（Forward Pass）：使用 bfloat16 或 fp8。这包括所有中间激活值（activations）。因为激活值通常不需要极高的精度，使用低精度可以显著节省内存。
- 其余部分：使用 float32。这包括模型参数（parameters）、梯度（gradients）以及优化器状态（optimizer states）。这些是训练的核心，需要更高的精度来保证数值稳定性和收敛性。

工具
- PyTorch 的 AMP 库 (Automatic Mixed Precision)：PyTorch 提供了自动混合精度（AMP）库，会自动将安全的操作（如矩阵乘法）转为 bf16，将危险操作（如 exp、softmax）保留为 fp32：
```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    x = torch.zeros(4, 8)  # 自动以 bf16 创建
```
- NVIDIA Transformer Engine：这是一个专门针对 Transformer 模型优化的库，它支持在矩阵乘法等核心操作中使用 FP8 精度。目标是实现全链路的 FP8 训练，即在整个训练过程中都使用 FP8，以达到极致的性能和效率。


### 算术强度

GPU 的工作模型：

1. 从 HBM（高带宽内存）发送输入到计算核心
2. 执行计算
3. 将输出从计算核心送回 HBM

以一个简单的 ReLU 操作为例（1024 × 1024 维的 bf16 向量）：

内存移动（Bytes）：

读入 x：2 × n（bf16 是 2 字节/元素）
写出 y：2 × n
总计：4n
计算量（FLOPs）：

n 次比较（max(x, 0)）
共 n FLOPs

通信时间 = 4n / (3.35 × 10¹²) ≈ 1.2 × 10⁻⁶ 秒 计算时间 = n / (989.5 × 10¹²) ≈ 1.0 × 10⁻⁹ 秒


为了避免每次都要计算两个时间再比较，引入算术强度（Arithmetic Intensity）：
```text
加速器强度 = FLOP/s / Bytes/s  → H100 约 295 FLOP/byte
算术强度 = FLOPs / Bytes        → 该操作每搬运 1 字节能做多少 FLOP
```

算术强度 < 加速器强度 → Memory-bound（瓶颈在数据传输）
算术强度 > 加速器强度 → Compute-bound（瓶颈在计算）

