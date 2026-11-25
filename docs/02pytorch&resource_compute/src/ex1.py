import torch
import torch.nn as nn

def create_tensor():
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # 从Python列表创建
    x = torch.zeros(4, 8)  # 4x8的零矩阵
    x = torch.ones(4, 8)   # 4x8的全1矩阵
    x = torch.randn(4, 8)  # 4x8的正态分布随机数

    # 分配但不初始化值（用于自定义初始化的值）
    x = torch.empty(4, 8)  # 未初始化的4x8矩阵
    nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # 截断正态分布初始化，分布均值为0，标准差为1，截断范围，只保留[-2, 2]区间内的值，超出此范围的值会被重新采样

def rank_of_tensor():
    x = torch.zeros(4)        # 秩 1 张量（向量），shape = (4,)
    x = torch.zeros(4, 8)     # 秩 2 张量（矩阵），shape = (4, 8)
    x = torch.zeros(4, 8, 2)  # 秩 3 张量，shape = (4, 8, 2)

def view_of_tensor():
    '''
    许多操作只是提供张量的一个不同“视图”。这不会创建副本，因此在一个张量中的修改会影响另一个。
    '''
    def same_storage(x, y):
        return x.data_ptr() == y.data_ptr()
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])

    # 这些操作不复制数据
    y = x[0]           # 获取第0行
    y = x[:, 1]        # 获取第1列
    y = x.view(3, 2)   # 重塑为3x2
    y = x.transpose(1, 0)  # 转置

    # 验证是否共享存储
    assert same_storage(x, y)  # same_storage检查底层存储是否相同

    # 注意：修改x会影响y
    x[0][0] = 100
    assert y[0][0] == 100  # 值也被修改
    
    # 注意：并非所有视图都是“连续”的。当一个视图的数据在内存中不是按顺序存储时，它就是非连续的（Non-contiguous）。
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])
    y = x.transpose(1, 0)
    assert not y.is_contiguous()  # 非连续张量

    # 你不能对非连续张量直接执行某些操作，比如再次 view:
    # 尝试重塑会失败
    try:
        y.view(2, 3)
    except RuntimeError as e:
        print("view error")
        assert "view size is not compatible with input tensor's size and stride" in str(e)

    # 如果需要对一个非连续张量进行进一步操作，可以先调用 .contiguous() 方法：
    # .contiguous() 会创建一个新的张量，并将数据按顺序复制到新的连续内存块中，这样后续操作就不会出错。
    y = x.transpose(1, 0).contiguous().view(2, 3)
    assert not same_storage(x, y)  # 现在创建了新存储

def ele_wise_op():
    '''
    逐元素操作是对每个元素独立应用函数，并返回相同形状的新张量。
    这意味着，当你对一个张量执行 x.pow(2) 或 x + x 时，操作会独立地作用于张量中的每一个数字，
    而不会像矩阵乘法那样考虑元素之间的行列关系。
    '''
    x = torch.tensor([1, 4, 9])
    assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))    # 幂运算 (pow)，每个元素平方
    assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))      # 对每个元素开平方根
    assert torch.equal(x.rsqrt(), torch.tensor([1, 1/2, 1/3])) # rsqrt 是 reciprocal of sqrt 的缩写，即对每个元素先开方再取倒数
    assert torch.equal(x + x, torch.tensor([2, 8, 18])) # 每个元素自加
    assert torch.equal(x * 2, torch.tensor([2, 8, 18])) # 每个元素乘以标量2
    assert torch.equal(x / 0.5, torch.tensor([2, 8, 18])) # 每个元素除以标量0.5
    # 实用操作：下三角矩阵（用于因果注意力掩码）
    x = torch.ones(3, 3).tril()  # 创建一个3x3全1矩阵，然后取其下三角部分（含对角线）
    print("创建一个3x3全1矩阵，然后取其下三角部分（含对角线）")
    print(x)
    # 结果：
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]
    # 含义：第0个词只能看到第0个词，第1个词能看到第0、1个词，依此类推

    x = torch.ones(3, 3).triu()   # 上三角（含对角线）
    print("上三角（含对角线）")
    print(x)
    # [[1, 1, 1],
    #  [0, 1, 1],
    #  [0, 0, 1]]

    y = torch.ones(3, 3).triu(1)  # 严格上三角（不含对角线），取反后即为因果掩码
    print("严格上三角（不含对角线），取反后即为因果掩码")
    print(y)
    # [[0, 1, 1],
    #  [0, 0, 1],
    #  [0, 0, 0]]

def mat_mul():
    '''
    一个标准的矩阵乘法形式为：一个 M×K 的矩阵乘以一个 K×N 的矩阵，得到一个 M×N 的结果矩阵
    '''
    # 基本矩阵乘法
    x = torch.ones(16, 32)  # 16x32矩阵
    w = torch.ones(32, 2)   # 32x2权重矩阵
    y = x @ w               # 结果：16x2矩阵
    assert y.size() == torch.Size([16, 2])

    '''
    在 PyTorch 进行上述运算时，会自动遍历 x 的前两个维度（即 4 和 8），
    对每一个 16×32 的子矩阵执行与 w 的矩阵乘法。
    最终结果 y 的形状是 (4, 8, 16, 2)，其中前两个维度保持不变，后两个维度根据矩阵乘法规则变化。
    '''
    x = torch.ones(4, 8, 16, 32) # 形状为 [批大小, 序列长度, 样本特征, 特征维度]
    w = torch.ones(32, 2)        # 权重矩阵
    y = x @ w                    # 矩阵乘法
    assert y.size() == torch.Size([4, 8, 16, 2]) # 结果形状


    # 传统方式（容易写错维度顺序）
    x = torch.ones(2, 2, 1, 3)  # batch seq heads hidden

    # Jaxtyping方式（在类型注解中命名维度）
    from jaxtyping import Float
    x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)

    from einops import einsum

    # 定义两个张量
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)  
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)  

    # 传统方式
    z = x @ y.transpose(-2, -1)  # batch, sequence, sequence

    # Einops方式
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
    # 未在输出中命名的维度（hidden）会被自动求和
    # 这种方法会将没在输出中出现的 hidden 维度，自动对该维度做求和。
    # 这本质上是爱因斯坦求和约定（Einstein summation） 的直观实现。更通用的写法是（支持任意前导维度）：
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2") #... 表示“任意数量的前导维度”



if __name__ == "__main__":
    create_tensor()
    view_of_tensor()
    ele_wise_op()
    mat_mul()