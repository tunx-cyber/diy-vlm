# 字符Tokenizer
class CharacterTokenizer:
    def __init__(self):
        pass  # 不需要额外参数，直接用ord、chr

    def encode(self, text):
        """
        将字符串编码为字符索引列表（Unicode code points）
        """
        return [ord(ch) for ch in text]

    def decode(self, indices):
        """
        将索引列表解码为字符串
        """
        return ''.join([chr(i) for i in indices])

# 测试代码
if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    string = "hi，很好的，terrific！🐋"  # 测试字符串

    # 编码
    indices = tokenizer.encode(string)
    print("编码ID:", indices)

    # 解码
    reconstructed_string = tokenizer.decode(indices)
    print("解码:", reconstructed_string)

    # 验证是否可逆
    assert string == reconstructed_string, "字符编码、解码不一致!"

    # 计算词汇量（最大Unicode code point+1）
    vocabulary_size = max(indices) + 1
    print("词汇量（上限）", vocabulary_size)

    # 简单压缩率计算
    def get_compression_ratio(text, indices):
        # 压缩率 = 原字符串字节数/编码索引字节数
        import sys
        original_bytes = len(text.encode('utf-8'))
        encoded_bytes = len(indices) * 4  # 假设每个Unicode code point用4字节存储
        return original_bytes / encoded_bytes

    compression_ratio = get_compression_ratio(string, indices)
    print("压缩比率:", compression_ratio)