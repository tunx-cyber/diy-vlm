# 基于空格和标点切分的实现示例
import re
         
def part(text):
   # 将标点符号单独拆开，并按照空格进行分割
   text = re.sub(r'([.,!?;:()"\'\[\]{}])', r' \1 ', text)
   tokens = text.split()
   return tokens

# 测试
if __name__ == "__main__":         
   s = "I like Datawhale."
   print(part(s))