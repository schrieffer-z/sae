import transformers
from utils import OpenWebTextDataset

# tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/internfs/lisihang/models/meta/Llama-3.2-1B-Instruct/")
# tokenizer.pad_token = tokenizer.eos_token
# dataset = OpenWebTextDataset('/mnt/internfs/lisihang/xAI-RLHF/Wei/10M', tokenizer, 512, 'text')

# print(len(dataset))


def validate_english_text(text: str, allow_extended_ascii=False) -> bool:
    """
    综合验证是否为英语文本
    :param text: 输入字符串
    :param allow_extended_ascii: 是否允许扩展ASCII字符（如é, ñ）
    :return: True表示全英语
    """
    # 定义ASCII及扩展ASCII范围
    max_code = 0xFF if allow_extended_ascii else 0x7F

    for char in text:
        code = ord(char)
        if code > max_code:
            return False
    return True

# 测试
print(validate_english_text("café", allow_extended_ascii=True))   # True
print(validate_english_text("café", allow_extended_ascii=False))  # False





[[]for i,pos in enumerate(max_pos)]