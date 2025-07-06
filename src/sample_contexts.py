import json
import random

# 假设你的 JSON 已经加载为一个 dict，命名为 data
# data = json.load(open("your_file.json", "r", encoding="utf-8"))

# 这里用一个示例结构代替
data = {
    "\"": [
        {"context": "25\"", "activation": 8.3125},
        {"context": "tactile\"", "activation": 7.75},
        {"context": "Index\"", "activation": 5.53125},
        {"context": "security strip\"", "activation": 5.28125},
        {"context": "x 3 5/8\"", "activation": 5.25},
        {"context": "Index and Tables\"", "activation": 5.25},
    ],
    ".": [
        {"context": "at the bottom of the list.\n\n   - Top: 0.", "activation": 14.5625},
        # … 其他条目 …
    ],
    # … 其他 key …
}

def sample_contexts(data: dict, threshold: float, k: int = 10):
    all_entries = []
    for lst in data.values():
        all_entries.extend(lst)

    filtered = [entry["context"] for entry in all_entries if entry["activation"] > threshold]

    if len(filtered) <= k:
        return filtered
    return random.sample(filtered, k)

if __name__ == "__main__":
    threshold = 6.0
    k = 10
    sampled = sample_contexts(data, threshold, k)
    print(f"随机抽到的 {len(sampled)} 条 context：")
    for ctx in sampled:
        print("—", ctx)
