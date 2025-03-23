import json

# 文件路径（修改为你的实际文件路径）
input_file = "mmlu_high_school_math.json"  # 原始 JSON 文件
output_file = "mmlu_high_school_math_100.json"  # 处理后的 JSON 文件

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 过滤出 "answer" 不是 0 的数据（即答案不是 A）
filtered_data = [sample for sample in data if sample["answer"] != 0]

# 只保留前 100 条
filtered_data = filtered_data[:100]

# 保存处理后的 JSON 数据
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"✅ 处理完成，已保存为 {output_file}")