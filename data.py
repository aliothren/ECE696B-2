import json

# 文件路径（可根据实际情况修改）
original_json_path = 'dataset/mmlu_high_school_math_100.json'
rephrased_txt_path = 'para.txt'
output_json_path = 'dataset/mmlu_paraphrased.json'

# Step 1: 读取原始 JSON 文件
with open(original_json_path, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Step 2: 读取改写后的题干文本
with open(rephrased_txt_path, 'r', encoding='utf-8') as f:
    new_questions = [line.strip() for line in f if line.strip()]

# 检查是否行数匹配
if len(original_data) != len(new_questions):
    raise ValueError(f"题目数量不匹配：JSON中有{len(original_data)}条，文本中有{len(new_questions)}条。")

# Step 3: 替换每条数据的 question 字段
for item, new_question in zip(original_data, new_questions):
    item['question'] = new_question

# Step 4: 写入新 JSON 文件
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(original_data, f, ensure_ascii=False, indent=2)

print("✅ 新的 JSON 文件已生成：", output_json_path)
