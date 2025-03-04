import re

# 从本地文件加载文档内容
md_file_path = "data/form-eigine.md"
with open(md_file_path, "r", encoding="utf-8") as file:
    md_content = file.read()


# 提取字段信息并转换为QA对
def generate_qa_pairs(section, field_type):
    lines = section.split("\n")
    qa_pairs = []
    for line in lines:
        if line.strip():
            question = f"什么是{field_type}？"
            answer = line.strip()
            qa_pairs.append((question, answer))
    return qa_pairs


# 提取文档中的各个字段的定义
system_field_section = re.findall(
    r"### 系统字段\n\n(.*?)###", md_content, re.DOTALL
)
inherent_field_section = re.findall(
    r"### 固有字段\n\n(.*?)###", md_content, re.DOTALL
)
virtual_field_section = re.findall(
    r"### 虚拟字段\n\n(.*?)###", md_content, re.DOTALL
)

# 生成问答对
system_field_qa = (
    generate_qa_pairs(system_field_section[0], "系统字段")
    if system_field_section
    else []
)
inherent_field_qa = (
    generate_qa_pairs(inherent_field_section[0], "固有字段")
    if inherent_field_section
    else []
)
virtual_field_qa = (
    generate_qa_pairs(virtual_field_section[0], "虚拟字段")
    if virtual_field_section
    else []
)

# 合并所有问答对
qa_pairs = system_field_qa + inherent_field_qa + virtual_field_qa

# 输出数据统计和样本
print(f"总问答对数量: {len(qa_pairs)}")
print(f"系统字段问答对: {len(system_field_qa)}")
print(f"固有字段问答对: {len(inherent_field_qa)}")
print(f"虚拟字段问答对: {len(virtual_field_qa)}")

# 显示一些样本
print("\n样本问答对:")
for i, (q, a) in enumerate(qa_pairs[:5]):
    print(f"问题 {i + 1}: {q}")
    print(f"回答 {i + 1}: {a}")
    print("-" * 50)

# 检查是否有异常长度的回答
answer_lengths = [len(a) for _, a in qa_pairs]
print(
    f"\n回答长度统计: 最小 {min(answer_lengths)}, 最大 {max(answer_lengths)}, 平均 {sum(answer_lengths) / len(answer_lengths):.2f}"
)

# 检查是否有重复的问题
questions = [q for q, _ in qa_pairs]
duplicates = {
    q: questions.count(q) for q in set(questions) if questions.count(q) > 1
}
print(f"\n重复问题: {duplicates}")

# 检查提示格式
print("\n使用的提示格式样例:")
q, a = qa_pairs[0]
formatted_prompt = f"USER: {q}\nASSISTANT: {a}"
print(formatted_prompt)
