import logging
import os
import re

from datasets import Dataset as HF_Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 加载.env文件
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 可选：调整 MPS 内存限制或禁用 MPS
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

MODEL_PATH = os.getenv("MODEL_PATH")

if not MODEL_PATH:
    raise ValueError("MODEL_PATH not found in .env file.")


md_file_path = "data/form-eigine.md"

# 从本地文件加载文档内容
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

# 将问答对转换为Hugging Face Dataset格式
qa_data = [{"question": q, "answer": a} for q, a in qa_pairs]
train_dataset = HF_Dataset.from_list(qa_data)

# 加载本地模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)


# 数据处理函数
def preprocess_function(examples):
    prompts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        # 创建对话格式
        prompt = f"USER: {question}\nASSISTANT: {answer}"
        prompts.append(prompt)

    # 对输入进行标记化
    model_inputs = tokenizer(
        prompts, max_length=512, truncation=True, padding="max_length"
    )

    # 创建标签（对因果语言模型来说，标签与输入相同）
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


# 划分训练集和验证集
train_size = int(0.9 * len(train_dataset))  # 90% 作为训练集
eval_size = len(train_dataset) - train_size  # 剩余的 10% 作为验证集
train_dataset, eval_dataset = train_dataset.train_test_split(
    train_size=train_size, test_size=eval_size
).values()

# 应用预处理函数到数据集
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 每个周期进行评估
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    use_cpu=True,
)

# 使用 Trainer 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 提供验证集
)


# 开始训练
trainer.train()
