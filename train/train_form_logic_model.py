import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import json
import re
import os
import logging
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os.path
from torch.optim import AdamW

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载.env文件
load_dotenv()

# 全局变量 - 本地模型路径
MODEL_PATH = os.getenv("MODEL_PATH")

# 1. 数据准备类
class FormLogicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 2. 数据处理函数：从LOGIC.md文档中提取规则
def extract_logic_rules(md_content):
    # 提取不同字段类型的规则
    system_fields = re.findall(r'####\s+`([^`]+)`.*?系统字段', md_content, re.DOTALL)
    fixed_fields = re.findall(r'####\s+`([^`]+)`.*?固有字段', md_content, re.DOTALL)
    virtual_fields = re.findall(r'####\s+`([^`]+)`.*?虚拟字段', md_content, re.DOTALL)

    # 提取字段依赖关系
    field_dependencies = []
    field_sections = re.findall(r'####\s+`([^`]+)`(.*?)(?=####|\Z)', md_content, re.DOTALL)

    for field, section in field_sections:
        # 查找依赖的其他字段
        dependencies = re.findall(r'`([^`]+)`', section)
        # 移除重复的字段名称（包括自身）
        dependencies = list(set([d for d in dependencies if d != field]))

        if dependencies:
            field_dependencies.append({
                'field': field,
                'dependencies': dependencies,
                'description': section.strip()
            })

    return {
        'system_fields': system_fields,
        'fixed_fields': fixed_fields,
        'virtual_fields': virtual_fields,
        'field_dependencies': field_dependencies
    }

# 3. 生成训练数据
def generate_training_data(rules):
    texts = []
    labels = []

    # 为字段类型创建样本
    for field in rules['system_fields']:
        texts.append(f"字段 {field} 的类型是什么?")
        labels.append(0)  # 0 表示系统字段

    for field in rules['fixed_fields']:
        texts.append(f"字段 {field} 的类型是什么?")
        labels.append(1)  # 1 表示固有字段

    for field in rules['virtual_fields']:
        texts.append(f"字段 {field} 的类型是什么?")
        labels.append(2)  # 2 表示虚拟字段

    # 为字段依赖关系创建样本
    for dep in rules['field_dependencies']:
        field = dep['field']
        for dependency in dep['dependencies']:
            texts.append(f"字段 {field} 是否依赖于字段 {dependency}?")
            labels.append(3)  # 3 表示存在依赖关系

            texts.append(f"字段 {dependency} 是否影响字段 {field}?")
            labels.append(3)

            # 添加一些负面例子
            random_field = np.random.choice(
                [f for f in rules['system_fields'] + rules['fixed_fields']
                 if f != field and f != dependency]
            )
            texts.append(f"字段 {field} 是否依赖于字段 {random_field}?")
            labels.append(4)  # 4 表示不存在依赖关系

    return texts, labels

# 4. 加载本地模型和tokenizer
def load_local_model(model_path, num_labels=5):
    """
    从本地路径加载模型和tokenizer

    参数:
    model_path (str): 本地模型路径
    num_labels (int): 分类标签数量

    返回:
    tokenizer, model
    """
    logger.info(f"尝试从本地路径加载模型和tokenizer: {model_path}")

    try:
        # 验证路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("成功加载tokenizer")

        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
        )
        logger.info("成功加载模型")

        return tokenizer, model

    except Exception as e:
        logger.error(f"加载本地模型失败: {str(e)}")
        logger.info("尝试从Hugging Face加载默认模型...")

        # 加载默认模型作为备选
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=num_labels
        )
        return tokenizer, model

# 5. 模型训练函数
def train_model(model, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    model.to(device)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"训练损失: {avg_train_loss}")

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                val_loss += loss.item()

                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = correct / total
        logger.info(f"验证损失: {avg_val_loss}, 准确率: {accuracy:.4f}")

    return model

# 6. 模型评估函数
def evaluate_model(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logger.info(f"测试准确率: {accuracy:.4f}")
    return accuracy

# 7. 主函数
def main():
    # 读取LOGIC.md文件
    md_file = 'data/LOGIC.md'
    try:
        with open(md_file, 'r', encoding='utf-8') as file:
            md_content = file.read()
        logger.info(f"成功读取LOGIC.md文件")
    except Exception as e:
        logger.error(f"读取LOGIC.md文件失败: {str(e)}")
        return

    # 提取规则
    rules = extract_logic_rules(md_content)
    logger.info(f"从文档中提取了 {len(rules['system_fields'])} 个系统字段, "
               f"{len(rules['fixed_fields'])} 个固有字段, "
               f"{len(rules['virtual_fields'])} 个虚拟字段, "
               f"{len(rules['field_dependencies'])} 个字段依赖关系")

    # 生成训练数据
    texts, labels = generate_training_data(rules)
    logger.info(f"生成了 {len(texts)} 个训练样本")

    # 划分数据集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    # 加载本地模型和tokenizer
    num_labels = 5  # 0: 系统字段, 1: 固有字段, 2: 虚拟字段, 3: 有依赖关系, 4: 无依赖关系
    tokenizer, model = load_local_model(MODEL_PATH, num_labels)

    # 创建数据集
    train_dataset = FormLogicDataset(train_texts, train_labels, tokenizer)
    val_dataset = FormLogicDataset(val_texts, val_labels, tokenizer)
    test_dataset = FormLogicDataset(test_texts, test_labels, tokenizer)

    # 创建数据加载器
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # 训练模型
    model = train_model(model, train_dataloader, val_dataloader, epochs=3)

    # 评估模型
    accuracy = evaluate_model(model, test_dataloader)

    # 保存微调后的模型
    output_dir = './fine_tuned_form_logic_model'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"模型已保存到 {output_dir}")

    # 保存规则数据
    with open(os.path.join(output_dir, 'form_logic_rules.json'), 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    logger.info("规则数据已保存")

    # 创建一个简单的推理函数来测试模型
    def query_model(query):
        inputs = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

        labels_map = {
            0: "系统字段",
            1: "固有字段",
            2: "虚拟字段",
            3: "有依赖关系",
            4: "无依赖关系"
        }

        return {
            "prediction": labels_map[prediction],
            "confidence": probabilities[0][prediction].item()
        }

    # 测试一些查询
    test_queries = [
        "字段 header_type_id 的类型是什么?",
        "字段 branch_id 是否依赖于字段 header_type_id?",
        "字段 claim_amount 是否依赖于字段 receipt_amount?"
    ]

    logger.info("\n测试查询结果:")
    for query in test_queries:
        result = query_model(query)
        logger.info(f"查询: {query}")
        logger.info(f"预测: {result['prediction']}, 置信度: {result['confidence']:.4f}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()
