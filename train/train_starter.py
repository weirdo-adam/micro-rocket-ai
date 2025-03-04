import logging
import os

from dotenv import load_dotenv
from torch.utils.data import Dataset
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
    raise ValueError("环境变量MODEL_PATH未设置")

logger.info(f"使用模型路径: {MODEL_PATH}")


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def main():
    try:
        logger.info("正在加载模型和分词器...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # 使用 CPU 加载模型，降低内存需求
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cpu",  # 强制在CPU上加载
            # 如果需要量化，可以添加量化配置
        )
        logger.info("模型和分词器加载成功!")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # 简单的示例数据
    sample_texts = [
        "这是一个微调模型的示例。",
        "我们正在使用本地模型文件进行微调。",
        "机器学习模型可以通过微调来适应特定任务。",
        "深度学习模型的训练需要大量的计算资源。",
        "迁移学习是一种有效的学习方法。",
    ]

    # 创建数据集
    train_dataset = SimpleDataset(sample_texts, tokenizer)

    # 训练参数 - 调整以减少内存使用
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # 减小批次大小
        gradient_accumulation_steps=2,  # 使用梯度累积
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        # 如果需要在CPU上训练
        use_cpu=True,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存微调后的模型
    output_dir = "./fine_tuned_model"
    logger.info(f"保存微调后的模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("微调完成!")


if __name__ == "__main__":
    main()
