import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载.env文件
load_dotenv()


def test_model():
    # 使用原始模型的tokenizer和微调后的模型
    original_model_path = os.getenv("MODEL_PATH")  # 你的原始模型路径
    checkpoint_path = "./results/checkpoint-6"  # 微调后的检查点路径

    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,device_map="cpu")

    # 设置为评估模式
    model.eval()

    # 准备测试提示（包括训练数据中的问题和新问题）
    test_prompts = [
        "什么是系统字段？",
        "什么是固有字段？",
        "什么是虚拟字段？",
        # 添加更多相关问题
    ]

    # 为每个提示生成文本
    for prompt in test_prompts:
        print(f"\n输入提示: {prompt}")

        # 构建完整的提示格式
        full_prompt = f"USER: {prompt}\nASSISTANT: "

        # 对输入进行编码
        inputs = tokenizer(full_prompt, return_tensors="pt")

        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取助手的回答部分
        assistant_response = generated_text.split("ASSISTANT: ")[-1]
        print(f"生成回答: {assistant_response}")


if __name__ == "__main__":
    test_model()
