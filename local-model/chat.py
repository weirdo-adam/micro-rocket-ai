import logging
import os.path
import warnings

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置日志级别，减少不必要的警告
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 加载.env文件
load_dotenv()


def load_local_model():
    # 指定本地模型路径
    model_path = os.getenv("MODEL_PATH")
    # 加载tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True
    )

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    print("Generating response for prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        print("Starting generation...")
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print("Generation completed")

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Decoded response:", response)
    return response


if __name__ == "__main__":
    # 加载模型
    model, tokenizer = load_local_model()
    prompt = "天空为什么是蓝色的？"
    response = generate_response(model, tokenizer, prompt)
    print("\nResponse:", response)
