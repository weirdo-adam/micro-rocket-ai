import nltk
from evaluate import load
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

nltk.download("punkt")


def evaluate_with_metrics():
    # 加载微调后的模型
    model_path = "./fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 准备测试数据
    test_data = [
        {
            "input": "解释机器学习是什么？",
            "reference": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需显式编程。",
        },
        {
            "input": "描述神经网络的基本结构",
            "reference": "神经网络由多层神经元组成，包括输入层、隐藏层和输出层。每个神经元接收输入，应用激活函数，并产生输出。",
        },
    ]

    # 加载评估指标
    bleu = load("bleu")
    rouge = load("rouge")

    # 存储结果
    bleu_scores = []
    rouge_scores = []

    for item in test_data:
        input_text = item["input"]
        reference = item["reference"]

        # 生成模型输出
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 计算BLEU分数
        bleu_result = bleu.compute(
            predictions=[prediction], references=[[reference]]
        )
        bleu_scores.append(bleu_result["bleu"])

        # 计算ROUGE分数
        rouge_result = rouge.compute(
            predictions=[prediction], references=[reference]
        )
        rouge_scores.append(rouge_result["rougeL"])

        # 打印单个样例结果
        print(f"\n输入: {input_text}")
        print(f"参考: {reference}")
        print(f"预测: {prediction}")
        print(f"BLEU: {bleu_result['bleu']:.4f}")
        print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")

    # 打印平均分数
    print(f"\n平均BLEU分数: {sum(bleu_scores) / len(bleu_scores):.4f}")
    print(f"平均ROUGE-L分数: {sum(rouge_scores) / len(rouge_scores):.4f}")


if __name__ == "__main__":
    evaluate_with_metrics()
