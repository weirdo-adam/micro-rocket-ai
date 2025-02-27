from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from langchain.prompts import PromptTemplate


def chain_example():
    # 创建提示模板
    prompt = PromptTemplate(
        input_variables=["topic"], template="请给我一个关于{topic}的简短介绍"
    )

    # 创建链
    llm = OllamaLLM(model="deepseek-r1:7b")
    chain = chain = prompt | llm | StrOutputParser()

    # 执行链
    response = chain.invoke({"topic": "人工智能"})
    print(response)


if __name__ == "__main__":
    print("\n链式调用示例:")
    chain_example()
