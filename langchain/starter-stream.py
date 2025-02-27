from langchain_community.llms import Ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# 流式输出示例
def streaming_example():
    # 设置流式输出
    llm = Ollama(
        model="deepseek-r1:7b",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    response = llm("讲一个简短的故事")
    return response


if __name__ == "__main__":
    print("\n流式输出示例:")
    result = streaming_example()
    if result:
        print("\n完整响应:", result)
