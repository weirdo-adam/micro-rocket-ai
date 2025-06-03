import json

import requests

# Ollama API 端点
OLLAMA_API_URL = "http://localhost:11434/api/generate"


# 定义可调用的函数
def get_weather(location, unit="celsius"):
    """获取指定位置的天气情况"""
    # 在实际应用中，这里会调用真实的天气 API
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "晴朗",
    }


# 函数定义，将提供给模型
function_definitions = [
    {
        "name": "get_weather",
        "description": "获取指定位置的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、广州",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位",
                },
            },
            "required": ["location"],
        },
    }
]

# 可用函数映射
available_functions = {"get_weather": get_weather}


def call_ollama_with_functions(prompt, model="llama3", functions=None):
    """调用 Ollama 模型并支持函数调用"""

    system_prompt = """你是一个有用的AI助手。
如果需要获取信息，你可以调用提供的函数。
"""

    # 构建请求体
    request_body = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
    }

    # 如果提供了函数定义，则添加到请求中
    if functions:
        request_body["functions"] = functions

    # 发送请求到 Ollama API
    response = requests.post(OLLAMA_API_URL, json=request_body)
    result = response.json()

    # 处理响应
    if "response" in result:
        response_text = result["response"]

        # 检查是否有函数调用请求
        if "tool_calls" in result:
            for tool_call in result["tool_calls"]:
                function_name = tool_call.get("name")
                function_args = json.loads(tool_call.get("arguments", "{}"))

                if function_name in available_functions:
                    # 执行函数调用
                    function_response = available_functions[function_name](
                        **function_args
                    )

                    # 发送函数执行结果回模型
                    follow_up = f"函数 {function_name} 的调用结果：{json.dumps(function_response, ensure_ascii=False)}"
                    second_response = call_ollama_with_functions(
                        f"{prompt}\n\n{response_text}\n\n{follow_up}",
                        model=model,
                    )
                    return second_response

        return response_text

    return "模型响应出错"


# 测试函数调用
if __name__ == "__main__":
    user_query = "北京今天的天气如何？"
    response = call_ollama_with_functions(
        user_query, model="deepseek-r1:7b", functions=function_definitions
    )
    print(f"用户问题: {user_query}")
    print(f"AI回答: {response}")
