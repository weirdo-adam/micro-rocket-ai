import json
import requests
from openai import OpenAI
import os.path
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)

# API基础URL
BASE_URL = os.getenv("CLOUDPENSE_OPENAPI")

# 定义函数描述
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_file_download_url",
            "description": "获取文件的下载地址",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "需要获取下载地址的文件名列表"
                    }
                },
                "required": ["data"]
            }
        }
    }
]

def get_token():
    """获取API访问令牌"""
    token_url = f"{BASE_URL}/common/unAuth/tokens/get"
    params = {
        "grant_type": "client_credentials",
        "client_id": "cloudpense6427507247",
        "client_secret": "4f068579e99dc1f05603e9546ee1e13141889d1ee91d565ca8764bdaa95bc6ca9960de19"
    }

    response = requests.get(token_url, params=params)
    if response.status_code == 200:
        return response.json()["data"]["access_token"]
    else:
        raise Exception(f"获取token失败: {response.text}")

def get_file_download_url(file_paths):
    """获取文件下载URL的函数实现"""
    # 首先获取token
    token = get_token()

    # 准备请求获取下载URL
    download_url_endpoint = f"{BASE_URL}/common/files/v2/downloadUrl"
    headers = {
        "access_token": token,
        "Content-Type": "application/json"
    }

    payload = {
        "data": file_paths
    }

    print(file_paths)
    # 发送请求获取下载URL
    response = requests.post(download_url_endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"获取文件下载URL失败: {response.text}")

# 用于处理OpenAI函数调用的函数
def process_function_call(message):
    function_call = message.tool_calls[0].function
    function_name = function_call.name
    function_args = json.loads(function_call.arguments)
    if function_name == "get_file_download_url":
        file_paths = function_args.get("data")
        result = get_file_download_url(file_paths)
        return result
    else:
        return {"error": "Unknown function"}

# 示例对话
def run_conversation():
    messages = [{"role": "user", "content": "我需要获取这个文件的下载链接: previewImg/test/4a6cfdf1-4867-423e-9d70-68a4f9f80ace.png"}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=functions,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # 检查是否有函数调用
    if response_message.tool_calls:
        # 获取函数调用结果
        function_response = process_function_call(response_message)

        # 将函数调用和结果添加到消息历史
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": response_message.tool_calls[0].id,
            "content": json.dumps(function_response)
        })

        # 获取AI对函数调用结果的解释
        second_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        return second_response.choices[0].message.content
    else:
        return response_message.content

# 执行对话
if __name__ == "__main__":
    print(run_conversation())
