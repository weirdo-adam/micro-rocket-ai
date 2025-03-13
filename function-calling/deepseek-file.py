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
                    "file_paths": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "需要获取下载地址的文件路径列表"
                    },
                    "company_id": {
                        "type": "integer",
                        "description": "公司ID"
                    },
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID，默认为0"
                    }
                },
                "required": ["file_paths", "company_id"]
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

def get_file_download_url(file_paths, company_id, user_id=0):
    """获取文件下载URL的函数实现"""
    # 首先获取token
    token = get_token()

    # 准备请求获取下载URL
    download_url_endpoint = f"{BASE_URL}/sc-file/fileService/files/getDownloadUrl"
    headers = {
        "access_token": token,
        "Content-Type": "application/json"
    }
    params = {
        "companyId": company_id,
        "userId": user_id
    }

    # 发送请求获取下载URL
    response = requests.post(download_url_endpoint, headers=headers, params=params, json=file_paths)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"获取文件下载URL失败: {response.text}")

# 用于处理OpenAI函数调用的函数
def process_function_call(message):
    function_call = message.tool_calls[0].function
    function_name = function_call.name
    function_args = json.loads(function_call.arguments)
    print(function_args)
    if function_name == "get_file_download_url":
        file_paths = function_args.get("file_paths")
        company_id = function_args.get("company_id")
        user_id = function_args.get("user_id", 0)

        result = get_file_download_url(file_paths, company_id, user_id)
        return result
    else:
        return {"error": "Unknown function"}

# 示例对话
def run_conversation():
    messages = [{"role": "user", "content": "我需要获取这个文件的下载链接: previewImg/test/4a6cfdf1-4867-423e-9d70-68a4f9f80ace.png，公司ID是3809"}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=functions,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    print(response_message)

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
