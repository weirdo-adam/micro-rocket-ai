import logging
import os.path
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# log配置
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 加载.env文件
load_dotenv()

# bge-base embedding model
# 将文本转换成向量(数值表示),用于文本相似度计算、检索等任务。bge-base-en-v1.5 是英文基础版本
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

# ollama
Settings.llm = Ollama(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", request_timeout=3600.0)

# 检查存储是否存在
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # 加载文档并创建索引
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # 持久化存储
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # 加载已存在的索引
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# 创建查询引擎
query_engine = index.as_query_engine()
response = query_engine.query("请使用中文回答，代理人是什么意思?")
print(response)
