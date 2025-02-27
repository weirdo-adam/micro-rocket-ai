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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# log配置
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 加载.env文件
load_dotenv()

# 配置OpenAI设置
Settings.llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_API_URL")
)

# 设置embedding模型
Settings.embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_API_URL")
)

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
response = query_engine.query("代理人是什么意思?中文回答")
print(response)
