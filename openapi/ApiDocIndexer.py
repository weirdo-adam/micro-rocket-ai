import logging
import os
import sys
import glob
from typing import List, Optional, Any

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class ApiDocIndexer:
    """API文档索引管理器"""

    def __init__(self, persist_dir: str = "./storage"):
        """
        初始化API文档索引管理器

        Args:
            persist_dir: 索引存储目录
        """
        # 配置日志
        self._setup_logging()

        # 加载环境变量
        load_dotenv()

        # 配置LLM和嵌入模型
        self._setup_models()

        # 设置存储目录
        self.persist_dir = persist_dir
        self.api_docs_dir = os.getenv("OPENAPI_DOCS_DIR")

        # 索引和查询引擎
        self.index = None
        self.query_engine = None

        # 初始化索引
        self._initialize_index()

    def _setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        self.logger = logging.getLogger(__name__)

    def _setup_models(self):
        """配置LLM和嵌入模型"""
        Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text')
        Settings.llm = Ollama(model="deepseek-r1:7b", request_timeout=3600.0)

    def load_api_documents(self, api_docs_dir: str) -> List[Document]:
        """
        加载目录下的所有API文档并返回Document对象列表

        Args:
            api_docs_dir: API文档目录路径

        Returns:
            Document对象列表
        """
        self.logger.info(f"开始加载API文档，目录: {api_docs_dir}")

        if not os.path.exists(api_docs_dir):
            self.logger.error(f"API文档目录不存在: {api_docs_dir}")
            return []

        # 获取目录下所有文件
        all_files = []
        for extension in ['*.json', '*.yaml', '*.yml', '*.md', '*.txt']:
            pattern = os.path.join(api_docs_dir, "**", extension)
            all_files.extend(glob.glob(pattern, recursive=True))

        self.logger.info(f"找到 {len(all_files)} 个文档文件")

        # 使用SimpleDirectoryReader加载文档
        try:
            documents = SimpleDirectoryReader(input_files=all_files).load_data()
            self.logger.info(f"成功加载 {len(documents)} 个文档")
            return documents
        except Exception as e:
            self.logger.error(f"加载文档时出错: {str(e)}")
            return []

    def build_index(self, documents: List[Document]) -> bool:
        """
        从文档构建索引并持久化存储

        Args:
            documents: Document对象列表

        Returns:
            构建是否成功
        """
        if not documents:
            self.logger.error("没有文档可用于构建索引")
            return False

        try:
            self.logger.info("开始构建向量索引...")
            self.index = VectorStoreIndex.from_documents(documents)

            # 持久化存储
            self.logger.info(f"保存索引到 {self.persist_dir}")
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            self.logger.info("索引构建和保存完成")

            # 创建查询引擎
            self.query_engine = self.index.as_query_engine()
            return True
        except Exception as e:
            self.logger.error(f"构建索引时出错: {str(e)}")
            return False

    def load_index(self) -> bool:
        """
        加载现有索引

        Returns:
            加载是否成功
        """
        try:
            self.logger.info(f"从 {self.persist_dir} 加载现有索引...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            self.logger.info("索引加载完成")

            # 创建查询引擎
            self.query_engine = self.index.as_query_engine()
            return True
        except Exception as e:
            self.logger.error(f"加载索引时出错: {str(e)}")
            return False

    def _initialize_index(self):
        """初始化索引，如果不存在则创建新索引"""
        if not os.path.exists(self.persist_dir):
            self.logger.info("未找到现有索引，开始构建新索引...")
            documents = self.load_api_documents(self.api_docs_dir)
            if not self.build_index(documents):
                self.logger.error("初始化索引失败")
        else:
            if not self.load_index():
                self.logger.error("加载现有索引失败")

    def query(self, query_text: str) -> str:
        """
        执行查询

        Args:
            query_text: 查询文本

        Returns:
            查询结果
        """
        if not self.query_engine:
            self.logger.error("查询引擎未初始化")
            return "系统错误：查询引擎未初始化"

        try:
            self.logger.info(f"执行查询: {query_text}")
            response = self.query_engine.query(query_text)
            return str(response)
        except Exception as e:
            self.logger.error(f"查询时出错: {str(e)}")
            return f"查询过程中出现错误: {str(e)}"

    def rebuild_index(self) -> bool:
        """
        重新构建索引

        Returns:
            重建是否成功
        """
        self.logger.info("开始重新构建索引...")
        documents = self.load_api_documents(self.api_docs_dir)
        return self.build_index(documents)

    def stream_query(self, query_text):
        """流式查询方法，返回一个生成器，用于逐步生成响应"""
        # 使用LlamaIndex的流式查询功能
        # 这里假设您使用的是LlamaIndex的最新版本，可能需要根据实际版本调整
        from llama_index.schema import StreamingResponse

        # 获取查询引擎
        query_engine = self.index.as_query_engine(streaming=True)

        # 执行查询并获取流式响应
        streaming_response = query_engine.query(query_text)

        # 如果流式响应是字符串生成器，直接返回
        if isinstance(streaming_response, StreamingResponse):
            for token in streaming_response.get_response_gen():
                yield token
        else:
            # 如果返回的不是流式响应，模拟流式返回
            full_response = str(streaming_response)

            # 按句子分割
            sentences = full_response.split('. ')
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    if i < len(sentences) - 1 and not sentence.endswith('.'):
                        yield sentence + '. '
                    else:
                        yield sentence
