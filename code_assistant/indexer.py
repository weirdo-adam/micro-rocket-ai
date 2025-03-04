import os
from typing import List, Optional
from llama_index import (
    VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage
)
from llama_index.node_parser import CodeSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeIndexer:
    def __init__(self, persist_dir: str = "./code_index"):
        self.persist_dir = persist_dir
        self.index = None

    def index_code_directory(self, code_dir: str, file_exts: Optional[List[str]] = None, force_reindex: bool = False):
        """对代码目录建立索引"""
        if file_exts is None:
            file_exts = [".py", ".js", ".java", ".cpp", ".c", ".h", ".html", ".css", ".go", ".rs"]

        # 检查是否已有持久化索引
        if os.path.exists(self.persist_dir) and not force_reindex:
            try:
                logger.info(f"尝试加载现有索引: {self.persist_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self.index = load_index_from_storage(storage_context)
                logger.info("成功加载现有索引")
                return
            except Exception as e:
                logger.warning(f"加载索引失败: {e}. 将重新创建索引。")

        # 创建索引目录
        os.makedirs(self.persist_dir, exist_ok=True)

        # 读取文件
        logger.info(f"从目录加载代码文件: {code_dir}")
        documents = SimpleDirectoryReader(
            input_dir=code_dir,
            recursive=True,
            file_extns=file_exts
        ).load_data()

        logger.info(f"加载了 {len(documents)} 个文档")

        # 使用代码分割器
        code_splitter = CodeSplitter(
            language="python",
            chunk_lines=100,
            chunk_overlap_lines=20,
            max_chars=4000
        )

        # 创建和保存索引
        logger.info("创建索引...")
        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=[code_splitter]
        )

        # 持久化索引
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info(f"索引已保存到: {self.persist_dir}")
