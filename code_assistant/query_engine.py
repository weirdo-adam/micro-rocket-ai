from llama_index import Settings
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
import logging

logger = logging.getLogger(__name__)

class CodeQueryEngine:
    def __init__(self, indexer, model_name="gpt-3.5-turbo", temperature=0.2):
        self.indexer = indexer

        # 设置嵌入模型和LLM
        Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        Settings.llm = OpenAI(model=model_name, temperature=temperature)

        if self.indexer.index is None:
            raise ValueError("索引尚未初始化，请先运行index_code_directory方法")

        self.query_engine = self.indexer.index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )

    def query(self, query_text):
        """查询代码库"""
        logger.info(f"处理查询: {query_text}")
        response = self.query_engine.query(query_text)
        return response
