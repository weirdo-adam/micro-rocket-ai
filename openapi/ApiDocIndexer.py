import logging
import os
import sys
import glob
from typing import List, Optional, Any, Dict, Tuple

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document,
    ServiceContext,
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import StreamingResponse


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

        # 索引和查询引擎
        self.index = None
        self.query_engine = None
        self.nodes = None

        # 设置存储目录
        self.persist_dir = persist_dir
        self.api_docs_dir = os.getenv("OPENAPI_DOCS_DIR")

        # 配置LLM和嵌入模型
        self._setup_models()

        # 初始化索引
        self._initialize_index()

    def _setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        self.logger = logging.getLogger(__name__)

    def _setup_models(self):
        """配置LLM、嵌入模型和服务上下文"""
        # 使用更高质量的嵌入模型
        embed_model = OllamaEmbedding(model_name='nomic-embed-text')
        llm = Ollama(model="deepseek-r1:7b", request_timeout=3600.0)

        # 创建ServiceContext以便统一管理
        self.service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            # 增加上下文窗口提高回答质量
            context_window=4096,
            # 增加生成文本的多样性
            temperature=0.2
        )

        # 配置全局设置
        Settings.embed_model = embed_model
        Settings.llm = llm

    def _create_text_splitter(self):
        """创建优化的文本分割器"""
        # 使用语义化分块，保持API文档的上下文连贯性
        return SentenceSplitter(
            chunk_size=512,  # 适合API文档的分块大小
            chunk_overlap=100,  # 增加重叠以保持上下文
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.。!?！？]",  # 支持中英文语句边界
            include_metadata=True,
            metadata_keys=["filename", "document_type"]  # 保留文件信息
        )

    def load_api_documents(self, api_docs_dir: str) -> Tuple[List[Document], List]:
        """
        加载目录下的所有API文档并返回Document对象列表和处理后的节点

        Args:
            api_docs_dir: API文档目录路径

        Returns:
            包含Document对象列表和处理后节点的元组
        """
        self.logger.info(f"开始加载API文档，目录: {api_docs_dir}")

        if not os.path.exists(api_docs_dir):
            self.logger.error(f"API文档目录不存在: {api_docs_dir}")
            return [], []

        # 获取目录下所有文件
        all_files = []
        for extension in ['*.json', '*.yaml', '*.yml', '*.md', '*.txt']:
            pattern = os.path.join(api_docs_dir, "**", extension)
            all_files.extend(glob.glob(pattern, recursive=True))

        self.logger.info(f"找到 {len(all_files)} 个文档文件")

        # 使用SimpleDirectoryReader加载文档
        try:
            # 为文档增加元数据
            documents = []
            for file_path in all_files:
                file_type = os.path.splitext(file_path)[1].lower()
                file_name = os.path.basename(file_path)

                doc_type = "unknown"
                if file_type in ['.json', '.yaml', '.yml']:
                    doc_type = "api_spec"
                elif file_type == '.md':
                    doc_type = "markdown"
                elif file_type == '.txt':
                    doc_type = "text"

                # 使用读取器加载单个文件
                file_docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

                # 为每个文档添加元数据
                for doc in file_docs:
                    doc.metadata.update({
                        "filename": file_name,
                        "document_type": doc_type,
                        "file_path": file_path
                    })
                    documents.append(doc)

            self.logger.info(f"成功加载 {len(documents)} 个文档")

            # 使用优化的分块策略处理文档
            text_splitter = self._create_text_splitter()
            nodes = text_splitter.get_nodes_from_documents(documents)
            self.logger.info(f"文档分块完成，生成了 {len(nodes)} 个节点")

            return documents, nodes
        except Exception as e:
            self.logger.error(f"加载文档时出错: {str(e)}")
            return [], []

    def build_index(self, documents: List[Document], nodes: List) -> bool:
        """
        从文档构建索引并持久化存储

        Args:
            documents: Document对象列表
            nodes: 处理后的文档节点

        Returns:
            构建是否成功
        """
        if not documents or not nodes:
            self.logger.error("没有文档可用于构建索引")
            return False

        try:
            self.logger.info("开始构建向量索引...")
            self.nodes = nodes

            # 使用优化的service_context来构建索引
            self.index = VectorStoreIndex(
                nodes,
                service_context=self.service_context
            )

            # 持久化存储
            self.logger.info(f"保存索引到 {self.persist_dir}")
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            self.logger.info("索引构建和保存完成")

            # 创建优化的查询引擎
            self._create_optimized_query_engine()
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
            self.index = load_index_from_storage(
                storage_context,
                service_context=self.service_context
            )
            self.logger.info("索引加载完成")

            # 加载节点以便使用混合检索
            self._load_nodes()

            # 创建优化的查询引擎
            self._create_optimized_query_engine()
            return True
        except Exception as e:
            self.logger.error(f"加载索引时出错: {str(e)}")
            return False

    def _load_nodes(self):
        """从索引中加载节点"""
        try:
            # 从索引中获取所有节点
            if self.index:
                self.nodes = self.index.docstore.get_all_nodes()
                self.logger.info(f"从索引中加载了 {len(self.nodes)} 个节点")
            else:
                self.logger.warning("索引未初始化，无法加载节点")
        except Exception as e:
            self.logger.error(f"加载节点时出错: {str(e)}")

    def _create_optimized_query_engine(self):
        """创建优化的查询引擎，使用混合检索策略和重排序"""
        if not self.index or not self.nodes:
            self.logger.error("索引或节点未初始化，无法创建查询引擎")
            return

        try:
            self.logger.info("创建优化的查询引擎...")

            # 1. 创建向量检索器
            vector_retriever = self.index.as_retriever(
                similarity_top_k=10,  # 检索更多候选结果以提高召回率
            )

            # 2. 创建BM25检索器
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.nodes,
                similarity_top_k=10,
            )

            # 3. 融合两种检索策略
            fusion_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=8,  # 最终选择的结果数量
                use_orig_query=True,  # 使用原始查询
                mode="reciprocal_rank_fusion",  # 使用RRF融合算法
            )

            # 4. 添加重排序处理器
            # 注意：如果您使用Ollama，可能需要自定义重排序器或使用其他兼容的重排序方案
            # 下面使用的是SentenceTransformer重排序器，如果不可用请替换为兼容方案
            try:
                reranker = SentenceTransformerRerank(
                    model_name="BAAI/bge-reranker-base",  # 轻量级重排序模型
                    top_n=5  # 保留前5个最相关结果
                )

                # 5. 创建响应合成器
                response_synthesizer = get_response_synthesizer(
                    service_context=self.service_context,
                    response_mode="compact"  # 使用紧凑模式生成响应
                )

                # 6. 构建查询引擎
                self.query_engine = RetrieverQueryEngine.from_args(
                    retriever=fusion_retriever,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=[reranker],
                    service_context=self.service_context
                )

                self.logger.info("优化的查询引擎创建完成")
            except ImportError:
                # 如果重排序器不可用，退回到基本引擎
                self.logger.warning("重排序器不可用，使用基本引擎")
                self.query_engine = RetrieverQueryEngine.from_args(
                    retriever=fusion_retriever,
                    service_context=self.service_context
                )
        except Exception as e:
            self.logger.error(f"创建查询引擎时出错: {str(e)}")
            # 回退到基本查询引擎
            self.query_engine = self.index.as_query_engine()

    def _initialize_index(self):
        """初始化索引，如果不存在则创建新索引"""
        if not os.path.exists(self.persist_dir):
            self.logger.info("未找到现有索引，开始构建新索引...")
            documents, nodes = self.load_api_documents(self.api_docs_dir)
            if not self.build_index(documents, nodes):
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

            # 预处理查询，添加提示以提高API文档相关性
            enhanced_query = self._enhance_query(query_text)

            # 执行查询
            response = self.query_engine.query(enhanced_query)

            # 记录查询结果和源节点（用于调试）
            if hasattr(response, 'source_nodes') and response.source_nodes:
                sources = [node.node.metadata.get('filename', 'unknown')
                          for node in response.source_nodes]
                self.logger.info(f"查询命中的文档源: {sources}")

            return str(response)
        except Exception as e:
            self.logger.error(f"查询时出错: {str(e)}")
            return f"查询过程中出现错误: {str(e)}"

    def _enhance_query(self, query_text: str) -> str:
        """增强查询以提高API文档检索相关性"""
        # 对于API文档检索，可以添加一些提示信息
        if not any(keyword in query_text.lower() for keyword in
                  ['api', 'endpoint', '接口', '参数', 'parameter', 'request', '请求']):
            # 如果查询中没有明确的API相关关键词，可能需要增强
            return f"关于API的以下问题: {query_text}"
        return query_text

    def rebuild_index(self) -> bool:
        """
        重新构建索引

        Returns:
            重建是否成功
        """
        self.logger.info("开始重新构建索引...")
        documents, nodes = self.load_api_documents(self.api_docs_dir)
        return self.build_index(documents, nodes)

    def stream_query(self, query_text: str):
        """
        流式查询方法，返回一个生成器，用于逐步生成响应

        Args:
            query_text: 查询文本

        Yields:
            响应的文本片段
        """
        if not self.index:
            self.logger.error("索引未初始化，无法执行流式查询")
            yield "系统错误：索引未初始化"
            return

        try:
            # 增强查询
            enhanced_query = self._enhance_query(query_text)

            # 创建流式查询引擎
            streaming_query_engine = self.index.as_query_engine(
                streaming=True,
                similarity_top_k=5,
                service_context=self.service_context
            )

            # 执行查询
            self.logger.info(f"执行流式查询: {enhanced_query}")
            streaming_response = streaming_query_engine.query(enhanced_query)

            # 处理流式响应
            if hasattr(streaming_response, "get_response_gen"):
                # 标准的StreamingResponse对象
                for token in streaming_response.get_response_gen():
                    yield token
            else:
                # 如果不是流式响应，模拟流式效果
                full_response = str(streaming_response)

                # 按句子分割
                import re
                sentences = re.split(r'(?<=[.!?。！？])\s+', full_response)

                for sentence in sentences:
                    if sentence.strip():
                        yield sentence + " "
        except Exception as e:
            self.logger.error(f"流式查询时出错: {str(e)}")
            yield f"查询过程中出现错误: {str(e)}"
