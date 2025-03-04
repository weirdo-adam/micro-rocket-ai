import argparse
import os
from .indexer import CodeIndexer
from .query_engine import CodeQueryEngine
from .utils import count_code_files
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="代码库AI助手")
    parser.add_argument("--code_dir", type=str, required=True, help="本地代码目录路径")
    parser.add_argument("--index_dir", type=str, default="./code_index", help="索引存储路径")
    parser.add_argument("--reindex", action="store_true", help="强制重新索引")
    parser.add_argument("--query", type=str, help="直接查询，不进入交互模式")

    args = parser.parse_args()

    # 检查代码目录是否存在
    if not os.path.exists(args.code_dir):
        logger.error(f"代码目录不存在: {args.code_dir}")
        return

    # 统计代码文件
    file_count = count_code_files(args.code_dir)
    logger.info(f"找到 {file_count} 个代码文件准备索引")

    # 创建索引
    indexer = CodeIndexer(persist_dir=args.index_dir)
    indexer.index_code_directory(args.code_dir, force_reindex=args.reindex)

    # 初始化查询引擎
    query_engine = CodeQueryEngine(indexer)

    # 处理查询
    if args.query:
        response = query_engine.query(args.query)
        print(f"\n回答: {response.response}\n")
        print("相关源代码:")
        for source_node in response.source_nodes:
            print(f"\n- {source_node.node.metadata['file_name']}:")
            print(f"  {source_node.node.get_content()[:150]}...\n")
    else:
        # 交互模式
        print(f"\n代码库AI助手已准备就绪 - 索引了 {file_count} 个文件")
        print("输入 'exit' 或 'quit' 退出\n")

        while True:
            query = input("请输入您的问题: ")
            if query.lower() in ['exit', 'quit']:
                break

            response = query_engine.query(query)
            print(f"\n回答: {response.response}\n")
            print("相关源代码:")
            for source_node in response.source_nodes:
                print(f"\n- {source_node.node.metadata['file_name']}:")
                print(f"  {source_node.node.get_content()[:150]}...\n")

if __name__ == "__main__":
    main()
