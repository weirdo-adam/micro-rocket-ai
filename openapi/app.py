from ApiDocIndexer import ApiDocIndexer

indexer = ApiDocIndexer()

if __name__ == "__main__":
    query_text = "如何获取文件下载地址"
    result = indexer.query(query_text)
    print(result)
