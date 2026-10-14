import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge_base.config import settings
from knowledge_base.vector_store import load_vector_store

# 动态获取项目根目录 (当前文件在 src/knowledge_base/ 下，向上找3层就是 guithub/)
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

def ingest_document():
    """加载 data/raw 下的文档，切分并写入向量数据库"""
    raw_dir =  settings.raw_dir

    # 检查目录是否存在
    if not raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
        print(f"创建目录 {raw_dir}，请放入文档后重新运行。")
        return

    # 1.加载文档（这里以读取 .txt 文件为例）
    print(f"正在从 {raw_dir} 加载文档...")
    loader = DirectoryLoader(
        raw_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if not documents:
        print("未在 data/raw 下找到任何 .txt 文档，请放入文档后再试。")
        return

    print(f"成功加载 {len(documents)} 个原始文档。")

    # 2.切分文档（使用 config.py 中的配置）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档已切分为 {len(chunks)} 个片段。")

    # 3.写入 Chroma 向量数据库
    print("正在将片段写入向量数据库（首次运行会下载本地 Embedding 模型，请耐心等待）...")
    vector_store = load_vector_store()
    vector_store.add_documents(chunks)

    print("✅ 知识库构建完成！数据已写入向量数据库。")

