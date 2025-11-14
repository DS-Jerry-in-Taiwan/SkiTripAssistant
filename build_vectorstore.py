import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 讀取 .env 檔案
load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./documents")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "travel_knowledge")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("載入文件...")
    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=lambda path: TextLoader(path, encoding="utf-8"))
    documents = loader.load()
    print(f"共載入 {len(documents)} 份文件")

    print("切割文件...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(documents)
    print(f"切割後共 {len(docs)} 個 chunks")

    print("初始化 OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    print("建立 Chroma 向量資料庫並持久化...")
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist()
    print(f"已完成向量資料庫建置，資料儲存於 {CHROMA_DIR}/")

if __name__ == "__main__":
    main()