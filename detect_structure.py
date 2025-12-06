import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
DOCS_DIR = os.getenv("DOCS_DIR", "./documents")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "travel_knowledge")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def detect_structure(text):
    # 判斷是否有 Markdown 標題
    if re.search(r"^#{1,3} ", text, re.MULTILINE):
        return "markdown"
    # 判斷是否有特殊主題符號（如【主題】）
    if re.search(r"^【[^】]+】", text, re.MULTILINE):
        return "custom_header"
    # 判斷是否為長自然語言段落
    if len(text) > 1000:
        return "long_text"
    return "plain"

def chunk_by_strategy(text):
    strategy = detect_structure(text)
    if strategy == "markdown":
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "header"), ("##", "subheader")], strip_headers=False)
        return splitter.split_text(text)
    elif strategy == "custom_header":
        # 正則分割，保留主題標題
        pattern = r"(?=\n【[^】]+】)"
        chunks = re.split(pattern, text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    elif strategy == "long_text":
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        # 需包裝成 Document 物件
        from langchain_core.documents import Document
        doc = Document(page_content=text)
        return splitter.split_documents([doc])
    else:
        # 預設用固定長度分割
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        from langchain_core.documents import Document
        doc = Document(page_content=text)
        return splitter.split_documents([doc])

def main():
    print("載入文件...")
    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=lambda path: TextLoader(path, encoding="utf-8"))
    documents = loader.load()
    print(f"共載入 {len(documents)} 份文件")

    print("自動判斷並分割文件...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_by_strategy(doc.page_content)
        all_chunks.extend(chunks)
    print(f"分割後共 {len(all_chunks)} 個 chunks")

    print("初始化 OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    print("建立 Chroma 向量資料庫並持久化...")
    vectorstore = Chroma.from_documents(
        all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist()
    print(f"已完成向量資料庫建置，資料儲存於 {CHROMA_DIR}/")

if __name__ == "__main__":
    main()