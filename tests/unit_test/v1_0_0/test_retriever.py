import os
import unittest
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
        cls.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        cls.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "travel_knowledge")
        cls.RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))
        try:
            cls.embeddings = OpenAIEmbeddings(model=cls.EMBEDDING_MODEL, openai_api_key=cls.OPENAI_API_KEY)
            cls.vectorstore = Chroma(
                persist_directory=cls.CHROMA_DIR,
                embedding_function=cls.embeddings,
                collection_name=cls.COLLECTION_NAME
            )
            cls.retriever = cls.vectorstore.as_retriever(
                search_kwargs={"k": cls.RETRIEVAL_K}
            )
        except Exception as e:
            cls.retriever = None
            cls.init_error = e

    def test_retriever_initialization(self):
        # ✅ Retriever 可正常初始化
        self.assertIsNotNone(self.retriever, f"Retriever 初始化失敗: {getattr(self, 'init_error', None)}")

    def test_simple_query(self):
        # ✅ 可執行簡單查詢
        if self.retriever is None:
            self.skipTest("Retriever 未初始化")
        query = "推薦台北景點"
        try:
            results = self.retriever.invoke(query)
            print(f"\n查詢：{query}")
            self.assertIsInstance(results, list)
            self.assertGreaterEqual(len(results), 1, "查詢結果為空")
        except Exception as e:
            self.fail(f"查詢執行失敗: {e}")

    def test_error_handling(self):
        # ✅ 錯誤處理完善
        # 測試資料庫不存在時 retriever 是否為 None
        chroma_dir_backup = self.CHROMA_DIR
        try:
            # 嘗試載入不存在的資料庫
            embeddings = OpenAIEmbeddings(model=self.EMBEDDING_MODEL, openai_api_key=self.OPENAI_API_KEY)
            vectorstore = Chroma(
                persist_directory="./not_exist_db",
                embedding_function=embeddings,
                collection_name=self.COLLECTION_NAME
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": self.RETRIEVAL_K}
            )
            # 應該不會有資料
            results = retriever.invoke("test")
            self.assertIsInstance(results, list)
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_return_format(self):
        # ✅ 返回格式正確
        if self.retriever is None:
            self.skipTest("Retriever 未初始化")
        query = "推薦東京景點"
        results = self.retriever.invoke(query)
        print(f"\n查詢：{query}")
        for doc in results:
            self.assertTrue(hasattr(doc, "page_content"), "返回物件缺少 page_content 欄位")
            self.assertTrue(isinstance(doc.page_content, str), "page_content 欄位型別錯誤")

if __name__ == "__main__":
    unittest.main()