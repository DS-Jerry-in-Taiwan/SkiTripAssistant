import os
import unittest
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

class TestSkiTravelVectorstore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.chroma_dir = "chroma_db"
        cls.collection_name = "travel_knowledge"
        cls.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        cls.openai_api_key = os.getenv("OPENAI_API_KEY")
        cls.embeddings = OpenAIEmbeddings(model=cls.embedding_model, openai_api_key=cls.openai_api_key)
        cls.vectorstore = Chroma(
            persist_directory=cls.chroma_dir,
            embedding_function=cls.embeddings,
            collection_name=cls.collection_name
        )

    def print_results(self, query, results):
        print(f"\n查詢：{query}")
        for i, doc in enumerate(results, 1):
            print(f"結果 {i}: {doc.page_content[:80]}...")

    def test_vectorstore_exists(self):
        self.assertTrue(os.path.exists(self.chroma_dir), "❌ chroma_db 資料夾不存在")

    def test_vector_count(self):
        count = self.vectorstore._collection.count()
        self.assertGreater(count, 0, "❌ 向量庫內無資料，請確認已建置完成")

    def test_query_japan_ski_resorts(self):
        query = "推薦日本滑雪場"
        results = self.vectorstore.similarity_search(query, k=3)
        self.print_results(query, results)
        self.assertGreaterEqual(len(results), 1, "❌ 查詢『推薦日本滑雪場』無檢索結果")
        found = any("日本滑雪場" in doc.page_content or "二世谷" in doc.page_content for doc in results)
        self.assertTrue(found, "❌ 未檢索到 japan_ski_resorts.txt 相關內容")

    def test_query_ski_travel_tips(self):
        query = "滑雪旅遊貼士"
        results = self.vectorstore.similarity_search(query, k=3)
        self.print_results(query, results)
        self.assertGreaterEqual(len(results), 1, "❌ 查詢『滑雪旅遊貼士』無檢索結果")
        found = any("貼士" in doc.page_content or "裝備" in doc.page_content for doc in results)
        self.assertTrue(found, "❌ 未檢索到 ski_travel_tips.txt 相關內容")

    def test_query_ski_budget_guide(self):
        query = "滑雪預算怎麼規劃"
        results = self.vectorstore.similarity_search(query, k=3)
        self.print_results(query, results)
        self.assertGreaterEqual(len(results), 1, "❌ 查詢『滑雪預算怎麼規劃』無檢索結果")
        found = any("預算" in doc.page_content or "費用" in doc.page_content for doc in results)
        self.assertTrue(found, "❌ 未檢索到 ski_budget_guide.txt 相關內容")

    def test_query_ski_transportation(self):
        query = "滑雪交通方式"
        results = self.vectorstore.similarity_search(query, k=3)
        self.print_results(query, results)
        self.assertGreaterEqual(len(results), 1, "❌ 查詢『滑雪交通方式』無檢索結果")
        found = any("交通" in doc.page_content or "機場" in doc.page_content for doc in results)
        self.assertTrue(found, "❌ 未檢索到 ski_transportation.txt 相關內容")

    def test_query_ski_itinerary_sample(self):
        query = "滑雪行程範例"
        results = self.vectorstore.similarity_search(query, k=3)
        self.print_results(query, results)
        self.assertGreaterEqual(len(results), 1, "❌ 查詢『滑雪行程範例』無檢索結果")
        found = any("行程" in doc.page_content or "Day" in doc.page_content for doc in results)
        self.assertTrue(found, "❌ 未檢索到 ski_itinerary_sample.txt 相關內容")

    def test_result_count_and_order(self):
        query = "滑雪"
        results = self.vectorstore.similarity_search(query, k=3)
        self.print_results(query, results)
        self.assertEqual(len(results), 3, "❌ 檢索結果數量不正確（應為3）")
        # Chroma similarity_search 預設已排序

if __name__ == "__main__":
    unittest.main()