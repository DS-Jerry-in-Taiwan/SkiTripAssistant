import os
import unittest
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

class TestVectorstoreBuild(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.chroma_dir = "chroma_db"
        self.collection_name = "travel_knowledge"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model, openai_api_key=self.openai_api_key)
        self.vectorstore = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def test_vectorstore_exists(self):
        # 檢查 Chroma 資料夾是否存在
        self.assertTrue(os.path.exists(self.chroma_dir), "chroma_db 資料夾不存在")

    def test_vector_count(self):
        # 檢查向量庫內是否有資料
        count = self.vectorstore._collection.count()
        self.assertGreater(count, 0, "向量庫內無資料，請確認已建置完成")

    def test_similarity_search(self):
        # 檢索測試
        query = "滑雪場介紹"
        results = self.vectorstore.similarity_search(query, k=3)
        self.assertGreaterEqual(len(results), 1, "檢索不到相關文件")
        for doc in results:
            self.assertTrue(len(doc.page_content) > 0, "檢索結果內容為空")

    def test_all_documents_retrievable(self):
        # 檢查所有 chunks 是否都能被檢索到
        count = self.vectorstore._collection.count()
        # 取全部向量的 id
        all_ids = self.vectorstore._collection.get(ids=None)["ids"]
        self.assertEqual(len(all_ids), count, "部分 chunks 無法檢索")
        # 隨機抽取一個 id 進行檢索
        if all_ids:
            doc = self.vectorstore._collection.get(ids=[all_ids[0]])["documents"][0]
            self.assertTrue(len(doc) > 0, "檢索到的 chunk 內容為空")

    def test_query_examples(self):
        # 多組查詢驗證
        queries = [
            "預算規劃",
            "交通方式",
            "滑雪貼士",
            "行程範例"
        ]
        for query in queries:
            results = self.vectorstore.similarity_search(query, k=2)
            self.assertGreaterEqual(len(results), 1, f"查詢 '{query}' 無檢索結果")
            for doc in results:
                print(f"查詢 '{query}' 檢索到的內容片段: {doc.page_content[:100]}...")
                self.assertTrue(len(doc.page_content) > 0, f"查詢 '{query}' 結果內容為空")

if __name__ == "__main__":
    unittest.main()