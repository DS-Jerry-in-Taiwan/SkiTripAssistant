import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import unittest
from langchain_text_splitters import MarkdownHeaderTextSplitter

class TestChunkSemantics(unittest.TestCase):
    def setUp(self):
        # 範例文本，模擬旅遊知識文件
        self.text = """
        # 日本滑雪場介紹
        # 北海道地區】
        介紹北海道主要滑雪場與特色。

        # 交通資訊】
        說明如何前往各大滑雪場。
        """
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "header")], strip_headers=False)

    def test_chunk_semantics(self):
        chunks = self.splitter.split_text(self.text)
        print("Generated Chunks:", chunks)
        # 測試每個 chunk 是否包含主題標記
        for chunk in chunks:
            print("Chunk:", chunk.page_content)
            self.assertTrue(len(chunk.page_content.strip()) > 0)
        self.assertTrue(any("北海道" in chunk.page_content for chunk in chunks))
        self.assertTrue(any("交通" in chunk.page_content for chunk in chunks))
        self.assertTrue(len(chunk.page_content) > 0)
        # 測試 chunk 數量是否合理
        self.assertGreaterEqual(len(chunks), 2)

if __name__ == "__main__":
    unittest.main()