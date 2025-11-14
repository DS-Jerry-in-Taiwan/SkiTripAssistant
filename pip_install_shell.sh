#!/bin/bash

# 建議先啟動虛擬環境
echo "請確認已啟動 Python 虛擬環境！"

# 安裝核心套件
pip install langchain langgraph langchain-openai langchain-community chromadb tiktoken python-dotenv

# 產生 requirements.txt
cat <<EOT > requirements.txt
langchain
langgraph
langchain-openai
langchain-community
chromadb
tiktoken
python-dotenv
EOT

echo "所有套件安裝完成，requirements.txt 已產生！"
