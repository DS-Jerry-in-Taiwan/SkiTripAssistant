FROM python:3.11-slim

# 安裝系統依賴與清理
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential vim && \
    rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝 Python 套件
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製專案檔案（排除 .git 及 cache）
COPY . .

# 設定預設啟動指令
CMD ["python", "main.py"]