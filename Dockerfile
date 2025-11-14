FROM python:3.11-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential vim \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 預設啟動 CLI 主程式
# CMD ["python", "main.py"]