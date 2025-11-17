import requests

url = "https://api.tavily.com/search"
headers = {"Content-Type": "application/json"}
params = {
    "query": "日本 滑雪 推薦",
    "api_key": "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",
    "location": "",
    "limit": 5
}

response = requests.post(url, json=params, headers=headers)
print("Tavily API Response Status Code:", response.status_code)
data = response.json()
print("Tavily API Search Results:")
# 整理資料，僅顯示主要欄位
for item in data.get("results", []):
    title = item.get("title", "")
    description = item.get("content", "")
    url = item.get("url", "")
    score = item.get("score", "")
    print(f"- {title}\n  描述: {description}\n  網址: {url}\n  分數: {score}\n")