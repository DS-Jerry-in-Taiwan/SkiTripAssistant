import os

DOCS_DIR = "documents"
summary_len = 50
total_words = 0

def get_file_summary(filepath):
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            word_count = len(content)
            summary = content[:summary_len].replace('\n', ' ')
            return word_count, summary
    except UnicodeDecodeError:
        return None, "❌ 編碼錯誤（非 UTF-8）"

def main():
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    print(f"文件數量：{len(files)}\n")
    for fname in files:
        fpath = os.path.join(DOCS_DIR, fname)
        word_count, summary = get_file_summary(fpath)
        if word_count is not None:
            total_words = word_count
            print(f"{fname} | 字數：{word_count} | 前 50 字：{summary}")
        else:
            print(f"{fname} | {summary}")
    print(f"\n所有文件總字數：{total_words}")

if __name__ == "__main__":
    main()