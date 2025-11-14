import os
import tiktoken
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load enviroment variables from .env file
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "travel_knowledge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.7"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
SYSTEM_PROMPT = (
    "你是一位專業旅遊規劃助理，請務必優先根據下方文件內容，針對使用者問題進行整理、摘要與彙整，不要直接貼出原文。\n"
    "請參考整個對話歷史，維持上下文連貫性，針對使用者持續的需求給出回應。\n"
    "若文件資訊不足，請用旅遊專業知識補充，並說明資料來源非文件庫。\n"
    "請避免重複前一輪回應，針對新問題給出新建議。\n"
    "若已無新資訊可補充，請主動告知使用者，並引導其詢問其他主題。\n"
    "請確保回應簡潔、友善、實用，並使用繁體中文。"
)

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

# Initialize the Chroma vector store
try:
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
except Exception as e:
    print(f"Error initializing Chroma vector store: {e}")
    vector_store = None

# retriever setup
RETRIEVER = None
if vector_store:
    RETRIEVER = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )
    print("Retriever successfully created.")    
else:
    print("Vector store is not initialized; retriever cannot be created.")

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """ Count the number of tokens in a given text for a specified model. """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_model_token_limit(model_name:str) -> int:
    """Get the token limit for a specified model"""
    model_token_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
    }
    return model_token_limits.get(model_name, 4096)

def rag_retrieval_node(state: Dict[str, Any], threshold: float = SCORE_THRESHOLD) -> Dict[str, Any]:
    """ 
    RAG Retrieval Node
    Retrieves relevant documents using the pre-configured retriever.
    """
    # Extract the latest user message as the query
    user_message = [msg for msg in state.get("messages", []) if msg.get("role") == "user"]
    if not user_message:
        return {"retrieved_docs": []}
    query = user_message[-1].get("content", "").strip()
    if not query:
        return {"retrieved_docs": []}
    
    # operate retrieve from the vector store
    try:
        results_with_score = vector_store.similarity_search_with_score(query, k=RETRIEVAL_K) if vector_store else []
        # 打印分數
        # for i, (doc, score) in enumerate(results_with_score, 1):
        #     print(f"結果 {i}: 分數={score:.4f}, 內容={doc.page_content[:40]}...")
        filtered_docs_scores = [(doc, score) for doc, score in results_with_score if score <= threshold]
        filtered_docs = [doc for doc, score in filtered_docs_scores]
        # return {"retrieved_docs": filtered_docs}
        # no result handling
        if not filtered_docs:
            return {"retrieved_docs": []}
        return {"retrieved_docs": filtered_docs,
                "retrieved_docs_with_score": filtered_docs_scores
                }
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"retrieved_docs": []}

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_model_token_limit(model_name: str = "gpt-3.5-turbo") -> int:
    model_token_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
    }
    return model_token_limits.get(model_name, 4096)

def truncate_prompt_by_token(prompt_messages: list, model_name: str = "gpt-3.5-turbo") -> list:
    token_limit = get_model_token_limit(model_name)
    total_tokens = 0
    truncated = []
    # 順序保留 system、文件資訊
    for msg in prompt_messages[:2]:
        msg_tokens = count_tokens(msg["content"], model_name)
        total_tokens += msg_tokens
        truncated.append(msg)
    # 倒序加入歷史訊息，直到達到限制
    history = prompt_messages[2:]
    for msg in reversed(history):
        msg_tokens = count_tokens(msg["content"], model_name)
        if total_tokens + msg_tokens > token_limit:
            truncated.insert(2, {"role": "system", "content": "（部分對話已省略）"})
            break
        truncated.insert(2, msg)
        total_tokens += msg_tokens
    return truncated[:2] + list(reversed(truncated[2:]))

def format_retrieval_node(state: Dict[str, Any], max_length: int = 1200,top_k: int = 5) -> Dict[str, Any]:
    """
    format retrieved documents in state. Add title and handling too long content.
    """
    docs_with_score = state.get("retrieved_docs_with_score", [])
    if not docs_with_score and state.get("retrieved_docs",[]):
        # if no score info, create dummy scores
        docs_with_score = [(doc, 1.0) for doc in state.get("retrieved_docs",[])]
    if not docs_with_score:
        return "（目前無相關文件，請用通用知識回答）"
    # Accendding sort by score
    sorted_docs = sorted(docs_with_score, key=lambda x: x[1])
    formatted_list = []
    total_length = 0
    for idx, (doc, score) in enumerate(sorted_docs[:top_k], 1):
        title = doc.metadata.get("title", f"document{idx}")
        content = doc.page_content.strip()
        # Handle too long content
        if len(content) > 400:
            content = content[:400] + "...content too long, truncated."
        doc_text = f"【{title}】(score={score:.3f})\n{content}\n---"
        if total_length + len(doc_text) > max_length:
            formatted_list.append("...additional content truncated due to length limit.")
            break
        formatted_list.append(doc_text)
        total_length += len(doc_text)
        
    return "\n".join(formatted_list)

def build_prompt(state: Dict[str, Any]) -> list:
    """
    merge system prompt and chat history to build the prompt for LLM
    """
    docs_text = format_retrieval_node(state)
    last_user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    user_focus = f"本輪問題重點：{last_user_msg}" if last_user_msg else ""
    recent_msgs = [msg["content"] for msg in state.get("messages", []) if msg.get("role") == "user"][-2:]
    context_summary = "對話摘要：" + " / ".join(recent_msgs) if recent_msgs else ""
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"以下是參考文件：\n{docs_text}\n{user_focus}\n{context_summary}"}
    ]
    # add full chat history (user/assistant), latest message at the end
    prompt_messages.extend(state.get("messages", []))
    return prompt_messages

def call_model_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call LLM model to generate response based on the built prompt
    """
    try:
        # merge system prompt and the formated retrieved documnets
        prompt_messages = build_prompt(state)
        prompt_messages = truncate_prompt_by_token(prompt_messages, model_name="gpt-3.5-turbo")
         
        # Initialize the LLM model
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.2)

        # Call the model
        response = llm.invoke(prompt_messages)
        ai_content = response.content if hasattr(response, "content") else str(response)
        
        # update state with the assistant message
        state["messages"].append({"role": "assistant", "content": ai_content})
        return state
    except Exception as e:
        print(f"Error calling LLM model: {e}")
        state["error_message"] = "抱歉，處理您的請求時發生錯誤。"
        return state

            