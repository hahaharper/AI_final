import os
import json
import re
import pandas as pd
from tqdm import tqdm
from autogen import ConversableAgent
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# === 設定 ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JSON_FOLDER = "/home/harper112/AI_final/regulation"
CHROMA_DB_DIR = "/home/harper112/AI_final/rag_db"
CSV_PATH = "/home/harper112/AI_final/final_project_query.csv"
OUTPUT_PATH = "/home/harper112/AI_final/out.csv"
LLM_MODEL = "gpt-4o-mini"

LLM_CFG = {"config_list": [{"model": LLM_MODEL, "api_key": OPENAI_API_KEY}]}

# === Info Extraction Agent ===
INFO_EXTRACTION_AGENT = ConversableAgent(
    name="info_extraction_agent",
    llm_config=LLM_CFG,
    system_message=(
    "You are a shopping assistant that extracts structured product data from raw product descriptions.\n"
    "Return a valid JSON object with:\n"
    "- product_name (string)\n- ad_description (string)\n"
    "Strictly ensure the JSON is properly formatted. Do not include any explanation, markdown, or extra text. Only return valid JSON like:\n"
    '{"product_name": "...", "ad_description": "..."}'
    )
)



# === 自定義 JSON Loader ===
class JsonLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return [Document(page_content=text, metadata={"source": self.file_path})]

# === 建立向量資料庫 ===
def build_vector_db(folder_path: str, persist_dir: str) -> Chroma:
    loader = DirectoryLoader(folder_path, glob="**/*.json", loader_cls=JsonLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    return vectordb

# === Prompt 設定 ===
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
### 廣告內容
{question}

### 法規比對摘要
{context}

## 回應格式
違法機率: X% 
違法內容分析: ... 
罰款額度: ... 
參考依據: [...]
"""
)

# === 執行 RAG 評估 ===
def detect_legality_with_rag(ad_text: str, vectordb: Chroma):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=LLM_MODEL),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    result = chain({"query": ad_text})
    return result["result"]

# === 抽取 violation_score ===
def extract_violation_score(response: str) -> float:
    match = re.search(r"違法機率[:：]\s*(\d+)%", response)
    if match:
        return float(match.group(1)) / 100
    return None

# === 判定是否合格 ===
def check_qualification(score: float) -> int:
    return 1 if score > 0.6 else 0

# === 安全呼叫 Agent 回應 ===
def safe_generate_reply(agent, messages, max_retries=5):
    import time
    for i in range(max_retries):
        try:
            return agent.generate_reply(messages)
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = 0.5 + i
                print(f"⚠️ Rate limit hit, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise

# === 抽取欄位值 ===
def extract_value_from_str_dict(text: str, field: str):
    try:
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"```$", "", text.strip())
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        data = json.loads(text)
        return data.get(field, None)
    except Exception as e:
        print("❌ JSON parse failed:", e)
        return None

# === 主程式 ===
def main():
    print("[1] 建立向量資料庫...")
    vectordb = build_vector_db(JSON_FOLDER, CHROMA_DB_DIR)

    print("[2] 讀取 CSV...")
    df = pd.read_csv(CSV_PATH)
    results = []

    print("[3] 開始處理...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
            query = str(row["Question"])
            info_response = safe_generate_reply(INFO_EXTRACTION_AGENT, [
                {"role": "user", "content": f"Extract product information from the following query:\n{query}"}
            ])

            product_name = extract_value_from_str_dict(info_response, "product_name")
            ad_text = extract_value_from_str_dict(info_response, "ad_description")
            if not ad_text:
                print(f"第 {idx} 筆缺 ad_description，猜測違規")
                results.append({"ID": idx, "Answer": 1})
                continue
            if not ad_text:
                print(f"第 {idx} 筆缺 ad_description，略過。")
                continue
            # 自動翻譯為中文
        # if not re.search(r'[\u4e00-\u9fff]', ad_text):  # 如果沒有中文字
        #     ad_text = translate_to_chinese(ad_text, INFO_EXTRACTION_AGENT)

            rag_response = detect_legality_with_rag(ad_text, vectordb)
            score = extract_violation_score(rag_response)
            if score is None:
                print(f"無法擷取違法機率（第 {idx} 筆），猜測違規")
                results.append({"ID": idx, "Answer": 1})
                continue


            label = check_qualification(score)
            results.append({"ID": idx, "Answer": label})

            print("[4] 寫出結果...")
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"完成！結果寫入 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()