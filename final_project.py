import pandas as pd
from autogen import ConversableAgent
import os, sys, json, ast, re, time

# ─── Set up LLM config ───
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ─── Agent 1: Extract product info ───
INFO_EXTRACTION_AGENT = ConversableAgent(
    name="info_extraction_agent",
    llm_config=LLM_CFG,
    system_message=(
        "You are a shopping assistant that extracts structured product data from raw product descriptions.\n"
        "For any given query, return a JSON object with:\n"
        "- product_name: the name of the product.\n"
        "- ad_description: the full advertising-style description of the product, including all relevant benefits, features, and use cases.\n\n"
        "Only return a JSON object. Do not include explanations or extra text."
    )
)


# ─── Agent 2: Violation checking ───
VIOLATION_CHECK_AGENT = ConversableAgent(
    name="violation_check_agent",
    llm_config=LLM_CFG,
    system_message=(
        "你是政策審查員，負責分析廣告文案是否可能違反既有規範。\n"
        "你會收到以下資料：\n"
        "- ad_description：產品的完整廣告文案。\n"
        "- reference_cases：以 JSON 格式呈現的既有違規案例清單。\n"
        "你需要判斷該廣告文案是否可能違反其中的規定。\n"
        "請回傳 JSON 物件，包含：\n"
        "violation_score：違規可能性的分數，0 到 1 之間的浮點數。\n"
        "matched_cases：與之相似的違規案例摘要。\n\n"
        "只回傳 JSON 格式，不要加上多餘說明或文字。\n"
        "請用繁體中文輸出。"
    )
)

# ─── Load violation case JSON files ───
def load_violation_cases(folder_path: str) -> list:
    all_cases = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    all_cases.append(data)
                except:
                    continue
    return all_cases

# ─── Call Agent 2 to analyze ad description ───
def analyze_violation(ad_description: str, reference_cases: list):
    input_data = {
        "ad_description": ad_description,
        "reference_cases": reference_cases
    }
    message = "Analyze the following ad for violations:\n\n" + json.dumps(input_data, ensure_ascii=False)
    response = safe_generate_reply(
        agent=VIOLATION_CHECK_AGENT,
        messages=[{"role": "user", "content": message}]
    )

    return response.strip()  # <--- 非常重要，去掉前後空白





# ─── Agent 3: Qualification Checker (English prompt) ───
def check_qualification(violation_score: float):
    if violation_score>0.6:
        return 1  # Return -1 if parsing fails
    else:
        return 0


def extract_value_from_str_dict(text: str, field: str):
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

    data = ast.literal_eval(text)

    if isinstance(data, dict):
        return data.get(field, None)
    return None




def safe_generate_reply(agent, messages, max_retries=5):
    for i in range(max_retries):
        try:
            return agent.generate_reply(messages)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                wait_time = 0.5 + i  # 遞增等待時間
                print(f"⚠️ Rate limit hit, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise  # 非 rate limit 錯誤就直接報

def main(csv_path: str, log_path: str, ref_json_folder: str):
    df = pd.read_csv(csv_path)
    reference_cases = load_violation_cases(ref_json_folder)
    results = {"ID": [], "Answer": []}

    with open(log_path, 'w', encoding='utf-8') as log_file:
        for idx, row in df.iterrows():
            query = str(row['Question'])
            log_file.write(f"\n▶ 處理第 {idx} 筆資料...\n")

            # 1. 抽取商品資訊
            info_response = safe_generate_reply(INFO_EXTRACTION_AGENT, [
                {"role": "user", "content": f"Extract product information from the following query:\n{query}"}
            ])
            log_file.write(f"INFO response:\n{info_response}\n")

            product_name = extract_value_from_str_dict(info_response, "product_name")
            ad_text = extract_value_from_str_dict(info_response, "ad_description")

            if not ad_text:
                log_file.write("❌ 無法擷取 ad_description，跳過此筆。\n")
                continue

            # 2. 檢查違規程度
            violation_data = analyze_violation(ad_text, reference_cases)
            log_file.write(f"Violation response:\n{violation_data}\n")

            score = extract_value_from_str_dict(violation_data, "violation_score")
            if score is None:
                log_file.write("❌ 無法擷取 violation_score，跳過此筆。\n")
                continue

            # 3. 判定是否合格
            qualification = check_qualification(score)

            # 4. 存入結果
            results["ID"].append(idx)
            results["Answer"].append(qualification)
            log_file.write(f"✅ 結果：{'不合格' if qualification == 1 else '合格'} (score={score})\n")
            log_file.flush()
            # 5. 即時寫出結果 CSV
            df_out = pd.DataFrame(results)
            df_out.to_csv('/home/harper112/AI_final/out.csv', index=False)

    print("✅ 所有資料處理完成，結果寫入 out.csv")


# ─── Run the full pipeline ───
if __name__ == "__main__":
    csv_path = "/home/harper112/AI_final/final_project_query.csv"
    log_path = "/home/harper112/AI_final/violation_log.txt"
    ref_json_folder = "/home/harper112/AI_final/regulation"
    main(csv_path, log_path, ref_json_folder)
