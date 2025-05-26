import pandas as pd
from autogen import ConversableAgent
import os, sys

# ───── Set up LLM config ─────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ───── Define Agent ─────
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

# ───── Main logic ─────
def main(csv_path: str, log_path: str):
    df = pd.read_csv(csv_path)
    for idx in range(len(df['Question'])):
        query = df['Question'][idx]
        response = INFO_EXTRACTION_AGENT.generate_reply(
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract product information from the following query:\n{query}"
                    }
                ]
            )

        print(response)
        break


# ───── Entry Point ─────
if __name__ == "__main__":
    csv_path = "/home/harper112/AI_final/final_project_query.csv"
    log_path = "/home/harper112/AI_final/output_log.txt"
    main(csv_path, log_path)
