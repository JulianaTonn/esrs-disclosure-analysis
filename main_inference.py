##### Main Inference Script #####

### Load Metadata of Reports to be analyzed
import pandas as pd
esrs_reports = pd.read_excel("./data_preparation/esrs_reports.xlsx")

test = esrs_reports[20:30]


import os
import re
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from rag_system import RAGSystem



# ----------------------------
# 0) Environment & Auth
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv_path = os.path.expanduser("~/thesis/esg_extraction/.env")
load_dotenv(dotenv_path=dotenv_path)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(HF_TOKEN)


# ----------------------------
# 1) Config
# ----------------------------
DB_ROOT = Path("./faiss_dbs")
DB_ROOT.mkdir(parents=True, exist_ok=True)
ESRS_METADATA_PATH = 'EsrsMetadata.xlsx'
REPORTS_CSV_PATH = './data_preparation/esrs_reports.csv' 

RESULTS_PATH = "all_results.jsonl" # nested dict: { report_id: { query_id: {verdict, analysis, sources} } }
SKIPPED_PATH = "skipped_reports.csv" # keep track of skipped reports

# Ensure skipped_reports.csv has a header if file doesn't exist
if not os.path.exists(SKIPPED_PATH):
    pd.DataFrame(columns=esrs_reports.columns).to_csv(SKIPPED_PATH, index=False)

# ----------------------------
# 2) Instantiate your system
# ----------------------------
rag = RAGSystem(ESRS_METADATA_PATH)


# ----------------------------
# 3) Process each report URL
# ----------------------------
import time
start_time = time.time()

for idx, row in test.iterrows():
    url = row.get("link", None)
    company_name = row['company']
    report_id = row['report_id']
    db_path = str(DB_ROOT / report_id)
    
    if pd.isna(url) or not isinstance(url, str) or not url.strip():
        print(f"Row {idx}, Company {report_id}: no valid 'link' URL -> skipping")
        # append directly to skipped_reports.csv
        row.to_frame().T.to_csv(SKIPPED_PATH, mode="a", header=False, index=False)
        continue

    print(f"\n=== Running pipeline for: {idx} - {report_id} ===")
    try:
        # Important: use the method your class exposes
        # (expects report_id, db_path, and either pdf_url or pdf_path)
        result = rag.process_and_analyze_report(
            report_id=report_id,
            db_path=db_path,
            pdf_url=url        # we’re using the 'link' column (a URL)
        )

        # augment result with metadata before saving
        record = {
            "report_id": report_id,
            "company": company_name,
            "row_index": int(idx),
            "result": result.get(report_id, {}),
        }

        # append result as one JSON line
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Failed on {report_id}: {e}")
        # also log failure in skipped_reports.csv
        row.to_frame().T.to_csv(SKIPPED_PATH, mode="a", header=False, index=False)


time = (time.time() - start_time)/ 60
print(time, "minutes needed for 5 reports")