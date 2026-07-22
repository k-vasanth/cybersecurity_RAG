from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent
VECTOR_DB_PATH=BASE_DIR / "data" / "vectors"
METADATA_DB_PATH=BASE_DIR / "data" / "metadata"
OWASP_DATA_PATH=BASE_DIR / "data" / "owasp.json"
report_path="/hellhound-spider/hellhound_report.json"
