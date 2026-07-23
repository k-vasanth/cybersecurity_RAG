from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_PATH = BASE_DIR / "data" / "vectors"
METADATA_DB_PATH = BASE_DIR / "data" / "metadata"
FILE_DATA_PATH = BASE_DIR / "data" / "payloads.json"
report_path = BASE_DIR / "hellhound-spider" / "hellhound_report.json"
