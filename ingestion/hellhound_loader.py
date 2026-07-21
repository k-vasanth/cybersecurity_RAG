import os
import json
from config import report_path

class HellhoundLoader:
    def __init__(self):
        self.report_path = report_path

    def load(self):
        if not os.path.exists(self.report_path):
            print(f"Hellhound report not found: {self.report_path}")
            return []

        with open(self.report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        def walk(obj):
            if isinstance(obj, dict):
                url = obj.get("url") or obj.get("endpoint") or obj.get("path")
                params = obj.get("params") or obj.get("parameters") or []
                method = obj.get("method", "")
                title = obj.get("title") or url or "Hellhound Finding"

                if url:
                    content = f"""
                    Endpoint: {url}
                    Method: {method}
                    Parameters: {json.dumps(params, indent=2)}
                    Raw finding: {json.dumps(obj, indent=2)}
                    """
                    documents.append({
                        "source": "hellhound",
                        "title": title,
                        "url": url,
                        "content": content
                    })
                else:
                    for value in obj.values():
                        walk(value)

                for value in obj.values():
                    walk(value)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return documents
