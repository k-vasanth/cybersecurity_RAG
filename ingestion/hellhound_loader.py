import os
import json

class HellhoundLoader:
    def __init__(self, report_path="/hellhound-spider/hellhound_report.json"):
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
                    Parameters: {params}
                    Raw finding: {obj}
                    """
                    documents.append({
                        "source": "hellhound",
                        "title": title,
                        "url": url,
                        "content": content
                    })

                for value in obj.values():
                    walk(value)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return documents
