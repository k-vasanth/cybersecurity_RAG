from ingestion.owasp_loader import OWASPLoader
from ingestion.hellhound_loader import HellhoundLoader
from processing.cleaner import TextCleaner
from processing.chunker import TextChunker
from processing.metadata_builder import MetadataBuilder
from embedding.embedder import Embedder
from storage.vector_db import VectorDB
from storage.metadata_db import MetadataDB
import faiss
import pickle
import os

def pipeline():

    documents = []
    documents.extend(OWASPLoader().load())
    documents.extend(HellhoundLoader("hellhound-spider/hellhound_report.json").load())
    cleaner = TextCleaner()
    chunker = TextChunker()
    metadata_builder = MetadataBuilder()
    embedder = Embedder()
    vector_db = VectorDB(dimension=768, rebuild=True)
    metadata_db = MetadataDB(rebuild=True)
    print(f"Total documents loaded: {len(documents)}")
    all_chunks = []
    all_payloads = []
    chunk_counter = 0
    for doc in documents:
        cleaned = cleaner.clean(doc["content"])
        chunks = chunker.chunk(cleaned)
        for chunk in chunks:
            chunk_id = f"chunk_{chunk_counter}"
            metadata = metadata_builder.build(
                source=doc["source"],
                title=doc["title"],
                chunk_id=chunk_id,
                url=doc.get("url"),
                severity="high",
                tags=[doc.get("type", "general")]
            )

            metadata["content"] = chunk
            all_chunks.append(chunk)
            all_payloads.append(metadata)
            metadata_db.insert(metadata)
            chunk_counter += 1
    vectors = embedder.encode(all_chunks)
    vector_db.store(vectors,all_payloads)
    os.makedirs("data", exist_ok=True)
    faiss.write_index(vector_db.index,"data/faiss_index.bin")
    with open("data/payloads.pkl", "wb") as f:
        pickle.dump(vector_db.payloads, f)


if __name__ == "__main__":
    pipeline()
