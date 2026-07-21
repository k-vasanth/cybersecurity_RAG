class ReciprocalRankFusion:
    def reciprocal_rank_fusion(self, faiss_docs, bm25_docs, k=60):
        scores = {}
        documents = {}

        for rank, doc in enumerate(faiss_docs, start=1):
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                continue

            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank))
            documents[chunk_id] = doc
            documents[chunk_id]["retrieval_type"] = "faiss"

        for rank, doc in enumerate(bm25_docs, start=1):
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                continue

            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank))

            if chunk_id in documents:
                documents[chunk_id]["retrieval_type"] = "faiss+bm25"
            else:
                documents[chunk_id] = doc
                documents[chunk_id]["retrieval_type"] = "bm25"

        fused_docs = sorted(
            documents.values(),
            key=lambda doc: scores[doc["chunk_id"]],
            reverse=True
        )

        for doc in fused_docs:
            doc["rrf_score"] = scores[doc["chunk_id"]]

        return fused_docs