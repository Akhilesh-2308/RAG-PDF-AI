from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="documents", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))] 
        self.client.upsert(self.collection, points=points)

    def search(self, q_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=q_vector,
            limit=top_k
        )
        contexts = []
        sources = set()

        for r in results.points:
            payload = r.payload or {}
            source = payload.get("source", "")
            text = payload.get("text", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
