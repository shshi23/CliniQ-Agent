import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class MedicalRetriever:
    def __init__(self,
                 model_path: str = str(BASE_DIR / "local_model"),
                 db_path: str = str(BASE_DIR / "data" / "vector_db"),
                 collection_name: str = "medical_guidelines"):

        if Path(model_path).exists() and (Path(model_path) / "config.json").exists():
            self.model = SentenceTransformer(model_path)
        else:
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        ) 

        docs = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            docs.append({
                "text": text,
                "metadata": meta,
                "similarity": round(1 - distance, 4)
            })
        return docs

if __name__ == "__main__":
    retriever = MedicalRetriever()

    queries = [
        "Ребенок 8 лет, быстро устает при чтении и приближает книгу к глазам",
        "У новорожденного слезотечение, светобоязнь и увеличен глаз",
        "Плохо видит вдаль, в очках зрение улучшается до 1.0",
    ]

    for q in queries:
        print(f"\n=== Запрос: {q} ===")
        for i, doc in enumerate(retriever.search(q, k=2), 1):
            m = doc["metadata"]
            print(f"{i}. [{doc['similarity']:.3f}] {m['title']} | {m['hierarchy']}")