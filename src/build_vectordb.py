import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
from tqdm import tqdm

def build_vectordb(
        chunks_path: str = "../data/chunks.json",
        persist_directory: str = "../data/vector_db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    model = SentenceTransformer(model_name)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = []
    metadatas = []
    ids = []

    for i, chunk in tqdm(enumerate(chunks), desc="chunks processing"):
        texts.append(chunk["page_content"])
        metadata = chunk["metadata"].copy()

        if isinstance(metadata.get("icd_code"), list):
            metadata["icd_code"] = ", ".join(metadata["icd_code"])
        if isinstance(metadata.get("age_category"), list):
            metadata["age_category"] = ", ".join(metadata["age_category"])
        if isinstance(metadata.get("hierarchy"), list):
            metadata["hierarchy"] = " -> ".join(metadata["hierarchy"])

        metadatas.append(metadata)
        ids.append(metadata["chunk_id"])

    batch_size = 32
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="embedding processing"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)

        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / norms

        embeddings.extend(batch_embeddings.tolist())

    if Path(persist_directory).exists():
        import shutil
        shutil.rmtree(persist_directory)

    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(
        name="medical_guidelines",
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

if __name__ == "__main__":
    build_vectordb()