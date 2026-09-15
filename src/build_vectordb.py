import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_model(local_path: str = "../local_model",
               fallback_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):

    if Path(local_path).exists() and (Path(local_path) / "config.json").exists():
        return SentenceTransformer(local_path)
    else:
        model = SentenceTransformer(fallback_name)
        return model

def clean_value(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return ", ".join(str(x).strip() for x in v)
    return v

def build_vectordb(
        chunks_path: str = "../data/chunks.json",
        persist_directory: str = "../data/vector_db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    model = load_model()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = []
    metadatas = []
    ids = []

    for i, chunk in tqdm(enumerate(chunks), desc="chunks processing"):
        raw = {k.strip(): v for k, v in chunk.items()}
        texts.append(raw["page_content"].strip())

        metadata = {k.strip(): clean_value(v) for k, v in raw["metadata"].items()}
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