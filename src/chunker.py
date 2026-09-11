from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import json
from pathlib import Path

def get_chunks_at_file(md_filepath: str, metadata: dict) -> list[Document]:
    """
    Разбивает .md-файл на чанки по заголовкам 
    """
    with open(md_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    content_start = 0
    in_frontmatter = False

    for i, line in enumerate(lines):
        if line.strip() == '#CliniQAgent':
            in_frontmatter = True
            continue
        if in_frontmatter and line.strip() == '':
            content_start = i + 1
            break

    md_text = '\n'.join(lines[content_start:])

    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4")
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )

    chunks = md_splitter.split_text(md_text)

    documents = []
    for i, chunk in enumerate(chunks):
        hierarchy = []
        if "header_2" in chunk.metadata:
            hierarchy.append(chunk.metadata["header_2"])
        if "header_3" in chunk.metadata:
            hierarchy.append(chunk.metadata["header_3"])
        if "header_4" in chunk.metadata:
            hierarchy.append(chunk.metadata["header_4"])

        context_prefix = (
            f"[Документ: {metadata['title']} | "
            f"МКБ: {', '.join(metadata['icd_code'])} | "
            f"Возраст: {', '.join(metadata['age_category'])} | "
            f"Раздел: {' -> '.join(hierarchy) if hierarchy else 'Введение'}]\n\n"
        )

        doc = Document(
            page_content=context_prefix + chunk.page_content,
            metadata={
                "source": metadata['filename'],
                "title": metadata['title'],
                "icd_code": metadata['icd_code'],
                "age_category": metadata['age_category'],
                "approval_year": int(metadata['approval_year']),
                "hierarchy": hierarchy,
                "chunk_id": f"{metadata['filename']}_chunk_{i}"
            }
        )
        documents.append(doc)

    return documents

def get_chunks_all_files(metadata_path: str = "../data/metadata.json", data_dir: str = "../data") -> list[Document]:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    all_chunks = []

    for filename, metadata in all_metadata.items():
        filepath = f"{data_dir}/{filename}"
        try:
            chunks = get_chunks_at_file(filepath, metadata)
            all_chunks.extend(chunks)
        except FileNotFoundError:
            print(f"File not found!")
        except Exception as e:
            print(f"Error: {e}")

    return all_chunks

def save_chunks_to_json(chunks: list[Document], output_path: str = "../data/chunks.json"):
    data = []
    for chunk in chunks:
        data.append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    all_chunks = get_chunks_all_files()
    save_chunks_to_json(all_chunks)