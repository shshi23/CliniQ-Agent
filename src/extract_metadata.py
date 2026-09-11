import json
import ast
from pathlib import Path

def extract_metadata(filepath: Path) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    metadata = {}
    in_frontmatter = False

    for line in lines:
        if line.strip() == '#CliniQAgent':
            in_frontmatter = True
            continue

        if in_frontmatter and line.strip() == '':
            break

        if in_frontmatter and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if value.startswith('[') and value.endswith(']'):
                try:
                    value = ast.literal_eval(value)
                    if isinstance(value, list):
                        value = [str(v).strip() for v in value]
                except (ValueError, SyntaxError):
                    pass

            metadata[key] = value

    metadata['filename'] = filepath.name

    return metadata

def extract_all_metadata(data_dir: str = "../data") -> dict:
    data_path = Path(data_dir)
    all_metadata = {}

    md_files = list(data_path.glob("*.md"))

    for md_file in md_files:
        metadata = extract_metadata(md_file)
        all_metadata[md_file.name] = metadata

    return all_metadata

if __name__ == "__main__":
    all_metadata = extract_all_metadata("../data")
    output_path = Path("../data/metadata.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
