# src/parser.py
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pymupdf4llm


class ClinicalRecommendationParser:
    """Парсер клинических рекомендаций"""
    
    SECTIONS = [
        "Список сокращений",
        "Термины и определения",
        "1. Краткая информация",
        "1.1 Определение",
        "1.2 Этиология",
        "1.3 Эпидемиология",
        "1.4 Особенности кодирования",
        "1.5 Классификация",
        "1.6 Клиническая картина",
        "2. Диагностика",
        "2.1 Жалобы и анамнез",
        "2.2 Физикальное обследование",
        "2.3 Лабораторные",
        "2.4 Инструментальные",
        "2.5 Иные диагностические",
        "3. Лечение",
        "4. Медицинская реабилитация",
        "5. Профилактика",
        "6. Организация оказания",
        "7. Дополнительная информация",
        "Критерии оценки качества",
    ]
    
    # Мусорные блоки, которые нужно удалить
    NOISE_PATTERNS = [
        r"Приложение А1\..*?(?=Приложение А2|\Z)",  # Состав рабочей группы
        r"Приложение А2\..*?(?=Приложение А3|\Z)",  # Методология разработки
        r"Список литературы.*?(?=Приложение|\Z)",   # Список литературы
    ]
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.raw_text = ""
        self.metadata = {}
        self.sections = {}
    
    def extract_text(self) -> str:
        """Извлекает текст из PDF в Markdown."""
        md_text = pymupdf4llm.to_markdown(str(self.pdf_path))
        self.raw_text = md_text
        return md_text
    
    def extract_metadata(self) -> Dict:
        """Извлекает метаданные из шапки документа"""
        text = self.raw_text
        
        title_match = re.search(
            r"Клинические рекомендации\s*\n(.+?)(?:\n|Кодирование)", 
            text, 
            re.IGNORECASE
        )
        title = title_match.group(1).strip() if title_match else "Неизвестно"
        
        icd_match = re.search(r"здоровье[:\s]*([A-Z]\d+(?:[.,]\d+)*(?:\s*,\s*[A-Z]\d+(?:[.,]\d+)*)*)", text)
        icd_codes = []
        if icd_match:
            icd_codes = [code.strip() for code in icd_match.group(1).split(",")]
        
        year_match = re.search(r"Год утверждения.*?[:\s]*(\d{4})", text)
        year = int(year_match.group(1)) if year_match else None
        
        age_match = re.search(r"Возрастная категория[:\s]*(.+?)(?:\n|Специальность)", text)
        age_category = age_match.group(1).strip() if age_match else None
        
        id_match = re.search(r"ID[:\s]*(\d+_\d+)", text)
        doc_id = id_match.group(1) if id_match else None
        
        self.metadata = {
            "file_name": self.pdf_path.name,
            "disease_name": title,
            "icd_codes": icd_codes,
            "year": year,
            "age_category": age_category,
            "doc_id": doc_id,
            "specialty": "Офтальмология",
        }
        return self.metadata
    
    def clean_text(self, text: str) -> str:
        """Очищает текст от артефактов"""

        for pattern in self.NOISE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        
        text = re.sub(r"\[\d+(?:[,\s]*\d+)*(?:\s*-\s*\d+)?\]", "", text)

        text = re.sub(r"([.,;:!?])([А-ЯA-Zа-яa-z])", r"\1 \2", text)
 
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()
    
    def split_by_sections(self, text: str) -> Dict[str, str]:
        """Разбивает текст на секции по заголовкам"""
        sections = {}

        header_pattern = r"^(#{1,3}\s*)?(.+?)$"
        
        current_section = "Введение"
        current_content = []
        
        for line in text.split("\n"):
            line_stripped = line.strip()
            
            is_header = False
            for section_name in self.SECTIONS:
                if line_stripped.startswith(section_name) or (
                    re.match(r"^\d+\.\d*\s+\S", line_stripped) and 
                    len(line_stripped) < 100
                ):
                    if current_content:
                        sections[current_section] = "\n".join(current_content).strip()
                    
                    current_section = line_stripped
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()
        
        self.sections = sections
        return sections
    
    def parse(self) -> Tuple[Dict, Dict[str, str]]:
        """Полный цикл парсинга."""
        print(f"Парсинг: {self.pdf_path.name}")
        
        self.extract_text()
        
        metadata = self.extract_metadata()
        print(f"Название: {metadata['disease_name']}")
        print(f"МКБ: {', '.join(metadata['icd_codes'])}")
        print(f"Год: {metadata['year']}")
        
        cleaned_text = self.clean_text(self.raw_text)
    
        sections = self.split_by_sections(cleaned_text)
        print(f"Найдено секций: {len(sections)}")
        
        return metadata, sections


def parse_all_pdfs(input_dir: str, output_dir: str) -> List[Dict]:
    """Парсит все PDF из директории"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_metadata = []
    
    for pdf_file in input_path.glob("*.pdf"):
        parser = ClinicalRecommendationParser(str(pdf_file))
        metadata, sections = parser.parse()
        
        # Сохраняем очищенный текст в Markdown
        md_output = output_path / f"{pdf_file.stem}.md"
        with open(md_output, "w", encoding="utf-8") as f:
            f.write(f"# {metadata['disease_name']}\n\n")
            f.write(f"**МКБ:** {', '.join(metadata['icd_codes'])}  \n")
            f.write(f"**Год:** {metadata['year']}  \n")
            f.write(f"**Возрастная категория:** {metadata['age_category']}\n\n")
            f.write("---\n\n")
            
            for section_name, content in sections.items():
                f.write(f"## {section_name}\n\n")
                f.write(content)
                f.write("\n\n")
        
        all_metadata.append(metadata)
        print(f"Сохранено: {md_output.name}\n")
    
    metadata_file = output_path.parent / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"Метаданные сохранены: {metadata_file}")
    
    return all_metadata


if __name__ == "__main__":
    parse_all_pdfs(
        input_dir="../data/raw",
        output_dir="../data/processed"
    )