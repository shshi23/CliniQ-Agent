import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pymupdf4llm


class ClinicalRecommendationParser:
    """Парсер клинических рекомендаций Минздрава РФ."""
    
    # Паттерны секций — гибкие, учитывают Markdown-артефакты
    SECTION_PATTERNS = [
        r"Список сокращений",
        r"Термины и определения",
        r"1\.\s*Краткая информация",
        r"1\.1\s+Определение",
        r"1\.2\s+Этиология",
        r"1\.3\s+Эпидемиология",
        r"1\.4\s+Особенности кодирования",
        r"1\.5\s+Классификация",
        r"1\.6\s+Клиническая картина",
        r"2\.\s*Диагностика",
        r"2\.1\s+Жалобы",
        r"2\.2\s+Физикальное",
        r"2\.3\s+Лабораторные",
        r"2\.4\s+Инструментальные",
        r"2\.5\s+Иные диагностические",
        r"3\.\s*Лечение",
        r"3\.\d+\.\d+\s+",
        r"3\.\d+\s+",
        r"4\.\s*Медицинская реабилитация",
        r"5\.\s*Профилактика",
        r"6\.\s*Организация",
        r"7\.\s*Дополнительная информация",
        r"Критерии оценки качества",
    ]
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.raw_md = ""
        self.cleaned_md = ""
        self.metadata: Dict = {}
        self.sections: Dict[str, str] = {}
    
    # ==================== ШАГ 1: Конвертация PDF -> Markdown ====================
    
    def extract_markdown(self) -> str:
        """Конвертирует PDF в Markdown через pymupdf4llm."""
        self.raw_md = pymupdf4llm.to_markdown(str(self.pdf_path))
        return self.raw_md
    
    # ==================== ШАГ 2: Очистка от мусора ====================
    
    @staticmethod
    def _strip_md(text: str) -> str:
        """Удаляет Markdown-артефакты: #, **, *, _."""
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        return text.strip()
    
    def _remove_toc(self, text: str) -> str:
        """
        Удаляет оглавление.
        Оглавление идёт от слова "Оглавление" до первого реального заголовка
        "Список сокращений", за которым идёт список сокращений (с тире).
        """
        # Ищем "Оглавление" и удаляем всё до первого "Список сокращений\n<СОКРАЩЕНИЕ>"
        pattern = r"(?is)Оглавление\s*\n.*?(?=\n\s*Список сокращений\s*\n\s*[A-ZА-ЯЁ]{2,}\s*[–\-])"
        text = re.sub(pattern, "", text, count=1)
        
        # Если не сработало — пробуем более простой вариант
        if "Оглавление" in text:
            # Удаляем всё от "Оглавление" до первого вхождения "Список сокращений"
            # после которого идёт строка с тире (определение сокращения)
            lines = text.split("\n")
            new_lines = []
            in_toc = False
            for i, line in enumerate(lines):
                if re.search(r"Оглавление", line, re.IGNORECASE):
                    in_toc = True
                    continue
                if in_toc:
                    # Проверяем, это реальный заголовок "Список сокращений"?
                    if re.search(r"^\s*Список сокращений\s*$", line):
                        # Смотрим следующую строку — если там сокращение, то это реальный заголовок
                        if i + 1 < len(lines) and re.search(r"[A-ZА-ЯЁ]{2,}\s*[–\-]", lines[i + 1]):
                            in_toc = False
                            new_lines.append(line)
                    # Иначе пропускаем (это часть оглавления)
                else:
                    new_lines.append(line)
            text = "\n".join(new_lines)
        
        return text
    
    def _remove_literature(self, text: str) -> str:
        """Удаляет список литературы."""
        pattern = r"(?is)\n\s*Список литературы\s*\n.*?(?=\n\s*Приложение\s+[А-ЯA-Z]|\Z)"
        return re.sub(pattern, "", text)
    
    def _remove_appendix_a1_a2(self, text: str) -> str:
        """Удаляет мусорные приложения А1 и А2."""
        pattern_a1 = r"(?is)\n\s*Приложение\s+[АA]1\..*?(?=\n\s*Приложение\s+[А-ЯA-Z]|\Z)"
        text = re.sub(pattern_a1, "", text)
        
        pattern_a2 = r"(?is)\n\s*Приложение\s+[АA]2\..*?(?=\n\s*Приложение\s+[А-ЯA-Z]|\Z)"
        text = re.sub(pattern_a2, "", text)
        
        return text
    
    def _remove_citation_numbers(self, text: str) -> str:
        """Удаляет сноски на источники вида [1], [1, 2], [1-3]."""
        return re.sub(r"\[\d+(?:\s*[,;]\s*\d+)*(?:\s*[-–]\s*\d+)?\]", "", text)
    
    def _fix_merged_words(self, text: str) -> str:
        """Исправляет слипшиеся слова."""
        # Пробел после знаков препинания перед заглавной буквой
        text = re.sub(r"([.,:;!?])([A-ZА-ЯЁa-zа-яё])", r"\1 \2", text)
        # Пробел между строчной русской и заглавной латинской
        text = re.sub(r"([а-яё])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([a-z])([А-ЯЁ])", r"\1 \2", text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Убирает лишние пробелы и переносы."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)
    
    def clean(self, text: str) -> str:
        """Полная очистка текста от артефактов."""
        text = self._remove_toc(text)
        text = self._remove_literature(text)
        text = self._remove_appendix_a1_a2(text)
        text = self._remove_citation_numbers(text)
        text = self._fix_merged_words(text)
        text = self._normalize_whitespace(text)
        return text.strip()
    
    # ==================== ШАГ 3: Извлечение метаданных ====================
    
    def extract_metadata(self, text: str) -> Dict:
        """Извлекает метаданные из шапки документа."""
        # Сначала очищаем от Markdown-артефактов для поиска
        clean_text = self._strip_md(text)
        metadata = {"file_name": self.pdf_path.name}
        
        # Название заболевания
        title_match = re.search(
            r"Клинические рекомендации\s*\n(.+?)(?:\n|Кодирование)",
            clean_text,
            re.IGNORECASE
        )
        title = title_match.group(1).strip() if title_match else "Неизвестно"
        # Очищаем название от Markdown
        title = self._strip_md(title)
        metadata["disease_name"] = title
        
        # Коды МКБ — ищем в контексте "здоровье"
        # Учитываем, что между "здоровье" и кодом может быть перенос строки
        icd_match = re.search(
            r"здоровье[:\s]+([A-Z]\d+(?:[.,]\d+)?(?:\s*,\s*[A-Z]\d+(?:[.,]\d+)?)*)",
            clean_text,
            re.IGNORECASE
        )
        if icd_match:
            codes_str = icd_match.group(1)
            # Разделяем по запятым и пробелам
            codes = re.findall(r"[A-Z]\d+(?:[.,]\d+)?", codes_str, re.IGNORECASE)
            metadata["icd_codes"] = [c.upper() for c in codes]
        else:
            # Фоллбэк: ищем любые коды МКБ в первых 500 символах
            header = clean_text[:500]
            codes = re.findall(r"\b([A-Z]\d{1,2}(?:\.\d+)?)\b", header)
            metadata["icd_codes"] = [c.upper() for c in codes] if codes else []
        
        # Год утверждения
        year_match = re.search(r"Год утверждения.*?[:\s]*(\d{4})", clean_text)
        metadata["year"] = int(year_match.group(1)) if year_match else None
        
        # Возрастная категория — останавливаемся на "Специальность"
        age_match = re.search(
            r"Возрастная категория[:\s]*(.+?)(?:\s*Специальность|$)",
            clean_text,
            re.IGNORECASE
        )
        age = age_match.group(1).strip() if age_match else None
        age = self._strip_md(age) if age else None
        metadata["age_category"] = age
        
        # ID документа
        id_match = re.search(r"ID[:\s]*(\d+_\d+)", clean_text)
        metadata["doc_id"] = id_match.group(1) if id_match else None
        
        metadata["specialty"] = "Офтальмология"
        
        self.metadata = metadata
        return metadata
    
    # ==================== ШАГ 4: Разбиение на секции ====================
    
    def split_into_sections(self, text: str) -> Dict[str, str]:
        """Разбивает текст на смысловые секции по заголовкам."""
        sections = {}
        current_section = "Введение"
        current_content = []
        
        for line in text.split("\n"):
            line_stripped = line.strip()
            # Очищаем от Markdown для сравнения
            line_clean = self._strip_md(line_stripped)
            
            if not line_clean:
                current_content.append(line)
                continue
            
            # Проверяем, является ли строка заголовком раздела
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                # Используем search, а не match — паттерн может быть не с начала строки
                if re.search(pattern, line_clean) and len(line_clean) < 150:
                    # Сохраняем предыдущую секцию
                    if current_content:
                        content = "\n".join(current_content).strip()
                        if content:
                            sections[current_section] = content
                    
                    current_section = line_clean
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line)
        
        # Сохраняем последнюю секцию
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                sections[current_section] = content
        
        self.sections = sections
        return sections
    
    # ==================== ГЛАВНЫЙ МЕТОД ====================
    
    def parse(self) -> Tuple[Dict, Dict[str, str], str]:
        """Полный цикл парсинга."""
        print(f"📄 Парсинг: {self.pdf_path.name}")
        
        # 1. Конвертация
        self.extract_markdown()
        
        # 2. Извлечение метаданных (из сырого текста)
        self.extract_metadata(self.raw_md)
        print(f"   🏷  {self.metadata['disease_name']}")
        print(f"   🔢 МКБ: {', '.join(self.metadata['icd_codes'])}")
        print(f"   📅 {self.metadata['year']} | 👥 {self.metadata['age_category']}")
        
        # 3. Очистка
        self.cleaned_md = self.clean(self.raw_md)
        
        # 4. Разбиение на секции
        self.split_into_sections(self.cleaned_md)
        print(f"   📑 Найдено секций: {len(self.sections)}")
        
        return self.metadata, self.sections, self.cleaned_md


# ==================== ФУНКЦИИ ДЛЯ МАССОВОЙ ОБРАБОТКИ ====================

def parse_single_pdf(pdf_path: str, output_dir: str) -> Dict:
    """Парсит один PDF и сохраняет результат."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    parser = ClinicalRecommendationParser(pdf_path)
    metadata, sections, cleaned_md = parser.parse()
    
    # Сохраняем очищенный markdown
    md_file = output_path / f"{Path(pdf_path).stem}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# {metadata['disease_name']}\n\n")
        f.write(f"**МКБ:** {', '.join(metadata['icd_codes'])}  \n")
        f.write(f"**Год:** {metadata['year']}  \n")
        f.write(f"**Возрастная категория:** {metadata['age_category']}  \n")
        f.write(f"**ID:** {metadata['doc_id']}\n\n")
        f.write("---\n\n")
        
        for section_name, content in sections.items():
            f.write(f"## {section_name}\n\n")
            f.write(content)
            f.write("\n\n")
    
    print(f"   ✅ Сохранено: {md_file.name}\n")
    return metadata


def parse_all_pdfs(input_dir: str, output_dir: str) -> List[Dict]:
    """Парсит все PDF из директории."""
    input_path = Path(input_dir)
    pdf_files = sorted(input_path.glob("*.pdf"))
    
    print(f"📂 Найдено PDF файлов: {len(pdf_files)}\n")
    print("=" * 60)
    
    all_metadata = []
    for pdf_file in pdf_files:
        metadata = parse_single_pdf(str(pdf_file), output_dir)
        all_metadata.append(metadata)
        print("=" * 60)
    
    # Сохраняем общий metadata.json
    metadata_file = Path(output_dir).parent / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 Метаданные сохранены: {metadata_file}")
    return all_metadata


if __name__ == "__main__":
    INPUT_DIR = "data/raw"
    OUTPUT_DIR = "data/processed"
    parse_all_pdfs(INPUT_DIR, OUTPUT_DIR)