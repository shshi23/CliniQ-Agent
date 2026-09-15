import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class MedicalGenerator:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found!")

        self.model = model or os.getenv("LLM_MODEL", "google/gemma-4-26b-a4b-it:free")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

    def generate_response(self, query: str, context_chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join([
            f"[Документ: {chunk['metadata']['title']}\n"
            f"Раздел: {chunk['metadata']['hierarchy']}\n"
            f"МКБ: {chunk['metadata']['icd_code']}]\n\n"
            f"{chunk['text']}"
            for chunk in context_chunks[:3]
        ])

        system_prompt = """Ты — медицинский ассистент, который помогает врачам и пациентам на основе официальных клинических рекомендаций Минздрава РФ.
        СТРОГИЕ ПРАВИЛА:
            1. Отвечай ТОЛЬКО на основе предоставленного контекста из клинических рекомендаций
            2. Если информации недостаточно — честно скажи "В предоставленных рекомендациях нет информации для ответа на этот вопрос"
            3. НЕ выдумывай диагнозы, дозировки, методы лечения — используй только данные из контекста
            4. Всегда указывай источник (название КР и раздел)
            5. Используй профессиональный, но понятный язык
            6. Если вопрос касается симптомов — укажи, к какому врачу обратиться
            7. Для детей указывай возрастные особенности из рекомендаций
        Формат ответа:
            - Краткий ответ на вопрос (1-2 предложения)
            - Обоснование из рекомендаций (с цитированием раздела и источника)
            - Рекомендации по дальнейшим действиям"""

        user_prompt = f"""Контекст из клинических рекомендаций: 
            {context}
            Вопрос пользователя: {query}
            Дай ответ на основе предоставленного контекста."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generation: {str(e)}"