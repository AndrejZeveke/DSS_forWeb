from dataclasses import dataclass
from typing import List

@dataclass
class ServiceAddress:
    """Класс для хранения информации об адресе сервиса"""
    url: str
    title: str
    description: str
    keywords: List[str]  # ключевые слова для обучения
    category: str
    relevance_score: float = 0.0