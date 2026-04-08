import spacy
from spacy.matcher import Matcher
import numpy as np
from spacy.tokenizer import Tokenizer
from typing import Dict
import json
from components.NeuralAddressSearch import NeuralAddressSearch
from components.ServiceAddress import ServiceAddress
from typing import TYPE_CHECKING

def setup_search_system(nlp_model) -> NeuralAddressSearch:
    """
    Настройка системы поиска с предопределенными адресами
    """
    search_system = NeuralAddressSearch(nlp_model)
    
    # Список адресов с ключевыми словами (прописывается отдельно в коде)
    addresses = [
        ServiceAddress(
            url="https://service.example.com/gds/main-dashboard",
            title="Главная панель управления ГСД",
            description="Централизованная панель управления глобальной системой дистрибуции с данными о бронировании авиабилетов и отелей.",
            keywords=["гсд", "глобальная система дистрибуции", "панель управления", "главная страница", "бронирование", "полеты", "отели"],
            category="основное"
        ),
        ServiceAddress(
            url="https://service.example.com/gds/analytics",
            title="Аналитика и отчеты ГСД",
            description="Комплексный анализ и отчетность по транзакциям ГСД и показателям эффективности.",
            keywords=["гсд", "анализ", "отчет", "статистика", "метрика", "эффективность", "анализ данных"],
            category="аналитика"
        ),
        ServiceAddress(
            url="https://service.example.com/gds/configuration",
            title="Центр конфигурации ГСД",
            description="Настройтка параметров интеграции с ГСД, ключи API и системные настройки.",
            keywords=["гсд", "настройка", "параметры", "установка", "ключи api", "интеграция", "предпочтения"],
            category="настройки"
        ),
        ServiceAddress(
            url="https://service.example.com/gds/help-support",
            title="Помощь и поддержка ГСД",
            description="Центр поддержки по вопросам, связанным с ГСД, руководства по устранению неполадок, и часто задаваемые вопросы.",
            keywords=["гсд", "помощь", "поддержка", "устранение", "неполадок", "часто", "задаваемые", "вопросы", "руководство", "документация"],
            category="поддержка"
        ),
        ServiceAddress(
            url="https://service.example.com/gds/booking-management",
            title="Система управления бронированием",
            description="Управление и отслеживание всех бронирований через глобальную систему дистрибуции.",
            keywords=["гсд", "бронирование", "управление", "отслеживание", "заказы"],
            category="действия"
        ),
        ServiceAddress(
            url="https://service.example.com/gds/inventory",
            title="Запасы и наличие товаров на складе",
            description="Управление запасами и проверка наличия товаров в режиме реального времени через ГСД.",
            keywords=["гсд", "запасы", "доступность", "учет", "реального", "времени", "проверка", "товары"],
            category="запасы"
        ),
        ServiceAddress(
            url="https://service.example.com/gds/reports/weekly",
            title="Еженедельный отчет о результатах работы ГСД",
            description="Еженедельные отчеты по показателям эффективности и КПЭ для операций гсд.",
            keywords=["гсд", "еженедельный отчет", "эффективность", "кпэ", "метрика", "отчет"],
            category="отчеты"
        ),
        ServiceAddress(
            url="https://service.example.com/gds/api-documentation",
            title="Документация по API ГСД",
            description="Полная документация по API для интеграции с ГСД и разработки.",
            keywords=["гсд", "api", "документация", "разработка", "интеграция", "конечная точка", "эндпоинт"],
            category="разработчик"
        )
    ]
    
    search_system.add_addresses_from_list(addresses)
    
    return search_system

def extract_noun_chunks_with_matcher(doc):
    # Создаём объект Matcher, используя словарь модели
    matcher = Matcher(doc.vocab)

    # Определяем паттерн для поиска
    # 'OP': '+' означает "один или более"
    pattern = [
    {"POS": "ADV", "OP": "*"},   # Ноль или более наречий
    {"POS": "ADJ", "OP": "+"},   # Одно или более прилагательных
    {"POS": "NUM", "OP": "*"},   # Ноль или более числительных
    {"POS": "NOUN", "OP": "+"}   # Одно или более существительных
]
    # Паттерн 2: Прилагательное + Существительное
    pattern_adj_noun = [{"POS": "ADJ"}, {"POS": "NOUN", "OP": "+"}]

    # Паттерн 3: Существительное + Предлог + Существительное
    pattern_noun_adp_noun = [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN", "OP": "+"}]

    # Добавляем все паттерны
    matcher.add("NOUN_PHRASE", [pattern, pattern_adj_noun, pattern_noun_adp_noun])

    # Применяем matcher к документу
    matches = matcher(doc)

    # Извлекаем и возвращаем span'ы (фрагменты документа) для найденных совпадений
    noun_chunks = []
    for match_id, start, end in matches:
        span = doc[start:end]
        noun_chunks.append(span)
    return noun_chunks

def process_user_query(query: str, search_system: NeuralAddressSearch) -> Dict:
    """
    Обработка запроса пользователя с выводом структурированного результата
    """
    print(f"\n{'='*70}")
    print(f"🔍 ОБРАБОТКА ЗАПРОСА: \"{query}\"")
    print(f"{'='*70}")
    
    # Анализ запроса с помощью существующего функционала spaCy
    query_doc = search_system.nlp(query)
    noun_chunks = extract_noun_chunks_with_matcher(query_doc)

    print("\n📊 Анализ запроса:")
    print(f"   Токены: {[token.text for token in query_doc]}")
    print(f"   Части речи: {[(token.text, token.pos_) for token in query_doc]}")
    print(f"   Именные конструкции: {[chunk.text for chunk in noun_chunks]}")
    print(f"   Ключевые термины: {[token.lemma_ for token in query_doc if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB']]}")
    
    # Поиск релевантных адресов
    results = search_system.search(query, top_n=5)
    
    # Формирование структурированного вывода
    structured_results = {
        'query': query,
        'query_analysis': {
            'tokens': [token.text for token in query_doc],
            'pos_tags': [(token.text, token.pos_) for token in query_doc],
            'noun_chunks': [chunk.text for chunk in noun_chunks],
            'key_terms': [token.lemma_ for token in query_doc if not token.is_stop]
        },
        'results': []
    }
    
    print(f"\n{'='*70}")
    print("🎯 НАИБОЛЕЕ РЕЛЕВАНТНЫЕ АДРЕСА:")
    print(f"{'='*70}")
    
    for i, (addr, score, details) in enumerate(results, 1):
        relevance_percent = min(100, int(score * 100))
        
        result_item = {
            'rank': i,
            'url': addr.url,
            'title': addr.title,
            'description': addr.description,
            'category': addr.category,
            'relevance_score': score,
            'relevance_percent': relevance_percent,
            'match_details': details
        }
        structured_results['results'].append(result_item)
        
        print(f"\n{i}. {addr.title} (Релевантность: {relevance_percent}%)")
        print(f"   🔗 URL: {addr.url}")
        print(f"   📝 Описание: {addr.description}")
        print(f"   🏷️  Категория: {addr.category}")
        print(f"   🔑 Совпавшие ключевые слова: {', '.join(details['matched_keywords']) if details['matched_keywords'] else 'нет прямых совпадений'}")
        print(f"   📈 Детали оценки:")
        print(f"      - Нейросетевое сходство: {details['cosine_sim']:.3f}")
        print(f"      - Бонус за ключевые слова: {details['keyword_bonus']:.2f}")
        print(f"      - Бонус за категорию: {details['category_bonus']:.2f}")
        print(f"      - Бонус за важные слова: {details['pos_bonus']:.2f}")
    
    return structured_results

def convert_to_serializable(obj):
    """Рекурсивно преобразует NumPy типы в стандартные Python типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def main():
    """
    Главная функция, объединяющая весь функционал
    """
    print("🚀 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ПОИСКА АДРЕСОВ")
    print("="*70)
    
    # Загрузка модели spaCy (используем существующий nlp)
    nlp = spacy.load("ru_core_news_lg")
    
    # Добавляем кастомные правила токенизации
    custom_infixes = [r"@"]
    infix_re = spacy.util.compile_infix_regex(list(nlp.Defaults.infixes) + custom_infixes)
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=spacy.util.compile_prefix_regex(nlp.Defaults.prefixes).search,
        suffix_search=spacy.util.compile_suffix_regex(nlp.Defaults.suffixes).search,
        infix_finditer=infix_re.finditer,
    )
    
    # Создание системы поиска с нейросетью
    search_system = setup_search_system(nlp)
    
    # Обучение нейросети на ключевых словах адресов
    print("\n📚 ОБУЧЕНИЕ НЕЙРОСЕТИ...")
    search_system.train(epochs=30)  # 30 эпох обучения
    
    # Сохранение обученной модели (закомментировано на время проверки)
    # search_system.save("saved_search_system")
    
    about_text = "Где найти страницу с системой управления бронированием?"
    
    print("\n" + "="*70)
    print("📝 ОБРАБОТКА ЗАПРОСА ИЗ ПЕРЕМЕННОЙ about_text")
    print("="*70)
    
    # Основная обработка запроса
    results = process_user_query(about_text, search_system)
    
    # Сохранение результатов в структурированном формате
    print("\n" + "="*70)
    print("📄 СТРУКТУРИРОВАННЫЙ РЕЗУЛЬТАТ (JSON):")
    print("="*70)
    print(json.dumps(convert_to_serializable(results), ensure_ascii=False, indent=2))
    
    return results

if __name__ == "__main__":
    results = main()