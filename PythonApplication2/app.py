from flask import Flask, render_template, request, redirect, url_for
import spacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
import numpy as np
import os
from components.ServiceAddress import ServiceAddress
from components.NeuralAddressSearch import NeuralAddressSearch
from components.IntentClassifier import IntentClassifier
import random
from datetime import datetime, timedelta

app = Flask(__name__, template_folder='templates')
print(">>> Template folder:", app.template_folder)
print(">>> Contents:", os.listdir(app.template_folder))

# ---------- Глобальная переменная поисковой системы ----------
search = None

# ------------------- Функции анализа запроса (из MainApplication1.py) -------------------
def extract_noun_chunks_with_matcher(doc):
    """Извлекает именные группы с помощью Matcher (старая логика)."""
    matcher = Matcher(doc.vocab)
    pattern1 = [
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADJ", "OP": "+"},
        {"POS": "NUM", "OP": "*"},
        {"POS": "NOUN", "OP": "+"}
    ]
    pattern_adj_noun = [{"POS": "ADJ"}, {"POS": "NOUN", "OP": "+"}]
    pattern_noun_adp_noun = [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN", "OP": "+"}]
    matcher.add("NOUN_PHRASE", [pattern1, pattern_adj_noun, pattern_noun_adp_noun])
    matches = matcher(doc)
    return [doc[start:end] for _, start, end in matches]

def build_query_analysis(doc):
    """Возвращает словарь с анализом запроса."""
    noun_chunks = extract_noun_chunks_with_matcher(doc)
    return {
        'tokens': [token.text for token in doc],
        'pos_tags': [(token.text, token.pos_) for token in doc],
        'noun_chunks': [chunk.text for chunk in noun_chunks],
        'key_terms': [token.lemma_ for token in doc if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB']]
    }

def convert_to_serializable(obj):
    """Преобразует numpy-типы в стандартные Python-типы (для отладки)."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def generate_booking_data_full():
    random.seed(42)  # фиксируем зерно для воспроизводимости
    data = []
    base_date = datetime.now()
    for i in range(100):
        # дата от 0 до 365 дней назад
        days_ago = random.randint(0, 365)
        d = base_date - timedelta(days=days_ago)
        data.append({
            'date': d.strftime('%Y-%m-%d'),
            'flight': f'SU-{random.randint(100, 999)}',
            'passenger': random.choice(['Иванов И.И.', 'Петров П.П.', 'Сидорова А.В.', 'Кузнецов Д.М.']),
            'age': random.randint(18, 70)
        })
    data.sort(key=lambda x: x['date'], reverse=True)
    return data

def generate_inventory_data_full():
    random.seed(42)
    data = []
    base_date = datetime.now()
    products = ['Кресло пилота', 'Топливный насос', 'Бортовая электроника', 'Аварийный трап', 'Кислородная маска']
    warehouses = ['Склад А', 'Склад B', 'Центральный']
    for i in range(100):
        days_ago = random.randint(0, 365)
        d = base_date - timedelta(days=days_ago)
        data.append({
            'date': d.strftime('%Y-%m-%d'),
            'product': random.choice(products),
            'quantity': random.randint(1, 500),
            'warehouse': random.choice(warehouses)
        })
    data.sort(key=lambda x: x['date'], reverse=True)
    return data

def generate_kpi_data_full():
    random.seed(42)
    data = []
    base_date = datetime.now()
    metrics = ['Загрузка рейсов', 'Точность расписания', 'Удовлетворённость пассажиров', 'Время обработки багажа']
    for i in range(100):
        days_ago = random.randint(0, 365)
        d = base_date - timedelta(days=days_ago)
        value = round(random.uniform(70.0, 100.0), 1)
        target = round(random.uniform(80.0, 100.0), 1)
        data.append({
            'date': d.strftime('%Y-%m-%d'),
            'metric': random.choice(metrics),
            'value': value,
            'target': target
        })
    data.sort(key=lambda x: x['date'], reverse=True)
    return data

def create_search_system():
    """Создание и наполнение системы адресами."""
    nlp = spacy.load("ru_core_news_lg")
    custom_infixes = [r"@"]
    infix_re = spacy.util.compile_infix_regex(list(nlp.Defaults.infixes) + custom_infixes)
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=spacy.util.compile_prefix_regex(nlp.Defaults.prefixes).search,
        suffix_search=spacy.util.compile_suffix_regex(nlp.Defaults.suffixes).search,
        infix_finditer=infix_re.finditer,
    )
    search_system = NeuralAddressSearch(nlp)
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
            description="Настройка параметров интеграции с ГСД, ключи API и системные настройки.",
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
            url="https://service.example.com/gds/reports-weekly",
            title="Еженедельный отчет о результатах работы ГСД",
            description="Еженедельные отчеты по показателям эффективности и КПЭ для операций ГСД.",
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

def get_slug(addr: ServiceAddress) -> str:
    """Извлекает последнюю часть URL для использования как slug."""
    return addr.url.rstrip('/').split('/')[-1]

# ---------- Фиксированные маршруты для каждого адреса ----------
@app.route('/main-dashboard')
def main_dashboard():
    slug = 'main-dashboard'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

@app.route('/analytics')
def analytics():
    slug = 'analytics'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

@app.route('/configuration')
def configuration():
    slug = 'configuration'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

@app.route('/help-support')
def help_support():
    slug = 'help-support'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

@app.route('/booking-management')
def booking_management():
    slug = 'booking-management'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

@app.route('/inventory')
def inventory():
    slug = 'inventory'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

@app.route('/reports-weekly')
def reports_weekly():
    slug = 'reports-weekly'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

@app.route('/api-documentation')
def api_documentation():
    slug = 'api-documentation'
    addr = search.addresses[list(get_slug(a) == slug for a in search.addresses).index(True)]
    return render_template('address_detail.html', addr=addr)

#Маршруты для отчетов:

@app.route('/report/booking')
def report_booking():
    days = request.args.get('period', 30, type=int)
    cutoff = datetime.now() - timedelta(days=days)
    # фильтруем записи, дата которых >= cutoff
    filtered = [row for row in booking_data 
                if datetime.strptime(row['date'], '%Y-%m-%d').date() >= cutoff.date()]
    return render_template('report_booking.html', period=days, data=filtered)

@app.route('/report/inventory')
def report_inventory():
    days = request.args.get('period', 30, type=int)
    cutoff = datetime.now() - timedelta(days=days)
    filtered = [row for row in inventory_data 
                if datetime.strptime(row['date'], '%Y-%m-%d').date() >= cutoff.date()]
    return render_template('report_inventory.html', period=days, data=filtered)

@app.route('/report/kpi')
def report_kpi():
    days = request.args.get('period', 30, type=int)
    cutoff = datetime.now() - timedelta(days=days)
    filtered = [row for row in kpi_data 
                if datetime.strptime(row['date'], '%Y-%m-%d').date() >= cutoff.date()]
    return render_template('report_kpi.html', period=days, data=filtered)

# ---------- Главная и поиск ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search_results():
    query = request.args.get('q', '').strip()
    if not query:
        return redirect(url_for('index'))

    # Проверяем отчётную интенцию (вся логика теперь внутри intent_clf)
    report_type, period = intent_clf.predict_intent(query)
    if report_type:
        return redirect(url_for(f'report_{report_type}', period=period))

    # Анализ запроса средствами spaCy + Matcher
    doc = search.nlp(query)
    query_analysis = build_query_analysis(doc)

    # Поиск адресов (нейросеть + бонусы)
    raw_results = search.search(query, top_n=5)

    # Формируем список результатов с деталями
    results_for_template = []
    for rank, (addr, score, details) in enumerate(raw_results, 1):
        endpoint = get_slug(addr).replace('-', '_')
        results_for_template.append({
            'rank': rank,
            'endpoint': endpoint,
            'title': addr.title,
            'description': addr.description,
            'category': addr.category,
            'url': addr.url,
            'relevance_score': score,
            'relevance_percent': min(100, int(score * 100)),
            'cosine_sim': details['cosine_sim'],
            'keyword_bonus': details['keyword_bonus'],
            'category_bonus': details['category_bonus'],
            'pos_bonus': details['pos_bonus'],
            'matched_keywords': details.get('matched_keywords', []),
            'important_tokens': details.get('important_tokens', [])
        })

    structured_results = {
        'query': query,
        'query_analysis': query_analysis,
        'results': results_for_template
    }
    return render_template('results.html', data=structured_results)

if __name__ == '__main__':
    # Инициализация поисковой системы ДО запуска сервера
    print("Инициализация поисковой системы...")
    search = create_search_system()
    model_dir = "saved_model"
    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "neural_model", "model.pth")):
        print("Загрузка обученной модели...")
        search.load(model_dir)
    else:
        print("Обучение модели...")
        search.train(epochs=30)
        search.save(model_dir)

    global booking_data, inventory_data, kpi_data
    booking_data = generate_booking_data_full()
    inventory_data = generate_inventory_data_full()
    kpi_data = generate_kpi_data_full()
    print("Данные для отчётов подготовлены.")

    print("Обучение классификатора интентов...")
    intent_clf = IntentClassifier(search.neural_processor, nlp=search.nlp)
    print("Готово.")

    print("Сервер готов к работе.")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)