import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random
from spacy.matcher import Matcher

class IntentClassifier:
    """Классификатор: является ли запрос отчётом, тип отчёта и период."""
    def __init__(self, text_processor, nlp=None):
        self.processor = text_processor
        self.nlp = nlp
        self.scaler = StandardScaler()
        self.clf_is_report = None
        self._train()

        # Паттерны Matcher для извлечения периода
        self.period_patterns = [
            [{"LOWER": "за"}, {"LOWER": "последние", "OP": "?"}, {"POS": "NUM", "OP": "+"},
             {"LEMMA": {"IN": ["день", "неделя", "месяц", "квартал", "год"]}}],
            [{"LOWER": "за"}, {"LOWER": {"IN": ["последний", "этот", "текущий"]}, "OP": "?"},
             {"LEMMA": {"IN": ["день", "неделя", "месяц", "квартал", "год"]}}],
            [{"LOWER": "за"}, {"LOWER": {"IN": ["вчера", "сегодня"]}}],
            [{"LOWER": "за"}, {"LOWER": {"IN": ["эту", "прошлую"]}}, {"LEMMA": "неделя"}],
            [{"LOWER": "за"}, {"LOWER": {"IN": ["эту", "последнюю"]}}, {"LEMMA": "неделя"}],
        ]

    def _generate_synthetic_data(self, n_samples=2000):
        data = []
        report_keywords = ['отчёт', 'отчет', 'составь', 'покажи данные', 'выгрузи', 'сформируй']
        type_keywords = {
            'booking': ['бронирование', 'бронь', 'рейс', 'пассажир'],
            'inventory': ['запасы', 'склад', 'наличие', 'товар', 'инвентар'],
            'kpi': ['кпэ', 'эффективность', 'показатели', 'метрики', 'ключевые показатели']
        }
        for _ in range(n_samples):
            if random.random() < 0.7:
                rtype = random.choice(list(type_keywords.keys()))
                type_word = random.choice(type_keywords[rtype])
                phrase = random.choice(['за день', 'за неделю', 'за месяц', 'за квартал', 'за год'])
                kw = random.choice(report_keywords)
                query = f"{kw} по {type_word} {phrase}"
                data.append((query, True))
            else:
                nav = random.choice([
                    "главная панель управления", "настройки системы",
                    "помощь и поддержка", "аналитика и отчёты",
                    "документация api", "центр конфигурации",
                    "система бронирования"
                ])
                data.append((nav, False))
        return data

    def _train(self):
        raw = self._generate_synthetic_data()
        texts = [item[0] for item in raw]
        y = [1 if item[1] else 0 for item in raw]
        X = self.processor.encode_batch(texts)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.clf_is_report = LogisticRegression(max_iter=1000)
        self.clf_is_report.fit(X_scaled, y)

    def _get_report_type(self, query):
        """Определение типа отчёта по ключевым словам."""
        q = query.lower()
        if any(w in q for w in ['бронирование', 'бронь', 'брони', 'рейс', 'пассажир']):
            return 'booking'
        if any(w in q for w in ['запасы', 'склад', 'наличие', 'товар', 'инвентар']):
            return 'inventory'
        if any(w in q for w in ['кпэ', 'эффективность', 'показатели', 'метрики', 'ключевые показатели']):
            return 'kpi'
        return 'booking'  # fallback

    def _extract_period(self, query):
        """Извлечение периода в днях через Matcher."""
        if not self.nlp:
            return 30
        doc = self.nlp(query)
        matcher = Matcher(self.nlp.vocab)
        for i, pat in enumerate(self.period_patterns):
            matcher.add(f"P{i}", [pat])
        matches = matcher(doc)
        if not matches:
            return 30
        _, start, end = matches[-1]
        span = doc[start:end]
        # Ищем числительное
        number = 1
        num_tokens = [t for t in span if t.pos_ == "NUM"]
        if num_tokens:
            num_text = num_tokens[-1].text
            if num_text.isdigit():
                number = int(num_text)
            else:
                number = 1  # для словесных числительных можно расширить
        # Ищем существительное с единицей времени
        unit_lemma = None
        for t in reversed(span):
            if t.pos_ == "NOUN":
                unit_lemma = t.lemma_
                break
        if unit_lemma is None:
            return 30
        if 'день' in unit_lemma:
            return number
        elif 'недел' in unit_lemma:
            return number * 7
        elif 'месяц' in unit_lemma:
            return number * 30
        elif 'квартал' in unit_lemma:
            return number * 90
        elif 'год' in unit_lemma or 'лет' in unit_lemma:
            return number * 365
        return number

    def predict_intent(self, query):
        """Возвращает (report_type, period_days) или (None, None)."""
        feat = self.processor.encode_batch([query])
        feat_scaled = self.scaler.transform(feat)
        if not self.clf_is_report.predict(feat_scaled)[0]:
            return None, None
        rtype = self._get_report_type(query)
        period = self._extract_period(query)
        return rtype, period