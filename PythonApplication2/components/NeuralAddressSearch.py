from components.NeuralTextProcessor import NeuralTextProcessor
from components.ServiceAddress import ServiceAddress
import numpy as np
from typing import List, Tuple, Dict
import json

class NeuralAddressSearch:
    """
    Система поиска адресов с использованием нейросети
    """
    def __init__(self, nlp_model, embedding_dim: int = 100):
        self.nlp = nlp_model
        self.neural_processor = NeuralTextProcessor(embedding_dim=embedding_dim)
        self.addresses: List['ServiceAddress'] = []
        self.address_features: List[np.ndarray] = []
        
    def add_address(self, address: ServiceAddress):
        """Добавление адреса в базу"""
        self.addresses.append(address)
    
    def add_addresses_from_list(self, addresses: List['ServiceAddress']):
        """Добавление списка адресов"""
        self.addresses.extend(addresses)
    
    def prepare_training_data(self) -> Tuple[List[str], List[np.ndarray]]:
        """
        Подготовка данных для обучения нейросети
        На основе ключевых слов создаются векторы-цели
        """
        texts = []
        labels = []
        
        for i, addr in enumerate(self.addresses):
            # Создание текста для обучения (комбинация заголовка, описания и ключевых слов)
            text = f"{addr.title} {addr.description} {' '.join(addr.keywords)}"
            texts.append(text)
            
            # Создание метки-вектора на основе ключевых слов
            # Простое представление: единицы на позициях ключевых слов
            label = np.zeros(64)  # размерность выхода нейросети
            
            # Распределение ключевых слов в векторе
            for j, keyword in enumerate(addr.keywords[:5]):  # не более 5 ключевых слов
                # Хэширование ключевого слова для получения позиции
                pos = hash(keyword) % 60
                label[pos] = 1.0 / (j + 1)  # убывающий вес
            
            # Добавление категории
            category_hash = hash(addr.category) % 4
            label[60 + category_hash] = 1.0
            
            labels.append(label)
        
        return texts, labels
    
    def train(self, epochs: int = 50):
        """Обучение нейросети на основе добавленных адресов"""
        texts, labels = self.prepare_training_data()
        
        if len(texts) == 0:
            print("⚠️ Нет данных для обучения")
            return
        
        self.neural_processor.train(texts, labels, epochs=epochs)
        
        # Вычисляем признаки для всех адресов
        self.address_features = []
        for addr in self.addresses:
            text = f"{addr.title} {addr.description} {' '.join(addr.keywords)}"
            features = self.neural_processor.process(text)
            self.address_features.append(features)
    
    def search(self, query: str, top_n: int = 5) -> List[Tuple['ServiceAddress', float, Dict]]:
        """
        Поиск наиболее релевантных адресов по запросу
        Возвращает список адресов с оценками релевантности и деталями
        """
        # Анализ запроса с помощью spaCy
        query_doc = self.nlp(query)
        
        # обработка запроса нейросетью
        query_features = self.neural_processor.process(query)
        
        # Вычисление сходства с каждым адресом
        scores = []
        for i, (addr, addr_features) in enumerate(zip(self.addresses, self.address_features)):
            # Косинусное сходство нейросетевых признаков
            cosine_sim = np.dot(query_features, addr_features) / (
                np.linalg.norm(query_features) * np.linalg.norm(addr_features) + 1e-8
            )
            
            # Бонус за совпадение ключевых слов
            keyword_bonus = 0
            matched_keywords = []
            query_lower = query.lower()
            for keyword in addr.keywords:
                if keyword.lower() in query_lower or query_lower in keyword.lower():
                    keyword_bonus += 0.2
                    matched_keywords.append(keyword)
            
            # Бонус за совпадение категории
            category_bonus = 0
            if addr.category.lower() in query_lower:
                category_bonus = 0.3
            
            # Бонус за важные слова из запроса (существительные и глаголы)
            pos_bonus = 0
            important_tokens = [token.text.lower() for token in query_doc 
                              if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop]
            for token in important_tokens:
                if token in addr.title.lower() or token in addr.description.lower():
                    pos_bonus += 0.1
            
            # Итоговая оценка
            total_score = cosine_sim + keyword_bonus + category_bonus + pos_bonus
            scores.append((addr, total_score, {
                'cosine_sim': cosine_sim,
                'keyword_bonus': keyword_bonus,
                'category_bonus': category_bonus,
                'pos_bonus': pos_bonus,
                'matched_keywords': matched_keywords,
                'important_tokens': important_tokens
            }))
        
        # Сортировка по убыванию релевантности
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Нормализация оценки после сортировки
        if scores:
            max_score = scores[0][1]  # максимальная оценка
            normalized_results = []
            for addr, score, details in scores:
               normalized_score = score / max_score if max_score > 0 else 0
               normalized_results.append((addr, normalized_score, details))
            return normalized_results[:top_n]
    
        return scores[:top_n]
    
    def save(self, path: str):
        """Сохранение системы"""
        self.neural_processor.save(f"{path}/neural_model")
        
        # Сохранение адресов
        addresses_data = []
        for addr in self.addresses:
            addresses_data.append({
                'url': addr.url,
                'title': addr.title,
                'description': addr.description,
                'keywords': addr.keywords,
                'category': addr.category
            })
        
        with open(f"{path}/addresses.json", 'w', encoding='utf-8') as f:
            json.dump(addresses_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Система сохранена в {path}")
    
    def load(self, path: str):
        """Загрузка системы"""
        self.neural_processor.load(f"{path}/neural_model")
        
        with open(f"{path}/addresses.json", 'r', encoding='utf-8') as f:
            addresses_data = json.load(f)
        
        self.addresses = []
        for data in addresses_data:
            self.addresses.append(ServiceAddress(
                url=data['url'],
                title=data['title'],
                description=data['description'],
                keywords=data['keywords'],
                category=data['category']
            ))
        
        # Восстанавление признаков
        self.address_features = []
        for addr in self.addresses:
            text = f"{addr.title} {addr.description} {' '.join(addr.keywords)}"
            features = self.neural_processor.process(text)
            self.address_features.append(features)
        
        print(f"Система загружена из {path}, найдено {len(self.addresses)} адресов")