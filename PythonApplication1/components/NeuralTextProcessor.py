import torch
from typing import List
from collections import Counter
from components.SimpleTextNN import SimpleTextNN
from components.TextDataset import TextDataset
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle
import os

class NeuralTextProcessor:
    """
    Обработка текста нейросетью для извлечения признаков
    """
    def __init__(self, embedding_dim: int = 100, hidden_dim: int = 128, output_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_vocabulary(self, texts: List[str]):
        """Построение словаря из текстов"""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Добавляем слова в словарь
        for word, count in word_counts.most_common(10000):  # ограничение часто встречающихся слов в 10000 слов
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def train(self, texts: List[str], labels: List[np.ndarray], epochs: int = 50, batch_size: int = 32):
        """Обучение нейросети"""
        print("🚀 Начало обучения нейросети...")
        
        # Построение словаря
        self.build_vocabulary(texts)
        
        # Создание модели
        self.model = SimpleTextNN(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim)
        self.model.to(self.device)
        
        # Подготовка данных
        dataset = TextDataset(texts, labels, self.word2idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Обучение
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Эпоха {epoch + 1}/{epochs}, Потери: {total_loss / len(dataloader):.4f}")
        
        print("✅ Обучение завершено!")
    
    def process(self, text: str) -> np.ndarray:
        """Обработка текста нейросетью, возвращает вектор признаков"""
        self.model.eval()
        
        # Преобразование текста
        indices = [self.word2idx.get(word, 1) for word in text.lower().split()]
        if len(indices) > 100:
            indices = indices[:100]
        else:
            indices = indices + [0] * (100 - len(indices))
        
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
        
        return features.cpu().numpy().flatten()
    
    def save(self, path: str):
        """Сохранение модели"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        with open(f"{path}/vocab.pkl", 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word, 'vocab_size': self.vocab_size}, f)
        print(f"💾 Модель сохранена в {path}")
    
    def load(self, path: str):
        """Загрузка модели"""
        self.model = SimpleTextNN(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim)
        self.model.load_state_dict(torch.load(f"{path}/model.pth", map_location=self.device))
        self.model.to(self.device)
        with open(f"{path}/vocab.pkl", 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
        print(f"📥 Модель загружена из {path}")