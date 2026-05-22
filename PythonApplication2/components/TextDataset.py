import torch
from typing import List, Dict
from torch.utils.data import Dataset
import numpy as np

class TextDataset(Dataset):
    """Dataset для обучения нейросети"""
    def __init__(self, texts: List[str], labels: List[np.ndarray], word2idx: Dict):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Преобразование текста в индексы
        indices = [self.word2idx.get(word, 1) for word in text.split()]  # 1 = UNK
        if len(indices) > 100:  # обрезаем до 100 токенов
            indices = indices[:100]
        else:  # паддинг
            indices = indices + [0] * (100 - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)