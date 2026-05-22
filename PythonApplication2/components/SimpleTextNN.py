import torch.nn as nn

class SimpleTextNN(nn.Module):
    """
    Простая нейросеть для обработки текста и извлечения признаков
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 128, output_dim: int = 64):
        super(SimpleTextNN, self).__init__()
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Сверточные слои для извлечения n-граммных признаков
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=4, padding=1)
        
        # Пулинг и полносвязные слои
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Эмбеддинги
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Свертки
        conv1_out = self.relu(self.conv1(embedded))  # (batch_size, 64, seq_len)
        conv2_out = self.relu(self.conv2(conv1_out))  # (batch_size, 64, seq_len)
        
        # Глобальный пулинг
        pooled = self.pool(conv2_out).squeeze(-1)  # (batch_size, 64)
        
        # Полносвязные слои
        fc1_out = self.relu(self.fc1(pooled))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)  # (batch_size, output_dim)
        
        return output