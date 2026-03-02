import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output, mask=None):
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = F.softmax(attention_weights, dim=1)
        
        if mask is not None:
            attention_weights = attention_weights * mask.unsqueeze(2)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        weighted_output = torch.sum(lstm_output * attention_weights, dim=1)
        return weighted_output, attention_weights


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 num_classes, dropout, pretrained_embeddings=None, use_attention=True):
        super(SentimentLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * 2)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
    
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        if lengths is not None:
            lengths = lengths.cpu()
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        lstm_output = self.layer_norm(lstm_output)
        
        if self.use_attention:
            mask = (x != 0).float()
            output, attention_weights = self.attention(lstm_output, mask)
        else:
            hidden_forward = hidden[-2]
            hidden_backward = hidden[-1]
            output = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits
    
    def predict(self, x, lengths=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


def create_model(config, pretrained_embeddings=None):
    model = SentimentLSTM(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        pretrained_embeddings=pretrained_embeddings,
        use_attention=True
    )
    
    model = model.to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return model


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config
    
    model = create_model(Config)
    print("\n模型架构:")
    print(model)
