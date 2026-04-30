import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "隐藏维度必须能被注意力头数整除"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, lstm_output, mask=None):
        batch_size = lstm_output.size(0)
        seq_len = lstm_output.size(1)
        
        # 线性变换并分割为多个注意力头
        Q = self.query(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            energy = energy.masked_fill(mask == 0, -1e4)
        
        # 计算注意力权重
        attention_weights = F.softmax(energy, dim=-1)
        
        # 应用注意力权重到值
        out = torch.matmul(attention_weights, V)
        
        # 重新组合注意力头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 最终线性变换
        out = self.fc_out(out)
        
        # 对序列维度求和
        weighted_output = torch.sum(out, dim=1)
        
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
        
        # 渐进式LSTM层设计
        self.lstm_layers = nn.ModuleList()
        input_dim = embedding_dim
        
        # 第一层LSTM
        self.lstm_layers.append(nn.LSTM(
            input_dim,
            64,
            1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        ))
        
        # 第二层LSTM
        self.lstm_layers.append(nn.LSTM(
            64 * 2,
            256,
            1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        ))
        
        # 第三层LSTM
        self.lstm_layers.append(nn.LSTM(
            256 * 2,
            512,
            1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        ))
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(64 * 2),
            nn.LayerNorm(256 * 2),
            nn.LayerNorm(512 * 2)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        if use_attention:
            self.attention = MultiHeadAttention(512 * 2)
            self.fc = nn.Linear(512 * 2, num_classes)
        else:
            self.fc = nn.Linear(512 * 2, num_classes)
        
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
            elif 'linear' in name or 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        current_input = embedded
        
        # 逐层处理LSTM
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            if lengths is not None:
                lengths = lengths.cpu()
                packed_embedded = nn.utils.rnn.pack_padded_sequence(
                    current_input, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, (hidden, cell) = lstm(packed_embedded)
                lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                lstm_output, (hidden, cell) = lstm(current_input)
            
            # 应用层归一化
            lstm_output = layer_norm(lstm_output)
            lstm_output = self.dropout(lstm_output)
            
            current_input = lstm_output
        
        # 最终的LSTM输出
        final_output = current_input
        
        if self.use_attention:
            mask = (x != 0).float()
            output, attention_weights = self.attention(final_output, mask)
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
