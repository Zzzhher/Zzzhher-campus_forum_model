import torch
import torch.nn.functional as F
import pickle
import jieba
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model import SentimentLSTM


class SentimentInference:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else Config.DEVICE
        self.model = None
        self.word2idx = None
        self.idx2word = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        print(f"加载模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config_dict = checkpoint['config']
        
        self.model = SentimentLSTM(
            vocab_size=config_dict['vocab_size'],
            embedding_dim=config_dict['embedding_dim'],
            hidden_dim=config_dict['hidden_dim'],
            num_layers=config_dict['num_layers'],
            num_classes=config_dict['num_classes'],
            dropout=config_dict['dropout'],
            use_attention=True
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功 (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%)")
        
        self.load_vocab()
    
    def load_vocab(self):
        print("加载词表...")
        
        with open(Config.WORD2IDX_FILE, 'rb') as f:
            self.word2idx = pickle.load(f)
        
        with open(Config.IDX2WORD_FILE, 'rb') as f:
            self.idx2word = pickle.load(f)
        
        print(f"词表加载成功 (大小: {len(self.word2idx)})")
    
    def tokenize_text(self, text):
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []
        
        tokens = list(jieba.cut(text))
        tokens = [token for token in tokens if token.strip()]
        return tokens
    
    def text_to_sequence(self, text):
        tokens = self.tokenize_text(text)
        sequence = [self.word2idx.get(token, self.word2idx.get('<UNK>', 1)) for token in tokens]
        
        if len(sequence) > Config.MAX_LEN:
            sequence = sequence[:Config.MAX_LEN]
        else:
            sequence = sequence + [self.word2idx.get('<PAD>', 0)] * (Config.MAX_LEN - len(sequence))
        
        return sequence
    
    def predict(self, text, return_probabilities=False):
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model() 方法")
        
        sequence = self.text_to_sequence(text)
        sequence_tensor = torch.LongTensor([sequence]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        label_names = {0: '负向', 1: '正向', 2: '中性'}
        label = label_names[predicted_class]
        
        result = {
            'text': text,
            'predicted_class': predicted_class,
            'predicted_label': label,
            'confidence': probabilities[0][predicted_class].item()
        }
        
        if return_probabilities:
            result['probabilities'] = {
                '负向': probabilities[0][0].item(),
                '正向': probabilities[0][1].item(),
                '中性': probabilities[0][2].item()
            }
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model() 方法")
        
        sequences = [self.text_to_sequence(text) for text in texts]
        sequences_tensor = torch.LongTensor(sequences).to(self.device)
        
        with torch.no_grad():
            logits = self.model(sequences_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
        
        label_names = {0: '负向', 1: '正向', 2: '中性'}
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'predicted_class': int(predicted_classes[i]),
                'predicted_label': label_names[predicted_classes[i]],
                'confidence': probabilities[i][predicted_classes[i]].item()
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    '负向': probabilities[i][0].item(),
                    '正向': probabilities[i][1].item(),
                    '中性': probabilities[i][2].item()
                }
            
            results.append(result)
        
        return results
    
    def get_attention_weights(self, text):
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model() 方法")
        
        sequence = self.text_to_sequence(text)
        sequence_tensor = torch.LongTensor([sequence]).to(self.device)
        
        with torch.no_grad():
            embedded = self.model.embedding(sequence_tensor)
            lstm_output, (hidden, cell) = self.model.lstm(embedded)
            lstm_output = self.model.layer_norm(lstm_output)
            
            mask = (sequence_tensor != 0).float()
            attention_weights = self.model.attention.attention(lstm_output)
            attention_weights = F.softmax(attention_weights, dim=1)
            attention_weights = attention_weights * mask.unsqueeze(2)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        tokens = self.tokenize_text(text)
        attention_weights = attention_weights[0].cpu().numpy().flatten()[:len(tokens)]
        
        word_attention = []
        for i, token in enumerate(tokens):
            if i < len(attention_weights):
                word_attention.append({
                    'word': token,
                    'attention': attention_weights[i]
                })
        
        word_attention.sort(key=lambda x: x['attention'], reverse=True)
        
        return word_attention[:10]


def main():
    print("=" * 60)
    print("情感分析推理测试")
    print("=" * 60)
    
    model_path = Config.BEST_MODEL_FILE
    
    inference = SentimentInference(model_path)
    
    test_texts = [
        "这个酒店真的太棒了，服务很好，房间也很干净！",
        "外卖送得太慢了，而且味道也不好，非常失望。",
        "这本书还可以，没有什么特别的地方，一般般。",
        "非常糟糕的体验，再也不会来了！",
        "价格合理，性价比不错，推荐购买。"
    ]
    
    print("\n" + "=" * 60)
    print("单条文本预测")
    print("=" * 60)
    
    for text in test_texts:
        result = inference.predict(text, return_probabilities=True)
        print(f"\n文本: {text}")
        print(f"预测: {result['predicted_label']} (置信度: {result['confidence']:.4f})")
        print(f"概率分布: 负向={result['probabilities']['负向']:.4f}, "
              f"正向={result['probabilities']['正向']:.4f}, "
              f"中性={result['probabilities']['中性']:.4f}")
    
    print("\n" + "=" * 60)
    print("批量预测")
    print("=" * 60)
    
    batch_results = inference.predict_batch(test_texts[:3])
    for result in batch_results:
        print(f"\n文本: {result['text']}")
        print(f"预测: {result['predicted_label']} (置信度: {result['confidence']:.4f})")
    
    print("\n" + "=" * 60)
    print("注意力权重分析")
    print("=" * 60)
    
    attention_result = inference.get_attention_weights(test_texts[0])
    print(f"\n文本: {test_texts[0]}")
    print("最重要的词语（按注意力权重排序）:")
    for i, item in enumerate(attention_result, 1):
        print(f"  {i}. {item['word']}: {item['attention']:.4f}")


if __name__ == '__main__':
    main()
