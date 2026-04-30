import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import time
import json
from torch.amp import autocast, GradScaler
import sys
import random
import jieba
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model import create_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def data_augmentation(text, augmentation_rate=0.3):
    """
    数据增强函数
    1. 随机删除部分词语
    2. 同义词替换
    3. 随机打乱句子顺序
    """
    # 同义词词典（简单示例）
    synonyms = {
        '很好': ['不错', '很棒', '优秀', '出色'],
        '很差': ['糟糕', '差劲', '不好', '低劣'],
        '喜欢': ['喜爱', '喜欢', '钟爱', '热爱'],
        '讨厌': ['厌恶', '反感', '不喜欢', '憎恶'],
        '开心': ['高兴', '愉快', '快乐', '喜悦'],
        '难过': ['悲伤', '伤心', '痛苦', '忧伤']
    }
    
    # 分词
    words = list(jieba.cut(text))
    
    # 随机删除部分词语
    if random.random() < augmentation_rate:
        keep_rate = random.uniform(0.7, 0.9)
        words = [word for word in words if random.random() < keep_rate]
    
    # 同义词替换
    if random.random() < augmentation_rate:
        for i, word in enumerate(words):
            if word in synonyms and random.random() < 0.5:
                words[i] = random.choice(synonyms[word])
    
    # 随机打乱句子顺序（如果有多个句子）
    if random.random() < augmentation_rate and len(words) > 5:
        # 简单地随机打乱词语顺序
        random.shuffle(words)
    
    return ''.join(words)


class SentimentDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        self.best_model_state = model.state_dict().copy()


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.criterion = nn.CrossEntropyLoss(weight=self._get_class_weights())
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler('cuda') if self.device == 'cuda' else None
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_val_acc = 0.0
        
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def _get_class_weights(self):
        weight_file = os.path.join(self.config.RESULTS_DIR, 'class_weights.json')
        if os.path.exists(weight_file):
            with open(weight_file, 'r') as f:
                weights = json.load(f)
            return torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        return None
    
    def load_data(self):
        print("加载预处理数据...")
        
        with open(self.config.PREPROCESSED_DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        
        train_sequences = data['train']['sequences']
        train_labels = data['train']['labels']
        val_sequences = data['val']['sequences']
        val_labels = data['val']['labels']
        test_sequences = data['test']['sequences']
        test_labels = data['test']['labels']
        
        print(f"数据集大小:")
        print(f"  训练集: {len(train_sequences)}")
        print(f"  验证集: {len(val_sequences)}")
        print(f"  测试集: {len(test_sequences)}")
        
        self._calculate_class_weights(train_labels)
        
        train_dataset = SentimentDataset(train_sequences, train_labels)
        val_dataset = SentimentDataset(val_sequences, val_labels)
        test_dataset = SentimentDataset(test_sequences, test_labels)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        print(f"批次数: 训练={len(self.train_loader)}, 验证={len(self.val_loader)}, 测试={len(self.test_loader)}")
    
    def _calculate_class_weights(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        weights = []
        for label in sorted(unique):
            weight = total / (len(unique) * counts[np.where(unique == label)[0][0]])
            weights.append(weight)
        
        print(f"\n类别权重:")
        for label, weight in zip(sorted(unique), weights):
            label_name = {0: '负向', 1: '正向', 2: '中性'}[label]
            print(f"  {label_name} ({label}): {weight:.4f}")
        
        weight_file = os.path.join(self.config.RESULTS_DIR, 'class_weights.json')
        with open(weight_file, 'w') as f:
            json.dump(weights, f)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self.device)
        )
    
    def setup_model(self):
        print("\n初始化模型...")
        self.model = create_model(self.config)
        
        # 尝试使用RAdam优化器
        try:
            from torch.optim import RAdam
            self.optimizer = RAdam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=0.01
            )
            print(f"优化器: RAdam")
        except ImportError:
            # 如果RAdam不可用，使用AdamW作为备选
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=0.01
            )
            print(f"优化器: AdamW (RAdam不可用)")
        
        # 使用余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.NUM_EPOCHS,
            eta_min=1e-6
        )
        
        print(f"学习率调度器: CosineAnnealingLR")
        print(f"初始学习率: {self.config.LEARNING_RATE}")
        print(f"设备: {self.device}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [训练]')
        
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.device == 'cuda' and self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, loader, desc='验证'):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(loader, desc=desc)
        
        with torch.no_grad():
            for sequences, labels in pbar:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                if self.device == 'cuda' and self.scaler is not None:
                    with autocast('cuda'):
                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self):
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, _, _ = self.validate(self.val_loader, '验证')
            
            epoch_time = time.time() - epoch_start_time
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} - {epoch_time:.2f}s")
            print(f"训练集 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"验证集 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}")
            
            # 余弦退火学习率调度器不需要传入验证准确率
            self.scheduler.step()
            
            if current_lr != self.optimizer.param_groups[0]['lr']:
                print(f"学习率调整: {current_lr:.6f} -> {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(self.config.BEST_MODEL_FILE, epoch, val_acc)
                print(f"保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            
            self.save_model(self.config.LAST_MODEL_FILE, epoch, val_acc)
            
            self.early_stopping(val_acc, self.model)
            
            if self.early_stopping.early_stop:
                print(f"\n早停触发！在epoch {epoch+1}停止训练")
                if self.early_stopping.best_model_state is not None:
                    self.model.load_state_dict(self.early_stopping.best_model_state)
                    print("已加载最佳模型权重")
                break
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print("=" * 60)
        
        self.plot_training_history()
        self.save_training_history()
        
        return self.history
    
    def evaluate(self):
        print("\n" + "=" * 60)
        print("在测试集上评估模型")
        print("=" * 60)
        
        self.model.eval()
        test_loss, test_acc, predictions, labels = self.validate(self.test_loader, '测试')
        
        print(f"\n测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        
        label_names = ['负向', '正向', '中性']
        
        print(f"\n分类报告:")
        print(classification_report(labels, predictions, target_names=label_names, digits=4))
        
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision_macro = precision_score(labels, predictions, average='macro')
        recall_macro = recall_score(labels, predictions, average='macro')
        
        print(f"\n宏平均指标:")
        print(f"  F1-Score: {f1_macro:.4f}")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall: {recall_macro:.4f}")
        
        print(f"\n加权平均指标:")
        print(f"  F1-Score: {f1_weighted:.4f}")
        
        self.plot_confusion_matrix(labels, predictions, label_names)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
        
        results_file = os.path.join(self.config.RESULTS_DIR, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def save_model(self, filepath, epoch, val_acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': {
                'vocab_size': self.config.VOCAB_SIZE,
                'embedding_dim': self.config.EMBEDDING_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'num_layers': self.config.NUM_LAYERS,
                'num_classes': self.config.NUM_CLASSES,
                'dropout': self.config.DROPOUT
            }
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"从 {filepath} 加载模型 (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%)")
    
    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('训练和验证损失', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('训练和验证准确率', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('学习率变化', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, self.history['epoch_time'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
        axes[1, 1].set_title('每个Epoch的训练时间', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        history_plot = os.path.join(self.config.RESULTS_DIR, 'training_history.png')
        plt.savefig(history_plot, dpi=300, bbox_inches='tight')
        print(f"\n训练历史图表已保存到: {history_plot}")
        plt.close()

    def plot_confusion_matrix(self, labels, predictions, label_names):
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names,
                    cbar_kws={'label': '样本数量'})
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        cm_plot = os.path.join(self.config.RESULTS_DIR, 'confusion_matrix.png')
        plt.savefig(cm_plot, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {cm_plot}")
        plt.close()
    
    def save_training_history(self):
        history_file = os.path.join(self.config.RESULTS_DIR, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"训练历史已保存到: {history_file}")


def main():
    print("=" * 60)
    print("校园论坛情感分析 - 模型训练")
    print("=" * 60)
    
    print(f"\n设备信息:")
    print(f"  使用设备: {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"\n配置信息:")
    Config.display()
    
    trainer = Trainer(Config)
    
    trainer.load_data()
    trainer.setup_model()
    
    history = trainer.train()
    
    results = trainer.evaluate()
    
    print("\n" + "=" * 60)
    print("训练和评估完成!")
    print("=" * 60)
    print(f"\n最终测试结果:")
    print(f"  准确率: {results['test_accuracy']:.2f}%")
    print(f"  宏平均F1: {results['f1_macro']:.4f}")
    print(f"  加权平均F1: {results['f1_weighted']:.4f}")


if __name__ == '__main__':
    main()
