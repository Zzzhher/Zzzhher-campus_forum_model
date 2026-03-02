import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns
import time
import json
import requests
from datetime import datetime
from torch.amp import autocast, GradScaler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model import create_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

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

        # 后端API配置
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:8080')
        self.api_token = os.environ.get('API_TOKEN', '')

    def _get_class_weights(self):
        weight_file = os.path.join(self.config.RESULTS_DIR, 'class_weights.json')
        if os.path.exists(weight_file):
            with open(weight_file, 'r') as f:
                weights = json.load(f)
            return torch.tensor(weights, dtype=torch.float32).to(self.device)

        return None

    def load_data(self):
        print("\n加载数据...")

        # 从预处理数据文件中加载数据
        data = pickle.load(open(self.config.PREPROCESSED_DATA_FILE, 'rb'))
        train_data = data['train']
        val_data = data['val']
        test_data = data['test']

        train_dataset = SentimentDataset(train_data['sequences'], train_data['labels'])
        val_dataset = SentimentDataset(val_data['sequences'], val_data['labels'])
        test_dataset = SentimentDataset(test_data['sequences'], test_data['labels'])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f"训练样本: {len(train_dataset)}")
        print(f"验证样本: {len(val_dataset)}")
        print(f"测试样本: {len(test_dataset)}")

        return len(train_dataset), len(val_dataset), len(test_dataset)

    def setup_model(self):
        print("\n初始化模型...")
        self.model = create_model(self.config).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [训练]')

        start_time = time.time()

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

        end_time = time.time()
        epoch_time = end_time - start_time

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        print(f"  训练完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%, 时间: {epoch_time:.2f}s")

        return avg_loss, accuracy, epoch_time

    def validate(self, loader, desc='验证'):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        pbar = tqdm(loader, desc=desc)

        start_time = time.time()

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

        end_time = time.time()
        epoch_time = end_time - start_time

        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total

        print(f"  {desc}完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%, 时间: {epoch_time:.2f}s")

        return avg_loss, accuracy, all_predictions, all_labels

    def train(self):
        print("=" * 60)
        print("开始训练")
        print("=" * 60)

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()

            train_loss, train_acc, train_time = self.train_epoch(epoch)
            val_loss, val_acc, _, _ = self.validate(self.val_loader, '验证')

            epoch_time = time.time() - epoch_start_time

            current_lr = self.optimizer.param_groups[0]['lr']

            # 确保历史记录列表已初始化
            if 'train_loss' not in self.history:
                self.history['train_loss'] = []
                self.history['train_acc'] = []
                self.history['val_loss'] = []
                self.history['val_acc'] = []
                self.history['learning_rate'] = []
                self.history['epoch_time'] = []

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} 完成:")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  耗时: {epoch_time:.2f}s")

            self.scheduler.step(val_acc)

            self.early_stopping(val_acc, self.model)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(
                    self.config.BEST_MODEL_FILE,
                    epoch, val_acc
                )
                print(f"  保存最佳模型 (Val Acc: {val_acc:.2f}%)")

            if self.early_stopping.early_stop:
                print(f"\n早停触发! 最佳验证准确率: {self.early_stopping.best_score:.2f}%")
                break

        self.model.load_state_dict(self.early_stopping.best_model_state)
        self.save_training_history()
        self.plot_training_history()

        return self.history

    def evaluate(self):
        print("\n" + "=" * 60)
        print("最终评估")
        print("=" * 60)

        test_loss, test_acc, predictions, labels = self.validate(self.test_loader, '测试')

        print(f"\n测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")

        label_names = ['负向', '正向', '中性']

        print(f"\n分类报告:")
        report = classification_report(labels, predictions, target_names=label_names, digits=4, output_dict=True)
        print(classification_report(labels, predictions, target_names=label_names, digits=4))

        # 计算各项指标
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision_macro = precision_score(labels, predictions, average='macro')
        recall_macro = recall_score(labels, predictions, average='macro')
        accuracy = accuracy_score(labels, predictions)

        print(f"\n宏平均指标:")
        print(f"  F1-Score: {f1_macro:.4f}")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall: {recall_macro:.4f}")

        print(f"\n加权平均指标:")
        print(f"  F1-Score: {f1_weighted:.4f}")

        self.plot_confusion_matrix(labels, predictions, label_names)

        # 构建混淆矩阵JSON
        cm = confusion_matrix(labels, predictions)
        confusion_matrix_json = {
            'labels': label_names,
            'matrix': cm.tolist()
        }

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': confusion_matrix_json,
            'class_report': report
        }

        results_file = os.path.join(self.config.RESULTS_DIR, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 发送评估指标到后端
        self._send_evaluation_to_backend(results, len(labels))

        return results

    def _send_evaluation_to_backend(self, results, test_samples):
        """将模型评估指标发送到后端API进行持久化"""
        try:
            print("\n正在将评估指标发送到后端...")

            # 生成模型版本号
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 构建评估数据
            evaluation_data = {
                'modelVersion': model_version,
                'accuracy': round(results['accuracy'], 4),
                'precisionScore': round(results['precision_macro'], 4),
                'recall': round(results['recall_macro'], 4),
                'f1Score': round(results['f1_macro'], 4),
                'evalDate': datetime.now().strftime('%Y-%m-%d'),
                'evalTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'testSamples': test_samples,
                'trainingSamples': len(self.train_loader.dataset) if self.train_loader else 0,
                'epochs': len(self.history['train_loss']),
                'batchSize': self.config.BATCH_SIZE,
                'learningRate': self.config.LEARNING_RATE,
                'modelPath': self.config.BEST_MODEL_FILE,
                'confusionMatrix': json.dumps(results.get('confusion_matrix', {})),
                'classReport': json.dumps(results.get('class_report', {})),
                'isBest': 0,  # 默认不是最佳，需要管理员确认
                'remark': f'自动训练完成，验证准确率: {self.best_val_acc:.2f}%'
            }

            # 发送请求到后端
            url = f"{self.backend_url}/api/admin/model/evaluations"
            headers = {
                'Content-Type': 'application/json'
            }
            if self.api_token:
                headers['Authorization'] = f'Bearer {self.api_token}'

            response = requests.post(url, json=evaluation_data, headers=headers, timeout=10)

            if response.status_code == 200:
                print(f"✓ 评估指标已成功发送到后端 (版本: {model_version})")
                print(f"  发送的数据: {json.dumps(evaluation_data, indent=2, default=str)}")
            else:
                print(f"✗ 发送评估指标失败: HTTP {response.status_code}")
                print(f"  响应: {response.text}")
                print(f"  请求数据: {json.dumps(evaluation_data, indent=2, default=str)}")

        except requests.exceptions.ConnectionError:
            print("✗ 无法连接到后端服务，评估指标未持久化")
            print(f"  请确保后端服务运行在 {self.backend_url}")
        except Exception as e:
            print(f"✗ 发送评估指标时出错: {str(e)}")
            print(f"  评估指标已保存到本地文件")

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

        confusion_matrix_path = os.path.join(self.config.RESULTS_DIR, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图表已保存到: {confusion_matrix_path}")
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
