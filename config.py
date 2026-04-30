import os
import torch


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    VOCAB_DIR = os.path.join(MODELS_DIR, 'vocab')
    DATA_PROCESSED_DIR = os.path.join(MODELS_DIR, 'data')
    MODEL_WEIGHTS_DIR = os.path.join(MODELS_DIR, 'model_weights')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    os.makedirs(VOCAB_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    VOCAB_SIZE = 30000
    MAX_LEN = 250
    
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 3
    DROPOUT = 0.5
    NUM_CLASSES = 3
    
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    GRADIENT_CLIP = 5.0
    
    DEVICE = 'cuda' if (os.environ.get('USE_GPU', '1') == '1' and torch.cuda.is_available()) else 'cpu'
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    RANDOM_SEED = 42
    
    WORD2IDX_FILE = os.path.join(VOCAB_DIR, 'word2idx.pkl')
    IDX2WORD_FILE = os.path.join(VOCAB_DIR, 'idx2word.pkl')
    PREPROCESSED_DATA_FILE = os.path.join(DATA_PROCESSED_DIR, 'preprocessed_data.pkl')
    BEST_MODEL_FILE = os.path.join(MODEL_WEIGHTS_DIR, 'best_model.pth')
    LAST_MODEL_FILE = os.path.join(MODEL_WEIGHTS_DIR, 'last_model.pth')
    
    @classmethod
    def display(cls):
        print("=" * 50)
        print("配置信息")
        print("=" * 50)
        print(f"基础目录: {cls.BASE_DIR}")
        print(f"数据目录: {cls.DATA_DIR}")
        print(f"模型目录: {cls.MODELS_DIR}")
        print(f"词表目录: {cls.VOCAB_DIR}")
        print(f"预处理数据目录: {cls.DATA_PROCESSED_DIR}")
        print(f"模型权重目录: {cls.MODEL_WEIGHTS_DIR}")
        print(f"结果目录: {cls.RESULTS_DIR}")
        print("-" * 50)
        print(f"词表大小: {cls.VOCAB_SIZE}")
        print(f"最大序列长度: {cls.MAX_LEN}")
        print("-" * 50)
        print(f"嵌入维度: {cls.EMBEDDING_DIM}")
        print(f"隐藏层维度: {cls.HIDDEN_DIM}")
        print(f"LSTM层数: {cls.NUM_LAYERS}")
        print(f"Dropout率: {cls.DROPOUT}")
        print(f"分类数: {cls.NUM_CLASSES}")
        print("-" * 50)
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"梯度裁剪: {cls.GRADIENT_CLIP}")
        print("-" * 50)
        print(f"训练集比例: {cls.TRAIN_RATIO}")
        print(f"验证集比例: {cls.VAL_RATIO}")
        print(f"测试集比例: {cls.TEST_RATIO}")
        print("-" * 50)
        print(f"随机种子: {cls.RANDOM_SEED}")
        print(f"设备: {cls.DEVICE}")
        print("=" * 50)


if __name__ == '__main__':
    Config.display()
