# 校园论坛内容审核系统

基于LSTM和注意力机制的情感分析与内容审核系统，支持对校园论坛用户生成内容进行自动化审核。

## 功能特性

- **情感分析**：基于BiLSTM+注意力机制的深度学习模型，预测文本情感倾向（正面/中性/负面）
- **内容审核**：内置多维度敏感词库，支持高危/中危风险识别与自动处理
- **RESTful API**：基于FastAPI的高性能Web服务，支持单条和批量预测
- **模型量化**：INT8动态量化优化，提升推理效率
- **热词管理**：支持运行时动态添加/删除敏感词

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.0+ |
| Web框架 | FastAPI + Uvicorn |
| 中文分词 | Jieba |
| 数据处理 | Pandas, NumPy |
| 模型评估 | Scikit-learn, Matplotlib |

## 目录结构

```
campus_forum_model/
├── config.py                 # 项目配置
├── requirements.txt          # 依赖清单
├── data/                     # 原始数据
│   ├── ChnSentiCorp_htl_all.csv   # 情感分析数据集
│   ├── waimai_10k.csv        # 外卖评论数据集
│   └── Vocabulary/           # 敏感词库
├── models/                   # 模型文件
│   ├── vocab/                # 词表（word2idx, idx2word）
│   ├── model_weights/        # 训练权重
│   └── data/                 # 预处理数据
├── results/                  # 训练结果
├── src/
│   ├── app.py               # FastAPI应用入口
│   ├── model.py             # LSTM模型定义
│   ├── train.py             # 模型训练脚本
│   ├── inference.py          # 推理引擎
│   ├── moderation.py         # 审核引擎
│   ├── preprocess.py         # 数据预处理
│   └── sensitive_words.py    # 敏感词管理
└── merge_vocab.py            # 词库合并工具
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python -m src.train
```

### 3. 启动API服务

```bash
python -m src.app
```

服务默认运行在 `http://localhost:8000`，访问 `/docs` 查看API文档。

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/predict` | POST | 单条情感预测 |
| `/predict/batch` | POST | 批量情感预测 |
| `/moderate` | POST | 单条内容审核 |
| `/moderate/batch` | POST | 批量内容审核 |
| `/keywords` | GET | 获取敏感词列表 |
| `/keywords` | POST | 添加敏感词 |
| `/keywords` | DELETE | 删除敏感词 |

## 敏感词库

系统内置以下类别的敏感词库：

- 政治类
- 涉枪涉爆类
- 色情类
- 反动类
- 贪腐类
- 暴恐类
- 涉COVID-19类
- 非法网址类
- 广告类
- 民生类

## 模型架构

```
SentimentLSTM
├── Embedding Layer (30000 vocab, 256 dim)
├── BiLSTM Layer 1 (64 hidden * 2)
├── BiLSTM Layer 2 (256 hidden * 2)
├── BiLSTM Layer 3 (512 hidden * 2)
├── Multi-Head Attention (4 heads)
├── Fully Connected Layers
└── Softmax Output (3 classes)
```

## 配置说明

主要超参数可在 `config.py` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| VOCAB_SIZE | 30000 | 词表大小 |
| MAX_LEN | 250 | 最大序列长度 |
| EMBEDDING_DIM | 256 | 嵌入维度 |
| HIDDEN_DIM | 512 | 隐藏层维度 |
| NUM_LAYERS | 3 | LSTM层数 |
| BATCH_SIZE | 64 | 批次大小 |
| LEARNING_RATE | 0.001 | 学习率 |
| NUM_EPOCHS | 30 | 训练轮数 |
