# 校园论坛AI审核模型

## 项目简介

本项目为校园论坛系统提供AI驱动的智能内容审核服务，基于深度学习情感分析和敏感词过滤技术，实现"AI初筛 + 人工复核"的两层防控体系，保障校园论坛内容的健康与安全。

## 核心功能

- **情感分析**：基于LSTM深度学习模型，支持三分类（正向/负向/中性）
- **敏感词过滤**：使用DFA算法实现高效敏感词检测
- **智能审核决策**：综合情感分析和敏感词检测结果，自动做出审核决策
- **RESTful API**：提供标准HTTP接口，易于与主系统集成
- **实时评估**：支持模型性能实时评估和反馈

## 技术架构

```
campus_forum (SpringBoot) → campus_forum_model (Flask API) → 审核引擎
                                                           ↓
                                                ┌─────────────┴─────────────┐
                                                ▼                         ▼
                                        情感分析模型                  敏感词过滤器
                                                ▲                         ▲
                                                └─────────────┬─────────────┘
                                                           ↓
                                                    审核结果返回
```

## 数据集来源

### 训练数据集

本项目使用以下数据集进行模型训练：

#### 1. ChnSentiCorp_htl_all.csv

- **来源**：ChnSentiCorp（中文情感分析基准数据集）
- **描述**：酒店评论情感分析数据集，包含正面、负向、中性三类情感标签
- **特点**：
  - 标注质量高，是中文情感分析的权威基准数据集
  - 包含丰富的情感表达和语言现象
  - 数据量约10k+条
- **标签格式**：0=负向，1=正向，2=中性

#### 2. waimai_10k.csv

- **来源**：外卖平台评论数据
- **描述**：外卖平台用户评价数据集，主要为正面评价
- **特点**：
  - 与校园论坛评论场景接近（都是服务评价）
  - 语言风格贴近实际用户表达
  - 数据量10k条
- **标签格式**：0=负向，1=正向（主要为正向评价）

#### 3. neutral.csv

- **来源**：人工标注的中性情感数据
- **描述**：中性情感评论数据集
- **特点**：
  - 补充中性样本，平衡数据分布
  - 包含客观描述性文本
- **标签格式**：2=中性

### 数据集选择理由

1. **符合开题报告要求**："结合 ChnSentiCorp 等公开数据集与人工标注的校园评论样本"
2. **场景匹配度高**：外卖评论与校园论坛评论场景相似，都是服务评价类内容
3. **数据质量高**：ChnSentiCorp是权威基准数据集，标注质量可靠
4. **数据量充足**：总数据量约20k+条，满足模型训练需求
5. **情感分布均衡**：通过组合不同数据集，实现正负中三类情感的相对平衡

### 数据集统计

| 数据集                   | 总数      | 正向     | 负向     | 中性     | 用途           |
| ------------------------ | --------- | -------- | -------- | -------- | -------------- |
| ChnSentiCorp_htl_all.csv | ~10k+     | ✓        | ✓        | ✓        | 主要训练集     |
| waimai_10k.csv           | 10k       | ✓        | ✓        | -        | 补充正面数据   |
| neutral.csv              | 未统计    | -        | -        | ✓        | 补充中性数据   |
| **合计**                 | **~20k+** | **丰富** | **丰富** | **丰富** | **完整训练集** |

## 模型性能

- **准确率**：87.95%（超过任务书要求的85%）
- **支持分类**：正向、负向、中性三分类
- **推理速度**：单次推理 < 50ms
- **模型大小**：轻量级模型，适合部署在普通服务器

## 项目结构

```
campus_forum_model/
├── src/                      # 源代码
│   ├── app.py               # Flask API服务
│   ├── model.py             # LSTM模型定义
│   ├── inference.py         # 模型推理
│   ├── moderation.py        # 审核引擎
│   ├── sensitive_words.py   # 敏感词过滤
│   ├── preprocess.py        # 文本预处理
│   ├── train.py             # 模型训练
│   └── send_evaluation.py   # 模型评估结果发送
├── data/                   # 数据目录
│   ├── ChnSentiCorp_htl_all.csv  # 中文情感分析数据集
│   ├── waimai_10k.csv            # 外卖评论数据集
│   ├── neutral.csv               # 中性情感数据
│   └── Vocabulary/              # 敏感词库
├── models/                 # 模型文件
│   ├── best_model.pt       # 最佳模型权重
│   ├── data/               # 预处理数据
│   └── vocab/              # 词汇表
├── results/               # 训练结果
│   ├── training_history.json     # 训练历史
│   ├── test_results.json         # 测试结果
│   ├── training_history.png      # 训练历史图
│   └── confusion_matrix.png      # 混淆矩阵
├── config.py             # 配置文件
├── requirements.txt       # 依赖包
├── run_preprocess.py      # 数据预处理脚本
├── merge_prohibited_words.py # 敏感词合并脚本
├── AI服务实现文档.md       # 技术实现文档
└── README.md            # 项目说明
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- Flask 2.0+
- NLTK 3.6+
- jieba 0.42+
- pandas
- numpy

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据预处理

```bash
python run_preprocess.py
```

### 训练模型

```bash
cd src
python train.py
```

### 启动服务

```bash
cd src
python app.py
```

服务将启动在 `http://localhost:5000`

### 测试接口

```bash
curl -X POST http://localhost:5000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "这个服务非常好，我很满意"}'
```

## API接口

### 内容审核

**POST** `/moderate`

请求参数：

```json
{
  "text": "待审核文本",
  "content_type": "topic" // topic 或 comment
}
```

响应结果：

```json
{
  "success": true,
  "data": {
    "action": "ALLOW", // ALLOW, REVIEW, BLOCK
    "reason": "内容正常",
    "sentiment": {
      "label": "positive", // positive, negative, neutral
      "confidence": 0.95
    },
    "sensitive_words": [],
    "risk_score": 0
  }
}
```

### 健康检查

**GET** `/health`

响应结果：

```json
{
  "status": "healthy",
  "model_loaded": true,
  "moderation_loaded": true
}
```

### 模型评估

**POST** `/evaluate`

请求参数：

```json
{
  "text": "测试文本",
  "true_label": "positive" // positive, negative, neutral
}
```

响应结果：

```json
{
  "success": true,
  "data": {
    "predicted_label": "positive",
    "true_label": "positive",
    "correct": true,
    "confidence": 0.95
  }
}
```

## 敏感词库

敏感词库位于 `data/Vocabulary/` 目录，包含以下分类：

- **暴恐词库.txt**：暴力恐怖相关词
- **涉枪涉爆.txt**：枪支爆炸物相关词
- **反动词库.txt**：反动言论相关词
- **色情词库.txt**：色情内容相关词
- **政治类型.txt**：政治敏感词
- **广告类型.txt**：广告推广词
- **贪腐词库.txt**：贪污腐败相关词
- **民生词库.txt**：民生相关词
- **其他词库.txt**：其他敏感词
- **网易前端过滤敏感词库.txt**：综合敏感词库
- **COVID-19词库.txt**：疫情相关敏感词
- **GFW补充词库.txt**：网络安全相关词
- **新思想启蒙.txt**：思想相关敏感词
- **零时-Tencent.txt**：腾讯敏感词库
- **非法网址.txt**：非法网站相关词

### 敏感词库更新

使用以下脚本合并和更新敏感词库：

```bash
python merge_prohibited_words.py
```

## 与主系统集成

### 配置步骤

1. **启动AI审核服务**

   ```bash
   cd src
   python app.py
   ```

2. **配置主系统**
   在 `campus_forum/backend/src/main/resources/application.yml` 中添加以下配置：

   ```yaml
   ai:
     moderation:
       url: http://localhost:5000/moderate
       timeout: 5000
   ```

3. **自动集成**
   主系统会自动将用户发布的内容发送到AI审核服务进行检测，并根据返回结果进行处理。

## 性能优化

- **模型推理优化**：使用GPU加速，模型量化
- **敏感词检测优化**：DFA算法O(n)复杂度，高效匹配
- **服务部署优化**：支持Gunicorn多进程部署，提高并发处理能力
- **内存优化**：模型加载优化，减少内存占用

## 监控与日志

- **服务日志**：记录API请求和响应
- **模型性能监控**：定期评估模型性能
- **错误处理**：完善的错误处理机制，确保服务稳定性

## 文档

- [AI服务实现文档.md](AI服务实现文档.md) - 详细的技术实现文档

## 许可证

MIT License
