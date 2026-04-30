# -*- coding: utf-8 -*-
"""
合并敏感词库脚本
将Vocabulary目录下的词库文件合并到high_risk_words.json和medium_risk_words.json
"""

import os
import json

# 定义高危词库文件
HIGH_RISK_FILES = [
    "暴恐词库.txt",
    "色情类型.txt",
    "涉枪涉爆.txt",
    "贪腐词库.txt",
    "政治类型.txt"
]

# 词库目录
VOCAB_DIR = r"d:\Projects\Python\campus_forum_model\models\data\Vocabulary"
HIGH_RISK_OUTPUT = r"d:\Projects\Python\campus_forum_model\models\data\high_risk_words.json"
MEDIUM_RISK_OUTPUT = r"d:\Projects\Python\campus_forum_model\models\data\medium_risk_words.json"


def load_words_from_file(filepath):
    """从词库文件加载词汇"""
    words = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    words.append(word)
    except Exception as e:
        print(f"加载文件失败 {filepath}: {e}")
    return words


def main():
    # 获取所有词库文件
    all_files = set(os.listdir(VOCAB_DIR))

    # 加载高危词库
    high_risk_words = []
    for filename in HIGH_RISK_FILES:
        filepath = os.path.join(VOCAB_DIR, filename)
        if os.path.exists(filepath):
            words = load_words_from_file(filepath)
            high_risk_words.extend(words)
            print(f"加载高危词库 {filename}: {len(words)} 个词")
        else:
            print(f"文件不存在: {filepath}")

    # 加载中危词库 (排除高危词库文件)
    medium_risk_words = []
    for filename in all_files:
        if filename not in HIGH_RISK_FILES and filename.endswith('.txt'):
            filepath = os.path.join(VOCAB_DIR, filename)
            words = load_words_from_file(filepath)
            medium_risk_words.extend(words)
            print(f"加载中危词库 {filename}: {len(words)} 个词")

    # 去重
    high_risk_words = list(set(high_risk_words))
    medium_risk_words = list(set(medium_risk_words))

    # 保存高危词库
    high_risk_data = {
        "name": "高危敏感词库",
        "description": "包含暴恐、色情、涉枪涉爆、贪腐、政治等高危敏感词",
        "version": "1.0.0",
        "word_count": len(high_risk_words),
        "words": high_risk_words
    }

    with open(HIGH_RISK_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(high_risk_data, f, ensure_ascii=False, indent=2)
    print(f"\n高危词库已保存: {HIGH_RISK_OUTPUT}")
    print(f"高危词库词汇数量: {len(high_risk_words)}")

    # 保存中危词库
    medium_risk_data = {
        "name": "中危敏感词库",
        "description": "包含广告、民生、COVID-19等其他敏感词",
        "version": "1.0.0",
        "word_count": len(medium_risk_words),
        "words": medium_risk_words
    }

    with open(MEDIUM_RISK_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(medium_risk_data, f, ensure_ascii=False, indent=2)
    print(f"\n中危词库已保存: {MEDIUM_RISK_OUTPUT}")
    print(f"中危词库词汇数量: {len(medium_risk_words)}")


if __name__ == "__main__":
    main()