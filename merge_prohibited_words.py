#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将高危敏感词库合并到 prohibited.json
高危词库包括：暴恐词库、涉枪涉爆、反动词库
"""
import os
import json

# 高危词库文件
HIGH_RISK_VOCABULARIES = [
    "暴恐词库.txt",
    "涉枪涉爆.txt",
    "反动词库.txt"
]

# 词库目录
VOCABULARY_DIR = r"d:\JavaProjects\campus_forum_model\data\Vocabulary"
# 输出文件
OUTPUT_FILE = r"d:\JavaProjects\campus_forum\prohibited.json"

def read_vocabulary_file(filepath):
    """读取词库文件"""
    words = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释行
                if line and not line.startswith('#'):
                    words.append(line)
        print(f"读取文件 {os.path.basename(filepath)}: {len(words)} 个词")
    except Exception as e:
        print(f"读取文件失败 {filepath}: {e}")
    return words

def merge_high_risk_words():
    """合并高危敏感词"""
    all_words = []

    print("=" * 60)
    print("开始合并高危敏感词库...")
    print("=" * 60)

    for vocab_file in HIGH_RISK_VOCABULARIES:
        filepath = os.path.join(VOCABULARY_DIR, vocab_file)
        if os.path.exists(filepath):
            words = read_vocabulary_file(filepath)
            all_words.extend(words)
        else:
            print(f"警告: 文件不存在 {filepath}")

    # 去重
    unique_words = list(set(all_words))
    print(f"\n合并后总词数: {len(all_words)}")
    print(f"去重后词数: {len(unique_words)}")
    print(f"去重数量: {len(all_words) - len(unique_words)}")

    # 写入 JSON 文件
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(unique_words, f, ensure_ascii=False, indent=2)
        print(f"\n成功写入文件: {OUTPUT_FILE}")
        print(f"共写入 {len(unique_words)} 个敏感词")
    except Exception as e:
        print(f"写入文件失败 {OUTPUT_FILE}: {e}")

    print("=" * 60)
    print("完成！")
    print("=" * 60)

    # 显示前10个词作为预览
    if unique_words:
        print("\n前10个敏感词预览:")
        for i, word in enumerate(unique_words[:10], 1):
            print(f"  {i}. {word}")
        if len(unique_words) > 10:
            print(f"  ... 还有 {len(unique_words) - 10} 个词")

if __name__ == "__main__":
    merge_high_risk_words()
