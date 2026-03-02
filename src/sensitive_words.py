"""
敏感词过滤模块 - 基于DFA(确定有限状态自动机)算法
支持多类别词库管理和热更新
"""

import os
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WordCategory(Enum):
    """敏感词类别"""
    PORN = "色情"
    POLITICAL = "政治"
    VIOLENCE = "暴恐"
    AD = "广告"
    ILLEGAL = "违法"
    CORRUPTION = "贪腐"
    COVID = "疫情"
    REBELLIOUS = "反动"
    PEOPLE = "民生"
    OTHER = "其他"


@dataclass
class MatchResult:
    """匹配结果"""
    is_match: bool
    matched_words: List[str]
    categories: List[WordCategory]
    risk_score: float
    positions: List[Tuple[int, int]]


class DFAFilter:
    """DFA敏感词过滤器"""

    def __init__(self):
        self.keyword_chains = {}
        self.delimit = '\x00'
        self.word_categories = {}
        self.word_severity = {}

    def add_word(self, word: str, category: WordCategory = WordCategory.OTHER, severity: int = 1):
        """添加敏感词到DFA树"""
        if not word:
            return

        word = word.strip().lower()
        if not word:
            return

        # 只有当词不存在或者新的类别更严重时才更新类别和严重程度
        if word not in self.word_categories or category in SensitiveWordManager.SEVERE_CATEGORIES:
            self.word_categories[word] = category
            self.word_severity[word] = severity

        level = self.keyword_chains
        for char in word:
            if char not in level:
                level[char] = {}
            level = level[char]
        level[self.delimit] = {}

    def parse(self, text: str) -> MatchResult:
        """解析文本，返回匹配结果"""
        if not text:
            return MatchResult(False, [], [], 0.0, [])

        text = text.lower()
        matched_words = []
        categories = set()
        positions = []
        total_risk = 0

        i = 0
        while i < len(text):
            level = self.keyword_chains
            match_len = 0
            match_word = ""
            start_pos = i

            for j in range(i, len(text)):
                char = text[j]
                if char in level:
                    match_len += 1
                    match_word += char
                    level = level[char]
                    if self.delimit in level:
                        matched_words.append(match_word)
                        positions.append((start_pos, j + 1))
                        if match_word in self.word_categories:
                            categories.add(self.word_categories[match_word])
                            total_risk += self.word_severity.get(match_word, 1) * 20
                else:
                    break

            if match_len > 0:
                i += match_len
            else:
                i += 1

        risk_score = min(total_risk, 100)

        return MatchResult(
            is_match=len(matched_words) > 0,
            matched_words=list(set(matched_words)),
            categories=list(categories),
            risk_score=risk_score,
            positions=positions
        )

    def replace(self, text: str, replace_char: str = '*') -> str:
        """替换敏感词"""
        if not text:
            return text

        result = list(text.lower())
        match_result = self.parse(text)

        for start, end in match_result.positions:
            for i in range(start, end):
                result[i] = replace_char

        return ''.join(result)

    def check(self, text: str) -> bool:
        """检查是否包含敏感词"""
        return self.parse(text).is_match


class SensitiveWordManager:
    """敏感词管理器 - 支持多词库管理"""

    VOCABULARY_FILES = {
        WordCategory.PORN: ["色情词库.txt", "色情类型.txt"],
        WordCategory.POLITICAL: ["政治类型.txt", "新思想启蒙.txt"],
        WordCategory.VIOLENCE: ["暴恐词库.txt"],
        WordCategory.AD: ["广告类型.txt"],
        WordCategory.ILLEGAL: ["涉枪涉爆.txt", "非法网址.txt"],
        WordCategory.CORRUPTION: ["贪腐词库.txt"],
        WordCategory.COVID: ["COVID-19词库.txt"],
        WordCategory.REBELLIOUS: ["反动词库.txt"],
        WordCategory.PEOPLE: ["民生词库.txt"],
        WordCategory.OTHER: ["其他词库.txt", "补充词库.txt", "GFW补充词库.txt"]
    }

    # 严重敏感词类别 - 直接拦截
    SEVERE_CATEGORIES = [WordCategory.VIOLENCE, WordCategory.ILLEGAL, WordCategory.REBELLIOUS]

    def __init__(self, vocab_dir: str = None):
        """
        初始化敏感词管理器
        :param vocab_dir: 词库目录路径
        """
        self.vocab_dir = vocab_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Vocabulary"
        )
        self.filter = DFAFilter()
        self.word_count = 0
        self._load_all_vocabularies()

    def _load_all_vocabularies(self):
        """加载所有词库文件"""
        logger.info(f"开始加载敏感词库，目录: {self.vocab_dir}")

        # 首先加载严重类别的词库
        severe_categories = [WordCategory.VIOLENCE, WordCategory.ILLEGAL, WordCategory.REBELLIOUS]
        for category in severe_categories:
            if category in self.VOCABULARY_FILES:
                for filename in self.VOCABULARY_FILES[category]:
                    filepath = os.path.join(self.vocab_dir, filename)
                    if os.path.exists(filepath):
                        self._load_vocabulary_file(filepath, category)
                    else:
                        logger.warning(f"词库文件不存在: {filepath}")

        # 然后加载其他类别的词库
        for category, files in self.VOCABULARY_FILES.items():
            if category not in severe_categories:
                for filename in files:
                    filepath = os.path.join(self.vocab_dir, filename)
                    if os.path.exists(filepath):
                        self._load_vocabulary_file(filepath, category)
                    else:
                        logger.warning(f"词库文件不存在: {filepath}")

        logger.info(f"敏感词库加载完成，共 {self.word_count} 个词")

    def _load_vocabulary_file(self, filepath: str, category: WordCategory):
        """加载单个词库文件"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                count = 0
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        # 严重类别标记为严重程度3
                        severity = 3 if category in self.SEVERE_CATEGORIES else 2
                        # 只有当词不存在或者当前类别比已存在的类别更严重时才添加/更新
                        word_lower = word.lower()
                        if word_lower not in self.filter.word_categories or category in self.SEVERE_CATEGORIES:
                            self.filter.add_word(word, category, severity)
                            count += 1
                self.word_count += count
                logger.info(f"加载词库 {os.path.basename(filepath)} ({category.value}): {count} 个词")
        except Exception as e:
            logger.error(f"加载词库文件失败 {filepath}: {str(e)}")

    def check(self, text: str) -> MatchResult:
        """检查文本是否包含敏感词"""
        return self.filter.parse(text)

    def replace(self, text: str, replace_char: str = '*') -> str:
        """替换敏感词"""
        return self.filter.replace(text, replace_char)

    def is_severe(self, result: MatchResult) -> bool:
        """判断是否包含严重敏感词"""
        for category in result.categories:
            if category in self.SEVERE_CATEGORIES:
                return True
        return False

    def get_stats(self) -> Dict:
        """获取词库统计信息"""
        return {
            "total_words": self.word_count,
            "categories": {cat.value: len(files) for cat, files in self.VOCABULARY_FILES.items()}
        }


# 全局敏感词管理器实例
_swm_instance = None


def get_sensitive_word_manager(vocab_dir: str = None) -> SensitiveWordManager:
    """获取敏感词管理器单例"""
    global _swm_instance
    if _swm_instance is None:
        _swm_instance = SensitiveWordManager(vocab_dir)
    return _swm_instance
