# -*- coding: utf-8 -*-
"""
敏感词过滤模块 - 基于DFA(确定有限状态自动机)算法
支持多类别词库管理和热更新
"""

import os
import re
import json
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

    # 严重敏感词类别 - 直接拦截
    SEVERE_CATEGORIES = [WordCategory.VIOLENCE, WordCategory.ILLEGAL, WordCategory.REBELLIOUS]

    def __init__(self, data_dir: str = None):
        """
        初始化敏感词管理器
        :param data_dir: 数据目录路径
        """
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "data"
        )
        self.filter = DFAFilter()
        self.word_count = 0
        self.word_frequency = {}
        self.high_risk_words = set()
        self.medium_risk_words = set()
        self._load_all_vocabularies()

    def _load_all_vocabularies(self):
        """加载所有词库文件"""
        logger.info(f"开始加载敏感词库，目录: {self.data_dir}")

        high_risk_file = os.path.join(self.data_dir, "high_risk_words.json")
        medium_risk_file = os.path.join(self.data_dir, "medium_risk_words.json")

        # 加载高危词库
        if os.path.exists(high_risk_file):
            self._load_json_vocabulary(high_risk_file, risk_level="high")
        else:
            logger.warning(f"高危词库文件不存在: {high_risk_file}")

        # 加载中危词库
        if os.path.exists(medium_risk_file):
            self._load_json_vocabulary(medium_risk_file, risk_level="medium")
        else:
            logger.warning(f"中危词库文件不存在: {medium_risk_file}")

        logger.info(f"敏感词库加载完成，共 {self.word_count} 个词")
        logger.info(f"高危词库: {len(self.high_risk_words)} 个词")
        logger.info(f"中危词库: {len(self.medium_risk_words)} 个词")

    def _load_json_vocabulary(self, filepath: str, risk_level: str = "medium"):
        """加载JSON格式的词库文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                words = data.get('words', [])
                count = 0
                for word in words:
                    word = word.strip()
                    if word:
                        word_lower = word.lower()
                        severity = 3 if risk_level == "high" else 2
                        category = self._get_category_for_word(word_lower)
                        self.filter.add_word(word, category, severity)
                        self.word_frequency[word_lower] = 0
                        count += 1
                        if risk_level == "high":
                            self.high_risk_words.add(word_lower)
                        else:
                            self.medium_risk_words.add(word_lower)
                self.word_count += count
                logger.info(f"加载词库 {os.path.basename(filepath)} ({risk_level}): {count} 个词")
        except Exception as e:
            logger.error(f"加载词库文件失败 {filepath}: {str(e)}")

    def _get_category_for_word(self, word: str) -> WordCategory:
        """根据词汇特征判断类别"""
        if any(w in word for w in ['赌博', '博彩', '彩票', '下注', '投注', '赌钱', '赌场', '赌球', '赌注', '庄家', '开赌', '外围', '黑彩', '私彩', 'bocai']):
            return WordCategory.ILLEGAL
        elif any(w in word for w in ['兼职', '招聘', '代购', '代理', '加盟', '招商', '推广', '营销', '广告', '推销', '销售']):
            return WordCategory.AD
        elif any(w in word for w in ['色情', '黄色', '淫秽', '性爱', '成人', 'av', 'a片', '三级片', '性交', '做爱', '自慰', '手淫', '口交', '肛交', '阴道', '阴茎', '精液', '乳头', '乳房', '臀部', '射精', '性欲', '性高潮', '性行为', '性交易', '性服务', '性伴侣', '性虐待', 'sm', '捆绑', '滴蜡', '皮鞭', '情趣', '制服', '丝袜', '内衣', '透视', '走光', '露点', '喷血', '爆乳', '巨乳', '美乳', '骚', '浪', '媚', '妖', '勾引', '诱惑', '勾搭', '偷拍', '自拍', '偷窥', '黄色电影', '黄色小说', '黄色图片', '成人电影', '成人图片', '成人小说', '成人游戏', '成人论坛', '同城交友', '一夜情', '约炮', '援交', '买春', '卖春', '嫖娼', '妓女', '出台', '包夜', '全套']):
            return WordCategory.PORN
        elif any(w in word for w in ['暴恐', '恐怖', '炸弹', '爆炸', '枪支', '枪械', '子弹', '刀具', '砍刀', '匕首', '杀人', '自杀', '自残', '袭击', '恐怖分子', 'ISIS', '基地组织', '博科圣地', '伊斯兰国', '圣战', '殉教', '自杀式袭击', '汽车炸弹', '人体炸弹', '路边炸弹', '地雷', '手榴弹', '炸药', 'TNT', '黑火药', '硝化甘油', '雷管', '引爆', '纵火', '投毒', '暗杀', '刺杀', '绑架', '劫持', '劫机', '劫船', '劫车', '人质', '勒索', '抢劫', '盗窃', '诈骗', '贪污', '贿赂', '走私', '贩毒', '吸毒', '洗钱', '偷渡', '非法移民', '黑市', '军火', '武器']):
            return WordCategory.VIOLENCE
        elif any(w in word for w in ['领导', '总书记', '主席', '总理', '国家', '政府', '政党', '党建', '党章', '党性', '党政', '党内', '党外', '党委', '纪委', '组织部', '宣传部', '统战部', '政法委', '综治办', '610办公室', '维权', '上访', '静坐', '抗议', '示威', '游行', '罢工', '罢课', '罢市', '抗税', '抗租', '抗粮', '抗债', '抗费', '抗捐', '法轮功', '邪教', '颠覆', '推翻', '反动', '分裂', '独立', '叛国', '卖国', '汉奸', '卖国贼', '叛徒', '内奸', '间谍', '特务', '走狗', '奴才', '洋奴', '卖国求荣', '崇洋媚外', '数典忘祖', '卖国主义', '民族败类', '国家公敌', '人民公敌', '历史罪人', '千古罪人', '民族罪人', '国家罪人', '人民罪人']):
            return WordCategory.POLITICAL
        elif any(w in word for w in ['贪腐', '腐败', '贪污', '受贿', '行贿', '索贿', '挪用', '侵占', '私分', '挥霍', '浪费', '渎职', '失职', '滥用职权', '玩忽职守', '徇私舞弊', '权钱交易', '权色交易', '权权交易', '官商勾结', '权大于法', '以权谋私', '以权压法', '以权乱法', '执法犯法', '司法腐败', '行政腐败', '吏治腐败', '经济腐败', '政治腐败', '社会腐败']):
            return WordCategory.CORRUPTION
        elif any(w in word for w in ['新冠', '肺炎', '疫情', '病毒', '感染', '传播', '防控', '隔离', '疫苗', '核酸', '检测', '确诊', '疑似', '死亡', '治愈', '病例', '无症状', '密切接触', '次密切接触', '封城', '封区', '封村', '封路', '停课', '停工', '停产', '停业', '限行', '管控', '解封', '复工', '复产', '复学', '常态化', '疫情防控', '新冠肺炎', 'COVID-19', 'coronavirus', 'pandemic', 'epidemic']):
            return WordCategory.COVID
        elif any(w in word for w in ['反动', '反党', '反政府', '反社会', '反人类', '反民族', '反国家', '反人民', '反社会主义', '反共产主义', '反马克思主义', '反列宁主义', '反毛泽东思想', '反邓小平理论', '反三个代表', '反科学发展观', '反中国梦', '反核心价值观', '反改革开放', '反四项基本原则']):
            return WordCategory.REBELLIOUS
        elif any(w in word for w in ['民生', '民意', '民情', '民心', '民怨', '民愤', '民忧', '民困', '民苦', '民怨沸腾', '民不聊生', '民生疾苦', '民生问题', '民生改善', '民生工程', '民生政策', '民生保障', '民生服务', '民生投入', '民生支出', '民生福利', '民生水平', '民生质量', '民生状况', '民生诉求', '民生需求', '民生期盼', '民生愿景', '民生目标', '民生规划', '民生发展', '民生改革', '民生创新', '民生进步', '民生幸福', '民生安康', '民生和谐', '民生稳定', '民生繁荣', '民生昌盛']):
            return WordCategory.PEOPLE
        else:
            return WordCategory.OTHER

    def check(self, text: str) -> MatchResult:
        """检查文本是否包含敏感词"""
        result = self.filter.parse(text)
        # 更新词频
        for word in result.matched_words:
            if word in self.word_frequency:
                self.word_frequency[word] += 1
        return result

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
        # 统计每个类别的词数
        category_counts = {}
        for word, category in self.filter.word_categories.items():
            category_name = category.value
            if category_name not in category_counts:
                category_counts[category_name] = 0
            category_counts[category_name] += 1
        
        return {
            "total_words": self.word_count,
            "high_risk_words": len(self.high_risk_words),
            "medium_risk_words": len(self.medium_risk_words),
            "categories": category_counts
        }

    def get_word_cloud_data(self, limit: int = 100) -> List[Dict]:
        """获取敏感词云数据
        
        Args:
            limit: 返回词的数量限制
            
        Returns:
            敏感词云数据列表，每个元素包含word和count
        """
        # 按词频排序，获取高频敏感词
        sorted_words = sorted(
            self.word_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        # 转换为词云数据格式
        word_cloud_data = []
        for word, count in sorted_words:
            word_cloud_data.append({
                "word": word,
                "count": count
            })
        
        # 如果没有词频大于0的词，返回前limit个词
        if not word_cloud_data and self.word_frequency:
            top_words = list(self.word_frequency.items())[:limit]
            for word, count in top_words:
                word_cloud_data.append({
                    "word": word,
                    "count": count
                })
        
        return word_cloud_data

    def add_keyword(self, word: str, category: WordCategory = WordCategory.OTHER, severity: int = 1):
        """添加敏感词"""
        self.filter.add_word(word, category, severity)
        self.word_count += 1
        word_lower = word.lower()
        if word_lower not in self.word_frequency:
            self.word_frequency[word_lower] = 0
        return True

    def _save_to_json(self):
        """保存敏感词到JSON文件"""
        try:
            # 分别收集高危和中危敏感词
            high_risk_words = []
            medium_risk_words = []
            
            for word, severity in self.filter.word_severity.items():
                if severity == 3:
                    high_risk_words.append(word)
                else:
                    medium_risk_words.append(word)
            
            # 保存高危词库
            high_risk_file = os.path.join(self.data_dir, "high_risk_words.json")
            with open(high_risk_file, 'w', encoding='utf-8') as f:
                json.dump({'words': high_risk_words}, f, ensure_ascii=False, indent=2)
            
            # 保存中危词库
            medium_risk_file = os.path.join(self.data_dir, "medium_risk_words.json")
            with open(medium_risk_file, 'w', encoding='utf-8') as f:
                json.dump({'words': medium_risk_words}, f, ensure_ascii=False, indent=2)
            
            logger.info(f"敏感词库保存成功: 高危词 {len(high_risk_words)} 个, 中危词 {len(medium_risk_words)} 个")
        except Exception as e:
            logger.error(f"保存敏感词库失败: {str(e)}")
    
    def remove_keyword(self, word: str):
        """移除敏感词"""
        word_lower = word.lower()
        if word_lower in self.filter.word_categories:
            del self.filter.word_categories[word_lower]
            if word_lower in self.filter.word_severity:
                del self.filter.word_severity[word_lower]
            if word_lower in self.word_frequency:
                del self.word_frequency[word_lower]
            # 从集合中移除
            if word_lower in self.high_risk_words:
                self.high_risk_words.remove(word_lower)
            if word_lower in self.medium_risk_words:
                self.medium_risk_words.remove(word_lower)
            self.word_count -= 1
            # 保存更改到JSON文件
            self._save_to_json()
            return True
        return False


# 全局敏感词管理器实例
_swm_instance = None


def get_sensitive_word_manager(vocab_dir: str = None) -> SensitiveWordManager:
    """获取敏感词管理器单例"""
    global _swm_instance
    if _swm_instance is None:
        _swm_instance = SensitiveWordManager(vocab_dir)
    return _swm_instance


class ForbiddenContentDetector:
    """
    违禁内容检测器
    """
    
    # 违禁内容正则表达式
    FORBIDDEN_PATTERNS = {
        # 联系方式
        'phone': r'1[3-9]\d{9}',  # 手机号
        'qq': r'QQ[\s]*[:：]\s*[1-9]\d{4,9}',  # QQ号
        'wechat': r'(微信|vx)[\s]*[:：]\s*[a-zA-Z0-9_-]{6,20}',  # 微信号
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 邮箱
        'url': r'https?://[\w\-._~:/?#[\]@!$&\'()*+,;=.]+',  # URL
        
        # 违法内容
        'gambling': r'(赌博|博彩|彩票|时时彩|六合彩|下注|投注)',  # 赌博
        'fraud': r'(诈骗|骗钱|刷单|刷钻|兼职|日赚|月入)',  # 诈骗
        'porn': r'(色情|黄色|淫秽|性爱|成人|AV|情色)',  # 色情
        'drugs': r'(毒品|大麻|冰毒|摇头丸|海洛因)',  # 毒品
        'violence': r'(暴力|杀人|自杀|自残|打架|斗殴)',  # 暴力
        
        # 政治违禁
        'political': r'(法轮功|邪教|颠覆|推翻|反动|抗议|示威)',  # 政治违禁
        
        # 广告
        'spam': r'(广告|推广|营销|代购|代理|加盟|招商)',  # 广告
    }
    
    @classmethod
    def check(cls, text: str) -> Tuple[bool, List[str], List[str]]:
        """
        检查文本是否包含违禁内容
        
        Args:
            text: 待检查文本
            
        Returns:
            Tuple[是否包含违禁内容, 命中的模式列表, 命中的类型列表]
        """
        matched_patterns = []
        matched_types = []
        
        for pattern_type, pattern in cls.FORBIDDEN_PATTERNS.items():
            if re.search(pattern, text):
                matched_patterns.append(pattern)
                matched_types.append(pattern_type)
        
        return len(matched_patterns) > 0, matched_patterns, matched_types
    
    @classmethod
    def get_forbidden_types(cls) -> List[str]:
        """
        获取违禁内容类型
        
        Returns:
            违禁内容类型列表
        """
        return list(cls.FORBIDDEN_PATTERNS.keys())


if __name__ == "__main__":
    # 测试敏感词库管理器
    manager = get_sensitive_word_manager()
    
    # 测试检查
    test_texts = [
        "这课太难了",  # 合理负面情绪
        "刷单日赚500+",  # 违法诱导
        "习近平总书记",  # 政治内容
        "赌博网站，注册送100元",  # 赌博
        "微信：wx12345678",  # 联系方式
    ]
    
    print("测试敏感词检测:")
    for text in test_texts:
        result = manager.check(text)
        has_forbidden, matched_patterns, matched_types = ForbiddenContentDetector.check(text)
        
        print(f"\n文本: {text}")
        print(f"包含敏感词: {result.is_match}")
        print(f"命中敏感词: {result.matched_words}")
        print(f"命中分类: {[cat.value for cat in result.categories]}")
        print(f"风险评分: {result.risk_score}")
        print(f"包含违禁内容: {has_forbidden}")
        print(f"命中违禁类型: {matched_types}")
    
    # 打印统计信息
    print("\n敏感词库统计:")
    stats = manager.get_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    # 打印词云数据
    print("\n敏感词云数据:")
    word_cloud_data = manager.get_word_cloud_data(20)
    print(json.dumps(word_cloud_data, ensure_ascii=False, indent=2))