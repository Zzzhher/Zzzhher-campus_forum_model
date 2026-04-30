import re
from typing import Dict, List, Tuple
from enum import Enum
from datetime import datetime
import json
import os
import numpy as np
import torch
from sensitive_words import get_sensitive_word_manager


class ModerationAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REVIEW = "review"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModerationResult:
    def __init__(self):
        self.action = None
        self.risk_level = None
        self.sentiment = None
        self.reason = ""
        self.matched_high_risk = []
        self.matched_medium_risk = []
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        # 合并高危词和中危词
        all_sensitive_words = self.matched_high_risk + self.matched_medium_risk
        
        # 构建符合 Java 端期望的 sensitive_words 对象
        sensitive_words = {
            'matched': len(all_sensitive_words) > 0,
            'words': all_sensitive_words,
            'risk_score': len(self.matched_high_risk) * 3 + len(self.matched_medium_risk) * 2
        }
        
        return {
            'action': self.action.value if self.action else None,
            'risk_level': self.risk_level.value if self.risk_level else None,
            'sentiment': self.sentiment,
            'reason': self.reason,
            'matched_high_risk': self.matched_high_risk,
            'matched_medium_risk': self.matched_medium_risk,
            'sensitive_words': sensitive_words,
            'timestamp': self.timestamp
        }


class ModerationEngine:
    def __init__(self, sentiment_inference=None):
        """
        初始化审核引擎
        
        Args:
            sentiment_inference: 情感分析引擎实例
        """
        self.sentiment_inference = sentiment_inference
        self.moderation_log = []
        
        # 词库路径
        self.models_data_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'data')
        
        # 初始化敏感词管理器
        self.sensitive_word_manager = get_sensitive_word_manager()
        
        # 加载高危和中危词库
        self.high_risk_words = self._load_high_risk_words()
        self.medium_risk_words = self._load_medium_risk_words()
        
        # 规则优先级和权重
        self.rule_priorities = {
            'high_risk': 3,  # 最高优先级
            'medium_risk': 2,
            'contact_info': 3,  # 联系方式检测
            'advertisement': 3,  # 广告检测
            'political': 3,  # 政治内容检测
            'illegal': 3,  # 违法内容检测
            'spam': 1  # 垃圾内容检测
        }
        
        # 规则权重动态调整
        self.rule_weights = {
            'high_risk': 1.0,
            'medium_risk': 0.7,
            'contact_info': 0.9,
            'advertisement': 0.8,
            'political': 0.95,
            'illegal': 1.0,
            'spam': 0.5
        }
        
        # 审核策略配置
        self.config = {
            # 情感分析阈值
            "negative_threshold": 0.85,        # 负向情感置信度阈值
            "negative_block_threshold": 0.98,  # 负向情感直接拦截阈值

            # 敏感词阈值
            "sensitive_risk_threshold": 40,    # 敏感词风险分数阈值
            "sensitive_block_threshold": 60    # 敏感词直接拦截阈值
        }
        
        # 词向量模型（用于语义敏感词检测）
        self.word_embeddings = None
        self._load_word_embeddings()
        
        print(f"加载高危词: {len(self.high_risk_words)} 个")
        print(f"加载中危词: {len(self.medium_risk_words)} 个")
    
    def _load_high_risk_words(self):
        """
        加载高危词库
        """
        high_risk_words = set()
        
        # 从文件加载高危词
        high_risk_file = os.path.join(self.models_data_path, 'high_risk_words.json')
        if os.path.exists(high_risk_file):
            try:
                with open(high_risk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    high_risk_words.update(data.get('words', []))
            except Exception as e:
                print(f"加载高危词库文件失败: {e}")
        else:
            print(f"高危词库文件不存在: {high_risk_file}")
        
        # 添加一些常见的侮辱性词汇作为备用
        high_risk_words.update([
            '傻逼', '白痴', '智障', '蠢货', '畜生', '禽兽',
        ])
        
        return high_risk_words
    
    def _load_medium_risk_words(self):
        """
        加载中危词库
        """
        medium_risk_words = set()
        
        # 从文件加载中危词
        medium_risk_file = os.path.join(self.models_data_path, 'medium_risk_words.json')
        if os.path.exists(medium_risk_file):
            try:
                with open(medium_risk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    medium_risk_words.update(data.get('words', []))
            except Exception as e:
                print(f"加载中危词库文件失败: {e}")
        else:
            print(f"中危词库文件不存在: {medium_risk_file}")
        
        return medium_risk_words
    
    def _load_word_embeddings(self):
        """
        加载词向量模型，用于语义敏感词检测
        """
        try:
            # 这里使用随机初始化的词向量作为示例
            # 实际应用中可以加载预训练的词向量模型
            vocab_size = 30000
            embedding_dim = 100
            self.word_embeddings = np.random.randn(vocab_size, embedding_dim)
            print(f"词向量模型加载完成，形状: {self.word_embeddings.shape}")
        except Exception as e:
            print(f"加载词向量模型失败: {e}")
            self.word_embeddings = None
    
    def _calculate_similarity(self, word1, word2):
        """
        计算两个词的语义相似度
        """
        if self.word_embeddings is None:
            return 0.0
        
        # 这里使用简单的余弦相似度计算
        # 实际应用中需要根据词向量模型的实现进行调整
        return 0.5  # 示例值
    
    def _detect_semantic_sensitive(self, text: str) -> List[str]:
        """
        语义敏感词检测
        
        Args:
            text: 待检测文本
            
        Returns:
            命中的语义敏感词列表
        """
        matched_words = []
        
        # 简单的语义敏感词检测实现
        # 实际应用中需要结合词向量模型进行更复杂的语义匹配
        sensitive_patterns = [
            r'违法', r'犯罪', r'暴力', r'色情', r'赌博', r'毒品',
            r'政治', r'反动', r'恐怖', r'极端', r'邪教'
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, text):
                matched_words.append(pattern)
        
        return matched_words
    
    def _detect_contact_info(self, text: str) -> bool:
        """
        检测联系方式
        """
        # 检测手机号
        phone_pattern = r'1[3-9]\d{9}'
        if re.search(phone_pattern, text):
            return True
        
        # 检测邮箱
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        if re.search(email_pattern, text):
            return True
        
        # 检测QQ号
        qq_pattern = r'[qQ][qQ][\s:：]*[1-9]\d{4,}'
        if re.search(qq_pattern, text):
            return True
        
        return False
    
    def _detect_advertisement(self, text: str) -> bool:
        """
        检测广告内容
        """
        ad_patterns = [
            r'推广', r'广告', r'宣传', r'促销', r'优惠',
            r'限时', r'抢购', r'折扣', r'免费', r'赠送'
        ]
        
        for pattern in ad_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_political(self, text: str) -> bool:
        """
        检测政治内容
        """
        political_patterns = [
            r'政府', r'政策', r'政治', r'领导人', r'国家'
        ]
        
        for pattern in political_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_illegal(self, text: str) -> bool:
        """
        检测违法内容
        """
        illegal_patterns = [
            r'毒品', r'赌博', r'色情', r'暴力', r'恐怖'
        ]
        
        for pattern in illegal_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_spam(self, text: str) -> bool:
        """
        检测垃圾内容
        """
        # 检测重复内容
        if len(text) > 20:
            # 简单的重复模式检测
            for i in range(len(text) - 3):
                substring = text[i:i+4]
                if text.count(substring) > 2:
                    return True
        
        return False
    
    def _adjust_rule_weights(self, rule_name, is_correct):
        """
        动态调整规则权重
        
        Args:
            rule_name: 规则名称
            is_correct: 规则判断是否正确
        """
        if rule_name in self.rule_weights:
            if is_correct:
                # 增加权重
                self.rule_weights[rule_name] = min(1.0, self.rule_weights[rule_name] + 0.05)
            else:
                # 减少权重
                self.rule_weights[rule_name] = max(0.1, self.rule_weights[rule_name] - 0.1)
            
            print(f"规则 {rule_name} 权重调整为: {self.rule_weights[rule_name]:.2f}")
    def _detect_high_risk(self, text: str) -> List[str]:
        """
        检测高危词
        
        Args:
            text: 待检测文本
            
        Returns:
            命中的高危词列表
        """
        matched_words = []
        for word in self.high_risk_words:
            if word in text:
                matched_words.append(word)
        return matched_words
    
    def _detect_medium_risk(self, text: str) -> List[str]:
        """
        检测中危词
        
        Args:
            text: 待检测文本
            
        Returns:
            命中的中危词列表
        """
        matched_words = []
        for word in self.medium_risk_words:
            if word in text:
                matched_words.append(word)
        return matched_words
    
    def layer1_sentiment_analysis(self, text: str) -> Tuple[Dict, float]:
        """
        第一层：情感分析
        
        Args:
            text: 待分析文本
            
        Returns:
            Tuple[情感分析结果对象, 置信度]
        """
        if self.sentiment_inference is None:
            return {"label": "中性", "confidence": 0.5}, 0.5
        
        try:
            result = self.sentiment_inference.predict(text, return_probabilities=True)
            sentiment_obj = {
                "label": result['predicted_label'],
                "confidence": result.get('confidence', 0.5)
            }
            return sentiment_obj, result.get('confidence', 0.5)
        except Exception as e:
            print(f"情感分析失败: {e}")
            return {"label": "中性", "confidence": 0.5}, 0.5
    
    def layer2_risk_detection(self, text: str) -> Tuple[List[str], List[str]]:
        """
        第二层：风险词检测
        
        Args:
            text: 待检测文本
            
        Returns:
            Tuple[命中的高危词列表, 命中的中危词列表]
        """
        # 使用敏感词管理器检测敏感词
        match_result = self.sensitive_word_manager.check(text)
        
        # 分离高危词和中危词
        matched_high_risk = []
        matched_medium_risk = []
        
        for word in match_result.matched_words:
            word_lower = word.lower()
            if word_lower in self.high_risk_words:
                matched_high_risk.append(word)
            elif word_lower in self.medium_risk_words:
                matched_medium_risk.append(word)
        
        return matched_high_risk, matched_medium_risk
    
    def layer3_final_decision(self, text: str, sentiment: Dict, matched_high_risk: List[str], matched_medium_risk: List[str], 
                             matched_semantic_sensitive: List[str], rule_results: Dict, risk_score: float) -> ModerationResult:
        """
        第三层：最终决策
        
        Args:
            text: 待审核文本
            sentiment: 情感分析结果对象
            matched_high_risk: 命中的高危词列表
            matched_medium_risk: 命中的中危词列表
            matched_semantic_sensitive: 命中的语义敏感词列表
            rule_results: 规则检测结果
            risk_score: 风险分数
            
        Returns:
            审核结果
        """
        result = ModerationResult()
        result.sentiment = sentiment
        result.matched_high_risk = matched_high_risk
        result.matched_medium_risk = matched_medium_risk
        
        # 构建详细的原因
        reasons = []
        
        # 检查高优先级规则
        if rule_results.get('high_risk'):
            reasons.append("含高危词")
        if rule_results.get('illegal'):
            reasons.append("含违法内容")
        if rule_results.get('contact_info'):
            reasons.append("含联系方式")
        if rule_results.get('advertisement'):
            reasons.append("含广告内容")
        if rule_results.get('political'):
            reasons.append("含政治内容")
        if rule_results.get('medium_risk'):
            reasons.append("含中危词")
        if rule_results.get('spam'):
            reasons.append("含垃圾内容")
        if matched_semantic_sensitive:
            reasons.append("含语义敏感内容")
        
        # 获取情感标签和置信度
        sentiment_label = sentiment.get('label', '中性')
        sentiment_confidence = sentiment.get('confidence', 0.0)
        
        # 检查是否需要人工审核
        needs_manual_review = False
        
        # 高风险直接拦截
        if risk_score >= 5.0:
            # 高风险
            result.action = ModerationAction.BLOCK
            result.risk_level = RiskLevel.HIGH
            result.reason = "命中高风险内容，自动拦截: " + ", ".join(reasons) if reasons else "命中高风险内容，自动拦截"
        else:
            # 检查负向情感
            if sentiment_label == '负向':
                # 负向情感置信度极高，直接拦截
                if sentiment_confidence >= self.config["negative_block_threshold"]:
                    result.action = ModerationAction.BLOCK
                    result.risk_level = RiskLevel.HIGH
                    result.reason = f"负向情感置信度过高: {sentiment_confidence:.2%}"
                # 负向情感置信度较高，需要人工审核
                elif sentiment_confidence >= self.config["negative_threshold"]:
                    result.action = ModerationAction.REVIEW
                    result.risk_level = RiskLevel.MEDIUM
                    result.reason = f"负向情感，需人工审核: {sentiment_confidence:.2%}"
                # 负向情感但置信度较低，结合风险分数判断
                elif risk_score >= 2.0:
                    result.action = ModerationAction.REVIEW
                    result.risk_level = RiskLevel.MEDIUM
                    result.reason = "命中中风险内容，需人工审核: " + ", ".join(reasons) if reasons else "命中中风险内容，需人工审核"
                else:
                    # 负向情感但风险较低，允许发布（合理吐槽）
                    result.action = ModerationAction.ALLOW
                    result.risk_level = RiskLevel.LOW
                    result.reason = "负面情绪但无高风险内容，允许发布"
            else:
                # 非负向情感
                if risk_score >= 2.0:
                    result.action = ModerationAction.REVIEW
                    result.risk_level = RiskLevel.MEDIUM
                    result.reason = "命中中风险内容，需人工审核: " + ", ".join(reasons) if reasons else "命中中风险内容，需人工审核"
                else:
                    result.action = ModerationAction.ALLOW
                    result.risk_level = RiskLevel.LOW
                    result.reason = "非负面情绪，允许发布"
        
        return result
    
    def moderate(self, text: str) -> ModerationResult:
        """
        审核文本
        
        Args:
            text: 待审核文本
            
        Returns:
            审核结果
        """
        result = ModerationResult()
        
        try:
            # 第一层：情感分析
            sentiment, confidence = self.layer1_sentiment_analysis(text)
            
            # 第二层：风险词检测
            matched_high_risk, matched_medium_risk = self.layer2_risk_detection(text)
            
            # 新增：语义敏感词检测
            matched_semantic_sensitive = self._detect_semantic_sensitive(text)
            
            # 新增：多维度检测
            has_contact_info = self._detect_contact_info(text)
            has_advertisement = self._detect_advertisement(text)
            has_political = self._detect_political(text)
            has_illegal = self._detect_illegal(text)
            has_spam = self._detect_spam(text)
            
            # 构建规则检测结果
            rule_results = {
                'high_risk': len(matched_high_risk) > 0,
                'medium_risk': len(matched_medium_risk) > 0,
                'contact_info': has_contact_info,
                'advertisement': has_advertisement,
                'political': has_political,
                'illegal': has_illegal,
                'spam': has_spam
            }
            
            # 计算风险分数
            risk_score = 0.0
            for rule_name, is_detected in rule_results.items():
                if is_detected:
                    risk_score += self.rule_priorities[rule_name] * self.rule_weights[rule_name]
            
            # 第三层：最终决策
            result = self.layer3_final_decision(
                text, sentiment, matched_high_risk, matched_medium_risk,
                matched_semantic_sensitive, rule_results, risk_score
            )
            
            self._log_moderation(text, result)
            
        except Exception as e:
            print(f"审核过程出错: {e}")
            result.action = ModerationAction.REVIEW
            result.risk_level = RiskLevel.MEDIUM
            result.reason = f"审核过程出错: {str(e)}"
            # 设置默认的sentiment对象，确保返回格式一致
            result.sentiment = {"label": "中性", "confidence": 0.5}
        
        return result
    
    def moderate_batch(self, texts: List[str]) -> List[ModerationResult]:
        """
        批量审核文本
        
        Args:
            texts: 待审核文本列表
            
        Returns:
            审核结果列表
        """
        results = []
        for text in texts:
            result = self.moderate(text)
            results.append(result)
        return results
    
    def _log_moderation(self, text: str, result: ModerationResult):
        """
        记录审核日志
        """
        # 提取情感标签用于日志记录
        sentiment_label = result.sentiment.get('label', '未知') if isinstance(result.sentiment, dict) else str(result.sentiment)
        
        log_entry = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'action': result.action.value if result.action else None,
            'risk_level': result.risk_level.value if result.risk_level else None,
            'sentiment': sentiment_label,
            'reason': result.reason,
            'matched_high_risk': result.matched_high_risk,
            'matched_medium_risk': result.matched_medium_risk,
            'timestamp': result.timestamp
        }
        
        self.moderation_log.append(log_entry)
    
    def get_moderation_stats(self) -> Dict:
        """
        获取审核统计信息
        
        Returns:
            统计信息
        """
        if not self.moderation_log:
            return {}
        
        total = len(self.moderation_log)
        stats = {
            'total': total,
            'allowed': 0,
            'blocked': 0,
            'reviewed': 0,
            'by_risk_level': {
                'low': 0,
                'medium': 0,
                'high': 0
            },
            'by_sentiment': {
                '正向': 0,
                '中性': 0,
                '负向': 0
            }
        }
        
        for log in self.moderation_log:
            action = log.get('action')
            if action == ModerationAction.ALLOW.value:
                stats['allowed'] += 1
            elif action == ModerationAction.BLOCK.value:
                stats['blocked'] += 1
            elif action == ModerationAction.REVIEW.value:
                stats['reviewed'] += 1
            
            risk = log.get('risk_level')
            if risk:
                stats['by_risk_level'][risk] += 1
            
            sentiment = log.get('sentiment')
            if sentiment:
                # 确保sentiment是字符串
                sentiment_str = sentiment.get('label', '未知') if isinstance(sentiment, dict) else str(sentiment)
                if sentiment_str in stats['by_sentiment']:
                    stats['by_sentiment'][sentiment_str] += 1
        
        return stats
    
    def update_config(self, **kwargs):
        """
        更新审核策略配置
        
        Args:
            **kwargs: 配置项
        """
        self.config.update(kwargs)
        print(f"审核策略已更新: {kwargs}")
    
    def get_config(self) -> Dict:
        """
        获取当前审核策略配置
        
        Returns:
            配置字典
        """
        return self.config.copy()


if __name__ == '__main__':
    print("=" * 60)
    print("审核引擎测试")
    print("=" * 60)
    
    # 测试审核引擎
    engine = ModerationEngine()
    
    test_texts = [
        "这个酒店真的太棒了，服务很好，房间也很干净！",  # 正向情感
        "作业太多了，累死了，老师太变态了！",  # 负向情感，合理吐槽
        "我要杀了你，你给我等着！",  # 负向情感，含高危词
        "这个游戏有私服，一起去玩啊！",  # 中性情感，含中危词
        "政府政策真的很好，支持！",  # 正向情感
        "你是个傻逼，滚开！",  # 负向情感，含高危词
        "这个产品很棒，推荐购买！",  # 正向情感
        "我对这个决定有意见，建议重新考虑！",  # 中性情感，合理表达
    ]
    
    print("\n" + "=" * 60)
    print("单条文本审核")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. 文本: {text}")
        
        result = engine.moderate(text)
        
        action_emoji = {
            ModerationAction.ALLOW: "✅",
            ModerationAction.BLOCK: "❌",
            ModerationAction.REVIEW: "⚠️"
        }
        
        print(f"   情感: {result.sentiment}")
        print(f"   风险等级: {result.risk_level.value if result.risk_level else 'N/A'}")
        print(f"   审核结果: {action_emoji.get(result.action, '?')} {result.action.value if result.action else 'N/A'}")
        print(f"   原因: {result.reason}")
        if result.matched_high_risk:
            print(f"   命中高危词: {result.matched_high_risk}")
        if result.matched_medium_risk:
            print(f"   命中中危词: {result.matched_medium_risk}")
    
    print("\n" + "=" * 60)
    print("审核统计")
    print("=" * 60)
    
    stats = engine.get_moderation_stats()
    print(f"总审核数: {stats.get('total', 0)}")
    print(f"放行: {stats.get('allowed', 0)}")
    print(f"拦截: {stats.get('blocked', 0)}")
    print(f"人工审核: {stats.get('reviewed', 0)}")
    
    print("\n按风险等级分布:")
    for level, count in stats.get('by_risk_level', {}).items():
        print(f"  {level}: {count}")
    
    print("\n按情感分布:")
    for sentiment, count in stats.get('by_sentiment', {}).items():
        print(f"  {sentiment}: {count}")
