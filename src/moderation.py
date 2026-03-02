"""
内容审核引擎 - 综合敏感词过滤和情感分析
实现"AI初筛 + 人工复核"双层防控体系
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from sensitive_words import get_sensitive_word_manager, WordCategory, MatchResult
from inference import SentimentInference

logger = logging.getLogger(__name__)


class ModerationAction(Enum):
    """审核决策"""
    ALLOW = "allow"           # 允许通过
    BLOCK = "block"           # 拒绝通过
    MANUAL_REVIEW = "manual"  # 需要人工审核


class ContentType(Enum):
    """内容类型"""
    TOPIC = "topic"           # 帖子
    COMMENT = "comment"       # 评论
    USERNAME = "username"     # 用户名
    SIGNATURE = "signature"   # 个性签名


@dataclass
class ModerationResult:
    """审核结果"""
    action: ModerationAction          # 审核决策
    content_type: ContentType         # 内容类型
    text: str                         # 原始文本
    sentiment_label: str              # 情感标签
    sentiment_confidence: float       # 情感置信度
    sentiment_probabilities: Dict[str, float] = field(default_factory=dict)  # 情感概率分布
    sensitive_match: bool = False     # 是否命中敏感词
    sensitive_words: List[str] = field(default_factory=list)  # 命中的敏感词
    sensitive_categories: List[str] = field(default_factory=list)  # 敏感词类别
    sensitive_risk_score: float = 0.0  # 敏感词风险分数
    is_severe: bool = False           # 是否严重违规
    reason: str = ""                  # 审核原因
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action": self.action.value,
            "content_type": self.content_type.value,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "sentiment": {
                "label": self.sentiment_label,
                "confidence": round(self.sentiment_confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in self.sentiment_probabilities.items()}
            },
            "sensitive_words": {
                "matched": self.sensitive_match,
                "words": self.sensitive_words,
                "categories": self.sensitive_categories,
                "risk_score": round(self.sensitive_risk_score, 2),
                "is_severe": self.is_severe
            },
            "reason": self.reason,
            "timestamp": self.timestamp
        }


class ModerationEngine:
    """
    内容审核引擎
    整合敏感词过滤和情感分析，输出最终审核决策
    """

    def __init__(self, inference_engine: SentimentInference):
        """
        初始化审核引擎
        :param inference_engine: 情感分析推理引擎
        """
        self.inference = inference_engine
        self.swm = get_sensitive_word_manager()

        # 审核策略配置
        self.config = {
            # 情感分析阈值
            "negative_threshold": 0.85,        # 负向情感置信度阈值 (提高阈值减少误判)
            "negative_block_threshold": 0.98,  # 负向情感直接拦截阈值

            # 敏感词阈值
            "sensitive_risk_threshold": 40,    # 敏感词风险分数阈值
            "sensitive_block_threshold": 60    # 敏感词直接拦截阈值
        }

        logger.info("审核引擎初始化完成")

    def moderate(self, text: str, content_type: ContentType = ContentType.COMMENT) -> ModerationResult:
        """
        执行内容审核
        :param text: 待审核文本
        :param content_type: 内容类型
        :return: 审核结果
        """
        if not text or not text.strip():
            return ModerationResult(
                action=ModerationAction.ALLOW,
                content_type=content_type,
                text=text or "",
                sentiment_label="中性",
                sentiment_confidence=1.0,
                reason="空内容自动通过"
            )

        # 1. 敏感词检测 (第一层过滤)
        sensitive_result = self.swm.check(text)

        # 2. 情感分析 (第二层过滤)
        sentiment_result = self.inference.predict(text, return_probabilities=True)

        # 3. 综合决策
        action, reason = self._make_decision(
            sensitive_result,
            sentiment_result,
            content_type
        )

        # 构建审核结果
        result = ModerationResult(
            action=action,
            content_type=content_type,
            text=text,
            sentiment_label=sentiment_result.get("predicted_label", "未知"),
            sentiment_confidence=sentiment_result.get("confidence", 0.0),
            sentiment_probabilities=sentiment_result.get("probabilities", {}),
            sensitive_match=sensitive_result.is_match,
            sensitive_words=sensitive_result.matched_words,
            sensitive_categories=[cat.value for cat in sensitive_result.categories],
            sensitive_risk_score=sensitive_result.risk_score,
            is_severe=self.swm.is_severe(sensitive_result),
            reason=reason
        )

        logger.info(f"审核完成: {content_type.value} -> {action.value}, 原因: {reason}")
        return result

    def _make_decision(self, sensitive_result: MatchResult,
                       sentiment_result: Dict,
                       content_type: ContentType) -> tuple:
        """
        综合决策逻辑
        :return: (action, reason)
        """
        # 获取情感分析结果
        sentiment_label = sentiment_result.get("predicted_label", "中性")
        sentiment_confidence = sentiment_result.get("confidence", 0.0)
        probabilities = sentiment_result.get("probabilities", {})
        negative_prob = probabilities.get("负向", 0.0)

        # 严重敏感词 - 直接拦截
        if self.swm.is_severe(sensitive_result):
            return ModerationAction.BLOCK, f"包含严重敏感词: {sensitive_result.matched_words[:3]}"

        # 敏感词风险分数过高 - 直接拦截
        if sensitive_result.risk_score >= self.config["sensitive_block_threshold"]:
            return ModerationAction.BLOCK, f"敏感词风险分数过高: {sensitive_result.risk_score}"

        # 负向情感置信度极高 - 直接拦截
        if negative_prob >= self.config["negative_block_threshold"]:
            return ModerationAction.BLOCK, f"负向情感置信度过高: {negative_prob:.2%}"

        # 统一审核标准：敏感词风险分数中等 或 负向情感置信度中等 -> 人工审核
        needs_manual_review = False
        reasons = []

        if sensitive_result.risk_score >= self.config["sensitive_risk_threshold"]:
            needs_manual_review = True
            reasons.append(f"敏感词风险: {sensitive_result.risk_score}")

        if sentiment_label == "负向" and negative_prob >= self.config["negative_threshold"]:
            needs_manual_review = True
            reasons.append(f"负向情感: {negative_prob:.2%}")

        if needs_manual_review:
            return ModerationAction.MANUAL_REVIEW, "; ".join(reasons)

        # 通过审核
        return ModerationAction.ALLOW, "内容正常"

    def moderate_batch(self, texts: List[str], content_type: ContentType = ContentType.COMMENT) -> List[ModerationResult]:
        """
        批量审核
        :param texts: 文本列表
        :param content_type: 内容类型
        :return: 审核结果列表
        """
        results = []
        for text in texts:
            result = self.moderate(text, content_type)
            results.append(result)
        return results

    def update_config(self, **kwargs):
        """
        更新审核策略配置
        :param kwargs: 配置项
        """
        self.config.update(kwargs)
        logger.info(f"审核策略已更新: {kwargs}")

    def get_config(self) -> Dict:
        """获取当前审核策略配置"""
        return self.config.copy()
