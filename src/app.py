from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from inference import SentimentInference
from moderation import ModerationEngine, ContentType
from sensitive_words import get_sensitive_word_manager

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

inference_engine = None
moderation_engine = None


def load_inference_engine():
    global inference_engine, moderation_engine
    try:
        model_path = Config.BEST_MODEL_FILE
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False

        inference_engine = SentimentInference(model_path)
        moderation_engine = ModerationEngine(inference_engine)
        logger.info("推理引擎加载成功")
        logger.info("审核引擎初始化成功")
        return True
    except Exception as e:
        logger.error(f"加载推理引擎失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': inference_engine is not None,
        'moderation_loaded': moderation_engine is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': '缺少必需参数: text',
                'code': 400
            }), 400

        text = data['text']

        if not isinstance(text, str):
            return jsonify({
                'error': 'text参数必须是字符串',
                'code': 400
            }), 400

        if not text.strip():
            return jsonify({
                'error': 'text参数不能为空',
                'code': 400
            }), 400

        return_probabilities = data.get('return_probabilities', False)

        result = inference_engine.predict(text, return_probabilities=return_probabilities)

        response = {
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"预测成功: {text[:50]}... -> {result['predicted_label']}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'预测失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({
                'error': '缺少必需参数: texts',
                'code': 400
            }), 400

        texts = data['texts']

        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts参数必须是列表',
                'code': 400
            }), 400

        if len(texts) == 0:
            return jsonify({
                'error': 'texts列表不能为空',
                'code': 400
            }), 400

        if len(texts) > 100:
            return jsonify({
                'error': '批量预测最多支持100条文本',
                'code': 400
            }), 400

        return_probabilities = data.get('return_probabilities', False)

        results = inference_engine.predict_batch(texts, return_probabilities=return_probabilities)

        response = {
            'success': True,
            'data': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"批量预测成功: {len(texts)}条文本")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'批量预测失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': '缺少必需参数: text',
                'code': 400
            }), 400

        text = data['text']

        if not isinstance(text, str):
            return jsonify({
                'error': 'text参数必须是字符串',
                'code': 400
            }), 400

        if not text.strip():
            return jsonify({
                'error': 'text参数不能为空',
                'code': 400
            }), 400

        prediction_result = inference_engine.predict(text, return_probabilities=True)
        attention_result = inference_engine.get_attention_weights(text)

        response = {
            'success': True,
            'data': {
                'prediction': prediction_result,
                'attention': attention_result
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"分析成功: {text[:50]}... -> {prediction_result['predicted_label']}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'分析失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    try:
        if inference_engine is None:
            return jsonify({
                'error': '模型未加载',
                'code': 503
            }), 503

        info = {
            'vocab_size': Config.VOCAB_SIZE,
            'max_len': Config.MAX_LEN,
            'num_classes': Config.NUM_CLASSES,
            'embedding_dim': Config.EMBEDDING_DIM,
            'hidden_dim': Config.HIDDEN_DIM,
            'num_layers': Config.NUM_LAYERS,
            'dropout': Config.DROPOUT,
            'device': Config.DEVICE,
            'model_path': Config.BEST_MODEL_FILE
        }

        return jsonify({
            'success': True,
            'data': info
        }), 200

    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({
            'error': f'获取模型信息失败: {str(e)}',
            'code': 500
        }), 500


# ==================== 审核相关接口 ====================

@app.route('/moderate', methods=['POST'])
def moderate():
    """
    内容审核接口
    请求参数:
    - text: 待审核文本 (必需)
    - content_type: 内容类型 (可选, 默认comment) [topic/comment/username/signature]
    """
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': '缺少必需参数: text',
                'code': 400
            }), 400

        text = data['text']

        if not isinstance(text, str):
            return jsonify({
                'error': 'text参数必须是字符串',
                'code': 400
            }), 400

        if not text.strip():
            return jsonify({
                'error': 'text参数不能为空',
                'code': 400
            }), 400

        # 解析内容类型
        content_type_str = data.get('content_type', 'comment')
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            return jsonify({
                'error': f'无效的内容类型: {content_type_str}',
                'code': 400
            }), 400

        result = moderation_engine.moderate(text, content_type)

        response = {
            'success': True,
            'data': result.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"审核成功: {text[:50]}... -> {result.action.value}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"审核失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'审核失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/moderate/batch', methods=['POST'])
def moderate_batch():
    """
    批量审核接口
    请求参数:
    - texts: 待审核文本列表 (必需)
    - content_type: 内容类型 (可选, 默认comment)
    """
    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({
                'error': '缺少必需参数: texts',
                'code': 400
            }), 400

        texts = data['texts']

        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts参数必须是列表',
                'code': 400
            }), 400

        if len(texts) == 0:
            return jsonify({
                'error': 'texts列表不能为空',
                'code': 400
            }), 400

        if len(texts) > 100:
            return jsonify({
                'error': '批量审核最多支持100条文本',
                'code': 400
            }), 400

        content_type_str = data.get('content_type', 'comment')
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            return jsonify({
                'error': f'无效的内容类型: {content_type_str}',
                'code': 400
            }), 400

        results = moderation_engine.moderate_batch(texts, content_type)

        response = {
            'success': True,
            'data': [result.to_dict() for result in results],
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"批量审核成功: {len(texts)}条文本")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"批量审核失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'批量审核失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/moderation/config', methods=['GET'])
def get_moderation_config():
    """获取审核策略配置"""
    try:
        if moderation_engine is None:
            return jsonify({
                'error': '审核引擎未加载',
                'code': 503
            }), 503

        config = moderation_engine.get_config()

        return jsonify({
            'success': True,
            'data': config
        }), 200

    except Exception as e:
        logger.error(f"获取审核配置失败: {str(e)}")
        return jsonify({
            'error': f'获取审核配置失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/moderation/config', methods=['POST'])
def update_moderation_config():
    """更新审核策略配置"""
    try:
        if moderation_engine is None:
            return jsonify({
                'error': '审核引擎未加载',
                'code': 503
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'error': '请求体不能为空',
                'code': 400
            }), 400

        moderation_engine.update_config(**data)

        return jsonify({
            'success': True,
            'message': '审核策略已更新',
            'data': moderation_engine.get_config()
        }), 200

    except Exception as e:
        logger.error(f"更新审核配置失败: {str(e)}")
        return jsonify({
            'error': f'更新审核配置失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/sensitive_words/check', methods=['POST'])
def check_sensitive_words():
    """敏感词检测接口"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': '缺少必需参数: text',
                'code': 400
            }), 400

        text = data['text']

        if not isinstance(text, str):
            return jsonify({
                'error': 'text参数必须是字符串',
                'code': 400
            }), 400

        swm = get_sensitive_word_manager()
        result = swm.check(text)

        return jsonify({
            'success': True,
            'data': {
                'is_match': result.is_match,
                'matched_words': result.matched_words,
                'categories': [cat.value for cat in result.categories],
                'risk_score': result.risk_score,
                'is_severe': swm.is_severe(result)
            },
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"敏感词检测失败: {str(e)}")
        return jsonify({
            'error': f'敏感词检测失败: {str(e)}',
            'code': 500
        }), 500


@app.route('/sensitive_words/stats', methods=['GET'])
def get_sensitive_words_stats():
    """获取敏感词库统计信息"""
    try:
        swm = get_sensitive_word_manager()
        stats = swm.get_stats()

        return jsonify({
            'success': True,
            'data': stats
        }), 200

    except Exception as e:
        logger.error(f"获取敏感词统计失败: {str(e)}")
        return jsonify({
            'error': f'获取敏感词统计失败: {str(e)}',
            'code': 500
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': '接口不存在',
        'code': 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': '服务器内部错误',
        'code': 500
    }), 500


if __name__ == '__main__':
    if load_inference_engine():
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("启动失败：无法加载推理引擎")
        sys.exit(1)
