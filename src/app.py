from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys
import traceback
from datetime import datetime
import logging

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from inference import SentimentInference
from moderation import ModerationEngine
from sensitive_words import get_sensitive_word_manager, WordCategory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="校园论坛内容审核系统",
    description="基于LSTM的情感分析和内容审核API",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
inference_engine = None
moderation_engine = None

# 数据模型
class PredictRequest(BaseModel):
    text: str = Field(..., description="待预测的文本", min_length=1)
    return_probabilities: Optional[bool] = Field(False, description="是否返回概率分布")

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="待预测的文本列表", min_items=1, max_items=100)
    return_probabilities: Optional[bool] = Field(False, description="是否返回概率分布")

class ModerateRequest(BaseModel):
    text: str = Field(..., description="待审核的文本", min_length=1)

class ModerateBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="待审核的文本列表", min_items=1, max_items=100)

# 敏感词管理数据模型
class AddKeywordRequest(BaseModel):
    word: str = Field(..., description="敏感词", min_length=1)
    category: str = Field(..., description="敏感词类别")
    severity: Optional[int] = Field(1, description="严重程度", ge=1, le=3)

class RemoveKeywordRequest(BaseModel):
    word: str = Field(..., description="敏感词", min_length=1)

class WordCloudRequest(BaseModel):
    limit: Optional[int] = Field(100, description="返回词的数量限制", ge=1, le=1000)

# 加载推理引擎
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

# 启动时加载引擎
@app.on_event("startup")
def startup_event():
    logger.info("=" * 60)
    logger.info("启动情感分析API服务")
    logger.info("=" * 60)
    
    if not load_inference_engine():
        logger.error("无法加载推理引擎，服务启动失败")
        raise Exception("无法加载推理引擎")
    
    logger.info(f"模型配置:")
    logger.info(f"  词表大小: {Config.VOCAB_SIZE}")
    logger.info(f"  最大序列长度: {Config.MAX_LEN}")
    logger.info(f"  分类数: {Config.NUM_CLASSES}")
    logger.info(f"  设备: {Config.DEVICE}")

# 健康检查
@app.get("/health", summary="健康检查", description="检查服务和模型状态")
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': inference_engine is not None,
        'moderation_loaded': moderation_engine is not None
    }

# 单条预测
@app.post("/predict", summary="单条文本预测", description="对单条文本进行情感分析预测")
def predict(request: PredictRequest):
    try:
        text = request.text
        return_probabilities = request.return_probabilities
        
        result = inference_engine.predict(text, return_probabilities=return_probabilities)
        
        response = {
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"预测成功: {text[:50]}... -> {result['predicted_label']}")
        
        return response
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'预测失败: {str(e)}')

# 批量预测
@app.post("/predict/batch", summary="批量文本预测", description="对多条文本进行情感分析预测")
def predict_batch(request: PredictBatchRequest):
    try:
        texts = request.texts
        return_probabilities = request.return_probabilities
        
        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="批量预测最多支持100条文本")
        
        results = inference_engine.predict_batch(texts, return_probabilities=return_probabilities)
        
        response = {
            'success': True,
            'data': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"批量预测成功: {len(texts)}条文本")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'批量预测失败: {str(e)}')

# 文本分析
@app.post("/analyze", summary="文本分析", description="对文本进行情感分析和注意力权重分析")
def analyze(request: PredictRequest):
    try:
        text = request.text
        
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
        
        return response
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'分析失败: {str(e)}')

# 模型信息
@app.get("/model/info", summary="模型信息", description="获取模型配置信息")
def model_info():
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        
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
        
        return {
            'success': True,
            'data': info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f'获取模型信息失败: {str(e)}')

# 单条审核
@app.post("/moderate", summary="单条文本审核", description="对单条文本进行内容审核")
def moderate(request: ModerateRequest):
    try:
        text = request.text
        
        result = moderation_engine.moderate(text)
        
        response = {
            'success': True,
            'data': result.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"审核成功: {text[:50]}... -> {result.action.value if result.action else 'N/A'}")
        
        return response
    except Exception as e:
        logger.error(f"审核失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'审核失败: {str(e)}')

# 批量审核
@app.post("/moderate/batch", summary="批量文本审核", description="对多条文本进行内容审核")
def moderate_batch(request: ModerateBatchRequest):
    try:
        texts = request.texts
        
        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="批量审核最多支持100条文本")
        
        results = moderation_engine.moderate_batch(texts)
        
        response = {
            'success': True,
            'data': [result.to_dict() for result in results],
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"批量审核成功: {len(texts)}条文本")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量审核失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'批量审核失败: {str(e)}')

# 审核统计
@app.get("/moderation/stats", summary="审核统计", description="获取审核统计信息")
def moderation_stats():
    try:
        if moderation_engine is None:
            raise HTTPException(status_code=503, detail="审核引擎未加载")
        
        stats = moderation_engine.get_moderation_stats()
        
        return {
            'success': True,
            'data': stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取审核统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f'获取审核统计失败: {str(e)}')

# 敏感词云数据
@app.get("/sensitive/wordcloud", summary="敏感词云数据", description="获取高频敏感词云数据")
def get_word_cloud(limit: int = 100):
    try:
        swm = get_sensitive_word_manager()
        word_cloud_data = swm.get_word_cloud_data(limit)
        
        return {
            'success': True,
            'data': word_cloud_data,
            'count': len(word_cloud_data),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取敏感词云数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f'获取敏感词云数据失败: {str(e)}')

# 敏感词库统计
@app.get("/sensitive/stats", summary="敏感词库统计", description="获取敏感词库统计信息")
def get_sensitive_stats():
    try:
        swm = get_sensitive_word_manager()
        stats = swm.get_stats()
        
        return {
            'success': True,
            'data': stats
        }
    except Exception as e:
        logger.error(f"获取敏感词库统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f'获取敏感词库统计失败: {str(e)}')

# 添加敏感词
@app.post("/sensitive/add", summary="添加敏感词", description="添加新的敏感词")
def add_sensitive_word(request: AddKeywordRequest):
    try:
        swm = get_sensitive_word_manager()
        
        # 验证类别是否有效
        try:
            category = WordCategory(request.category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f'无效的敏感词类别: {request.category}')
        
        success = swm.add_keyword(request.word, category, request.severity)
        
        if success:
            logger.info(f"添加敏感词成功: {request.word} (类别: {category.value})")
            return {
                'success': True,
                'message': f"敏感词 '{request.word}' 添加成功",
                'timestamp': datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="添加敏感词失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加敏感词失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f'添加敏感词失败: {str(e)}')

# 移除敏感词
@app.post("/sensitive/remove", summary="移除敏感词", description="移除敏感词")
def remove_sensitive_word(request: RemoveKeywordRequest):
    try:
        swm = get_sensitive_word_manager()
        
        success = swm.remove_keyword(request.word)
        
        if success:
            logger.info(f"移除敏感词成功: {request.word}")
            return {
                'success': True,
                'message': f"敏感词 '{request.word}' 移除成功",
                'timestamp': datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"敏感词 '{request.word}' 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"移除敏感词失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f'移除敏感词失败: {str(e)}')

# 获取敏感词类别
@app.get("/sensitive/categories", summary="敏感词类别", description="获取所有敏感词类别")
def get_sensitive_categories():
    try:
        categories = [{
            'value': cat.value,
            'name': cat.name
        } for cat in WordCategory]
        
        return {
            'success': True,
            'data': categories
        }
    except Exception as e:
        logger.error(f"获取敏感词类别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f'获取敏感词类别失败: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    
    host = os.environ.get('HOST', 'localhost')
    port = int(os.environ.get('PORT', 8000))
    
    logger.info(f"服务启动在: http://{host}:{port}")
    logger.info(f"API文档地址: http://{host}:{port}/docs")
    
    uvicorn.run(app, host='localhost', port=port, log_config=None)