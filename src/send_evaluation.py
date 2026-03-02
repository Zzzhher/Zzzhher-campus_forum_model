"""
手动发送模型评估数据到后端
用于补充之前训练完成但未保存到数据库的评估指标
"""
import json
import requests
from datetime import datetime
import os

# 后端API地址
BACKEND_URL = "http://localhost:8080"

# 从测试结果文件中加载数据
def load_test_results():
    results_file = os.path.join('..', 'results', 'test_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"警告: 未找到测试结果文件 {results_file}")
        return None

# 从训练历史文件中加载数据
def load_training_history():
    history_file = os.path.join('..', 'results', 'training_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"警告: 未找到训练历史文件 {history_file}")
        return None

def send_evaluation_to_backend():
    """将模型评估指标发送到后端API进行持久化"""
    try:
        print("=" * 60)
        print("手动发送模型评估数据到后端")
        print("=" * 60)

        # 加载测试结果
        results = load_test_results()
        if not results:
            print("✗ 无法加载测试结果，请确保已经运行过 train.py")
            return False

        # 加载训练历史
        history = load_training_history()
        if not history:
            print("✗ 无法加载训练历史")
            return False

        # 生成模型版本号（使用当前时间）
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 构建评估数据（注意：evalDate 和 evalTime 由后端自动设置）
        evaluation_data = {
            'modelVersion': model_version,
            'accuracy': round(results.get('accuracy', 0), 4),
            'precisionScore': round(results.get('precision_macro', 0), 4),
            'recall': round(results.get('recall_macro', 0), 4),
            'f1Score': round(results.get('f1_macro', 0), 4),
            'testSamples': 3717,  # 测试集样本数
            'trainingSamples': 29731,  # 训练集样本数
            'epochs': len(history.get('train_loss', [])),
            'batchSize': 64,
            'learningRate': 0.001,
            'modelPath': 'D:\\JavaProjects\\campus_forum_model\\models\\best_model.pth',
            'confusionMatrix': json.dumps(results.get('confusion_matrix', {})),
            'classReport': json.dumps(results.get('class_report', {})),
            'isBest': 0,  # 默认不是最佳，需要管理员确认
            'remark': f'手动上传，测试准确率: {results.get("accuracy", 0)*100:.2f}%'
        }

        print(f"\n模型版本: {model_version}")
        print(f"准确率: {evaluation_data['accuracy']:.4f}")
        print(f"F1分数: {evaluation_data['f1Score']:.4f}")
        print(f"训练轮数: {evaluation_data['epochs']}")

        # 发送请求到后端
        url = f"{BACKEND_URL}/api/admin/model/evaluations"
        headers = {
            'Content-Type': 'application/json'
        }

        print(f"\n正在发送到: {url}")
        response = requests.post(url, json=evaluation_data, headers=headers, timeout=10)

        if response.status_code == 200:
            print(f"✓ 评估指标已成功发送到后端 (版本: {model_version})")
            print(f"\n发送的数据:")
            print(json.dumps(evaluation_data, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"✗ 发送评估指标失败: HTTP {response.status_code}")
            print(f"  响应: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到后端服务")
        print(f"  请确保后端服务运行在 {BACKEND_URL}")
        return False
    except Exception as e:
        print(f"✗ 发送评估指标时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = send_evaluation_to_backend()
    if success:
        print("\n" + "=" * 60)
        print("操作成功！请在数据库中查看记录")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("操作失败！")
        print("=" * 60)
