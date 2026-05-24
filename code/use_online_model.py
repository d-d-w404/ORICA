import numpy as np
from sgd_online_learning import SGDOnlineLearner

def use_online_model():
    """使用在线学习模型的示例"""
    
    print("🎯 使用SGD在线学习模型")
    print("="*50)
    
    # 1. 加载预训练模型
    model_path = './Quick30/sgd_final_online_model.pkl'
    
    try:
        learner = SGDOnlineLearner()
        learner.load_model(model_path)
        print("✅ 模型加载成功")
        
        # 显示模型信息
        info = learner.get_model_info()
        print(f"📊 模型信息:")
        print(f"   模型名称: {info['model_name']}")
        print(f"   类别: {info['classes']}")
        print(f"   在线学习次数: {info['training_samples']}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 请先运行 sgd_online_learning.py 训练模型")
        return
    
    # 2. 预测新数据
    print(f"\n🔮 预测新数据:")
    
    # 生成模拟的新EEG特征数据
    new_features = np.random.randn(3, 85)  # 3个样本，每个85个特征
    print(f"   新数据形状: {new_features.shape}")
    
    # 进行预测
    predictions = learner.predict(new_features)
    probabilities = learner.predict_proba(new_features)
    
    print(f"   预测结果: {predictions}")
    print(f"   预测概率:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"     样本 {i+1}: 标签={pred}, 概率={prob}")
    
    # 3. 在线学习新数据
    print(f"\n🔄 在线学习新数据:")
    
    # 模拟新的带标签数据
    new_X = np.random.randn(2, 85)  # 2个新样本
    new_y = np.array([1, 0])  # 对应的标签
    
    print(f"   新训练数据: {new_X.shape}")
    print(f"   新标签: {new_y}")
    
    # 在线学习
    success = learner.online_learn(new_X, new_y)
    
    if success:
        print("✅ 在线学习成功")
        
        # 再次预测，看看模型是否改进
        print(f"\n🔮 学习后的预测:")
        new_predictions = learner.predict(new_features)
        new_probabilities = learner.predict_proba(new_features)
        
        print(f"   新预测结果: {new_predictions}")
        print(f"   新预测概率:")
        for i, (pred, prob) in enumerate(zip(new_predictions, new_probabilities)):
            print(f"     样本 {i+1}: 标签={pred}, 概率={prob}")
        
        # 保存更新后的模型
        updated_model_path = './Quick30/sgd_updated_online_model.pkl'
        learner.save_model(updated_model_path)
        print(f"💾 更新后的模型已保存: {updated_model_path}")
    
    # 4. 批量在线学习演示
    print(f"\n📦 批量在线学习演示:")
    
    # 生成多批新数据
    for batch_num in range(3):
        batch_X = np.random.randn(3, 85)
        batch_y = np.random.choice([0, 1], size=3)
        
        print(f"   批次 {batch_num + 1}:")
        print(f"     数据: {batch_X.shape}")
        print(f"     标签: {batch_y}")
        
        learner.online_learn(batch_X, batch_y)
        
        # 预测并显示结果
        batch_predictions = learner.predict(batch_X)
        accuracy = np.mean(batch_predictions == batch_y)
        print(f"     预测准确率: {accuracy:.3f}")
        print()
    
    print("✅ 在线学习演示完成!")

if __name__ == "__main__":
    use_online_model() 