#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试EEGNet在线学习功能
"""

import numpy as np
import os
from eegnet_online_learning import EEGNetOnlineLearner

def test_eegnet_online_learning():
    """测试EEGNet在线学习功能"""
    
    print("🧪 Testing EEGNet Online Learning...")
    
    # 创建学习器
    learner = EEGNetOnlineLearner()
    
    # 生成模拟EEG数据 (16通道, 1000时间点)
    n_samples = 50
    n_channels = 16
    time_length = 1000
    
    print(f"📊 Generating {n_samples} EEG samples: {n_channels} channels x {time_length} time points")
    
    # 生成模拟EEG数据
    X = np.random.randn(n_samples, n_channels, time_length)
    y = np.random.choice([7, 8], n_samples)  # 左手和右手想象
    
    # 预训练
    print("\n📝 Starting pre-training...")
    accuracy = learner.pretrain(X, y)
    print(f"✅ Pre-training completed. Accuracy: {accuracy:.4f}")
    
    # 在线学习测试
    print("\n🔄 Testing online learning...")
    for i in range(10):
        # 生成新的EEG样本
        features = np.random.randn(n_channels, time_length)
        label = np.array([np.random.choice([7, 8])])
        
        # 预测
        prediction = learner.predict(features)
        proba = learner.predict_proba(features)
        
        # 在线学习
        success = learner.online_learn(features, label)
        
        print(f"Sample {i+1}: Pred={prediction[0]}, True={label[0]}, Success={success}")
        print(f"   Probabilities: [{proba[0, 0]:.3f}, {proba[0, 1]:.3f}]")
    
    # 保存模型
    print("\n💾 Saving model...")
    model_path = "./Quick30/eegnet_online_model.pt"
    success = learner.save_model(model_path)
    
    if success:
        print(f"✅ Model saved: {model_path}")
        
        # 测试加载模型
        print("\n📂 Testing model loading...")
        new_learner = EEGNetOnlineLearner()
        load_success = new_learner.load_model(model_path)
        
        if load_success:
            print("✅ Model loaded successfully!")
            
            # 测试加载后的模型
            test_features = np.random.randn(n_channels, time_length)
            prediction = new_learner.predict(test_features)
            print(f"✅ Loaded model prediction: {prediction[0]}")
        else:
            print("❌ Failed to load model")
    else:
        print("❌ Failed to save model")
    
    print("\n✅ EEGNet online learning test completed!")

if __name__ == "__main__":
    test_eegnet_online_learning() 