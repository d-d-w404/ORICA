#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试预训练模型的预测输出
"""

import numpy as np
import torch
from eegnet_online_learning import EEGNetOnlineLearner

def test_model_predictions():
    """测试模型预测输出"""
    print("🧠 Testing Model Predictions")
    print("=" * 50)
    
    # 创建学习器
    learner = EEGNetOnlineLearner()
    
    # 加载预训练模型
    model_path = './Quick30/eegnet_pretrained_model_c3c4.pt'
    
    try:
        if learner.load_model(model_path):
            print("✅ Model loaded successfully")
        else:
            print("❌ Failed to load model")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 生成测试数据
    print("\n📊 Generating test data...")
    
    # 模拟EEG数据 (25通道, 1000时间点)
    test_data_1 = np.random.randn(25, 1000)  # 测试样本1
    test_data_2 = np.random.randn(25, 1000)  # 测试样本2
    
    print(f"Test data shape: {test_data_1.shape}")
    
    # 进行预测
    print("\n🔍 Making predictions...")
    
    # 预测类别
    pred_1 = learner.predict(test_data_1)
    pred_2 = learner.predict(test_data_2)
    
    # 预测概率
    proba_1 = learner.predict_proba(test_data_1)
    proba_2 = learner.predict_proba(test_data_2)
    
    # 显示结果
    print("\n📈 Prediction Results:")
    print("=" * 30)
    
    print(f"Sample 1:")
    print(f"   Prediction: {pred_1[0]} (类别 {pred_1[0]})")
    print(f"   Probabilities: [{proba_1[0][0]:.3f}, {proba_1[0][1]:.3f}]")
    print(f"   Original Label: {7 if pred_1[0] == 0 else 8}")
    print(f"   Meaning: {'左手想象' if pred_1[0] == 0 else '右手想象'}")
    
    print(f"\nSample 2:")
    print(f"   Prediction: {pred_2[0]} (类别 {pred_2[0]})")
    print(f"   Probabilities: [{proba_2[0][0]:.3f}, {proba_2[0][1]:.3f}]")
    print(f"   Original Label: {7 if pred_2[0] == 0 else 8}")
    print(f"   Meaning: {'左手想象' if pred_2[0] == 0 else '右手想象'}")
    
    # 解释预测值
    print("\n📋 Prediction Value Explanation:")
    print("=" * 40)
    print("Model Output Values:")
    print("   0 → 左手想象 (原始标签: 7)")
    print("   1 → 右手想象 (原始标签: 8)")
    print("\nProbability Format:")
    print("   [P(类别0), P(类别1)]")
    print("   [P(左手想象), P(右手想象)]")
    
    # 模型信息
    print("\n🔧 Model Information:")
    print("=" * 25)
    model_info = learner.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")

def test_with_real_data():
    """使用真实数据测试"""
    print("\n🧪 Testing with Real Data")
    print("=" * 40)
    
    try:
        # 加载真实数据
        data = np.load('./Quick30/labeled_raw_eeg_data.npz')
        X = data['X']
        y = data['y']
        
        print(f"Loaded real data: {X.shape}")
        print(f"Labels: {y.shape}")
        
        # 创建学习器
        learner = EEGNetOnlineLearner()
        model_path = './Quick30/eegnet_pretrained_model_c3c4.pt'
        
        if learner.load_model(model_path):
            print("✅ Model loaded")
            
            # 测试几个真实样本
            for i in range(min(5, len(X))):
                sample = X[i]
                true_label = y[i]
                
                # 预测
                pred = learner.predict(sample)
                proba = learner.predict_proba(sample)
                
                print(f"\nSample {i+1}:")
                print(f"   True Label: {true_label}")
                print(f"   Prediction: {pred[0]}")
                print(f"   Probabilities: [{proba[0][0]:.3f}, {proba[0][1]:.3f}]")
                print(f"   Correct: {'✅' if pred[0] == (0 if true_label[0] == 7 else 1) else '❌'}")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")

if __name__ == "__main__":
    test_model_predictions()
    test_with_real_data() 