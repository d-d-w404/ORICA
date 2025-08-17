#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试EEGNet标签映射功能
"""

import numpy as np
from eegnet_online_learning import EEGNetOnlineLearner

def test_label_mapping():
    """测试标签映射功能"""
    print("🧪 Testing EEGNet Label Mapping...")
    
    # 创建EEGNet模型
    learner = EEGNetOnlineLearner("Test_Label_Mapping")
    
    # 测试标签映射
    print("\n📊 Testing label mapping:")
    print(f"External label 7 -> Internal label: {learner.label_mapping[7]}")
    print(f"External label 8 -> Internal label: {learner.label_mapping[8]}")
    print(f"Internal label 0 -> External label: {learner.reverse_mapping[0]}")
    print(f"Internal label 1 -> External label: {learner.reverse_mapping[1]}")
    
    # 模拟EEG数据
    n_channels = 16
    time_length = 1000
    features = np.random.randn(n_channels, time_length)
    
    # 测试在线学习
    print("\n🔄 Testing online learning with external labels:")
    labels_7 = np.array([7])  # 左手想象
    labels_8 = np.array([8])  # 右手想象
    
    result1 = learner.online_learn(features, labels_7)
    print(f"✅ Learning with label 7: {result1}")
    
    result2 = learner.online_learn(features, labels_8)
    print(f"✅ Learning with label 8: {result2}")
    
    # 检查内部缓存
    print(f"📊 Internal label cache: {learner.label_cache}")
    
    # 测试预测
    print("\n🔄 Testing prediction:")
    prediction = learner.predict(features)
    print(f"✅ Prediction result: {prediction}")
    print(f"✅ Prediction type: {type(prediction[0])}")
    
    print("\n✅ Label mapping test completed!")

if __name__ == "__main__":
    test_label_mapping() 