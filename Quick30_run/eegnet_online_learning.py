#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEGNet Online Learning Wrapper
支持在线学习的EEGNet模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from braindecode.models import EEGNetv1 as EEGNet
from sklearn.preprocessing import StandardScaler
import time

class EEGNetOnlineLearner:
    """EEGNet在线学习器"""
    
    def __init__(self, model_name="EEGNet_Online_Learner"):
        self.model_name = model_name
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 模型参数
        self.n_channels = 25 # 根据您的EEG通道数调整
        self.input_time_length = 1000  # 2秒 * 500Hz
        self.n_classes = 2  # 左手和右手想象 (7, 8)
        
        # 在线学习参数
        self.learning_rate = 1e-4
        self.batch_size = 1
        self.epochs_per_update = 1
        
        # 数据缓存
        self.feature_cache = []
        self.label_cache = []
        self.max_cache_size = 100  # 最大缓存大小
        
        print(f"🧠 Initialized {model_name}")
    
    def _create_model(self):
        """创建EEGNet模型"""
        if self.model is None:
            self.model = EEGNet(
                in_chans=self.n_channels,
                n_classes=self.n_classes,
                input_window_samples=self.input_time_length
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.CrossEntropyLoss()
            print(f"✅ EEGNet model created: {self.n_channels} channels, {self.n_classes} classes")
    
    def pretrain(self, X, y, model_path=None):
        """预训练模型"""
        print("📝 Starting EEGNet pre-training...")
        
        self._create_model()
        
        # 数据预处理
        X = np.array(X)
        y = np.array(y)
        
        # 检查数据形状并调整
        if len(X.shape) == 2:
            # 如果是2D (samples, features)，需要重塑为3D (samples, channels, time)
            n_samples, n_features = X.shape
            n_channels = self.n_channels
            time_length = n_features // n_channels
            
            if n_features % n_channels == 0:
                X = X.reshape(n_samples, n_channels, time_length)
                print(f"✅ Reshaped data: {X.shape}")
            else:
                print(f"⚠️ Cannot reshape {n_features} features into {n_channels} channels")
                return 0.0
        
        # 标准化每个通道
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_scaled[i, j, :] = (X[i, j, :] - np.mean(X[i, j, :])) / (np.std(X[i, j, :]) + 1e-8)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # 训练模型
        self.model.train()
        for epoch in range(50):  # 预训练50个epoch
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # 计算准确率
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        
        self.is_trained = True
        print(f"✅ Pre-training completed. Accuracy: {accuracy:.4f}")
        
        # 保存模型
        if model_path:
            self.save_model(model_path)
        
        return accuracy
    
    def online_learn(self, features, labels):
        """在线学习"""
        try:
            # 如果模型还没有创建，先创建模型
            if self.model is None:
                self._create_model()
            
            # 处理特征数据
            if len(features.shape) == 2:
                # 如果是2D (channels, time)，添加batch维度
                features = features.reshape(1, features.shape[0], features.shape[1])
            
            # 添加到缓存
            self.feature_cache.append(features)
            self.label_cache.append(labels[0])
            
            # 限制缓存大小
            if len(self.feature_cache) > self.max_cache_size:
                self.feature_cache.pop(0)
                self.label_cache.pop(0)
            
            # 当缓存达到一定大小时进行训练
            if len(self.feature_cache) >= 3:  # 每3个样本训练一次
                print("training")
                self._train_on_cache()
                self.is_trained = True  # 标记模型已训练
            
            return True
            
        except Exception as e:
            print(f"❌ Online learning error: {e}")
            return False
    
    def _train_on_cache(self):
        """基于缓存数据进行训练"""
        if len(self.feature_cache) < 2:
            return
        
        # 准备数据
        X_cache = np.concatenate(self.feature_cache, axis=0)  # (batch, channels, time)
        y_cache = np.array(self.label_cache)
        
        # 标准化每个样本的每个通道
        X_scaled = np.zeros_like(X_cache)
        for i in range(X_cache.shape[0]):
            for j in range(X_cache.shape[1]):
                X_scaled[i, j, :] = (X_cache[i, j, :] - np.mean(X_cache[i, j, :])) / (np.std(X_cache[i, j, :]) + 1e-8)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y_cache)
        
        # 训练
        self.model.train()
        for _ in range(self.epochs_per_update):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
    
    def predict(self, features):
        """预测"""
        if not self.is_trained or self.model is None:
            print("⚠️ Model not trained yet.")
            # 返回随机预测 (7 或 8)
            import random
            return np.array([random.choice([7, 8])])
        
        try:
            self.model.eval()
            with torch.no_grad():
                # 处理特征数据
                if len(features.shape) == 2:
                    # 如果是2D (channels, time)，添加batch维度
                    features = features.reshape(1, features.shape[0], features.shape[1])
                
                # 标准化
                features_scaled = np.zeros_like(features)
                for i in range(features.shape[0]):
                    for j in range(features.shape[1]):
                        features_scaled[i, j, :] = (features[i, j, :] - np.mean(features[i, j, :])) / (np.std(features[i, j, :]) + 1e-8)
                
                X_tensor = torch.FloatTensor(features_scaled)
                
                # 预测
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                
                return predicted.numpy()
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return np.array([0])
    
    def predict_proba(self, features):
        """预测概率"""
        if not self.is_trained or self.model is None:
            print("⚠️ Model not trained yet.")
            return np.array([[0.5, 0.5]])
        
        try:
            self.model.eval()
            with torch.no_grad():
                # 处理特征数据
                if len(features.shape) == 2:
                    # 如果是2D (channels, time)，添加batch维度
                    features = features.reshape(1, features.shape[0], features.shape[1])
                
                # 标准化
                features_scaled = np.zeros_like(features)
                for i in range(features.shape[0]):
                    for j in range(features.shape[1]):
                        features_scaled[i, j, :] = (features[i, j, :] - np.mean(features[i, j, :])) / (np.std(features[i, j, :]) + 1e-8)
                
                X_tensor = torch.FloatTensor(features_scaled)
                
                # 预测概率
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                return probabilities.numpy()
                
        except Exception as e:
            print(f"❌ Probability prediction error: {e}")
            return np.array([[0.5, 0.5]])
    
    def save_model(self, filepath):
        """保存模型"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存模型状态和相关信息
            model_data = {
                'model_state_dict': self.model.state_dict() if self.model else None,
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'model_params': {
                    'n_channels': self.n_channels,
                    'input_time_length': self.input_time_length,
                    'n_classes': self.n_classes
                },
                'is_trained': self.is_trained
            }
            
            torch.save(model_data, filepath)
            print(f"💾 EEGNet model saved: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            if not os.path.exists(filepath):
                print(f"❌ Model file not found: {filepath}")
                return False
            
            # 加载模型数据
            model_data = torch.load(filepath, map_location='cpu')
            
            # 检查模型格式（兼容新旧格式）
            if 'model_config' in model_data:
                # 新格式：来自预训练脚本
                print("📂 Loading pretrained model format...")
                self.n_channels = model_data['model_config']['n_channels']
                self.input_time_length = model_data['model_config']['input_time_length']
                self.n_classes = model_data['model_config']['n_classes']
                self.is_trained = True  # 预训练模型默认已训练
                
                # 创建模型
                self._create_model()
                
                # 加载模型状态
                if model_data['model_state_dict']:
                    self.model.load_state_dict(model_data['model_state_dict'])
                
                # 重新初始化优化器（预训练模型不需要加载优化器状态）
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
                print(f"✅ Pretrained EEGNet model loaded: {filepath}")
                print(f"   Channels: {self.n_channels}, Classes: {self.n_classes}")
                
            elif 'model_params' in model_data:
                # 旧格式：来自在线学习
                print("📂 Loading online learning model format...")
                self.n_channels = model_data['model_params']['n_channels']
                self.input_time_length = model_data['model_params']['input_time_length']
                self.n_classes = model_data['model_params']['n_classes']
                self.is_trained = model_data['is_trained']
                
                # 创建模型
                self._create_model()
                
                # 加载状态
                if model_data['model_state_dict']:
                    self.model.load_state_dict(model_data['model_state_dict'])
                if model_data['optimizer_state_dict']:
                    self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                
                print(f"✅ Online learning EEGNet model loaded: {filepath}")
            else:
                print("❌ Unknown model format")
                return False
            
            # 重新初始化scaler
            self.scaler = StandardScaler()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'EEGNet',
            'n_channels': self.n_channels,
            'input_time_length': self.input_time_length,
            'n_classes': self.n_classes,
            'is_trained': self.is_trained,
            'cache_size': len(self.feature_cache)
        }

# 测试函数
def test_eegnet_online_learning():
    """测试EEGNet在线学习功能"""
    print("🧪 Testing EEGNet Online Learning...")
    
    # 创建学习器
    learner = EEGNetOnlineLearner()
    
    # 生成模拟数据
    n_samples = 100
    n_features = 80  # 假设有80个特征
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([7, 8], n_samples)
    
    # 预训练
    accuracy = learner.pretrain(X, y)
    print(f"Pre-training accuracy: {accuracy:.4f}")
    
    # 在线学习测试
    for i in range(10):
        features = np.random.randn(1, n_features)
        label = np.array([np.random.choice([7, 8])])
        
        # 预测
        prediction = learner.predict(features)
        proba = learner.predict_proba(features)
        
        # 在线学习
        success = learner.online_learn(features, label)
        
        print(f"Sample {i+1}: Pred={prediction[0]}, True={label[0]}, Success={success}")
    
    # 保存模型
    learner.save_model("./Quick30/eegnet_online_model.pt")
    
    print("✅ EEGNet online learning test completed!")

if __name__ == "__main__":
    test_eegnet_online_learning() 