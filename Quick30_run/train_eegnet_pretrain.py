#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用收集的原始数据训练EEGNet预训练模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv1 as EEGNet
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EEGNetPretrainer:
    """EEGNet预训练器"""
    
    def __init__(self, model_name="EEGNet_Pretrainer"):
        self.model_name = model_name
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 模型参数
        self.n_channels = None  # 将从数据中自动获取
        self.input_time_length = 1000  # 2秒 * 500Hz
        self.n_classes = None  # 将从数据中自动获取
        
        # 训练参数
        self.learning_rate = 5e-4
        self.batch_size = 8
        self.epochs = 300
        self.validation_split = 0.15
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"🧠 Initialized {model_name}")
    
    def load_and_preprocess_data(self, data_path):
        """加载和预处理数据"""
        print(f"📂 Loading data from: {data_path}")
        
        try:
            # 加载数据
            data = np.load(data_path)
            X = data['X']  # 原始EEG数据
            y = data['y']  # 标签数据
            
            print(f"📊 Data loaded successfully:")
            print(f"   X shape: {X.shape}")
            print(f"   y shape: {y.shape}")
            print(f"   Data type: {X.dtype}")
            
            # 自动设置模型参数
            self.n_channels = X.shape[1]  # 通道数
            self.n_classes = len(np.unique(y))  # 类别数
            
            print(f"🔧 Model parameters set:")
            print(f"   Channels: {self.n_channels}")
            print(f"   Classes: {self.n_classes}")
            print(f"   Time length: {self.input_time_length}")
            
            # 数据预处理
            X_processed, y_processed = self._preprocess_data(X, y)
            
            return X_processed, y_processed
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def _preprocess_data(self, X, y):
        """预处理数据"""
        print("🔄 Preprocessing data...")
        
        # 1. 标准化每个通道的时间序列
        X_processed = np.zeros_like(X)
        for i in range(X.shape[0]):  # 对每个样本
            for j in range(X.shape[1]):  # 对每个通道
                # 标准化时间序列
                mean_val = np.mean(X[i, j, :])
                std_val = np.std(X[i, j, :])
                if std_val > 0:
                    X_processed[i, j, :] = (X[i, j, :] - mean_val) / std_val
                else:
                    X_processed[i, j, :] = X[i, j, :] - mean_val
        
        # 2. 处理标签
        # 如果标签是多维的，取第一个维度
        if len(y.shape) > 1:
            y_processed = y[:, 0]
        else:
            y_processed = y
        
        # 3. 确保标签是连续的整数 (0, 1, 2, ...)
        unique_labels = np.unique(y_processed)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        y_processed = np.array([label_mapping[label] for label in y_processed])
        
        print(f"✅ Data preprocessing completed:")
        print(f"   Processed X shape: {X_processed.shape}")
        print(f"   Processed y shape: {y_processed.shape}")
        print(f"   Unique labels: {unique_labels} -> {list(range(len(unique_labels)))}")
        
        return X_processed, y_processed
    
    def create_model(self):
        """创建EEGNet模型"""
        if self.n_channels is None or self.n_classes is None:
            print("❌ Model parameters not set. Please load data first.")
            return False
        
        print("🔨 Creating EEGNet model...")
        
        self.model = EEGNet(
            in_chans=self.n_channels,
            n_classes=self.n_classes,
            input_window_samples=self.input_time_length
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✅ EEGNet model created:")
        print(f"   Input channels: {self.n_channels}")
        print(f"   Output classes: {self.n_classes}")
        print(f"   Time length: {self.input_time_length}")
        
        return True
    
    def prepare_data_loaders(self, X, y):
        """准备数据加载器"""
        print("📦 Preparing data loaders...")
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=y
        )
        
        print(f"📊 Data split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader):
        """训练模型"""
        if self.model is None:
            print("❌ Model not created. Please create model first.")
            return False
        
        print(f"🚀 Starting training...")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # 保存训练历史
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        self.is_trained = True
        print("✅ Training completed!")
        
        return True
    
    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        if not self.is_trained:
            print("❌ Model not trained yet.")
            return None
        
        print("🔍 Evaluating model...")
        
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            # 计算准确率
            correct = (predicted == y_test_tensor).sum().item()
            total = y_test_tensor.size(0)
            accuracy = correct / total
            
            print(f"📊 Test Results:")
            print(f"   Accuracy: {accuracy:.4f} ({correct}/{total})")
            
            # 计算每个类别的准确率
            unique_labels = np.unique(y_test)
            for label in unique_labels:
                mask = y_test == label
                class_correct = (predicted[mask] == y_test_tensor[mask]).sum().item()
                class_total = mask.sum()
                class_accuracy = class_correct / class_total
                print(f"   Class {label}: {class_accuracy:.4f} ({class_correct}/{class_total})")
        
        return accuracy
    
    def save_model(self, filepath):
        """保存模型"""
        if not self.is_trained:
            print("❌ Model not trained yet.")
            return False
        
        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存模型状态
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_config': {
                    'n_channels': self.n_channels,
                    'n_classes': self.n_classes,
                    'input_time_length': self.input_time_length
                },
                'training_history': {
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies
                }
            }, filepath)
            
            print(f"✅ Model saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if not self.train_losses:
            print("❌ No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    print("🧠 EEGNet Pretraining Script")
    print("=" * 50)
    
    # 创建预训练器
    trainer = EEGNetPretrainer()
    
    # 数据路径
    data_path = './Quick30/labeled_raw_eeg_data_listen_processed.npz'
    model_save_path = './Quick30/eegnet_pretrained_model_listen_processed.pt'
    plot_save_path = './Quick30/training_history.png'
    
    # 1. 加载和预处理数据
    X, y = trainer.load_and_preprocess_data(data_path)
    if X is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # 2. 创建模型
    if not trainer.create_model():
        print("❌ Failed to create model. Exiting.")
        return
    
    # 3. 准备数据加载器
    train_loader, val_loader = trainer.prepare_data_loaders(X, y)
    
    # 4. 训练模型
    if not trainer.train_model(train_loader, val_loader):
        print("❌ Training failed. Exiting.")
        return
    
    # 5. 评估模型
    # 使用验证集进行评估
    # 正确获取验证集数据
    X_val_tensor = val_loader.dataset.tensors[0]  # 获取X数据
    y_val_tensor = val_loader.dataset.tensors[1]  # 获取y数据
    X_val = X_val_tensor.numpy()
    y_val = y_val_tensor.numpy()
    trainer.evaluate_model(X_val, y_val)
    
    # 6. 保存模型
    trainer.save_model(model_save_path)
    
    # 7. 绘制训练历史
    trainer.plot_training_history(plot_save_path)
    
    print("\n🎉 Pretraining completed successfully!")
    print(f"📁 Model saved: {model_save_path}")
    print(f"📊 Training history: {plot_save_path}")
    print("\n💡 You can now use this model for online learning!")

if __name__ == "__main__":
    main()