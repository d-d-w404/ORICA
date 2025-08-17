import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import os
import time

class SGDOnlineLearner:
    """SGD在线学习器"""
    
    def __init__(self, model_name="SGD_Online_Learner"):
        self.model_name = model_name
        self.scaler = StandardScaler()
        
        # 尝试不同的损失函数，优先选择支持概率预测的
        loss_options = ['log_loss', 'modified_huber', 'squared_loss']
        selected_loss = None
        
        for loss in loss_options:
            try:
                test_model = SGDClassifier(loss=loss, max_iter=1)
                # 如果能创建成功，就使用这个损失函数
                selected_loss = loss
                print(f"✅ 使用损失函数: {loss}")
                break
            except Exception as e:
                print(f"❌ 损失函数 {loss} 不可用: {e}")
                continue
        
        if selected_loss is None:
            # 如果都不行，使用默认的
            selected_loss = 'modified_huber'
            print(f"⚠️ 使用默认损失函数: {selected_loss}")
        
        self.model = SGDClassifier(
            loss=selected_loss,  # 使用检测到的损失函数
            penalty='l2',        # L2正则化
            alpha=0.001,         # 正则化强度
            max_iter=1,          # 每次只迭代1次，适合在线学习
            random_state=42,
            tol=1e-3
        )
        self.is_fitted = False
        self.classes_ = None
        self.training_history = []
        
    def pretrain(self, X, y, save_path='./Quick30/sgd_pretrained_model.pkl'):
        """预训练模型"""
        print(f"🔄 开始预训练 {self.model_name}...")
        print(f"   数据形状: X={X.shape}, y={y.shape}")
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 获取唯一类别
        self.classes_ = np.unique(y)
        print(f"   类别: {self.classes_}")
        
        # 预训练模型
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # 评估预训练模型
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f"✅ 预训练完成，准确率: {accuracy:.4f}")
        
        # 保存预训练模型
        self.save_model(save_path)
        print(f"💾 预训练模型已保存: {save_path}")
        
        return accuracy
    
    def online_learn(self, X_new, y_new, verbose=True):
        """在线学习新数据"""
        if not self.is_fitted:
            print("❌ 模型尚未预训练，请先调用 pretrain()")
            return False
        
        # 标准化新数据
        X_new_scaled = self.scaler.transform(X_new)
        
        # 在线学习
        self.model.partial_fit(X_new_scaled, y_new, classes=self.classes_)
        
        # 记录学习历史
        timestamp = time.time()
        self.training_history.append({
            'timestamp': timestamp,
            'samples_added': len(X_new),
            'new_labels': y_new.tolist() if hasattr(y_new, 'tolist') else y_new
        })
        
        if verbose:
            print(f"✅ 在线学习完成，添加了 {len(X_new)} 个样本")
        
        return True
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            print("❌ 模型尚未训练")
            return None
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            print("❌ 模型尚未训练")
            return None
        
        X_scaled = self.scaler.transform(X)
        
        # 检查模型是否支持概率预测
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # 如果不支持概率预测，使用决策函数转换为概率
            try:
                decision_scores = self.model.decision_function(X_scaled)
                # 使用sigmoid函数将决策分数转换为概率
                import numpy as np
                proba = 1 / (1 + np.exp(-decision_scores))
                # 确保概率在[0,1]范围内
                proba = np.clip(proba, 0, 1)
                # 返回二分类概率
                return np.column_stack([1 - proba, proba])
            except Exception as e:
                print(f"❌ 概率预测失败: {e}")
                # 返回默认概率
                return np.array([[0.5, 0.5]])
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        if not self.is_fitted:
            print("❌ 模型尚未训练")
            return None
        
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 模型评估结果:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   预测报告:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, path):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes_,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        """加载模型"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.classes_ = model_data['classes']
        self.is_fitted = model_data['is_fitted']
        self.training_history = model_data['training_history']
        print(f"✅ 模型已加载: {path}")
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'classes': self.classes_,
            'training_samples': len(self.training_history),
            'coef_shape': self.model.coef_.shape if self.is_fitted else None
        }
        return info

def demo_online_learning():
    """演示在线学习过程"""
    
    print("🚀 SGD在线学习演示")
    print("="*60)
    
    # 1. 加载数据
    data_file = './Quick30/labeled_eeg_data2.npz'
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    data = np.load(data_file)
    X = data['X']
    y = data['y']
    
    # 如果标签是多维的，取第一列
    if len(y.shape) > 1:
        y = y[:, 0]
    
    print(f"📊 数据形状: {X.shape}")
    print(f"🏷️ 标签分布: {np.unique(y, return_counts=True)}")
    
    # 2. 创建在线学习器
    learner = SGDOnlineLearner("EEG_SGD_Online_Learner")
    
    # 3. 预训练模型（使用前80%的数据）
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    print(f"\n📈 预训练阶段:")
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    pretrain_accuracy = learner.pretrain(X_train, y_train)
    
    # 4. 评估预训练模型
    print(f"\n📊 预训练模型评估:")
    learner.evaluate(X_test, y_test)
    
    # 5. 在线学习演示
    print(f"\n🔄 在线学习阶段:")
    print("="*40)
    
    # 将剩余数据分成小批次进行在线学习
    batch_size = 2  # 每次学习2个样本
    online_accuracies = []
    
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
        
        # 在线学习
        learner.online_learn(X_batch, y_batch)
        
        # 评估当前模型
        current_accuracy = learner.evaluate(X_test, y_test)
        online_accuracies.append(current_accuracy)
        
        print(f"   批次 {i//batch_size + 1}: 添加了 {len(X_batch)} 个样本")
        print(f"   当前准确率: {current_accuracy:.4f}")
        print("-" * 30)
    
    # 6. 可视化结果
    print(f"\n📈 生成可视化结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SGD在线学习结果', fontsize=16)
    
    # 1. 在线学习准确率变化
    batch_numbers = range(1, len(online_accuracies) + 1)
    axes[0, 0].plot(batch_numbers, online_accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=pretrain_accuracy, color='r', linestyle='--', 
                        label=f'预训练准确率: {pretrain_accuracy:.3f}')
    axes[0, 0].set_title('在线学习准确率变化')
    axes[0, 0].set_xlabel('批次')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 模型权重分布
    if learner.is_fitted:
        coef = learner.model.coef_[0]
        axes[0, 1].hist(coef, bins=30, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('模型权重分布')
        axes[0, 1].set_xlabel('权重值')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 学习历史
    if learner.training_history:
        timestamps = [h['timestamp'] for h in learner.training_history]
        samples_added = [h['samples_added'] for h in learner.training_history]
        
        axes[1, 0].plot(range(1, len(timestamps) + 1), samples_added, 'go-')
        axes[1, 0].set_title('在线学习历史')
        axes[1, 0].set_xlabel('学习次数')
        axes[1, 0].set_ylabel('添加样本数')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 模型信息
    model_info = learner.get_model_info()
    info_text = f"""
模型信息:
- 模型名称: {model_info['model_name']}
- 是否已训练: {model_info['is_fitted']}
- 类别数: {len(model_info['classes']) if model_info['classes'] is not None else 'N/A'}
- 训练样本数: {model_info['training_samples']}
- 权重形状: {model_info['coef_shape']}
- 预训练准确率: {pretrain_accuracy:.4f}
- 最终准确率: {online_accuracies[-1] if online_accuracies else 'N/A':.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                     fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    axes[1, 1].set_title('模型信息')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('./Quick30/sgd_online_learning_results.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: ./Quick30/sgd_online_learning_results.png")
    
    # 7. 保存最终模型
    final_model_path = './Quick30/sgd_final_online_model.pkl'
    learner.save_model(final_model_path)
    print(f"✅ 最终模型已保存: {final_model_path}")
    
    # 8. 使用示例
    print(f"\n💡 模型使用示例:")
    print(f"```python")
    print(f"# 加载模型")
    print(f"learner = SGDOnlineLearner()")
    print(f"learner.load_model('{final_model_path}')")
    print(f"")
    print(f"# 预测新数据")
    print(f"new_features = np.random.randn(1, {X.shape[1]})")
    print(f"prediction = learner.predict(new_features)")
    print(f"print(f'预测结果: {{prediction}}')")
    print(f"")
    print(f"# 在线学习新数据")
    print(f"new_X = np.random.randn(2, {X.shape[1]})")
    print(f"new_y = np.array([1, 0])")
    print(f"learner.online_learn(new_X, new_y)")
    print(f"```")
    
    print("\n" + "="*60)
    print("✅ SGD在线学习演示完成!")
    print("="*60)

if __name__ == "__main__":
    demo_online_learning() 