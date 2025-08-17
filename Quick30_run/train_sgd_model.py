import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_sgd_eeg_model():
    """使用SGD训练EEG数据模型"""
    
    # 数据文件路径
    data_file = './Quick30/labeled_eeg_data2.npz'
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    print("🔄 加载数据...")
    try:
        data = np.load(data_file)
        X = data['X']
        y = data['y']
        print(f"✅ 数据加载成功")
        print(f"   特征形状: {X.shape}")
        print(f"   标签形状: {y.shape}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 数据预处理
    print("\n🔄 数据预处理...")
    
    # 如果标签是多维的，取第一列作为主要标签
    if len(y.shape) > 1:
        y = y[:, 0]
    
    print(f"   标签分布: {np.unique(y, return_counts=True)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   标准化后特征范围: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   训练集大小: {X_train.shape[0]} 样本")
    print(f"   测试集大小: {X_test.shape[0]} 样本")
    
    # 训练SGD模型
    print("\n🔄 训练SGD模型...")
    
    # 创建SGD分类器
    sgd_model = SGDClassifier(
        loss='hinge',  # 使用SVM损失函数
        penalty='l2',  # L2正则化
        alpha=0.001,   # 正则化强度
        max_iter=1000, # 最大迭代次数
        random_state=42,
        tol=1e-3
    )
    
    # 训练模型
    sgd_model.fit(X_train, y_train)
    
    # 预测
    y_pred = sgd_model.predict(X_test)
    y_pred_train = sgd_model.predict(X_train)
    
    # 评估模型
    print("\n📊 模型评估结果:")
    print("="*50)
    
    # 训练集准确率
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"训练集准确率: {train_accuracy:.4f}")
    
    # 测试集准确率
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 交叉验证
    cv_scores = cross_val_score(sgd_model, X_scaled, y, cv=5)
    print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 详细分类报告
    print(f"\n📋 详细分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 混淆矩阵:")
    print(cm)
    
    # 特征重要性
    feature_importance = np.abs(sgd_model.coef_[0])
    top_features = np.argsort(feature_importance)[-10:]  # 前10个重要特征
    
    print(f"\n🔍 前10个重要特征:")
    for i, feature_idx in enumerate(reversed(top_features)):
        print(f"   特征 {feature_idx}: {feature_importance[feature_idx]:.6f}")
    
    # 可视化结果
    print(f"\n📈 生成可视化图表...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SGD EEG模型训练结果', fontsize=16)
    
    # 1. 准确率对比
    axes[0, 0].bar(['训练集', '测试集'], [train_accuracy, test_accuracy], 
                    color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('模型准确率对比')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate([train_accuracy, test_accuracy]):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. 混淆矩阵热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('混淆矩阵')
    axes[0, 1].set_xlabel('预测标签')
    axes[0, 1].set_ylabel('真实标签')
    
    # 3. 特征重要性
    top_10_importance = feature_importance[top_features]
    feature_labels = [f'F{i}' for i in top_features]
    axes[1, 0].barh(range(len(top_10_importance)), top_10_importance)
    axes[1, 0].set_yticks(range(len(feature_labels)))
    axes[1, 0].set_yticklabels(feature_labels)
    axes[1, 0].set_title('前10个重要特征')
    axes[1, 0].set_xlabel('重要性得分')
    
    # 4. 交叉验证结果
    axes[1, 1].plot(range(1, len(cv_scores)+1), cv_scores, 'bo-')
    axes[1, 1].axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                        label=f'平均: {cv_scores.mean():.3f}')
    axes[1, 1].set_title('交叉验证结果')
    axes[1, 1].set_xlabel('折数')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./Quick30/sgd_model_results.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: ./Quick30/sgd_model_results.png")
    
    # 保存模型
    import joblib
    model_save_path = './Quick30/sgd_eeg_model.pkl'
    scaler_save_path = './Quick30/sgd_scaler.pkl'
    
    joblib.dump(sgd_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    print(f"✅ 模型已保存: {model_save_path}")
    print(f"✅ 标准化器已保存: {scaler_save_path}")
    
    # 模型使用示例
    print(f"\n💡 模型使用示例:")
    print(f"```python")
    print(f"import joblib")
    print(f"import numpy as np")
    print(f"")
    print(f"# 加载模型")
    print(f"model = joblib.load('{model_save_path}')")
    print(f"scaler = joblib.load('{scaler_save_path}')")
    print(f"")
    print(f"# 预测新数据")
    print(f"new_features = np.random.randn(1, {X.shape[1]})  # 新特征")
    print(f"new_features_scaled = scaler.transform(new_features)")
    print(f"prediction = model.predict(new_features_scaled)")
    print(f"print(f'预测结果: {{prediction}}')")
    print(f"```")
    
    print("\n" + "="*50)
    print("✅ SGD模型训练完成!")
    print("="*50)

if __name__ == "__main__":
    train_sgd_eeg_model() 