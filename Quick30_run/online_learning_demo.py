import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def online_learning_demo():
    """在线学习模型演示"""
    
    print("🚀 在线学习模型演示")
    print("="*60)
    
    # 加载数据
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
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建在线学习模型
    models = {
        'SGD Classifier': SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.001,
            random_state=42,
            max_iter=1  # 每次只迭代1次
        ),
        'Passive Aggressive': PassiveAggressiveClassifier(
            C=1.0,
            random_state=42,
            max_iter=1
        ),
        'Perceptron': Perceptron(
            penalty='l2',
            alpha=0.001,
            random_state=42,
            max_iter=1
        )
    }
    
    # 在线学习过程
    print("\n🔄 开始在线学习...")
    print("-" * 60)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📈 训练 {model_name}...")
        
        accuracies = []
        predictions = []
        
        # 逐个样本进行在线学习
        for i in range(len(X_scaled)):
            # 获取当前样本
            X_current = X_scaled[i:i+1]
            y_current = y[i:i+1]
            
            # 在线学习（使用partial_fit）
            if i == 0:
                # 第一次需要指定所有类别
                unique_classes = np.unique(y)
                model.partial_fit(X_current, y_current, classes=unique_classes)
            else:
                model.partial_fit(X_current, y_current)
            
            # 预测当前样本
            pred = model.predict(X_current)
            predictions.append(pred[0])
            
            # 计算累积准确率
            current_accuracy = accuracy_score(y[:i+1], predictions)
            accuracies.append(current_accuracy)
            
            if (i + 1) % 5 == 0:  # 每5个样本显示一次进度
                print(f"   样本 {i+1}/{len(X_scaled)}, 准确率: {current_accuracy:.3f}")
        
        results[model_name] = {
            'accuracies': accuracies,
            'predictions': predictions,
            'final_accuracy': accuracies[-1]
        }
        
        print(f"✅ {model_name} 最终准确率: {accuracies[-1]:.3f}")
    
    # 可视化结果
    print("\n📊 生成可视化结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('在线学习模型性能对比', fontsize=16)
    
    # 1. 准确率随时间变化
    for model_name, result in results.items():
        axes[0, 0].plot(range(1, len(result['accuracies'])+1), 
                        result['accuracies'], 
                        label=model_name, linewidth=2)
    
    axes[0, 0].set_title('在线学习准确率变化')
    axes[0, 0].set_xlabel('样本数')
    axes[0, 0].set_ylabel('累积准确率')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 最终准确率对比
    model_names = list(results.keys())
    final_accuracies = [results[name]['final_accuracy'] for name in model_names]
    
    bars = axes[0, 1].bar(model_names, final_accuracies, 
                           color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('最终准确率对比')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_ylim(0, 1)
    
    # 在柱状图上添加数值
    for bar, acc in zip(bars, final_accuracies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
    
    # 3. 学习曲线（前20个样本）
    for model_name, result in results.items():
        early_accuracies = result['accuracies'][:20]
        axes[1, 0].plot(range(1, len(early_accuracies)+1), 
                        early_accuracies, 
                        label=model_name, marker='o')
    
    axes[1, 0].set_title('早期学习曲线（前20个样本）')
    axes[1, 0].set_xlabel('样本数')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 模型性能总结
    summary_data = []
    for model_name, result in results.items():
        summary_data.append([
            model_name,
            f"{result['final_accuracy']:.3f}",
            f"{np.mean(result['accuracies']):.3f}",
            f"{np.std(result['accuracies']):.3f}"
        ])
    
    summary_df = pd.DataFrame(summary_data, 
                             columns=['模型', '最终准确率', '平均准确率', '准确率标准差'])
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary_df.values,
                             colLabels=summary_df.columns,
                             cellLoc='center',
                             loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('模型性能总结')
    
    plt.tight_layout()
    plt.savefig('./Quick30/online_learning_results.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: ./Quick30/online_learning_results.png")
    
    # 保存最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['final_accuracy'])
    best_model = models[best_model_name]
    
    import joblib
    model_save_path = f'./Quick30/best_online_model_{best_model_name.replace(" ", "_").lower()}.pkl'
    joblib.dump(best_model, model_save_path)
    print(f"✅ 最佳模型已保存: {model_save_path}")
    
    # 打印详细结果
    print("\n" + "="*60)
    print("📊 详细结果:")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n🔍 {model_name}:")
        print(f"   最终准确率: {result['final_accuracy']:.3f}")
        print(f"   平均准确率: {np.mean(result['accuracies']):.3f}")
        print(f"   准确率标准差: {np.std(result['accuracies']):.3f}")
        print(f"   准确率范围: [{np.min(result['accuracies']):.3f}, {np.max(result['accuracies']):.3f}]")
    
    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"   最终准确率: {results[best_model_name]['final_accuracy']:.3f}")
    
    print("\n" + "="*60)
    print("✅ 在线学习演示完成!")
    print("="*60)

if __name__ == "__main__":
    online_learning_demo() 