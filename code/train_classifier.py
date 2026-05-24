import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 读取数据
DATA_PATH = 'labeled_eeg_data.npz'
data = np.load(DATA_PATH)
X = data['X']
y = data['y']

# 2. 选择标签（如只用arousal，或多标签都用）
# 假设y为[N, 2]，可选y[:, 0]（arousal）或y[:, 1]（valence）
if y.ndim == 2 and y.shape[1] > 1:
    print("标签为多维，默认只用第一个标签（arousal）进行分类")
    y = y[:, 0]
else:
    y = y.ravel()

# 3. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. 特征归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 训练SGDClassifier（支持增量学习）
clf = SGDClassifier(max_iter=1000, tol=1e-3)
# 需要先用partial_fit初始化类别
classes = np.unique(y_train)
clf.partial_fit(X_train, y_train, classes=classes)

# 6. 评估
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {acc:.3f}")
print("分类报告：")
print(classification_report(y_test, y_pred))

# 7. 保存模型和scaler（可选，便于后续增量学习）
import joblib
joblib.dump(clf, 'sgd_classifier.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("模型和scaler已保存，可用于后续增量学习。") 