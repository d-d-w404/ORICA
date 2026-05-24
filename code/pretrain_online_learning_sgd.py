import threading
import time
import os
import numpy as np
import signal
import sys
from scipy.signal import welch
from sgd_online_learning import SGDOnlineLearner

class PretrainOnlineLearning:
    """预训练和在线学习管理器"""
    
    def __init__(self, receiver=None, gui=None):
        self.receiver = receiver
        self.gui = gui
        self.learner = None
        self.is_running = False
        
        # 添加预测准确率统计
        self.prediction_history = []
        self.accuracy_history = []
        self.total_predictions = 0
        self.correct_predictions = 0
        
        # 自动标签生成相关
        self.current_target_label = None
        self.label_generation_time = None
        self.auto_label_mode = True  # Enable auto label mode by default
        self.label_classes = [7, 8]  # Label classes: 7=left hand, 8=right hand
        
        # 设置信号处理器，确保程序退出时保存数据
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器，确保程序正常退出时保存数据"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, saving data before exit...")
            self.stop_online_learning()
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        except:
            pass  # Windows可能不支持某些信号
        
    def start_online_learning(self):
        """启动在线学习"""
        # 启动新线程进行在线学习，避免阻塞主线程
        threading.Thread(target=self._online_learning_worker, daemon=True).start()

    def _online_learning_worker(self):
        """在线学习工作线程"""
        print("🚀 Starting online learning...")
        #self._update_gui("🔄 Initializing online learning...")
        
        try:
            # 检查预训练模型是否存在
            pretrained_model_path = './Quick30/sgd_pretrained_model.pkl'
            if not os.path.exists(pretrained_model_path):
                # 如果没有预训练模型，先训练一个
                print("📝 Pre-trained model not found, starting training...")
                #self._update_gui("📝 Training pre-trained model...")
                
                # 加载数据
                data_file = './Quick30/labeled_eeg_data_hand2.npz'
                if not os.path.exists(data_file):
                    #self._update_gui("❌ Data file does not exist, please collect data first")
                    print("❌ Data file does not exist: labeled_eeg_data2.npz")
                    return
                
                data = np.load(data_file)
                X = data['X']
                y = data['y']
                
                if len(y.shape) > 1:
                    y = y[:, 0]
                
                # 创建并训练模型
                self.learner = SGDOnlineLearner("EEG_Online_Learner")
                pretrain_accuracy = self.learner.pretrain(X, y, pretrained_model_path)
                
                print(f"✅ Pre-training completed, accuracy: {pretrain_accuracy:.4f}")
                #self._update_gui(f"✅ Pre-training completed, accuracy: {pretrain_accuracy:.4f}")
            
            # 加载预训练模型
            self.learner = SGDOnlineLearner()
            self.learner.load_model(pretrained_model_path)
            
            print("✅ Model loaded successfully, starting online learning...")
            #self._update_gui("✅ Model loaded successfully, starting online learning...")
            
            # 开始实时在线学习
            self._start_realtime_online_learning()
            
        except Exception as e:
            error_msg = f"❌ Online learning error: {str(e)}"
            print(error_msg)
            #self._update_gui(error_msg)

    def _start_realtime_online_learning(self):
        """开始实时在线学习"""
        print("🔄 Starting real-time online learning...")
        #self._update_gui("🔄 Starting real-time online learning...")
        
        # 在线学习参数
        LEARNING_INTERVAL = 8  # Learn every 5 seconds
        WINDOW_DURATION = 2   # 2 second window
        FS = 500
        
        # 自动标签生成参数
        PREPARATION_TIME = 4  # Preparation time after label display (seconds)
        ACTION_TIME = WINDOW_DURATION  # Action execution time (seconds)
        
        def extract_bandpower_features(data, fs=FS):
            bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
            features = []
            for ch in data:
                f, Pxx = welch(ch, fs=fs, nperseg=fs*2)
                for band in bands.values():
                    idx = np.logical_and(f >= band[0], f < band[1])
                    features.append(np.sum(Pxx[idx]))
            return np.array(features)
        
        learning_count = 0
        start_time = time.time()
        self.is_running = True
        
        # 生成第一个目标标签
        self._generate_new_target_label()
        
        while self.is_running:
            try:
                # 检查是否需要生成新的目标标签
                current_time = time.time()
                if (self.label_generation_time is None or 
                    current_time - self.label_generation_time >= LEARNING_INTERVAL):
                    # 生成新的目标标签
                    self._generate_new_target_label()
                
                # 获取当前EEG数据
                if self.receiver is None:
                    print("❌ Receiver not initialized")
                    time.sleep(1)
                    continue
                    
                buffer = self.receiver.get_buffer_data(data_type='processed')
                #buffer = self.receiver.get_buffer_data(data_type='raw')
                if buffer is not None and buffer.shape[1] >= FS * WINDOW_DURATION:
                    # 计算标签生成后的时间
                    time_since_label = current_time - self.label_generation_time
                    
                    # 只有在准备时间过后才开始收集数据
                    if time_since_label >= PREPARATION_TIME:
                        # 提取最新2秒数据
                        window = buffer[:, -FS*WINDOW_DURATION:]
                        
                        # 提取特征
                        features = extract_bandpower_features(window)
                        
                        # 根据模式选择标签
                        if self.auto_label_mode:
                            label = self.current_target_label
                        else:
                            label = self._get_user_label()
                        
                        # 预测当前数据
                        prediction = self.learner.predict(features.reshape(1, -1))[0]
                        proba = self.learner.predict_proba(features.reshape(1, -1))[0]
                        
                        # 在线学习
                        self.learner.online_learn(features.reshape(1, -1), np.array([label]))
                        learning_count += 1
                        
                        # 统计预测准确率
                        self._update_prediction_accuracy(prediction, label, proba)
                        
                        # 更新GUI显示
                        elapsed_time = time.time() - start_time
                        current_accuracy = self.correct_predictions / max(1, self.total_predictions)
                        
                        # 计算剩余时间
                        remaining_time = max(0, LEARNING_INTERVAL - time_since_label)
                        
                        result_text = f"""
🔄 {self.get_label_mode()} label mode running...
⏱️ Runtime: {elapsed_time:.1f}s
🎯 Current target: {'Left hand imagination' if label == 7 else 'Right hand imagination'}
⏳ Time remaining: {remaining_time:.1f}s
📊 Prediction count: {self.total_predictions}
🎯 Current prediction: {prediction} ({'Left hand' if prediction == 7 else 'Right hand'})
📈 Prediction probability: [{proba[0]:.3f}, {proba[1]:.3f}]
✅ Prediction result: {'Correct' if prediction == label else 'Incorrect'}
📊 Cumulative accuracy: {current_accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})
📝 Label source: {self.get_label_mode()}
                    """.strip()
                        
                        #self._update_gui(result_text)
                        
                        print(f"🎯 Prediction #{self.total_predictions}: Target={label}({'Left hand' if label == 7 else 'Right hand'}), Prediction={prediction}({'Left hand' if prediction == 7 else 'Right hand'}), Accuracy={current_accuracy:.3f}")
                        
                        # 等待到下一个学习周期
                        time.sleep(LEARNING_INTERVAL - time_since_label)
                    else:
                        # 还在准备阶段，显示倒计时
                        remaining_prep = PREPARATION_TIME - time_since_label
                        print(f"⏳ Preparation phase: {remaining_prep:.1f}s until data collection starts...")
                        time.sleep(0.5)
                else:
                    time.sleep(0.5)
                
            except Exception as e:
                error_msg = f"❌ Prediction error: {str(e)}"
                print(error_msg)
                #self._update_gui(error_msg)
                time.sleep(5)  # Wait 5 seconds after error before continuing

    def _generate_new_target_label(self):
        """生成新的目标标签"""
        import random
        
        # 随机选择一个标签类别
        self.current_target_label = random.choice(self.label_classes)
        self.label_generation_time = time.time()
        
        # 显示目标标签
        action_name = "Left hand imagination" if self.current_target_label == 7 else "Right hand imagination"
        print(f"\n🎯 New target generated: {action_name} (Label: {self.current_target_label})")
        print("⏳ Please prepare to perform the corresponding action...")
        
        # 更新GUI显示（如果有的话）
        if self.gui is not None:
            try:
                if hasattr(self.gui, 'target_label_display'):
                    self.gui.target_label_display.setText(f"Current target: {action_name}")
                if hasattr(self.gui, 'countdown_display'):
                    self.gui.countdown_display.setText("Preparing...")
            except Exception as e:
                print(f"❌ GUI update failed: {e}")

    def _get_user_label(self):
        """获取用户输入的标签（保留用于手动模式）"""
        if self.gui is None:
            return 7
            
        label_text = self.gui.online_label_input.text().strip()
        if label_text:
            try:
                label = int(label_text)
                print(f"📝 User input label: {label}")
                return label
            except ValueError:
                print("❌ Label format error, using default label 7")
                return 7
        else:
            # 如果没有输入，使用默认标签
            print("⚠️ No label input, using default label 7")
            return 7

    def _get_label_input(self):
        """获取标签输入文本"""
        if self.gui is None:
            return ""
        return self.gui.online_label_input.text().strip()

    def _update_gui(self, message):
        """更新GUI显示"""
        if self.gui is not None and hasattr(self.gui, 'online_result_label'):
            try:
                self.gui.online_result_label.setText(message)
            except Exception as e:
                print(f"❌ GUI update failed: {e}")

    def stop_online_learning(self):
        """停止在线学习"""
        self.is_running = False
        print("🛑 Stopping prediction mode")
        
        # 打印预测结果摘要
        self.print_prediction_summary()
        
        # 保存预测结果
        if self.total_predictions > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"./Results/prediction_results_{timestamp}.json"
            self.save_prediction_results(filename)

    def get_learner(self):
        """获取学习器实例"""
        return self.learner

    def save_model(self, path):
        """保存模型"""
        if self.learner is not None:
            self.learner.save_model(path)
            print(f"💾 Model saved: {path}")

    def load_model(self, path):
        """加载模型"""
        if self.learner is None:
            self.learner = SGDOnlineLearner()
        self.learner.load_model(path)
        print(f"✅ Model loaded: {path}")

    def predict(self, features):
        """预测"""
        if self.learner is not None:
            return self.learner.predict(features)
        return None

    def predict_proba(self, features):
        """预测概率"""
        if self.learner is not None:
            return self.learner.predict_proba(features)
        return None

    def online_learn(self, features, labels):
        """在线学习"""
        if self.learner is not None:
            return self.learner.online_learn(features, labels)
        return False

    def set_auto_label_mode(self, enabled=True):
        """设置自动标签模式"""
        self.auto_label_mode = enabled
        mode_str = "Auto label mode" if enabled else "Manual label mode"
        print(f"🔄 Switched to {mode_str}")

    def get_current_target_label(self):
        """获取当前目标标签"""
        return self.current_target_label

    def get_label_mode(self):
        """获取当前标签模式"""
        return "Auto" if self.auto_label_mode else "Manual"

    def get_current_status(self):
        """获取当前状态信息"""
        current_time = time.time()
        time_since_label = current_time - self.label_generation_time if self.label_generation_time else 0
        
        return {
            'is_running': self.is_running,
            'auto_label_mode': self.auto_label_mode,
            'current_target_label': self.current_target_label,
            'time_since_label': time_since_label,
            'total_predictions': self.total_predictions,
            'current_accuracy': self.correct_predictions / max(1, self.total_predictions)
        }

    def _update_prediction_accuracy(self, prediction, true_label, proba):
        """更新预测准确率统计"""
        self.total_predictions += 1
        
        # 检查预测是否正确
        is_correct = (prediction == true_label)
        if is_correct:
            self.correct_predictions += 1
        
        # 记录预测历史 - 确保所有数据都是JSON可序列化的
        prediction_record = {
            'timestamp': float(time.time()),
            'prediction': int(prediction),
            'true_label': int(true_label),
            'is_correct': bool(is_correct),
            'proba': [float(p) for p in proba.tolist()],
            'accuracy': float(self.correct_predictions / self.total_predictions)
        }
        self.prediction_history.append(prediction_record)
        self.accuracy_history.append(float(prediction_record['accuracy']))
        
        print(f"📊 Prediction statistics: Prediction={prediction}, Actual={true_label}, Result={'✅' if is_correct else '❌'}")
    
    def get_prediction_statistics(self):
        """获取预测统计信息"""
        if self.total_predictions == 0:
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'accuracy_history': [],
                'prediction_history': []
            }
        
        return {
            'total_predictions': int(self.total_predictions),
            'correct_predictions': int(self.correct_predictions),
            'accuracy': float(self.correct_predictions / self.total_predictions),
            'accuracy_history': [float(acc) for acc in self.accuracy_history],
            'prediction_history': self.prediction_history.copy()
        }
    
    def save_prediction_results(self, filepath='prediction_results.json'):
        """保存预测结果到文件"""
        import json
        from datetime import datetime
        import os
        
        # 确保Results目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_prediction_statistics(),
            'model_info': {
                'model_type': 'SGDOnlineLearner',
                'feature_type': 'bandpower',
                'feature_dim': 80
            }
        }
        
        try:
            # 先写入临时文件，确保完整性
            temp_filepath = filepath + '.tmp'
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 验证临时文件完整性
            with open(temp_filepath, 'r', encoding='utf-8') as f:
                json.load(f)  # 测试JSON是否完整
            
            # 如果验证通过，重命名为最终文件
            if os.path.exists(filepath):
                backup_filepath = filepath + '.backup'
                os.rename(filepath, backup_filepath)
                print(f"💾 Previous file backed up as: {backup_filepath}")
            
            os.rename(temp_filepath, filepath)
            print(f"💾 Prediction results saved: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save prediction results: {e}")
            # 清理临时文件
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            return False
    
    def print_prediction_summary(self):
        """打印预测结果摘要"""
        stats = self.get_prediction_statistics()
        
        print("\n" + "="*60)
        print("📊 Prediction Results Summary")
        print("="*60)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Correct predictions: {stats['correct_predictions']}")
        print(f"Final accuracy: {stats['accuracy']:.3f} ({stats['correct_predictions']}/{stats['total_predictions']})")
        
        if len(stats['accuracy_history']) > 1:
            print(f"Accuracy range: {min(stats['accuracy_history']):.3f} - {max(stats['accuracy_history']):.3f}")
        
        print("="*60)

# 独立运行测试
if __name__ == "__main__":
    print("🧪 测试预训练在线学习模块")
    
    # 创建实例
    learning_manager = PretrainOnlineLearning()
    
    # 测试预训练
    print("📝 测试预训练功能...")
    try:
        learning_manager.start_online_learning()
        print("✅ 在线学习启动成功")
    except Exception as e:
        print(f"❌ 在线学习启动失败: {e}") 