import threading
import time
import os
import numpy as np
import signal
import sys
from scipy.signal import welch
from eegnet_online_learning import EEGNetOnlineLearner

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
        self.label_classes = [0, 1]  # Label classes: 0=left hand, 1=right hand
        
        # 语音提示相关设置
        self.voice_enabled = True  # 是否启用语音提示
        self.voice_engine = None  # 语音引擎实例
        
        # 初始化语音引擎
        self._init_voice_engine()
        
        # 设置信号处理器，确保程序退出时保存数据
        self._setup_signal_handlers()
    
    def _init_voice_engine(self):
        """初始化语音引擎"""
        try:
            import pyttsx3
            self.voice_engine = pyttsx3.init()
            
            # 设置语音参数
            self.voice_engine.setProperty('rate', 150)  # 语速
            self.voice_engine.setProperty('volume', 0.8)  # 音量
            
            # 尝试设置中文语音（如果可用）
            voices = self.voice_engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.voice_engine.setProperty('voice', voice.id)
                    break
            
            print("✅ Voice engine initialized successfully")
            
        except ImportError:
            print("⚠️ pyttsx3 not installed. Install with: pip install pyttsx3")
            self.voice_engine = None
        except Exception as e:
            print(f"❌ Voice engine initialization failed: {e}")
            self.voice_engine = None
    
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
            # 检查是否存在预训练模型
            pretrained_model_path = './Quick30/eegnet_pretrained_model_listen_processed_copy.pt'
            if os.path.exists(pretrained_model_path):
                print("📝 Loading existing pretrained EEGNet model...")
                #self._update_gui("📝 Loading pretrained model...")
                
                # 加载预训练模型
                self.learner = EEGNetOnlineLearner("EEGNet_Online_Learner")
                self.learner.load_model(pretrained_model_path)
                
                print("✅ Pretrained model loaded successfully, starting online learning...")
                #self._update_gui("✅ Pretrained model loaded successfully, starting online learning...")
            else:
                # 创建新的EEGNet模型
                print("📝 Creating new EEGNet model for online learning...")
                #self._update_gui("📝 Creating new EEGNet model...")
                
                # 创建EEGNet模型
                self.learner = EEGNetOnlineLearner("EEGNet_Online_Learner")
                
                print("✅ EEGNet model created successfully, starting online learning...")
                #self._update_gui("✅ Model created successfully, starting online learning...")
            
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
        LEARNING_INTERVAL =  8  # Learn every 5 seconds
        WINDOW_DURATION = 2   # 2 second window
        FS = 500
        
        # 自动标签生成参数
        PREPARATION_TIME = 6  # Preparation time after label display (seconds)
        ACTION_TIME = WINDOW_DURATION  # Action execution time (seconds)
        
        def extract_features(data, fs=FS):
            """提取特征 - 对于EEGNet使用原始时间序列数据"""
            # 对于EEGNet，直接返回原始时间序列数据
            # 数据形状: (n_channels, n_samples) -> (n_channels, time_length)
            return data
        
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
                    
                #buffer = self.receiver.get_buffer_data(data_type='processed')
                buffer = self.receiver.get_buffer_data(data_type='processed')
                if buffer is not None and buffer.shape[1] >= FS * WINDOW_DURATION:
                    # 计算标签生成后的时间
                    time_since_label = current_time - self.label_generation_time
                    
                    # 只有在准备时间过后才开始收集数据
                    if time_since_label >= PREPARATION_TIME:
                        # 提取最新2秒数据
                        window = buffer[:, -FS*WINDOW_DURATION:]
                        
                        # 提取特征
                        features = extract_features(window)
                        
                        # 根据模式选择标签
                        if self.auto_label_mode:
                            label = self.current_target_label
                        else:
                            label = self._get_user_label()
                        
                        # 直接使用预训练模型进行预测
                        prediction = self.learner.predict(features)[0]
                        print(self.learner.predict(features))
                        print(prediction)
                        proba = self.learner.predict_proba(features)[0]
                        print(self.learner.predict_proba(features))
                        
                        # 统计预测准确率
                        self._update_prediction_accuracy(prediction, label, proba)

                        #增量学习
                        self.learner.online_learn(features, np.array([label]))
                        
                        # 更新学习计数
                        learning_count += 1
                        
                        # 更新GUI显示
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, LEARNING_INTERVAL - time_since_label)
                        
                        # 显示完整信息
                        current_accuracy = self.correct_predictions / max(1, self.total_predictions)
                        result_text = f"""
🔄 {self.get_label_mode()} label mode running...
⏱️ Runtime: {elapsed_time:.1f}s
🎯 Current target: {'Left hand imagination' if label == 0 else 'Right hand imagination'}
⏳ Time remaining: {remaining_time:.1f}s
📊 Prediction count: {self.total_predictions}
🎯 Current prediction: {prediction} ({'Left hand' if prediction == 0 else 'Right hand'})
📈 Prediction probability: [{proba[0]:.3f}, {proba[1]:.3f}]
✅ Prediction result: {'Correct' if prediction == label else 'Incorrect'}
📊 Cumulative accuracy: {current_accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})
📝 Label source: {self.get_label_mode()}
                    """.strip()
                        
                        print(f"🎯 Prediction #{self.total_predictions}: Target={label}({'Left hand' if label == 0 else 'Right hand'}), Prediction={prediction}({'Left hand' if prediction == 0 else 'Right hand'}), Accuracy={current_accuracy:.3f}")
                        
                        #self._update_gui(result_text)
                        
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
        import threading
        
        # 随机选择一个标签类别
        self.current_target_label = random.choice(self.label_classes)
        self.label_generation_time = time.time()
        
        # 显示目标标签
        action_name = "Left hand imagination" if self.current_target_label == 0 else "Right hand imagination"
        print(f"\n🎯 New target generated: {action_name} (Label: {self.current_target_label})")
        print("⏳ Please prepare to perform the corresponding action...")
        
        # 播放语音提示
        self._play_voice_prompt(self.current_target_label)
        
        # 更新GUI显示（如果有的话）
        if self.gui is not None:
            try:
                if hasattr(self.gui, 'target_label_display'):
                    self.gui.target_label_display.setText(f"Current target: {action_name}")
                if hasattr(self.gui, 'countdown_display'):
                    self.gui.countdown_display.setText("Preparing...")
            except Exception as e:
                print(f"❌ GUI update failed: {e}")
    
    def _play_voice_prompt(self, label):
        """播放语音提示"""
        if not self.voice_enabled:
            return
            
        # 根据标签生成语音内容
        if label == 0:
            voice_text = "左"
        else:  # label == 1
            voice_text = "右"
        
        # 如果语音引擎可用，使用它
        if self.voice_engine is not None:
            def play_voice():
                try:
                    self.voice_engine.say(voice_text)
                    self.voice_engine.runAndWait()
                except Exception as e:
                    print(f"❌ Voice playback error: {e}")
                    # 如果语音引擎失败，使用备用方案
                    self._play_voice_fallback(label)
            
            # 启动语音播放线程
            voice_thread = threading.Thread(target=play_voice, daemon=True)
            voice_thread.start()
            
            print(f"🔊 Playing voice prompt: {voice_text}")
        else:
            # 使用备用方案
            self._play_voice_fallback(label)
    
    def _play_voice_fallback(self, label):
        """备用语音播放方案（使用系统命令）"""
        try:
            import subprocess
            import platform
            
            # 根据标签生成语音内容
            if label == 0:
                voice_text = "左"
            else:  # label == 1
                voice_text = "右"
            
            # 根据操作系统选择不同的语音命令
            system = platform.system()
            
            if system == "Windows":
                # Windows 使用 PowerShell 的 SpeechSynthesizer
                ps_script = f'Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("{voice_text}")'
                subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
            elif system == "Darwin":  # macOS
                # macOS 使用 say 命令
                subprocess.run(["say", voice_text], capture_output=True)
            elif system == "Linux":
                # Linux 使用 espeak 或 festival
                try:
                    subprocess.run(["espeak", voice_text], capture_output=True)
                except FileNotFoundError:
                    try:
                        subprocess.run(["festival", "--tts"], input=voice_text.encode(), capture_output=True)
                    except FileNotFoundError:
                        print("⚠️ No text-to-speech tool found on Linux")
            
            print(f"🔊 Playing voice prompt (fallback): {voice_text}")
            
        except Exception as e:
            print(f"❌ Fallback voice prompt error: {e}")
    
    def enable_voice_prompt(self, enabled=True):
        """启用或禁用语音提示"""
        self.voice_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🔊 Voice prompt {status}")
    
    def set_voice_rate(self, rate):
        """设置语音语速"""
        if self.voice_engine is not None:
            try:
                self.voice_engine.setProperty('rate', rate)
                print(f"🔊 Voice rate set to {rate}")
            except Exception as e:
                print(f"❌ Failed to set voice rate: {e}")
    
    def set_voice_volume(self, volume):
        """设置语音音量 (0.0 - 1.0)"""
        if self.voice_engine is not None:
            try:
                self.voice_engine.setProperty('volume', volume)
                print(f"🔊 Voice volume set to {volume}")
            except Exception as e:
                print(f"❌ Failed to set voice volume: {e}")
    
    def test_voice(self):
        """测试语音功能"""
        print("🔊 Testing voice prompt...")
        self._play_voice_prompt(0)  # 测试左手语音
        import time
        time.sleep(2)  # 等待2秒
        self._play_voice_prompt(1)  # 测试右手语音

    def _get_user_label(self):
        """获取用户输入的标签（保留用于手动模式）"""
        if self.gui is None:
            return 0
            
        label_text = self.gui.online_label_input.text().strip()
        if label_text:
            try:
                label = int(label_text)
                print(f"📝 User input label: {label}")
                return label
            except ValueError:
                print("❌ Label format error, using default label 0")
                return 0
        else:
            # 如果没有输入，使用默认标签
            print("⚠️ No label input, using default label 0")
            return 0

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
                'model_type': 'EEGNetOnlineLearner',
                'feature_type': 'raw_time_series',
                'feature_dim': 16000  # 16通道 × 1000时间点
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