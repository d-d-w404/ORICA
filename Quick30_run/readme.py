"""

ORICA/
├── main_gui.py                        # 主程序入口，GUI 启动逻辑（EEGGUI类）
├── stream_receiver.py                # LSL 数据接收与通道管理（LSLStreamReceiver, ChannelManager）
├── viewer.py                         # 实时波形绘图（LSLStreamVisualizer）
├── filter_utils.py                   # EEG 预处理工具类（EEGSignalProcessor）
├── regression_model.py               # 实时情绪回归模型（RealTimeRegressor）
├── attention_estimator.py            # 注意力得分提取类（RealTimeAttentionEstimator）
├── attention_ball.py                 # 注意力动态小球可视化（AttentionBallWindow）
├── bandpower_plot.py                 # 实时频段能量图（BandpowerStreamVisualizer）
├── bandpower_analysis.py             # `analyze_bandpower` 和其他分析函数
├── channel_selector.py               # 通道选择弹窗（ChannelSelectorDialog）
├── ORICA/
│   ├── __init__.py
│   └── orica_algorithm.py            # ORICA 算法类定义（如需要）
├── assets/                           # 可选：图标、配置、预定义布局等
└── .venv/                            # 虚拟环境（不要提交到 Git）



"""