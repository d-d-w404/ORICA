import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # 明确指定后端
import matplotlib.pyplot as plt
import mne
from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QWidget, QLabel, QSpinBox, 
                             QHBoxLayout, QMainWindow, QCheckBox, QScrollArea, QFrame)
from PyQt5.QtCore import pyqtSignal, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.Qt import Qt
from PyQt5.QtGui import QImage, QPixmap
from io import BytesIO
from PIL import Image
from topomap_analyzer import TopomapDataWorker


class TopomapWorker(QThread):
    image_ready = pyqtSignal(QImage)

    def __init__(self, receiver, get_topomap_func, interval_sec=2, n_components=5):
        super().__init__()
        self.receiver = receiver
        self.get_topomap_func = get_topomap_func
        self.interval_sec = interval_sec
        self.n_components = n_components
        self.running = True

    def run(self):
        import time
        while self.running:
            try:
                fig = self.get_topomap_func(self.receiver, self.n_components)
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf).convert("RGBA")
                qimg = QImage(img.tobytes("raw", "RGBA"), img.width, img.height, QImage.Format_RGBA8888)
                self.image_ready.emit(qimg)
                plt.close(fig)
            except Exception as e:
                print(f"❌ Topomap线程绘图失败: {e}")
            time.sleep(self.interval_sec)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class TopomapWindow(QMainWindow):
    """
    实时更新的Topomap窗口
    可以显示动态变化的ICA成分topomap
    """
    
    def __init__(self, receiver=None):
        super().__init__()
        self.receiver = receiver
        self.base_figsize = (3.5, 3)  # 每个topomap的基础大小
        self.columns_per_row = 4  # 每行4个topomap
        # 缓存变量
        self.cached_mixing_matrix = None
        self.cached_info = None
        self.cached_ch_names = None
        self.last_n_components = 0
        self.fig = None
        self.canvas = None
        # 防抖机制
        self.last_update_time = 0
        self.min_update_interval = 0.5  # 最小更新间隔（秒）
        self.worker = None
        self.image_label = None
        self.n_components_spinbox = None
        self.update_interval_spinbox = None
        self.status_label = None
        self.auto_update_checkbox = None
        self.last_data = None  # 缓存上一次数据
        self.setup_ui()
        self.start_worker()

    def setup_ui(self):
        self.setWindowTitle("Real time ICA Topomap")
        self.setGeometry(200, 200, 1000, 800)
        central_widget = QWidget()
        layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("ICA_num"))
        self.n_components_spinbox = QSpinBox()
        self.n_components_spinbox.setRange(1, 20)
        self.n_components_spinbox.setValue(12)
        self.n_components_spinbox.setToolTip("select IC number")
        self.n_components_spinbox.valueChanged.connect(self.restart_worker)
        control_layout.addWidget(self.n_components_spinbox)
        control_layout.addWidget(QLabel("update_freq:"))
        self.update_interval_spinbox = QSpinBox()
        self.update_interval_spinbox.setRange(1, 10)
        self.update_interval_spinbox.setValue(5)
        self.update_interval_spinbox.setToolTip("set topomap update_freq")
        self.update_interval_spinbox.valueChanged.connect(self.restart_worker)
        control_layout.addWidget(self.update_interval_spinbox)
        self.auto_update_checkbox = QCheckBox("auto update")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.setToolTip("start/stop auto update")
        self.auto_update_checkbox.toggled.connect(self.toggle_auto_update)
        control_layout.addWidget(self.auto_update_checkbox)
        layout.addLayout(control_layout)
        self.status_label = QLabel("ready")
        self.status_label.setStyleSheet("color: gray; margin: 5px;")
        layout.addWidget(self.status_label)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def make_topomap_figure(self, receiver, n_components_to_plot):
        """根据成分数量创建topomap图形"""
        import mne
        import numpy as np
        fig = plt.figure(figsize=(12, 8))
        try:
            if receiver is None or receiver.orica is None or receiver.orica.ica is None or receiver.channel_manager is None:
                return fig
            
            mixing_matrix = np.linalg.pinv(receiver.orica.ica.get_W())

            #对W排序按照能量大小排序。
            # W_sorted = receiver.orica.sorted_W
            # mixing_matrix = np.linalg.pinv(W_sorted)

            ch_names = receiver.channel_manager.get_labels_by_indices(receiver.channel_range)
            info = mne.create_info(ch_names=ch_names, sfreq=receiver.srate, ch_types='eeg')
            info.set_montage('standard_1020')
            n_components = mixing_matrix.shape[1]
            n_to_plot = min(n_components_to_plot, n_components)
            if n_to_plot <= 0:
                return fig
            if n_to_plot <= 2:
                cols = n_to_plot
            else:
                cols = self.columns_per_row
            rows = (n_to_plot + cols - 1) // cols
            axes = []
            for i in range(n_to_plot):
                ax = fig.add_subplot(rows, cols, i + 1)
                axes.append(ax)
            for i in range(n_to_plot):
                try:
                    mne.viz.plot_topomap(
                        mixing_matrix[:, i],
                        pos=info,
                        axes=axes[i],
                        show=False,
                        cmap='RdBu_r',
                        names=ch_names
                    )
                    axes[i].set_title(f'IC {i}', fontsize=12)
                    if (hasattr(receiver, 'latest_eog_indices') and 
                        receiver.latest_eog_indices is not None and 
                        i in receiver.latest_eog_indices):
                        axes[i].set_title(f'IC {i} (EOG)', color='red', fontsize=12)
                except Exception as e:
                    print(f"❌ Topomap绘制错误: {e}")
                    continue
            if cols == 1:
                wspace = 0.0
            elif cols == 2:
                wspace = 0.1
            else:
                wspace = 0.3
            if rows == 1:
                hspace = 0.1
            elif rows == 2:
                hspace = 0.2
            else:
                hspace = 0.4
            fig.subplots_adjust(
                left=0.05,
                right=0.95,
                top=0.95,
                bottom=0.05,
                wspace=wspace,
                hspace=hspace
            )
        except Exception as e:
            print(f"❌ 线程make_topomap_figure错误: {e}")
        return fig

    def start_worker(self):
        if self.worker is not None:
            self.worker.stop()
        interval = self.update_interval_spinbox.value() if self.update_interval_spinbox is not None else 2
        n_components = self.n_components_spinbox.value() if self.n_components_spinbox is not None else 5
        self.worker = TopomapDataWorker(self.receiver, interval_sec=interval, n_components=n_components)
        self.worker.data_ready.connect(self.update_topomap)
        if self.auto_update_checkbox is not None and self.auto_update_checkbox.isChecked():
            self.worker.start()

    def restart_worker(self):
        self.start_worker()

    def update_topomap(self, mixing_matrix, ch_names, info, n_to_plot, eog_indices, sorted_idx, ecd_locs, power_spectra, freqs, rv_list):
        import matplotlib.pyplot as plt
        import mne
        from io import BytesIO
        from PIL import Image
        from PyQt5.QtGui import QImage, QPixmap
        
        # sorted_idx通过rv_list参数传递
        sorted_idx = sorted_idx if sorted_idx is not None else list(range(mixing_matrix.shape[1]))
        
        if n_to_plot <= 2:
            cols = n_to_plot
        else:
            cols = self.columns_per_row
        rows = (n_to_plot + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        
        if n_to_plot == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.flatten()
            
        for i in range(n_to_plot):
            try:
                mne.viz.plot_topomap(
                    mixing_matrix[:, i],
                    pos=info,
                    axes=axes[i],
                    show=False,
                    cmap='RdBu_r',
                    names=ch_names
                )
                # 只有当排序前的IC编号在eog_indices中时才标红
                if (eog_indices is not None and sorted_idx[i] in eog_indices):
                    axes[i].set_title(f'IC {i} (EOG)', color='red', fontsize=10)
                else:
                    axes[i].set_title(f'IC {i}', fontsize=10)
            except Exception as e:
                axes[i].set_title(f'Topomap Error')
                print(f"❌ Topomap绘制错误: {e}")
        
        for i in range(n_to_plot, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        qimg = QImage(img.tobytes("raw", "RGBA"), img.width, img.height, QImage.Format_RGBA8888)
        if self.image_label is not None:
            self.image_label.setPixmap(QPixmap.fromImage(qimg))
        else:
            print("❌ image_label为None")
        if self.status_label is not None:
            self.status_label.setText(f"Topomap updated ({n_to_plot} components)")
        plt.close(fig)

    def toggle_auto_update(self, checked):
        """切换自动更新"""
        try:
            if checked:
                if self.worker is not None and not self.worker.isRunning():
                    self.worker.start()
                if self.status_label is not None:
                    self.status_label.setText("✅ 自动更新已启用")
            else:
                if self.worker is not None:
                    self.worker.stop()
                if self.status_label is not None:
                    self.status_label.setText("⏸️ 自动更新已暂停")
        except Exception as e:
            print(f"❌ 切换自动更新失败: {e}")
            
    def set_receiver(self, receiver):
        """设置数据接收器"""
        self.receiver = receiver
        # 清除缓存，强制重新计算
        self.cached_mixing_matrix = None
        self.cached_info = None
        self.cached_ch_names = None
        self.last_n_components = 0  # 强制重新创建图形
        
        # 设置receiver后立即更新一次
        self.restart_worker()
        
        # 如果自动更新被勾选，启动线程
        if self.auto_update_checkbox is not None and self.auto_update_checkbox.isChecked():
            if self.worker is not None:
                self.worker.start()
        
    def closeEvent(self, event):
        """窗口关闭时停止定时器"""
        if self.worker is not None:
            self.worker.stop()
        event.accept()


class TopomapButton(QPushButton):
    """
    简化的Topomap按钮组件
    点击后打开实时更新的topomap窗口
    """
    
    def __init__(self, receiver=None, parent=None):
        super().__init__("Show Topomap", parent)
        self.receiver = receiver
        self.topomap_window = None
        self.clicked.connect(self.show_topomap_window)
    
    def set_receiver(self, receiver):
        """设置数据接收器"""
        self.receiver = receiver
        if self.topomap_window is not None:
            self.topomap_window.set_receiver(receiver)
        
    def show_topomap_window(self):
        """显示实时更新的topomap窗口"""
        try:
            if self.receiver is None:
                print("❌ 数据接收器未设置")
                return
                
            # 如果窗口已经存在，就显示它
            if self.topomap_window is not None:
                self.topomap_window.show()
                self.topomap_window.raise_()
                return
                
            # 创建新的topomap窗口
            self.topomap_window = TopomapWindow(self.receiver)
            self.topomap_window.show()
            
        except Exception as e:
            print(f"❌ 打开Topomap窗口失败: {e}") 