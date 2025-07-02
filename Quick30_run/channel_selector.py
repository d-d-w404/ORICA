from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton, QScrollArea, QWidget

class ChannelSelectorDialog(QDialog):
    def __init__(self, parent_gui, receiver):
        super().__init__()
        self.setWindowTitle("Select EEG Channels")
        self.receiver = receiver
        self.parent_gui = parent_gui
        self.checkboxes = []

        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 从 ChannelManager 获取全部通道
        self.channel_info = self.receiver.channel_manager.channels

        # 当前选中的 index 列表
        current_range = set(self.receiver.channel_range)

        for ch in self.channel_info:
            cb = QCheckBox(ch["label"])
            # 勾选当前在 range 中的通道
            cb.setChecked(ch["index"] in current_range)
            self.checkboxes.append(cb)
            scroll_layout.addWidget(cb)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.clicked.connect(self.apply_selection)
        layout.addWidget(confirm_btn)

        self.setLayout(layout)
        self.resize(300, 400)

    def apply_selection(self):
        selected_indices = []
        selected_labels = []

        for cb, ch in zip(self.checkboxes, self.channel_info):
            if cb.isChecked():
                selected_indices.append(ch["index"])
                selected_labels.append(ch["label"])

        # 更新 receiver 的通道范围和标签
        #self.receiver.channel_range = (
        self.receiver.set_channel_range_and_labels(selected_indices,selected_labels)



        print(f"✅ Selected channel indices: {selected_indices}")
        print(f"✅ Selected channel labels: {selected_labels}")
        self.accept()