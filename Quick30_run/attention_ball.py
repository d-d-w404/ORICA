from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPointF, QPoint, QRect
from PyQt5.QtGui import QColor, QPainter, QFont
import numpy as np
from PyQt5.QtCore import QTimer, QRect, QPoint, QPointF, Qt

class AttentionBallWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 Attention Ball View")
        self.resize(800, 600)
        self.ball_pos = QPoint(400, 300)
        self.ball_radius = 30
        self.color = QColor("gray")
        self.score = 0.0
        self.velocity = QPointF(0, 0)

        # 初始化表达式与显示控制
        self.current_expression = ""
        self.current_result = ""
        self.showing_result = False

        # 小球漂移计时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.move_ball)
        self.timer.start(16)



    def show_result(self):
        self.showing_result = True
        self.update()

    def move_ball(self):
        jitter = 0.5*(0.1+abs(max(0,(1.0 - self.score))))
        ax = np.random.uniform(-jitter, jitter) * 5
        ay = np.random.uniform(-jitter, jitter) * 5
        self.velocity += QPointF(ax, ay)
        max_speed = 12.0
        self.velocity.setX(np.clip(self.velocity.x(), -max_speed, max_speed))
        self.velocity.setY(np.clip(self.velocity.y(), -max_speed, max_speed))
        self.ball_pos += QPoint(int(self.velocity.x()), int(self.velocity.y()))
        margin = self.ball_radius
        self.ball_pos.setX(np.clip(self.ball_pos.x(), margin, self.width() - margin))
        self.ball_pos.setY(np.clip(self.ball_pos.y(), margin, self.height() - margin))
        self.velocity *= 0.92
        self.update()

    def update_attention(self, score):
        self.score = float(score)
        self.ball_radius = int(40 + 50 * self.score)

        # 🎨 将注意力分数映射到红→黄→绿的渐变
        # 0.0 → 红 (255, 0, 0)
        # 0.5 → 黄 (255, 255, 0)
        # 1.0 → 绿 (0, 255, 0)
        if self.score <= 0.5:
            # 红 → 黄 线性插值
            r = 255
            g = int(255 * (self.score / 0.5))  # 0→255
            b = 0
        else:
            # 黄 → 绿 线性插值
            r = int(255 * (1 - (self.score - 0.5) / 0.5))  # 255→0
            g = 255
            b = 0

        self.color = QColor(r, g, b)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.ball_pos, self.ball_radius, self.ball_radius)

        # ✏️ 显示注意力得分（保留两位小数）
        painter.setPen(Qt.black)
        font = QFont("Arial", 14)
        font.setBold(True)
        painter.setFont(font)

        text = f"{self.score:.2f}"
        text_rect = QRect(self.ball_pos.x() - self.ball_radius,
                          self.ball_pos.y() - 10,
                          self.ball_radius * 2,
                          20)
        painter.drawText(text_rect, Qt.AlignCenter, text)