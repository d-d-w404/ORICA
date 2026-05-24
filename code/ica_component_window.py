# ica_component_window.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class ICAComponentWindow(QWidget):
    def __init__(self, ica_sources=None, max_components=10):
        super().__init__()
        self.setWindowTitle("ICA Components")
        self.setGeometry(200, 200, 800, 600)

        self.sources = ica_sources
        self.max_components = max_components

        self.canvas = FigureCanvas(plt.Figure(figsize=(10, 6)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        if self.sources is not None:
            self.plot_components()

    def plot_components(self):
        self.canvas.figure.clear()
        n_plot = min(self.sources.shape[0], self.max_components)
        for i in range(n_plot):
            ax = self.canvas.figure.add_subplot(n_plot, 1, i + 1)
            ax.plot(self.sources[i])
            ax.set_ylabel(f"IC {i}")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def update_sources(self, ica_sources):
        self.sources = ica_sources
        self.plot_components()

    def set_eog_indices(self, indices):
        self.eog_indices = indices

    def plot_components(self):
        self.canvas.figure.clear()
        n_plot = min(self.sources.shape[0], self.max_components)
        for i in range(n_plot):
            ax = self.canvas.figure.add_subplot(n_plot, 1, i + 1)
            color = 'red' if hasattr(self, 'eog_indices') and i in self.eog_indices else 'blue'
            ax.plot(self.sources[i], color=color)
            ax.set_ylabel(f"IC {i}")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

