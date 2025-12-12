from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HistogramWindow(QMainWindow):
    """
    Окно для отображения гистограммы.
    """
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Гистограмма")
        self.setGeometry(200, 200, 500, 400)

        self.canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.ax = self.canvas.figure.subplots()
        self.ax.hist(data, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
        self.ax.set_xlabel("Яркость")
        self.ax.set_ylabel("Частота")
        self.ax.set_title("Гистограмма яркости")

        self.save_button = QPushButton("Сохранить гистограмму")
        self.save_button.clicked.connect(self.save_histogram)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas)
        layout.addWidget(self.save_button)

    def save_histogram(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить гистограмму", "", "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)")
        if file_path:
            self.canvas.figure.savefig(file_path)
