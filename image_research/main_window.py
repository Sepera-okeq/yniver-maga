from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QLabel, QInputDialog, QToolBar
from PyQt6.QtGui import QAction, QImage, QIcon, QPainterPath, QActionGroup
from PyQt6.QtCore import QSize, QRectF
from image_viewer import ImageViewer
import image_processing
from histogram_window import HistogramWindow

class MainWindow(QMainWindow):
    """
    Главное окно приложения.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Research")
        self.setGeometry(100, 100, 800, 600)

        self.image_viewer = ImageViewer(self)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.image_viewer)

        self._create_menus()
        self._create_toolbars()
        self._create_status_bar()

    def _create_toolbars(self):
        """
        Создание панелей инструментов.
        """
        toolbar = QToolBar("Инструменты")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        rect_selection_action = QAction(QIcon(), "Прямоугольное выделение", self)
        rect_selection_action.setCheckable(True)
        rect_selection_action.setChecked(True)
        rect_selection_action.triggered.connect(lambda: self.set_selection_mode('rect'))
        toolbar.addAction(rect_selection_action)

        lasso_selection_action = QAction(QIcon(), "Лассо", self)
        lasso_selection_action.setCheckable(True)
        lasso_selection_action.triggered.connect(lambda: self.set_selection_mode('lasso'))
        toolbar.addAction(lasso_selection_action)

        # Группировка кнопок, чтобы только одна могла быть активна
        selection_group = QActionGroup(self)
        selection_group.addAction(rect_selection_action)
        selection_group.addAction(lasso_selection_action)
        selection_group.setExclusive(True)

    def set_selection_mode(self, mode):
        """
        Устанавливает режим выделения в ImageViewer.
        """
        self.image_viewer.set_selection_mode(mode)

    def _create_status_bar(self):
        """
        Создание строки состояния.
        """
        self.status_bar = self.statusBar()
        self.pixel_info_label = QLabel()
        self.status_bar.addPermanentWidget(self.pixel_info_label)

    def _create_menus(self):
        """
        Создание меню.
        """
        # Меню "Файл"
        file_menu = self.menuBar().addMenu("&Файл")

        open_action = QAction("&Открыть", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("&Сохранить", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("&Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Меню "Правка"
        edit_menu = self.menuBar().addMenu("&Правка")
        self.undo_action = QAction("&Отменить", self)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)

        # Меню "Обработка"
        processing_menu = self.menuBar().addMenu("&Обработка")
        grayscale_action = QAction("&Градации серого", self)
        grayscale_action.triggered.connect(self.convert_to_grayscale)
        processing_menu.addAction(grayscale_action)

        stats_action = QAction("&Статистика", self)
        stats_action.triggered.connect(self.show_statistics)
        processing_menu.addAction(stats_action)

        histogram_action = QAction("&Гистограмма", self)
        histogram_action.triggered.connect(self.show_histogram)
        processing_menu.addAction(histogram_action)

        contrast_action = QAction("&Линейное контрастирование", self)
        contrast_action.triggered.connect(self.apply_linear_contrast)
        processing_menu.addAction(contrast_action)

        smooth_action = QAction("&Сглаживание", self)
        smooth_action.triggered.connect(self.apply_smoothing)
        processing_menu.addAction(smooth_action)

        posterize_action = QAction("Постеризация (Карты)", self)
        posterize_action.triggered.connect(self.apply_posterization)
        processing_menu.addAction(posterize_action)

        # Меню "Масштабирование"
        scale_menu = self.menuBar().addMenu("&Масштаб")
        
        scale_nearest_action = QAction("Выборкой", self)
        scale_nearest_action.triggered.connect(lambda: self.scale_image("nearest"))
        scale_menu.addAction(scale_nearest_action)

        scale_bilinear_action = QAction("Интерполяцией", self)
        scale_bilinear_action.triggered.connect(lambda: self.scale_image("bilinear"))
        scale_menu.addAction(scale_bilinear_action)

        # Меню "Шум"
        noise_menu = self.menuBar().addMenu("&Шум")
        
        add_noise_action = QAction("Добавить шум", self)
        add_noise_action.triggered.connect(self.add_noise)
        noise_menu.addAction(add_noise_action)

        reduce_noise_action = QAction("Уменьшить шум (медианный)", self)
        reduce_noise_action.triggered.connect(self.reduce_noise)
        noise_menu.addAction(reduce_noise_action)

        # Меню "Пиксель"
        pixel_menu = self.menuBar().addMenu("&Пиксель")
        
        get_pixel_action = QAction("Получить значение", self)
        get_pixel_action.triggered.connect(self.get_pixel)
        pixel_menu.addAction(get_pixel_action)

        set_pixel_action = QAction("Установить значение", self)
        set_pixel_action.triggered.connect(self.set_pixel)
        pixel_menu.addAction(set_pixel_action)

        # Меню "Генерация"
        generate_menu = self.menuBar().addMenu("&Генерация")
        
        gen_uniform_action = QAction("Равномерный шум", self)
        gen_uniform_action.triggered.connect(lambda: self.generate_image("uniform"))
        generate_menu.addAction(gen_uniform_action)

        gen_normal_action = QAction("Нормальный шум", self)
        gen_normal_action.triggered.connect(lambda: self.generate_image("normal"))
        generate_menu.addAction(gen_normal_action)

        gen_sliding_sum_action = QAction("Скользящее суммирование", self)
        gen_sliding_sum_action.triggered.connect(self.generate_sliding_sum)
        generate_menu.addAction(gen_sliding_sum_action)

        # Меню "Поворот"
        rotate_menu = self.menuBar().addMenu("&Поворот")
        rotate_action = QAction("Повернуть", self)
        rotate_action.triggered.connect(self.rotate_image)
        rotate_menu.addAction(rotate_action)

    def rotate_image(self):
        """
        Поворачивает изображение.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для обработки.")
            return

        angle, ok = QInputDialog.getDouble(self, "Поворот", "Введите угол:", 0.0, -360.0, 360.0, 2)
        if ok:
            new_image = image_processing.rotate_image(self.image_viewer.get_image(), angle)
            self.image_viewer.set_image(new_image)
            self.undo_action.setEnabled(True)

    def generate_sliding_sum(self):
        """
        Генерирует изображение однородной сцены методом скользящего суммирования.
        """
        width, ok_w = QInputDialog.getInt(self, "Ширина", "Ширина:", 512, 1, 4096, 1)
        if ok_w:
            height, ok_h = QInputDialog.getInt(self, "Высота", "Высота:", 512, 1, 4096, 1)
            if ok_h:
                radius, ok_r = QInputDialog.getInt(self, "Радиус", "Радиус:", 5, 1, 100, 1)
                if ok_r:
                    mean, ok_m = QInputDialog.getDouble(self, "Среднее", "Среднее:", 128.0, 0.0, 255.0, 2)
                    if ok_m:
                        std, ok_s = QInputDialog.getDouble(self, "Ст. отклонение", "Ст. отклонение:", 20.0, 0.1, 100.0, 2)
                        if ok_s:
                            new_image = image_processing.generate_sliding_sum_image(width, height, radius, mean, std)
                            self.image_viewer.set_image(new_image)
                            self.undo_action.setEnabled(True)

    def generate_image(self, type):
        """
        Генерирует новое изображение.
        """
        width, ok_w = QInputDialog.getInt(self, "Ширина", "Ширина:", 512, 1, 4096, 1)
        if ok_w:
            height, ok_h = QInputDialog.getInt(self, "Высота", "Высота:", 512, 1, 4096, 1)
            if ok_h:
                if type == "uniform":
                    low, ok_l = QInputDialog.getInt(self, "Нижняя граница", "Нижняя граница:", 0, 0, 255, 1)
                    if ok_l:
                        high, ok_h = QInputDialog.getInt(self, "Верхняя граница", "Верхняя граница:", 255, 0, 255, 1)
                        if ok_h:
                            new_image = image_processing.generate_uniform_noise_image(width, height, low, high)
                            self.image_viewer.set_image(new_image)
                            self.undo_action.setEnabled(True)
                elif type == "normal":
                    mean, ok_m = QInputDialog.getDouble(self, "Среднее", "Среднее:", 128.0, 0.0, 255.0, 2)
                    if ok_m:
                        std, ok_s = QInputDialog.getDouble(self, "Ст. отклонение", "Ст. отклонение:", 20.0, 0.1, 100.0, 2)
                        if ok_s:
                            new_image = image_processing.generate_normal_noise_image(width, height, mean, std)
                            self.image_viewer.set_image(new_image)
                            self.undo_action.setEnabled(True)

    def get_pixel(self):
        """
        Получает и отображает значение пикселя.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения.")
            return

        x, ok_x = QInputDialog.getInt(self, "Координата X", "Введите X:", 0, 0, self.image_viewer.get_image().width() - 1, 1)
        if ok_x:
            y, ok_y = QInputDialog.getInt(self, "Координата Y", "Введите Y:", 0, 0, self.image_viewer.get_image().height() - 1, 1)
            if ok_y:
                color = image_processing.get_pixel_value(self.image_viewer.get_image(), x, y)
                if color:
                    QMessageBox.information(self, f"Цвет пикселя ({x}, {y})", f"R: {color.red()}, G: {color.green()}, B: {color.blue()}")

    def set_pixel(self):
        """
        Устанавливает значение пикселя.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения.")
            return

        x, ok_x = QInputDialog.getInt(self, "Координата X", "Введите X:", 0, 0, self.image_viewer.get_image().width() - 1, 1)
        if ok_x:
            y, ok_y = QInputDialog.getInt(self, "Координата Y", "Введите Y:", 0, 0, self.image_viewer.get_image().height() - 1, 1)
            if ok_y:
                r, ok_r = QInputDialog.getInt(self, "R", "R:", 0, 0, 255, 1)
                if ok_r:
                    g, ok_g = QInputDialog.getInt(self, "G", "G:", 0, 0, 255, 1)
                    if ok_g:
                        b, ok_b = QInputDialog.getInt(self, "B", "B:", 0, 0, 255, 1)
                        if ok_b:
                            new_image = image_processing.set_pixel_value(self.image_viewer.get_image(), x, y, QColor(r, g, b))
                            self.image_viewer.set_image(new_image)
                            self.undo_action.setEnabled(True)

    def add_noise(self):
        """
        Добавляет гауссовский шум к изображению.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для обработки.")
            return

        sigma, ok = QInputDialog.getDouble(self, "Добавить шум", "Введите сигму:", 10.0, 0.1, 100.0, 2)
        if ok:
            new_image = image_processing.add_gaussian_noise(self.image_viewer.get_image(), sigma)
            self.image_viewer.set_image(new_image)
            self.undo_action.setEnabled(True)

    def reduce_noise(self):
        """
        Уменьшает шум с помощью медианного фильтра.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для обработки.")
            return

        radius, ok = QInputDialog.getInt(self, "Уменьшить шум", "Введите радиус медианного фильтра:", 1, 1, 10, 1)
        if ok:
            new_image = image_processing.reduce_noise_median(self.image_viewer.get_image(), radius)
            self.image_viewer.set_image(new_image)
            self.undo_action.setEnabled(True)

    def scale_image(self, method: str):
        """
        Масштабирует изображение с использованием выбранного метода.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для обработки.")
            return

        factor, ok = QInputDialog.getDouble(self, "Масштаб", "Введите коэффициент масштабирования:", 1.0, 0.1, 10.0, 2)
        if ok:
            if method == "nearest":
                new_image = image_processing.scale_image_nearest(self.image_viewer.get_image(), factor)
            elif method == "bilinear":
                new_image = image_processing.scale_image_bilinear(self.image_viewer.get_image(), factor)
            else:
                return

            self.image_viewer.set_image(new_image)
            self.undo_action.setEnabled(True)

    def apply_posterization(self):
        """
        Применяет постеризацию к изображению.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для обработки.")
            return

        levels, ok = QInputDialog.getInt(self, "Уровни постеризации", "Введите количество уровней:", 4, 2, 255, 1)
        if ok:
            selection = self.image_viewer.get_selection()
            new_image = image_processing.posterize_image(self.image_viewer.get_image(), levels, selection)
            self.image_viewer.set_image(new_image)
            self.undo_action.setEnabled(True)

    def apply_smoothing(self):
        """
        Применяет сглаживание к выделенной области или всему изображению.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для обработки.")
            return

        radius, ok = QInputDialog.getInt(self, "Радиус сглаживания", "Введите радиус:", 1, 1, 100, 1)
        if ok:
            selection = self.image_viewer.get_selection()
            new_image = image_processing.smooth_image(self.image_viewer.get_image(), radius, selection)
            self.image_viewer.set_image(new_image)
            self.undo_action.setEnabled(True)

    def apply_linear_contrast(self):
        """
        Применяет линейное контрастирование к выделенной области или всему изображению.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для обработки.")
            return

        selection = self.image_viewer.get_selection()
        new_image = image_processing.linear_contrast(self.image_viewer.get_image(), selection)
        self.image_viewer.set_image(new_image)
        self.undo_action.setEnabled(True)

    def show_histogram(self):
        """
        Отображает гистограмму для выделенной области или всего изображения.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для анализа.")
            return

        selection = self.image_viewer.get_selection()
        data = image_processing.get_histogram_data(self.image_viewer.get_image(), selection)

        if data:
            self.histogram_window = HistogramWindow(data)
            self.histogram_window.show()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось получить данные для гистограммы.")

    def show_statistics(self):
        """
        Отображает статистику для выделенной области или всего изображения.
        """
        if not self.image_viewer.get_image():
            QMessageBox.information(self, "Информация", "Нет изображения для анализа.")
            return

        selection = self.image_viewer.get_selection()
        stats = image_processing.calculate_statistics(self.image_viewer.get_image(), selection)

        if selection:
            title = "Статистика выделенной области"
        else:
            title = "Статистика всего изображения"

        if stats:
            message = (
                f"Минимум: {stats['min']}\n"
                f"Максимум: {stats['max']}\n"
                f"Среднее: {stats['mean']:.2f}\n"
                f"Стандартное отклонение: {stats['std']:.2f}"
            )
            QMessageBox.information(self, title, message)
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось рассчитать статистику.")

    def convert_to_grayscale(self):
        """
        Преобразует изображение в градации серого.
        """
        if self.image_viewer.get_image():
            selection = self.image_viewer.get_selection()
            new_image = image_processing.to_grayscale(self.image_viewer.get_image(), selection)
            self.image_viewer.set_image(new_image)
            self.undo_action.setEnabled(True)

    def open_image(self):
        """
        Открытие изображения.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            image = QImage(file_name)
            if image.isNull():
                QMessageBox.information(self, "Ошибка", "Не удалось загрузить изображение.")
                return
            self.image_viewer.set_image(image)
            self.undo_action.setEnabled(True)

    def save_image(self):
        """
        Сохранение изображения.
        """
        if self.image_viewer.get_image() is None:
            QMessageBox.information(self, "Информация", "Нет изображения для сохранения.")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "PNG (*.png);;JPEG (*.jpg);;Bitmap (*.bmp)")
        if file_name:
            self.image_viewer.get_image().save(file_name)

    def undo(self):
        """
        Отмена последнего действия.
        """
        if not self.image_viewer.undo():
            self.undo_action.setEnabled(False)
