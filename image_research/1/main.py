import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QScrollArea, QMenuBar, QAction, 
                             QFileDialog, QStatusBar, QMessageBox, QToolBar,
                             QSlider, QPushButton, QButtonGroup, QFrame, QDialog,
                             QTextEdit, QDialogButtonBox, QFormLayout, QLineEdit,
                             QRadioButton, QGroupBox)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QCursor, 
                         QIcon, QFont, QBrush)
from PIL import Image, ImageQt
import numpy as np
import copy

class ImageLabel(QLabel):
    """Кастомный QLabel для отображения изображения с поддержкой зуммирования и выделения областей"""
    
    mouse_moved = pyqtSignal(int, int, int, int, int)  # x, y, r, g, b
    zoom_changed = pyqtSignal(float)  # Сигнал для изменения зума
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self.setStyleSheet("border: 1px solid gray;")
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setFocusPolicy(Qt.WheelFocus)  # Для обработки колесика мыши
        
        # Изображение и масштабирование
        self.original_pixmap = None
        self.current_pixmap = None
        self.scale_factor = 1.0
        
        # Выделение области
        self.selection_mode = None  # 'rectangle', 'freehand', None
        self.selecting = False
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        # Координаты выделения в координатах оригинального изображения
        self.selection_rect_image = QRect()  # В координатах изображения
        self.freehand_points_image = []  # В координатах изображения
        
        # История для отмены операций
        self.history = []
        self.max_history = 10
        
    def set_image(self, pixmap):
        """Установка нового изображения"""
        if pixmap:
            self.save_state()
            self.original_pixmap = pixmap.copy()
            self.current_pixmap = pixmap.copy()
            self.scale_factor = 1.0
            self.update_display()
            
    def save_state(self):
        """Сохранение текущего состояния для отмены"""
        if self.original_pixmap:
            state = {
                'original_pixmap': self.original_pixmap.copy(),
                'current_pixmap': self.current_pixmap.copy(),
                'scale_factor': self.scale_factor,
                'selection_mode': self.selection_mode,
                'selection_rect_image': QRect(self.selection_rect_image),
                'freehand_points_image': self.freehand_points_image.copy()
            }
            self.history.append(state)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
    def undo(self):
        """Отмена последней операции"""
        if self.history:
            state = self.history.pop()
            self.original_pixmap = state['original_pixmap']
            self.current_pixmap = state['current_pixmap']
            self.scale_factor = state['scale_factor']
            
            # Восстановление состояния выделения
            self.selection_mode = state.get('selection_mode', None)
            self.selection_rect_image = QRect(state.get('selection_rect_image', QRect()))
            self.freehand_points_image = state.get('freehand_points_image', []).copy()
            
            self.update_display()
            self.update()  # Обновление отображения выделения
            return True
        return False
        
    def set_scale(self, scale):
        """Установка масштаба изображения"""
        if self.original_pixmap:
            self.scale_factor = scale
            self.update_display()
            
    def update_display(self):
        """Обновление отображения изображения"""
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            # Устанавливаем размер виджета равным размеру масштабированного изображения
            self.resize(scaled_pixmap.size())
            
    def set_selection_mode(self, mode):
        """Установка режима выделения"""
        # Сохранение состояния перед изменением режима выделения
        if self.original_pixmap and (self.selection_mode != mode):
            self.save_state()
        
        self.selection_mode = mode
        self.selection_rect_image = QRect()
        self.freehand_points_image = []
        self.update()
        
    def screen_to_image_coords(self, screen_point):
        """Преобразование экранных координат в координаты изображения"""
        if not self.original_pixmap:
            return QPoint()
            
        label_rect = self.rect()
        pixmap_rect = self.pixmap().rect() if self.pixmap() else QRect()
        
        if pixmap_rect.isEmpty():
            return QPoint()
            
        # Центрирование изображения в label
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2
        
        # Координаты относительно изображения
        img_x = int((screen_point.x() - x_offset) / self.scale_factor)
        img_y = int((screen_point.y() - y_offset) / self.scale_factor)
        
        # Ограничение координат границами изображения
        img_x = max(0, min(img_x, self.original_pixmap.width() - 1))
        img_y = max(0, min(img_y, self.original_pixmap.height() - 1))
        
        return QPoint(img_x, img_y)
        
    def image_to_screen_coords(self, image_point):
        """Преобразование координат изображения в экранные координаты"""
        if not self.original_pixmap:
            return QPoint()
            
        label_rect = self.rect()
        pixmap_rect = self.pixmap().rect() if self.pixmap() else QRect()
        
        if pixmap_rect.isEmpty():
            return QPoint()
            
        # Центрирование изображения в label
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2
        
        # Экранные координаты
        screen_x = int(image_point.x() * self.scale_factor + x_offset)
        screen_y = int(image_point.y() * self.scale_factor + y_offset)
        
        return QPoint(screen_x, screen_y)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.selection_mode:
            self.selecting = True
            self.selection_start = event.pos()
            
            # Преобразование в координаты изображения
            image_start = self.screen_to_image_coords(event.pos())
            
            if self.selection_mode == 'freehand':
                self.freehand_points_image = [image_start]
            elif self.selection_mode == 'rectangle':
                self.selection_rect_image = QRect(image_start, image_start)
        elif event.button() == Qt.RightButton and self.selection_mode == 'freehand' and self.selecting:
            # Правая кнопка мыши завершает рисование произвольной области
            self.selecting = False
            self.update()
                
    def mouseMoveEvent(self, event):
        # Отправка координат и RGB значений
        if self.current_pixmap and not self.current_pixmap.isNull():
            # Преобразование координат с учетом масштаба
            label_rect = self.rect()
            pixmap_rect = self.pixmap().rect() if self.pixmap() else QRect()
            
            if not pixmap_rect.isEmpty():
                # Центрирование изображения в label
                x_offset = (label_rect.width() - pixmap_rect.width()) // 2
                y_offset = (label_rect.height() - pixmap_rect.height()) // 2
                
                # Координаты относительно изображения
                img_x = int((event.x() - x_offset) / self.scale_factor)
                img_y = int((event.y() - y_offset) / self.scale_factor)
                
                # Получение RGB значений
                if (0 <= img_x < self.original_pixmap.width() and 
                    0 <= img_y < self.original_pixmap.height()):
                    
                    image = self.original_pixmap.toImage()
                    color = image.pixelColor(img_x, img_y)
                    self.mouse_moved.emit(img_x, img_y, color.red(), color.green(), color.blue())
        
        # Обработка выделения области
        if self.selecting and self.selection_mode:
            # Преобразование текущей позиции в координаты изображения
            image_current = self.screen_to_image_coords(event.pos())
            
            if self.selection_mode == 'rectangle':
                # Обновление прямоугольника в координатах изображения
                image_start = self.screen_to_image_coords(self.selection_start)
                self.selection_rect_image = QRect(image_start, image_current).normalized()
            elif self.selection_mode == 'freehand':
                # Улучшенное добавление точек для произвольного выделения
                if len(self.freehand_points_image) == 0:
                    # Первая точка
                    self.freehand_points_image.append(image_current)
                else:
                    # Проверяем расстояние от последней точки
                    last_point = self.freehand_points_image[-1]
                    distance = ((image_current.x() - last_point.x()) ** 2 + 
                               (image_current.y() - last_point.y()) ** 2) ** 0.5
                    
                    # Добавляем точку если расстояние больше минимального порога
                    # или если прошло достаточно времени с последней точки
                    min_distance = max(1, int(2 / self.scale_factor))  # Адаптивное расстояние в зависимости от масштаба
                    
                    if distance >= min_distance:
                        # Если расстояние большое, добавляем промежуточные точки для плавности
                        if distance > min_distance * 3:
                            self._add_interpolated_points(last_point, image_current, min_distance)
                        else:
                            self.freehand_points_image.append(image_current)
                    else:
                        # Даже если расстояние маленькое, обновляем последнюю точку для плавности
                        if len(self.freehand_points_image) > 0:
                            self.freehand_points_image[-1] = image_current
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.selecting:
            self.selecting = False
            
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.selection_mode and (self.selection_rect_image.isValid() or len(self.freehand_points_image) > 0):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            if self.selection_mode == 'rectangle' and self.selection_rect_image.isValid():
                # Преобразование прямоугольника из координат изображения в экранные
                top_left = self.image_to_screen_coords(self.selection_rect_image.topLeft())
                bottom_right = self.image_to_screen_coords(self.selection_rect_image.bottomRight())
                screen_rect = QRect(top_left, bottom_right)
                
                pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(screen_rect)
                
            elif self.selection_mode == 'freehand' and len(self.freehand_points_image) > 1:
                # Преобразование точек из координат изображения в экранные
                screen_points = [self.image_to_screen_coords(point) for point in self.freehand_points_image]
                
                pen = QPen(QColor(255, 0, 0), 2)
                painter.setPen(pen)
                
                # Рисование линий между точками
                for i in range(1, len(screen_points)):
                    painter.drawLine(screen_points[i-1], screen_points[i])
                
                # Автоматическое замыкание фигуры: линия от последней точки к первой
                if len(screen_points) > 2 and not self.selecting:
                    painter.drawLine(screen_points[-1], screen_points[0])
                    
    def wheelEvent(self, event):
        """Обработка колесика мыши для зуммирования с центрированием на курсоре"""
        if self.original_pixmap:
            # Получение направления прокрутки
            delta = event.angleDelta().y()
            
            # Вычисление нового масштаба
            zoom_factor = 1.1 if delta > 0 else 0.9
            new_scale = self.scale_factor * zoom_factor
            
            # Ограничение масштаба
            new_scale = max(0.1, min(5.0, new_scale))
            
            # Получение позиции курсора для центрирования
            cursor_pos = event.pos()
            
            # Отправка сигнала об изменении зума с позицией курсора
            self.zoom_changed.emit(new_scale)
            
            # Центрирование на позиции курсора будет обработано в главном окне
            self.zoom_center_point = cursor_pos
            
    def is_point_in_selection(self, x, y):
        """Проверка принадлежности точки к выделенной области (координаты в пикселях изображения)"""
        if self.selection_mode == 'rectangle' and self.selection_rect_image.isValid():
            # Проверка принадлежности точки прямоугольному выделению
            return self.selection_rect_image.contains(x, y)
        elif self.selection_mode == 'freehand' and len(self.freehand_points_image) > 2:
            # Использование алгоритма ray casting для проверки принадлежности точки многоугольнику
            return self._point_in_polygon(x, y, self.freehand_points_image)
        return False
        
    def _point_in_polygon(self, x, y, polygon_points):
        """Алгоритм ray casting для определения принадлежности точки многоугольнику"""
        if len(polygon_points) < 3:
            return False
            
        inside = False
        j = len(polygon_points) - 1
        
        for i in range(len(polygon_points)):
            xi, yi = polygon_points[i].x(), polygon_points[i].y()
            xj, yj = polygon_points[j].x(), polygon_points[j].y()
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
            
        return inside
        
    def _point_to_line_distance(self, point, line_start, line_end):
        """Вычисление расстояния от точки до отрезка"""
        # Векторы
        line_vec = QPoint(line_end.x() - line_start.x(), line_end.y() - line_start.y())
        point_vec = QPoint(point.x() - line_start.x(), point.y() - line_start.y())
        
        # Длина отрезка
        line_len_sq = line_vec.x() ** 2 + line_vec.y() ** 2
        if line_len_sq == 0:
            # Отрезок вырожден в точку
            return ((point.x() - line_start.x()) ** 2 + (point.y() - line_start.y()) ** 2) ** 0.5
        
        # Проекция точки на прямую
        t = max(0, min(1, (point_vec.x() * line_vec.x() + point_vec.y() * line_vec.y()) / line_len_sq))
        
        # Ближайшая точка на отрезке
        closest_x = line_start.x() + t * line_vec.x()
        closest_y = line_start.y() + t * line_vec.y()
        
        # Расстояние
        return ((point.x() - closest_x) ** 2 + (point.y() - closest_y) ** 2) ** 0.5
        
    def _add_interpolated_points(self, start_point, end_point, step_size):
        """Добавление промежуточных точек между двумя точками для плавности линии"""
        dx = end_point.x() - start_point.x()
        dy = end_point.y() - start_point.y()
        distance = (dx * dx + dy * dy) ** 0.5
        
        if distance <= step_size:
            self.freehand_points_image.append(end_point)
            return
            
        # Количество промежуточных точек
        num_steps = int(distance / step_size)
        
        for i in range(1, num_steps + 1):
            t = i / num_steps
            interp_x = int(start_point.x() + dx * t)
            interp_y = int(start_point.y() + dy * t)
            self.freehand_points_image.append(QPoint(interp_x, interp_y))


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer - Обработка изображений")
        self.setGeometry(100, 100, 1000, 700)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Панель инструментов
        self.create_toolbar()
        
        # Область прокрутки для изображения
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)  # Важно: False для показа скроллбаров
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Виджет для отображения изображения
        self.image_label = ImageLabel()
        self.image_label.mouse_moved.connect(self.update_status_bar)
        self.image_label.zoom_changed.connect(self.on_wheel_zoom)
        self.scroll_area.setWidget(self.image_label)
        
        main_layout.addWidget(self.scroll_area)
        
        # Создание меню и строки состояния
        self.create_menu()
        self.create_status_bar()
        
        # Текущий файл
        self.current_file = None
        
    def create_menu(self):
        """Создание меню"""
        menubar = self.menuBar()
        
        # Меню "Файл"
        file_menu = menubar.addMenu('Файл')
        
        open_action = QAction('Открыть', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction('Сохранить', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        save_as_action = QAction('Сохранить как...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню "Правка"
        edit_menu = menubar.addMenu('Правка')
        
        undo_action = QAction('Отменить', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo_operation)
        edit_menu.addAction(undo_action)
        
        # Меню "Изображение"
        image_menu = menubar.addMenu('Изображение')
        
        grayscale_action = QAction('Преобразовать в градации серого...', self)
        grayscale_action.triggered.connect(self.show_grayscale_dialog)
        image_menu.addAction(grayscale_action)
        
        image_menu.addSeparator()
        
        bitmap_to_bytes_action = QAction('Экспорт в массив байтов', self)
        bitmap_to_bytes_action.triggered.connect(self.export_to_byte_array)
        image_menu.addAction(bitmap_to_bytes_action)
        
        bytes_to_bitmap_action = QAction('Импорт из массива байтов', self)
        bytes_to_bitmap_action.triggered.connect(self.import_from_byte_array)
        image_menu.addAction(bytes_to_bitmap_action)
        
        image_menu.addSeparator()
        
        image_info_action = QAction('О изображении...', self)
        image_info_action.triggered.connect(self.show_image_info)
        image_menu.addAction(image_info_action)
        
        # Меню "Выделение"
        selection_menu = menubar.addMenu('Выделение')
        
        rect_selection_action = QAction('Прямоугольное выделение', self)
        rect_selection_action.triggered.connect(lambda: self.set_selection_mode('rectangle'))
        selection_menu.addAction(rect_selection_action)
        
        freehand_selection_action = QAction('Произвольное выделение', self)
        freehand_selection_action.triggered.connect(lambda: self.set_selection_mode('freehand'))
        selection_menu.addAction(freehand_selection_action)
        
        clear_selection_action = QAction('Очистить выделение', self)
        clear_selection_action.triggered.connect(lambda: self.set_selection_mode(None))
        selection_menu.addAction(clear_selection_action)
        
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Кнопки масштабирования
        zoom_in_btn = QPushButton("Увеличить")
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Уменьшить")
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar.addWidget(zoom_out_btn)
        
        zoom_fit_btn = QPushButton("По размеру")
        zoom_fit_btn.clicked.connect(self.zoom_fit)
        toolbar.addWidget(zoom_fit_btn)
        
        zoom_100_btn = QPushButton("100%")
        zoom_100_btn.clicked.connect(self.zoom_100)
        toolbar.addWidget(zoom_100_btn)
        
        toolbar.addSeparator()
        
        # Слайдер масштаба
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        toolbar.addWidget(QLabel("Масштаб:"))
        toolbar.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)
        
    def create_status_bar(self):
        """Создание строки состояния"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.coords_label = QLabel("Координаты: -")
        self.rgb_label = QLabel("RGB: -")
        
        self.status_bar.addWidget(self.coords_label)
        self.status_bar.addPermanentWidget(self.rgb_label)
        
    def update_status_bar(self, x, y, r, g, b):
        """Обновление строки состояния"""
        self.coords_label.setText(f"Координаты: ({x}, {y})")
        self.rgb_label.setText(f"RGB: ({r}, {g}, {b})")
        
    def open_image(self):
        """Открытие изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Открыть изображение",
            "",
            "Изображения (*.bmp *.png *.jpg *.jpeg);;Все файлы (*)"
        )
        
        if file_path:
            try:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    self.image_label.set_image(pixmap)
                    self.current_file = file_path
                    self.setWindowTitle(f"Image Viewer - {os.path.basename(file_path)}")
                    self.zoom_fit()
                else:
                    QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке изображения: {str(e)}")
                
    def save_image(self):
        """Сохранение изображения"""
        if self.current_file and self.image_label.current_pixmap:
            try:
                self.image_label.current_pixmap.save(self.current_file)
                QMessageBox.information(self, "Успех", "Изображение сохранено")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")
        else:
            self.save_image_as()
            
    def save_image_as(self):
        """Сохранение изображения как..."""
        if not self.image_label.current_pixmap:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для сохранения")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            "",
            "BMP (*.bmp);;PNG (*.png);;JPEG (*.jpg);;Все файлы (*)"
        )
        
        if file_path:
            try:
                self.image_label.current_pixmap.save(file_path)
                self.current_file = file_path
                self.setWindowTitle(f"Image Viewer - {os.path.basename(file_path)}")
                QMessageBox.information(self, "Успех", "Изображение сохранено")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")
                
    def convert_to_grayscale(self):
        """Преобразование в градации серого"""
        if not self.image_label.original_pixmap:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для обработки")
            return
            
        try:
            # Сохранение состояния
            self.image_label.save_state()
            
            # Преобразование QPixmap в QImage
            image = self.image_label.current_pixmap.toImage()
            
            # Проверка наличия активного выделения
            has_selection = (
                (self.image_label.selection_mode == 'rectangle' and self.image_label.selection_rect_image.isValid()) or
                (self.image_label.selection_mode == 'freehand' and len(self.image_label.freehand_points_image) > 2)
            )
            
            if has_selection:
                # Оптимизированное селективное преобразование в градации серого
                result_image = image.copy()
                
                if self.image_label.selection_mode == 'rectangle':
                    # Для прямоугольного выделения - простая обработка области
                    rect = self.image_label.selection_rect_image
                    
                    # Ограничение области границами изображения
                    x1 = max(0, rect.left())
                    y1 = max(0, rect.top())
                    x2 = min(image.width() - 1, rect.right())
                    y2 = min(image.height() - 1, rect.bottom())
                    
                    # Обработка только пикселей в прямоугольнике
                    for y in range(y1, y2 + 1):
                        for x in range(x1, x2 + 1):
                            color = image.pixelColor(x, y)
                            gray = int(0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())
                            result_image.setPixelColor(x, y, QColor(gray, gray, gray))
                            
                elif self.image_label.selection_mode == 'freehand':
                    # Для произвольного выделения - оптимизация с bounding box
                    points = self.image_label.freehand_points_image
                    
                    # Вычисление bounding box для ограничения области проверки
                    min_x = max(0, min(p.x() for p in points))
                    max_x = min(image.width() - 1, max(p.x() for p in points))
                    min_y = max(0, min(p.y() for p in points))
                    max_y = min(image.height() - 1, max(p.y() for p in points))
                    
                    # Предварительное создание маски для быстрой проверки
                    # Используем scanline алгоритм для заполнения полигона
                    mask = self._create_polygon_mask(points, min_x, min_y, max_x, max_y)
                    
                    # Обработка только пикселей в bounding box с использованием маски
                    for y in range(min_y, max_y + 1):
                        for x in range(min_x, max_x + 1):
                            if mask.get((x, y), False):
                                color = image.pixelColor(x, y)
                                gray = int(0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())
                                result_image.setPixelColor(x, y, QColor(gray, gray, gray))
                
                # Обратное преобразование в QPixmap
                grayscale_pixmap = QPixmap.fromImage(result_image)
            else:
                # Полное преобразование в градации серого
                grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)
                grayscale_pixmap = QPixmap.fromImage(grayscale_image)
            
            # Обновление изображения
            self.image_label.current_pixmap = grayscale_pixmap
            self.image_label.update_display()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при преобразовании: {str(e)}")
            
    def _create_polygon_mask(self, points, min_x, min_y, max_x, max_y):
        """Создание маски для полигона с использованием scanline алгоритма"""
        mask = {}
        
        if len(points) < 3:
            return mask
            
        # Создание списка рёбер полигона
        edges = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            # Убеждаемся, что p1.y <= p2.y
            if p1.y() > p2.y():
                p1, p2 = p2, p1
                
            if p1.y() != p2.y():  # Игнорируем горизонтальные рёбра
                edges.append((p1.y(), p2.y(), p1.x(), p2.x()))
        
        # Scanline алгоритм
        for y in range(min_y, max_y + 1):
            intersections = []
            
            for y1, y2, x1, x2 in edges:
                if y1 <= y < y2:
                    # Вычисление x-координаты пересечения
                    if y2 != y1:
                        x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                        intersections.append(int(x))
            
            # Сортировка пересечений
            intersections.sort()
            
            # Заполнение между парами пересечений
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = max(min_x, intersections[i])
                    x_end = min(max_x, intersections[i + 1])
                    
                    for x in range(x_start, x_end + 1):
                        mask[(x, y)] = True
        
        return mask
            
    def rgb_to_byte_array(self, pixmap):
        """Преобразование RGB изображения в массив байтов"""
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        
        byte_array = []
        for y in range(height):
            for x in range(width):
                color = image.pixelColor(x, y)
                # Преобразование в градации серого по формуле
                gray = int(0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())
                byte_array.append(gray)
                
        return np.array(byte_array, dtype=np.uint8), width, height
        
    def byte_array_to_pixmap(self, byte_array, width, height):
        """Преобразование массива байтов в QPixmap"""
        # Создание QImage из массива байтов
        image = QImage(byte_array.data, width, height, width, QImage.Format_Grayscale8)
        return QPixmap.fromImage(image)
        
    def set_selection_mode(self, mode):
        """Установка режима выделения"""
        self.image_label.set_selection_mode(mode)
        
    def undo_operation(self):
        """Отмена последней операции"""
        if self.image_label.undo():
            QMessageBox.information(self, "Отмена", "Последняя операция отменена")
        else:
            QMessageBox.information(self, "Отмена", "Нет операций для отмены")
            
    def zoom_in(self):
        """Увеличение масштаба"""
        current_value = self.zoom_slider.value()
        new_value = min(500, current_value + 25)
        self.zoom_slider.setValue(new_value)
        
    def zoom_out(self):
        """Уменьшение масштаба"""
        current_value = self.zoom_slider.value()
        new_value = max(10, current_value - 25)
        self.zoom_slider.setValue(new_value)
        
    def zoom_fit(self):
        """Масштабирование по размеру окна"""
        if not self.image_label.original_pixmap:
            return
            
        scroll_size = self.scroll_area.size()
        image_size = self.image_label.original_pixmap.size()
        
        # Вычисление масштаба для помещения изображения в окно
        scale_x = scroll_size.width() / image_size.width()
        scale_y = scroll_size.height() / image_size.height()
        scale = min(scale_x, scale_y, 1.0) * 100
        
        self.zoom_slider.setValue(int(scale))
        
    def zoom_100(self):
        """Масштаб 100%"""
        self.zoom_slider.setValue(100)
        
    def on_zoom_slider_changed(self, value):
        """Обработка изменения слайдера масштаба"""
        scale = value / 100.0
        self.image_label.set_scale(scale)
        self.zoom_label.setText(f"{value}%")
        
    def on_wheel_zoom(self, scale):
        """Обработка зуммирования колесиком мыши с центрированием на курсоре"""
        # Сохранение текущей позиции скроллбаров
        old_h_value = self.scroll_area.horizontalScrollBar().value()
        old_v_value = self.scroll_area.verticalScrollBar().value()
        
        # Получение позиции курсора относительно scroll area
        if hasattr(self.image_label, 'zoom_center_point'):
            cursor_pos = self.image_label.zoom_center_point
            
            # Преобразование позиции курсора в координаты изображения
            image_point = self.image_label.screen_to_image_coords(cursor_pos)
            
            # Обновление слайдера без вызова сигнала
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(scale * 100))
            self.zoom_slider.blockSignals(False)
            
            # Обновление масштаба и метки
            old_scale = self.image_label.scale_factor
            self.image_label.set_scale(scale)
            self.zoom_label.setText(f"{int(scale * 100)}%")
            
            # Вычисление новой позиции скроллбаров для центрирования на курсоре
            if self.image_label.original_pixmap:
                # Новые экранные координаты той же точки изображения
                new_screen_point = self.image_label.image_to_screen_coords(image_point)
                
                # Размеры scroll area
                scroll_rect = self.scroll_area.viewport().rect()
                
                # Вычисление смещения для центрирования
                center_x = scroll_rect.width() // 2
                center_y = scroll_rect.height() // 2
                
                # Новые значения скроллбаров
                new_h_value = new_screen_point.x() - center_x
                new_v_value = new_screen_point.y() - center_y
                
                # Установка новых значений скроллбаров
                self.scroll_area.horizontalScrollBar().setValue(new_h_value)
                self.scroll_area.verticalScrollBar().setValue(new_v_value)
        else:
            # Обычное зуммирование без центрирования
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(scale * 100))
            self.zoom_slider.blockSignals(False)
            
            self.image_label.set_scale(scale)
            self.zoom_label.setText(f"{int(scale * 100)}%")
        
    def export_to_byte_array(self):
        """Экспорт изображения в массив байтов"""
        if not self.image_label.current_pixmap:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для экспорта")
            return
            
        try:
            # Преобразование в массив байтов
            byte_array, width, height = self.rgb_to_byte_array(self.image_label.current_pixmap)
            
            # Создание диалога для экспорта
            dialog = ByteArrayExportDialog(byte_array, width, height, self)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте: {str(e)}")
            
    def import_from_byte_array(self):
        """Импорт изображения из массива байтов"""
        try:
            # Создание диалога для импорта
            dialog = ByteArrayImportDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                byte_array, width, height = dialog.get_data()
                
                # Преобразование в QPixmap
                pixmap = self.byte_array_to_pixmap(byte_array, width, height)
                
                # Установка изображения
                self.image_label.set_image(pixmap)
                self.current_file = None
                self.setWindowTitle("Image Viewer - Импортированное изображение")
                self.zoom_fit()
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при импорте: {str(e)}")
            
    def show_grayscale_dialog(self):
        """Показать диалог выбора параметров преобразования в градации серого"""
        if not self.image_label.original_pixmap:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для обработки")
            return
            
        try:
            # Создание диалога выбора параметров
            dialog = GrayscaleOptionsDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                method, output_format = dialog.get_options()
                self.convert_to_grayscale_with_options(method, output_format)
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при отображении диалога: {str(e)}")
            
    def convert_to_grayscale_with_options(self, method, output_format):
        """Преобразование в градации серого с выбранными параметрами"""
        if not self.image_label.original_pixmap:
            return
            
        try:
            # Сохранение состояния
            self.image_label.save_state()
            
            # Преобразование QPixmap в QImage
            image = self.image_label.current_pixmap.toImage()
            
            # Проверка наличия активного выделения
            has_selection = (
                (self.image_label.selection_mode == 'rectangle' and self.image_label.selection_rect_image.isValid()) or
                (self.image_label.selection_mode == 'freehand' and len(self.image_label.freehand_points_image) > 2)
            )
            
            if has_selection:
                # Селективное преобразование в градации серого
                result_image = image.copy()
                
                if self.image_label.selection_mode == 'rectangle':
                    # Для прямоугольного выделения
                    rect = self.image_label.selection_rect_image
                    x1 = max(0, rect.left())
                    y1 = max(0, rect.top())
                    x2 = min(image.width() - 1, rect.right())
                    y2 = min(image.height() - 1, rect.bottom())
                    
                    for y in range(y1, y2 + 1):
                        for x in range(x1, x2 + 1):
                            color = image.pixelColor(x, y)
                            gray_value = self._calculate_grayscale(color, method)
                            result_image.setPixelColor(x, y, QColor(gray_value, gray_value, gray_value))
                            
                elif self.image_label.selection_mode == 'freehand':
                    # Для произвольного выделения
                    points = self.image_label.freehand_points_image
                    min_x = max(0, min(p.x() for p in points))
                    max_x = min(image.width() - 1, max(p.x() for p in points))
                    min_y = max(0, min(p.y() for p in points))
                    max_y = min(image.height() - 1, max(p.y() for p in points))
                    
                    mask = self._create_polygon_mask(points, min_x, min_y, max_x, max_y)
                    
                    for y in range(min_y, max_y + 1):
                        for x in range(min_x, max_x + 1):
                            if mask.get((x, y), False):
                                color = image.pixelColor(x, y)
                                gray_value = self._calculate_grayscale(color, method)
                                result_image.setPixelColor(x, y, QColor(gray_value, gray_value, gray_value))
                
                # Применение выходного формата
                final_image = self._apply_output_format(result_image, output_format)
                grayscale_pixmap = QPixmap.fromImage(final_image)
            else:
                # Полное преобразование изображения
                if method == 'qt_builtin':
                    # Использование встроенного метода Qt
                    grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)
                else:
                    # Применение пользовательского метода ко всему изображению
                    result_image = image.copy()
                    for y in range(image.height()):
                        for x in range(image.width()):
                            color = image.pixelColor(x, y)
                            gray_value = self._calculate_grayscale(color, method)
                            result_image.setPixelColor(x, y, QColor(gray_value, gray_value, gray_value))
                    grayscale_image = result_image
                
                # Применение выходного формата
                final_image = self._apply_output_format(grayscale_image, output_format)
                grayscale_pixmap = QPixmap.fromImage(final_image)
            
            # Обновление изображения
            self.image_label.current_pixmap = grayscale_pixmap
            self.image_label.update_display()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при преобразовании: {str(e)}")
            
    def _calculate_grayscale(self, color, method):
        """Вычисление значения серого по выбранному методу"""
        r, g, b = color.red(), color.green(), color.blue()
        
        if method == 'luminance':
            # Стандартная формула яркости (ITU-R BT.709)
            return int(0.299 * r + 0.587 * g + 0.114 * b)
        elif method == 'average':
            # Простое среднее арифметическое
            return int((r + g + b) / 3)
        elif method == 'lightness':
            # Среднее между максимальным и минимальным значением
            return int((max(r, g, b) + min(r, g, b)) / 2)
        elif method == 'red_channel':
            # Только красный канал
            return r
        elif method == 'green_channel':
            # Только зеленый канал
            return g
        elif method == 'blue_channel':
            # Только синий канал
            return b
        else:
            # По умолчанию - стандартная формула
            return int(0.299 * r + 0.587 * g + 0.114 * b)
            
    def _apply_output_format(self, image, output_format):
        """Применение выходного формата к изображению"""
        if output_format == 'grayscale8':
            return image.convertToFormat(QImage.Format_Grayscale8)
        elif output_format == 'rgb32':
            return image.convertToFormat(QImage.Format_RGB32)
        elif output_format == 'argb32':
            return image.convertToFormat(QImage.Format_ARGB32)
        elif output_format == 'original':
            return image
        else:
            return image.convertToFormat(QImage.Format_Grayscale8)

    def show_image_info(self):
        """Показать информацию об изображении"""
        if not self.image_label.current_pixmap:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для отображения информации")
            return
            
        try:
            # Создание диалога с информацией об изображении
            dialog = ImageInfoDialog(self.image_label.current_pixmap, self.current_file, self)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при отображении информации: {str(e)}")

    def resizeEvent(self, event):
        """Обработка изменения размера окна"""
        super().resizeEvent(event)
        # При изменении размера окна скроллбары автоматически обновятся
        # благодаря правильной настройке QScrollArea
        if hasattr(self, 'scroll_area') and hasattr(self, 'image_label'):
            # Обновляем область прокрутки
            self.scroll_area.updateGeometry()


class ByteArrayExportDialog(QDialog):
    """Диалог для экспорта массива байтов"""
    
    def __init__(self, byte_array, width, height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Экспорт в массив байтов")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Информация о массиве
        info_layout = QFormLayout()
        info_layout.addRow("Ширина:", QLabel(str(width)))
        info_layout.addRow("Высота:", QLabel(str(height)))
        info_layout.addRow("Размер массива:", QLabel(f"{len(byte_array)} байт"))
        
        layout.addLayout(info_layout)
        
        # Текстовое поле с массивом байтов
        layout.addWidget(QLabel("Массив байтов (первые 1000 элементов):"))
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 9))
        
        # Отображение первых 1000 элементов для производительности
        display_array = byte_array[:1000] if len(byte_array) > 1000 else byte_array
        array_text = ", ".join(map(str, display_array))
        if len(byte_array) > 1000:
            array_text += f"\n... и еще {len(byte_array) - 1000} элементов"
        
        self.text_edit.setPlainText(f"[{array_text}]")
        layout.addWidget(self.text_edit)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        copy_button = QPushButton("Копировать в буфер")
        copy_button.clicked.connect(lambda: self.copy_to_clipboard(byte_array))
        button_layout.addWidget(copy_button)
        
        save_button = QPushButton("Сохранить в файл")
        save_button.clicked.connect(lambda: self.save_to_file(byte_array, width, height))
        button_layout.addWidget(save_button)
        
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
    def copy_to_clipboard(self, byte_array):
        """Копирование массива в буфер обмена"""
        try:
            from PyQt5.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            array_text = ", ".join(map(str, byte_array))
            clipboard.setText(f"[{array_text}]")
            QMessageBox.information(self, "Успех", "Массив скопирован в буфер обмена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при копировании: {str(e)}")
            
    def save_to_file(self, byte_array, width, height):
        """Сохранение массива в файл"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить массив байтов",
                "",
                "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Массив байтов изображения\n")
                    f.write(f"# Ширина: {width}\n")
                    f.write(f"# Высота: {height}\n")
                    f.write(f"# Размер: {len(byte_array)} байт\n\n")
                    
                    array_text = ", ".join(map(str, byte_array))
                    f.write(f"[{array_text}]")
                    
                QMessageBox.information(self, "Успех", "Массив сохранен в файл")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")


class ByteArrayImportDialog(QDialog):
    """Диалог для импорта массива байтов"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Импорт из массива байтов")
        self.setModal(True)
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Поля для ввода размеров
        size_layout = QFormLayout()
        
        self.width_edit = QLineEdit()
        self.width_edit.setPlaceholderText("Например: 256")
        size_layout.addRow("Ширина:", self.width_edit)
        
        self.height_edit = QLineEdit()
        self.height_edit.setPlaceholderText("Например: 256")
        size_layout.addRow("Высота:", self.height_edit)
        
        layout.addLayout(size_layout)
        
        # Текстовое поле для ввода массива
        layout.addWidget(QLabel("Массив байтов (значения от 0 до 255, разделенные запятыми):"))
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Введите массив в формате: [123, 45, 67, 89, ...] или просто: 123, 45, 67, 89, ...")
        self.text_edit.setFont(QFont("Courier", 9))
        layout.addWidget(self.text_edit)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        load_button = QPushButton("Загрузить из файла")
        load_button.clicked.connect(self.load_from_file)
        button_layout.addWidget(load_button)
        
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
    def load_from_file(self):
        """Загрузка массива из файла"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Загрузить массив байтов",
                "",
                "Текстовые файлы (*.txt);;Все файлы (*)"
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Попытка извлечь размеры из комментариев
                lines = content.split('\n')
                for line in lines:
                    if line.startswith('# Ширина:'):
                        width = line.split(':')[1].strip()
                        self.width_edit.setText(width)
                    elif line.startswith('# Высота:'):
                        height = line.split(':')[1].strip()
                        self.height_edit.setText(height)
                
                # Поиск массива в файле
                import re
                array_match = re.search(r'\[([\d\s,]+)\]', content)
                if array_match:
                    self.text_edit.setPlainText(array_match.group(1))
                else:
                    self.text_edit.setPlainText(content)
                    
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке файла: {str(e)}")
            
    def get_data(self):
        """Получение данных из диалога"""
        try:
            # Получение размеров
            width = int(self.width_edit.text())
            height = int(self.height_edit.text())
            
            if width <= 0 or height <= 0:
                raise ValueError("Размеры должны быть положительными числами")
            
            # Парсинг массива
            text = self.text_edit.toPlainText().strip()
            
            # Удаление квадратных скобок если есть
            text = text.strip('[]')
            
            # Разделение по запятым и преобразование в числа
            values = []
            for item in text.split(','):
                item = item.strip()
                if item:
                    value = int(item)
                    if not (0 <= value <= 255):
                        raise ValueError(f"Значение {value} должно быть в диапазоне 0-255")
                    values.append(value)
            
            # Проверка размера массива
            expected_size = width * height
            if len(values) != expected_size:
                raise ValueError(f"Размер массива ({len(values)}) не соответствует размерам изображения ({width}x{height}={expected_size})")
            
            return np.array(values, dtype=np.uint8), width, height
            
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка в данных: {str(e)}")
            raise
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обработке данных: {str(e)}")
            raise


class GrayscaleOptionsDialog(QDialog):
    """Диалог для выбора параметров преобразования в градации серого"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры преобразования в градации серого")
        self.setModal(True)
        self.resize(400, 350)
        
        layout = QVBoxLayout(self)
        
        # Группа методов преобразования
        method_group = QGroupBox("Метод преобразования:")
        method_layout = QVBoxLayout(method_group)
        
        self.method_buttons = QButtonGroup()
        
        # Стандартная формула яркости (по умолчанию)
        self.luminance_radio = QRadioButton("Яркость (ITU-R BT.709): 0.299×R + 0.587×G + 0.114×B")
        self.luminance_radio.setChecked(True)
        self.method_buttons.addButton(self.luminance_radio, 0)
        method_layout.addWidget(self.luminance_radio)
        
        # Простое среднее
        self.average_radio = QRadioButton("Среднее арифметическое: (R + G + B) / 3")
        self.method_buttons.addButton(self.average_radio, 1)
        method_layout.addWidget(self.average_radio)
        
        # Lightness
        self.lightness_radio = QRadioButton("Lightness: (max(R,G,B) + min(R,G,B)) / 2")
        self.method_buttons.addButton(self.lightness_radio, 2)
        method_layout.addWidget(self.lightness_radio)
        
        # Отдельные каналы
        self.red_radio = QRadioButton("Только красный канал (R)")
        self.method_buttons.addButton(self.red_radio, 3)
        method_layout.addWidget(self.red_radio)
        
        self.green_radio = QRadioButton("Только зеленый канал (G)")
        self.method_buttons.addButton(self.green_radio, 4)
        method_layout.addWidget(self.green_radio)
        
        self.blue_radio = QRadioButton("Только синий канал (B)")
        self.method_buttons.addButton(self.blue_radio, 5)
        method_layout.addWidget(self.blue_radio)
        
        # Встроенный метод Qt
        self.qt_builtin_radio = QRadioButton("Встроенный метод Qt (быстрый)")
        self.method_buttons.addButton(self.qt_builtin_radio, 6)
        method_layout.addWidget(self.qt_builtin_radio)
        
        layout.addWidget(method_group)
        
        # Группа выходного формата
        format_group = QGroupBox("Выходной формат:")
        format_layout = QVBoxLayout(format_group)
        
        self.format_buttons = QButtonGroup()
        
        # 8-битные градации серого (по умолчанию)
        self.grayscale8_radio = QRadioButton("8-битные градации серого (Grayscale8)")
        self.grayscale8_radio.setChecked(True)
        self.format_buttons.addButton(self.grayscale8_radio, 0)
        format_layout.addWidget(self.grayscale8_radio)
        
        # 24-битный RGB
        self.rgb32_radio = QRadioButton("24-битный RGB (RGB32)")
        self.format_buttons.addButton(self.rgb32_radio, 1)
        format_layout.addWidget(self.rgb32_radio)
        
        # 32-битный ARGB
        self.argb32_radio = QRadioButton("32-битный ARGB (ARGB32)")
        self.format_buttons.addButton(self.argb32_radio, 2)
        format_layout.addWidget(self.argb32_radio)
        
        # Оригинальный формат
        self.original_radio = QRadioButton("Сохранить оригинальный формат")
        self.format_buttons.addButton(self.original_radio, 3)
        format_layout.addWidget(self.original_radio)
        
        layout.addWidget(format_group)
        
        # Кнопки
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_options(self):
        """Получение выбранных параметров"""
        # Определение метода
        method_id = self.method_buttons.checkedId()
        method_map = {
            0: 'luminance',
            1: 'average', 
            2: 'lightness',
            3: 'red_channel',
            4: 'green_channel',
            5: 'blue_channel',
            6: 'qt_builtin'
        }
        method = method_map.get(method_id, 'luminance')
        
        # Определение формата
        format_id = self.format_buttons.checkedId()
        format_map = {
            0: 'grayscale8',
            1: 'rgb32',
            2: 'argb32',
            3: 'original'
        }
        output_format = format_map.get(format_id, 'grayscale8')
        
        return method, output_format


class ImageInfoDialog(QDialog):
    """Диалог с информацией об изображении"""
    
    def __init__(self, pixmap, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О изображении")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Основная информация
        info_layout = QFormLayout()
        
        # Размеры
        info_layout.addRow("Ширина:", QLabel(f"{pixmap.width()} пикселей"))
        info_layout.addRow("Высота:", QLabel(f"{pixmap.height()} пикселей"))
        info_layout.addRow("Разрешение:", QLabel(f"{pixmap.width()} × {pixmap.height()}"))
        
        # Информация о файле
        if file_path:
            info_layout.addRow("Файл:", QLabel(os.path.basename(file_path)))
            info_layout.addRow("Путь:", QLabel(file_path))
            
            try:
                file_size = os.path.getsize(file_path)
                if file_size < 1024:
                    size_str = f"{file_size} байт"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} КБ"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} МБ"
                info_layout.addRow("Размер файла:", QLabel(size_str))
            except:
                pass
        else:
            info_layout.addRow("Файл:", QLabel("Не сохранен"))

        # Информация о формате
        image = pixmap.toImage()
        format_names = {
            QImage.Format_RGB32: "RGB32",
            QImage.Format_ARGB32: "ARGB32",
            QImage.Format_ARGB32_Premultiplied: "ARGB32 (Premultiplied)",
            QImage.Format_RGB16: "RGB16",
            QImage.Format_ARGB8565_Premultiplied: "ARGB8565 (Premultiplied)",
            QImage.Format_RGB666: "RGB666",
            QImage.Format_ARGB6666_Premultiplied: "ARGB6666 (Premultiplied)",
            QImage.Format_RGB555: "RGB555",
            QImage.Format_ARGB8555_Premultiplied: "ARGB8555 (Premultiplied)",
            QImage.Format_RGB888: "RGB888",
            QImage.Format_RGB444: "RGB444",
            QImage.Format_ARGB4444_Premultiplied: "ARGB4444 (Premultiplied)",
            QImage.Format_RGBX8888: "RGBX8888",
            QImage.Format_RGBA8888: "RGBA8888",
            QImage.Format_RGBA8888_Premultiplied: "RGBA8888 (Premultiplied)",
            QImage.Format_BGR30: "BGR30",
            QImage.Format_A2BGR30_Premultiplied: "A2BGR30 (Premultiplied)",
            QImage.Format_RGB30: "RGB30",
            QImage.Format_A2RGB30_Premultiplied: "A2RGB30 (Premultiplied)",
            QImage.Format_Alpha8: "Alpha8",
            QImage.Format_Grayscale8: "Grayscale8"
        }
        
        format_name = format_names.get(image.format(), f"Неизвестный ({image.format()})")
        info_layout.addRow("Формат:", QLabel(format_name))
        
        # Глубина цвета
        depth = image.depth()
        info_layout.addRow("Глубина цвета:", QLabel(f"{depth} бит"))
        
        # Количество пикселей
        total_pixels = pixmap.width() * pixmap.height()
        info_layout.addRow("Всего пикселей:", QLabel(f"{total_pixels:,}"))
        
        layout.addLayout(info_layout)
        
        # Статистика цветов (для небольших изображений)
        if total_pixels <= 1000000:  # Только для изображений до 1 мегапикселя
            layout.addWidget(QLabel("\nСтатистика цветов:"))
            
            stats_layout = QFormLayout()
            
            # Вычисление статистики
            r_values, g_values, b_values = [], [], []
            
            for y in range(min(100, image.height())):  # Выборка для производительности
                for x in range(min(100, image.width())):
                    color = image.pixelColor(x, y)
                    r_values.append(color.red())
                    g_values.append(color.green())
                    b_values.append(color.blue())
            
            if r_values:
                stats_layout.addRow("Средний R:", QLabel(f"{sum(r_values) // len(r_values)}"))
                stats_layout.addRow("Средний G:", QLabel(f"{sum(g_values) // len(g_values)}"))
                stats_layout.addRow("Средний B:", QLabel(f"{sum(b_values) // len(b_values)}"))
            
            layout.addLayout(stats_layout)
        
        # Кнопка закрытия
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль интерфейса
    
    viewer = ImageViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
