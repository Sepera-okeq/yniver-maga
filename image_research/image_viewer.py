from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QGraphicsRectItem, QGraphicsPathItem
from PyQt6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QPainterPath
from PyQt6.QtCore import Qt, QPointF, QRectF

class ImageViewer(QGraphicsView):
    """
    Виджет для отображения и взаимодействия с изображением.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)

        self.current_image = None
        self.undo_stack = []

        self.selection_mode = 'rect'  # 'rect' or 'lasso'
        self.selection_item = None # Can be QGraphicsRectItem or QGraphicsPathItem
        self.start_pos = None
        self.lasso_path = None

    def set_selection_mode(self, mode: str):
        """
        Устанавливает режим выделения ('rect' или 'lasso').
        """
        if mode in ['rect', 'lasso']:
            self.selection_mode = mode
            self.clear_selection()

    def clear_selection(self):
        """
        Очищает текущее выделение.
        """
        if self.selection_item:
            self.scene.removeItem(self.selection_item)
            self.selection_item = None
            self.lasso_path = None

    def set_image(self, image: QImage):
        """
        Устанавливает новое изображение для отображения.
        """
        if self.current_image:
            self.undo_stack.append(self.current_image.copy())
            # Ограничим стек отмены, чтобы не занимать слишком много памяти
            if len(self.undo_stack) > 10:
                self.undo_stack.pop(0)
        
        self.current_image = image
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def get_image(self) -> QImage:
        """
        Возвращает текущее изображение.
        """
        return self.current_image

    def undo(self):
        """
        Отменяет последнее действие.
        """
        if self.undo_stack:
            previous_image = self.undo_stack.pop()
            self.current_image = previous_image
            pixmap = QPixmap.fromImage(previous_image)
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            return True
        return False

    def mouseMoveEvent(self, event):
        """
        Обрабатывает движение мыши для отображения координат и цвета пикселя.
        """
        if self.current_image:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())

            if 0 <= x < self.current_image.width() and 0 <= y < self.current_image.height():
                color = self.current_image.pixelColor(x, y)
                self.window().pixel_info_label.setText(f"X: {x}, Y: {y} | R: {color.red()}, G: {color.green()}, B: {color.blue()}")
            else:
                self.window().pixel_info_label.setText("")
        
        if self.start_pos:
            pos = self.mapToScene(event.pos())
            if self.selection_mode == 'rect':
                rect = QRectF(self.start_pos, pos).normalized()
                if self.selection_item:
                    self.selection_item.setRect(rect)
            elif self.selection_mode == 'lasso':
                if self.lasso_path:
                    self.lasso_path.lineTo(pos)
                    self.selection_item.setPath(self.lasso_path)

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        """
        Обрабатывает нажатие кнопки мыши для начала выделения области.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.clear_selection()
            self.start_pos = self.mapToScene(event.pos())
            
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)

            if self.selection_mode == 'rect':
                self.selection_item = QGraphicsRectItem(QRectF(self.start_pos, self.start_pos))
                self.selection_item.setPen(pen)
                self.scene.addItem(self.selection_item)
            elif self.selection_mode == 'lasso':
                self.lasso_path = QPainterPath()
                self.lasso_path.moveTo(self.start_pos)
                self.selection_item = QGraphicsPathItem(self.lasso_path)
                self.selection_item.setPen(pen)
                self.scene.addItem(self.selection_item)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Обрабатывает отпускание кнопки мыши для завершения выделения области.
        """
        if event.button() == Qt.MouseButton.LeftButton and self.start_pos:
            if self.selection_mode == 'lasso' and self.lasso_path:
                self.lasso_path.closeSubpath()
                self.selection_item.setPath(self.lasso_path)
            self.start_pos = None

        super().mouseReleaseEvent(event)
    
    def get_selection(self):
        """
        Возвращает выделенную область (QRectF или QPainterPath).
        """
        if isinstance(self.selection_item, QGraphicsRectItem):
            return self.selection_item.rect()
        elif isinstance(self.selection_item, QGraphicsPathItem):
            return self.selection_item.path()
        return None
