import numpy as np
from PyQt6.QtGui import QImage, QColor, QTransform, QPainterPath
from PyQt6.QtCore import Qt, QRectF, QPointF

def _get_pixel_iterator(width, height, selection=None):
    """
    Вспомогательный итератор, который возвращает координаты пикселей
    для обработки в зависимости от наличия выделения.
    """
    if selection:
        if isinstance(selection, QRectF):
            # Итерация по прямоугольнику
            x_start, y_start = int(selection.left()), int(selection.top())
            x_end, y_end = int(selection.right()), int(selection.bottom())
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if 0 <= x < width and 0 <= y < height:
                        yield x, y
        elif isinstance(selection, QPainterPath):
            # Для оптимизации итерируем по ограничивающему прямоугольнику контура
            bounding_rect = selection.boundingRect()
            x_start, y_start = int(bounding_rect.left()), int(bounding_rect.top())
            x_end, y_end = int(bounding_rect.right()), int(bounding_rect.bottom())
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if 0 <= x < width and 0 <= y < height and selection.contains(QPointF(x, y)):
                        yield x, y
    else:
        # Если выделения нет, итерация по всему изображению
        for y in range(height):
            for x in range(width):
                yield x, y

def to_grayscale(image: QImage, selection=None) -> QImage:
    """
    Преобразует цветное изображение в градации серого.
    Применяется к выделенной области, если она есть.
    """
    if image.isNull():
        return QImage()

    new_image = image.copy()
    
    for x, y in _get_pixel_iterator(new_image.width(), new_image.height(), selection):
        color = QColor(new_image.pixel(x, y))
        gray_value = int(0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())
        new_image.setPixelColor(x, y, QColor(gray_value, gray_value, gray_value))

    return new_image

def image_to_byte_array(image: QImage) -> bytearray:
    """
    Преобразует QImage в массив байтов.
    """
    if image.isNull():
        return bytearray()
    
    if image.format() != QImage.Format.Format_Grayscale8:
        image = image.convertToFormat(QImage.Format.Format_Grayscale8)

    bits = image.bits()
    size = image.width() * image.height()
    return bytearray(bits.asarray(size))

def byte_array_to_image(byte_arr: bytearray, width: int, height: int) -> QImage:
    """
    Преобразует массив байтов в QImage.
    """
    if not byte_arr or width <= 0 or height <= 0:
        return QImage()

    expected_size = width * height
    if len(byte_arr) != expected_size:
        raise ValueError(f"Длина массива байтов ({len(byte_arr)}) не соответствует ожидаемому размеру ({expected_size})")

    image = QImage(bytes(byte_arr), width, height, QImage.Format.Format_Grayscale8)
    return image.convertToFormat(QImage.Format.Format_RGB32)

def calculate_statistics(image: QImage, selection=None):
    """
    Вычисляет статистику для пикселей в заданной области.
    """
    if image.isNull():
        return None

    img_gray = image if image.isGrayscale() else image.convertToFormat(QImage.Format.Format_Grayscale8)
    
    pixels = []
    for x, y in _get_pixel_iterator(img_gray.width(), img_gray.height(), selection):
        pixels.append(img_gray.pixelColor(x, y).red())

    if not pixels:
        return None

    pixels = np.array(pixels)
    return {
        'min': np.min(pixels),
        'max': np.max(pixels),
        'mean': np.mean(pixels),
        'std': np.std(pixels)
    }

def get_histogram_data(image: QImage, selection=None):
    """
    Возвращает данные для гистограммы для заданной области.
    """
    if image.isNull():
        return None

    img_gray = image if image.isGrayscale() else image.convertToFormat(QImage.Format.Format_Grayscale8)
    
    pixels = []
    for x, y in _get_pixel_iterator(img_gray.width(), img_gray.height(), selection):
        pixels.append(img_gray.pixelColor(x, y).red())

    return pixels

def linear_contrast(image: QImage, selection=None) -> QImage:
    """
    Выполняет линейное контрастирование для заданной области.
    """
    if image.isNull():
        return QImage()

    new_image = image.copy()
    
    stats = calculate_statistics(new_image, selection)
    if not stats or stats['min'] == stats['max']:
        return image

    m, M = stats['min'], stats['max']
    
    for x, y in _get_pixel_iterator(new_image.width(), new_image.height(), selection):
        color = QColor(new_image.pixel(x, y))
        
        r = int(255 * (color.red() - m) / (M - m))
        g = int(255 * (color.green() - m) / (M - m))
        b = int(255 * (color.blue() - m) / (M - m))
        
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        new_image.setPixelColor(x, y, QColor(r, g, b))

    return new_image

def smooth_image(image: QImage, radius: int, selection=None) -> QImage:
    """
    Сглаживает изображение с использованием квадратной окрестности заданного радиуса.
    """
    if image.isNull() or radius <= 0:
        return image

    new_image = image.copy()
    source_image = image.copy()

    for x, y in _get_pixel_iterator(source_image.width(), source_image.height(), selection):
        r_total, g_total, b_total = 0, 0, 0
        count = 0
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                nx, ny = x + i, y + j
                if 0 <= nx < source_image.width() and 0 <= ny < source_image.height():
                    color = QColor(source_image.pixel(nx, ny))
                    r_total += color.red()
                    g_total += color.green()
                    b_total += color.blue()
                    count += 1
        
        if count > 0:
            r = int(r_total / count)
            g = int(g_total / count)
            b = int(b_total / count)
            new_image.setPixelColor(x, y, QColor(r, g, b))

    return new_image

def posterize_image(image: QImage, levels: int, selection=None) -> QImage:
    """
    Создает изображение с кусочно-постоянными амплитудами (постеризация).
    """
    if image.isNull() or levels <= 1:
        return image

    new_image = image.copy()
    
    step = 256 / levels
    lookup_table = [int(int(i / step) * (255 / (levels - 1))) for i in range(256)]

    for x, y in _get_pixel_iterator(new_image.width(), new_image.height(), selection):
        color = QColor(new_image.pixel(x, y))
        
        r = lookup_table[color.red()]
        g = lookup_table[color.green()]
        b = lookup_table[color.blue()]
        
        new_image.setPixelColor(x, y, QColor(r, g, b))

    return new_image

def scale_image_nearest(image: QImage, scale_factor: float) -> QImage:
    """
    Масштабирует изображение методом ближайшего соседа.
    """
    if image.isNull() or scale_factor <= 0:
        return image

    new_width = int(image.width() * scale_factor)
    new_height = int(image.height() * scale_factor)
    new_image = QImage(new_width, new_height, image.format())

    for y in range(new_height):
        for x in range(new_width):
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            new_image.setPixel(x, y, image.pixel(orig_x, orig_y))

    return new_image

def scale_image_bilinear(image: QImage, scale_factor: float) -> QImage:
    """
    Масштабирует изображение методом билинейной интерполяции.
    """
    if image.isNull() or scale_factor <= 0:
        return image

    return image.scaled(int(image.width() * scale_factor), int(image.height() * scale_factor), 
                        Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

def add_gaussian_noise(image: QImage, sigma: float) -> QImage:
    """
    Добавляет аддитивный белый гауссовский шум к изображению.
    """
    if image.isNull():
        return QImage()

    new_image = image.copy()
    width, height = new_image.width(), new_image.height()
    noise = np.random.normal(0, sigma, (height, width)).astype(np.int32)

    for y in range(height):
        for x in range(width):
            original_color = QColor(new_image.pixel(x, y))
            r = max(0, min(255, original_color.red() + noise[y, x]))
            g = max(0, min(255, original_color.green() + noise[y, x]))
            b = max(0, min(255, original_color.blue() + noise[y, x]))
            new_image.setPixelColor(x, y, QColor(r, g, b))
            
    return new_image

def reduce_noise_median(image: QImage, radius: int) -> QImage:
    """
    Уменьшает шум с помощью медианного фильтра.
    """
    if image.isNull() or radius <= 0:
        return image

    new_image = image.copy()
    width, height = new_image.width(), new_image.height()

    for y in range(height):
        for x in range(width):
            pixels_r, pixels_g, pixels_b = [], [], []
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    nx, ny = x + i, y + j
                    if 0 <= nx < width and 0 <= ny < height:
                        color = QColor(image.pixel(nx, ny))
                        pixels_r.append(color.red())
                        pixels_g.append(color.green())
                        pixels_b.append(color.blue())
            
            if pixels_r:
                median_r = int(np.median(pixels_r))
                median_g = int(np.median(pixels_g))
                median_b = int(np.median(pixels_b))
                new_image.setPixelColor(x, y, QColor(median_r, median_g, median_b))

    return new_image

def get_pixel_value(image: QImage, x: int, y: int) -> QColor:
    """
    Возвращает цвет пикселя по заданным координатам.
    """
    if image.isNull() or not (0 <= x < image.width() and 0 <= y < image.height()):
        return None
    return image.pixelColor(x, y)

def set_pixel_value(image: QImage, x: int, y: int, color: QColor) -> QImage:
    """
    Устанавливает цвет пикселя по заданным координатам.
    """
    if image.isNull() or not (0 <= x < image.width() and 0 <= y < image.height()):
        return image
    
    new_image = image.copy()
    new_image.setPixelColor(x, y, color)
    return new_image

def generate_uniform_noise_image(width: int, height: int, low: int, high: int) -> QImage:
    """
    Генерирует изображение с равномерным шумом.
    """
    if width <= 0 or height <= 0:
        return QImage()
    
    image = QImage(width, height, QImage.Format.Format_RGB32)
    for y in range(height):
        for x in range(width):
            val = np.random.randint(low, high + 1)
            image.setPixel(x, y, (val << 16) | (val << 8) | val)
    return image

def generate_normal_noise_image(width: int, height: int, mean: float, std: float) -> QImage:
    """
    Генерирует изображение с нормальным шумом.
    """
    if width <= 0 or height <= 0:
        return QImage()

    image = QImage(width, height, QImage.Format.Format_RGB32)
    noise = np.random.normal(mean, std, (height, width))
    
    for y in range(height):
        for x in range(width):
            val = int(noise[y, x])
            val = max(0, min(255, val))
            image.setPixel(x, y, (val << 16) | (val << 8) | val)
    return image

def rotate_image(image: QImage, angle: float) -> QImage:
    """
    Поворачивает изображение на заданный угол.
    """
    if image.isNull():
        return QImage()

    transform = QTransform().rotate(angle)
    return image.transformed(transform, Qt.TransformationMode.SmoothTransformation)

def generate_sliding_sum_image(width: int, height: int, radius: int, mean: float, std: float) -> QImage:
    """
    Генерирует изображение однородной сцены методом скользящего суммирования.
    """
    if width <= 0 or height <= 0 or radius <= 0:
        return QImage()

    source_image = generate_normal_noise_image(width, height, mean, std)
    return smooth_image(source_image, radius)
