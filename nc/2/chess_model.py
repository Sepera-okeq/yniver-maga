import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import sys

# Ценности фигур
PIECE_VALUES = {
    'pawn_w': 1, 'pawn_b': 1,
    'knight_w': 3, 'knight_b': 3,
    'bishop_w': 3, 'bishop_b': 3,
    'rook_w': 5, 'rook_b': 5,
    'queen_w': 9, 'queen_b': 9,
    'king_w': 0, 'king_b': 0
}

def load_data(data_dir):
    """
    Загружает изображения и метки из датасета.
    """
    images = []
    labels_count = []  # количество фигур
    labels_more_pieces = []  # у кого больше фигур: 0 - белые, 1 - черные, 2 - равенство
    labels_more_value = []  # у кого больше ценность: 0 - белые, 1 - черные, 2 - равенство

    files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    total_files = len(files)
    print(f"Начинаем загрузку {total_files} изображений...")

    for i, filename in enumerate(files):
        if i % 100 == 0:
            print(f"Обработано {i}/{total_files} изображений...")

        # Загружаем изображение
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Изменяем размер для модели
        img = img / 255.0  # Нормализация
        images.append(img)

        # Загружаем JSON с конфигурацией
        json_path = os.path.join(data_dir, filename.replace('.jpg', '.json'))
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Вычисляем метки
        pieces = data['config']
        white_count = 0
        black_count = 0
        white_value = 0
        black_value = 0

        for piece in pieces.values():
            if piece.endswith('_w'):
                white_count += 1
                white_value += PIECE_VALUES[piece]
            elif piece.endswith('_b'):
                black_count += 1
                black_value += PIECE_VALUES[piece]

        total_count = white_count + black_count
        labels_count.append(total_count)

        # У кого больше фигур
        if white_count > black_count:
            labels_more_pieces.append(0)  # белые
        elif black_count > white_count:
            labels_more_pieces.append(1)  # черные
        else:
            labels_more_pieces.append(2)  # равенство

        # У кого больше ценность
        if white_value > black_value:
            labels_more_value.append(0)  # белые
        elif black_value > white_value:
            labels_more_value.append(1)  # черные
        else:
            labels_more_value.append(2)  # равенство

    print(f"Загрузка завершена. Всего {len(images)} изображений.")
    return np.array(images), {
        'count': np.array(labels_count),
        'more_pieces': np.array(labels_more_pieces),
        'more_value': np.array(labels_more_value)
    }

def build_model(num_classes_count, num_classes_more):
    """
    Строит упрощенную и более эффективную модель CNN для трех задач.
    """
    input_layer = layers.Input(shape=(224, 224, 3))

    # Сверточные слои
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # GlobalAveragePooling2D для резкого уменьшения количества параметров
    x = layers.GlobalAveragePooling2D()(x)

    # Плотный слой с Dropout для регуляризации
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Выходы для трех задач
    output_count = layers.Dense(num_classes_count, activation='softmax', name='count')(x)
    output_more_pieces = layers.Dense(num_classes_more, activation='softmax', name='more_pieces')(x)
    output_more_value = layers.Dense(num_classes_more, activation='softmax', name='more_value')(x)

    model = models.Model(inputs=input_layer, outputs=[output_count, output_more_pieces, output_more_value])

    # Оптимизатор Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss={
            'count': 'sparse_categorical_crossentropy',
            'more_pieces': 'sparse_categorical_crossentropy',
            'more_value': 'sparse_categorical_crossentropy'
        },
        metrics={
            'count': ['accuracy'],
            'more_pieces': ['accuracy'],
            'more_value': ['accuracy']
        }
    )

    return model

def visualize_predictions(model, X_test, y_test_count, y_test_pieces, y_test_value, le_count, num_images=5):
    """
    Визуализирует предсказания модели на нескольких тестовых изображениях.
    """
    print("Шаг 8: Визуализация предсказаний...")
    
    # Получаем предсказания
    predictions = model.predict(X_test)
    pred_count = predictions[0]
    pred_pieces = predictions[1]
    pred_value = predictions[2]

    # Декодируем реальные и предсказанные метки
    real_count = le_count.inverse_transform(y_test_count)
    predicted_count_indices = np.argmax(pred_count, axis=1)
    predicted_count = le_count.inverse_transform(predicted_count_indices)

    predicted_pieces_indices = np.argmax(pred_pieces, axis=1)
    predicted_value_indices = np.argmax(pred_value, axis=1)

    # Словарь для расшифровки меток
    more_map = {0: 'Белые', 1: 'Черные', 2: 'Равенство'}

    # Выбираем случайные изображения для отображения
    indices = np.random.choice(len(X_test), num_images, replace=False)

    plt.figure(figsize=(15, 5 * num_images))
    
    for i, idx in enumerate(indices):
        # Изображение
        img = X_test[idx]

        # Задача 1: Количество фигур
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title(f"Кол-во фигур\nРеально: {real_count[idx]}\nПредсказано: {predicted_count[idx]}")
        plt.axis('off')

        # Задача 2: У кого больше фигур
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        real_label_pieces = more_map[y_test_pieces[idx]]
        pred_label_pieces = more_map[predicted_pieces_indices[idx]]
        plt.title(f"Больше фигур\nРеально: {real_label_pieces}\nПредсказано: {pred_label_pieces}")
        plt.axis('off')

        # Задача 3: У кого больше ценность
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        real_label_value = more_map[y_test_value[idx]]
        pred_label_value = more_map[predicted_value_indices[idx]]
        plt.title(f"Больше ценность\nРеально: {real_label_value}\nПредсказано: {pred_label_value}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

def main():
    # Перенаправляем вывод в файл лога
    log_file = open('training_log.txt', 'w')
    sys.stdout = log_file

    # Шаг 1: Загружаем данные
    print("Шаг 1: Загружаем данные...")
    data_dir = 'data/data'
    images, labels = load_data(data_dir)

    # Шаг 2: Преобразуем метки и разделяем данные
    print("Шаг 2: Преобразуем метки и разделяем данные...")
    
    # Преобразуем метки 'count' в последовательные индексы
    le_count = LabelEncoder()
    y_count_encoded = le_count.fit_transform(labels['count'])
    
    # Разделяем на обучающую и тестовую выборки один раз
    X_train, X_test, y_train_count, y_test_count, y_train_pieces, y_test_pieces, y_train_value, y_test_value = train_test_split(
        images, 
        y_count_encoded, 
        labels['more_pieces'], 
        labels['more_value'], 
        test_size=0.2, 
        random_state=42
    )

    # Шаг 3: Строим модель
    print("Шаг 3: Строим модель...")
    num_classes_count = len(np.unique(y_count_encoded))
    num_classes_more = 3  # белые, черные, равенство
    
    print(f"Количество классов для количества фигур: {num_classes_count}")
    print(f"Количество классов для сравнения: {num_classes_more}")
    
    model = build_model(num_classes_count, num_classes_more)

    # Выводим summary модели
    print("\nАрхитектура модели:")
    model.summary(print_fn=lambda x: print(x, file=sys.stdout))

    # Шаг 4: Обучаем модель
    print("Шаг 4: Обучаем модель...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Больше терпения для сходимости
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train,
        {
            'count': y_train_count,
            'more_pieces': y_train_pieces,
            'more_value': y_train_value
        },
        epochs=50,
        batch_size=32,  # Увеличиваем батч для ускорения и стабильности
        validation_data=(
            X_test,
            {
                'count': y_test_count,
                'more_pieces': y_test_pieces,
                'more_value': y_test_value
            }
        ),
        callbacks=callbacks
    )

    # Шаг 5: Оцениваем модель
    print("Шаг 5: Оцениваем модель...")
    results = model.evaluate(
        X_test,
        {
            'count': y_test_count,
            'more_pieces': y_test_pieces,
            'more_value': y_test_value
        }
    )

    print(f"Общая потеря: {results[0]:.4f}")
    print(f"Потеря для количества фигур: {results[1]:.4f}")
    print(f"Потеря для 'у кого больше фигур': {results[2]:.4f}")
    print(f"Потеря для 'у кого больше ценность': {results[3]:.4f}")
    print(f"Точность для количества фигур: {results[4]:.4f}")
    print(f"Точность для 'у кого больше фигур': {results[5]:.4f}")
    print(f"Точность для 'у кого больше ценность': {results[6]:.4f}")

    # Сохраняем модель
    print("Шаг 6: Сохраняем модель...")
    model.save('chess_model.h5')

    # Графики обучения
    print("Шаг 7: Строим графики обучения...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['count_accuracy'], label='train')
    plt.plot(history.history['val_count_accuracy'], label='val')
    plt.title('Точность: количество фигур')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['more_pieces_accuracy'], label='train')
    plt.plot(history.history['val_more_pieces_accuracy'], label='val')
    plt.title('Точность: у кого больше фигур')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['more_value_accuracy'], label='train')
    plt.plot(history.history['val_more_value_accuracy'], label='val')
    plt.title('Точность: у кого больше ценность')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Шаг 8: Визуализация предсказаний
    visualize_predictions(model, X_test, y_test_count, y_test_pieces, y_test_value, le_count)

    print("Обучение завершено!")

if __name__ == '__main__':
    main()

# предобработка
# классификация наличия в клетке (или регрессию)