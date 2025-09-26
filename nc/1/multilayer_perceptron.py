"""
Многослойный перцептрон (Multilayer Perceptron)

Ссылки на Wikipedia:
- Многослойный перцептрон: https://ru.wikipedia.org/wiki/Многослойный_перцептрон
- Искусственная нейронная сеть: https://ru.wikipedia.org/wiki/Искусственная_нейронная_сеть
- Алгоритм обратного распространения ошибки: https://ru.wikipedia.org/wiki/Метод_обратного_распространения_ошибки
- Функция активации: https://ru.wikipedia.org/wiki/Функция_активации
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


class MultilayerPerceptron:
    """
    Класс многослойного перцептрона
    
    Многослойный перцептрон - это тип искусственной нейронной сети прямого распространения,
    состоящий из нескольких слоев нейронов, где каждый слой полностью соединен со следующим.
    """
    
    def __init__(self, layer_sizes):
        """
        Инициализация сети
        
        Args:
            layer_sizes (list): Список с количеством нейронов в каждом слое
                               Например: [2, 4, 3, 1] означает:
                               - входной слой: 2 нейрона
                               - первый скрытый слой: 4 нейрона  
                               - второй скрытый слой: 3 нейрона
                               - выходной слой: 1 нейрон
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Инициализация весов и смещений
        # Веса инициализируются случайными значениями из нормального распределения
        # https://translated.turbopages.org/proxy_u/en-ru.ru.d9e0dd90-68d6889f-d5c0eb98-74722d776562/https/en.wikipedia.org/wiki/Weight_initialization
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Веса между слоем i и слоем i+1
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.5
            self.weights.append(weight_matrix)
            
            # Смещения для слоя i+1
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.biases.append(bias_vector)
    
    def sigmoid(self, x):
        """
        Сигмоидная функция активации
        
        Сигмоида - это S-образная функция, которая сжимает любое вещественное число
        в диапазон (0, 1), что делает её полезной для вероятностных интерпретаций.
        
        Формула: σ(x) = 1 / (1 + e^(-x))
        
        https://ru.wikipedia.org/wiki/Сигмоида
        """
        # Ограничиваем x для предотвращения переполнения
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Производная сигмоидной функции
        
        Используется при обратном распространении ошибки для вычисления градиентов.
        Производная сигмоиды: σ'(x) = σ(x) * (1 - σ(x))
        """
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward_propagation(self, X):
        """
        Прямое распространение сигнала через сеть
        
        Процесс, при котором входные данные проходят через все слои сети
        от входного к выходному слою.
        
        https://en.wikipedia.org/wiki/Feedforward_neural_network
        
        Args:
            X (numpy.ndarray): Входные данные
            
        Returns:
            list: Активации всех слоев
        """
        activations = [X]  # Активации каждого слоя
        
        for i in range(self.num_layers - 1):
            # Вычисляем взвешенную сумму: z = X * W + b
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            
            # Применяем функцию активации
            a = self.sigmoid(z)
            activations.append(a)
        
        return activations
    
    def backward_propagation(self, X, y, activations):
        """
        Обратное распространение ошибки
        
        Алгоритм для вычисления градиентов функции потерь по весам и смещениям.
        Работает путем распространения ошибки от выходного слоя к входному.
        
        https://ru.wikipedia.org/wiki/Метод_обратного_распространения_ошибки
        
        Args:
            X (numpy.ndarray): Входные данные
            y (numpy.ndarray): Целевые значения
            activations (list): Активации всех слоев
            
        Returns:
            tuple: Градиенты весов и смещений
        """
        m = X.shape[0]  # Количество примеров
        
        # Инициализируем градиенты
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Вычисляем ошибку выходного слоя
        # δ = (a - y) * σ'(z)
        output_error = activations[-1] - y
        deltas = [output_error]
        
        # Обратное распространение ошибки через скрытые слои
        for i in range(self.num_layers - 2, 0, -1):
            # Вычисляем z для текущего слоя
            z = np.dot(activations[i-1], self.weights[i-1]) + self.biases[i-1]
            
            # Ошибка текущего слоя
            delta = np.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(z)
            deltas.insert(0, delta)
        
        # Вычисляем градиенты
        for i in range(self.num_layers - 1):
            weight_gradients[i] = np.dot(activations[i].T, deltas[i]) / m
            bias_gradients[i] = np.mean(deltas[i], axis=0, keepdims=True)
        
        return weight_gradients, bias_gradients
    
    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=False):
        """
        Обучение нейронной сети
        
        Использует алгоритм градиентного спуска для минимизации функции потерь.
        
        https://ru.wikipedia.org/wiki/Градиентный_спуск
        
        Args:
            X (numpy.ndarray): Обучающие данные
            y (numpy.ndarray): Целевые значения
            epochs (int): Количество эпох обучения
            learning_rate (float): Скорость обучения
            verbose (bool): Выводить ли информацию о процессе обучения
            
        Returns:
            list: История потерь
        """
        loss_history = []
        
        for epoch in range(epochs):
            # Прямое распространение
            activations = self.forward_propagation(X)
            
            # Вычисление функции потерь (среднеквадратичная ошибка)
            # MSE = (1/2m) * Σ(y_pred - y_true)²
            loss = np.mean((activations[-1] - y) ** 2)
            loss_history.append(loss)
            
            # Обратное распространение
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations)
            
            # Обновление весов и смещений
            # w = w - α * ∇w (где α - скорость обучения)
            for i in range(self.num_layers - 1):
                self.weights[i] -= learning_rate * weight_gradients[i]
                self.biases[i] -= learning_rate * bias_gradients[i]
            
            # Вывод информации о прогрессе
            if verbose and epoch % 100 == 0:
                print(f"Эпоха {epoch}, Потеря: {loss:.6f}")
        
        return loss_history
    
    def predict(self, X):
        """
        Предсказание для новых данных
        
        Args:
            X (numpy.ndarray): Входные данные
            
        Returns:
            numpy.ndarray: Предсказания сети
        """
        activations = self.forward_propagation(X)
        return activations[-1]
    
    def get_network_info(self):
        """Получить информацию о структуре сети"""
        info = f"Структура сети: {' -> '.join(map(str, self.layer_sizes))}\n"
        info += f"Количество слоев: {self.num_layers}\n"
        
        total_params = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            layer_params = w.size + b.size
            total_params += layer_params
            info += f"Слой {i+1}: {w.shape[0]} -> {w.shape[1]}, параметров: {layer_params}\n"
        
        info += f"Общее количество параметров: {total_params}"
        return info


class MLPInterface:
    """Графический интерфейс для работы с многослойным перцептроном"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Многослойный перцептрон")
        self.root.geometry("800x600")
        
        self.mlp = None
        self.training_data = None
        self.training_labels = None
        
        self.create_widgets()
        self.generate_sample_data()
    
    def create_widgets(self):
        """Создание элементов интерфейса"""
        
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка сети
        network_frame = ttk.LabelFrame(main_frame, text="Настройка сети", padding="5")
        network_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(network_frame, text="Структура сети (через запятую):").grid(row=0, column=0, sticky=tk.W)
        self.structure_var = tk.StringVar(value="2,4,3,1")
        ttk.Entry(network_frame, textvariable=self.structure_var, width=20).grid(row=0, column=1, padx=5)
        
        ttk.Button(network_frame, text="Создать сеть", command=self.create_network).grid(row=0, column=2, padx=5)
        
        # Параметры обучения
        training_frame = ttk.LabelFrame(main_frame, text="Параметры обучения", padding="5")
        training_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(training_frame, text="Эпохи:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="1000")
        ttk.Entry(training_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(training_frame, text="Скорость обучения:").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        self.lr_var = tk.StringVar(value="0.1")
        ttk.Entry(training_frame, textvariable=self.lr_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Button(training_frame, text="Обучить", command=self.train_network).grid(row=0, column=4, padx=5)
        
        # Тестирование
        test_frame = ttk.LabelFrame(main_frame, text="Тестирование", padding="5")
        test_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(test_frame, text="Входные данные (через запятую):").grid(row=0, column=0, sticky=tk.W)
        self.input_var = tk.StringVar(value="0.17,0.33")
        ttk.Entry(test_frame, textvariable=self.input_var, width=20).grid(row=0, column=1, padx=5)
        
        ttk.Button(test_frame, text="Предсказать", command=self.predict).grid(row=0, column=2, padx=5)
        
        # Результаты
        results_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=70)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def log_message(self, message):
        """Добавить сообщение в лог"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def generate_sample_data(self):
        """
        Генерация примера данных для обучения
        
        Создаем задачу классификации состояния воды по температуре и давлению.
        
        Физика процесса:
        - При низких температурах вода замерзает (лед)
        - При комнатной температуре остается жидкой (вода)  
        - При высоких температурах превращается в пар
        - Давление влияет на точки фазовых переходов
        
        https://ru.wikipedia.org/wiki/Фазовые_переходы
        """
        # Данные для классификации температуры воды
        # Вход: [температура_C, давление_атм] -> Выход: [состояние] (0=лед, 0.5=вода, 1=пар)
        self.training_data = np.array([
            [-10, 1.0],  # -10°C, 1 атм
            [0, 1.0],    # 0°C, 1 атм  
            [25, 1.0],   # 25°C, 1 атм
            [50, 1.0],   # 50°C, 1 атм
            [75, 1.0],   # 75°C, 1 атм
            [100, 1.0],  # 100°C, 1 атм
            [120, 1.0],  # 120°C, 1 атм
            [0, 0.5],    # 0°C, 0.5 атм
            [100, 0.5],  # 100°C, 0.5 атм
            [100, 2.0],  # 100°C, 2 атм
        ])
        
        # Нормализация входных данных для лучшего обучения
        self.training_data[:, 0] = self.training_data[:, 0] / 150.0  # температура от -10 до 150
        self.training_data[:, 1] = self.training_data[:, 1] / 3.0    # давление от 0 до 3 атм
        
        self.training_labels = np.array([
            [0.0],   # лед
            [0.1],   # почти лед
            [0.5],   # вода
            [0.6],   # теплая вода
            [0.8],   # горячая вода
            [0.9],   # почти пар
            [1.0],   # пар
            [0.1],   # лед при низком давлении
            [0.8],   # горячая вода при низком давлении
            [0.5],   # вода при высоком давлении
        ])
        
        self.log_message("Сгенерированы тестовые данные (Состояние воды):")
        self.log_message("Вход: [температура_норм, давление_норм] -> Выход: состояние")
        self.log_message("0.0 = лед, 0.5 = вода, 1.0 = пар")
        self.log_message("Примеры: [-10°C,1атм] -> лед, [25°C,1атм] -> вода, [120°C,1атм] -> пар")
        self.log_message("-" * 50)
    
    def create_network(self):
        """Создание новой нейронной сети"""
        try:
            # Парсинг структуры сети
            structure_str = self.structure_var.get().strip()
            layer_sizes = [int(x.strip()) for x in structure_str.split(',')]
            
            if len(layer_sizes) < 2:
                raise ValueError("Сеть должна иметь минимум 2 слоя")
            
            # Создание сети
            self.mlp = MultilayerPerceptron(layer_sizes)
            
            self.log_message(f"Создана новая сеть:")
            self.log_message(self.mlp.get_network_info())
            self.log_message("-" * 50)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания сети: {str(e)}")
    
    def train_network(self):
        """Обучение нейронной сети"""
        if self.mlp is None:
            messagebox.showwarning("Предупреждение", "Сначала создайте сеть!")
            return
        
        try:
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.lr_var.get())
            
            self.log_message(f"Начинаем обучение...")
            self.log_message(f"Эпохи: {epochs}, Скорость обучения: {learning_rate}")
            
            # Запуск обучения в отдельном потоке
            def train_thread():
                loss_history = self.mlp.train(
                    self.training_data, 
                    self.training_labels, 
                    epochs=epochs, 
                    learning_rate=learning_rate,
                    verbose=True
                )
                
                # Результаты обучения
                final_loss = loss_history[-1]
                self.log_message(f"Обучение завершено!")
                self.log_message(f"Финальная потеря: {final_loss:.6f}")
                
                # Тестирование на обучающих данных
                predictions = self.mlp.predict(self.training_data)
                self.log_message("\nРезультаты на обучающих данных:")
                for i, (input_data, target, pred) in enumerate(zip(self.training_data, self.training_labels, predictions)):
                    self.log_message(f"Вход: {input_data}, Цель: {target[0]:.3f}, Предсказание: {pred[0]:.3f}")
                
                self.log_message("-" * 50)
            
            threading.Thread(target=train_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")
    
    def predict(self):
        """Предсказание для введенных данных"""
        if self.mlp is None:
            messagebox.showwarning("Предупреждение", "Сначала создайте и обучите сеть!")
            return
        
        try:
            # Парсинг входных данных
            input_str = self.input_var.get().strip()
            input_data = [float(x.strip()) for x in input_str.split(',')]
            
            if len(input_data) != self.mlp.layer_sizes[0]:
                raise ValueError(f"Ожидается {self.mlp.layer_sizes[0]} входных значений")
            
            # Предсказание
            input_array = np.array([input_data])
            prediction = self.mlp.predict(input_array)
            
            self.log_message(f"Предсказание для входа {input_data}:")
            self.log_message(f"Выход: {prediction[0]}")
            self.log_message("-" * 30)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка предсказания: {str(e)}")


def main():
    """Главная функция"""
    # Создание и запуск GUI
    root = tk.Tk()
    app = MLPInterface(root)
    
    # Добавляем информацию о программе
    app.log_message("=== МНОГОСЛОЙНЫЙ ПЕРЦЕПТРОН ===")
    app.log_message("")
    app.log_message("Инструкция:")
    app.log_message("1. Задайте структуру сети (например: 2,4,3,1)")
    app.log_message("2. Нажмите 'Создать сеть'")
    app.log_message("3. Настройте параметры обучения")
    app.log_message("4. Нажмите 'Обучить'")
    app.log_message("5. Введите данные для предсказания и нажмите 'Предсказать'")
    app.log_message("")
    app.log_message("По умолчанию используются данные состояния воды для демонстрации")
    app.log_message("=" * 50)
    
    root.mainloop()


if __name__ == "__main__":
    main()
