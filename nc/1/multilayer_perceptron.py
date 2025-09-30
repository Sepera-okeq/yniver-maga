"""
Многослойный перцептрон (Multilayer Perceptron)

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
from matplotlib.figure import Figure
import matplotlib.patches as patches
import threading
import seaborn as sns


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
        self.root.title("Многослойный перцептрон - Визуализация")
        self.root.geometry("1400x900")
        
        self.mlp = None
        self.training_data = None
        self.training_labels = None
        self.loss_history = []
        
        self.create_widgets()
        self.generate_sample_data()
    
    def create_widgets(self):
        """Создание элементов интерфейса"""
        
        # Главный фрейм с двумя колонками
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Левая панель управления
        control_frame = ttk.Frame(main_frame, width=400)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.grid_propagate(False)
        
        # Правая панель с графиками
        viz_frame = ttk.Frame(main_frame)
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === ЛЕВАЯ ПАНЕЛЬ ===
        
        # Настройка сети
        network_frame = ttk.LabelFrame(control_frame, text="Настройка сети", padding="5")
        network_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(network_frame, text="Структура сети:").grid(row=0, column=0, sticky=tk.W)
        self.structure_var = tk.StringVar(value="2,4,3,1")
        ttk.Entry(network_frame, textvariable=self.structure_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Button(network_frame, text="Создать сеть", command=self.create_network).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Параметры обучения
        training_frame = ttk.LabelFrame(control_frame, text="Параметры обучения", padding="5")
        training_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(training_frame, text="Эпохи:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="1000")
        ttk.Entry(training_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(training_frame, text="Скорость обучения:").grid(row=1, column=0, sticky=tk.W)
        self.lr_var = tk.StringVar(value="0.1")
        ttk.Entry(training_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Button(training_frame, text="Обучить", command=self.train_network).grid(row=2, column=0, columnspan=2, pady=5)
        
        # Тестирование
        test_frame = ttk.LabelFrame(control_frame, text="Тестирование", padding="5")
        test_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(test_frame, text="Входные данные:").grid(row=0, column=0, sticky=tk.W)
        self.input_var = tk.StringVar(value="0.17,0.33")
        ttk.Entry(test_frame, textvariable=self.input_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Button(test_frame, text="Предсказать", command=self.predict).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Кнопки визуализации
        viz_buttons_frame = ttk.LabelFrame(control_frame, text="Визуализация", padding="5")
        viz_buttons_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(viz_buttons_frame, text="График потерь", command=self.plot_loss).grid(row=0, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(viz_buttons_frame, text="Структура сети", command=self.plot_network).grid(row=1, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(viz_buttons_frame, text="Предсказания vs Цели", command=self.plot_predictions).grid(row=2, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(viz_buttons_frame, text="Функция активации", command=self.plot_activation).grid(row=3, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(viz_buttons_frame, text="Веса сети", command=self.plot_weights).grid(row=4, column=0, pady=2, sticky=(tk.W, tk.E))
        
        # Результаты
        results_frame = ttk.LabelFrame(control_frame, text="Результаты", padding="5")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, width=45)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === ПРАВАЯ ПАНЕЛЬ ===
        
        # Notebook для вкладок с графиками
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создание вкладок для графиков
        self.create_plot_tabs()
        
        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        control_frame.rowconfigure(4, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        viz_buttons_frame.columnconfigure(0, weight=1)
    
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
    
    def create_plot_tabs(self):
        """Создание вкладок для графиков"""
        # Вкладка для графика потерь
        self.loss_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.loss_frame, text="График потерь")
        
        # Вкладка для структуры сети
        self.network_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.network_frame, text="Структура сети")
        
        # Вкладка для предсказаний
        self.pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_frame, text="Предсказания")
        
        # Вкладка для функции активации
        self.activation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.activation_frame, text="Функция активации")
        
        # Вкладка для весов
        self.weights_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.weights_frame, text="Веса сети")

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
                self.loss_history = self.mlp.train(
                    self.training_data, 
                    self.training_labels, 
                    epochs=epochs, 
                    learning_rate=learning_rate,
                    verbose=True
                )
                
                # Результаты обучения
                final_loss = self.loss_history[-1]
                self.log_message(f"Обучение завершено!")
                self.log_message(f"Финальная потеря: {final_loss:.6f}")
                
                # Тестирование на обучающих данных
                predictions = self.mlp.predict(self.training_data)
                self.log_message("\nРезультаты на обучающих данных:")
                for i, (input_data, target, pred) in enumerate(zip(self.training_data, self.training_labels, predictions)):
                    self.log_message(f"Вход: {input_data}, Цель: {target[0]:.3f}, Предсказание: {pred[0]:.3f}")
                
                # Автоматически показать график потерь после обучения
                self.plot_loss()
                
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

    def plot_loss(self):
        """График потерь во время обучения"""
        if not self.loss_history:
            messagebox.showwarning("Предупреждение", "Сначала обучите сеть!")
            return
        
        # Очистка предыдущего графика
        for widget in self.loss_frame.winfo_children():
            widget.destroy()
        
        # Создание фигуры
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # График потерь
        epochs = range(len(self.loss_history))
        ax.plot(epochs, self.loss_history, 'b-', linewidth=2, label='Потеря обучения')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Потеря (MSE)')
        ax.set_title('График потерь во время обучения')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Добавление аннотаций
        if len(self.loss_history) > 10:
            # Показать начальную и конечную потерю
            ax.annotate(f'Начальная: {self.loss_history[0]:.4f}', 
                       xy=(0, self.loss_history[0]), 
                       xytext=(len(self.loss_history)*0.2, self.loss_history[0]*1.2),
                       arrowprops=dict(arrowstyle='->', color='red'))
            
            ax.annotate(f'Финальная: {self.loss_history[-1]:.4f}', 
                       xy=(len(self.loss_history)-1, self.loss_history[-1]), 
                       xytext=(len(self.loss_history)*0.7, self.loss_history[-1]*1.5),
                       arrowprops=dict(arrowstyle='->', color='green'))
        
        fig.tight_layout()
        
        # Встраивание в tkinter
        canvas = FigureCanvasTkAgg(fig, self.loss_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Переключение на вкладку с графиком
        self.notebook.select(self.loss_frame)

    def plot_network(self):
        """Визуализация структуры нейронной сети"""
        if self.mlp is None:
            messagebox.showwarning("Предупреждение", "Сначала создайте сеть!")
            return
        
        # Очистка предыдущего графика
        for widget in self.network_frame.winfo_children():
            widget.destroy()
        
        # Создание фигуры
        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        layer_sizes = self.mlp.layer_sizes
        max_neurons = max(layer_sizes)
        
        # Позиции слоев
        layer_positions = []
        for i, size in enumerate(layer_sizes):
            x = i * 3  # Расстояние между слоями
            y_positions = np.linspace(-max_neurons/2, max_neurons/2, size)
            layer_positions.append([(x, y) for y in y_positions])
        
        # Рисование нейронов
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for i, (layer_pos, size) in enumerate(zip(layer_positions, layer_sizes)):
            color = colors[i % len(colors)]
            for x, y in layer_pos:
                circle = patches.Circle((x, y), 0.3, color=color, ec='black', linewidth=1.5)
                ax.add_patch(circle)
        
        # Рисование связей (весов)
        for i in range(len(layer_positions) - 1):
            current_layer = layer_positions[i]
            next_layer = layer_positions[i + 1]
            weights = self.mlp.weights[i]
            
            # Нормализация весов для цвета
            w_min, w_max = weights.min(), weights.max()
            w_range = w_max - w_min if w_max != w_min else 1
            
            for j, (x1, y1) in enumerate(current_layer):
                for k, (x2, y2) in enumerate(next_layer):
                    weight = weights[j, k]
                    # Цвет линии зависит от веса
                    intensity = abs(weight - w_min) / w_range
                    color = 'red' if weight > 0 else 'blue'
                    alpha = 0.3 + 0.7 * intensity
                    linewidth = 0.5 + 2 * intensity
                    
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)
        
        # Подписи слоев
        layer_names = ['Входной'] + [f'Скрытый {i}' for i in range(1, len(layer_sizes)-1)] + ['Выходной']
        for i, (name, size) in enumerate(zip(layer_names, layer_sizes)):
            x = i * 3
            ax.text(x, max_neurons/2 + 1, f'{name}\n({size} нейронов)', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-1, (len(layer_sizes)-1) * 3 + 1)
        ax.set_ylim(-max_neurons/2 - 2, max_neurons/2 + 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Структура нейронной сети\n(Красные линии - положительные веса, Синие - отрицательные)', 
                    fontsize=12, fontweight='bold')
        
        fig.tight_layout()
        
        # Встраивание в tkinter
        canvas = FigureCanvasTkAgg(fig, self.network_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Переключение на вкладку с графиком
        self.notebook.select(self.network_frame)

    def plot_predictions(self):
        """График предсказаний против целевых значений"""
        if self.mlp is None:
            messagebox.showwarning("Предупреждение", "Сначала создайте и обучите сеть!")
            return
        
        # Очистка предыдущего графика
        for widget in self.pred_frame.winfo_children():
            widget.destroy()
        
        # Получение предсказаний
        predictions = self.mlp.predict(self.training_data)
        targets = self.training_labels
        
        # Создание фигуры с подграфиками
        fig = Figure(figsize=(12, 10), dpi=100)
        
        # График 1: Предсказания vs Цели
        ax1 = fig.add_subplot(221)
        ax1.scatter(targets, predictions, alpha=0.7, s=100, c='blue', edgecolors='black')
        
        # Идеальная линия
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальная линия')
        
        ax1.set_xlabel('Целевые значения')
        ax1.set_ylabel('Предсказания')
        ax1.set_title('Предсказания vs Целевые значения')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Ошибки
        ax2 = fig.add_subplot(222)
        errors = predictions.flatten() - targets.flatten()
        ax2.bar(range(len(errors)), errors, alpha=0.7, 
               color=['red' if e > 0 else 'blue' for e in errors])
        ax2.set_xlabel('Образец')
        ax2.set_ylabel('Ошибка (Предсказание - Цель)')
        ax2.set_title('Ошибки предсказаний')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # График 3: Распределение ошибок
        ax3 = fig.add_subplot(223)
        ax3.hist(errors, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Ошибка')
        ax3.set_ylabel('Частота')
        ax3.set_title('Распределение ошибок')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.grid(True, alpha=0.3)
        
        # График 4: Временной ряд предсказаний
        ax4 = fig.add_subplot(224)
        x_range = range(len(targets))
        ax4.plot(x_range, targets.flatten(), 'o-', label='Целевые', linewidth=2, markersize=8)
        ax4.plot(x_range, predictions.flatten(), 's-', label='Предсказания', linewidth=2, markersize=8)
        ax4.set_xlabel('Образец')
        ax4.set_ylabel('Значение')
        ax4.set_title('Сравнение по образцам')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Статистики
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        r2 = 1 - np.sum(errors**2) / np.sum((targets - np.mean(targets))**2)
        
        fig.suptitle(f'Анализ предсказаний\nMSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}', 
                    fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        
        # Встраивание в tkinter
        canvas = FigureCanvasTkAgg(fig, self.pred_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Переключение на вкладку с графиком
        self.notebook.select(self.pred_frame)

    def plot_activation(self):
        """График функции активации и её производной"""
        # Очистка предыдущего графика
        for widget in self.activation_frame.winfo_children():
            widget.destroy()
        
        # Создание фигуры
        fig = Figure(figsize=(12, 8), dpi=100)
        
        # Диапазон значений
        x = np.linspace(-10, 10, 1000)
        
        # Сигмоида и её производная
        if self.mlp:
            sigmoid_y = self.mlp.sigmoid(x)
            sigmoid_derivative_y = self.mlp.sigmoid_derivative(x)
        else:
            # Если сети нет, показываем стандартную сигмоиду
            sigmoid_y = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            sigmoid_derivative_y = sigmoid_y * (1 - sigmoid_y)
        
        # График 1: Функция активации
        ax1 = fig.add_subplot(221)
        ax1.plot(x, sigmoid_y, 'b-', linewidth=3, label='σ(x) = 1/(1+e^(-x))')
        ax1.set_xlabel('x')
        ax1.set_ylabel('σ(x)')
        ax1.set_title('Сигмоидная функция активации')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.axhline(y=1, color='black', linewidth=0.5, linestyle='--')
        ax1.axvline(x=0, color='black', linewidth=0.5)
        
        # График 2: Производная
        ax2 = fig.add_subplot(222)
        ax2.plot(x, sigmoid_derivative_y, 'r-', linewidth=3, label="σ'(x) = σ(x)(1-σ(x))")
        ax2.set_xlabel('x')
        ax2.set_ylabel("σ'(x)")
        ax2.set_title('Производная сигмоиды')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.axvline(x=0, color='black', linewidth=0.5)
        
        # График 3: Сравнение с другими функциями активации
        ax3 = fig.add_subplot(223)
        
        # ReLU
        relu_y = np.maximum(0, x)
        ax3.plot(x, relu_y, 'g-', linewidth=2, label='ReLU')
        
        # Tanh
        tanh_y = np.tanh(x)
        ax3.plot(x, tanh_y, 'm-', linewidth=2, label='Tanh')
        
        # Sigmoid
        ax3.plot(x, sigmoid_y, 'b-', linewidth=2, label='Sigmoid')
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('f(x)')
        ax3.set_title('Сравнение функций активации')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.axvline(x=0, color='black', linewidth=0.5)
        ax3.set_ylim(-1.5, 1.5)
        
        # График 4: Свойства сигмоиды
        ax4 = fig.add_subplot(224)
        
        # Показать проблему затухающих градиентов
        x_grad = np.linspace(-6, 6, 100)
        sigmoid_grad = 1 / (1 + np.exp(-x_grad))
        derivative_grad = sigmoid_grad * (1 - sigmoid_grad)
        
        ax4.fill_between(x_grad, 0, derivative_grad, alpha=0.3, color='red', 
                        label='Область малых градиентов')
        ax4.plot(x_grad, derivative_grad, 'r-', linewidth=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel("σ'(x)")
        ax4.set_title('Проблема затухающих градиентов')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axhline(y=0.25, color='orange', linewidth=2, linestyle='--', 
                   label='Максимум производной = 0.25')
        
        fig.tight_layout()
        
        # Встраивание в tkinter
        canvas = FigureCanvasTkAgg(fig, self.activation_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Переключение на вкладку с графиком
        self.notebook.select(self.activation_frame)

    def plot_weights(self):
        """Визуализация весов нейронной сети"""
        if self.mlp is None:
            messagebox.showwarning("Предупреждение", "Сначала создайте сеть!")
            return
        
        # Очистка предыдущего графика
        for widget in self.weights_frame.winfo_children():
            widget.destroy()
        
        # Создание фигуры
        num_layers = len(self.mlp.weights)
        fig = Figure(figsize=(15, 4 * num_layers), dpi=100)
        
        for i, weights in enumerate(self.mlp.weights):
            # Тепловая карта весов
            ax1 = fig.add_subplot(num_layers, 3, i*3 + 1)
            im = ax1.imshow(weights.T, cmap='RdBu', aspect='auto', interpolation='nearest')
            ax1.set_title(f'Веса слоя {i+1}\n({weights.shape[0]} → {weights.shape[1]})')
            ax1.set_xlabel('Входные нейроны')
            ax1.set_ylabel('Выходные нейроны')
            fig.colorbar(im, ax=ax1, shrink=0.8)
            
            # Гистограмма весов
            ax2 = fig.add_subplot(num_layers, 3, i*3 + 2)
            ax2.hist(weights.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title(f'Распределение весов слоя {i+1}')
            ax2.set_xlabel('Значение веса')
            ax2.set_ylabel('Частота')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.grid(True, alpha=0.3)
            
            # Статистики весов
            ax3 = fig.add_subplot(num_layers, 3, i*3 + 3)
            stats = {
                'Среднее': np.mean(weights),
                'Медиана': np.median(weights),
                'Ст. откл.': np.std(weights),
                'Мин': np.min(weights),
                'Макс': np.max(weights),
                'Норма L2': np.linalg.norm(weights)
            }
            
            y_pos = np.arange(len(stats))
            values = list(stats.values())
            bars = ax3.barh(y_pos, values, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
            
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(list(stats.keys()))
            ax3.set_title(f'Статистики весов слоя {i+1}')
            ax3.set_xlabel('Значение')
            
            # Добавление значений на столбцы
            for j, (bar, value) in enumerate(zip(bars, values)):
                ax3.text(value + 0.01 * max(values), bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center', fontsize=8)
            
            ax3.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle('Анализ весов нейронной сети', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        # Встраивание в tkinter
        canvas = FigureCanvasTkAgg(fig, self.weights_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Переключение на вкладку с графиком
        self.notebook.select(self.weights_frame)


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
