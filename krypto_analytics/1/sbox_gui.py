"""
GUI для лабораторной работы 1: 
Исследование свойств S-боксов и их нелинейности
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from sbox_lab1 import (
    BooleanFunction, LinearFunction, SBox, 
    nonlinearity, generate_all_linear_functions,
    generate_random_sbox
)
import itertools


class SBoxGUI:
    """Главный класс GUI приложения"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("S-боксы и степень нелинейности")
        self.root.geometry("1200x750")
        self.root.config(bg="#f0f0f0")
        
        # Стили
        self.setup_styles()
        
        # Создаем главный фрейм с вкладками
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Создаем вкладки
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        self.tab5 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="📊 Основные операции")
        self.notebook.add(self.tab2, text="🎯 Нелинейность")
        self.notebook.add(self.tab3, text="🔷 Bent-функции")
        self.notebook.add(self.tab4, text="📦 S-боксы")
        self.notebook.add(self.tab5, text="📈 Анализ")
        
        # Инициализируем вкладки
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()
        self.setup_tab5()
        
        # Флаг для отмены операций
        self.cancel_operation = False
    
    def setup_styles(self):
        """Настройка стилей ttk"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Normal.TLabel', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
    
    # =====================================================================
    # ВКЛАДКА 1: ОСНОВНЫЕ ОПЕРАЦИИ
    # =====================================================================
    
    def setup_tab1(self):
        """Вкладка для основных операций"""
        main_frame = ttk.Frame(self.tab1, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Заголовок
        title = ttk.Label(main_frame, text="Расстояние Хэмминга", style='Title.TLabel')
        title.pack(pady=10)
        
        # Левая часть
        left_frame = ttk.LabelFrame(main_frame, text="Функция 1", padding="10")
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        ttk.Label(left_frame, text="Размерность n:").grid(row=0, column=0, sticky="w")
        self.tab1_n1_var = tk.StringVar(value="2")
        ttk.Entry(left_frame, textvariable=self.tab1_n1_var, width=5).grid(row=0, column=1, sticky="w")
        
        ttk.Label(left_frame, text="Таблица истинности:", font=('Arial', 10)).grid(row=1, column=0, columnspan=2, pady=10, sticky="nw")
        
        self.tab1_text1 = tk.Text(left_frame, height=8, width=20, font=('Courier', 9))
        self.tab1_text1.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.tab1_text1.insert("1.0", "0\n0\n0\n1")
        
        ttk.Label(left_frame, text="(каждое значение на новой строке)", font=('Arial', 8)).grid(row=3, column=0, columnspan=2, sticky="w")
        
        # Правая часть
        right_frame = ttk.LabelFrame(main_frame, text="Функция 2", padding="10")
        right_frame.pack(side="right", fill="both", expand=True, padx=5)
        
        ttk.Label(right_frame, text="Размерность n:").grid(row=0, column=0, sticky="w")
        self.tab1_n2_var = tk.StringVar(value="2")
        ttk.Entry(right_frame, textvariable=self.tab1_n2_var, width=5).grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(right_frame, text="Таблица истинности:", font=('Arial', 10)).grid(row=1, column=0, columnspan=2, pady=10, sticky="nw")
        
        self.tab1_text2 = tk.Text(right_frame, height=8, width=20, font=('Courier', 9))
        self.tab1_text2.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        self.tab1_text2.insert("1.0", "0\n1\n1\n1")
        
        ttk.Label(right_frame, text="(каждое значение на новой строке)", font=('Arial', 8)).grid(row=3, column=0, columnspan=2, sticky="w")
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(button_frame, text="🔢 Вычислить расстояние", 
                  command=self.calculate_hamming).pack(side="left", padx=5)
        ttk.Button(button_frame, text="🔄 Пример: И и ИЛИ", 
                  command=self.example_and_or).pack(side="left", padx=5)
        
        # Результат
        result_frame = ttk.LabelFrame(main_frame, text="Результат", padding="10")
        result_frame.pack(fill="both", expand=True, pady=10)
        
        self.tab1_result = scrolledtext.ScrolledText(result_frame, height=10, font=('Courier', 10))
        self.tab1_result.pack(fill="both", expand=True)
    
    def calculate_hamming(self):
        """Вычисляет расстояние Хэмминга"""
        try:
            n1 = int(self.tab1_n1_var.get())
            n2 = int(self.tab1_n2_var.get())
            
            if n1 != n2:
                messagebox.showerror("Ошибка", "Размерности функций должны быть одинаковыми!")
                return
            
            # Парсим таблицы истинности
            tt1_str = self.tab1_text1.get("1.0", "end-1c").strip().split('\n')
            tt2_str = self.tab1_text2.get("1.0", "end-1c").strip().split('\n')
            
            if len(tt1_str) != 2**n1 or len(tt2_str) != 2**n2:
                messagebox.showerror("Ошибка", f"Таблица истинности должна иметь {2**n1} элементов!")
                return
            
            tt1 = [int(x) for x in tt1_str]
            tt2 = [int(x) for x in tt2_str]
            
            f1 = BooleanFunction(n1, tt1)
            f2 = BooleanFunction(n2, tt2)
            
            distance = BooleanFunction.hamming_distance(f1, f2)
            
            self.tab1_result.delete("1.0", "end")
            self.tab1_result.insert("end", f"Функция 1: {f1.truth_table}\n")
            self.tab1_result.insert("end", f"Функция 2: {f2.truth_table}\n\n")
            self.tab1_result.insert("end", f"{'='*50}\n")
            self.tab1_result.insert("end", f"Расстояние Хэмминга: {distance}\n")
            self.tab1_result.insert("end", f"{'='*50}\n\n")
            self.tab1_result.insert("end", f"Интерпретация:\n")
            self.tab1_result.insert("end", f"Функции отличаются в {distance} точках из {2**n1}\n")
            
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
    
    def example_and_or(self):
        """Пример: И и ИЛИ"""
        self.tab1_text1.delete("1.0", "end")
        self.tab1_text1.insert("1.0", "0\n0\n0\n1")
        self.tab1_n1_var.set("2")
        
        self.tab1_text2.delete("1.0", "end")
        self.tab1_text2.insert("1.0", "0\n1\n1\n1")
        self.tab1_n2_var.set("2")
        
        self.calculate_hamming()
    
    # =====================================================================
    # ВКЛАДКА 2: НЕЛИНЕЙНОСТЬ
    # =====================================================================
    
    def setup_tab2(self):
        """Вкладка для анализа нелинейности"""
        main_frame = ttk.Frame(self.tab2, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        title = ttk.Label(main_frame, text="Анализ степени нелинейности функции", style='Title.TLabel')
        title.pack(pady=10)
        
        # Входные данные
        input_frame = ttk.LabelFrame(main_frame, text="Введите функцию", padding="10")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(input_frame, text="Размерность n:").grid(row=0, column=0, sticky="w")
        self.tab2_n_var = tk.StringVar(value="2")
        ttk.Entry(input_frame, textvariable=self.tab2_n_var, width=5).grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(input_frame, text="Таблица истинности:").grid(row=1, column=0, sticky="nw")
        self.tab2_text = tk.Text(input_frame, height=6, width=40, font=('Courier', 9))
        self.tab2_text.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        self.tab2_text.insert("1.0", "0\n0\n0\n1")
        
        # Примеры
        example_frame = ttk.LabelFrame(main_frame, text="Встроенные примеры", padding="10")
        example_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(example_frame, text="f = x₁*x₂ (нелинейная)", 
                  command=self.example_x1x2).pack(side="left", padx=5)
        ttk.Button(example_frame, text="f = x₁ (линейная)", 
                  command=self.example_x1).pack(side="left", padx=5)
        ttk.Button(example_frame, text="f = x₁⊕x₂ (линейная)", 
                  command=self.example_xor).pack(side="left", padx=5)
        
        # Кнопка вычисления
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        ttk.Button(button_frame, text="🎯 Вычислить нелинейность", 
                  command=self.calculate_nonlinearity).pack(side="left", padx=5)
        
        # Результаты
        result_frame = ttk.LabelFrame(main_frame, text="Результаты анализа", padding="10")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tab2_result = scrolledtext.ScrolledText(result_frame, height=12, font=('Courier', 9))
        self.tab2_result.pack(fill="both", expand=True)
    
    def calculate_nonlinearity(self):
        """Вычисляет нелинейность функции"""
        try:
            n = int(self.tab2_n_var.get())
            tt_str = self.tab2_text.get("1.0", "end-1c").strip().split('\n')
            
            if len(tt_str) != 2**n:
                messagebox.showerror("Ошибка", f"Таблица должна иметь {2**n} элементов!")
                return
            
            tt = [int(x) for x in tt_str]
            f = BooleanFunction(n, tt)
            
            nl = nonlinearity(f)
            weight = sum(tt)
            
            self.tab2_result.delete("1.0", "end")
            self.tab2_result.insert("end", f"{'='*60}\n")
            self.tab2_result.insert("end", f"АНАЛИЗ БУЛЕВОЙ ФУНКЦИИ\n")
            self.tab2_result.insert("end", f"{'='*60}\n\n")
            
            self.tab2_result.insert("end", f"Размерность: n = {n}\n")
            self.tab2_result.insert("end", f"Таблица истинности: {f.truth_table}\n")
            self.tab2_result.insert("end", f"Вес функции (кол-во единиц): {weight}\n\n")
            
            self.tab2_result.insert("end", f"{'='*60}\n")
            self.tab2_result.insert("end", f"Степень нелинейности: {nl}\n")
            self.tab2_result.insert("end", f"{'='*60}\n\n")
            
            if nl == 0:
                status = "✓ ЛИНЕЙНАЯ функция"
            else:
                status = f"✓ НЕЛИНЕЙНАЯ функция (расстояние до линейной = {nl})"
            
            self.tab2_result.insert("end", status + "\n\n")
            
            # Максимальная возможная нелинейность
            if n % 2 == 0:
                max_nl = 2**(n-1) - 2**(n//2 - 1)
                self.tab2_result.insert("end", f"Максимальная нелинейность для n={n}: {max_nl}\n")
                if nl == max_nl:
                    self.tab2_result.insert("end", f"🎉 ЭТО BENT-ФУНКЦИЯ!\n")
            
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
    
    def example_x1x2(self):
        """Пример: x1*x2"""
        self.tab2_n_var.set("2")
        self.tab2_text.delete("1.0", "end")
        self.tab2_text.insert("1.0", "0\n0\n0\n1")
        self.calculate_nonlinearity()
    
    def example_x1(self):
        """Пример: x1"""
        self.tab2_n_var.set("2")
        self.tab2_text.delete("1.0", "end")
        self.tab2_text.insert("1.0", "0\n0\n1\n1")
        self.calculate_nonlinearity()
    
    def example_xor(self):
        """Пример: x1 XOR x2"""
        self.tab2_n_var.set("2")
        self.tab2_text.delete("1.0", "end")
        self.tab2_text.insert("1.0", "0\n1\n1\n0")
        self.calculate_nonlinearity()
    
    # =====================================================================
    # ВКЛАДКА 3: BENT-ФУНКЦИИ
    # =====================================================================
    
    def setup_tab3(self):
        """Вкладка для работы с bent-функциями"""
        main_frame = ttk.Frame(self.tab3, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        title = ttk.Label(main_frame, text="Построение Bent-функций", style='Title.TLabel')
        title.pack(pady=10)
        
        # Информация
        info_frame = ttk.LabelFrame(main_frame, text="Информация о Bent-функциях", padding="10")
        info_frame.pack(fill="x", padx=5, pady=5)
        
        info_text = (
            "Bent-функция - это булева функция от четного числа переменных,\n"
            "имеющая максимальную степень нелинейности.\n\n"
            "Для n переменных (четное): max NL = 2^(n-1) - 2^(n/2-1)\n\n"
            "n     max NL\n"
            "─────────────\n"
            "2        1\n"
            "4        6\n"
            "6        28\n"
            "8        120\n"
        )
        info_label = ttk.Label(info_frame, text=info_text, justify="left", font=('Courier', 10))
        info_label.pack(fill="x")
        
        # Примеры bent-функций
        examples_frame = ttk.LabelFrame(main_frame, text="Примеры известных Bent-функций", padding="10")
        examples_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(examples_frame, text="Проверить f = x₁*x₂ + x₃*x₄ (n=4)", 
                  command=self.check_bent_n4).pack(side="left", padx=5)
        ttk.Button(examples_frame, text="Проверить f = x₁*x₃ + x₂*x₄ (n=6)", 
                  command=self.check_bent_n6).pack(side="left", padx=5)
        
        # Поиск bent-функций - используем простой фрейм с pack
        search_frame = ttk.LabelFrame(main_frame, text="Поиск Bent-функций", padding="10")
        search_frame.pack(fill="x", padx=5, pady=5)
        
        # Верхняя строка параметров
        param_row1 = ttk.Frame(search_frame)
        param_row1.pack(fill="x", pady=5)
        
        ttk.Label(param_row1, text="Размерность n (четная):").pack(side="left", padx=5)
        self.tab3_n_var = tk.StringVar(value="4")
        ttk.Entry(param_row1, textvariable=self.tab3_n_var, width=5).pack(side="left", padx=5)
        
        ttk.Label(param_row1, text="Макс функций:").pack(side="left", padx=(20, 5))
        self.tab3_max_var = tk.StringVar(value="5000")
        ttk.Entry(param_row1, textvariable=self.tab3_max_var, width=10).pack(side="left", padx=5)
        
        # Кнопки и прогресс
        button_row = ttk.Frame(search_frame)
        button_row.pack(fill="x", pady=5)
        
        ttk.Button(button_row, text="🔷 Найти Bent-функции", 
                  command=self.search_bent).pack(side="left", padx=5)
        
        self.tab3_cancel_btn = ttk.Button(button_row, text="⏹ Отмена", 
                  command=self.cancel_search, state="disabled")
        self.tab3_cancel_btn.pack(side="left", padx=5)
        
        # Прогресс
        self.tab3_progress = ttk.Progressbar(button_row, length=300, mode='indeterminate')
        self.tab3_progress.pack(side="left", padx=10, fill="x", expand=True)
        
        # Результаты
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tab3_result = scrolledtext.ScrolledText(result_frame, height=10, font=('Courier', 10))
        self.tab3_result.pack(fill="both", expand=True)
    
    def check_bent_n4(self):
        """Проверяет bent-функцию для n=4: f = x1*x2 + x3*x4"""
        tt = []
        for x in range(16):
            x1 = (x >> 0) & 1
            x2 = (x >> 1) & 1
            x3 = (x >> 2) & 1
            x4 = (x >> 3) & 1
            value = (x1 & x2) ^ (x3 & x4)
            tt.append(value)
        
        f = BooleanFunction(4, tt)
        nl = nonlinearity(f)
        
        self.tab3_result.delete("1.0", "end")
        self.tab3_result.insert("end", "="*60 + "\n")
        self.tab3_result.insert("end", "Проверка: f = x₁*x₂ + x₃*x₄ (n=4)\n")
        self.tab3_result.insert("end", "="*60 + "\n\n")
        self.tab3_result.insert("end", f"Таблица истинности:\n{f.truth_table}\n\n")
        self.tab3_result.insert("end", f"Нелинейность: {nl}\n")
        self.tab3_result.insert("end", f"Максимальная для n=4: 6\n\n")
        
        if nl == 6:
            self.tab3_result.insert("end", "✓ ЭТО BENT-ФУНКЦИЯ!")
        else:
            self.tab3_result.insert("end", "✗ Не bent-функция")
    
    def check_bent_n6(self):
        """Проверяет bent-функцию для n=6: f = x1*x3 + x2*x4"""
        tt = []
        for x in range(64):
            x1 = (x >> 0) & 1
            x2 = (x >> 1) & 1
            x3 = (x >> 2) & 1
            x4 = (x >> 3) & 1
            value = (x1 & x3) ^ (x2 & x4)
            tt.append(value)
        
        f = BooleanFunction(6, tt)
        nl = nonlinearity(f)
        
        self.tab3_result.delete("1.0", "end")
        self.tab3_result.insert("end", "="*60 + "\n")
        self.tab3_result.insert("end", "Проверка: f = x₁*x₃ + x₂*x₄ (n=6)\n")
        self.tab3_result.insert("end", "="*60 + "\n\n")
        self.tab3_result.insert("end", f"Размер таблицы: {len(f.truth_table)}\n")
        self.tab3_result.insert("end", f"Нелинейность: {nl}\n")
        self.tab3_result.insert("end", f"Максимальная для n=6: 28\n\n")
        
        if nl == 28:
            self.tab3_result.insert("end", "✓ ЭТО BENT-ФУНКЦИЯ!")
        else:
            self.tab3_result.insert("end", "✗ Не bent-функция")
    
    def search_bent(self):
        """Поиск bent-функций в отдельном потоке"""
        try:
            n = int(self.tab3_n_var.get())
            max_search = int(self.tab3_max_var.get())
            
            if n % 2 != 0:
                messagebox.showerror("Ошибка", "n должно быть четным!")
                return
            
            self.cancel_operation = False
            self.tab3_cancel_btn.config(state="normal")
            
            thread = threading.Thread(target=self._search_bent_thread, args=(n, max_search))
            thread.daemon = True
            thread.start()
            
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Введите корректные значения")
    
    def _search_bent_thread(self, n, max_search):
        """Поиск bent-функций в отдельном потоке"""
        self.tab3_progress.start()
        
        try:
            bent_funcs = []
            max_nl = 2**(n-1) - 2**(n//2 - 1)
            
            self.tab3_result.delete("1.0", "end")
            self.tab3_result.insert("end", f"Поиск bent-функций для n={n}...\n")
            self.tab3_result.insert("end", f"Максимальная нелинейность: {max_nl}\n\n")
            
            checked = 0
            for tt_tuple in itertools.islice(
                itertools.product([0, 1], repeat=2**n),
                max_search
            ):
                if self.cancel_operation:
                    break
                
                f = BooleanFunction(n, list(tt_tuple))
                nl = nonlinearity(f)
                
                if nl == max_nl:
                    bent_funcs.append(f)
                    self.tab3_result.insert("end", f"✓ Найдена bent-функция #{len(bent_funcs)}\n")
                    self.tab3_result.see("end")
                    self.root.update_idletasks()
                
                checked += 1
                if checked % 1000 == 0:
                    progress = (checked / max_search) * 100
                    self.tab3_progress['value'] = progress
                    self.root.update_idletasks()
            
            self.tab3_result.insert("end", f"\n{'='*60}\n")
            self.tab3_result.insert("end", f"Поиск завершен!\n")
            self.tab3_result.insert("end", f"Проверено функций: {checked}\n")
            self.tab3_result.insert("end", f"Найдено bent-функций: {len(bent_funcs)}\n")
            
        except Exception as e:
            self.tab3_result.insert("end", f"\nОшибка: {str(e)}\n")
        
        finally:
            self.tab3_progress.stop()
            self.tab3_cancel_btn.config(state="disabled")
    
    def cancel_search(self):
        """Отмена поиска"""
        self.cancel_operation = True
        self.tab3_cancel_btn.config(state="disabled")
    
    # =====================================================================
    # ВКЛАДКА 4: S-БОКСЫ
    # =====================================================================
    
    def setup_tab4(self):
        """Вкладка для работы с S-боксами"""
        main_frame = ttk.Frame(self.tab4, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        title = ttk.Label(main_frame, text="Генерирование S-боксов", style='Title.TLabel')
        title.pack(pady=10)
        
        # Параметры
        param_frame = ttk.LabelFrame(main_frame, text="Параметры S-бокса", padding="10")
        param_frame.pack(fill="x", padx=5, pady=5)
        
        param_row = ttk.Frame(param_frame)
        param_row.pack(fill="x")
        
        ttk.Label(param_row, text="Размерность n:").pack(side="left", padx=5)
        self.tab4_n_var = tk.StringVar(value="3")
        ttk.Spinbox(param_row, from_=2, to=8, textvariable=self.tab4_n_var, width=5).pack(side="left", padx=5)
        
        ttk.Label(param_row, text="Количество итераций:").pack(side="left", padx=(20, 5))
        self.tab4_iter_var = tk.StringVar(value="100")
        ttk.Entry(param_row, textvariable=self.tab4_iter_var, width=10).pack(side="left", padx=5)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(button_frame, text="🎲 Случайный S-бокс", 
                  command=self.generate_random).pack(side="left", padx=5)
        ttk.Button(button_frame, text="🔍 Найти с макс нелинейностью", 
                  command=self.find_best_sbox).pack(side="left", padx=5)
        self.tab4_cancel_btn = ttk.Button(button_frame, text="⏹ Отмена", 
                  command=self.cancel_sbox, state="disabled")
        self.tab4_cancel_btn.pack(side="left", padx=5)
        
        # Прогресс
        self.tab4_progress = ttk.Progressbar(button_frame, length=300, mode='determinate')
        self.tab4_progress.pack(side="left", padx=10, fill="x", expand=True)
        
        # Результаты
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tab4_result = scrolledtext.ScrolledText(result_frame, height=12, font=('Courier', 9))
        self.tab4_result.pack(fill="both", expand=True)
    
    def generate_random(self):
        """Генерирует случайный S-бокс"""
        try:
            n = int(self.tab4_n_var.get())
            sbox = generate_random_sbox(n)
            nl = sbox.nonlinearity()
            
            self.tab4_result.delete("1.0", "end")
            self.tab4_result.insert("end", f"{'='*60}\n")
            self.tab4_result.insert("end", f"Случайный S-бокс размерности ({n}, {n})\n")
            self.tab4_result.insert("end", f"{'='*60}\n\n")
            
            self.tab4_result.insert("end", f"Нелинейность: {nl}\n\n")
            
            self.tab4_result.insert("end", "Базовые функции:\n")
            for i, f in enumerate(sbox.base_functions):
                self.tab4_result.insert("end", f"  f{i+1}: {f.truth_table[:8]}...\n")
            
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
    
    def find_best_sbox(self):
        """Поиск S-бокса с максимальной нелинейностью"""
        try:
            n = int(self.tab4_n_var.get())
            iterations = int(self.tab4_iter_var.get())
            
            self.cancel_operation = False
            self.tab4_cancel_btn.config(state="normal")
            
            thread = threading.Thread(target=self._find_best_sbox_thread, args=(n, iterations))
            thread.daemon = True
            thread.start()
            
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Введите корректные значения")
    
    def _find_best_sbox_thread(self, n, iterations):
        """Поиск лучшего S-бокса в отдельном потоке"""
        try:
            self.tab4_result.delete("1.0", "end")
            self.tab4_result.insert("end", f"Поиск S-бокса размерности {n}...\n\n")
            
            best_nl = 0
            best_sbox = None
            
            for i in range(iterations):
                if self.cancel_operation:
                    break
                
                sbox = generate_random_sbox(n)
                nl = sbox.nonlinearity()
                
                if nl > best_nl:
                    best_nl = nl
                    best_sbox = sbox
                    self.tab4_result.insert("end", f"Итерация {i+1}: найден S-бокс с NL = {best_nl}\n")
                    self.tab4_result.see("end")
                    self.root.update_idletasks()
                
                self.tab4_progress['value'] = ((i + 1) / iterations) * 100
                self.root.update_idletasks()
            
            if best_sbox:
                self.tab4_result.insert("end", f"\n{'='*60}\n")
                self.tab4_result.insert("end", f"Лучший найденный S-бокс:\n")
                self.tab4_result.insert("end", f"Размерность: ({n}, {n})\n")
                self.tab4_result.insert("end", f"Нелинейность: {best_nl}\n")
                self.tab4_result.insert("end", f"{'='*60}\n")
        
        except Exception as e:
            self.tab4_result.insert("end", f"\nОшибка: {str(e)}\n")
        
        finally:
            self.tab4_cancel_btn.config(state="disabled")
    
    def cancel_sbox(self):
        """Отмена поиска S-бокса"""
        self.cancel_operation = True
        self.tab4_cancel_btn.config(state="disabled")
    
    # =====================================================================
    # ВКЛАДКА 5: АНАЛИЗ
    # =====================================================================
    
    def setup_tab5(self):
        """Вкладка для анализа"""
        main_frame = ttk.Frame(self.tab5, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        title = ttk.Label(main_frame, text="Справочная информация и таблицы", style='Title.TLabel')
        title.pack(pady=10)
        
        # Вкладки для разных типов информации
        sub_notebook = ttk.Notebook(main_frame)
        sub_notebook.pack(fill="both", expand=True)
        
        # Таблица максимальной нелинейности
        info1_frame = ttk.Frame(sub_notebook)
        sub_notebook.add(info1_frame, text="Максимальная нелинейность")
        
        info1_text = """
МАКСИМАЛЬНАЯ СТЕПЕНЬ НЕЛИНЕЙНОСТИ ДЛЯ BENT-ФУНКЦИЙ

Для четного n булева функция может иметь максимальную нелинейность:
    NL_max = 2^(n-1) - 2^(n/2-1)

Таблица значений:
┌───┬──────────┬─────────────────────────────┐
│ n │ NL_max   │ 2^(2^n) всего функций       │
├───┼──────────┼─────────────────────────────┤
│ 2 │    1     │ 16                          │
│ 4 │    6     │ 65536                       │
│ 6 │   28     │ ~ 1.8 × 10^18               │
│ 8 │  120     │ ~ 1.1 × 10^77               │
│10 │  496     │ огромное число              │
└───┴──────────┴─────────────────────────────┘

Bent-функции существуют ТОЛЬКО для четных n!
При нечетных n нельзя достичь максимальной нелинейности.
        """
        
        text1 = scrolledtext.ScrolledText(info1_frame, font=('Courier', 10))
        text1.pack(fill="both", expand=True, padx=5, pady=5)
        text1.insert("1.0", info1_text)
        text1.config(state="disabled")
        
        # Примеры функций
        info2_frame = ttk.Frame(sub_notebook)
        sub_notebook.add(info2_frame, text="Примеры функций")
        
        info2_text = """
ПРИМЕРЫ БУЛЕВЫХ ФУНКЦИЙ И ИХ СВОЙСТВА

1. ЛИНЕЙНЫЕ ФУНКЦИИ (нелинейность = 0)
   • f = 0 (нулевая функция)
   • f = 1 (единичная функция)
   • f = x₁ (проекция)
   • f = x₁ ⊕ x₂ (XOR)
   • f = 1 ⊕ x₁ ⊕ x₂ (отрицание XOR)

2. НЕЛИНЕЙНЫЕ ФУНКЦИИ
   • f = x₁ ∧ x₂ (И)
   • f = x₁ ∨ x₂ (ИЛИ)
   • f = x₁ ⊙ x₂ (XNOR)
   • f = x₁ ∧ x₂ ⊕ x₃ (смешанная)

3. BENT-ФУНКЦИИ (максимальная нелинейность)
   Для n = 4:  f = x₁*x₂ + x₃*x₄
   Для n = 6:  f = x₁*x₃ + x₂*x₄
   
   NL = 6 (max для n=4)
   NL = 28 (max для n=6)
        """
        
        text2 = scrolledtext.ScrolledText(info2_frame, font=('Courier', 10))
        text2.pack(fill="both", expand=True, padx=5, pady=5)
        text2.insert("1.0", info2_text)
        text2.config(state="disabled")
        
        # Формулы
        info3_frame = ttk.Frame(sub_notebook)
        sub_notebook.add(info3_frame, text="Основные формулы")
        
        info3_text = """
ОСНОВНЫЕ ОПРЕДЕЛЕНИЯ И ФОРМУЛЫ

1. РАССТОЯНИЕ ХЭММИНГА между функциями f и g:
   ρ(f, g) = |{x ∈ F₂ⁿ : f(x) ≠ g(x)}|
   
   Это количество точек, в которых функции отличаются.

2. НОРМА ХЭММИНГА вектора a = (a₁, ..., aₙ):
   H(a) = количество ненулевых координат
   Пример: H(101) = 2

3. ПОЛИНОМ ЖЕГАЛКИНА:
   f = a₀ + ∑ aᵢxᵢ + ∑ aᵢⱼxᵢxⱼ + ...
   где aᵢ ∈ {0, 1}

4. СТЕПЕНЬ НЕЛИНЕЙНОСТИ:
   NL(f) = min{ρ(f, g) : g ∈ LFₙ}
   (минимальное расстояние до линейной функции)

5. BENT-ФУНКЦИЯ (n четное):
   NL(f) = 2^(n-1) - 2^(n/2-1) (максимум!)

6. S-БОК (n, m):
   Отображение из F₂ⁿ в F₂ᵐ
   Задается m булевыми функциями f₁, ..., fₘ
   
   Нелинейность S-бокса = минимальная нелинейность
   всех линейных комбинаций его базовых функций
        """
        
        text3 = scrolledtext.ScrolledText(info3_frame, font=('Courier', 10))
        text3.pack(fill="both", expand=True, padx=5, pady=5)
        text3.insert("1.0", info3_text)
        text3.config(state="disabled")


def main():
    """Запуск GUI приложения"""
    root = tk.Tk()
    app = SBoxGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()