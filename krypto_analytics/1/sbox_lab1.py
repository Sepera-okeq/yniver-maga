"""
Лабораторная работа 1: Исследование свойств S-боксов и построение 
S-бокса наибольшей степени нелинейности
"""

import itertools
from typing import List, Tuple, Set
import numpy as np


# ============================================================================
# ЧАСТЬ 1: Расстояние Хэмминга и степень нелинейности
# ============================================================================

class BooleanFunction:
    """Класс для работы с булевыми функциями от n переменных"""
    
    def __init__(self, n: int, truth_table: List[int]):
        """
        Инициализация булевой функции
        
        Args:
            n: количество переменных
            truth_table: вектор значений функции (список из 2^n элементов)
        """
        if len(truth_table) != 2**n:
            raise ValueError(f"Таблица истинности должна иметь {2**n} элементов")
        
        self.n = n
        self.truth_table = tuple(truth_table)
    
    def __call__(self, x: int) -> int:
        """Вычисление значения функции на входе x"""
        return self.truth_table[x]
    
    def __repr__(self):
        return f"BF(n={self.n}, tt={self.truth_table})"
    
    @staticmethod
    def hamming_distance(f1: 'BooleanFunction', f2: 'BooleanFunction') -> int:
        """
        Расстояние Хэмминга между двумя булевыми функциями
        
        Args:
            f1, f2: две булевы функции одинаковой размерности
            
        Returns:
            Количество точек, в которых функции отличаются
        """
        if f1.n != f2.n:
            raise ValueError("Функции должны иметь одинаковую размерность")
        
        distance = sum(f1(x) != f2(x) for x in range(2**f1.n))
        return distance
    
    @staticmethod
    def hamming_norm(vector: List[int]) -> int:
        """Норма Хэмминга вектора (количество ненулевых элементов)"""
        return sum(1 for x in vector if x != 0)


class LinearFunction(BooleanFunction):
    """Класс для линейной (аффинной) булевой функции"""
    
    def __init__(self, n: int, a0: int, coeffs: List[int]):
        """
        Инициализация линейной функции f = a0 + a1*x1 + ... + an*xn
        
        Args:
            n: количество переменных
            a0: свободный член (0 или 1)
            coeffs: коэффициенты при переменных (список из n элементов)
        """
        if len(coeffs) != n:
            raise ValueError(f"Должно быть {n} коэффициентов")
        
        self.a0 = a0
        self.coeffs = coeffs
        
        # Вычисляем таблицу истинности
        truth_table = []
        for x in range(2**n):
            # Раскладываем x на биты
            bits = [(x >> i) & 1 for i in range(n)]
            # Вычисляем скалярное произведение
            value = a0
            for i in range(n):
                value ^= (bits[i] & coeffs[i])
            truth_table.append(value)
        
        super().__init__(n, truth_table)
    
    def __repr__(self):
        return f"LF(a0={self.a0}, coeffs={self.coeffs})"


def generate_all_linear_functions(n: int) -> List[LinearFunction]:
    """
    Генерирует все линейные функции от n переменных
    
    Returns:
        Список всех $2^{n+1}$ линейных функций размерности n
    """
    linear_functions = []
    
    # Перебираем все возможные коэффициенты
    for a0 in [0, 1]:
        for coeffs_tuple in itertools.product([0, 1], repeat=n):
            lf = LinearFunction(n, a0, list(coeffs_tuple))
            linear_functions.append(lf)
    
    return linear_functions


def nonlinearity(f: BooleanFunction) -> int:
    """
    Вычисляет степень нелинейности (nonlinearity) булевой функции
    
    Args:
        f: булева функция
        
    Returns:
        Минимальное расстояние Хэмминга от f до множества линейных функций
    """
    linear_functions = generate_all_linear_functions(f.n)
    
    min_distance = float('inf')
    closest_linear = None
    
    for lf in linear_functions:
        distance = BooleanFunction.hamming_distance(f, lf)
        if distance < min_distance:
            min_distance = distance
            closest_linear = lf
    
    return min_distance


# ============================================================================
# ЧАСТЬ 2: Построение bent-функций
# ============================================================================

def is_bent_function(f: BooleanFunction) -> bool:
    """
    Проверяет, является ли функция bent-функцией
    (имеет максимальную нелинейность для четных n)
    """
    if f.n % 2 != 0:
        return False  # Bent-функции существуют только для четных n
    
    expected_max_nl = 2**(f.n - 1) - 2**(f.n // 2 - 1)
    actual_nl = nonlinearity(f)
    
    return actual_nl == expected_max_nl


def find_bent_functions(n: int, max_search: int = None) -> List[BooleanFunction]:
    """
    Поиск bent-функций размерности n
    
    Args:
        n: размерность (должна быть четной)
        max_search: максимальное количество функций для поиска 
                   (для экономии времени)
    
    Returns:
        Список найденных bent-функций
    """
    if n % 2 != 0:
        print(f"Bent-функции существуют только для четных n. n={n} нечетное.")
        return []
    
    print(f"\n{'='*70}")
    print(f"Поиск bent-функций размерности n={n}")
    print(f"{'='*70}")
    
    bent_functions = []
    max_nl = 2**(n - 1) - 2**(n // 2 - 1)
    
    print(f"Максимальная нелинейность для n={n}: {max_nl}")
    print(f"Всего булевых функций: {2**(2**n)}")
    
    total = 2**(2**n)
    if max_search:
        total = min(total, max_search)
    
    # Поиск осуществляется перебором всех возможных таблиц истинности
    checked = 0
    for truth_table_tuple in itertools.islice(
        itertools.product([0, 1], repeat=2**n), 
        total
    ):
        f = BooleanFunction(n, list(truth_table_tuple))
        nl = nonlinearity(f)
        
        if nl == max_nl:
            bent_functions.append(f)
            print(f"Найдена bent-функция #{len(bent_functions)}")
            print(f"  Таблица истинности: {f.truth_table[:8]}...")
        
        checked += 1
        if checked % 10000 == 0:
            print(f"  Проверено функций: {checked}...", end='\r')
    
    print(f"\nВсего найдено bent-функций: {len(bent_functions)}")
    return bent_functions


def build_bent_functions_specific():
    """Построение bent-функций для n=4 и n=6"""
    
    print("\n" + "="*70)
    print("ПОИСК BENT-ФУНКЦИЙ")
    print("="*70)
    
    # Для n=4: максимальная нелинейность = 6
    print("\nДля n=4: Максимальная нелинейность = 6")
    print("Примеры известных bent-функций:")
    
    # f = x1*x2 + x3*x4 (пример bent-функции для n=4)
    bent_4_example = BooleanFunction(4, [0, 0, 0, 1, 0, 1, 1, 0, 
                                         0, 1, 1, 0, 1, 0, 0, 1])
    print(f"f = x1*x2 + x3*x4: NL = {nonlinearity(bent_4_example)}")
    
    # Для n=6: максимальная нелинейность = 28
    print("\nДля n=6: Максимальная нелинейность = 28")
    print("Примеры известных bent-функций:")
    
    # f = x1*x3 + x2*x4 (пример bent-функции для n=6)
    # Строим таблицу истинности для этой функции
    truth_table_6 = []
    for x in range(2**6):
        x1 = (x >> 0) & 1
        x2 = (x >> 1) & 1
        x3 = (x >> 2) & 1
        x4 = (x >> 3) & 1
        value = (x1 & x3) ^ (x2 & x4)
        truth_table_6.append(value)
    
    bent_6_example = BooleanFunction(6, truth_table_6)
    nl_6 = nonlinearity(bent_6_example)
    print(f"f = x1*x3 + x2*x4: NL = {nl_6}")
    
    if nl_6 == 28:
        print("✓ Это bent-функция!")
    
    return bent_4_example, bent_6_example


# ============================================================================
# ЧАСТЬ 3: S-боксы и их нелинейность
# ============================================================================

class SBox:
    """Класс для представления S-бокса (n, m)"""
    
    def __init__(self, n: int, m: int, base_functions: List[BooleanFunction]):
        """
        Инициализация S-бокса
        
        Args:
            n: размерность входа
            m: размерность выхода
            base_functions: список m булевых функций от n переменных
        """
        if len(base_functions) != m:
            raise ValueError(f"Должно быть {m} базовых функций")
        
        if any(f.n != n for f in base_functions):
            raise ValueError(f"Все функции должны быть от {n} переменных")
        
        self.n = n
        self.m = m
        self.base_functions = base_functions
    
    def evaluate(self, x: int) -> int:
        """
        Вычисляет значение S-бокса на входе x
        
        Returns:
            Целое число, чьи биты - значения базовых функций
        """
        result = 0
        for i, f in enumerate(self.base_functions):
            result |= (f(x) << i)
        return result
    
    def get_all_linear_combinations(self) -> List[BooleanFunction]:
        """
        Генерирует все нетривиальные линейные комбинации базовых функций
        
        Returns:
            Список булевых функций - все линейные комбинации
        """
        combinations = []
        
        # Перебираем все возможные коэффициенты (кроме нулевого вектора)
        for coeffs_tuple in itertools.product([0, 1], repeat=self.m):
            if all(c == 0 for c in coeffs_tuple):
                continue  # Пропускаем нулевую комбинацию
            
            # Строим функцию-комбинацию
            truth_table = []
            for x in range(2**self.n):
                value = 0
                for i in range(self.m):
                    if coeffs_tuple[i] == 1:
                        value ^= self.base_functions[i](x)
                truth_table.append(value)
            
            combinations.append(BooleanFunction(self.n, truth_table))
        
        return combinations
    
    def nonlinearity(self) -> int:
        """
        Вычисляет степень нелинейности S-бокса как минимальную
        нелинейность среди всех линейных комбинаций базовых функций
        """
        combinations = self.get_all_linear_combinations()
        
        min_nl = float('inf')
        for combo in combinations:
            nl = nonlinearity(combo)
            min_nl = min(min_nl, nl)
        
        return min_nl
    
    def __repr__(self):
        return f"SBox({self.n}, {self.m})"


def create_sbox_from_function(n: int, f: BooleanFunction) -> SBox:
    """
    Создает S-бокс (n, n) из одной булевой функции
    путем создания n функций через линейные комбинации
    """
    if f.n != n:
        raise ValueError(f"Функция должна быть от {n} переменных")
    
    # Создаем базовые функции как линейные комбинации
    base_functions = []
    for mask in range(1, 2**n):  # Исключаем нулевую функцию
        # Строим линейную комбинацию с коэффициентом a = mask
        truth_table = []
        for x in range(2**n):
            value = 0
            for i in range(n):
                if (mask >> i) & 1:
                    # Добавляем i-й бит x
                    value ^= (x >> i) & 1
            value ^= f(x)  # Добавляем нелинейную часть
            truth_table.append(value)
        
        base_functions.append(BooleanFunction(n, truth_table))
        if len(base_functions) == n:
            break
    
    return SBox(n, n, base_functions)


def generate_random_sbox(n: int) -> SBox:
    """Генерирует случайный S-бокс размерности (n, n)"""
    import random
    
    base_functions = []
    for _ in range(n):
        # Генерируем случайную булеву функцию
        truth_table = [random.randint(0, 1) for _ in range(2**n)]
        base_functions.append(BooleanFunction(n, truth_table))
    
    return SBox(n, n, base_functions)


def generate_sbox_with_high_nonlinearity(n: int, max_iterations: int = 1000):
    """
    Генерирует S-бокс с высокой нелинейностью путем полного перебора
    
    Args:
        n: размерность S-бокса
        max_iterations: максимальное количество попыток
        
    Returns:
        Лучший найденный S-бокс и его нелинейность
    """
    import random
    
    print(f"\n{'='*70}")
    print(f"Генерирование S-бокса размерности {n} с высокой нелинейностью")
    print(f"{'='*70}")
    
    best_sbox = None
    best_nl = 0
    
    for iteration in range(max_iterations):
        sbox = generate_random_sbox(n)
        nl = sbox.nonlinearity()
        
        if nl > best_nl:
            best_nl = nl
            best_sbox = sbox
            print(f"Итерация {iteration + 1}: найден S-бокс с NL = {best_nl}")
        
        if (iteration + 1) % 100 == 0:
            print(f"  Проверено S-боксов: {iteration + 1}...", end='\r')
    
    print(f"\nЛучший найденный S-бокс (n={n}):")
    print(f"  Нелинейность: {best_nl}")
    
    return best_sbox, best_nl


# ============================================================================
# ПРИМЕРЫ И ТЕСТИРОВАНИЕ
# ============================================================================

def test_hamming_distance():
    """Тест вычисления расстояния Хэмминга"""
    print("\n" + "="*70)
    print("ТЕСТ 1: Расстояние Хэмминга")
    print("="*70)
    
    # Логическое И: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1
    and_func = BooleanFunction(2, [0, 0, 0, 1])
    
    # Логическое ИЛИ: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→1
    or_func = BooleanFunction(2, [0, 1, 1, 1])
    
    distance = BooleanFunction.hamming_distance(and_func, or_func)
    print(f"\nРасстояние между И и ИЛИ: {distance}")
    print(f"Ожидается: 2")
    
    # Норма Хэмминга
    norm = BooleanFunction.hamming_norm([1, 0, 1])
    print(f"\nНорма Хэмминга для (1,0,1): {norm}")


def test_nonlinearity():
    """Тест вычисления нелинейности"""
    print("\n" + "="*70)
    print("ТЕСТ 2: Степень нелинейности")
    print("="*70)
    
    # f = x1*x2 (нелинейная функция)
    x1x2 = BooleanFunction(2, [0, 0, 0, 1])
    nl = nonlinearity(x1x2)
    print(f"\nНелинейность для f = x1*x2: {nl}")
    print(f"Ожидается: 1")
    
    # f = x1 (линейная функция)
    x1 = BooleanFunction(2, [0, 0, 1, 1])
    nl = nonlinearity(x1)
    print(f"Нелинейность для f = x1: {nl}")
    print(f"Ожидается: 0")


def test_linear_functions():
    """Тест генерирования линейных функций"""
    print("\n" + "="*70)
    print("ТЕСТ 3: Линейные (аффинные) функции")
    print("="*70)
    
    linear_funcs = generate_all_linear_functions(2)
    print(f"\nКоличество линейных функций от 2 переменных: {len(linear_funcs)}")
    print(f"Ожидается: {2**(2+1)} = 8")
    
    # Выводим первые несколько
    print("\nПримеры линейных функций:")
    for i, lf in enumerate(linear_funcs[:4]):
        print(f"  {i+1}. {lf} → {lf.truth_table}")


def demonstrate_sbox():
    """Демонстрация работы S-бокса"""
    print("\n" + "="*70)
    print("ДЕМОНСТРАЦИЯ: S-бокс размерности (3, 2)")
    print("="*70)
    
    # Создаем функции для примера из теории
    f1 = BooleanFunction(3, [0, 0, 1, 1, 0, 0, 1, 1])  # проекция на x1
    f2 = BooleanFunction(3, [0, 1, 0, 1, 0, 1, 0, 1])  # проекция на x2
    
    sbox = SBox(3, 2, [f1, f2])
    
    print("\nТаблица S-бокса:")
    print("x (dec) | x (bin) | S(x) (bin) | S(x) (dec)")
    print("-" * 50)
    for x in range(2**3):
        x_bin = format(x, '03b')
        sx = sbox.evaluate(x)
        sx_bin = format(sx, '02b')
        print(f"{x:6d} | {x_bin:7s} | {sx_bin:10s} | {sx:10d}")
    
    print(f"\nНелинейность S-бокса: {sbox.nonlinearity()}")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция - запуск всех тестов и задач"""
    
    print("\n" + "#"*70)
    print("# ЛАБОРАТОРНАЯ РАБОТА 1")
    print("# Исследование свойств S-боксов и построение S-бокса")
    print("# наибольшей степени нелинейности")
    print("#"*70)
    
    # ТЕСТ 1: Расстояние Хэмминга
    test_hamming_distance()
    
    # ТЕСТ 2: Нелинейность
    test_nonlinearity()
    
    # ТЕСТ 3: Линейные функции
    test_linear_functions()
    
    # ЗАДАЧА 1: Демонстрация S-бокса
    demonstrate_sbox()
    
    # ЗАДАЧА 2: Bent-функции
    bent_4, bent_6 = build_bent_functions_specific()
    
    # ЗАДАЧА 3: Генерирование S-бокса с высокой нелинейностью
    print("\n" + "="*70)
    print("ЗАДАЧА 3: S-бокс размерности 5")
    print("="*70)
    
    sbox_5, nl_5 = generate_sbox_with_high_nonlinearity(5, max_iterations=100)
    
    # Выводим результаты
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"\n1. Bent-функция для n=4:")
    print(f"   Нелинейность: {nonlinearity(bent_4)} (max = 6)")
    
    print(f"\n2. Bent-функция для n=6:")
    print(f"   Нелинейность: {nonlinearity(bent_6)} (max = 28)")
    
    print(f"\n3. S-бокс размерности 5:")
    print(f"   Нелинейность: {nl_5}")


if __name__ == "__main__":
    main()