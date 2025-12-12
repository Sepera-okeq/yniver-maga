"""
Лабораторная работа 2: Построение таблицы линейных аппроксимаций S-бокса
Все варианты (1-20) с возможностью выбора
"""

import numpy as np
from itertools import product

# Все варианты S-боксов
VARIANTS = {
    1: {
        'F1': [0,1,0,1,0,1,0,1,1,0,0,0,1,0,1,0],
        'F2': [0,1,1,1,1,0,1,0,0,1,0,1,1,0,1,0],
        'F3': [1,1,0,0,0,0,1,1,0,0,1,0,1,1,0,0],
        'F4': [0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0]
    },
    2: {
        'F1': [0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0],
        'F2': [0,0,1,1,0,1,1,1,1,1,0,0,1,1,0,0],
        'F3': [0,1,0,1,1,0,0,0,0,1,0,1,1,0,1,0],
        'F4': [1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1]
    },
    3: {
        'F1': [1,1,1,0,0,0,1,1,0,0,1,1,1,1,0,0],
        'F2': [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1],
        'F3': [0,1,0,0,1,0,1,0,0,1,0,1,1,0,1,0],
        'F4': [0,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1]
    },
    4: {
        'F1': [0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,0],
        'F2': [0,1,1,0,0,0,1,0,1,0,0,1,1,0,0,1],
        'F3': [1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1],
        'F4': [0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0]
    },
    5: {
        'F1': [0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,1],
        'F2': [0,1,1,0,1,0,0,1,0,1,1,1,1,0,0,1],
        'F3': [1,1,1,0,1,1,0,0,0,0,1,1,0,0,1,1],
        'F4': [0,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0]
    },
    6: {
        'F1': [0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,1],
        'F2': [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1],
        'F3': [1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,1],
        'F4': [0,0,0,1,1,1,0,0,1,1,0,0,0,0,1,1]
    },
    7: {
        'F1': [0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0],
        'F2': [0,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0],
        'F3': [1,0,1,0,0,1,0,1,1,1,1,0,0,1,0,1],
        'F4': [0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,1]
    },
    8: {
        'F1': [0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0],
        'F2': [0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0],
        'F3': [0,1,1,1,1,0,0,1,0,1,1,0,1,0,0,1],
        'F4': [1,0,0,0,0,1,0,1,1,0,1,0,0,1,0,1]
    },
    9: {
        'F1': [0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,0],
        'F2': [1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0],
        'F3': [0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0],
        'F4': [0,0,0,0,1,1,1,1,1,1,1,1,0,1,0,0]
    },
    10: {
        'F1': [1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,1],
        'F2': [0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1],
        'F3': [0,1,0,1,1,0,1,0,1,0,0,0,0,1,0,1],
        'F4': [0,1,1,0,0,1,1,0,1,0,0,0,1,0,0,1]
    },
    11: {
        'F1': [1,1,0,1,0,1,1,0,1,0,0,1,0,1,1,0],
        'F2': [0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0],
        'F3': [0,1,1,0,0,1,1,0,1,0,0,0,1,0,0,1],
        'F4': [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1]
    },
    12: {
        'F1': [0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,0],
        'F2': [0,1,0,1,0,1,0,1,1,0,1,0,0,0,1,0],
        'F3': [0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0],
        'F4': [1,0,0,0,0,1,0,1,1,0,1,0,0,1,0,1]
    },
    13: {
        'F1': [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1],
        'F2': [0,1,0,1,0,1,0,1,1,0,1,0,1,0,0,0],
        'F3': [1,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1],
        'F4': [0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1]
    },
    14: {
        'F1': [1,0,1,0,0,1,0,1,0,1,0,1,0,0,1,0],
        'F2': [0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0],
        'F3': [0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0],
        'F4': [0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,0]
    },
    15: {
        'F1': [0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0],
        'F2': [0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,0],
        'F3': [1,1,0,0,0,0,1,1,0,0,1,1,1,0,0,0],
        'F4': [0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0]
    },
    16: {
        'F1': [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
        'F2': [1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0],
        'F3': [0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0],
        'F4': [0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0]
    },
    17: {
        'F1': [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1],
        'F2': [0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,1],
        'F3': [0,1,0,1,0,1,0,1,1,0,1,0,1,0,0,0],
        'F4': [1,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1]
    },
    18: {
        'F1': [0,1,0,1,1,0,1,0,1,0,1,0,0,0,0,1],
        'F2': [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1],
        'F3': [0,1,0,1,0,1,0,1,1,0,1,0,1,0,0,0],
        'F4': [1,1,0,0,0,0,1,1,0,1,1,1,1,1,0,0]
    },
    19: {
        'F1': [1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,1],
        'F2': [0,0,0,1,0,1,0,1,1,0,1,0,1,0,1,0],
        'F3': [0,1,0,1,1,0,1,0,1,1,1,0,0,1,0,1],
        'F4': [0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,0]
    },
    20: {
        'F1': [0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1],
        'F2': [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
        'F3': [1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,1],
        'F4': [0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1]
    }
}

# Размерность S-бокса
n = 4  # количество входных бит
m = 4  # количество выходных бит

def xor_sum(bits):
    """Вычисляет XOR сумму битов"""
    result = 0
    for bit in bits:
        result ^= bit
    return result

def get_output(variant_data, x1, x2, x3, x4):
    """Получить выходные значения S-бокса для входа (x1, x2, x3, x4)"""
    index = x1 * 8 + x2 * 4 + x3 * 2 + x4
    return (variant_data['F1'][index], 
            variant_data['F2'][index], 
            variant_data['F3'][index], 
            variant_data['F4'][index])

def build_linear_approximation_table(variant_data):
    """
    Построение таблицы линейных аппроксимаций
    Размер таблицы: 2^n x 2^m = 16 x 16
    """
    table = np.zeros((2**m, 2**n), dtype=int)
    
    # Перебираем все возможные комбинации коэффициентов A (входные биты)
    for a_idx, (a1, a2, a3, a4) in enumerate(product([0, 1], repeat=n)):
        # Перебираем все возможные комбинации коэффициентов B (выходные биты)
        for b_idx, (b1, b2, b3, b4) in enumerate(product([0, 1], repeat=m)):
            count = 0
            
            # Перебираем все возможные входные значения X
            for x1, x2, x3, x4 in product([0, 1], repeat=n):
                # Получаем выходные значения Y
                y1, y2, y3, y4 = get_output(variant_data, x1, x2, x3, x4)
                
                # Вычисляем левую часть: a1*X1 + a2*X2 + a3*X3 + a4*X4
                left = xor_sum([a1*x1, a2*x2, a3*x3, a4*x4])
                
                # Вычисляем правую часть: b1*Y1 + b2*Y2 + b3*Y3 + b4*Y4
                right = xor_sum([b1*y1, b2*y2, b3*y3, b4*y4])
                
                # Проверяем равенство
                if left == right:
                    count += 1
            
            # Записываем в таблицу (count - среднее значение)
            c_avg = 2**(n-1)  # для n=4, c_avg = 8
            table[b_idx, a_idx] = count - c_avg
    
    return table

def format_binary(num, bits):
    """Форматирование числа в двоичный вид"""
    return format(num, f'0{bits}b')

def print_table(table, variant_num):
    """Красивый вывод таблицы линейных аппроксимаций"""
    print(f"\n{'='*150}")
    print(f"Таблица линейных аппроксимаций S-бокса (Вариант {variant_num})")
    print(f"{'='*150}")
    
    # Заголовок столбцов (коэффициенты A)
    print("B\\A  ", end="")
    for a in range(2**n):
        print(f"{format_binary(a, n):>6}", end=" ")
    print()
    print("-" * 150)
    
    # Строки таблицы
    for b in range(2**m):
        print(f"{format_binary(b, m):>4} ", end="")
        for a in range(2**n):
            value = table[b, a]
            print(f"{value:>6}", end=" ")
        print()
    print("=" * 150)

def find_best_approximations(table, top_n=10):
    """Найти лучшие линейные аппроксимации (с максимальным отклонением от 0)"""
    approximations = []
    
    # Пропускаем первую строку (B = 0000), так как она тривиальна
    for b in range(1, 2**m):
        for a in range(2**n):
            value = table[b, a]
            if value != 0:  # Интересуют только ненулевые значения
                approximations.append({
                    'A': format_binary(a, n),
                    'B': format_binary(b, m),
                    'value': value,
                    'abs_value': abs(value),
                    'count': value + 8  # Восстанавливаем исходное количество совпадений
                })
    
    # Сортируем по абсолютному значению (по убыванию)
    approximations.sort(key=lambda x: x['abs_value'], reverse=True)
    
    return approximations[:top_n]

def create_linear_equation(a_bits, b_bits, value):
    """Создать линейное уравнение из битовых векторов"""
    left_terms = []
    right_terms = []
    
    # Левая часть (входные биты X)
    for i, bit in enumerate(a_bits):
        if bit == '1':
            left_terms.append(f"X{i+1}")
    
    # Правая часть (выходные биты Y)
    for i, bit in enumerate(b_bits):
        if bit == '1':
            right_terms.append(f"Y{i+1}")
    
    # Формируем уравнение
    left = " + ".join(left_terms) if left_terms else "0"
    right = " + ".join(right_terms) if right_terms else "0"
    
    # Если значение отрицательное, добавляем +1 в правую часть
    if value < 0:
        right += " + 1"
    
    return f"{left} = {right}"

def print_best_approximations(approximations, variant_num):
    """Вывод лучших аппроксимаций"""
    print(f"\n{'='*100}")
    print(f"Лучшие линейные аппроксимации для варианта {variant_num}")
    print(f"{'='*100}")
    print(f"{'№':<4} {'A (вход)':<12} {'B (выход)':<12} {'Отклонение':<12} {'Совпадений':<12} {'Уравнение':<40}")
    print("-" * 100)
    
    for i, approx in enumerate(approximations, 1):
        equation = create_linear_equation(approx['A'], approx['B'], approx['value'])
        print(f"{i:<4} {approx['A']:<12} {approx['B']:<12} {approx['value']:<12} "
              f"{approx['count']:<12} {equation:<40}")
    
    print("=" * 100)

def create_approximation_system(approximations, num_equations=5):
    """Создать систему линейных уравнений"""
    print(f"\n{'='*100}")
    print(f"Система линейных уравнений, аппроксимирующих S-бокс (топ-{num_equations} уравнений)")
    print(f"{'='*100}")
    
    equations = []
    for i, approx in enumerate(approximations[:num_equations], 1):
        equation = create_linear_equation(approx['A'], approx['B'], approx['value'])
        equations.append(equation)
        print(f"{i}. {equation}")
    
    print("\nИли в виде системы:")
    print("⎧")
    for i, eq in enumerate(equations):
        if i == len(equations) - 1:
            print(f"⎩ {eq}")
        else:
            print(f"⎨ {eq}")
    
    print("=" * 100)
    
    return equations

def analyze_nonlinearity(table):
    """Анализ степени нелинейности S-бокса"""
    # Находим максимальное абсолютное значение в таблице (исключая первую строку)
    max_abs_value = 0
    for b in range(1, 2**m):
        for a in range(2**n):
            max_abs_value = max(max_abs_value, abs(table[b, a]))
    
    print(f"\n{'='*100}")
    print("Анализ нелинейности S-бокса")
    print(f"{'='*100}")
    print(f"Максимальное отклонение от среднего: {max_abs_value}")
    print(f"Максимальное количество совпадений: {max_abs_value + 8}")
    
    if max_abs_value == 8:
        print("Степень нелинейности: 0 (S-бокс полностью ЛИНЕЙНЫЙ)")
    elif max_abs_value >= 6:
        print(f"Степень нелинейности: НИЗКАЯ (значение {max_abs_value} близко к линейному)")
    elif max_abs_value >= 4:
        print(f"Степень нелинейности: СРЕДНЯЯ")
    else:
        print(f"Степень нелинейности: ВЫСОКАЯ (хорошо для криптографии)")
    
    print("=" * 100)

def main():
    """Главная функция программы"""
    print("="*100)
    print("Лабораторная работа 2: Построение таблицы линейных аппроксимаций S-бокса")
    print("="*100)
    
    # Выбор варианта
    while True:
        try:
            print("\nДоступные варианты: 1-20")
            variant_num = int(input("Введите номер варианта (или 0 для выхода): "))
            
            if variant_num == 0:
                print("Выход из программы.")
                break
            
            if variant_num not in VARIANTS:
                print(f"Ошибка: вариант {variant_num} не существует. Выберите от 1 до 20.")
                continue
            
            # Получаем данные варианта
            variant_data = VARIANTS[variant_num]
            
            print(f"\nВыбран вариант {variant_num}")
            print(f"F1 = {variant_data['F1']}")
            print(f"F2 = {variant_data['F2']}")
            print(f"F3 = {variant_data['F3']}")
            print(f"F4 = {variant_data['F4']}")
            
            # Построение таблицы линейных аппроксимаций
            print("\nПостроение таблицы линейных аппроксимаций...")
            table = build_linear_approximation_table(variant_data)
            
            # Вывод таблицы
            print_table(table, variant_num)
            
            # Анализ нелинейности
            analyze_nonlinearity(table)
            
            # Поиск лучших аппроксимаций
            best_approximations = find_best_approximations(table, top_n=15)
            print_best_approximations(best_approximations, variant_num)
            
            # Создание системы уравнений
            create_approximation_system(best_approximations, num_equations=5)
            
            # Предложение сохранить результаты
            save = input("\nСохранить результаты в файл? (y/n): ").lower()
            if save == 'y':
                filename = f"variant_{variant_num}_results.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    # Перенаправляем вывод в файл
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = f
                    
                    print(f"Результаты для варианта {variant_num}")
                    print(f"F1 = {variant_data['F1']}")
                    print(f"F2 = {variant_data['F2']}")
                    print(f"F3 = {variant_data['F3']}")
                    print(f"F4 = {variant_data['F4']}")
                    print_table(table, variant_num)
                    analyze_nonlinearity(table)
                    print_best_approximations(best_approximations, variant_num)
                    create_approximation_system(best_approximations, num_equations=5)
                    
                    sys.stdout = old_stdout
                
                print(f"Результаты сохранены в файл: {filename}")
            
            # Предложение продолжить
            continue_choice = input("\nПроанализировать другой вариант? (y/n): ").lower()
            if continue_choice != 'y':
                print("Выход из программы.")
                break
                
        except ValueError:
            print("Ошибка: введите целое число.")
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем.")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
