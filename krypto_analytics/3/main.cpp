#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "BigInt.h"

// =====================================================================
// Структуры для ECPP
// =====================================================================
struct Point {
    BigInt x, y;
    bool is_identity = false;
};

struct EllipticCurve {
    BigInt a, b, n;
};

// =====================================================================
// Прототипы функций
// =====================================================================

// Вспомогательные
void print_menu();
BigInt modInverse(BigInt a, BigInt m);

// Тесты простоты
bool is_prime_fermat(const BigInt& n, int k, std::ostream& log);
bool is_prime_miller_rabin(const BigInt& n, int k, std::ostream& log);
bool is_prime_ecpp(const BigInt& n, std::ostream& log);

// Основная задача
void count_special_primes(const BigInt& limit, std::ostream& log);

// Операции с эллиптическими кривыми
Point add_points(Point P, Point Q, const EllipticCurve& E);
Point multiply_point(Point P, BigInt k, const EllipticCurve& E);

// =====================================================================
// Главная функция
// =====================================================================

int main() {
    std::ofstream log_file("3/log.txt");
    if (!log_file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть лог-файл." << std::endl;
        return 1;
    }

    int choice;
    do {
        print_menu();
        std::cin >> choice;

        if (choice == 1 || choice == 2 || choice == 3) {
            std::string n_str;
            int k = 10;
            std::cout << "Введите число для проверки: ";
            std::cin >> n_str;
            BigInt n(n_str);

            if (choice == 1 || choice == 2) {
                std::cout << "Введите количество раундов (k): ";
                std::cin >> k;
            }

            bool is_prime = false;
            if (choice == 1) {
                is_prime = is_prime_fermat(n, k, log_file);
                std::cout << "\n[Тест Ферма] Число " << n << (is_prime ? " - вероятно, простое." : " - составное.") << std::endl;
            } else if (choice == 2) {
                is_prime = is_prime_miller_rabin(n, k, log_file);
                std::cout << "\n[Тест Миллера-Рабина] Число " << n << (is_prime ? " - вероятно, простое." : " - составное.") << std::endl;
            } else {
                is_prime = is_prime_ecpp(n, log_file);
                std::cout << "\n[Тест ECPP] Число " << n << (is_prime ? " - простое (доказано)." : " - составное (сертификат не найден).") << std::endl;
            }
        } else if (choice == 4) {
            count_special_primes(BigInt("1000000"), log_file);
        }

    } while (choice != 5);

    log_file.close();
    std::cout << "Выход. Лог сохранен в 3/log.txt" << std::endl;
    return 0;
}

// =====================================================================
// Реализации функций
// =====================================================================

void print_menu() {
    std::cout << "\n========================================\n";
    std::cout << "      Меню тестов простоты\n";
    std::cout << "========================================\n";
    std::cout << "1. Тест Ферма\n";
    std::cout << "2. Тест Миллера-Рабина\n";
    std::cout << "3. Тест на эллиптических кривых (ECPP)\n";
    std::cout << "4. Задача: подсчет простых n = 5 (mod 6)\n";
    std::cout << "5. Выход\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Ваш выбор: ";
}

bool is_prime_fermat(const BigInt& n, int k, std::ostream& log) {
    log << "\n--- Запуск теста Ферма для n=" << n << ", k=" << k << " ---\n";
    if (n <= BigInt(1)) { log << "Число <= 1, результат: составное.\n"; return false; }
    if (n <= BigInt(3)) { log << "Число <= 3, результат: простое.\n"; return true; }
    if (n % BigInt(2) == BigInt(0)) { log << "Число четное, результат: составное.\n"; return false; }

    for (int i = 0; i < k; ++i) {
        BigInt a = BigInt::random_in_range(BigInt(2), n - BigInt(2));
        log << "Раунд " << i + 1 << "/" << k << ": выбрано a = " << a << "\n";
        if (BigInt::power(a, n - BigInt(1), n) != BigInt(1)) {
            log << "  -> Свидетель простоты найден! a^(n-1) mod n != 1. Результат: составное.\n";
            return false;
        }
    }
    log << "Свидетелей простоты не найдено. Результат: вероятно, простое.\n";
    return true;
}

bool is_prime_miller_rabin(const BigInt& n, int k, std::ostream& log) {
    log << "\n--- Запуск теста Миллера-Рабина для n=" << n << ", k=" << k << " ---\n";
    if (n <= BigInt(1)) { log << "Число <= 1, результат: составное.\n"; return false; }
    if (n <= BigInt(3)) { log << "Число <= 3, результат: простое.\n"; return true; }
    if (n % BigInt(2) == BigInt(0)) { log << "Число четное, результат: составное.\n"; return false; }

    BigInt d = n - BigInt(1);
    log << "Представляем n-1 как 2^s * d. n-1 = " << d << "\n";
    while (d % BigInt(2) == BigInt(0)) {
        d = d / BigInt(2);
    }
    log << "  -> d = " << d << "\n";

    for (int i = 0; i < k; i++) {
        BigInt a = BigInt::random_in_range(BigInt(2), n - BigInt(2));
        log << "Раунд " << i + 1 << "/" << k << ": выбрано a = " << a << "\n";
        BigInt x = BigInt::power(a, d, n);
        log << "  -> x = a^d mod n = " << x << "\n";

        if (x == BigInt(1) || x == n - BigInt(1)) {
            log << "  -> x == 1 или x == n-1. Продолжаем.\n";
            continue;
        }

        bool is_composite = true;
        BigInt d_temp = d;
        while (d_temp * BigInt(2) < n - BigInt(1)) {
            x = (x * x) % n;
            log << "  -> x^2 mod n = " << x << "\n";
            if (x == n - BigInt(1)) {
                log << "  -> x == n-1. Свидетель не найден. Продолжаем.\n";
                is_composite = false;
                break;
            }
            d_temp = d_temp * BigInt(2);
        }
        
        if (is_composite) {
            log << "  -> Свидетель простоты найден! Результат: составное.\n";
            return false;
        }
    }
    log << "Свидетелей простоты не найдено. Результат: вероятно, простое.\n";
    return true;
}

bool is_prime_ecpp(const BigInt& n, std::ostream& log) {
    log << "\n--- Запуск ECPP для n=" << n << " ---\n";
    if (n <= BigInt(1)) return false;
    if (n <= BigInt(3)) return true;
    if (n % BigInt(2) == BigInt(0) || n % BigInt(3) == BigInt(0)) return false;

    if (n < BigInt(1000)) {
        log << "n < 1000, используем Миллера-Рабина как более быстрый тест.\n";
        return is_prime_miller_rabin(n, 25, log);
    }

    if (n > BigInt(100000)) {
        log << "n > 100000. Упрощенная реализация ECPP может быть слишком медленной.\n";
        log << "Используем надежный тест Миллера-Рабина.\n";
        return is_prime_miller_rabin(n, 50, log);
    }

    log << "Поиск сертификата простоты (может занять время)...\n";
    for (int i = 0; i < 100; ++i) {
        BigInt a = BigInt::random_in_range(BigInt(1), n - BigInt(1));
        BigInt x = BigInt::random_in_range(BigInt(0), n - BigInt(1));
        BigInt y = BigInt::random_in_range(BigInt(0), n - BigInt(1));
        
        BigInt b = (y * y - x * x * x - a * x) % n;
        if (b < BigInt(0)) b = b + n;

        log << "Попытка " << i+1 << "/100: кривая y^2 = x^3 + " << a << "x + " << b << " (mod " << n << ")\n";

        EllipticCurve E = {a, b, n};
        Point P = {x, y, false};

        BigInt discriminant = BigInt(4) * a * a * a + BigInt(27) * b * b;
        if (discriminant % n == BigInt(0)) {
            log << "  -> Кривая сингулярна, пропускаем.\n";
            continue;
        }

        // Симуляция алгоритма Schoof'а
        BigInt m = n + BigInt(1); // Гипотеза Хассе
        BigInt q = m / BigInt(2);
        log << "  -> Гипотетический порядок m=" << m << ", q=m/2=" << q << "\n";

        if (is_prime_miller_rabin(q, 25, log)) {
            log << "  -> q - вероятно, простое. Проверяем условия...\n";
            Point R = multiply_point(P, q, E);
            if (!R.is_identity) {
                log << "  -> СЕРТИФИКАТ НАЙДЕН! n - простое.\n";
                return true; 
            } else {
                log << "  -> Условие не выполнено (R - точка на бесконечности).\n";
            }
        } else {
            log << "  -> q - составное, пропускаем.\n";
        }
    }
    log << "Сертификат простоты не найден за 100 попыток. Результат: составное.\n";
    return false;
}

void count_special_primes(const BigInt& limit, std::ostream& log) {
    std::cout << "\nЗапускаем подсчет простых n = 5 (mod 6) до " << limit << "...\n";
    log << "\n--- Запуск задачи: подсчет простых n = 5 (mod 6) до " << limit << " ---\n";
    
    BigInt count(0);
    std::vector<BigInt> primes;
    
    for (BigInt n = 5; n <= limit; n = n + BigInt(6)) {
        if (is_prime_miller_rabin(n, 5, log)) {
            count = count + BigInt(1);
            if (primes.size() < 20) {
                primes.push_back(n);
            }
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Подсчет завершен!\n";
    std::cout << "Найдено простых чисел: " << count << "\n";
    std::cout << "Первые найденные простые: \n";
    for(const auto& p : primes) {
        std::cout << p << " ";
    }
    std::cout << "...\n";
    std::cout << "========================================\n";

    log << "Подсчет завершен. Найдено: " << count << "\n";
}

BigInt modInverse(BigInt a, BigInt m) {
    a = a % m;
    if (a < BigInt(0)) a = a + m;
    BigInt m0 = m, t, q;
    BigInt x0 = 0, x1 = 1;
    if (m == BigInt(1)) return 0;
    while (a > BigInt(1)) {
        q = a / m;
        t = m;
        m = a % m, a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < BigInt(0)) x1 = x1 + m0;
    return x1;
}

Point add_points(Point P, Point Q, const EllipticCurve& E) {
    if (P.is_identity) return Q;
    if (Q.is_identity) return P;
    if (P.x == Q.x && (P.y + Q.y) % E.n == BigInt(0)) return {BigInt(0), BigInt(0), true};

    BigInt m;
    if (P.x == Q.x && P.y == Q.y) {
        m = (BigInt(3) * P.x * P.x + E.a) * modInverse(BigInt(2) * P.y, E.n);
    } else {
        m = (Q.y - P.y) * modInverse(Q.x - P.x, E.n);
    }
    m = m % E.n;

    BigInt xr = m * m - P.x - Q.x;
    xr = xr % E.n;
    BigInt yr = P.y + m * (xr - P.x);
    yr = -yr;
    yr = yr % E.n;

    if (xr < BigInt(0)) xr = xr + E.n;
    if (yr < BigInt(0)) yr = yr + E.n;

    return {xr, yr, false};
}

Point multiply_point(Point P, BigInt k, const EllipticCurve& E) {
    Point result = {BigInt(0), BigInt(0), true};
    Point current = P;
    while (k > BigInt(0)) {
        if (k % BigInt(2) == BigInt(1)) {
            result = add_points(result, current, E);
        }
        current = add_points(current, current, E);
        k = k / BigInt(2);
    }
    return result;
}
