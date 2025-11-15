#ifndef BIG_INT_H
#define BIG_INT_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <chrono>

class BigInt {
private:
    std::string value;
    bool is_negative;

    static std::string add(std::string a, std::string b);
    static std::string subtract(std::string a, std::string b);
    static std::string multiply(std::string a, std::string b);
    static std::pair<std::string, std::string> divide(std::string a, std::string b);
    static bool is_smaller(std::string a, std::string b);

public:
    BigInt() : value("0"), is_negative(false) {}
    BigInt(long long n);
    BigInt(std::string s);

    BigInt operator+(const BigInt& other) const;
    BigInt operator-(const BigInt& other) const;
    BigInt operator*(const BigInt& other) const;
    BigInt operator/(const BigInt& other) const;
    BigInt operator%(const BigInt& other) const;

    bool operator<(const BigInt& other) const;
    bool operator>(const BigInt& other) const;
    bool operator<=(const BigInt& other) const;
    bool operator>=(const BigInt& other) const;
    bool operator==(const BigInt& other) const;
    bool operator!=(const BigInt& other) const;

    BigInt operator-() const;

    std::string toString() const;
    friend std::ostream& operator<<(std::ostream& os, const BigInt& bi);

    static BigInt power(BigInt base, BigInt exp, BigInt mod);
    static BigInt random_in_range(const BigInt& min, const BigInt& max);
};

BigInt::BigInt(long long n) {
    if (n == 0) {
        value = "0";
        is_negative = false;
    } else {
        if (n < 0) {
            is_negative = true;
            n = -n;
        } else {
            is_negative = false;
        }
        value = std::to_string(n);
    }
}

BigInt::BigInt(std::string s) {
    if (s.empty() || (s.length() == 1 && s[0] == '0')) {
        value = "0";
        is_negative = false;
        return;
    }
    if (s[0] == '-') {
        is_negative = true;
        value = s.substr(1);
    } else {
        is_negative = false;
        value = s;
    }
    size_t first_digit = value.find_first_not_of('0');
    if (std::string::npos != first_digit) {
        value = value.substr(first_digit);
    } else {
        value = "0";
        is_negative = false;
    }
}

bool BigInt::is_smaller(std::string a, std::string b) {
    if (a.length() < b.length()) return true;
    if (a.length() > b.length()) return false;
    return a < b;
}

std::string BigInt::add(std::string a, std::string b) {
    std::string result = "";
    int carry = 0;
    int i = a.length() - 1;
    int j = b.length() - 1;
    while (i >= 0 || j >= 0 || carry) {
        int sum = carry;
        if (i >= 0) sum += a[i--] - '0';
        if (j >= 0) sum += b[j--] - '0';
        result += std::to_string(sum % 10);
        carry = sum / 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::string BigInt::subtract(std::string a, std::string b) {
    if (is_smaller(a, b)) throw std::invalid_argument("Subtraction would result in negative");
    std::string result = "";
    int borrow = 0;
    int i = a.length() - 1;
    int j = b.length() - 1;
    while (i >= 0) {
        int diff = (a[i] - '0') - borrow;
        if (j >= 0) diff -= (b[j--] - '0');
        if (diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result += std::to_string(diff);
        i--;
    }
    std::reverse(result.begin(), result.end());
    size_t first_digit = result.find_first_not_of('0');
    if (std::string::npos != first_digit) {
        return result.substr(first_digit);
    }
    return "0";
}

std::string BigInt::multiply(std::string a, std::string b) {
    if (a == "0" || b == "0") return "0";
    std::vector<int> res(a.length() + b.length(), 0);
    int i_n1 = 0, i_n2 = 0;
    for (int i = a.length() - 1; i >= 0; i--) {
        int carry = 0;
        int n1 = a[i] - '0';
        i_n2 = 0;
        for (int j = b.length() - 1; j >= 0; j--) {
            int n2 = b[j] - '0';
            int sum = n1 * n2 + res[i_n1 + i_n2] + carry;
            carry = sum / 10;
            res[i_n1 + i_n2] = sum % 10;
            i_n2++;
        }
        if (carry > 0) res[i_n1 + i_n2] += carry;
        i_n1++;
    }
    int i = res.size() - 1;
    while (i >= 0 && res[i] == 0) i--;
    if (i == -1) return "0";
    std::string s = "";
    while (i >= 0) s += std::to_string(res[i--]);
    return s;
}

std::pair<std::string, std::string> BigInt::divide(std::string dividend, std::string divisor) {
    if (divisor == "0") throw std::invalid_argument("Division by zero");
    if (is_smaller(dividend, divisor)) return {"0", dividend};
    std::string quotient = "";
    std::string current = "";
    for (char d : dividend) {
        current += d;
        int count = 0;
        while (BigInt(current) >= BigInt(divisor)) {
            current = subtract(current, divisor);
            count++;
        }
        quotient += std::to_string(count);
    }
    size_t first_digit = quotient.find_first_not_of('0');
    if (std::string::npos != first_digit) quotient = quotient.substr(first_digit);
    else quotient = "0";
    first_digit = current.find_first_not_of('0');
    if (std::string::npos != first_digit) current = current.substr(first_digit);
    else current = "0";
    return {quotient, current};
}

BigInt BigInt::operator+(const BigInt& other) const {
    if (is_negative == other.is_negative) {
        BigInt result;
        result.value = add(value, other.value);
        result.is_negative = is_negative;
        return result;
    } else {
        if (is_smaller(value, other.value)) {
            BigInt result;
            result.value = subtract(other.value, value);
            result.is_negative = other.is_negative;
            return result;
        } else {
            BigInt result;
            result.value = subtract(value, other.value);
            result.is_negative = is_negative;
            return result;
        }
    }
}

BigInt BigInt::operator-(const BigInt& other) const { return *this + (-other); }

BigInt BigInt::operator*(const BigInt& other) const {
    BigInt result;
    result.value = multiply(value, other.value);
    result.is_negative = (is_negative != other.is_negative) && (result.value != "0");
    return result;
}

BigInt BigInt::operator/(const BigInt& other) const {
    auto div_res = divide(value, other.value);
    BigInt result;
    result.value = div_res.first;
    result.is_negative = (is_negative != other.is_negative) && (result.value != "0");
    return result;
}

BigInt BigInt::operator%(const BigInt& other) const {
    auto div_res = divide(value, other.value);
    BigInt result;
    result.value = div_res.second;
    result.is_negative = is_negative;
    return result;
}

bool BigInt::operator<(const BigInt& other) const {
    if (is_negative != other.is_negative) return is_negative;
    if (is_negative) return is_smaller(other.value, value);
    return is_smaller(value, other.value);
}

bool BigInt::operator>(const BigInt& other) const { return other < *this; }
bool BigInt::operator<=(const BigInt& other) const { return !(*this > other); }
bool BigInt::operator>=(const BigInt& other) const { return !(*this < other); }
bool BigInt::operator==(const BigInt& other) const { return is_negative == other.is_negative && value == other.value; }
bool BigInt::operator!=(const BigInt& other) const { return !(*this == other); }

BigInt BigInt::operator-() const {
    BigInt result = *this;
    if (result.value != "0") result.is_negative = !is_negative;
    return result;
}

std::string BigInt::toString() const {
    if (is_negative) return "-" + value;
    return value;
}

std::ostream& operator<<(std::ostream& os, const BigInt& bi) {
    os << bi.toString();
    return os;
}

BigInt BigInt::power(BigInt base, BigInt exp, BigInt mod) {
    BigInt res(1);
    base = base % mod;
    while (exp > BigInt(0)) {
        if (exp % BigInt(2) == BigInt(1)) res = (res * base) % mod;
        base = (base * base) % mod;
        exp = exp / BigInt(2);
    }
    return res;
}

BigInt BigInt::random_in_range(const BigInt& min, const BigInt& max) {
    BigInt range = max - min + BigInt(1);
    int len = range.toString().length();
    std::string random_str = "";
    static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    for(int i = 0; i < len; ++i) {
        random_str += std::to_string(rng() % 10);
    }
    BigInt random_num(random_str);
    return min + (random_num % range);
}

#endif // BIG_INT_H
