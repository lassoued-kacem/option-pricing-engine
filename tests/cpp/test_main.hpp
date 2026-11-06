#pragma once
// Minimal dependency-free test harness for the C++ unit tests (run via CTest).

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace t {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}
inline int& failures() {
    static int f = 0;
    return f;
}

struct Reg {
    Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

inline void check_near(double got, double want, double tol, const char* expr, const char* file,
                       int line) {
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("  FAIL %s:%d  %s\n    got %.12g want %.12g |diff| %.3g > tol %.3g\n", file,
                    line, expr, got, want, std::fabs(got - want), tol);
        ++failures();
    }
}

inline void check_true(bool ok, const char* expr, const char* file, int line) {
    if (!ok) {
        std::printf("  FAIL %s:%d  %s\n", file, line, expr);
        ++failures();
    }
}

inline int run_all() {
    for (auto& c : registry()) {
        const int before = failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            std::printf("  FAIL %s threw: %s\n", c.name, e.what());
            ++failures();
        }
        std::printf("[%s] %s\n", failures() == before ? " OK " : "FAIL", c.name);
    }
    std::printf("%zu tests, %d failed checks\n", registry().size(), failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace t

#define T_CAT2(a, b) a##b
#define T_CAT(a, b) T_CAT2(a, b)
#define TEST(name)                                              \
    static void name();                                         \
    static t::Reg T_CAT(reg_, name)(#name, name);               \
    static void name()
#define CHECK_NEAR(got, want, tol) t::check_near((got), (want), (tol), #got, __FILE__, __LINE__)
#define CHECK(cond) t::check_true((cond), #cond, __FILE__, __LINE__)
#define CHECK_THROWS(stmt)                                      \
    do {                                                        \
        bool threw_ = false;                                    \
        try {                                                   \
            stmt;                                               \
        } catch (...) {                                         \
            threw_ = true;                                      \
        }                                                       \
        t::check_true(threw_, "throws: " #stmt, __FILE__, __LINE__); \
    } while (0)
#define TEST_MAIN() \
    int main() { return t::run_all(); }
