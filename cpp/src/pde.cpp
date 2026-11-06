#include "optlib/pde.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "optlib/tridiag.hpp"

namespace optlib {

namespace {

struct Boundary {
    double lo, hi;
};

Boundary boundary_values(OptionType type, const Market& m, double s_min, double s_max,
                         double tau) {
    const double df_r = std::exp(-m.r * tau), df_q = std::exp(-m.q * tau);
    if (type == OptionType::Call) return {0.0, s_max * df_q - m.K * df_r};
    return {m.K * df_r - s_min * df_q, 0.0};
}

// One theta-scheme step from tau to tau + dt:
//   (I - theta dt A) V^{n+1} = (I + (1 - theta) dt A) V^n
// where A is the tridiagonal spatial operator with coefficients (l, d, u).
// v holds V^n on entry (all nodes, including boundaries) and V^{n+1} on exit.
void theta_step(std::vector<double>& v, double theta, double dt, double l, double d, double u,
                Boundary next, std::vector<double>& lower, std::vector<double>& diag,
                std::vector<double>& upper, std::vector<double>& rhs,
                std::vector<double>& scratch) {
    const std::size_t n_int = v.size() - 2;  // interior unknowns j = 1..N-1
    const double ex = (1.0 - theta) * dt;    // explicit weight
    const double im = theta * dt;            // implicit weight

    for (std::size_t k = 0; k < n_int; ++k) {
        const std::size_t j = k + 1;
        rhs[k] = v[j] + ex * (l * v[j - 1] + d * v[j] + u * v[j + 1]);
        lower[k] = -im * l;
        diag[k] = 1.0 - im * d;
        upper[k] = -im * u;
    }
    // Known boundary values at the new time level move to the right-hand side.
    rhs[0] += im * l * next.lo;
    rhs[n_int - 1] += im * u * next.hi;

    solve_tridiagonal(lower, diag, upper, rhs, scratch);
    v.front() = next.lo;
    v.back() = next.hi;
    std::copy(rhs.begin(), rhs.end(), v.begin() + 1);
}

// Cubic Lagrange interpolation of grid values at x, using the 4 nodes around x.
double interp_cubic(const std::vector<double>& v, double x0, double dx, double x) {
    const std::size_t n = v.size();
    const double pos = (x - x0) / dx;
    std::size_t i = static_cast<std::size_t>(std::floor(pos));
    i = std::clamp<std::size_t>(i, 1, n - 3);  // stencil i-1 .. i+2
    double result = 0.0;
    for (std::size_t a = i - 1; a <= i + 2; ++a) {
        double w = 1.0;
        for (std::size_t b = i - 1; b <= i + 2; ++b)
            if (b != a) w *= (pos - static_cast<double>(b)) / (static_cast<double>(a) - static_cast<double>(b));
        result += w * v[a];
    }
    return result;
}

}  // namespace

double pde_price(OptionType type, const Market& m, std::size_t n_space, std::size_t n_time,
                 double width_sd) {
    validate(m);
    if (n_space < 4 || n_space % 2 != 0)
        throw std::invalid_argument("n_space must be even and >= 4");
    if (n_time < 2) throw std::invalid_argument("n_time must be >= 2");
    if (!(width_sd > 0.0) || !std::isfinite(width_sd))
        throw std::invalid_argument("width_sd must be finite and > 0");
    if (m.T == 0.0)
        return type == OptionType::Call ? std::max(m.S - m.K, 0.0) : std::max(m.K - m.S, 0.0);
    if (m.sigma <= 0.0) throw std::invalid_argument("PDE solver requires sigma > 0");

    // Grid in x = ln S, centred on ln K so the payoff kink is a node.
    const double x_k = std::log(m.K);
    const double half_width = std::fabs(std::log(m.S / m.K)) + width_sd * m.sigma * std::sqrt(m.T);
    const double x0 = x_k - half_width;
    const double dx = 2.0 * half_width / static_cast<double>(n_space);
    const double dt = m.T / static_cast<double>(n_time);
    const double s_min = std::exp(x0), s_max = std::exp(x_k + half_width);

    // Spatial operator A V_j = l V_{j-1} + d V_j + u V_{j+1}:
    //   a V_xx + b V_x - r V with a = sigma^2/2, b = r - q - sigma^2/2,
    //   central differences for both derivatives.
    const double a = 0.5 * m.sigma * m.sigma;
    const double b = m.r - m.q - a;
    const double l = a / (dx * dx) - b / (2.0 * dx);
    const double d = -2.0 * a / (dx * dx) - m.r;
    const double u = a / (dx * dx) + b / (2.0 * dx);

    // Terminal condition (tau = 0): payoff at each node.
    std::vector<double> v(n_space + 1);
    for (std::size_t j = 0; j <= n_space; ++j) {
        const double s = std::exp(x0 + static_cast<double>(j) * dx);
        v[j] = type == OptionType::Call ? std::max(s - m.K, 0.0) : std::max(m.K - s, 0.0);
    }

    const std::size_t n_int = n_space - 1;
    std::vector<double> lower(n_int), diag(n_int), upper(n_int), rhs(n_int), scratch(n_int);

    // Rannacher start: the first step is two implicit-Euler half steps.
    const double half = 0.5 * dt;
    theta_step(v, 1.0, half, l, d, u, boundary_values(type, m, s_min, s_max, half), lower, diag,
               upper, rhs, scratch);
    theta_step(v, 1.0, half, l, d, u, boundary_values(type, m, s_min, s_max, dt), lower, diag,
               upper, rhs, scratch);
    // Remaining steps: Crank-Nicolson (theta = 1/2).
    for (std::size_t n = 1; n < n_time; ++n) {
        const double tau_next = static_cast<double>(n + 1) * dt;
        theta_step(v, 0.5, dt, l, d, u, boundary_values(type, m, s_min, s_max, tau_next), lower,
                   diag, upper, rhs, scratch);
    }

    return interp_cubic(v, x0, dx, std::log(m.S));
}

}  // namespace optlib
