// pybind11 bindings: exposes Black-Scholes, Monte Carlo and Crank-Nicolson
// pricers as the Python module `optlib`.
//
// std::invalid_argument thrown by the core is translated to ValueError by
// pybind11's default exception translators.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

#include "optlib/black_scholes.hpp"
#include "optlib/monte_carlo.hpp"
#include "optlib/pde.hpp"

namespace py = pybind11;
using namespace optlib;

namespace {
Market make_market(double S, double K, double r, double sigma, double T, double q) {
    return Market{S, K, r, q, sigma, T};
}
}  // namespace

PYBIND11_MODULE(optlib, mod) {
    mod.doc() =
        "European option pricing: Black-Scholes closed form, Monte Carlo "
        "(antithetic / control variate) and Crank-Nicolson PDE.\n\n"
        "All functions take (option_type, S, K, r, sigma, T) positionally or by "
        "keyword; every further argument (q = continuous dividend yield, default "
        "0, plus n_paths/seed or n_space/n_time/width_sd) is keyword-only. Vega "
        "and rho are per 1.00 change; theta is dV/dt per year.";

    py::enum_<OptionType>(mod, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put);

    py::class_<McResult>(mod, "McResult")
        .def_readonly("price", &McResult::price)
        .def_readonly("std_error", &McResult::std_error, "standard error of price")
        .def_readonly("n_paths", &McResult::n_paths)
        .def("__repr__", [](const McResult& r) {
            return "McResult(price=" + std::to_string(r.price) +
                   ", std_error=" + std::to_string(r.std_error) +
                   ", n_paths=" + std::to_string(r.n_paths) + ")";
        });

    mod.def(
        "bs_price",
        [](OptionType t, double S, double K, double r, double sigma, double T, double q) {
            return bs_price(t, make_market(S, K, r, sigma, T, q));
        },
        py::arg("option_type"), py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"),
        py::arg("T"), py::kw_only(), py::arg("q") = 0.0, "Black-Scholes-Merton closed-form price.");

    mod.def(
        "bs_greeks",
        [](OptionType t, double S, double K, double r, double sigma, double T, double q) {
            const Greeks g = bs_greeks(t, make_market(S, K, r, sigma, T, q));
            py::dict d;
            d["delta"] = g.delta;
            d["gamma"] = g.gamma;
            d["vega"] = g.vega;
            d["theta"] = g.theta;
            d["rho"] = g.rho;
            return d;
        },
        py::arg("option_type"), py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"),
        py::arg("T"), py::kw_only(), py::arg("q") = 0.0,
        "Analytic Greeks as a dict: delta, gamma, vega (per 1.00 vol), theta (dV/dt per year), "
        "rho (per 1.00 rate).");

    using McFn = McResult (*)(OptionType, const Market&, std::size_t, std::uint64_t);
    auto def_mc = [&mod](const char* name, McFn fn, const char* doc) {
        mod.def(
            name,
            [fn](OptionType t, double S, double K, double r, double sigma, double T,
                 std::size_t n_paths, std::uint64_t seed, double q) {
                const Market m = make_market(S, K, r, sigma, T, q);
                py::gil_scoped_release release;
                return fn(t, m, n_paths, seed);
            },
            py::arg("option_type"), py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"),
            py::arg("T"), py::kw_only(), py::arg("n_paths"), py::arg("seed") = 42, py::arg("q") = 0.0, doc);
    };
    def_mc("mc_plain", &mc_plain, "Plain Monte Carlo price with standard error.");
    def_mc("mc_antithetic", &mc_antithetic,
           "Monte Carlo with antithetic variates; n_paths even, >= 4. SE over n_paths/2 pairs.");
    def_mc("mc_control_variate", &mc_control_variate,
           "Antithetic Monte Carlo with discounted S_T as control variate; n_paths even, >= 6.");

    mod.def(
        "pde_price",
        [](OptionType t, double S, double K, double r, double sigma, double T,
           std::size_t n_space, std::size_t n_time, double q, double width_sd) {
            const Market m = make_market(S, K, r, sigma, T, q);
            py::gil_scoped_release release;
            return pde_price(t, m, n_space, n_time, width_sd);
        },
        py::arg("option_type"), py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"),
        py::arg("T"), py::kw_only(), py::arg("n_space") = 400, py::arg("n_time") = 400, py::arg("q") = 0.0,
        py::arg("width_sd") = 6.0, "Crank-Nicolson finite-difference price.");
}
