#pragma once

#include <cstddef>

namespace optlib {

enum class OptionType { Call, Put };

// European option contract plus flat market data.
//   S     spot price            (> 0)
//   K     strike                (> 0)
//   r     continuously compounded risk-free rate
//   q     continuous dividend yield (risk-neutral drift of S is r - q)
//   sigma volatility            (>= 0)
//   T     time to maturity in years (>= 0)
struct Market {
    double S;
    double K;
    double r;
    double q;
    double sigma;
    double T;
};

// Throws std::invalid_argument if any field is non-finite or out of range.
void validate(const Market& m);

}  // namespace optlib
