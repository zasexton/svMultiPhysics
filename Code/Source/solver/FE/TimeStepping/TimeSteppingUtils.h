/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_TIME_STEPPING_UTILS_H
#define SVMP_FE_TIMESTEPPING_TIME_STEPPING_UTILS_H

#include "Core/FEException.h"
#include "Math/FiniteDifference.h"
#include "TimeStepping/TimeHistory.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {
namespace utils {

[[nodiscard]] inline double clampDt(double dt, double min_dt, double max_dt) noexcept
{
    double out = dt;
    if (min_dt > 0.0) {
        out = std::max(out, min_dt);
    }
    if (max_dt > 0.0) {
        out = std::min(out, max_dt);
    }
    return out;
}

struct GeneralizedAlphaFirstOrderParams {
    double alpha_m{0.0};
    double alpha_f{0.0};
    double gamma{0.0};
};

/**
 * @brief Convert spectral radius at infinity to generalized-α parameters for 1st-order systems.
 *
 * Jansen–Whiting–Hulbert (2000) style parameterization:
 *   alpha_m = (3 - rho_inf) / (2 * (1 + rho_inf))
 *   alpha_f = 1 / (1 + rho_inf)
 *   gamma   = 1/2 + alpha_m - alpha_f
 */
[[nodiscard]] inline GeneralizedAlphaFirstOrderParams
generalizedAlphaFirstOrderFromRhoInf(double rho_inf)
{
    FE_THROW_IF(!(rho_inf >= 0.0 && rho_inf <= 1.0) || !std::isfinite(rho_inf),
                InvalidArgumentException,
                "generalizedAlphaFirstOrderFromRhoInf: rho_inf must be finite and in [0,1]");

    GeneralizedAlphaFirstOrderParams p;
    p.alpha_m = (3.0 - rho_inf) / (2.0 * (1.0 + rho_inf));
    p.alpha_f = 1.0 / (1.0 + rho_inf);
    p.gamma = 0.5 + p.alpha_m - p.alpha_f;
    return p;
}

struct GeneralizedAlphaSecondOrderParams {
    double alpha_m{0.0};
    double alpha_f{0.0};
    double gamma{0.0};
    double beta{0.0};
};

/**
 * @brief Convert spectral radius at infinity to generalized-α parameters for 2nd-order systems.
 *
 * Chung–Hulbert (1993) spectral-radius parameterization expressed in this library’s stage convention:
 *   u_stage = (1 - alpha_f) * u_n + alpha_f * u_{n+1}
 *   a_stage = (1 - alpha_m) * a_n + alpha_m * a_{n+1}
 *
 * With this convention the commonly published Chung–Hulbert parameters
 *   alpha_m^CH = (2*rho_inf - 1) / (rho_inf + 1),
 *   alpha_f^CH = rho_inf / (rho_inf + 1)
 * are transformed as:
 *   alpha_m = 1 - alpha_m^CH = (2 - rho_inf) / (1 + rho_inf)
 *   alpha_f = 1 - alpha_f^CH = 1 / (1 + rho_inf)
 *
 * The resulting (beta,gamma) match the standard second-order accuracy conditions:
 *   gamma = 1/2 + alpha_m - alpha_f
 *   beta  = 1/4 * (1 + alpha_m - alpha_f)^2
 */
[[nodiscard]] inline GeneralizedAlphaSecondOrderParams
generalizedAlphaSecondOrderFromRhoInf(double rho_inf)
{
    FE_THROW_IF(!(rho_inf >= 0.0 && rho_inf <= 1.0) || !std::isfinite(rho_inf),
                InvalidArgumentException,
                "generalizedAlphaSecondOrderFromRhoInf: rho_inf must be finite and in [0,1]");

    GeneralizedAlphaSecondOrderParams p;
    p.alpha_m = (2.0 - rho_inf) / (1.0 + rho_inf);
    p.alpha_f = 1.0 / (1.0 + rho_inf);
    p.gamma = 0.5 + p.alpha_m - p.alpha_f;
    const double c = 1.0 + p.alpha_m - p.alpha_f;
    p.beta = 0.25 * c * c;
    return p;
}

struct SecondOrderStateInitReport {
    bool initialized_velocity{false};
    bool initialized_acceleration{false};
    int velocity_points{0};
    int acceleration_points{0};
};

/**
 * @brief Initialize (u̇,ü) by differentiating displacement history in-place.
 *
 * Uses Fornberg finite-difference weights on the time grid implied by
 * `TimeHistory::dtHistory()`. If insufficient displacement history exists for a
 * derivative order, the corresponding output is left unchanged.
 *
 * Notes:
 * - Velocity uses at least 2 displacement states.
 * - Acceleration uses at least 3 displacement states.
 */
[[nodiscard]] inline SecondOrderStateInitReport
initializeSecondOrderStateFromDisplacementHistory(const TimeHistory& history,
                                                  std::span<Real> out_u_dot,
                                                  std::span<Real> out_u_ddot,
                                                  bool overwrite_u_dot = true,
                                                  bool overwrite_u_ddot = true,
                                                  int max_points = 6)
{
    SecondOrderStateInitReport rep;

    const auto u_hist = history.uHistorySpans();
    FE_THROW_IF(u_hist.empty(), InvalidArgumentException,
                "initializeSecondOrderStateFromDisplacementHistory: TimeHistory has no displacement history");

    FE_CHECK_ARG(out_u_dot.size() == u_hist[0].size(), "initializeSecondOrderStateFromDisplacementHistory: uDot size mismatch");
    FE_CHECK_ARG(out_u_ddot.size() == u_hist[0].size(), "initializeSecondOrderStateFromDisplacementHistory: uDDot size mismatch");

    const double dt_prev = (history.dtPrev() > 0.0 && std::isfinite(history.dtPrev()))
        ? history.dtPrev()
        : history.dt();
    FE_THROW_IF(!(dt_prev > 0.0) || !std::isfinite(dt_prev), InvalidArgumentException,
                "initializeSecondOrderStateFromDisplacementHistory: dtPrev must be finite and > 0");

    const auto dt_hist = history.dtHistory();
    auto historyDt = [&](int idx) -> double {
        if (idx < 0 || idx >= static_cast<int>(dt_hist.size())) {
            return dt_prev;
        }
        const double v = dt_hist[static_cast<std::size_t>(idx)];
        if (v > 0.0 && std::isfinite(v)) {
            return v;
        }
        return dt_prev;
    };

    auto fillDerivative = [&](int derivative_order,
                              std::span<Real> dst,
                              int min_points,
                              int& points_used,
                              bool& flag,
                              bool overwrite) {
        if (!overwrite) {
            points_used = 0;
            return;
        }
        const int available = static_cast<int>(u_hist.size());
        const int n_points = std::min(available, std::max(1, max_points));
        if (n_points < min_points) {
            points_used = 0;
            return;
        }

        std::vector<double> nodes;
        nodes.reserve(static_cast<std::size_t>(n_points));
        nodes.push_back(0.0);
        double accum = 0.0;
        for (int j = 1; j < n_points; ++j) {
            accum += historyDt(j - 1);
            nodes.push_back(-accum);
        }

        const auto w = math::finiteDifferenceWeights(derivative_order, /*x0=*/0.0, nodes);
        FE_THROW_IF(static_cast<int>(w.size()) != n_points, InvalidArgumentException,
                    "initializeSecondOrderStateFromDisplacementHistory: internal weight size mismatch");

        std::fill(dst.begin(), dst.end(), static_cast<Real>(0.0));
        for (int j = 0; j < n_points; ++j) {
            const auto uj = u_hist[static_cast<std::size_t>(j)];
            const double alpha = w[static_cast<std::size_t>(j)];
            for (std::size_t i = 0; i < dst.size(); ++i) {
                dst[i] += static_cast<Real>(alpha) * uj[i];
            }
        }

        points_used = n_points;
        flag = true;
    };

    fillDerivative(/*derivative_order=*/1, out_u_dot, /*min_points=*/2,
                   rep.velocity_points, rep.initialized_velocity, overwrite_u_dot);
    fillDerivative(/*derivative_order=*/2, out_u_ddot, /*min_points=*/3,
                   rep.acceleration_points, rep.initialized_acceleration, overwrite_u_ddot);

    return rep;
}

} // namespace utils
} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_TIME_STEPPING_UTILS_H
