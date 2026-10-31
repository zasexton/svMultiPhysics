/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_TIME_SERIES_H
#define SVMP_FE_SYSTEMS_TIME_SERIES_H

/**
 * @file TimeSeries.h
 * @brief Tabulated time data and time-function parameters of a system
 *
 * Time-dependent boundary data is a parameter of the forms, not a callback:
 *
 *   const auto Q = systems::timeFunction(system, "Q_inlet",
 *                                        systems::readTimeSeries("inflow.flw", TimeSeriesEnd::Periodic));
 *   bcs.add(std::make_unique<bc::EssentialBC>(inlet, Q * profile, "u"));   // strong value
 *   R = R - (Q * inner(n, v)).ds(outlet);                                   // weak term
 *
 * In a weak form the parameter is a slot of the system's parameter registry,
 * filled with f(t) at the state time whenever the system evaluates its
 * parameters, so the kernel stays JIT-strict and cacheable; its values agree
 * with those of a callback returning f(t) g(x) to round-off (the kernels
 * differ).  In a strong value (StrongDirichletConstraint) f is evaluated at
 * the time of each constraint update, and `Q * g(x)` gives the inhomogeneities
 * of that callback bit for bit.
 */

#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

class FESystem;

/// What a time series does outside its first and last time.
enum class TimeSeriesEnd : std::uint8_t {
    Clamp,   ///< the first value before the first time, the last after the last
    Periodic ///< repeats with the period t_last - t_first
};

/**
 * @brief A tabulated function of time, linear between its points
 *
 * Values at strictly increasing times t_0 < ... < t_{n-1}, with one or more
 * components per time.  The interpolation is that of the solver's
 * temporal-values files, bit for bit.
 */
class TimeSeries {
public:
    /// @p values holds @p components values per time, time by time.  Throws
    /// InvalidArgumentException for fewer than two times, times that do not
    /// increase strictly, non-finite entries or a size mismatch.
    TimeSeries(std::vector<Real> times, std::vector<Real> values, TimeSeriesEnd end, int components = 1);

    /// Component 0 at time @p t.
    [[nodiscard]] Real operator()(Real t) const { return value(t, 0); }

    /// Component @p component at time @p t.
    [[nodiscard]] Real value(Real t, int component = 0) const;

    [[nodiscard]] std::span<const Real> times() const noexcept { return times_; }
    [[nodiscard]] int components() const noexcept { return components_; }
    [[nodiscard]] TimeSeriesEnd end() const noexcept { return end_; }
    [[nodiscard]] Real sample(std::size_t time_index, int component = 0) const;

private:
    std::vector<Real> times_{};
    std::vector<Real> values_{};
    TimeSeriesEnd end_{TimeSeriesEnd::Clamp};
    int components_{1};
};

/**
 * @brief Reads a temporal-values file: "<n> <count>", then n rows
 * "t v_0 ... v_{components-1}" with strictly increasing t
 *
 * Throws InvalidArgumentException, naming the file and the row, when the file
 * cannot be read or is malformed.
 */
[[nodiscard]] TimeSeries readTimeSeries(const std::string& path, TimeSeriesEnd end, int components = 1);

/// One time function of a system, by name.
struct TimeFunctionDeclaration {
    std::string name{};
    std::function<Real(Real)> function{};
};

/// The time functions declared on a system (its attachment), in order.
struct TimeFunctionStore {
    std::vector<TimeFunctionDeclaration> declarations{};
};

/**
 * @brief Declares the parameter @p name of @p system with value f(t) at time t,
 * and returns `FormExpr::parameter(name)`
 *
 * Throws InvalidArgumentException after setup, for an empty name or function,
 * or for a name already declared.
 */
[[nodiscard]] forms::FormExpr timeFunction(FESystem& system, std::string name, std::function<Real(Real)> f);

/// The time functions declared on @p system, in declaration order (empty when none).
[[nodiscard]] std::span<const TimeFunctionDeclaration> timeFunctions(const FESystem& system) noexcept;

/// The time function @p name of @p system, or null when none is declared.
[[nodiscard]] const std::function<Real(Real)>* findTimeFunction(const FESystem& system, std::string_view name);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_TIME_SERIES_H
