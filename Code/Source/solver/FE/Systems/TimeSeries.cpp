/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/TimeSeries.h"

#include "Core/FEException.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

namespace svmp {
namespace FE {
namespace systems {

TimeSeries::TimeSeries(std::vector<Real> times, std::vector<Real> values, TimeSeriesEnd end, int components)
    : times_(std::move(times))
    , values_(std::move(values))
    , end_(end)
    , components_(components)
{
    FE_THROW_IF(components_ < 1, InvalidArgumentException, "TimeSeries: at least one component is required");
    FE_THROW_IF(times_.size() < 2u, InvalidArgumentException, "TimeSeries: at least two times are required");
    FE_THROW_IF(values_.size() != times_.size() * static_cast<std::size_t>(components_), InvalidArgumentException,
                "TimeSeries: expected " + std::to_string(times_.size() * static_cast<std::size_t>(components_)) +
                    " values, got " + std::to_string(values_.size()));
    for (std::size_t i = 0; i < times_.size(); ++i) {
        FE_THROW_IF(!std::isfinite(times_[i]), InvalidArgumentException, "TimeSeries: a time is not finite");
        FE_THROW_IF(i > 0 && !(times_[i] > times_[i - 1]), InvalidArgumentException,
                    "TimeSeries: the times must increase strictly");
    }
    FE_THROW_IF(std::any_of(values_.begin(), values_.end(), [](Real v) { return !std::isfinite(v); }),
                InvalidArgumentException, "TimeSeries: a value is not finite");
    FE_THROW_IF(end_ != TimeSeriesEnd::Clamp && end_ != TimeSeriesEnd::Periodic, InvalidArgumentException,
                "TimeSeries: invalid end behavior");
}

Real TimeSeries::sample(std::size_t time_index, int component) const
{
    FE_THROW_IF(time_index >= times_.size(), InvalidArgumentException, "TimeSeries::sample: time index out of range");
    FE_THROW_IF(component < 0 || component >= components_, InvalidArgumentException,
                "TimeSeries::sample: component out of range");
    return values_[time_index * static_cast<std::size_t>(components_) + static_cast<std::size_t>(component)];
}

Real TimeSeries::value(Real time, int component) const
{
    FE_THROW_IF(component < 0 || component >= components_, InvalidArgumentException,
                "TimeSeries::value: component out of range");

    // The interpolation of the temporal-values files, operation by operation.
    const std::size_t last = times_.size() - 1u;
    const Real t0 = times_.front();
    const Real t1 = times_.back();
    const Real span = t1 - t0;
    Real tt = time;
    if (end_ == TimeSeriesEnd::Periodic && span > 0.0 && std::isfinite(span)) {
        tt = std::fmod(time - t0, span);
        if (tt < 0.0) {
            tt += span;
        }
        tt += t0;
    }
    if (tt <= t0) {
        return sample(0, component);
    }
    if (tt >= t1) {
        return sample(last, component);
    }

    const auto upper = std::upper_bound(times_.begin(), times_.end(), tt);
    const auto i1 = static_cast<std::size_t>(std::distance(times_.begin(), upper));
    const std::size_t i0 = i1 > 0u ? i1 - 1u : 0u;
    const Real ta = times_[i0];
    const Real tb = times_[i1];
    const Real dt = tb - ta;
    const Real alpha = (dt > 0.0) ? ((tt - ta) / dt) : 0.0;
    const Real va = sample(i0, component);
    const Real vb = sample(i1, component);
    return (1.0 - alpha) * va + alpha * vb;
}

TimeSeries readTimeSeries(const std::string& path, TimeSeriesEnd end, int components)
{
    const std::string where = "FE::systems::readTimeSeries('" + path + "'): ";
    FE_THROW_IF(components < 1, InvalidArgumentException, where + "at least one component is required");
    std::ifstream in(path);
    FE_THROW_IF(!in.is_open(), InvalidArgumentException, where + "cannot open the file");

    long long count = 0;
    long long legacy_count = 0;
    in >> count >> legacy_count;
    FE_THROW_IF(in.fail() || count < 2, InvalidArgumentException,
                where + "the file must start with '<number of times> <count>' and hold at least two times");

    std::vector<Real> times(static_cast<std::size_t>(count));
    std::vector<Real> values(static_cast<std::size_t>(count) * static_cast<std::size_t>(components));
    for (long long i = 0; i < count; ++i) {
        const auto row = static_cast<std::size_t>(i);
        in >> times[row];
        FE_THROW_IF(in.fail(), InvalidArgumentException, where + "cannot read the time of row " + std::to_string(i));
        for (int c = 0; c < components; ++c) {
            in >> values[row * static_cast<std::size_t>(components) + static_cast<std::size_t>(c)];
            FE_THROW_IF(in.fail(), InvalidArgumentException,
                        where + "cannot read component " + std::to_string(c) + " of row " + std::to_string(i));
        }
    }
    return TimeSeries(std::move(times), std::move(values), end, components);
}

forms::FormExpr timeFunction(FESystem& system, std::string name, std::function<Real(Real)> f)
{
    const std::string where = "FE::systems::timeFunction('" + name + "'): ";
    FE_THROW_IF(system.isSetup(), InvalidArgumentException, where + "time functions cannot be declared after setup");
    FE_THROW_IF(name.empty(), InvalidArgumentException, where + "the name must be nonempty");
    FE_THROW_IF(!f, InvalidArgumentException, where + "the function is empty");
    auto& store = system.attachment<TimeFunctionStore>();
    FE_THROW_IF(std::any_of(store.declarations.begin(), store.declarations.end(),
                            [&](const TimeFunctionDeclaration& existing) { return existing.name == name; }),
                InvalidArgumentException, where + "a time function of this name is already declared");
    store.declarations.push_back(TimeFunctionDeclaration{name, std::move(f)});
    return forms::FormExpr::parameter(std::move(name));
}

std::span<const TimeFunctionDeclaration> timeFunctions(const FESystem& system) noexcept
{
    const auto* store = system.findAttachment<TimeFunctionStore>();
    if (store == nullptr) {
        return {};
    }
    return store->declarations;
}

const std::function<Real(Real)>* findTimeFunction(const FESystem& system, std::string_view name)
{
    for (const auto& declaration : timeFunctions(system)) {
        if (declaration.name == name) {
            return &declaration.function;
        }
    }
    return nullptr;
}

} // namespace systems
} // namespace FE
} // namespace svmp
