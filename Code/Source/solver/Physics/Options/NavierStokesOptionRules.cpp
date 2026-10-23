/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#include "Physics/Options/NavierStokesOptionRules.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

[[nodiscard]] std::optional<FE::Real> constantScalarValue(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value)
{
    const auto* real = std::get_if<FE::Real>(&value);
    if (real == nullptr) {
        return std::nullopt;
    }
    return *real;
}

} // namespace

void validateWallNormalDimension(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim,
    std::string_view context)
{
    const auto normal = normalizedWallNormal(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    for (int component = dim; component < 3; ++component) {
        if (std::abs(normal[static_cast<std::size_t>(component)]) > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) +
                " wall_normal must lie in the velocity space dimension");
        }
    }
}

void validateFiniteConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && !std::isfinite(*literal)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be finite");
    }
}

void validateNonnegativeConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && *literal < FE::Real{0.0}) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be nonnegative");
    }
}

void validatePositiveConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && !(*literal > FE::Real{0.0})) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be positive");
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
