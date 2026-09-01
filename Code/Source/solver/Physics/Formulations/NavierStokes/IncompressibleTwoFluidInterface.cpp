/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidInterface.h"

#include "FE/Forms/Vocabulary.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

void validateParameters(
    const IncompressibleTwoFluidInterfaceParameters &parameters) {
  if (parameters.dimension != 2 && parameters.dimension != 3) {
    throw std::invalid_argument(
        "incompressible two-fluid interface requires dimension 2 or 3");
  }
  if (parameters.interface_marker < 0) {
    throw std::invalid_argument(
        "incompressible two-fluid interface marker must be nonnegative");
  }
  const auto finite_positive = [](FE::Real value) {
    return std::isfinite(value) && value > FE::Real{0.0};
  };
  if (!finite_positive(parameters.negative_density) ||
      !finite_positive(parameters.positive_density)) {
    throw std::invalid_argument("incompressible two-fluid interface densities "
                                "must be finite and positive");
  }
  if (!finite_positive(parameters.negative_viscosity) ||
      !finite_positive(parameters.positive_viscosity)) {
    throw std::invalid_argument("incompressible two-fluid interface "
                                "viscosities must be finite and positive");
  }
  if (!finite_positive(parameters.nitsche_gamma)) {
    throw std::invalid_argument("incompressible two-fluid interface Nitsche "
                                "gamma must be finite and positive");
  }
  if (!std::isfinite(parameters.surface_tension) ||
      parameters.surface_tension < FE::Real{0.0}) {
    throw std::invalid_argument("incompressible two-fluid interface surface "
                                "tension must be finite and nonnegative");
  }
  if (parameters.prescribed_pressure_jump.has_value() &&
      !std::isfinite(*parameters.prescribed_pressure_jump)) {
    throw std::invalid_argument(
        "incompressible two-fluid prescribed pressure jump must be finite");
  }
}

void validateExpression(const FE::forms::FormExpr &expression,
                        const char *name) {
  if (!expression.isValid()) {
    throw std::invalid_argument(
        std::string("incompressible two-fluid interface requires ") + name);
  }
}

struct StablePositivePair {
  FE::Real first_fraction{0.5};
  FE::Real second_fraction{0.5};
  FE::Real harmonic_mean{1.0};
};

[[nodiscard]] StablePositivePair stablePositivePair(FE::Real first,
                                                    FE::Real second) {
  const auto maximum = std::max(first, second);
  const auto first_scaled = first / maximum;
  const auto second_scaled = second / maximum;
  const auto scaled_sum = first_scaled + second_scaled;
  const auto minimum = std::min(first, second);
  return StablePositivePair{
      .first_fraction = first_scaled / scaled_sum,
      .second_fraction = second_scaled / scaled_sum,
      .harmonic_mean =
          minimum * (FE::Real{2.0} / (FE::Real{1.0} + minimum / maximum)),
  };
}

} // namespace

IncompressibleTwoFluidInterfaceWeights incompressibleTwoFluidInterfaceWeights(
    const IncompressibleTwoFluidInterfaceParameters &parameters) {
  validateParameters(parameters);

  const auto viscosity_pair = stablePositivePair(parameters.negative_viscosity,
                                                 parameters.positive_viscosity);
  const auto density_pair = stablePositivePair(parameters.negative_density,
                                               parameters.positive_density);
  return IncompressibleTwoFluidInterfaceWeights{
      .negative_traction = viscosity_pair.second_fraction,
      .positive_traction = viscosity_pair.first_fraction,
      .negative_complement = viscosity_pair.first_fraction,
      .positive_complement = viscosity_pair.second_fraction,
      .harmonic_viscosity = viscosity_pair.harmonic_mean,
      .harmonic_density = density_pair.harmonic_mean,
  };
}

IncompressibleTwoFluidInterfaceForms buildIncompressibleTwoFluidInterfaceForms(
    const FE::forms::FormExpr &negative_velocity,
    const FE::forms::FormExpr &negative_pressure,
    const FE::forms::FormExpr &negative_velocity_test,
    const FE::forms::FormExpr &negative_pressure_test,
    const FE::forms::FormExpr &positive_velocity,
    const FE::forms::FormExpr &positive_pressure,
    const FE::forms::FormExpr &positive_velocity_test,
    const FE::forms::FormExpr &positive_pressure_test,
    const IncompressibleTwoFluidInterfaceParameters &parameters) {
  validateParameters(parameters);
  validateExpression(negative_velocity, "negative velocity");
  validateExpression(negative_pressure, "negative pressure");
  validateExpression(negative_velocity_test, "negative velocity test");
  validateExpression(negative_pressure_test, "negative pressure test");
  validateExpression(positive_velocity, "positive velocity");
  validateExpression(positive_pressure, "positive pressure");
  validateExpression(positive_velocity_test, "positive velocity test");
  validateExpression(positive_pressure_test, "positive pressure test");

  using namespace FE::forms;

  const auto weights = incompressibleTwoFluidInterfaceWeights(parameters);
  const auto n = FormExpr::normal();
  const auto identity = FormExpr::identity(parameters.dimension);
  const auto two = FormExpr::constant(FE::Real{2.0});
  const auto mu_negative = FormExpr::constant(parameters.negative_viscosity);
  const auto mu_positive = FormExpr::constant(parameters.positive_viscosity);
  const auto negative_viscous_traction =
      (two * mu_negative * sym(grad(negative_velocity))) * n;
  const auto negative_pressure_traction = (-negative_pressure * identity) * n;
  const auto positive_viscous_traction =
      (two * mu_positive * sym(grad(positive_velocity))) * n;
  const auto positive_pressure_traction = (-positive_pressure * identity) * n;
  const auto negative_viscous_test_traction =
      (two * mu_negative * sym(grad(negative_velocity_test))) * n;
  const auto negative_pressure_test_traction =
      (-negative_pressure_test * identity) * n;
  const auto positive_viscous_test_traction =
      (two * mu_positive * sym(grad(positive_velocity_test))) * n;
  const auto positive_pressure_test_traction =
      (-positive_pressure_test * identity) * n;

  const auto negative_weight = FormExpr::constant(weights.negative_traction);
  const auto positive_weight = FormExpr::constant(weights.positive_traction);

  auto penalty_scale =
      FormExpr::constant(weights.harmonic_viscosity) / hNormal();
  if (parameters.include_transient_penalty) {
    penalty_scale = penalty_scale +
                    FormExpr::constant(weights.harmonic_density) * hNormal() /
                        FormExpr::effectiveTimeStep();
  }
  penalty_scale = FormExpr::constant(parameters.nitsche_gamma) * penalty_scale;

  IncompressibleTwoFluidInterfaceForms forms;
  forms.weights = weights;
  // Keep every generated-interface integral block-separable: each term has
  // exactly one test and at most one trial function. This is algebraically
  // identical to -<{sigma(u,p)n}_w, [v]> and lets the mixed compiler retain
  // all four velocity/pressure fields without an opaque multi-field term.
  forms.consistency = (-negative_weight *
                       inner(negative_viscous_traction, negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (-negative_weight * inner(negative_pressure_traction,
                                                negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (negative_weight *
                       inner(negative_viscous_traction, positive_velocity_test))
                          .dI(parameters.interface_marker) +
                      (negative_weight * inner(negative_pressure_traction,
                                               positive_velocity_test))
                          .dI(parameters.interface_marker) +
                      (-positive_weight *
                       inner(positive_viscous_traction, negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (-positive_weight * inner(positive_pressure_traction,
                                                negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (positive_weight *
                       inner(positive_viscous_traction, positive_velocity_test))
                          .dI(parameters.interface_marker) +
                      (positive_weight * inner(positive_pressure_traction,
                                               positive_velocity_test))
                          .dI(parameters.interface_marker);

  // The adjoint channel is the expanded -<{sigma(v,q)n}_w, [u]> term.
  forms.adjoint = (-negative_weight *
                   inner(negative_viscous_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (-negative_weight *
                   inner(negative_pressure_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (negative_weight *
                   inner(negative_viscous_test_traction, positive_velocity))
                      .dI(parameters.interface_marker) +
                  (negative_weight *
                   inner(negative_pressure_test_traction, positive_velocity))
                      .dI(parameters.interface_marker) +
                  (-positive_weight *
                   inner(positive_viscous_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (-positive_weight *
                   inner(positive_pressure_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (positive_weight *
                   inner(positive_viscous_test_traction, positive_velocity))
                      .dI(parameters.interface_marker) +
                  (positive_weight *
                   inner(positive_pressure_test_traction, positive_velocity))
                      .dI(parameters.interface_marker);

  forms.penalty =
      (penalty_scale * inner(negative_velocity, negative_velocity_test))
          .dI(parameters.interface_marker) +
      (-penalty_scale * inner(negative_velocity, positive_velocity_test))
          .dI(parameters.interface_marker) +
      (-penalty_scale * inner(positive_velocity, negative_velocity_test))
          .dI(parameters.interface_marker) +
      (penalty_scale * inner(positive_velocity, positive_velocity_test))
          .dI(parameters.interface_marker);
  forms.residual = forms.consistency + forms.adjoint + forms.penalty;

  if (parameters.prescribed_pressure_jump.has_value()) {
    const auto target =
        FormExpr::constant(*parameters.prescribed_pressure_jump);
    forms.prescribed_pressure_jump =
        (target * FormExpr::constant(weights.negative_complement) *
         inner(n, negative_velocity_test))
            .dI(parameters.interface_marker) +
        (target * FormExpr::constant(weights.positive_complement) *
         inner(n, positive_velocity_test))
            .dI(parameters.interface_marker);
    forms.residual = forms.residual + forms.prescribed_pressure_jump;
  }

  if (parameters.surface_tension > FE::Real{0.0}) {
    const auto projector = identity - outer(n, n);
    const auto surface_tension = FormExpr::constant(parameters.surface_tension);
    forms.surface_energy =
        (surface_tension * FormExpr::constant(weights.negative_complement) *
         inner(projector, grad(negative_velocity_test)))
            .dI(parameters.interface_marker) +
        (surface_tension * FormExpr::constant(weights.positive_complement) *
         inner(projector, grad(positive_velocity_test)))
            .dI(parameters.interface_marker);
    forms.residual = forms.residual + forms.surface_energy;
  }
  return forms;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
