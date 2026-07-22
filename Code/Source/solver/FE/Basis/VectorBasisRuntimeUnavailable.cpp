/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"

#include "Basis/BasisExceptions.h"

#include <string>

namespace svmp {
namespace FE {
namespace basis {
namespace {

[[noreturn]] void throwVectorBasisRuntimeUnavailable(const char* family)
{
    throw BasisEvaluationException(
        std::string(family) +
        " vector-basis runtime evaluation is unavailable in this build");
}

} // namespace

void BDMBasis::evaluate_vector_values(
    const math::Vector<Real, 3>&,
    std::vector<math::Vector<Real, 3>>&) const
{
    throwVectorBasisRuntimeUnavailable("BDM");
}

void BDMBasis::evaluate_vector_jacobians(
    const math::Vector<Real, 3>&,
    std::vector<VectorJacobian>&) const
{
    throwVectorBasisRuntimeUnavailable("BDM");
}

void BDMBasis::evaluate_divergence(const math::Vector<Real, 3>&,
                                   std::vector<Real>&) const
{
    throwVectorBasisRuntimeUnavailable("BDM");
}

void BDMBasis::evaluate_vector_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>&,
    std::size_t,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT) const
{
    throwVectorBasisRuntimeUnavailable("BDM");
}

void NedelecBasis::evaluate_vector_values(
    const math::Vector<Real, 3>&,
    std::vector<math::Vector<Real, 3>>&) const
{
    throwVectorBasisRuntimeUnavailable("Nedelec");
}

void NedelecBasis::evaluate_vector_jacobians(
    const math::Vector<Real, 3>&,
    std::vector<VectorJacobian>&) const
{
    throwVectorBasisRuntimeUnavailable("Nedelec");
}

void NedelecBasis::evaluate_curl(
    const math::Vector<Real, 3>&,
    std::vector<math::Vector<Real, 3>>&) const
{
    throwVectorBasisRuntimeUnavailable("Nedelec");
}

void NedelecBasis::evaluate_vector_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>&,
    std::size_t,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT) const
{
    throwVectorBasisRuntimeUnavailable("Nedelec");
}

void RaviartThomasBasis::evaluate_vector_values(
    const math::Vector<Real, 3>&,
    std::vector<math::Vector<Real, 3>>&) const
{
    throwVectorBasisRuntimeUnavailable("Raviart-Thomas");
}

void RaviartThomasBasis::evaluate_vector_jacobians(
    const math::Vector<Real, 3>&,
    std::vector<VectorJacobian>&) const
{
    throwVectorBasisRuntimeUnavailable("Raviart-Thomas");
}

void RaviartThomasBasis::evaluate_divergence(const math::Vector<Real, 3>&,
                                             std::vector<Real>&) const
{
    throwVectorBasisRuntimeUnavailable("Raviart-Thomas");
}

void RaviartThomasBasis::evaluate_vector_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>&,
    std::size_t,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT,
    Real* SVMP_RESTRICT) const
{
    throwVectorBasisRuntimeUnavailable("Raviart-Thomas");
}

} // namespace basis
} // namespace FE
} // namespace svmp
