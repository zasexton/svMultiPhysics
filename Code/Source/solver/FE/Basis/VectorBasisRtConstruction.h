/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_VECTORBASISRTCONSTRUCTION_H
#define SVMP_FE_BASIS_VECTORBASISRTCONSTRUCTION_H

#include "VectorBasis.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {
namespace detail {
namespace vector_construction {

using Vec3 = math::Vector<Real, 3>;

std::vector<Real> invert_dense_matrix(std::vector<Real> A, std::size_t n);
std::vector<Real> rank_revealing_pseudo_inverse_dense_matrix(
    const std::vector<Real>& A,
    std::size_t n);

void eval_rt_seed_values(ElementType type,
                         int order,
                         const Vec3& xi,
                         std::vector<Vec3>& values);
void eval_rt_seed_divergence(ElementType type,
                             int order,
                             const Vec3& xi,
                             std::vector<Real>& divergence);
std::vector<Real> build_rt_direct_transform(
    ElementType type,
    int order,
    std::size_t n,
    const std::vector<std::array<int, 4>>& extra_monomials);

void eval_nd_seed_values(ElementType type,
                         int order,
                         const Vec3& xi,
                         std::vector<Vec3>& values);
void eval_nd_seed_curl(ElementType type,
                       int order,
                       const Vec3& xi,
                       std::vector<Vec3>& curl);
std::vector<Real> build_nd_direct_transform(
    ElementType type,
    int order,
    std::size_t n,
    const std::vector<std::array<int, 4>>& extra_monomials);

} // namespace vector_construction
} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_VECTORBASISRTCONSTRUCTION_H
