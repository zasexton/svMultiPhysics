/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_SNAPSHOTFIELDEVALUATOR_H
#define SVMP_FE_INTERFACES_SNAPSHOTFIELDEVALUATOR_H

/**
 * @file SnapshotFieldEvaluator.h
 * @brief Point evaluators of FE fields at snapshot rule points.
 *
 * The evaluators return the value (and the physical or reference gradient)
 * of one FE field at a reference point of a parent cell, from a solution
 * vector in FE order: the callbacks that the free-surface geometry snapshot
 * functionals and the accepted-stage diagnostics take
 * (FreeSurfaceGeometrySnapshot.h).  What they read of the field is a
 * SnapshotFieldAccess, which systems::snapshotFieldAccess builds from an FE
 * system.
 */

#include "Core/Types.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"

#include <functional>
#include <memory>
#include <span>
#include <string_view>

namespace svmp {
namespace FE {

namespace assembly {
class IMeshAccess;
} // namespace assembly

namespace spaces {
class FunctionSpace;
} // namespace spaces

namespace interfaces {

/// What a snapshot field evaluator reads of one FE field.
struct SnapshotFieldAccess {
    /// The function space of the field.
    std::shared_ptr<const spaces::FunctionSpace> space{};
    /// The number of components of the field.
    int components{1};
    /// The offset of the field's DOFs in the solution vector.
    GlobalIndex dof_offset{-1};
    /// The mesh of the field; called when an evaluator needs it.
    std::function<const assembly::IMeshAccess&()> mesh{};
    /// The field's DOFs of a cell, looked up at every use.
    std::function<std::span<const GlobalIndex>(GlobalIndex cell)> cell_dofs{};
};

/**
 * @brief Vector values and physical gradients of a vector field.
 *
 * The field must have at least the mesh dimension of components.  A product
 * of a plain H1 scalar space is evaluated with one basis evaluation shared
 * by the components (bitwise as ProductSpace::evaluate and
 * evaluate_jacobian) unless @p allow_scalar_product_fast_path is false.  The
 * evaluator keeps the coefficients and geometry mapping of the last cell, so
 * it is single-threaded.  @p diagnostic_context starts its error messages.
 */
[[nodiscard]] FreeSurfaceDiscreteFunctionalVectorEvaluator makeSnapshotVectorFieldEvaluator(
    const SnapshotFieldAccess& field,
    std::span<const Real> solution,
    std::string_view diagnostic_context,
    bool allow_scalar_product_fast_path = true);

/// Values and reference gradients of a scalar field (one component).
[[nodiscard]] FreeSurfaceGeometryScalarEvaluator makeSnapshotScalarFieldEvaluator(
    const SnapshotFieldAccess& field,
    std::span<const Real> solution,
    std::string_view diagnostic_context);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_SNAPSHOTFIELDEVALUATOR_H
