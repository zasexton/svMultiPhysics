/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_SPACEFACTORY_H
#define SVMP_FE_SPACES_SPACEFACTORY_H

/**
 * @file SpaceFactory.h
 * @brief Factory for creating basic function spaces
 *
 * SpaceFactory provides convenience helpers for constructing common function
 * spaces (H¹, L², H(curl), H(div), vector-valued H¹) without depending on
 * any Mesh types. Higher-level modules pass element types and polynomial
 * orders; mesh topology is handled elsewhere.
 */

#include "Spaces/FunctionSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/ProductSpace.h"
#include "Spaces/IsogeometricSpace.h"
#include "Spaces/C1Space.h"
#include <memory>

namespace svmp {
namespace FE {
namespace assembly {
class IMeshAccess;
} // namespace assembly
namespace spaces {

class SpaceFactory {
public:
    /// Generic factory for core scalar/vector spaces
    static std::shared_ptr<FunctionSpace> create(SpaceType type,
                                                 ElementType element_type,
                                                 int order);

    static std::shared_ptr<H1Space> create_h1(ElementType element_type,
                                              int order) {
        return std::make_shared<H1Space>(element_type, order);
    }

    static std::shared_ptr<C1Space> create_c1(ElementType element_type,
                                              int order = 3) {
        return std::make_shared<C1Space>(element_type, order);
    }

    static std::shared_ptr<L2Space> create_l2(ElementType element_type,
                                              int order) {
        return std::make_shared<L2Space>(element_type, order);
    }

    static std::shared_ptr<HCurlSpace> create_hcurl(ElementType element_type,
                                                    int order) {
        return std::make_shared<HCurlSpace>(element_type, order);
    }

    static std::shared_ptr<HDivSpace> create_hdiv(ElementType element_type,
                                                  int order) {
        return std::make_shared<HDivSpace>(element_type, order);
    }

    /// Convenience: vector-valued H¹ space as product of scalar H¹
    static std::shared_ptr<ProductSpace> create_vector_h1(ElementType element_type,
                                                          int order,
                                                          int components);

    /// Factory for isogeometric spaces from external basis/quadrature
    static std::shared_ptr<IsogeometricSpace> create_isogeometric(
        std::shared_ptr<basis::BasisFunction> basis,
        std::shared_ptr<const quadrature::QuadratureRule> quadrature,
        FieldType field_type = FieldType::Scalar,
        Continuity continuity = Continuity::C0) {
        return std::make_shared<IsogeometricSpace>(
            std::move(basis), std::move(quadrature), field_type, continuity);
    }
};

/**
 * @brief Create a scalar- or vector-valued function space from a single call.
 *
 * This is a lightweight, user-facing wrapper around `SpaceFactory::create(...)`
 * that supports the common "vector H1" style by automatically constructing a
 * `ProductSpace` when the requested base space is scalar-valued.
 *
 * If @p components == 1, this returns the underlying space produced by
 * `SpaceFactory::create(...)`.
 *
 * If @p components > 1 and the selected space type is scalar-valued (e.g., H1/L2/C1),
 * this returns a `ProductSpace` with @p components copies of the scalar base space.
 *
 * For intrinsically vector-valued spaces (e.g., H(div)/H(curl)), @p components is
 * treated as a consistency check when > 1 (must match `value_dimension()`).
 *
 * @note We intentionally avoid naming this helper `FunctionSpace(...)` because
 *       it would collide with the existing `spaces::FunctionSpace` type name and
 *       force users to write `struct/class FunctionSpace` everywhere in C++.
 */
inline std::shared_ptr<FunctionSpace> Space(SpaceType type,
                                            ElementType element_type,
                                            int order,
                                            int components = 1)
{
    FE_THROW_IF(components < 1, InvalidArgumentException,
                "Space(SpaceType,...): components must be >= 1");

    auto base = SpaceFactory::create(type, element_type, order);
    FE_CHECK_NOT_NULL(base.get(), "Space(SpaceType,...): base space");

    if (base->field_type() == FieldType::Scalar) {
        if (components == 1) {
            return base;
        }
        return std::make_shared<ProductSpace>(std::move(base), components);
    }

    // Vector/tensor/mixed spaces are not replicated by ProductSpace here.
    if (components > 1 && base->value_dimension() != components) {
        FE_THROW(InvalidArgumentException,
                 "Space(SpaceType,...): components does not match intrinsic value dimension");
    }
    return base;
}

/**
 * @brief Convenience wrapper for vector-valued spaces.
 *
 * This is identical to `Space(type, element_type, order, components)`,
 * but reads better at call sites when constructing vector-valued unknowns.
 */
inline std::shared_ptr<FunctionSpace> VectorSpace(SpaceType type,
                                                  ElementType element_type,
                                                  int order,
                                                  int components)
{
    return Space(type, element_type, order, components);
}

/**
 * @brief Infer a uniform element type from a mesh access interface.
 *
 * This is a convenience for common workflows where the mesh is known to be
 * uniform (single reference element type). If the mesh contains multiple cell
 * types, this throws and callers must construct spaces explicitly per element
 * type (or extend the API to support heterogeneous spaces).
 *
 * If @p domain_id >= 0, only cells whose `mesh.getCellDomainId(cell_id)` equals
 * @p domain_id are considered.
 */
ElementType inferUniformElementType(const assembly::IMeshAccess& mesh, int domain_id = -1);

/// Overload: infer element type from a uniform mesh.
std::shared_ptr<FunctionSpace> Space(SpaceType type,
                                     const assembly::IMeshAccess& mesh,
                                     int order,
                                     int components = 1,
                                     int domain_id = -1);

/// Overload: infer element type from a uniform mesh.
std::shared_ptr<FunctionSpace> Space(SpaceType type,
                                     const std::shared_ptr<const assembly::IMeshAccess>& mesh,
                                     int order,
                                     int components = 1,
                                     int domain_id = -1);

/// Overload: infer element type from a uniform mesh.
std::shared_ptr<FunctionSpace> VectorSpace(SpaceType type,
                                           const assembly::IMeshAccess& mesh,
                                           int order,
                                           int components,
                                           int domain_id = -1);

/// Overload: infer element type from a uniform mesh.
std::shared_ptr<FunctionSpace> VectorSpace(SpaceType type,
                                           const std::shared_ptr<const assembly::IMeshAccess>& mesh,
                                           int order,
                                           int components,
                                           int domain_id = -1);

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_SPACEFACTORY_H
