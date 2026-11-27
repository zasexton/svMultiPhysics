/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceCompatibility.h"

#include "Basis/BasisTraits.h"

namespace svmp {
namespace FE {
namespace spaces {

SpaceCompatibility::Result
SpaceCompatibility::check_conformity(const FunctionSpace& a,
                                     const FunctionSpace& b) {
    Result r;

    if (a.element_type() != b.element_type()) {
        r.ok = false;
        r.message = "Element types differ";
        return r;
    }
    if (a.topological_dimension() != b.topological_dimension()) {
        r.ok = false;
        r.message = "Topological dimensions differ";
        return r;
    }
    if (a.field_type() != b.field_type()) {
        r.ok = false;
        r.message = "Field types differ";
        return r;
    }
    if (a.continuity() != b.continuity()) {
        r.ok = false;
        r.message = "Continuity types differ";
        return r;
    }

    r.ok = true;
    r.message = "Spaces are conforming";
    return r;
}

SpaceCompatibility::Result
SpaceCompatibility::check_inf_sup(const FunctionSpace& vector_field_space,
                                  const FunctionSpace& multiplier_space) {
    const auto type = vector_field_space.element_type();
    if (basis::topology(type) != basis::topology(multiplier_space.element_type()) ||
        vector_field_space.topological_dimension() != multiplier_space.topological_dimension()) {
        return {false, "Vector field and multiplier have different reference cells"};
    }
    if (vector_field_space.field_type() != FieldType::Vector ||
        multiplier_space.field_type() != FieldType::Scalar ||
        multiplier_space.continuity() != Continuity::L2) {
        return {false, "Unrecognized vector-field/multiplier pair"};
    }
    if (vector_field_space.continuity() == Continuity::C0) {
        return {false, "A continuous vector field with a discontinuous multiplier requires a mesh-dependent inf-sup check"};
    }

    if (vector_field_space.continuity() == Continuity::H_div &&
        multiplier_space.element().basis().basis_type() == BasisType::Lagrange) {
        const auto family = vector_field_space.element().basis().basis_type();
        const int order = vector_field_space.polynomial_order();
        int divergence_order = -1;
        if (family == BasisType::RaviartThomas &&
            (basis::is_simplex(type) || basis::is_quadrilateral(type) || basis::is_hexahedron(type))) {
            divergence_order = order;
        } else if (family == BasisType::BDM &&
                   (basis::is_simplex(type) || (basis::is_quadrilateral(type) && order == 1))) {
            divergence_order = order - 1;
        }
        if (multiplier_space.polynomial_order() <= divergence_order) {
            return {true, "The reference divergence space contains the multiplier polynomials"};
        }
    }
    return {false, "Reference divergence compatibility has not been established for this pair"};
}

} // namespace spaces
} // namespace FE
} // namespace svmp

