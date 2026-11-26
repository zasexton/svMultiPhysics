/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceCompatibility.h"

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
    Result r;

    const int dim_v = vector_field_space.topological_dimension();
    const int dim_p = multiplier_space.topological_dimension();

    if (dim_v != dim_p) {
        r.ok = false;
        r.message = "Vector field and multiplier have different spatial dimensions";
        return r;
    }

    // Vector-valued H1 field paired with a scalar L2 multiplier
    if (vector_field_space.field_type() == FieldType::Vector &&
        vector_field_space.continuity() == Continuity::C0 &&
        multiplier_space.field_type() == FieldType::Scalar &&
        multiplier_space.continuity() == Continuity::L2) {

        const int pv = vector_field_space.polynomial_order();
        const int pp = multiplier_space.polynomial_order();

        if (pv >= pp && pv > 0) {
            r.ok = true;
            r.message = "Heuristically inf-sup stable H1-L2 pair";
        } else {
            r.ok = false;
            r.message = "Vector field order too low relative to multiplier for stable H1-L2 pair";
        }
        return r;
    }

    // Mixed H(div)-L2 pair
    if (vector_field_space.field_type() == FieldType::Vector &&
        vector_field_space.continuity() == Continuity::H_div &&
        multiplier_space.field_type() == FieldType::Scalar &&
        multiplier_space.continuity() == Continuity::L2) {

        r.ok = true;
        r.message = "Heuristically compatible H(div)-L2 mixed pair";
        return r;
    }

    r.ok = false;
    r.message = "Unknown or unsupported mixed pair; unable to assess inf-sup stability";
    return r;
}

} // namespace spaces
} // namespace FE
} // namespace svmp

