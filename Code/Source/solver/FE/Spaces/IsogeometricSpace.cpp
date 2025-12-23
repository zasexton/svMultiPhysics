/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/IsogeometricSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

IsogeometricSpace::IsogeometricSpace(std::shared_ptr<basis::BasisFunction> basis,
                                     std::shared_ptr<const quadrature::QuadratureRule> quadrature,
                                     FieldType field_type,
                                     Continuity continuity) {
    auto elem = std::make_shared<elements::IsogeometricElement>(
        std::move(basis), std::move(quadrature), field_type, continuity);

    info_ = elem->info();
    element_dimension_ = elem->dimension();
    element_ = std::move(elem);
}

} // namespace spaces
} // namespace FE
} // namespace svmp

