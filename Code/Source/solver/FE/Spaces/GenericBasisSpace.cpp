/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/GenericBasisSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

GenericBasisSpace::GenericBasisSpace(std::shared_ptr<basis::BasisFunction> basis,
                                     std::shared_ptr<const quadrature::QuadratureRule> quadrature,
                                     FieldType field_type,
                                     Continuity continuity)
    : GenericBasisSpace(std::make_shared<elements::GeneralBasisElement>(
          std::move(basis), std::move(quadrature), field_type, continuity)) {
}

GenericBasisSpace::GenericBasisSpace(std::shared_ptr<elements::Element> element)
    : element_(std::move(element)) {
    FE_CHECK_NOT_NULL(element_.get(), "GenericBasisSpace element");
    info_ = element_->info();
    element_dimension_ = element_->dimension();
}

} // namespace spaces
} // namespace FE
} // namespace svmp
