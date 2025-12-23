/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_ENRICHEDSPACE_H
#define SVMP_FE_SPACES_ENRICHEDSPACE_H

/**
 * @file EnrichedSpace.h
 * @brief Enriched function spaces (XFEM, partition-of-unity)
 */

#include "Spaces/FunctionSpace.h"
#include <memory>

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Function space obtained by enriching a base space
 *
 * Represents spaces of the form V_enr = V_std ⊕ V_enrichment. The base and
 * enrichment spaces must live on the same reference element and share the
 * same field type. DOFs are concatenated as [std_coeffs, enrichment_coeffs].
 */
class EnrichedSpace : public FunctionSpace {
public:
    EnrichedSpace(std::shared_ptr<FunctionSpace> base_space,
                  std::shared_ptr<FunctionSpace> enrichment_space);

    SpaceType space_type() const noexcept override { return SpaceType::Enriched; }
    FieldType field_type() const noexcept override { return base_->field_type(); }
    Continuity continuity() const noexcept override { return base_->continuity(); }

    int value_dimension() const noexcept override { return base_->value_dimension(); }
    int topological_dimension() const noexcept override { return base_->topological_dimension(); }
    int polynomial_order() const noexcept override {
        return std::max(base_->polynomial_order(), enrichment_->polynomial_order());
    }
    ElementType element_type() const noexcept override { return base_->element_type(); }

    const elements::Element& element() const noexcept override { return base_->element(); }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override {
        return base_->element_ptr();
    }

    std::size_t dofs_per_element() const noexcept override {
        return base_->dofs_per_element() + enrichment_->dofs_per_element();
    }

    /// Evaluate u = u_std + u_enrichment
    Value evaluate(const Value& xi,
                   const std::vector<Real>& coefficients) const override;

    /// Interpolate by projecting into base space and zero-enriching by default
    void interpolate(const ValueFunction& function,
                     std::vector<Real>& coefficients) const override;

private:
    std::shared_ptr<FunctionSpace> base_;
    std::shared_ptr<FunctionSpace> enrichment_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_ENRICHEDSPACE_H

