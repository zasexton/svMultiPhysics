/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_MIXEDSPACE_H
#define SVMP_FE_SPACES_MIXEDSPACE_H

/**
 * @file MixedSpace.h
 * @brief Mixed function spaces combining multiple component spaces
 */

#include "Spaces/FunctionSpace.h"
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Mixed function space composed of several subspaces
 *
 * MixedSpace provides a light-weight container for multiple function spaces
 * that share the same reference element but may differ in continuity,
 * polynomial order, or field type (e.g., velocity-pressure pairs in mixed
 * formulations). It primarily encodes block structure and indexing; element-
 * level interpolation and evaluation are delegated to the component spaces.
 */
class MixedSpace : public FunctionSpace {
public:
    /// Description of a component field within a mixed space
    struct Component {
        std::string name;
        std::shared_ptr<FunctionSpace> space;
    };

    MixedSpace() = default;

    /// Add a component space with an optional name
    void add_component(const std::string& name,
                       std::shared_ptr<FunctionSpace> space);

    /// Number of component fields
    std::size_t num_components() const noexcept { return components_.size(); }

    /// Access component description
    const Component& component(std::size_t i) const { return components_.at(i); }

    /**
     * @brief Evaluate a single component field from an aggregated coefficient vector
     *
     * Extracts the coefficient sub-vector for the selected component and
     * delegates evaluation to the corresponding component space.
     */
    Value evaluate_component(std::size_t component_index,
                             const Value& xi,
                             const std::vector<Real>& coefficients) const;

    /**
     * @brief Evaluate all component fields from an aggregated coefficient vector
     *
     * Returns one Value per component in the order they were added.
     */
    std::vector<Value> evaluate_components(const Value& xi,
                                           const std::vector<Real>& coefficients) const;

    /// Total DOFs per element (sum over components)
    std::size_t dofs_per_element() const noexcept override;

    /// Block offset (starting DOF index) of component within concatenated vector
    std::size_t component_offset(std::size_t i) const;

    // FunctionSpace interface
    SpaceType space_type() const noexcept override { return SpaceType::Mixed; }
    FieldType field_type() const noexcept override { return FieldType::Mixed; }
    Continuity continuity() const noexcept override { return Continuity::Custom; }

    int value_dimension() const noexcept override;
    int topological_dimension() const noexcept override;
    int polynomial_order() const noexcept override;
    ElementType element_type() const noexcept override;

    const elements::Element& element() const noexcept override;
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override;

    /// Mixed spaces do not provide a unified evaluation; throw if called
    Value evaluate(const Value&,
                   const std::vector<Real>&) const override {
        FE_THROW(NotImplementedException,
                 "MixedSpace::evaluate is not defined for aggregated fields");
    }

    /// Mixed spaces do not provide a unified interpolation; throw if called
    void interpolate(const ValueFunction&,
                     std::vector<Real>&) const override {
        FE_THROW(NotImplementedException,
                 "MixedSpace::interpolate is not defined for aggregated fields (interpolate each component and pack)");
    }

private:
    std::vector<Component> components_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_MIXEDSPACE_H
