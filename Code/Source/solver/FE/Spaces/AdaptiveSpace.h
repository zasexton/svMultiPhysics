/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_ADAPTIVESPACE_H
#define SVMP_FE_SPACES_ADAPTIVESPACE_H

/**
 * @file AdaptiveSpace.h
 * @brief Spaces with variable polynomial order (p/hp-adaptivity)
 */

#include "Spaces/FunctionSpace.h"
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Adaptive function space with multiple polynomial levels
 *
 * AdaptiveSpace owns a set of FunctionSpace instances corresponding to
 * different polynomial orders. An "active" level can be selected for
 * element-local operations; global hp-adaptivity and DOF management are
 * handled by Adaptivity and Dofs modules.
 */
class AdaptiveSpace : public FunctionSpace {
public:
    struct Level {
        int order{0};
        std::shared_ptr<FunctionSpace> space;
    };

    AdaptiveSpace() = default;

    /// Add a level with given polynomial order
    void add_level(int order,
                   std::shared_ptr<FunctionSpace> space);

    /// Number of available levels
    std::size_t num_levels() const noexcept { return levels_.size(); }

    /// Set active level by index
    void set_active_level(std::size_t index);

    /// Set active level by polynomial order (throws if not found)
    void set_active_level_by_order(int order);

    /// Access current active level
    const Level& active_level() const;

    // FunctionSpace interface (delegated to active level)
    SpaceType space_type() const noexcept override { return SpaceType::Adaptive; }
    FieldType field_type() const noexcept override;
    Continuity continuity() const noexcept override;

    int value_dimension() const noexcept override;
    int topological_dimension() const noexcept override;
    int polynomial_order() const noexcept override;
    ElementType element_type() const noexcept override;

    const elements::Element& element() const noexcept override;
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override;

    std::size_t dofs_per_element() const noexcept override;

    Value evaluate(const Value& xi,
                   const std::vector<Real>& coefficients) const override;

    void interpolate(const ValueFunction& function,
                     std::vector<Real>& coefficients) const override;

private:
    std::vector<Level> levels_;
    std::size_t active_index_{0};
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_ADAPTIVESPACE_H

