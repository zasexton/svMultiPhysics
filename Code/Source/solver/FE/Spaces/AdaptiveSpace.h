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

    /// Set the active polynomial order for one element/cell
    void set_element_order(GlobalIndex cell_id,
                           int order);

    /// Resize per-element order storage and optionally initialize it
    void resize_element_orders(std::size_t n_cells,
                               int default_order = -1);

    /// Clear all cell-specific polynomial orders
    void clear_element_orders() { element_orders_.clear(); }

    /// Query the configured order for one element/cell
    int element_order(GlobalIndex cell_id) const;

    /// Number of cells with explicit order storage
    std::size_t num_element_orders() const noexcept { return element_orders_.size(); }

    /// Whether one element/cell has an explicit polynomial order
    bool has_element_order(GlobalIndex cell_id) const noexcept;

    /// Access the space used for one element/cell
    const FunctionSpace& element_space(GlobalIndex cell_id) const noexcept;

    /// Access the owning space pointer used for one element/cell
    std::shared_ptr<FunctionSpace> element_space_ptr(GlobalIndex cell_id) const noexcept;

    // FunctionSpace interface (delegated to active level)
    SpaceType space_type() const noexcept override { return SpaceType::Adaptive; }
    FieldType field_type() const noexcept override;
    Continuity continuity() const noexcept override;

    int value_dimension() const noexcept override;
    int topological_dimension() const noexcept override;
    int polynomial_order() const noexcept override;
    int polynomial_order(GlobalIndex cell_id) const noexcept override;
    bool is_variable_order() const noexcept override { return !element_orders_.empty(); }
    ElementType element_type() const noexcept override;

    const elements::Element& element() const noexcept override;
    const elements::Element& getElement(ElementType cell_type,
                                        GlobalIndex cell_id) const noexcept override;
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override;

    std::size_t dofs_per_element() const noexcept override;
    std::size_t dofs_per_element(GlobalIndex cell_id) const noexcept override;

    Value evaluate(const Value& xi,
                   const std::vector<Real>& coefficients) const override;

    void interpolate(const ValueFunction& function,
                     std::vector<Real>& coefficients) const override;

private:
    const Level* find_level(int order) const noexcept;
    const Level& level_for_cell(GlobalIndex cell_id) const noexcept;

    std::vector<Level> levels_;
    std::size_t active_index_{0};
    std::vector<int> element_orders_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_ADAPTIVESPACE_H
