/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONSTRAINT_DEPENDENCY_H
#define SVMP_FE_CONSTRAINTS_CONSTRAINT_DEPENDENCY_H

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

struct ConstraintDependencyMask {
    bool geometry{false};
    bool reference_rebase{false};
    bool topology{false};
    bool ownership{false};
    bool numbering{false};
    bool mesh_field_layout{false};
    bool mesh_field_values{false};
    bool labels{false};
    bool active_configuration{false};
    bool fe_space{false};
    bool fe_dof_layout{false};
    bool fe_constraint_layout{false};
    bool fe_block_layout{false};
    bool time{false};

    [[nodiscard]] bool any() const noexcept;

    [[nodiscard]] static ConstraintDependencyMask none() noexcept { return {}; }
    [[nodiscard]] static ConstraintDependencyMask meshBoundaryTopology() noexcept;
    [[nodiscard]] static ConstraintDependencyMask meshGeometry() noexcept;
    [[nodiscard]] static ConstraintDependencyMask feDofLayout() noexcept;
};

struct ConstraintRevisionSnapshot {
    bool valid{false};
    std::uint64_t geometry{0};
    std::uint64_t reference_rebase{0};
    std::uint64_t topology{0};
    std::uint64_t ownership{0};
    std::uint64_t numbering{0};
    std::uint64_t mesh_field_layout{0};
    std::uint64_t mesh_field_values{0};
    std::uint64_t labels{0};
    std::uint64_t active_configuration{0};
    std::uint64_t fe_space{0};
    std::uint64_t fe_dof_layout{0};
    std::uint64_t fe_constraint_layout{0};
    std::uint64_t fe_block_layout{0};
    std::uint64_t time_epoch{0};

    [[nodiscard]] bool operator==(
        const ConstraintRevisionSnapshot&) const noexcept = default;
};

enum class ConstraintTangentPolicy : std::uint8_t {
    None,
    Analytic,
    External,
    NotSupported
};

struct ConstraintDependencyDeclaration {
    ConstraintDependencyMask structural{};
    ConstraintDependencyMask value{};
    ConstraintTangentPolicy tangent_policy{ConstraintTangentPolicy::None};
    std::string tangent_hook_name{};

    /// When set, a mesh_field_values dependency reads only the mesh fields
    /// named in mesh_field_value_names (of any entity kind); otherwise it
    /// covers the values of every mesh field.
    bool mesh_field_values_scoped{false};
    std::vector<std::string> mesh_field_value_names{};

    [[nodiscard]] bool any() const noexcept { return structural.any() || value.any(); }

    [[nodiscard]] bool readsMeshFieldValues() const noexcept
    {
        return structural.mesh_field_values || value.mesh_field_values;
    }
};

struct ConstraintRefreshResult {
    bool dependency_changed{false};
    bool structural_rebuild{false};
    bool value_update{false};
    bool skipped_no_cached_setup{false};
    std::string reason{};
};

void merge_into(ConstraintDependencyMask& dst, const ConstraintDependencyMask& src) noexcept;

/// Merges masks and tangent policy. The merged mesh-field-value scope is the
/// union of the scoped declarations that read mesh field values, or unscoped
/// (every field) as soon as one of them is unscoped.
void merge_into(ConstraintDependencyDeclaration& dst,
                const ConstraintDependencyDeclaration& src);

[[nodiscard]] bool dependency_changed(const ConstraintDependencyMask& mask,
                                      const ConstraintRevisionSnapshot& cached,
                                      const ConstraintRevisionSnapshot& current) noexcept;

[[nodiscard]] bool structural_dependency_changed(
    const ConstraintDependencyDeclaration& declaration,
    const ConstraintRevisionSnapshot& cached,
    const ConstraintRevisionSnapshot& current) noexcept;

[[nodiscard]] bool value_dependency_changed(
    const ConstraintDependencyDeclaration& declaration,
    const ConstraintRevisionSnapshot& cached,
    const ConstraintRevisionSnapshot& current) noexcept;

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONSTRAINT_DEPENDENCY_H
