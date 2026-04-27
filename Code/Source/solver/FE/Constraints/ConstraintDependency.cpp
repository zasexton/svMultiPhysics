/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/ConstraintDependency.h"

namespace svmp {
namespace FE {
namespace constraints {

bool ConstraintDependencyMask::any() const noexcept
{
    return geometry || reference_rebase || topology || ownership || numbering ||
           mesh_field_layout || labels || active_configuration || fe_space ||
           fe_dof_layout || fe_constraint_layout || fe_block_layout || time;
}

ConstraintDependencyMask ConstraintDependencyMask::meshBoundaryTopology() noexcept
{
    ConstraintDependencyMask out;
    out.topology = true;
    out.ownership = true;
    out.numbering = true;
    out.labels = true;
    return out;
}

ConstraintDependencyMask ConstraintDependencyMask::meshGeometry() noexcept
{
    ConstraintDependencyMask out;
    out.geometry = true;
    out.reference_rebase = true;
    out.active_configuration = true;
    return out;
}

ConstraintDependencyMask ConstraintDependencyMask::feDofLayout() noexcept
{
    ConstraintDependencyMask out;
    out.fe_space = true;
    out.fe_dof_layout = true;
    return out;
}

void merge_into(ConstraintDependencyMask& dst, const ConstraintDependencyMask& src) noexcept
{
    dst.geometry = dst.geometry || src.geometry;
    dst.reference_rebase = dst.reference_rebase || src.reference_rebase;
    dst.topology = dst.topology || src.topology;
    dst.ownership = dst.ownership || src.ownership;
    dst.numbering = dst.numbering || src.numbering;
    dst.mesh_field_layout = dst.mesh_field_layout || src.mesh_field_layout;
    dst.labels = dst.labels || src.labels;
    dst.active_configuration = dst.active_configuration || src.active_configuration;
    dst.fe_space = dst.fe_space || src.fe_space;
    dst.fe_dof_layout = dst.fe_dof_layout || src.fe_dof_layout;
    dst.fe_constraint_layout = dst.fe_constraint_layout || src.fe_constraint_layout;
    dst.fe_block_layout = dst.fe_block_layout || src.fe_block_layout;
    dst.time = dst.time || src.time;
}

void merge_into(ConstraintDependencyDeclaration& dst,
                const ConstraintDependencyDeclaration& src) noexcept
{
    merge_into(dst.structural, src.structural);
    merge_into(dst.value, src.value);
    if (dst.tangent_policy == ConstraintTangentPolicy::None &&
        src.tangent_policy != ConstraintTangentPolicy::None) {
        dst.tangent_policy = src.tangent_policy;
        dst.tangent_hook_name = src.tangent_hook_name;
    }
}

bool dependency_changed(const ConstraintDependencyMask& mask,
                        const ConstraintRevisionSnapshot& cached,
                        const ConstraintRevisionSnapshot& current) noexcept
{
    if (!mask.any()) {
        return false;
    }
    if (!cached.valid || !current.valid) {
        return true;
    }
    return (mask.geometry && cached.geometry != current.geometry) ||
           (mask.reference_rebase && cached.reference_rebase != current.reference_rebase) ||
           (mask.topology && cached.topology != current.topology) ||
           (mask.ownership && cached.ownership != current.ownership) ||
           (mask.numbering && cached.numbering != current.numbering) ||
           (mask.mesh_field_layout && cached.mesh_field_layout != current.mesh_field_layout) ||
           (mask.labels && cached.labels != current.labels) ||
           (mask.active_configuration && cached.active_configuration != current.active_configuration) ||
           (mask.fe_space && cached.fe_space != current.fe_space) ||
           (mask.fe_dof_layout && cached.fe_dof_layout != current.fe_dof_layout) ||
           (mask.fe_constraint_layout && cached.fe_constraint_layout != current.fe_constraint_layout) ||
           (mask.fe_block_layout && cached.fe_block_layout != current.fe_block_layout) ||
           (mask.time && cached.time_epoch != current.time_epoch);
}

bool structural_dependency_changed(const ConstraintDependencyDeclaration& declaration,
                                   const ConstraintRevisionSnapshot& cached,
                                   const ConstraintRevisionSnapshot& current) noexcept
{
    return dependency_changed(declaration.structural, cached, current);
}

bool value_dependency_changed(const ConstraintDependencyDeclaration& declaration,
                              const ConstraintRevisionSnapshot& cached,
                              const ConstraintRevisionSnapshot& current) noexcept
{
    return dependency_changed(declaration.value, cached, current);
}

} // namespace constraints
} // namespace FE
} // namespace svmp
