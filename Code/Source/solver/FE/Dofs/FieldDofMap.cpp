/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "FieldDofMap.h"
#include "SubspaceView.h"

// Include function space if available
#if defined(__has_include)
#  if __has_include("Spaces/FunctionSpace.h")
#    include "Spaces/FunctionSpace.h"
#    define FIELDDOFMAP_HAS_SPACES 1
#  else
#    define FIELDDOFMAP_HAS_SPACES 0
#  endif
#else
#  define FIELDDOFMAP_HAS_SPACES 0
#endif

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

FieldDofMap::FieldDofMap() = default;
FieldDofMap::~FieldDofMap() = default;

FieldDofMap::FieldDofMap(FieldDofMap&&) noexcept = default;
FieldDofMap& FieldDofMap::operator=(FieldDofMap&&) noexcept = default;

// =============================================================================
// Field Registration
// =============================================================================

int FieldDofMap::addScalarField(const std::string& name, GlobalIndex n_dofs) {
    checkNotFinalized();

    if (name_to_index_.count(name) > 0) {
        throw FEException("FieldDofMap::addScalarField: field '" + name + "' already exists");
    }

    FieldDescriptor desc;
    desc.name = name;
    desc.n_components = 1;
    desc.n_dofs = n_dofs;
    desc.block_index = static_cast<int>(fields_.size());

    auto idx = static_cast<int>(fields_.size());
    fields_.push_back(std::move(desc));
    name_to_index_[name] = static_cast<std::size_t>(idx);

    return idx;
}

int FieldDofMap::addVectorField(const std::string& name, LocalIndex n_components,
                                 GlobalIndex n_dofs_per_component) {
    checkNotFinalized();

    if (name_to_index_.count(name) > 0) {
        throw FEException("FieldDofMap::addVectorField: field '" + name + "' already exists");
    }

    FieldDescriptor desc;
    desc.name = name;
    desc.n_components = n_components;
    desc.n_dofs = n_dofs_per_component * n_components;
    desc.block_index = static_cast<int>(fields_.size());

    auto idx = static_cast<int>(fields_.size());
    fields_.push_back(std::move(desc));
    name_to_index_[name] = static_cast<std::size_t>(idx);

    return idx;
}

#if FIELDDOFMAP_HAS_SPACES
int FieldDofMap::addField(const std::string& name, const spaces::FunctionSpace& space,
                          GlobalIndex n_mesh_entities) {
    checkNotFinalized();

    if (name_to_index_.count(name) > 0) {
        throw FEException("FieldDofMap::addField: field '" + name + "' already exists");
    }

    FieldDescriptor desc;
    desc.name = name;
    desc.n_components = static_cast<LocalIndex>(space.value_dimension());
    const GlobalIndex dofs_per_entity = static_cast<GlobalIndex>(space.dofs_per_element());
    desc.n_dofs = n_mesh_entities * dofs_per_entity;
    desc.block_index = static_cast<int>(fields_.size());
    desc.polynomial_order = space.polynomial_order();

    auto idx = static_cast<int>(fields_.size());
    fields_.push_back(std::move(desc));
    name_to_index_[name] = static_cast<std::size_t>(idx);

    return idx;
}
#else
int FieldDofMap::addField(const std::string& /*name*/,
                          const spaces::FunctionSpace& /*space*/,
                          GlobalIndex /*n_mesh_entities*/) {
    throw FEException("FieldDofMap::addField: FunctionSpace not available");
}
#endif

void FieldDofMap::setLayout(FieldLayout layout) {
    checkNotFinalized();
    layout_ = layout;
}

void FieldDofMap::finalize() {
    checkNotFinalized();
    computeOffsets();
    finalized_ = true;
}

// =============================================================================
// Field Queries
// =============================================================================

const FieldDescriptor& FieldDofMap::getField(std::size_t field_idx) const {
    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::getField: invalid field index");
    }
    return fields_[field_idx];
}

const FieldDescriptor& FieldDofMap::getField(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        throw FEException("FieldDofMap::getField: field '" + name + "' not found");
    }
    return fields_[it->second];
}

int FieldDofMap::getFieldIndex(const std::string& name) const noexcept {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        return -1;
    }
    return static_cast<int>(it->second);
}

bool FieldDofMap::hasField(const std::string& name) const noexcept {
    return name_to_index_.count(name) > 0;
}

std::vector<std::string> FieldDofMap::getFieldNames() const {
    std::vector<std::string> names;
    names.reserve(fields_.size());
    for (const auto& field : fields_) {
        names.push_back(field.name);
    }
    return names;
}

// =============================================================================
// DOF Range Queries
// =============================================================================

std::pair<GlobalIndex, GlobalIndex> FieldDofMap::getFieldDofRange(std::size_t field_idx) const {
    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::getFieldDofRange: invalid field index");
    }

    if (!finalized_) {
        throw FEException("FieldDofMap::getFieldDofRange: not finalized");
    }

    GlobalIndex start = field_offsets_[field_idx];
    GlobalIndex end = (field_idx + 1 < field_offsets_.size())
        ? field_offsets_[field_idx + 1]
        : total_dofs_;

    return {start, end};
}

std::pair<GlobalIndex, GlobalIndex> FieldDofMap::getFieldDofRange(const std::string& name) const {
    auto idx = getFieldIndex(name);
    if (idx < 0) {
        throw FEException("FieldDofMap::getFieldDofRange: field '" + name + "' not found");
    }
    return getFieldDofRange(static_cast<std::size_t>(idx));
}

GlobalIndex FieldDofMap::getFieldOffset(std::size_t field_idx) const {
    if (!finalized_) {
        throw FEException("FieldDofMap::getFieldOffset: not finalized");
    }
    if (field_idx >= field_offsets_.size()) {
        throw FEException("FieldDofMap::getFieldOffset: invalid field index");
    }
    return field_offsets_[field_idx];
}

// =============================================================================
// Component Queries
// =============================================================================

LocalIndex FieldDofMap::numComponents(std::size_t field_idx) const {
    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::numComponents: invalid field index");
    }
    return fields_[field_idx].n_components;
}

LocalIndex FieldDofMap::numComponents(const std::string& name) const {
    auto idx = getFieldIndex(name);
    if (idx < 0) {
        throw FEException("FieldDofMap::numComponents: field '" + name + "' not found");
    }
    return numComponents(static_cast<std::size_t>(idx));
}

IndexSet FieldDofMap::getComponentDofs(std::size_t field_idx, LocalIndex component) const {
    checkFinalized();

    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::getComponentDofs: invalid field index");
    }

    const auto& field = fields_[field_idx];
    if (component >= field.n_components) {
        throw FEException("FieldDofMap::getComponentDofs: invalid component index");
    }

    std::vector<GlobalIndex> dofs;
    GlobalIndex n_per_component = field.n_dofs / field.n_components;

    switch (layout_) {
        case FieldLayout::Interleaved: {
            // DOFs are interleaved across all fields at each node
            // For interleaved: stride by total_components_, start at field_offset + component
            GlobalIndex field_start = field_offsets_[field_idx];
            for (GlobalIndex i = 0; i < n_per_component; ++i) {
                GlobalIndex dof = field_start + i * field.n_components + component;
                dofs.push_back(dof);
            }
            break;
        }

        case FieldLayout::Block: {
            // Each component is a contiguous block
            GlobalIndex comp_offset = field_offsets_[field_idx] + component * n_per_component;
            dofs.reserve(static_cast<std::size_t>(n_per_component));
            for (GlobalIndex i = 0; i < n_per_component; ++i) {
                dofs.push_back(comp_offset + i);
            }
            break;
        }

        case FieldLayout::FieldBlock: {
            // Field is a block, components interleaved within field
            GlobalIndex field_start = field_offsets_[field_idx];
            for (GlobalIndex i = 0; i < n_per_component; ++i) {
                GlobalIndex dof = field_start + i * field.n_components + component;
                dofs.push_back(dof);
            }
            break;
        }
    }

    return IndexSet(std::move(dofs));
}

IndexSet FieldDofMap::getComponentDofs(const std::string& name, LocalIndex component) const {
    auto idx = getFieldIndex(name);
    if (idx < 0) {
        throw FEException("FieldDofMap::getComponentDofs: field '" + name + "' not found");
    }
    return getComponentDofs(static_cast<std::size_t>(idx), component);
}

std::optional<std::pair<int, LocalIndex>>
FieldDofMap::getComponentOfDof(GlobalIndex dof_id) const {
    checkFinalized();

    if (dof_id < 0 || dof_id >= total_dofs_) {
        return std::nullopt;
    }

    // Find which field this DOF belongs to
    for (std::size_t f = 0; f < fields_.size(); ++f) {
        auto [start, end] = getFieldDofRange(f);
        if (dof_id >= start && dof_id < end) {
            const auto& field = fields_[f];
            GlobalIndex local = dof_id - start;

            LocalIndex component = 0;
            switch (layout_) {
                case FieldLayout::Interleaved:
                case FieldLayout::FieldBlock:
                    component = static_cast<LocalIndex>(local % field.n_components);
                    break;
                case FieldLayout::Block: {
                    GlobalIndex n_per_component = field.n_dofs / field.n_components;
                    component = static_cast<LocalIndex>(local / n_per_component);
                    break;
                }
            }

            return std::make_pair(static_cast<int>(f), component);
        }
    }

    return std::nullopt;
}

// =============================================================================
// Subspace Views
// =============================================================================

std::unique_ptr<SubspaceView> FieldDofMap::getFieldView(std::size_t field_idx) const {
    checkFinalized();

    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::getFieldView: invalid field index");
    }

    auto [start, end] = getFieldDofRange(field_idx);

    // Collect all DOFs for this field
    std::vector<GlobalIndex> dofs;
    dofs.reserve(static_cast<std::size_t>(end - start));
    for (GlobalIndex d = start; d < end; ++d) {
        dofs.push_back(d);
    }

    return std::make_unique<SubspaceView>(
        IndexSet(std::move(dofs)),
        fields_[field_idx].name,
        static_cast<int>(field_idx));
}

std::unique_ptr<SubspaceView> FieldDofMap::getFieldView(const std::string& name) const {
    auto idx = getFieldIndex(name);
    if (idx < 0) {
        throw FEException("FieldDofMap::getFieldView: field '" + name + "' not found");
    }
    return getFieldView(static_cast<std::size_t>(idx));
}

std::unique_ptr<SubspaceView> FieldDofMap::getComponentView(
    std::size_t field_idx, std::span<const LocalIndex> components) const {
    checkFinalized();

    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::getComponentView: invalid field index");
    }

    // Collect DOFs for specified components
    IndexSet combined;
    for (auto comp : components) {
        auto comp_dofs = getComponentDofs(field_idx, comp);
        combined = combined.unionWith(comp_dofs);
    }

    std::string name = fields_[field_idx].name + "_components";
    return std::make_unique<SubspaceView>(
        std::move(combined),
        name,
        static_cast<int>(field_idx));
}

// =============================================================================
// Layout Information
// =============================================================================

std::vector<GlobalIndex> FieldDofMap::getBlockSizes() const {
    std::vector<GlobalIndex> sizes;
    sizes.reserve(fields_.size());
    for (const auto& field : fields_) {
        sizes.push_back(field.n_dofs);
    }
    return sizes;
}

std::vector<GlobalIndex> FieldDofMap::getBlockOffsets() const {
    return field_offsets_;
}

// =============================================================================
// DOF Mapping
// =============================================================================

GlobalIndex FieldDofMap::fieldToGlobal(std::size_t field_idx, GlobalIndex local_dof) const {
    checkFinalized();

    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::fieldToGlobal: invalid field index");
    }

    const auto& field = fields_[field_idx];
    if (local_dof < 0 || local_dof >= field.n_dofs) {
        throw FEException("FieldDofMap::fieldToGlobal: local DOF out of range");
    }

    return field_offsets_[field_idx] + local_dof;
}

GlobalIndex FieldDofMap::componentToGlobal(std::size_t field_idx, LocalIndex component,
                                            GlobalIndex local_dof) const {
    checkFinalized();

    if (field_idx >= fields_.size()) {
        throw FEException("FieldDofMap::componentToGlobal: invalid field index");
    }

    const auto& field = fields_[field_idx];
    if (component >= field.n_components) {
        throw FEException("FieldDofMap::componentToGlobal: invalid component");
    }

    GlobalIndex n_per_component = field.n_dofs / field.n_components;
    if (local_dof < 0 || local_dof >= n_per_component) {
        throw FEException("FieldDofMap::componentToGlobal: local DOF out of range");
    }

    GlobalIndex field_start = field_offsets_[field_idx];

    switch (layout_) {
        case FieldLayout::Interleaved:
        case FieldLayout::FieldBlock:
            return field_start + local_dof * field.n_components + component;

        case FieldLayout::Block:
            return field_start + component * n_per_component + local_dof;
    }

    return -1;  // Unreachable
}

std::optional<std::pair<int, GlobalIndex>>
FieldDofMap::globalToField(GlobalIndex dof_id) const {
    checkFinalized();

    if (dof_id < 0 || dof_id >= total_dofs_) {
        return std::nullopt;
    }

    for (std::size_t f = 0; f < fields_.size(); ++f) {
        auto [start, end] = getFieldDofRange(f);
        if (dof_id >= start && dof_id < end) {
            return std::make_pair(static_cast<int>(f), dof_id - start);
        }
    }

    return std::nullopt;
}

// =============================================================================
// Internal Helpers
// =============================================================================

void FieldDofMap::checkFinalized() const {
    if (!finalized_) {
        throw FEException("FieldDofMap: operation requires finalization");
    }
}

void FieldDofMap::checkNotFinalized() const {
    if (finalized_) {
        throw FEException("FieldDofMap: operation not allowed after finalization");
    }
}

void FieldDofMap::computeOffsets() {
    field_offsets_.clear();
    field_offsets_.reserve(fields_.size());

    total_dofs_ = 0;
    total_components_ = 0;

    for (auto& field : fields_) {
        field.dof_offset = total_dofs_;
        field_offsets_.push_back(total_dofs_);
        total_dofs_ += field.n_dofs;
        total_components_ += field.n_components;
    }
}

} // namespace dofs
} // namespace FE
} // namespace svmp
