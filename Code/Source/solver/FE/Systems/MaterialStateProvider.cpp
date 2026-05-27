/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/MaterialStateProvider.h"

#include "Core/FEException.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <span>
#include <string>

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] bool isPowerOfTwo(std::size_t x) noexcept
{
    return x != 0u && (x & (x - 1u)) == 0u;
}

[[nodiscard]] std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    FE_THROW_IF(alignment == 0u, InvalidArgumentException, "MaterialStateProvider: alignment must be non-zero");
    FE_THROW_IF(!isPowerOfTwo(alignment), InvalidArgumentException, "MaterialStateProvider: alignment must be power-of-two");
    return (value + alignment - 1u) & ~(alignment - 1u);
}

} // namespace

MaterialStateProvider::AlignedBuffer::~AlignedBuffer()
{
    reset();
}

MaterialStateProvider::AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
{
    *this = std::move(other);
}

MaterialStateProvider::AlignedBuffer& MaterialStateProvider::AlignedBuffer::operator=(AlignedBuffer&& other) noexcept
{
    if (this == &other) {
        return *this;
    }
    reset();
    data = other.data;
    size_bytes = other.size_bytes;
    alignment = other.alignment;
    other.data = nullptr;
    other.size_bytes = 0;
    other.alignment = 1;
    return *this;
}

void MaterialStateProvider::AlignedBuffer::allocate(std::size_t size, std::size_t align)
{
    reset();
    FE_THROW_IF(size == 0u, InvalidArgumentException, "MaterialStateProvider: buffer size must be > 0");
    FE_THROW_IF(align == 0u, InvalidArgumentException, "MaterialStateProvider: alignment must be > 0");
    FE_THROW_IF(!isPowerOfTwo(align), InvalidArgumentException, "MaterialStateProvider: alignment must be power-of-two");

    alignment = align;
    size_bytes = size;
    data = static_cast<std::byte*>(::operator new(size_bytes, std::align_val_t(alignment)));
    std::memset(data, 0, size_bytes);
}

void MaterialStateProvider::AlignedBuffer::reset() noexcept
{
    if (data != nullptr) {
        ::operator delete(data, std::align_val_t(alignment));
        data = nullptr;
    }
    size_bytes = 0;
    alignment = 1;
}

MaterialStateProvider::MaterialStateProvider(GlobalIndex num_cells,
                                             std::vector<GlobalIndex> boundary_face_ids,
                                             std::vector<GlobalIndex> interior_face_ids,
                                             std::vector<std::uint64_t> generated_interface_ids)
    : num_cells_(num_cells)
    , boundary_face_ids_(std::move(boundary_face_ids))
    , interior_face_ids_(std::move(interior_face_ids))
    , generated_interface_ids_(std::move(generated_interface_ids))
{
    FE_THROW_IF(num_cells_ < 0, InvalidArgumentException, "MaterialStateProvider: num_cells must be non-negative");

    boundary_face_index_.reserve(boundary_face_ids_.size());
    for (std::size_t i = 0; i < boundary_face_ids_.size(); ++i) {
        const auto id = boundary_face_ids_[i];
        const auto [ins_it, inserted] = boundary_face_index_.emplace(id, i);
        (void)ins_it;
        FE_THROW_IF(!inserted, InvalidArgumentException,
                    "MaterialStateProvider: duplicate boundary face id");
    }

    interior_face_index_.reserve(interior_face_ids_.size());
    for (std::size_t i = 0; i < interior_face_ids_.size(); ++i) {
        const auto id = interior_face_ids_[i];
        const auto [ins_it, inserted] = interior_face_index_.emplace(id, i);
        (void)ins_it;
        FE_THROW_IF(!inserted, InvalidArgumentException,
                    "MaterialStateProvider: duplicate interior face id");
    }

    generated_interface_index_.reserve(generated_interface_ids_.size());
    for (std::size_t i = 0; i < generated_interface_ids_.size(); ++i) {
        const auto id = generated_interface_ids_[i];
        const auto [ins_it, inserted] = generated_interface_index_.emplace(id, i);
        (void)ins_it;
        FE_THROW_IF(!inserted, InvalidArgumentException,
                    "MaterialStateProvider: duplicate generated interface id");
    }
}

MaterialStateProvider::~MaterialStateProvider() = default;

MaterialStateProvider::MaterialStateProvider(MaterialStateProvider&&) noexcept = default;
MaterialStateProvider& MaterialStateProvider::operator=(MaterialStateProvider&&) noexcept = default;

void MaterialStateProvider::addKernel(const assembly::AssemblyKernel& kernel,
                                      assembly::MaterialStateSpec spec,
                                      LocalIndex max_cell_qpts,
                                      LocalIndex max_boundary_face_qpts,
                                      LocalIndex max_interior_face_qpts,
                                      LocalIndex max_generated_interface_qpts)
{
    FE_THROW_IF(max_cell_qpts < 0, InvalidArgumentException, "MaterialStateProvider: max_cell_qpts must be >= 0");
    FE_THROW_IF(max_boundary_face_qpts < 0, InvalidArgumentException,
                "MaterialStateProvider: max_boundary_face_qpts must be >= 0");
    FE_THROW_IF(max_interior_face_qpts < 0, InvalidArgumentException,
                "MaterialStateProvider: max_interior_face_qpts must be >= 0");
    FE_THROW_IF(max_generated_interface_qpts < 0, InvalidArgumentException,
                "MaterialStateProvider: max_generated_interface_qpts must be >= 0");
    FE_THROW_IF(max_cell_qpts == 0 && max_boundary_face_qpts == 0 &&
                    max_interior_face_qpts == 0 &&
                    max_generated_interface_qpts == 0,
                InvalidArgumentException,
                "MaterialStateProvider: at least one max_*_qpts must be > 0");
    FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException, "MaterialStateProvider: bytes_per_qpt must be > 0");
    FE_THROW_IF(spec.alignment == 0u, InvalidArgumentException, "MaterialStateProvider: alignment must be > 0");
    state::validateStateVariableMetadata(spec.variables, spec.bytes_per_qpt,
                                         "MaterialStateProvider::addKernel");

    const auto* key = &kernel;
    auto it = states_.find(key);
    if (it != states_.end()) {
        FE_THROW_IF(it->second.bytes_per_qpt != spec.bytes_per_qpt, InvalidArgumentException,
                    "MaterialStateProvider: kernel state spec mismatch (bytes_per_qpt)");
        FE_THROW_IF(it->second.alignment != spec.alignment, InvalidArgumentException,
                    "MaterialStateProvider: kernel state spec mismatch (alignment)");
        FE_THROW_IF(it->second.variables != spec.variables, InvalidArgumentException,
                    "MaterialStateProvider: kernel state spec mismatch (state variables)");
        FE_THROW_IF(static_cast<bool>(it->second.frame_transform_hook) !=
                        static_cast<bool>(spec.frame_transform_hook),
                    InvalidArgumentException,
                    "MaterialStateProvider: kernel state spec mismatch (frame transform hook)");
        const auto prev_max_cell_qpts = it->second.max_cell_qpts;
        it->second.max_cell_qpts = std::max(it->second.max_cell_qpts, max_cell_qpts);
        if (it->second.max_cell_qpts != prev_max_cell_qpts) {
            it->second.cell_stride_bytes = static_cast<std::size_t>(it->second.max_cell_qpts) * it->second.stride_bytes;

            const std::size_t total_bytes = static_cast<std::size_t>(num_cells_) * it->second.cell_stride_bytes;
            if (total_bytes == 0u) {
                it->second.buffer_old.reset();
                it->second.buffer_work.reset();
            } else {
                AlignedBuffer new_old;
                AlignedBuffer new_work;
                new_old.allocate(total_bytes, it->second.alignment);
                new_work.allocate(total_bytes, it->second.alignment);

                const std::size_t prev_cell_stride = static_cast<std::size_t>(prev_max_cell_qpts) * it->second.stride_bytes;
                const std::size_t copy_bytes = std::min(prev_cell_stride, it->second.cell_stride_bytes);
                if (copy_bytes > 0u && it->second.buffer_old.data != nullptr && it->second.buffer_work.data != nullptr) {
                    for (std::size_t c = 0; c < static_cast<std::size_t>(num_cells_); ++c) {
                        std::memcpy(new_old.data + c * it->second.cell_stride_bytes,
                                    it->second.buffer_old.data + c * prev_cell_stride,
                                    copy_bytes);
                        std::memcpy(new_work.data + c * it->second.cell_stride_bytes,
                                    it->second.buffer_work.data + c * prev_cell_stride,
                                    copy_bytes);
                    }
                }

                it->second.buffer_old = std::move(new_old);
                it->second.buffer_work = std::move(new_work);
            }
        }

        const auto prev_max_bq = it->second.max_boundary_face_qpts;
        it->second.max_boundary_face_qpts = std::max(it->second.max_boundary_face_qpts, max_boundary_face_qpts);
        if (it->second.max_boundary_face_qpts != prev_max_bq) {
            it->second.boundary_face_stride_bytes =
                static_cast<std::size_t>(it->second.max_boundary_face_qpts) * it->second.stride_bytes;
            const std::size_t total_bytes =
                boundary_face_ids_.size() * it->second.boundary_face_stride_bytes;
            if (total_bytes == 0u) {
                it->second.boundary_old.reset();
                it->second.boundary_work.reset();
            } else {
                AlignedBuffer new_old;
                AlignedBuffer new_work;
                new_old.allocate(total_bytes, it->second.alignment);
                new_work.allocate(total_bytes, it->second.alignment);

                const std::size_t prev_stride = static_cast<std::size_t>(prev_max_bq) * it->second.stride_bytes;
                const std::size_t copy_bytes = std::min(prev_stride, it->second.boundary_face_stride_bytes);
                if (copy_bytes > 0u && it->second.boundary_old.data != nullptr && it->second.boundary_work.data != nullptr) {
                    for (std::size_t f = 0; f < boundary_face_ids_.size(); ++f) {
                        std::memcpy(new_old.data + f * it->second.boundary_face_stride_bytes,
                                    it->second.boundary_old.data + f * prev_stride,
                                    copy_bytes);
                        std::memcpy(new_work.data + f * it->second.boundary_face_stride_bytes,
                                    it->second.boundary_work.data + f * prev_stride,
                                    copy_bytes);
                    }
                }
                it->second.boundary_old = std::move(new_old);
                it->second.boundary_work = std::move(new_work);
            }
        }

        const auto prev_max_iq = it->second.max_interior_face_qpts;
        it->second.max_interior_face_qpts = std::max(it->second.max_interior_face_qpts, max_interior_face_qpts);
        if (it->second.max_interior_face_qpts != prev_max_iq) {
            it->second.interior_face_stride_bytes =
                static_cast<std::size_t>(it->second.max_interior_face_qpts) * it->second.stride_bytes;
            const std::size_t total_bytes =
                interior_face_ids_.size() * it->second.interior_face_stride_bytes;
            if (total_bytes == 0u) {
                it->second.interior_old.reset();
                it->second.interior_work.reset();
            } else {
                AlignedBuffer new_old;
                AlignedBuffer new_work;
                new_old.allocate(total_bytes, it->second.alignment);
                new_work.allocate(total_bytes, it->second.alignment);

                const std::size_t prev_stride = static_cast<std::size_t>(prev_max_iq) * it->second.stride_bytes;
                const std::size_t copy_bytes = std::min(prev_stride, it->second.interior_face_stride_bytes);
                if (copy_bytes > 0u && it->second.interior_old.data != nullptr && it->second.interior_work.data != nullptr) {
                    for (std::size_t f = 0; f < interior_face_ids_.size(); ++f) {
                        std::memcpy(new_old.data + f * it->second.interior_face_stride_bytes,
                                    it->second.interior_old.data + f * prev_stride,
                                    copy_bytes);
                        std::memcpy(new_work.data + f * it->second.interior_face_stride_bytes,
                                    it->second.interior_work.data + f * prev_stride,
                                    copy_bytes);
                    }
                }
                it->second.interior_old = std::move(new_old);
                it->second.interior_work = std::move(new_work);
            }
        }

        const auto prev_max_gq = it->second.max_generated_interface_qpts;
        it->second.max_generated_interface_qpts =
            std::max(it->second.max_generated_interface_qpts,
                     max_generated_interface_qpts);
        if (it->second.max_generated_interface_qpts != prev_max_gq) {
            it->second.generated_interface_stride_bytes =
                static_cast<std::size_t>(
                    it->second.max_generated_interface_qpts) *
                it->second.stride_bytes;
            const std::size_t total_bytes =
                generated_interface_ids_.size() *
                it->second.generated_interface_stride_bytes;
            if (total_bytes == 0u) {
                it->second.generated_interface_old.reset();
                it->second.generated_interface_work.reset();
            } else {
                AlignedBuffer new_old;
                AlignedBuffer new_work;
                new_old.allocate(total_bytes, it->second.alignment);
                new_work.allocate(total_bytes, it->second.alignment);

                const std::size_t prev_stride =
                    static_cast<std::size_t>(prev_max_gq) *
                    it->second.stride_bytes;
                const std::size_t copy_bytes =
                    std::min(prev_stride,
                             it->second.generated_interface_stride_bytes);
                if (copy_bytes > 0u &&
                    it->second.generated_interface_old.data != nullptr &&
                    it->second.generated_interface_work.data != nullptr) {
                    for (std::size_t f = 0; f < generated_interface_ids_.size(); ++f) {
                        std::memcpy(
                            new_old.data +
                                f * it->second.generated_interface_stride_bytes,
                            it->second.generated_interface_old.data +
                                f * prev_stride,
                            copy_bytes);
                        std::memcpy(
                            new_work.data +
                                f * it->second.generated_interface_stride_bytes,
                            it->second.generated_interface_work.data +
                                f * prev_stride,
                            copy_bytes);
                    }
                }
                it->second.generated_interface_old = std::move(new_old);
                it->second.generated_interface_work = std::move(new_work);
            }
        }

        return;
    }

    KernelState state;
    state.bytes_per_qpt = spec.bytes_per_qpt;
    state.alignment = spec.alignment;
    state.stride_bytes = alignUp(state.bytes_per_qpt, state.alignment);
    state.variables = std::move(spec.variables);
    state.frame_transform_hook = std::move(spec.frame_transform_hook);
    state.max_cell_qpts = max_cell_qpts;
    state.cell_stride_bytes = static_cast<std::size_t>(state.max_cell_qpts) * state.stride_bytes;

    const std::size_t cell_total_bytes = static_cast<std::size_t>(num_cells_) * state.cell_stride_bytes;
    if (cell_total_bytes > 0u) {
        state.buffer_old.allocate(cell_total_bytes, state.alignment);
        state.buffer_work.allocate(cell_total_bytes, state.alignment);
    }

    state.max_boundary_face_qpts = max_boundary_face_qpts;
    state.boundary_face_stride_bytes = static_cast<std::size_t>(state.max_boundary_face_qpts) * state.stride_bytes;
    const std::size_t boundary_total_bytes = boundary_face_ids_.size() * state.boundary_face_stride_bytes;
    if (boundary_total_bytes > 0u) {
        state.boundary_old.allocate(boundary_total_bytes, state.alignment);
        state.boundary_work.allocate(boundary_total_bytes, state.alignment);
    }

    state.max_interior_face_qpts = max_interior_face_qpts;
    state.interior_face_stride_bytes = static_cast<std::size_t>(state.max_interior_face_qpts) * state.stride_bytes;
    const std::size_t interior_total_bytes = interior_face_ids_.size() * state.interior_face_stride_bytes;
    if (interior_total_bytes > 0u) {
        state.interior_old.allocate(interior_total_bytes, state.alignment);
        state.interior_work.allocate(interior_total_bytes, state.alignment);
    }

    state.max_generated_interface_qpts = max_generated_interface_qpts;
    state.generated_interface_stride_bytes =
        static_cast<std::size_t>(state.max_generated_interface_qpts) *
        state.stride_bytes;
    const std::size_t generated_interface_total_bytes =
        generated_interface_ids_.size() * state.generated_interface_stride_bytes;
    if (generated_interface_total_bytes > 0u) {
        state.generated_interface_old.allocate(generated_interface_total_bytes,
                                               state.alignment);
        state.generated_interface_work.allocate(generated_interface_total_bytes,
                                                state.alignment);
    }

    states_.emplace(key, std::move(state));
}

assembly::MaterialStateView MaterialStateProvider::getCellState(const assembly::AssemblyKernel& kernel,
                                                                GlobalIndex cell_id,
                                                                LocalIndex num_qpts)
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }

    FE_THROW_IF(cell_id < 0 || cell_id >= num_cells_, InvalidArgumentException,
                "MaterialStateProvider: cell_id out of bounds");
    FE_THROW_IF(num_qpts < 0 || num_qpts > it->second.max_cell_qpts, InvalidArgumentException,
                "MaterialStateProvider: requested num_qpts exceeds allocated max_qpts");

    auto* old_base = it->second.buffer_old.data + static_cast<std::size_t>(cell_id) * it->second.cell_stride_bytes;
    auto* work_base = it->second.buffer_work.data + static_cast<std::size_t>(cell_id) * it->second.cell_stride_bytes;
    return assembly::MaterialStateView{old_base, work_base, it->second.bytes_per_qpt, it->second.stride_bytes,
                                       it->second.alignment,
                                       std::span<const state::StateVariableMetadata>(
                                           it->second.variables.data(), it->second.variables.size()),
                                       state::StateVariableLifecycle::CommittedOld,
                                       work_lifecycle_};
}

assembly::MaterialStateView MaterialStateProvider::getBoundaryFaceState(const assembly::AssemblyKernel& kernel,
                                                                        GlobalIndex face_id,
                                                                        LocalIndex num_qpts)
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }

    const auto idx_it = boundary_face_index_.find(face_id);
    if (idx_it == boundary_face_index_.end()) {
        return {};
    }

    FE_THROW_IF(num_qpts < 0 || num_qpts > it->second.max_boundary_face_qpts, InvalidArgumentException,
                "MaterialStateProvider: requested boundary-face num_qpts exceeds allocated max_qpts");

    if (it->second.boundary_old.data == nullptr || it->second.boundary_work.data == nullptr) {
        return {};
    }

    const auto idx = idx_it->second;
    auto* old_base = it->second.boundary_old.data + idx * it->second.boundary_face_stride_bytes;
    auto* work_base = it->second.boundary_work.data + idx * it->second.boundary_face_stride_bytes;
    return assembly::MaterialStateView{old_base, work_base, it->second.bytes_per_qpt, it->second.stride_bytes,
                                       it->second.alignment,
                                       std::span<const state::StateVariableMetadata>(
                                           it->second.variables.data(), it->second.variables.size()),
                                       state::StateVariableLifecycle::CommittedOld,
                                       work_lifecycle_};
}

assembly::MaterialStateView MaterialStateProvider::getInteriorFaceState(const assembly::AssemblyKernel& kernel,
                                                                        GlobalIndex face_id,
                                                                        LocalIndex num_qpts)
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }

    const auto idx_it = interior_face_index_.find(face_id);
    if (idx_it == interior_face_index_.end()) {
        return {};
    }

    FE_THROW_IF(num_qpts < 0 || num_qpts > it->second.max_interior_face_qpts, InvalidArgumentException,
                "MaterialStateProvider: requested interior-face num_qpts exceeds allocated max_qpts");

    if (it->second.interior_old.data == nullptr || it->second.interior_work.data == nullptr) {
        return {};
    }

    const auto idx = idx_it->second;
    auto* old_base = it->second.interior_old.data + idx * it->second.interior_face_stride_bytes;
    auto* work_base = it->second.interior_work.data + idx * it->second.interior_face_stride_bytes;
    return assembly::MaterialStateView{old_base, work_base, it->second.bytes_per_qpt, it->second.stride_bytes,
                                       it->second.alignment,
                                       std::span<const state::StateVariableMetadata>(
                                           it->second.variables.data(), it->second.variables.size()),
                                       state::StateVariableLifecycle::CommittedOld,
                                       work_lifecycle_};
}

assembly::MaterialStateView MaterialStateProvider::getGeneratedInterfaceState(
    const assembly::AssemblyKernel& kernel,
    GlobalIndex /*parent_cell_id*/,
    int /*interface_marker*/,
    std::uint64_t cut_topology_revision,
    LocalIndex num_qpts)
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }

    const auto idx_it = generated_interface_index_.find(cut_topology_revision);
    if (idx_it == generated_interface_index_.end()) {
        return {};
    }

    FE_THROW_IF(num_qpts < 0 ||
                    num_qpts > it->second.max_generated_interface_qpts,
                InvalidArgumentException,
                "MaterialStateProvider: requested generated-interface num_qpts exceeds allocated max_qpts");

    if (it->second.generated_interface_old.data == nullptr ||
        it->second.generated_interface_work.data == nullptr) {
        return {};
    }

    const auto idx = idx_it->second;
    auto* old_base =
        it->second.generated_interface_old.data +
        idx * it->second.generated_interface_stride_bytes;
    auto* work_base =
        it->second.generated_interface_work.data +
        idx * it->second.generated_interface_stride_bytes;
    return assembly::MaterialStateView{
        old_base,
        work_base,
        it->second.bytes_per_qpt,
        it->second.stride_bytes,
        it->second.alignment,
        std::span<const state::StateVariableMetadata>(
            it->second.variables.data(), it->second.variables.size()),
        state::StateVariableLifecycle::CommittedOld,
        work_lifecycle_};
}

void MaterialStateProvider::beginTimeStep()
{
    for (auto& kv : states_) {
        auto& state = kv.second;
        if (state.buffer_old.data == nullptr || state.buffer_work.data == nullptr) {
            // still copy face buffers if present
        } else {
            FE_THROW_IF(state.buffer_old.size_bytes != state.buffer_work.size_bytes, FEException,
                        "MaterialStateProvider: beginTimeStep size mismatch");
            std::memcpy(state.buffer_work.data, state.buffer_old.data, state.buffer_old.size_bytes);
        }

        if (state.boundary_old.data != nullptr && state.boundary_work.data != nullptr) {
            FE_THROW_IF(state.boundary_old.size_bytes != state.boundary_work.size_bytes, FEException,
                        "MaterialStateProvider: beginTimeStep boundary size mismatch");
            std::memcpy(state.boundary_work.data, state.boundary_old.data, state.boundary_old.size_bytes);
        }

        if (state.interior_old.data != nullptr && state.interior_work.data != nullptr) {
            FE_THROW_IF(state.interior_old.size_bytes != state.interior_work.size_bytes, FEException,
                        "MaterialStateProvider: beginTimeStep interior size mismatch");
            std::memcpy(state.interior_work.data, state.interior_old.data, state.interior_old.size_bytes);
        }

        if (state.generated_interface_old.data != nullptr &&
            state.generated_interface_work.data != nullptr) {
            FE_THROW_IF(
                state.generated_interface_old.size_bytes !=
                    state.generated_interface_work.size_bytes,
                FEException,
                "MaterialStateProvider: beginTimeStep generated-interface size mismatch");
            std::memcpy(state.generated_interface_work.data,
                        state.generated_interface_old.data,
                        state.generated_interface_old.size_bytes);
        }
    }
    work_lifecycle_ = state::StateVariableLifecycle::TrialWork;
}

void MaterialStateProvider::commitTimeStep()
{
    for (auto& kv : states_) {
        auto& state = kv.second;
        std::swap(state.buffer_old, state.buffer_work);
        std::swap(state.boundary_old, state.boundary_work);
        std::swap(state.interior_old, state.interior_work);
        std::swap(state.generated_interface_old,
                  state.generated_interface_work);
    }
    work_lifecycle_ = state::StateVariableLifecycle::Accepted;
}

void MaterialStateProvider::rollbackTimeStep()
{
    beginTimeStep();
    work_lifecycle_ = state::StateVariableLifecycle::RolledBack;
}

state::StateFrameTransformResult MaterialStateProvider::applyStateFrameTransform(
    const state::StateFrameTransformRequest& request)
{
    state::StateFrameTransformResult result;

    for (auto& kv : states_) {
        auto& kernel_state = kv.second;

        for (const auto& variable : kernel_state.variables) {
            ++result.variables_seen;
            const bool requires_action =
                state::stateVariableRequiresFrameAction(variable, request.event);
            if (!requires_action) {
                ++result.variable_instances_preserved;
                continue;
            }

            ++result.variables_requiring_action;
            FE_THROW_IF(!kernel_state.frame_transform_hook, InvalidStateException,
                        "MaterialStateProvider: state variable '" + variable.name +
                            "' declares a frame transform policy but no transform hook");

            const auto apply_buffer = [&](state::StateStorageDomain domain,
                                          std::string_view storage_name,
                                          const AlignedBuffer& old_buffer,
                                          AlignedBuffer& work_buffer,
                                          std::size_t entity_count,
                                          std::size_t entity_stride_bytes,
                                          LocalIndex max_qpts,
                                          const std::vector<GlobalIndex>* entity_ids) {
                if (old_buffer.data == nullptr || work_buffer.data == nullptr ||
                    entity_count == 0u || max_qpts <= 0) {
                    return;
                }
                for (std::size_t entity = 0; entity < entity_count; ++entity) {
                    const auto entity_id = (entity_ids != nullptr)
                        ? (*entity_ids)[entity]
                        : static_cast<GlobalIndex>(entity);
                    for (LocalIndex q = 0; q < max_qpts; ++q) {
                        const auto q_offset =
                            entity * entity_stride_bytes +
                            static_cast<std::size_t>(q) * kernel_state.stride_bytes +
                            variable.offset_bytes;
                        state::StateVariableTransformContext ctx;
                        ctx.request = request;
                        ctx.storage_domain = domain;
                        ctx.storage_name = storage_name;
                        ctx.entity_id = entity_id;
                        ctx.quadrature_point = static_cast<int>(q);
                        ctx.variable = variable;
                        ctx.old_value = std::span<const std::byte>(
                            old_buffer.data + q_offset, variable.size_bytes);
                        ctx.work_value = std::span<std::byte>(
                            work_buffer.data + q_offset, variable.size_bytes);
                        kernel_state.frame_transform_hook(ctx);
                        ++result.variable_instances_transformed;
                    }
                }
            };

            apply_buffer(state::StateStorageDomain::MaterialCellQuadrature,
                         "cell",
                         kernel_state.buffer_old,
                         kernel_state.buffer_work,
                         static_cast<std::size_t>(num_cells_),
                         kernel_state.cell_stride_bytes,
                         kernel_state.max_cell_qpts,
                         nullptr);
            apply_buffer(state::StateStorageDomain::MaterialBoundaryFaceQuadrature,
                         "boundary-face",
                         kernel_state.boundary_old,
                         kernel_state.boundary_work,
                         boundary_face_ids_.size(),
                         kernel_state.boundary_face_stride_bytes,
                         kernel_state.max_boundary_face_qpts,
                         &boundary_face_ids_);
            apply_buffer(state::StateStorageDomain::MaterialInteriorFaceQuadrature,
                         "interior-face",
                         kernel_state.interior_old,
                         kernel_state.interior_work,
                         interior_face_ids_.size(),
                         kernel_state.interior_face_stride_bytes,
                         kernel_state.max_interior_face_qpts,
                         &interior_face_ids_);

            if (kernel_state.generated_interface_old.data != nullptr &&
                kernel_state.generated_interface_work.data != nullptr &&
                !generated_interface_ids_.empty() &&
                kernel_state.max_generated_interface_qpts > 0) {
                for (std::size_t entity = 0;
                     entity < generated_interface_ids_.size();
                     ++entity) {
                    for (LocalIndex q = 0;
                         q < kernel_state.max_generated_interface_qpts;
                         ++q) {
                        const auto q_offset =
                            entity *
                                kernel_state
                                    .generated_interface_stride_bytes +
                            static_cast<std::size_t>(q) *
                                kernel_state.stride_bytes +
                            variable.offset_bytes;
                        state::StateVariableTransformContext ctx;
                        ctx.request = request;
                        ctx.storage_domain =
                            state::StateStorageDomain::
                                MaterialGeneratedInterfaceQuadrature;
                        ctx.storage_name = "generated-interface";
                        ctx.entity_id = static_cast<GlobalIndex>(
                            generated_interface_ids_[entity]);
                        ctx.quadrature_point = static_cast<int>(q);
                        ctx.variable = variable;
                        ctx.old_value = std::span<const std::byte>(
                            kernel_state.generated_interface_old.data + q_offset,
                            variable.size_bytes);
                        ctx.work_value = std::span<std::byte>(
                            kernel_state.generated_interface_work.data + q_offset,
                            variable.size_bytes);
                        kernel_state.frame_transform_hook(ctx);
                        ++result.variable_instances_transformed;
                    }
                }
            }
        }
    }

    return result;
}

std::span<const state::StateVariableMetadata>
MaterialStateProvider::materialStateVariables(const assembly::AssemblyKernel& kernel) const noexcept
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }
    return {it->second.variables.data(), it->second.variables.size()};
}

} // namespace systems
} // namespace FE
} // namespace svmp
