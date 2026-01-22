/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/MaterialStateProvider.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cstring>
#include <new>

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
                                             std::vector<GlobalIndex> interior_face_ids)
    : num_cells_(num_cells)
    , boundary_face_ids_(std::move(boundary_face_ids))
    , interior_face_ids_(std::move(interior_face_ids))
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
}

MaterialStateProvider::~MaterialStateProvider() = default;

MaterialStateProvider::MaterialStateProvider(MaterialStateProvider&&) noexcept = default;
MaterialStateProvider& MaterialStateProvider::operator=(MaterialStateProvider&&) noexcept = default;

void MaterialStateProvider::addKernel(const assembly::AssemblyKernel& kernel,
                                      assembly::MaterialStateSpec spec,
                                      LocalIndex max_cell_qpts,
                                      LocalIndex max_boundary_face_qpts,
                                      LocalIndex max_interior_face_qpts)
{
    FE_THROW_IF(max_cell_qpts < 0, InvalidArgumentException, "MaterialStateProvider: max_cell_qpts must be >= 0");
    FE_THROW_IF(max_boundary_face_qpts < 0, InvalidArgumentException,
                "MaterialStateProvider: max_boundary_face_qpts must be >= 0");
    FE_THROW_IF(max_interior_face_qpts < 0, InvalidArgumentException,
                "MaterialStateProvider: max_interior_face_qpts must be >= 0");
    FE_THROW_IF(max_cell_qpts == 0 && max_boundary_face_qpts == 0 && max_interior_face_qpts == 0,
                InvalidArgumentException,
                "MaterialStateProvider: at least one max_*_qpts must be > 0");
    FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException, "MaterialStateProvider: bytes_per_qpt must be > 0");
    FE_THROW_IF(spec.alignment == 0u, InvalidArgumentException, "MaterialStateProvider: alignment must be > 0");

    const auto* key = &kernel;
    auto it = states_.find(key);
    if (it != states_.end()) {
        FE_THROW_IF(it->second.bytes_per_qpt != spec.bytes_per_qpt, InvalidArgumentException,
                    "MaterialStateProvider: kernel state spec mismatch (bytes_per_qpt)");
        FE_THROW_IF(it->second.alignment != spec.alignment, InvalidArgumentException,
                    "MaterialStateProvider: kernel state spec mismatch (alignment)");
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

        return;
    }

    KernelState state;
    state.bytes_per_qpt = spec.bytes_per_qpt;
    state.alignment = spec.alignment;
    state.stride_bytes = alignUp(state.bytes_per_qpt, state.alignment);
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
                                       it->second.alignment};
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
                                       it->second.alignment};
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
                                       it->second.alignment};
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
    }
}

void MaterialStateProvider::commitTimeStep()
{
    for (auto& kv : states_) {
        auto& state = kv.second;
        std::swap(state.buffer_old, state.buffer_work);
        std::swap(state.boundary_old, state.boundary_work);
        std::swap(state.interior_old, state.interior_work);
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
