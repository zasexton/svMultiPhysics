#ifndef SVMP_FE_SYSTEMS_GLOBAL_KERNEL_STATE_PROVIDER_H
#define SVMP_FE_SYSTEMS_GLOBAL_KERNEL_STATE_PROVIDER_H

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Systems/GlobalKernel.h"

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Persistent state storage for systems::GlobalKernel instances
 *
 * Mirrors the per-kernel MaterialStateProvider used for element/face kernels,
 * but keyed on `systems::GlobalKernel` and exposed directly to global kernels.
 *
 * Storage is allocated per kernel instance and per entity (cell / boundary face
 * / interior face), with fixed max_qpts stride and old/work double-buffering.
 */
class GlobalKernelStateProvider final {
public:
    explicit GlobalKernelStateProvider(GlobalIndex num_cells,
                                       std::vector<GlobalIndex> boundary_face_ids = {},
                                       std::vector<GlobalIndex> interior_face_ids = {});
    ~GlobalKernelStateProvider();

    GlobalKernelStateProvider(GlobalKernelStateProvider&&) noexcept;
    GlobalKernelStateProvider& operator=(GlobalKernelStateProvider&&) noexcept;

    GlobalKernelStateProvider(const GlobalKernelStateProvider&) = delete;
    GlobalKernelStateProvider& operator=(const GlobalKernelStateProvider&) = delete;

    void addKernel(const GlobalKernel& kernel, GlobalStateSpec spec);

    [[nodiscard]] assembly::MaterialStateView getCellState(const GlobalKernel& kernel,
                                                           GlobalIndex cell_id,
                                                           LocalIndex num_qpts) const;

    [[nodiscard]] assembly::MaterialStateView getBoundaryFaceState(const GlobalKernel& kernel,
                                                                   GlobalIndex face_id,
                                                                   LocalIndex num_qpts) const;

    [[nodiscard]] assembly::MaterialStateView getInteriorFaceState(const GlobalKernel& kernel,
                                                                   GlobalIndex face_id,
                                                                   LocalIndex num_qpts) const;

    void beginTimeStep();
    void commitTimeStep();

private:
    struct AlignedBuffer {
        std::byte* data{nullptr};
        std::size_t size_bytes{0};
        std::size_t alignment{1};

        AlignedBuffer() = default;
        ~AlignedBuffer();

        AlignedBuffer(AlignedBuffer&& other) noexcept;
        AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

        AlignedBuffer(const AlignedBuffer&) = delete;
        AlignedBuffer& operator=(const AlignedBuffer&) = delete;

        void allocate(std::size_t size, std::size_t align);
        void reset() noexcept;
    };

    struct KernelState {
        std::size_t bytes_per_qpt{0};
        std::size_t stride_bytes{0};
        std::size_t alignment{1};

        LocalIndex max_cell_qpts{0};
        std::size_t cell_stride_bytes{0};
        AlignedBuffer cell_old{};
        AlignedBuffer cell_work{};

        LocalIndex max_boundary_face_qpts{0};
        std::size_t boundary_face_stride_bytes{0};
        AlignedBuffer boundary_old{};
        AlignedBuffer boundary_work{};

        LocalIndex max_interior_face_qpts{0};
        std::size_t interior_face_stride_bytes{0};
        AlignedBuffer interior_old{};
        AlignedBuffer interior_work{};
    };

    GlobalIndex num_cells_{0};
    std::vector<GlobalIndex> boundary_face_ids_{};
    std::vector<GlobalIndex> interior_face_ids_{};
    std::unordered_map<GlobalIndex, std::size_t> boundary_face_index_{};
    std::unordered_map<GlobalIndex, std::size_t> interior_face_index_{};
    std::unordered_map<const GlobalKernel*, KernelState> states_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_GLOBAL_KERNEL_STATE_PROVIDER_H

