#ifndef SVMP_FE_SYSTEMS_MATERIAL_STATE_PROVIDER_H
#define SVMP_FE_SYSTEMS_MATERIAL_STATE_PROVIDER_H

#include "Assembly/Assembler.h"
#include "Assembly/AssemblyKernel.h"
#include "Core/Types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Per-kernel, per-cell state storage provider
 *
 * Systems allocates one state buffer per kernel instance that requests
 * RequiredData::MaterialState. Storage is laid out as:
 *   [cell0 qp0][cell0 qp1]...[cell0 qp(max_qpts-1)]
 *   [cell1 qp0]...
 */
class MaterialStateProvider final : public assembly::IMaterialStateProvider {
public:
    explicit MaterialStateProvider(GlobalIndex num_cells,
                                   std::vector<GlobalIndex> boundary_face_ids = {},
                                   std::vector<GlobalIndex> interior_face_ids = {});
    ~MaterialStateProvider() override;

    MaterialStateProvider(MaterialStateProvider&&) noexcept;
    MaterialStateProvider& operator=(MaterialStateProvider&&) noexcept;

    MaterialStateProvider(const MaterialStateProvider&) = delete;
    MaterialStateProvider& operator=(const MaterialStateProvider&) = delete;

    void addKernel(const assembly::AssemblyKernel& kernel,
                   assembly::MaterialStateSpec spec,
                   LocalIndex max_cell_qpts,
                   LocalIndex max_boundary_face_qpts = 0,
                   LocalIndex max_interior_face_qpts = 0);

    [[nodiscard]] assembly::MaterialStateView getCellState(const assembly::AssemblyKernel& kernel,
                                                           GlobalIndex cell_id,
                                                           LocalIndex num_qpts) override;

    [[nodiscard]] assembly::MaterialStateView getBoundaryFaceState(const assembly::AssemblyKernel& kernel,
                                                                   GlobalIndex face_id,
                                                                   LocalIndex num_qpts) override;

    [[nodiscard]] assembly::MaterialStateView getInteriorFaceState(const assembly::AssemblyKernel& kernel,
                                                                   GlobalIndex face_id,
                                                                   LocalIndex num_qpts) override;

    void beginTimeStep() override;
    void commitTimeStep() override;

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
        AlignedBuffer buffer_old{};
        AlignedBuffer buffer_work{};

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
    std::unordered_map<const assembly::AssemblyKernel*, KernelState> states_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_MATERIAL_STATE_PROVIDER_H
