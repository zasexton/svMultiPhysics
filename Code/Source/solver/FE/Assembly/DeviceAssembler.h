/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_FE_ASSEMBLY_DEVICE_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_DEVICE_ASSEMBLER_H

/**
 * @file DeviceAssembler.h
 * @brief GPU-oriented assembly strategy with partial assembly
 *
 * DeviceAssembler provides GPU-accelerated finite element assembly following
 * the partial assembly paradigm (libCEED-style). Key design principles:
 *
 * 1. PARTIAL ASSEMBLY:
 *    Instead of forming the full sparse matrix, compute and store only
 *    element-local geometric data (G), then apply operators on-the-fly:
 *      y = A * x  where  A = B^T G B
 *    - B: basis operator (reference to physical transformation)
 *    - G: geometric/material data at quadrature points
 *
 * 2. MATRIX-FREE OPERATOR APPLICATION:
 *    GPU kernels apply operators without forming global matrices.
 *    - Setup phase: compute and cache G factors
 *    - Apply phase: efficient GPU kernels for B^T G B x
 *    - Memory efficient: O(n_elem * n_qpts) instead of O(nnz)
 *
 * 3. DEVICE MEMORY MANAGEMENT:
 *    Efficient host-device data transfer and memory pooling.
 *    - Asynchronous transfers with streams
 *    - Memory pools to avoid allocation overhead
 *    - Unified memory support for simplified programming
 *
 * 4. BACKEND ABSTRACTION:
 *    Supports multiple GPU programming models.
 *    - CUDA for NVIDIA GPUs
 *    - HIP for AMD GPUs
 *    - SYCL for Intel GPUs (future)
 *    - CPU fallback for testing
 *
 * Assembly Levels (following MFEM conventions):
 * - FULL: Traditional full assembly (device-accelerated)
 * - ELEMENT: Store element matrices (E-vector style)
 * - PARTIAL: Store quadrature data only
 * - NONE: Full matrix-free (recompute everything)
 *
 * Module Boundaries:
 * - DeviceAssembler OWNS: GPU orchestration, memory management, kernel launch
 * - DeviceAssembler does NOT OWN: physics kernels (DeviceKernel interface)
 *
 * @see MatrixFreeAssembler for CPU matrix-free operations
 * @see libCEED for inspiration on partial assembly patterns
 */

#include "Core/Types.h"
#include "Assembler.h"
#include "MatrixFreeAssembler.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"
#include "GlobalSystemView.h"

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <optional>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofHandler;
}

namespace spaces {
    class FunctionSpace;
}

namespace sparsity {
    class SparsityPattern;
}

namespace constraints {
    class AffineConstraints;
}

namespace assembly {

// ============================================================================
// Device Backend Types
// ============================================================================

/**
 * @brief GPU backend selection
 */
enum class DeviceBackend {
    CPU,      ///< CPU fallback (for testing/debugging)
    CUDA,     ///< NVIDIA CUDA
    HIP,      ///< AMD HIP (ROCm)
    SYCL,     ///< Intel SYCL/oneAPI
    Auto      ///< Automatically detect best available
};

/**
 * @brief Assembly level for device
 */
enum class DeviceAssemblyLevel {
    /**
     * @brief Full assembly into sparse matrix (on device)
     */
    Full,

    /**
     * @brief Store element matrices (E-vector approach)
     */
    Element,

    /**
     * @brief Partial assembly: store quadrature-point data only
     */
    Partial,

    /**
     * @brief Full matrix-free: recompute everything each apply
     */
    None
};

/**
 * @brief Memory allocation strategy
 */
enum class DeviceMemoryMode {
    /**
     * @brief Explicit host/device memory (manual transfers)
     */
    Explicit,

    /**
     * @brief Unified/managed memory (automatic migration)
     */
    Unified,

    /**
     * @brief Pinned host memory (fast transfers)
     */
    Pinned
};

// ============================================================================
// Device Configuration
// ============================================================================

/**
 * @brief Configuration for device assembly
 */
struct DeviceOptions {
    /**
     * @brief GPU backend to use
     */
    DeviceBackend backend{DeviceBackend::Auto};

    /**
     * @brief Assembly level
     */
    DeviceAssemblyLevel assembly_level{DeviceAssemblyLevel::Partial};

    /**
     * @brief Memory allocation mode
     */
    DeviceMemoryMode memory_mode{DeviceMemoryMode::Explicit};

    /**
     * @brief Device ID (for multi-GPU systems)
     */
    int device_id{0};

    /**
     * @brief Number of CUDA streams for async operations
     */
    int num_streams{2};

    /**
     * @brief Batch size for element processing
     */
    std::size_t batch_size{1024};

    /**
     * @brief Enable device memory pooling
     */
    bool use_memory_pool{true};

    /**
     * @brief Initial memory pool size (bytes)
     */
    std::size_t pool_initial_size{64 * 1024 * 1024};  // 64 MB

    /**
     * @brief Enable async host-device transfers
     */
    bool async_transfers{true};

    /**
     * @brief Prefetch data to device
     */
    bool prefetch{true};

    /**
     * @brief Verbose output
     */
    bool verbose{false};
};

/**
 * @brief Device information and capabilities
 */
struct DeviceInfo {
    std::string name;
    std::string driver_version;
    std::size_t total_memory{0};
    std::size_t free_memory{0};
    int compute_capability_major{0};
    int compute_capability_minor{0};
    int num_multiprocessors{0};
    int max_threads_per_block{0};
    int warp_size{0};
    bool supports_unified_memory{false};
    bool supports_cooperative_groups{false};
};

/**
 * @brief Statistics from device assembly
 */
struct DeviceAssemblyStats {
    GlobalIndex elements_processed{0};
    double total_seconds{0.0};
    double setup_seconds{0.0};
    double transfer_seconds{0.0};
    double kernel_seconds{0.0};
    std::size_t host_to_device_bytes{0};
    std::size_t device_to_host_bytes{0};
    std::size_t device_memory_used{0};
};

// ============================================================================
// Device Memory Handle
// ============================================================================

/**
 * @brief RAII wrapper for device memory allocation
 */
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() = default;
    ~DeviceMemory();

    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;

    // Non-copyable
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    /**
     * @brief Allocate device memory
     */
    void allocate(std::size_t count);

    /**
     * @brief Free device memory
     */
    void free();

    /**
     * @brief Copy from host to device
     */
    void copyFromHost(const T* host_ptr, std::size_t count);

    /**
     * @brief Copy from device to host
     */
    void copyToHost(T* host_ptr, std::size_t count) const;

    /**
     * @brief Get device pointer
     */
    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }

    /**
     * @brief Get allocation size
     */
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /**
     * @brief Check if allocated
     */
    [[nodiscard]] bool isAllocated() const noexcept { return ptr_ != nullptr; }

private:
    T* ptr_{nullptr};
    std::size_t size_{0};
    DeviceBackend backend_{DeviceBackend::CPU};
};

// ============================================================================
// Device Kernel Interface
// ============================================================================

/**
 * @brief Abstract interface for device-side element kernels
 *
 * DeviceKernel defines the interface that physics kernels must implement
 * for GPU execution. The kernel is responsible for:
 * 1. Setup: Compute and store quadrature-point data
 * 2. Apply: Perform operator application using stored data
 */
class DeviceKernel {
public:
    virtual ~DeviceKernel() = default;

    /**
     * @brief Get required data for kernel
     */
    [[nodiscard]] virtual RequiredData getRequiredData() const noexcept = 0;

    /**
     * @brief Setup phase: compute and store quadrature data
     *
     * Called once before operator applications. Computes geometric factors
     * and material data at quadrature points.
     *
     * @param num_elements Number of elements
     * @param element_data Element geometric data (device pointer)
     * @param qpt_data Output: quadrature point data (device pointer)
     */
    virtual void setup(
        std::size_t num_elements,
        const void* element_data,
        void* qpt_data) = 0;

    /**
     * @brief Apply operator: y = A * x
     *
     * @param num_elements Number of elements
     * @param qpt_data Quadrature point data from setup
     * @param x Input vector (device pointer)
     * @param y Output vector (device pointer)
     */
    virtual void apply(
        std::size_t num_elements,
        const void* qpt_data,
        const Real* x,
        Real* y) = 0;

    /**
     * @brief Get size of per-quadrature-point data (bytes)
     */
    [[nodiscard]] virtual std::size_t qptDataSize() const noexcept = 0;

    /**
     * @brief Get kernel name for debugging
     */
    [[nodiscard]] virtual std::string name() const { return "DeviceKernel"; }
};

// ============================================================================
// Partial Assembly Data
// ============================================================================

/**
 * @brief Storage for partial assembly quadrature data
 */
class PartialAssemblyData {
public:
    PartialAssemblyData();
    ~PartialAssemblyData();

    PartialAssemblyData(PartialAssemblyData&& other) noexcept;
    PartialAssemblyData& operator=(PartialAssemblyData&& other) noexcept;

    // Non-copyable
    PartialAssemblyData(const PartialAssemblyData&) = delete;
    PartialAssemblyData& operator=(const PartialAssemblyData&) = delete;

    /**
     * @brief Allocate storage
     *
     * @param num_elements Number of elements
     * @param num_qpts Quadrature points per element
     * @param data_per_qpt Bytes per quadrature point
     */
    void allocate(std::size_t num_elements, std::size_t num_qpts,
                  std::size_t data_per_qpt);

    /**
     * @brief Free storage
     */
    void free();

    /**
     * @brief Get device pointer to quadrature data
     */
    [[nodiscard]] void* deviceData() noexcept;
    [[nodiscard]] const void* deviceData() const noexcept;

    /**
     * @brief Get storage info
     */
    [[nodiscard]] std::size_t numElements() const noexcept { return num_elements_; }
    [[nodiscard]] std::size_t numQpts() const noexcept { return num_qpts_; }
    [[nodiscard]] std::size_t totalBytes() const noexcept { return total_bytes_; }

    /**
     * @brief Check if populated
     */
    [[nodiscard]] bool isPopulated() const noexcept { return populated_; }
    void setPopulated(bool val) noexcept { populated_ = val; }

private:
    std::size_t num_elements_{0};
    std::size_t num_qpts_{0};
    std::size_t total_bytes_{0};
    DeviceMemory<char> device_storage_;
    bool populated_{false};
};

// ============================================================================
// DeviceAssembler
// ============================================================================

/**
 * @brief GPU-accelerated assembler with partial assembly support
 *
 * DeviceAssembler provides high-performance finite element assembly on GPUs.
 * It supports full assembly, element-wise storage, and partial assembly modes.
 *
 * Usage for matrix-free solves:
 * @code
 *   DeviceOptions options{
 *       .backend = DeviceBackend::CUDA,
 *       .assembly_level = DeviceAssemblyLevel::Partial
 *   };
 *   DeviceAssembler assembler(options);
 *   assembler.setDofMap(dof_map);
 *
 *   // Setup phase (compute quadrature data)
 *   assembler.setup(mesh, space, kernel);
 *
 *   // Apply operator (repeated in iterative solver)
 *   for (int iter = 0; iter < max_iters; ++iter) {
 *       assembler.apply(x, y);  // y = A * x
 *       // ... solver iteration ...
 *   }
 * @endcode
 *
 * Usage for full assembly:
 * @code
 *   DeviceOptions options{
 *       .backend = DeviceBackend::CUDA,
 *       .assembly_level = DeviceAssemblyLevel::Full
 *   };
 *   DeviceAssembler assembler(options);
 *   assembler.setDofMap(dof_map);
 *
 *   // Assemble matrix on device
 *   assembler.assembleMatrix(mesh, test_space, trial_space, kernel, matrix_view);
 * @endcode
 */
class DeviceAssembler : public Assembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    DeviceAssembler();
    explicit DeviceAssembler(const DeviceOptions& options);
    ~DeviceAssembler() override;

    DeviceAssembler(DeviceAssembler&& other) noexcept;
    DeviceAssembler& operator=(DeviceAssembler&& other) noexcept;

    // Non-copyable
    DeviceAssembler(const DeviceAssembler&) = delete;
    DeviceAssembler& operator=(const DeviceAssembler&) = delete;

    // =========================================================================
    // Configuration (Assembler interface)
    // =========================================================================

    void setDofMap(const dofs::DofMap& dof_map) override;
    void setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset) override;
    void setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset) override;
    void setDofHandler(const dofs::DofHandler& dof_handler) override;
    void setConstraints(const constraints::AffineConstraints* constraints) override;
    void setSparsityPattern(const sparsity::SparsityPattern* sparsity) override;
    void setOptions(const AssemblyOptions& options) override;
    void setCurrentSolution(std::span<const Real> solution) override;
    void setCurrentSolutionView(const GlobalSystemView* solution_view) override;
    void setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields) override;
    void setPreviousSolution(std::span<const Real> solution) override;
    void setPreviousSolution2(std::span<const Real> solution) override;
    void setPreviousSolutionK(int k, std::span<const Real> solution) override;
    void setTimeIntegrationContext(const TimeIntegrationContext* ctx) override;
    void setTime(Real time) override;
    void setTimeStep(Real dt) override;
    void setRealParameterGetter(
        const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept override;
    void setParameterGetter(
        const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept override;
    void setUserData(const void* user_data) noexcept override;
    void setJITConstants(std::span<const Real> constants) noexcept override;
    void setCoupledValues(std::span<const Real> integrals,
                          std::span<const Real> aux_state) noexcept override;
    void setMaterialStateProvider(IMaterialStateProvider* provider) noexcept override;

    [[nodiscard]] const AssemblyOptions& getOptions() const noexcept override;
    [[nodiscard]] bool isConfigured() const noexcept override;
    [[nodiscard]] std::string name() const override { return "DeviceAssembler"; }
    [[nodiscard]] bool supportsDG() const noexcept override;
    [[nodiscard]] bool supportsFullContext() const noexcept override { return true; }
    [[nodiscard]] bool supportsSolution() const noexcept override { return true; }
    [[nodiscard]] bool supportsSolutionHistory() const noexcept override { return true; }
    [[nodiscard]] bool supportsTimeIntegrationContext() const noexcept override { return true; }
    [[nodiscard]] bool supportsDofOffsets() const noexcept override { return true; }
    [[nodiscard]] bool supportsFieldRequirements() const noexcept override { return true; }
    [[nodiscard]] bool supportsMaterialState() const noexcept override { return true; }

    // =========================================================================
    // Device-Specific Configuration
    // =========================================================================

    /**
     * @brief Set device options
     */
    void setDeviceOptions(const DeviceOptions& options);

    /**
     * @brief Get current device options
     */
    [[nodiscard]] const DeviceOptions& getDeviceOptions() const noexcept;

    /**
     * @brief Get device information
     */
    [[nodiscard]] DeviceInfo getDeviceInfo() const;

    /**
     * @brief Get assembly statistics
     */
    [[nodiscard]] const DeviceAssemblyStats& getLastStats() const noexcept;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void initialize() override;
    void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) override;
    void reset() override;

    // =========================================================================
    // Standard Assembly (Assembler interface)
    // =========================================================================

    AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override;

    AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override;

    // =========================================================================
    // Partial Assembly Interface
    // =========================================================================

    /**
     * @brief Setup partial assembly: compute and store quadrature data
     *
     * Must be called before apply() for partial assembly mode.
     */
    void setup(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        DeviceKernel& kernel);

    /**
     * @brief Apply operator: y = A * x
     *
     * For partial assembly, uses stored quadrature data.
     *
     * @param x Input vector (host or device, depending on options)
     * @param y Output vector (host or device)
     */
    void apply(std::span<const Real> x, std::span<Real> y);

    /**
     * @brief Apply operator with device pointers
     */
    void applyDevice(const Real* x_device, Real* y_device);

    /**
     * @brief Check if setup has been performed
     */
    [[nodiscard]] bool isSetup() const noexcept;

    // =========================================================================
    // Device Memory Management
    // =========================================================================

    /**
     * @brief Allocate device vector
     */
    [[nodiscard]] DeviceMemory<Real> allocateVector(std::size_t size);

    /**
     * @brief Copy vector to device
     */
    void copyToDevice(std::span<const Real> host, DeviceMemory<Real>& device);

    /**
     * @brief Copy vector from device
     */
    void copyFromDevice(const DeviceMemory<Real>& device, std::span<Real> host);

    /**
     * @brief Synchronize device operations
     */
    void synchronize();

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Check if GPU is available
     */
    [[nodiscard]] static bool isGPUAvailable();

    /**
     * @brief Get number of available GPUs
     */
    [[nodiscard]] static int getGPUCount();

    /**
     * @brief Get string describing backend capabilities
     */
    [[nodiscard]] static std::string getBackendInfo();

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create device assembler with default options
 */
std::unique_ptr<Assembler> createDeviceAssembler();

/**
 * @brief Create device assembler with specified options
 */
std::unique_ptr<Assembler> createDeviceAssembler(const DeviceOptions& options);

/**
 * @brief Create device assembler with automatic backend detection
 */
std::unique_ptr<Assembler> createDeviceAssemblerAuto();

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_DEVICE_ASSEMBLER_H
