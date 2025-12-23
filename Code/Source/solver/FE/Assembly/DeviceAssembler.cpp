/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DeviceAssembler.h"
#include "StandardAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Constraints/AffineConstraints.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// DeviceMemory Implementation (CPU fallback)
// ============================================================================

template<typename T>
DeviceMemory<T>::~DeviceMemory()
{
    free();
}

template<typename T>
DeviceMemory<T>::DeviceMemory(DeviceMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), backend_(other.backend_)
{
    other.ptr_ = nullptr;
    other.size_ = 0;
}

template<typename T>
DeviceMemory<T>& DeviceMemory<T>::operator=(DeviceMemory&& other) noexcept
{
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        size_ = other.size_;
        backend_ = other.backend_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template<typename T>
void DeviceMemory<T>::allocate(std::size_t count)
{
    free();
    if (count > 0) {
        // CPU fallback: regular allocation
        ptr_ = new T[count];
        size_ = count;
        backend_ = DeviceBackend::CPU;
    }
}

template<typename T>
void DeviceMemory<T>::free()
{
    if (ptr_) {
        delete[] ptr_;
        ptr_ = nullptr;
        size_ = 0;
    }
}

template<typename T>
void DeviceMemory<T>::copyFromHost(const T* host_ptr, std::size_t count)
{
    if (count > size_) {
        throw std::out_of_range("DeviceMemory::copyFromHost: count exceeds allocation");
    }
    std::memcpy(ptr_, host_ptr, count * sizeof(T));
}

template<typename T>
void DeviceMemory<T>::copyToHost(T* host_ptr, std::size_t count) const
{
    if (count > size_) {
        throw std::out_of_range("DeviceMemory::copyToHost: count exceeds allocation");
    }
    std::memcpy(host_ptr, ptr_, count * sizeof(T));
}

// Explicit instantiation
template class DeviceMemory<Real>;
template class DeviceMemory<char>;
template class DeviceMemory<int>;
template class DeviceMemory<GlobalIndex>;

// ============================================================================
// PartialAssemblyData Implementation
// ============================================================================

PartialAssemblyData::PartialAssemblyData() = default;
PartialAssemblyData::~PartialAssemblyData() = default;
PartialAssemblyData::PartialAssemblyData(PartialAssemblyData&& other) noexcept = default;
PartialAssemblyData& PartialAssemblyData::operator=(PartialAssemblyData&& other) noexcept = default;

void PartialAssemblyData::allocate(std::size_t num_elements, std::size_t num_qpts,
                                   std::size_t data_per_qpt)
{
    num_elements_ = num_elements;
    num_qpts_ = num_qpts;
    total_bytes_ = num_elements * num_qpts * data_per_qpt;

    device_storage_.allocate(total_bytes_);
    populated_ = false;
}

void PartialAssemblyData::free()
{
    device_storage_.free();
    num_elements_ = 0;
    num_qpts_ = 0;
    total_bytes_ = 0;
    populated_ = false;
}

void* PartialAssemblyData::deviceData() noexcept
{
    return device_storage_.data();
}

const void* PartialAssemblyData::deviceData() const noexcept
{
    return device_storage_.data();
}

// ============================================================================
// DeviceAssembler Implementation
// ============================================================================

struct DeviceAssembler::Impl {
    DeviceOptions device_options;
    AssemblyOptions options;
    std::unique_ptr<StandardAssembler> cpu_assembler;

    const dofs::DofMap* dof_map{nullptr};
    const dofs::DofHandler* dof_handler{nullptr};
    const constraints::AffineConstraints* constraints{nullptr};
    const sparsity::SparsityPattern* sparsity{nullptr};

    // Partial assembly data
    PartialAssemblyData partial_data;
    DeviceKernel* active_kernel{nullptr};

    // Device vectors for apply
    DeviceMemory<Real> x_device;
    DeviceMemory<Real> y_device;
    std::size_t num_dofs{0};

    // Statistics
    DeviceAssemblyStats last_stats;

    bool initialized{false};
    bool setup_complete{false};

    Impl() : cpu_assembler(std::make_unique<StandardAssembler>()) {}
};

DeviceAssembler::DeviceAssembler()
    : impl_(std::make_unique<Impl>())
{
}

DeviceAssembler::DeviceAssembler(const DeviceOptions& options)
    : impl_(std::make_unique<Impl>())
{
    impl_->device_options = options;
}

DeviceAssembler::~DeviceAssembler() = default;

DeviceAssembler::DeviceAssembler(DeviceAssembler&& other) noexcept = default;
DeviceAssembler& DeviceAssembler::operator=(DeviceAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void DeviceAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    impl_->dof_map = &dof_map;
    impl_->cpu_assembler->setDofMap(dof_map);
}

void DeviceAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    impl_->dof_handler = &dof_handler;
    impl_->dof_map = &dof_handler.getDofMap();
    impl_->cpu_assembler->setDofHandler(dof_handler);
}

void DeviceAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    impl_->constraints = constraints;
    impl_->cpu_assembler->setConstraints(constraints);
}

void DeviceAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    impl_->sparsity = sparsity;
    impl_->cpu_assembler->setSparsityPattern(sparsity);
}

void DeviceAssembler::setOptions(const AssemblyOptions& options)
{
    impl_->options = options;
    impl_->cpu_assembler->setOptions(options);
}

const AssemblyOptions& DeviceAssembler::getOptions() const noexcept
{
    return impl_->options;
}

bool DeviceAssembler::isConfigured() const noexcept
{
    return impl_->dof_map != nullptr;
}

void DeviceAssembler::setDeviceOptions(const DeviceOptions& options)
{
    impl_->device_options = options;
}

const DeviceOptions& DeviceAssembler::getDeviceOptions() const noexcept
{
    return impl_->device_options;
}

DeviceInfo DeviceAssembler::getDeviceInfo() const
{
    DeviceInfo info;

    // CPU fallback info
    info.name = "CPU (Fallback)";
    info.driver_version = "N/A";
    info.supports_unified_memory = true;

    // In a full implementation with CUDA/HIP, would query device properties

    return info;
}

const DeviceAssemblyStats& DeviceAssembler::getLastStats() const noexcept
{
    return impl_->last_stats;
}

// ============================================================================
// Lifecycle
// ============================================================================

void DeviceAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("DeviceAssembler::initialize: not configured");
    }

    impl_->cpu_assembler->initialize();

    // Initialize device resources
    impl_->num_dofs = static_cast<std::size_t>(impl_->dof_map->getNumLocalDofs());

    impl_->initialized = true;
}

void DeviceAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    impl_->cpu_assembler->finalize(matrix_view, vector_view);
}

void DeviceAssembler::reset()
{
    impl_->cpu_assembler->reset();
    impl_->partial_data.free();
    impl_->x_device.free();
    impl_->y_device.free();
    impl_->active_kernel = nullptr;
    impl_->setup_complete = false;
    impl_->initialized = false;
    impl_->last_stats = DeviceAssemblyStats{};
}

// ============================================================================
// Standard Assembly
// ============================================================================

AssemblyResult DeviceAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    auto start = std::chrono::steady_clock::now();

    AssemblyResult result;

    // For CPU fallback or Full assembly level, delegate to CPU assembler
    if (impl_->device_options.backend == DeviceBackend::CPU ||
        impl_->device_options.assembly_level == DeviceAssemblyLevel::Full) {
        result = impl_->cpu_assembler->assembleMatrix(mesh, test_space, trial_space,
                                                      kernel, matrix_view);
    } else {
        // For other levels, still use CPU for now
        // Real implementation would use device kernels
        result = impl_->cpu_assembler->assembleMatrix(mesh, test_space, trial_space,
                                                      kernel, matrix_view);
    }

    auto end = std::chrono::steady_clock::now();
    impl_->last_stats.total_seconds = std::chrono::duration<double>(end - start).count();
    impl_->last_stats.elements_processed = result.elements_assembled;

    return result;
}

AssemblyResult DeviceAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return impl_->cpu_assembler->assembleVector(mesh, space, kernel, vector_view);
}

AssemblyResult DeviceAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    return impl_->cpu_assembler->assembleBoth(mesh, test_space, trial_space,
                                              kernel, matrix_view, vector_view);
}

AssemblyResult DeviceAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    return impl_->cpu_assembler->assembleBoundaryFaces(mesh, boundary_marker, space,
                                                       kernel, matrix_view, vector_view);
}

AssemblyResult DeviceAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    return impl_->cpu_assembler->assembleInteriorFaces(mesh, test_space, trial_space,
                                                       kernel, matrix_view, vector_view);
}

// ============================================================================
// Partial Assembly Interface
// ============================================================================

void DeviceAssembler::setup(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& /*space*/,
    DeviceKernel& kernel)
{
    auto start = std::chrono::steady_clock::now();

    if (!impl_->initialized) {
        initialize();
    }

    const std::size_t num_elements = static_cast<std::size_t>(mesh.numCells());
    const std::size_t num_qpts = 27;  // Typical for Q2
    const std::size_t data_per_qpt = kernel.qptDataSize();

    // Allocate partial assembly storage
    impl_->partial_data.allocate(num_elements, num_qpts, data_per_qpt);

    // Allocate device vectors
    impl_->x_device.allocate(impl_->num_dofs);
    impl_->y_device.allocate(impl_->num_dofs);

    // Call kernel setup
    // In a real implementation, would prepare element data and call device kernel
    kernel.setup(num_elements, nullptr, impl_->partial_data.deviceData());

    impl_->partial_data.setPopulated(true);
    impl_->active_kernel = &kernel;
    impl_->setup_complete = true;

    auto end = std::chrono::steady_clock::now();
    impl_->last_stats.setup_seconds = std::chrono::duration<double>(end - start).count();
}

void DeviceAssembler::apply(std::span<const Real> x, std::span<Real> y)
{
    if (!impl_->setup_complete) {
        throw std::runtime_error("DeviceAssembler::apply: setup not called");
    }

    if (x.size() != impl_->num_dofs || y.size() != impl_->num_dofs) {
        throw std::invalid_argument("DeviceAssembler::apply: vector size mismatch");
    }

    auto transfer_start = std::chrono::steady_clock::now();

    // Copy input to device
    impl_->x_device.copyFromHost(x.data(), x.size());
    impl_->last_stats.host_to_device_bytes += x.size() * sizeof(Real);

    auto kernel_start = std::chrono::steady_clock::now();

    // Apply kernel
    applyDevice(impl_->x_device.data(), impl_->y_device.data());

    auto kernel_end = std::chrono::steady_clock::now();

    // Copy output from device
    impl_->y_device.copyToHost(y.data(), y.size());
    impl_->last_stats.device_to_host_bytes += y.size() * sizeof(Real);

    auto transfer_end = std::chrono::steady_clock::now();

    impl_->last_stats.kernel_seconds +=
        std::chrono::duration<double>(kernel_end - kernel_start).count();
    impl_->last_stats.transfer_seconds +=
        std::chrono::duration<double>(kernel_start - transfer_start).count() +
        std::chrono::duration<double>(transfer_end - kernel_end).count();
}

void DeviceAssembler::applyDevice(const Real* x_device, Real* y_device)
{
    if (!impl_->setup_complete || !impl_->active_kernel) {
        throw std::runtime_error("DeviceAssembler::applyDevice: not setup");
    }

    const std::size_t num_elements = impl_->partial_data.numElements();

    impl_->active_kernel->apply(
        num_elements,
        impl_->partial_data.deviceData(),
        x_device,
        y_device);
}

bool DeviceAssembler::isSetup() const noexcept
{
    return impl_->setup_complete;
}

// ============================================================================
// Device Memory Management
// ============================================================================

DeviceMemory<Real> DeviceAssembler::allocateVector(std::size_t size)
{
    DeviceMemory<Real> mem;
    mem.allocate(size);
    return mem;
}

void DeviceAssembler::copyToDevice(std::span<const Real> host, DeviceMemory<Real>& device)
{
    device.copyFromHost(host.data(), host.size());
}

void DeviceAssembler::copyFromDevice(const DeviceMemory<Real>& device, std::span<Real> host)
{
    device.copyToHost(host.data(), host.size());
}

void DeviceAssembler::synchronize()
{
    // CPU fallback: no-op
    // CUDA: cudaDeviceSynchronize()
    // HIP: hipDeviceSynchronize()
}

// ============================================================================
// Static Query Methods
// ============================================================================

bool DeviceAssembler::isGPUAvailable()
{
    // In a full implementation, would check for CUDA/HIP devices
    // For now, return false (CPU fallback only)
    return false;
}

int DeviceAssembler::getGPUCount()
{
    // In a full implementation, would query device count
    return 0;
}

std::string DeviceAssembler::getBackendInfo()
{
    std::ostringstream oss;

    oss << "DeviceAssembler Backend Information:\n";
    oss << "  Compiled backends:\n";
    oss << "    CPU: yes (fallback)\n";

#ifdef SVMP_HAS_CUDA
    oss << "    CUDA: yes\n";
#else
    oss << "    CUDA: no\n";
#endif

#ifdef SVMP_HAS_HIP
    oss << "    HIP: yes\n";
#else
    oss << "    HIP: no\n";
#endif

#ifdef SVMP_HAS_SYCL
    oss << "    SYCL: yes\n";
#else
    oss << "    SYCL: no\n";
#endif

    oss << "  Available GPUs: " << getGPUCount() << "\n";

    return oss.str();
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Assembler> createDeviceAssembler()
{
    return std::make_unique<DeviceAssembler>();
}

std::unique_ptr<Assembler> createDeviceAssembler(const DeviceOptions& options)
{
    return std::make_unique<DeviceAssembler>(options);
}

std::unique_ptr<Assembler> createDeviceAssemblerAuto()
{
    DeviceOptions options;
    options.backend = DeviceBackend::Auto;

    // Auto-detect best backend
    if (DeviceAssembler::isGPUAvailable()) {
#ifdef SVMP_HAS_CUDA
        options.backend = DeviceBackend::CUDA;
#elif defined(SVMP_HAS_HIP)
        options.backend = DeviceBackend::HIP;
#else
        options.backend = DeviceBackend::CPU;
#endif
    } else {
        options.backend = DeviceBackend::CPU;
    }

    return std::make_unique<DeviceAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
