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

#include "MatrixFreeAssembler.h"
#include "Core/FEException.h"
#include "Constraints/AffineConstraints.h"  // For full AffineConstraints definition

#include <chrono>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// Internal Operator Implementation
// ============================================================================

/**
 * @brief Internal implementation of MatrixFreeOperator
 */
class MatrixFreeOperatorImpl : public MatrixFreeOperator {
public:
    explicit MatrixFreeOperatorImpl(MatrixFreeAssembler& assembler)
        : assembler_(assembler)
    {
    }

    void apply(std::span<const Real> x, std::span<Real> y) override {
        assembler_.apply(x, y);
    }

    void applyAdd(std::span<const Real> x, std::span<Real> y) override {
        assembler_.applyAdd(x, y);
    }

    [[nodiscard]] GlobalIndex numRows() const noexcept override {
        return assembler_.numRows();
    }

    [[nodiscard]] GlobalIndex numCols() const noexcept override {
        return assembler_.numCols();
    }

    void getDiagonal(std::span<Real> diag) override {
        assembler_.getDiagonal(diag);
    }

private:
    MatrixFreeAssembler& assembler_;
};

// ============================================================================
// Kernel Wrapper for Standard AssemblyKernel
// ============================================================================

/**
 * @brief Wraps a standard AssemblyKernel as IMatrixFreeKernel
 */
class WrappedMatrixFreeKernel : public IMatrixFreeKernel {
public:
    explicit WrappedMatrixFreeKernel(AssemblyKernel& kernel)
        : kernel_(kernel)
    {
    }

    void applyLocal(
        const AssemblyContext& context,
        std::span<const Real> x_local,
        std::span<Real> y_local) override
    {
        LocalIndex n_dofs = context.numTestDofs();

        // Compute element matrix
        output_.clear();
        output_.reserve(n_dofs, n_dofs, true, false);

        // Note: This casts away const from context, which is a design limitation
        // A proper implementation would have a separate computeMatrix method
        kernel_.computeCell(const_cast<AssemblyContext&>(context), output_);

        // Apply: y = K * x
        auto n = static_cast<std::size_t>(n_dofs);
        std::fill(y_local.begin(), y_local.end(), Real(0));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                y_local[i] += output_.local_matrix[i * n + j] * x_local[j];
            }
        }
    }

    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return kernel_.getRequiredData();
    }

private:
    AssemblyKernel& kernel_;
    mutable KernelOutput output_;
};

// ============================================================================
// MatrixFreeAssembler Implementation
// ============================================================================

MatrixFreeAssembler::MatrixFreeAssembler()
    : options_{}
    , loop_(std::make_unique<AssemblyLoop>())
{
}

MatrixFreeAssembler::MatrixFreeAssembler(const MatrixFreeOptions& options)
    : options_(options)
    , loop_(std::make_unique<AssemblyLoop>())
{
}

MatrixFreeAssembler::~MatrixFreeAssembler() = default;

MatrixFreeAssembler::MatrixFreeAssembler(MatrixFreeAssembler&& other) noexcept = default;

MatrixFreeAssembler& MatrixFreeAssembler::operator=(MatrixFreeAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void MatrixFreeAssembler::setMesh(const IMeshAccess& mesh) {
    mesh_ = &mesh;
    loop_->setMesh(mesh);
    invalidateSetup();
}

void MatrixFreeAssembler::setDofMap(const dofs::DofMap& dof_map) {
    dof_map_ = &dof_map;
    loop_->setDofMap(dof_map);
    invalidateSetup();
}

void MatrixFreeAssembler::setSpace(const spaces::FunctionSpace& space) {
    test_space_ = &space;
    trial_space_ = &space;
    invalidateSetup();
}

void MatrixFreeAssembler::setSpaces(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space)
{
    test_space_ = &test_space;
    trial_space_ = &trial_space;
    invalidateSetup();
}

void MatrixFreeAssembler::setKernel(IMatrixFreeKernel& kernel) {
    kernel_ = &kernel;
    invalidateSetup();
}

void MatrixFreeAssembler::setConstraints(const constraints::AffineConstraints& constraints) {
    constraints_ = &constraints;
}

void MatrixFreeAssembler::setOptions(const MatrixFreeOptions& options) {
    options_ = options;

    LoopOptions loop_opts;
    loop_opts.num_threads = options.num_threads;
    loop_opts.batch_size = options.batch_size;
    loop_->setOptions(loop_opts);
}

bool MatrixFreeAssembler::isConfigured() const noexcept {
    return mesh_ != nullptr &&
           dof_map_ != nullptr &&
           test_space_ != nullptr &&
           trial_space_ != nullptr &&
           kernel_ != nullptr;
}

// ============================================================================
// Setup
// ============================================================================

void MatrixFreeAssembler::setup() {
    FE_THROW_IF(!isConfigured(), "MatrixFreeAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get dimensions
    // Note: This would come from DofMap in actual implementation
    num_rows_ = mesh_->numCells() * 8;  // Placeholder
    num_cols_ = num_rows_;  // Square for now

    GlobalIndex num_cells = mesh_->numCells();
    element_cache_.resize(static_cast<std::size_t>(num_cells));

    // Initialize thread-local storage
    int num_threads = options_.num_threads;
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    thread_contexts_.resize(static_cast<std::size_t>(num_threads));
    thread_x_local_.resize(static_cast<std::size_t>(num_threads));
    thread_y_local_.resize(static_cast<std::size_t>(num_threads));

    for (int i = 0; i < num_threads; ++i) {
        auto idx = static_cast<std::size_t>(i);
        thread_contexts_[idx] = std::make_unique<AssemblyContext>();
    }

    // Cache data based on assembly level
    if (options_.assembly_level >= AssemblyLevel::Partial) {
        if (options_.cache_geometry) {
            cacheGeometryData();
        }
        if (options_.cache_basis) {
            cacheBasisData();
        }
        cacheKernelData();
    }

    is_setup_ = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.setup_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    // Calculate cached memory
    stats_.cached_bytes = 0;
    for (const auto& elem : element_cache_) {
        stats_.cached_bytes += elem.dofs.size() * sizeof(GlobalIndex);
        stats_.cached_bytes += elem.jacobians.size() * sizeof(Real);
        stats_.cached_bytes += elem.basis_values.size() * sizeof(Real);
        stats_.cached_bytes += elem.basis_gradients.size() * sizeof(Real);
        stats_.cached_bytes += elem.quadrature_weights.size() * sizeof(Real);
        stats_.cached_bytes += elem.kernel_data.size() * sizeof(Real);
    }
}

void MatrixFreeAssembler::invalidateSetup() {
    is_setup_ = false;
    element_cache_.clear();
}

void MatrixFreeAssembler::cacheGeometryData() {
    // Iterate over cells and cache geometry data
    GlobalIndex cell_idx = 0;
    mesh_->forEachCell([this, &cell_idx](GlobalIndex cell_id) {
        auto idx = static_cast<std::size_t>(cell_idx);
        auto& elem = element_cache_[idx];

        elem.cell_id = cell_id;
        elem.cell_type = mesh_->getCellType(cell_id);

        // Get DOFs for this cell
        // elem.dofs would be populated from DofMap

        // Cache Jacobians at quadrature points
        // This would involve:
        // - Getting element geometry
        // - Computing Jacobian at each quadrature point
        // - Storing determinants and inverses

        // Placeholder: allocate typical sizes
        // Real implementation would compute actual values
        int n_quad = 8;  // Typical quadrature point count
        elem.det_jacobians.resize(static_cast<std::size_t>(n_quad));
        elem.quadrature_weights.resize(static_cast<std::size_t>(n_quad));

        ++cell_idx;
    });
}

void MatrixFreeAssembler::cacheBasisData() {
    // Cache basis function evaluations
    GlobalIndex cell_idx = 0;
    mesh_->forEachCell([this, &cell_idx](GlobalIndex /*cell_id*/) {
        auto idx = static_cast<std::size_t>(cell_idx);
        auto& elem = element_cache_[idx];

        // Cache basis values and gradients at quadrature points
        // This would involve:
        // - Getting element type
        // - Evaluating basis functions
        // - Transforming gradients to physical space

        // Placeholder sizes
        int n_dofs = 8;   // Typical DOFs per element
        int n_quad = 8;   // Typical quadrature points

        elem.basis_values.resize(static_cast<std::size_t>(n_dofs * n_quad));
        elem.basis_gradients.resize(static_cast<std::size_t>(n_dofs * n_quad * 3));

        ++cell_idx;
    });
}

void MatrixFreeAssembler::cacheKernelData() {
    // Let kernel cache its own data
    GlobalIndex cell_idx = 0;
    mesh_->forEachCell([this, &cell_idx](GlobalIndex /*cell_id*/) {
        auto idx = static_cast<std::size_t>(cell_idx);
        auto& elem = element_cache_[idx];

        // Would prepare context and call kernel
        // std::size_t data_size = kernel_->elementDataSize(context);
        // elem.kernel_data.resize(data_size);
        // kernel_->setupElement(context, elem.kernel_data);

        // Placeholder
        elem.kernel_data.clear();

        ++cell_idx;
    });
}

// ============================================================================
// Operator Application
// ============================================================================

void MatrixFreeAssembler::apply(std::span<const Real> x, std::span<Real> y) {
    // Zero output
    std::fill(y.begin(), y.end(), Real(0));

    // Apply and add
    applyAdd(x, y);
}

void MatrixFreeAssembler::applyAdd(std::span<const Real> x, std::span<Real> y) {
    FE_THROW_IF(!isConfigured(), "MatrixFreeAssembler not configured");

    if (!is_setup_ && options_.assembly_level != AssemblyLevel::None) {
        setup();
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Zero constrained input entries
    std::vector<Real> x_work(x.begin(), x.end());
    if (options_.apply_constraints && constraints_) {
        zeroConstrainedEntries(x_work);
    }

#ifdef _OPENMP
    int num_threads = options_.num_threads > 0 ? options_.num_threads
                                               : omp_get_max_threads();

    // Parallel element loop
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        auto& context = *thread_contexts_[static_cast<std::size_t>(tid)];
        auto& x_local = thread_x_local_[static_cast<std::size_t>(tid)];
        auto& y_local = thread_y_local_[static_cast<std::size_t>(tid)];

        #pragma omp for
        for (std::size_t i = 0; i < element_cache_.size(); ++i) {
            const auto& elem = element_cache_[i];

            // Extract local x values
            std::size_t n_dofs = elem.dofs.size();
            x_local.resize(n_dofs);
            y_local.resize(n_dofs);

            for (std::size_t j = 0; j < n_dofs; ++j) {
                x_local[j] = x_work[static_cast<std::size_t>(elem.dofs[j])];
            }

            // Prepare context (use cached data)
            // context.setFromCache(elem);

            // Apply kernel
            if (!elem.kernel_data.empty()) {
                kernel_->applyLocalCached(context, elem.kernel_data, x_local, y_local);
            } else {
                kernel_->applyLocal(context, x_local, y_local);
            }

            // Scatter to global y (needs atomic for parallel)
            for (std::size_t j = 0; j < n_dofs; ++j) {
                #pragma omp atomic
                y[static_cast<std::size_t>(elem.dofs[j])] += y_local[j];
            }
        }
    }
#else
    // Sequential
    if (thread_contexts_.empty()) {
        thread_contexts_.push_back(std::make_unique<AssemblyContext>());
        thread_x_local_.resize(1);
        thread_y_local_.resize(1);
    }

    auto& context = *thread_contexts_[0];
    auto& x_local = thread_x_local_[0];
    auto& y_local = thread_y_local_[0];

    for (const auto& elem : element_cache_) {
        std::size_t n_dofs = elem.dofs.size();
        x_local.resize(n_dofs);
        y_local.resize(n_dofs);

        for (std::size_t j = 0; j < n_dofs; ++j) {
            x_local[j] = x_work[static_cast<std::size_t>(elem.dofs[j])];
        }

        if (!elem.kernel_data.empty()) {
            kernel_->applyLocalCached(context, elem.kernel_data, x_local, y_local);
        } else {
            kernel_->applyLocal(context, x_local, y_local);
        }

        for (std::size_t j = 0; j < n_dofs; ++j) {
            y[static_cast<std::size_t>(elem.dofs[j])] += y_local[j];
        }
    }
#endif

    // Apply constraints to output
    if (options_.apply_constraints && constraints_) {
        applyConstraints(y);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double apply_time = std::chrono::duration<double>(end_time - start_time).count();

    // Update statistics
    ++stats_.num_applies;
    stats_.total_apply_seconds += apply_time;
    stats_.last_apply_seconds = apply_time;
    stats_.avg_apply_seconds = stats_.total_apply_seconds /
                               static_cast<double>(stats_.num_applies);
}

void MatrixFreeAssembler::applyDirect(std::span<const Real> x, std::span<Real> y) {
    // Force recomputation
    bool was_setup = is_setup_;
    AssemblyLevel prev_level = options_.assembly_level;

    options_.assembly_level = AssemblyLevel::None;
    is_setup_ = false;

    apply(x, y);

    options_.assembly_level = prev_level;
    is_setup_ = was_setup;
}

void MatrixFreeAssembler::getDiagonal(std::span<Real> diag) {
    FE_THROW_IF(!isConfigured(), "MatrixFreeAssembler not configured");

    std::fill(diag.begin(), diag.end(), Real(0));

    // Compute diagonal by applying to unit vectors
    // This is expensive but general

    std::vector<Real> e(static_cast<std::size_t>(num_cols_), Real(0));
    std::vector<Real> Ae(static_cast<std::size_t>(num_rows_));

    for (GlobalIndex i = 0; i < num_cols_; ++i) {
        // Set up unit vector
        if (i > 0) {
            e[static_cast<std::size_t>(i - 1)] = Real(0);
        }
        e[static_cast<std::size_t>(i)] = Real(1);

        // Apply
        apply(e, Ae);

        // Extract diagonal entry
        if (i < static_cast<GlobalIndex>(diag.size())) {
            diag[static_cast<std::size_t>(i)] = Ae[static_cast<std::size_t>(i)];
        }
    }
}

// ============================================================================
// Constraint Handling
// ============================================================================

void MatrixFreeAssembler::applyConstraints(std::span<Real> y) const {
    if (!constraints_) return;

    // Set constrained entries to zero (for homogeneous constraints)
    // or to appropriate values (for inhomogeneous)

    constraints_->forEach([&y](const constraints::AffineConstraints::ConstraintView& cv) {
        if (cv.slave_dof >= 0 &&
            static_cast<std::size_t>(cv.slave_dof) < y.size()) {
            y[static_cast<std::size_t>(cv.slave_dof)] = Real(0);
        }
    });
}

void MatrixFreeAssembler::zeroConstrainedEntries(std::span<Real> y) const {
    if (!constraints_) return;

    constraints_->forEach([&y](const constraints::AffineConstraints::ConstraintView& cv) {
        if (cv.slave_dof >= 0 &&
            static_cast<std::size_t>(cv.slave_dof) < y.size()) {
            y[static_cast<std::size_t>(cv.slave_dof)] = Real(0);
        }
    });
}

// ============================================================================
// Operator Interface
// ============================================================================

std::shared_ptr<MatrixFreeOperator> MatrixFreeAssembler::getOperator() {
    return std::make_shared<MatrixFreeOperatorImpl>(*this);
}

GlobalIndex MatrixFreeAssembler::numRows() const noexcept {
    return num_rows_;
}

GlobalIndex MatrixFreeAssembler::numCols() const noexcept {
    return num_cols_;
}

// ============================================================================
// Residual Assembly
// ============================================================================

void MatrixFreeAssembler::assembleResidual(std::span<Real> /*residual*/) {
    FE_THROW_IF(!isConfigured(), "MatrixFreeAssembler not configured");

    // Residual assembly for nonlinear problems
    // Would iterate over elements and compute local residuals

    // Placeholder implementation
}

void MatrixFreeAssembler::setCurrentSolution(std::span<const Real> solution) {
    current_solution_.assign(solution.begin(), solution.end());
}

// ============================================================================
// Statistics
// ============================================================================

void MatrixFreeAssembler::resetStats() {
    stats_ = MatrixFreeStats{};
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<MatrixFreeAssembler> createMatrixFreeAssembler(
    const MatrixFreeOptions& options)
{
    return std::make_unique<MatrixFreeAssembler>(options);
}

std::unique_ptr<IMatrixFreeKernel> wrapAsMatrixFreeKernel(
    AssemblyKernel& kernel)
{
    return std::make_unique<WrappedMatrixFreeKernel>(kernel);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
