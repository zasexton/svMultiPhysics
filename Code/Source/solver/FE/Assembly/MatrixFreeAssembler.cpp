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
#include "Dofs/DofMap.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"
#include "Quadrature/QuadratureFactory.h"
#include "Geometry/MappingFactory.h"
#include "Math/Vector.h"
#include "Basis/BasisFunction.h"

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

class OwningWrappedMatrixFreeKernel : public IMatrixFreeKernel {
public:
    explicit OwningWrappedMatrixFreeKernel(std::shared_ptr<AssemblyKernel> kernel)
        : kernel_(std::move(kernel))
    {
        FE_CHECK_NOT_NULL(kernel_.get(), "OwningWrappedMatrixFreeKernel: kernel");
    }

    void applyLocal(
        const AssemblyContext& context,
        std::span<const Real> x_local,
        std::span<Real> y_local) override
    {
        LocalIndex n_dofs = context.numTestDofs();

        output_.clear();
        output_.reserve(n_dofs, n_dofs, true, false);

        kernel_->computeCell(const_cast<AssemblyContext&>(context), output_);

        auto n = static_cast<std::size_t>(n_dofs);
        std::fill(y_local.begin(), y_local.end(), Real(0));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                y_local[i] += output_.local_matrix[i * n + j] * x_local[j];
            }
        }
    }

    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return kernel_->getRequiredData();
    }

private:
    std::shared_ptr<AssemblyKernel> kernel_;
    mutable KernelOutput output_;
};

namespace {

int defaultGeometryOrder(ElementType element_type) noexcept
{
    switch (element_type) {
        case ElementType::Line3:
        case ElementType::Triangle6:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Tetra10:
        case ElementType::Hex20:
        case ElementType::Hex27:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return 2;
        default:
            return 1;
    }
}

struct CellContextScratch {
    std::vector<std::array<Real, 3>> cell_coords;

    std::vector<AssemblyContext::Point3D> quad_points;
    std::vector<Real> quad_weights;
    std::vector<AssemblyContext::Point3D> phys_points;
    std::vector<AssemblyContext::Matrix3x3> jacobians;
    std::vector<AssemblyContext::Matrix3x3> inv_jacobians;
    std::vector<Real> jac_dets;
    std::vector<Real> integration_weights;

    std::vector<Real> test_basis_values;
    std::vector<AssemblyContext::Vector3D> test_ref_gradients;
    std::vector<AssemblyContext::Vector3D> test_phys_gradients;

    std::vector<Real> trial_basis_values;
    std::vector<AssemblyContext::Vector3D> trial_ref_gradients;
    std::vector<AssemblyContext::Vector3D> trial_phys_gradients;
};

CellContextScratch& cellScratch()
{
    static thread_local CellContextScratch scratch;
    return scratch;
}

void prepareCellContext(AssemblyContext& context,
                        const IMeshAccess& mesh,
                        GlobalIndex cell_id,
                        const spaces::FunctionSpace& test_space,
                        const spaces::FunctionSpace& trial_space,
                        RequiredData required_data)
{
    auto& scratch = cellScratch();

    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    const auto& test_element = test_space.getElement(cell_type, cell_id);
    const auto& trial_element = (&test_space == &trial_space)
                                  ? test_element
                                  : trial_space.getElement(cell_type, cell_id);

    auto quad_rule = test_element.quadrature();
    if (!quad_rule) {
        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            test_element.polynomial_order(), false);
        quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
    }

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());
    const auto n_test_dofs = static_cast<LocalIndex>(test_element.num_dofs());
    const auto n_trial_dofs = static_cast<LocalIndex>(trial_element.num_dofs());

    mesh.getCellCoordinates(cell_id, scratch.cell_coords);

    std::vector<math::Vector<Real, 3>> node_coords(scratch.cell_coords.size());
    for (std::size_t i = 0; i < scratch.cell_coords.size(); ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            scratch.cell_coords[i][0],
            scratch.cell_coords[i][1],
            scratch.cell_coords[i][2]};
    }

    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = defaultGeometryOrder(cell_type);
    map_request.use_affine = (map_request.geometry_order <= 1);
    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    scratch.quad_points.resize(n_qpts);
    scratch.quad_weights.resize(n_qpts);
    scratch.phys_points.resize(n_qpts);
    scratch.jacobians.resize(n_qpts);
    scratch.inv_jacobians.resize(n_qpts);
    scratch.jac_dets.resize(n_qpts);
    scratch.integration_weights.resize(n_qpts);

    scratch.test_basis_values.resize(static_cast<std::size_t>(n_test_dofs * n_qpts));
    scratch.test_ref_gradients.resize(static_cast<std::size_t>(n_test_dofs * n_qpts));
    scratch.test_phys_gradients.resize(static_cast<std::size_t>(n_test_dofs * n_qpts));

    const bool different_spaces = (&test_space != &trial_space);
    if (different_spaces) {
        scratch.trial_basis_values.resize(static_cast<std::size_t>(n_trial_dofs * n_qpts));
        scratch.trial_ref_gradients.resize(static_cast<std::size_t>(n_trial_dofs * n_qpts));
        scratch.trial_phys_gradients.resize(static_cast<std::size_t>(n_trial_dofs * n_qpts));
    } else {
        scratch.trial_basis_values.clear();
        scratch.trial_ref_gradients.clear();
        scratch.trial_phys_gradients.clear();
    }

    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& qpt = quad_points[q];
        scratch.quad_points[q] = {qpt[0], qpt[1], qpt[2]};
        scratch.quad_weights[q] = quad_weights[q];

        const math::Vector<Real, 3> xi{qpt[0], qpt[1], qpt[2]};
        const auto x_phys = mapping->map_to_physical(xi);
        scratch.phys_points[q] = {x_phys[0], x_phys[1], x_phys[2]};

        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch.jacobians[q][i][j] = J(i, j);
                scratch.inv_jacobians[q][i][j] = J_inv(i, j);
            }
        }
        scratch.jac_dets[q] = det_J;
        scratch.integration_weights[q] = quad_weights[q] * std::abs(det_J);
    }

    const auto& test_basis = test_element.basis();
    std::vector<Real> values_at_pt;
    std::vector<basis::Gradient> gradients_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch.quad_points[q][0],
            scratch.quad_points[q][1],
            scratch.quad_points[q][2]};

        test_basis.evaluate_values(xi, values_at_pt);
        test_basis.evaluate_gradients(xi, gradients_at_pt);

        for (LocalIndex i = 0; i < n_test_dofs; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
            const std::size_t idx_phys = static_cast<std::size_t>(q * n_test_dofs + i);
            scratch.test_basis_values[idx] = values_at_pt[i];
            scratch.test_ref_gradients[idx] = {
                gradients_at_pt[i][0],
                gradients_at_pt[i][1],
                gradients_at_pt[i][2]};

            const auto& grad_ref = scratch.test_ref_gradients[idx];
            const auto& J_inv = scratch.inv_jacobians[q];
            AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};
            for (int d1 = 0; d1 < dim; ++d1) {
                for (int d2 = 0; d2 < dim; ++d2) {
                    grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                }
            }
            scratch.test_phys_gradients[idx_phys] = grad_phys;
        }

        if (different_spaces) {
            const auto& trial_basis = trial_element.basis();
            trial_basis.evaluate_values(xi, values_at_pt);
            trial_basis.evaluate_gradients(xi, gradients_at_pt);

            for (LocalIndex j = 0; j < n_trial_dofs; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j * n_qpts + q);
                const std::size_t idx_phys = static_cast<std::size_t>(q * n_trial_dofs + j);
                scratch.trial_basis_values[idx] = values_at_pt[j];
                scratch.trial_ref_gradients[idx] = {
                    gradients_at_pt[j][0],
                    gradients_at_pt[j][1],
                    gradients_at_pt[j][2]};

                const auto& grad_ref = scratch.trial_ref_gradients[idx];
                const auto& J_inv = scratch.inv_jacobians[q];
                AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};
                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                    }
                }
                scratch.trial_phys_gradients[idx_phys] = grad_phys;
            }
        }
    }

    context.configure(cell_id, test_element, trial_element, required_data);
    context.setCellDomainId(mesh.getCellDomainId(cell_id));
    context.setQuadratureData(scratch.quad_points, scratch.quad_weights);
    context.setPhysicalPoints(scratch.phys_points);
    context.setJacobianData(scratch.jacobians, scratch.inv_jacobians, scratch.jac_dets);
    context.setIntegrationWeights(scratch.integration_weights);

    context.setTestBasisData(n_test_dofs, scratch.test_basis_values, scratch.test_ref_gradients);
    context.setPhysicalGradients(scratch.test_phys_gradients,
                                 different_spaces ? scratch.trial_phys_gradients
                                                  : scratch.test_phys_gradients);

    if (different_spaces) {
        context.setTrialBasisData(n_trial_dofs, scratch.trial_basis_values, scratch.trial_ref_gradients);
    }
}

} // namespace

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

    // Dimensions (single global DOF space)
    num_rows_ = dof_map_->getNumDofs();
    num_cols_ = num_rows_;

    // Element cache: stable slot per cell id.
    const GlobalIndex num_cells = mesh_->numCells();
    element_cache_.assign(static_cast<std::size_t>(num_cells), MatrixFreeElementData{});
    mesh_->forEachCell([&](GlobalIndex cell_id) {
        FE_THROW_IF(cell_id < 0 || cell_id >= num_cells, FEException,
                    "MatrixFreeAssembler::setup: invalid cell id");

        auto& elem = element_cache_[static_cast<std::size_t>(cell_id)];
        elem.cell_id = cell_id;
        elem.cell_type = mesh_->getCellType(cell_id);

        const auto dofs = dof_map_->getCellDofs(cell_id);
        elem.dofs.assign(dofs.begin(), dofs.end());

        // Optional kernel-specific caching (physics data, coefficients, etc.).
        elem.kernel_data.clear();
    });

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

    is_setup_ = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.setup_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    // Calculate cached memory
    stats_.cached_bytes = 0;
    for (const auto& elem : element_cache_) {
        stats_.cached_bytes += elem.dofs.size() * sizeof(GlobalIndex);
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

    if (!is_setup_) {
        setup();
    }

    FE_THROW_IF(static_cast<GlobalIndex>(x.size()) < num_cols_, FEException,
                "MatrixFreeAssembler::applyAdd: input vector too small");
    FE_THROW_IF(static_cast<GlobalIndex>(y.size()) < num_rows_, FEException,
                "MatrixFreeAssembler::applyAdd: output vector too small");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Zero constrained input entries
    std::vector<Real> x_work(x.begin(), x.end());
    if (options_.apply_constraints && constraints_) {
        zeroConstrainedEntries(x_work);
    }

    const RequiredData required_data = kernel_->getRequiredData();

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
            if (elem.cell_id < 0) {
                continue;
            }

            // Extract local x values
            std::size_t n_dofs = elem.dofs.size();
            x_local.resize(n_dofs);
            y_local.resize(n_dofs);

            for (std::size_t j = 0; j < n_dofs; ++j) {
                x_local[j] = x_work[static_cast<std::size_t>(elem.dofs[j])];
            }

            prepareCellContext(context, *mesh_, elem.cell_id,
                               *test_space_, *trial_space_, required_data);

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
        if (elem.cell_id < 0) {
            continue;
        }
        std::size_t n_dofs = elem.dofs.size();
        x_local.resize(n_dofs);
        y_local.resize(n_dofs);

        for (std::size_t j = 0; j < n_dofs; ++j) {
            x_local[j] = x_work[static_cast<std::size_t>(elem.dofs[j])];
        }

        prepareCellContext(context, *mesh_, elem.cell_id,
                           *test_space_, *trial_space_, required_data);

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

    if (!is_setup_) {
        setup();
    }

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

std::unique_ptr<IMatrixFreeKernel> wrapAsMatrixFreeKernel(
    std::shared_ptr<AssemblyKernel> kernel)
{
    return std::make_unique<OwningWrappedMatrixFreeKernel>(std::move(kernel));
}

} // namespace assembly
} // namespace FE
} // namespace svmp
