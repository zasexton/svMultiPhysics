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
#include "StandardAssembler.h"
#include "Constraints/AffineConstraints.h"
#include "Dofs/DofMap.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <optional>

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
class WrappedMatrixFreeKernel final : public IMatrixFreeKernel {
public:
    explicit WrappedMatrixFreeKernel(AssemblyKernel& kernel, std::shared_ptr<AssemblyKernel> owner = {})
        : kernel_(kernel), owner_(std::move(owner)),
          needs_external_state_(!kernel.fieldRequirements().empty() || kernel.hasExplicitTimeDependency() ||
                                kernel.maxTemporalDerivativeOrder() > 0 ||
                                hasFlag(kernel.getRequiredData(), RequiredData::MaterialState)) {}

    void applyLocal(const AssemblyContext& context, std::span<const Real> x,
                    std::span<Real> y) override {
        FE_THROW_IF(kernel_.isVectorOnly() || !canEvaluateCell(), NotImplementedException,
                    "WrappedMatrixFreeKernel: no cell matrix for the supplied fields and state");
        KernelOutput output;
        const auto rows = context.numTestDofs();
        const auto cols = context.numTrialDofs();
        FE_CHECK_ARG(x.size() >= static_cast<std::size_t>(cols) && y.size() >= static_cast<std::size_t>(rows),
                     "WrappedMatrixFreeKernel: local vector size does not match field block");
        output.reserve(rows, cols, true, !kernel_.isMatrixOnly());
        kernel_.computeCell(context, output);
        FE_THROW_IF(!output.has_matrix, NotImplementedException,
                    "WrappedMatrixFreeKernel: cell kernel did not provide a matrix");
        FE_CHECK_ARG(output.n_test_dofs == rows && output.n_trial_dofs == cols,
                     "WrappedMatrixFreeKernel: local matrix size does not match field block");
        for (LocalIndex i = 0; i < rows; ++i) {
            Real sum = 0.0;
            for (LocalIndex j = 0; j < cols; ++j) sum += output.matrixEntry(i, j) * x[j];
            y[i] = sum;
        }
    }

    bool supportsResidual() const noexcept override { return canEvaluateCell() && !kernel_.isMatrixOnly(); }
    void computeResidual(const AssemblyContext& context, std::span<Real> residual) override {
        FE_THROW_IF(!supportsResidual(), NotImplementedException,
                    "WrappedMatrixFreeKernel: no cell residual for the supplied fields and state");
        KernelOutput output;
        output.reserve(context.numTestDofs(), context.numTrialDofs(), !kernel_.isVectorOnly(), true);
        kernel_.computeCell(context, output);
        FE_THROW_IF(!output.has_vector, NotImplementedException,
                    "WrappedMatrixFreeKernel: cell kernel did not provide a residual");
        FE_CHECK_ARG(output.local_vector.size() == static_cast<std::size_t>(context.numTestDofs()) &&
                     residual.size() >= output.local_vector.size(),
                     "WrappedMatrixFreeKernel: local residual size does not match test field");
        std::copy(output.local_vector.begin(), output.local_vector.end(), residual.begin());
    }
    RequiredData getRequiredData() const noexcept override { return kernel_.getRequiredData(); }
    const spaces::FunctionSpace* cellRuleSpace() const noexcept override { return kernel_.cellRuleSpace(); }

private:
    bool canEvaluateCell() const noexcept { return kernel_.hasCell() && !needs_external_state_; }

    AssemblyKernel& kernel_;
    std::shared_ptr<AssemblyKernel> owner_;
    bool needs_external_state_;
};

namespace {

const constraints::AffineConstraints* closedConstraints(
    const constraints::AffineConstraints* constraints, std::optional<constraints::AffineConstraints>& copy,
    GlobalIndex size)
{
    if (!constraints) return nullptr;
    if (!constraints->isClosed()) {
        copy.emplace(*constraints);
        copy->close();
        constraints = &*copy;
    }
    constraints->forEach([&](const auto& line) {
        FE_CHECK_ARG(line.slave_dof >= 0 && line.slave_dof < size,
                     "MatrixFreeAssembler: constrained DOF outside its field map");
        for (const auto& master : line.entries) {
            FE_CHECK_ARG(master.master_dof >= 0 && master.master_dof < size,
                         "MatrixFreeAssembler: constraint master outside its field map");
        }
    });
    return constraints;
}

} // namespace

MatrixFreeAssembler::MatrixFreeAssembler() = default;
MatrixFreeAssembler::MatrixFreeAssembler(const MatrixFreeOptions& options) : options_(options) {}
MatrixFreeAssembler::~MatrixFreeAssembler() = default;
MatrixFreeAssembler::MatrixFreeAssembler(MatrixFreeAssembler&& other) noexcept = default;
MatrixFreeAssembler& MatrixFreeAssembler::operator=(MatrixFreeAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void MatrixFreeAssembler::setMesh(const IMeshAccess& mesh) {
    mesh_ = &mesh;
    invalidateSetup();
}

void MatrixFreeAssembler::setDofMap(const dofs::DofMap& dof_map) {
    setDofMaps(dof_map, dof_map);
}

void MatrixFreeAssembler::setDofMaps(const dofs::DofMap& test_map, const dofs::DofMap& trial_map) {
    test_map_ = &test_map;
    trial_map_ = &trial_map;
    invalidateSetup();
}

void MatrixFreeAssembler::setDofHandler(const dofs::DofHandler& handler) {
    orientation_handler_ = &handler;
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
    setConstraints(constraints, constraints);
}

void MatrixFreeAssembler::setConstraints(const constraints::AffineConstraints& test_constraints,
                                         const constraints::AffineConstraints& trial_constraints) {
    test_constraints_ = &test_constraints;
    trial_constraints_ = &trial_constraints;
}

void MatrixFreeAssembler::setOptions(const MatrixFreeOptions& options) {
    options_ = options;
    invalidateSetup();
}

bool MatrixFreeAssembler::isConfigured() const noexcept {
    return mesh_ != nullptr &&
           test_map_ != nullptr && trial_map_ != nullptr &&
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

    FE_THROW_IF(test_space_->space_type() == spaces::SpaceType::Mixed ||
                trial_space_->space_type() == spaces::SpaceType::Mixed, NotImplementedException,
                "MatrixFreeAssembler: supply each mixed field block with its test/trial spaces and DOF maps");
    num_rows_ = test_map_->getNumDofs();
    num_cols_ = trial_map_->getNumDofs();

    is_setup_ = false;
    owned_cells_.clear();
    owned_cells_.reserve(static_cast<std::size_t>(mesh_->numOwnedCells()));
    mesh_->forEachOwnedCell([&](GlobalIndex cell_id) {
        const auto rows = test_map_->getCellDofs(cell_id);
        const auto cols = trial_map_->getCellDofs(cell_id);
        FE_CHECK_ARG(rows.size() == test_space_->dofs_per_element(cell_id) &&
                     cols.size() == trial_space_->dofs_per_element(cell_id),
                     "MatrixFreeAssembler: cell DOF map does not match its field space");
        for (auto row : rows) FE_CHECK_ARG(row >= 0 && row < num_rows_, "Invalid matrix-free row DOF");
        for (auto col : cols) FE_CHECK_ARG(col >= 0 && col < num_cols_, "Invalid matrix-free column DOF");
        owned_cells_.push_back(cell_id);
    });

    is_setup_ = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.setup_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    stats_.cached_bytes = owned_cells_.size() * sizeof(GlobalIndex);
}

void MatrixFreeAssembler::invalidateSetup() {
    is_setup_ = false;
    owned_cells_.clear();
}

void MatrixFreeAssembler::apply(std::span<const Real> x, std::span<Real> y) {
    evaluate(Evaluation::Action, x, y, false);
}

void MatrixFreeAssembler::applyAdd(std::span<const Real> x, std::span<Real> y) {
    evaluate(Evaluation::Action, x, y, true);
}

void MatrixFreeAssembler::applyDirect(std::span<const Real> x, std::span<Real> y) {
    const bool was_setup = is_setup_;
    invalidateSetup();
    try {
        apply(x, y);
    } catch (...) {
        if (!was_setup) invalidateSetup();
        throw;
    }
    if (!was_setup) invalidateSetup();
}

void MatrixFreeAssembler::evaluate(Evaluation evaluation, std::span<const Real> x,
                                   std::span<Real> output, bool add) {
    FE_THROW_IF(!isConfigured(), "MatrixFreeAssembler not configured");
    const bool residual = evaluation == Evaluation::Residual;
    FE_THROW_IF(residual && !kernel_->supportsResidual(), NotImplementedException,
                "MatrixFreeAssembler: kernel does not provide a cell residual");
    if (!is_setup_) setup();
    FE_CHECK_ARG(static_cast<GlobalIndex>(output.size()) >= num_rows_, "Matrix-free output vector too small");
    FE_CHECK_ARG(residual || static_cast<GlobalIndex>(x.size()) >= num_cols_, "Matrix-free input vector too small");

    const auto start_time = std::chrono::steady_clock::now();
    auto required = kernel_->getRequiredData();
    if (hasFlag(required, RequiredData::SolutionHessians) || hasFlag(required, RequiredData::SolutionLaplacians)) {
        required |= RequiredData::BasisHessians;
    }
    const bool reads_state = hasFlag(required, RequiredData::SolutionCoefficients) ||
        hasFlag(required, RequiredData::SolutionValues) || hasFlag(required, RequiredData::SolutionGradients) ||
        hasFlag(required, RequiredData::SolutionHessians) || hasFlag(required, RequiredData::SolutionLaplacians);
    FE_CHECK_ARG(!reads_state || static_cast<GlobalIndex>(current_solution_.size()) >= num_cols_,
                 "MatrixFreeAssembler: current solution required for this cell kernel");

    std::optional<constraints::AffineConstraints> test_copy, trial_copy;
    const auto* test = closedConstraints(options_.apply_constraints ? test_constraints_ : nullptr, test_copy, num_rows_);
    const auto* trial = (trial_constraints_ == test_constraints_ && num_rows_ == num_cols_) ? test :
        closedConstraints(options_.apply_constraints ? trial_constraints_ : nullptr, trial_copy, num_cols_);
    std::span<const Real> state = current_solution_;
    std::vector<Real> prolonged_direction, prolonged_state;
    if (trial && !residual) {
        prolonged_direction.assign(x.begin(), x.end());
        trial->distributeHomogeneous(prolonged_direction);
        x = prolonged_direction;
    }
    if (trial && reads_state) {
        prolonged_state = current_solution_;
        trial->distribute(prolonged_state);
        state = prolonged_state;
    }

    // Accumulate a separate increment so applyAdd preserves the caller's vector,
    // including constrained entries. A failed local evaluation publishes nothing.
    std::vector<Real> result(static_cast<std::size_t>(num_rows_), 0.0);
    std::exception_ptr failure;
#ifdef _OPENMP
    const int requested_threads = options_.num_threads > 0 ? options_.num_threads : omp_get_max_threads();
    const auto num_threads = std::clamp<GlobalIndex>(mesh_->numOwnedCells(), 1, requested_threads);
#pragma omp parallel num_threads(num_threads)
#endif
    {
        StandardAssembler preparer;
        if (orientation_handler_) preparer.setDofHandler(*orientation_handler_);
        AssemblyContext context;
        KernelOutput local_output;
        std::vector<Real> local_x, local_state;
#ifdef _OPENMP
#pragma omp for
#endif
        for (std::size_t index = 0; index < owned_cells_.size(); ++index) {
            try {
                const auto cell = owned_cells_[index];
                const auto rows = test_map_->getCellDofs(cell);
                const auto cols = trial_map_->getCellDofs(cell);
                auto gather_trial = [&](std::span<const Real> global, std::vector<Real>& local) {
                    local.resize(cols.size());
                    for (std::size_t j = 0; j < cols.size(); ++j) local[j] = global[cols[j]];
                    preparer.applyVectorBasisGlobalToLocal(*mesh_, cell, *trial_space_, local);
                };
                preparer.prepareContext(context, *mesh_, cell, *test_space_, *trial_space_, required,
                                        kernel_->cellRuleSpace());
                if (reads_state) {
                    gather_trial(state, local_state);
                    context.setSolutionCoefficients(local_state);
                }
                local_output.reserve(context.numTestDofs(), context.numTrialDofs(), false, true);
                if (residual) {
                    kernel_->computeResidual(context, local_output.local_vector);
                } else {
                    gather_trial(x, local_x);
                    kernel_->applyLocal(context, local_x, local_output.local_vector);
                }
                preparer.applyVectorBasisOutputOrientation(*mesh_, cell, *test_space_, cell, *trial_space_, local_output);
                for (std::size_t i = 0; i < rows.size(); ++i) {
#ifdef _OPENMP
#pragma omp atomic update
#endif
                    result[rows[i]] += local_output.local_vector[i];
                }
            } catch (...) {
#ifdef _OPENMP
#pragma omp critical(svmp_matrix_free_failure)
#endif
                { if (!failure) failure = std::current_exception(); }
            }
        }
    }
    if (failure) std::rethrow_exception(failure);
    if (test) {
        test->forEach([&](const auto& line) {
            const Real value = result[line.slave_dof];
            for (const auto& master : line.entries) result[master.master_dof] += master.weight * value;
            result[line.slave_dof] = 0.0;
        });
    }
    if (!add) std::fill(output.begin(), output.end(), Real(0));
    for (std::size_t i = 0; i < result.size(); ++i) output[i] += result[i];

    if (!residual) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        ++stats_.num_applies;
        stats_.total_apply_seconds += elapsed;
        stats_.last_apply_seconds = elapsed;
        stats_.avg_apply_seconds = stats_.total_apply_seconds / static_cast<double>(stats_.num_applies);
    }
}

void MatrixFreeAssembler::getDiagonal(std::span<Real> diag) {
    FE_THROW_IF(!isConfigured(), "MatrixFreeAssembler not configured");

    if (!is_setup_) {
        setup();
    }

    const auto count = std::min({num_rows_, num_cols_, static_cast<GlobalIndex>(diag.size())});
    std::vector<Real> diagonal(diag.size(), 0.0);
    std::vector<Real> column(static_cast<std::size_t>(num_cols_), 0.0);
    std::vector<Real> action(static_cast<std::size_t>(num_rows_));
    for (GlobalIndex i = 0; i < count; ++i) {
        column[i] = 1.0;
        apply(column, action);
        diagonal[i] = action[i];
        column[i] = 0.0;
    }
    std::copy(diagonal.begin(), diagonal.end(), diag.begin());
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

void MatrixFreeAssembler::assembleResidual(std::span<Real> residual) {
    evaluate(Evaluation::Residual, {}, residual, false);
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
    FE_CHECK_NOT_NULL(kernel.get(), "wrapAsMatrixFreeKernel: kernel");
    auto& borrowed = *kernel;
    return std::make_unique<WrappedMatrixFreeKernel>(borrowed, std::move(kernel));
}

} // namespace assembly
} // namespace FE
} // namespace svmp
