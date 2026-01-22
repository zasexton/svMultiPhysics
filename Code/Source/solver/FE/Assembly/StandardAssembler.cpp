/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "StandardAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintDistributor.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"
#include "Elements/ElementTransform.h"
#include "Elements/ReferenceElement.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/QuadratureRule.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/GeometryMapping.h"
#include "Basis/BasisFunction.h"
#include "Basis/VectorBasis.h"
#include "Math/Vector.h"
#include "Math/Matrix.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/InterfaceMesh.h"
#endif

#include <chrono>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {

namespace {

[[nodiscard]] bool invertDenseMatrix(std::span<const Real> A,
                                     LocalIndex n,
                                     std::vector<Real>& inv,
                                     std::vector<Real>& work)
{
    FE_CHECK_ARG(n >= 0, "invertDenseMatrix: negative size");
    const auto nn = static_cast<std::size_t>(n);
    FE_THROW_IF(A.size() != nn * nn, FEException, "invertDenseMatrix: size mismatch");

    inv.assign(nn * nn, 0.0);
    work.assign(A.begin(), A.end());
    for (LocalIndex i = 0; i < n; ++i) {
        inv[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(i)] = 1.0;
    }

    constexpr Real eps = Real(1e-14);

    for (LocalIndex col = 0; col < n; ++col) {
        LocalIndex pivot_row = col;
        Real pivot_val = std::abs(work[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(col)]);
        for (LocalIndex r = col + 1; r < n; ++r) {
            const Real v = std::abs(work[static_cast<std::size_t>(r) * nn + static_cast<std::size_t>(col)]);
            if (v > pivot_val) {
                pivot_val = v;
                pivot_row = r;
            }
        }

        if (pivot_val < eps) {
            return false;
        }

        if (pivot_row != col) {
            for (LocalIndex c = 0; c < n; ++c) {
                std::swap(work[static_cast<std::size_t>(pivot_row) * nn + static_cast<std::size_t>(c)],
                          work[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(c)]);
                std::swap(inv[static_cast<std::size_t>(pivot_row) * nn + static_cast<std::size_t>(c)],
                          inv[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(c)]);
            }
        }

        const Real diag = work[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(col)];
        if (std::abs(diag) < eps) {
            return false;
        }
        const Real inv_diag = Real(1) / diag;

        for (LocalIndex c = 0; c < n; ++c) {
            work[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(c)] *= inv_diag;
            inv[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(c)] *= inv_diag;
        }

        for (LocalIndex r = 0; r < n; ++r) {
            if (r == col) {
                continue;
            }
            const Real factor = work[static_cast<std::size_t>(r) * nn + static_cast<std::size_t>(col)];
            if (factor == 0.0) {
                continue;
            }
            for (LocalIndex c = 0; c < n; ++c) {
                work[static_cast<std::size_t>(r) * nn + static_cast<std::size_t>(c)] -=
                    factor * work[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(c)];
                inv[static_cast<std::size_t>(r) * nn + static_cast<std::size_t>(c)] -=
                    factor * inv[static_cast<std::size_t>(col) * nn + static_cast<std::size_t>(c)];
            }
        }
    }

    return true;
}

int requiredHistoryStates(const TimeIntegrationContext* ctx) noexcept
{
    if (ctx == nullptr) {
        return 0;
    }
    int required = 0;
    if (ctx->dt1) {
        required = std::max(required, ctx->dt1->requiredHistoryStates());
    }
    if (ctx->dt2) {
        required = std::max(required, ctx->dt2->requiredHistoryStates());
    }
    return required;
}

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

class OwnedRowOnlyView final : public GlobalSystemView {
public:
    OwnedRowOnlyView(GlobalSystemView& base,
                     const dofs::DofMap& row_map,
                     GlobalIndex row_offset)
        : base_(&base)
        , row_map_(&row_map)
        , row_offset_(row_offset)
    {
    }

    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode) override
    {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        const std::size_t n_rows = row_dofs.size();
        const std::size_t n_cols = col_dofs.size();
        FE_THROW_IF(local_matrix.size() != n_rows * n_cols, FEException,
                    "OwnedRowOnlyView::addMatrixEntries: local_matrix size mismatch");

        bool all_owned = true;
        for (const auto row : row_dofs) {
            if (!isOwnedRow(row)) {
                all_owned = false;
                break;
            }
        }
        if (all_owned) {
            base_->addMatrixEntries(row_dofs, col_dofs, local_matrix, mode);
            return;
        }

        owned_rows_.clear();
        owned_values_.clear();
        owned_rows_.reserve(n_rows);
        owned_values_.reserve(local_matrix.size());

        for (std::size_t i = 0; i < n_rows; ++i) {
            const auto row = row_dofs[i];
            if (!isOwnedRow(row)) {
                continue;
            }
            owned_rows_.push_back(row);
            const std::size_t base_idx = i * n_cols;
            owned_values_.insert(owned_values_.end(),
                                 local_matrix.begin() + base_idx,
                                 local_matrix.begin() + base_idx + n_cols);
        }

        if (!owned_rows_.empty()) {
            base_->addMatrixEntries(std::span<const GlobalIndex>(owned_rows_),
                                    col_dofs,
                                    std::span<const Real>(owned_values_),
                                    mode);
        }
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, AddMode mode) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        if (!isOwnedRow(row)) {
            return;
        }
        base_->addMatrixEntry(row, col, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        FE_THROW_IF(dofs.size() != values.size(), FEException,
                    "OwnedRowOnlyView::setDiagonal: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            setDiagonal(dofs[i], values[i]);
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        if (!isOwnedRow(dof)) {
            return;
        }
        base_->setDiagonal(dof, value);
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        owned_rows_.clear();
        owned_rows_.reserve(rows.size());
        for (const auto row : rows) {
            if (!isOwnedRow(row)) {
                continue;
            }
            owned_rows_.push_back(row);
        }
        if (!owned_rows_.empty()) {
            base_->zeroRows(std::span<const GlobalIndex>(owned_rows_), set_diagonal);
        }
    }

    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          AddMode mode) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        FE_THROW_IF(dofs.size() != local_vector.size(), FEException,
                    "OwnedRowOnlyView::addVectorEntries: size mismatch");

        bool all_owned = true;
        for (const auto dof : dofs) {
            if (!isOwnedRow(dof)) {
                all_owned = false;
                break;
            }
        }
        if (all_owned) {
            base_->addVectorEntries(dofs, local_vector, mode);
            return;
        }

        owned_rows_.clear();
        owned_vector_.clear();
        owned_rows_.reserve(dofs.size());
        owned_vector_.reserve(local_vector.size());

        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const auto dof = dofs[i];
            if (!isOwnedRow(dof)) {
                continue;
            }
            owned_rows_.push_back(dof);
            owned_vector_.push_back(local_vector[i]);
        }

        if (!owned_rows_.empty()) {
            base_->addVectorEntries(std::span<const GlobalIndex>(owned_rows_),
                                    std::span<const Real>(owned_vector_),
                                    mode);
        }
    }

    void addVectorEntry(GlobalIndex dof, Real value, AddMode mode) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        if (!isOwnedRow(dof)) {
            return;
        }
        base_->addVectorEntry(dof, value, mode);
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        FE_THROW_IF(dofs.size() != values.size(), FEException,
                    "OwnedRowOnlyView::setVectorEntries: size mismatch");
        owned_rows_.clear();
        owned_vector_.clear();
        owned_rows_.reserve(dofs.size());
        owned_vector_.reserve(values.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const auto dof = dofs[i];
            if (!isOwnedRow(dof)) {
                continue;
            }
            owned_rows_.push_back(dof);
            owned_vector_.push_back(values[i]);
        }
        if (!owned_rows_.empty()) {
            base_->setVectorEntries(std::span<const GlobalIndex>(owned_rows_),
                                    std::span<const Real>(owned_vector_));
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        owned_rows_.clear();
        owned_rows_.reserve(dofs.size());
        for (const auto dof : dofs) {
            if (!isOwnedRow(dof)) {
                continue;
            }
            owned_rows_.push_back(dof);
        }
        if (!owned_rows_.empty()) {
            base_->zeroVectorEntries(std::span<const GlobalIndex>(owned_rows_));
        }
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        return base_->getVectorEntry(dof);
    }

    void beginAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        base_->beginAssemblyPhase();
    }

    void endAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        base_->endAssemblyPhase();
    }

    void finalizeAssembly() override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        base_->finalizeAssembly();
    }

    [[nodiscard]] AssemblyPhase getPhase() const noexcept override
    {
        return base_ ? base_->getPhase() : AssemblyPhase::NotStarted;
    }

    [[nodiscard]] bool hasMatrix() const noexcept override
    {
        return base_ ? base_->hasMatrix() : false;
    }

    [[nodiscard]] bool hasVector() const noexcept override
    {
        return base_ ? base_->hasVector() : false;
    }

    [[nodiscard]] GlobalIndex numRows() const noexcept override
    {
        return base_ ? base_->numRows() : 0;
    }

    [[nodiscard]] GlobalIndex numCols() const noexcept override
    {
        return base_ ? base_->numCols() : 0;
    }

    [[nodiscard]] GlobalIndex numLocalRows() const noexcept override
    {
        return base_ ? base_->numLocalRows() : 0;
    }

    [[nodiscard]] bool isDistributed() const noexcept override
    {
        return base_ ? base_->isDistributed() : false;
    }

    [[nodiscard]] std::string backendName() const override
    {
        return base_ ? base_->backendName() : std::string{};
    }

    void zero() override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        base_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        return base_->getMatrixEntry(row, col);
    }

private:
    [[nodiscard]] bool isOwnedRow(GlobalIndex global_row) const noexcept
    {
        if (!row_map_) {
            return true;
        }
        const GlobalIndex local = global_row - row_offset_;
        if (local < 0 || local >= row_map_->getNumDofs()) {
            return false;
        }
        return row_map_->isOwnedDof(local);
    }

    GlobalSystemView* base_{nullptr};
    const dofs::DofMap* row_map_{nullptr};
    GlobalIndex row_offset_{0};

    std::vector<GlobalIndex> owned_rows_;
    std::vector<Real> owned_values_;
    std::vector<Real> owned_vector_;
};

} // namespace

// ============================================================================
// Construction
// ============================================================================

StandardAssembler::StandardAssembler() = default;

StandardAssembler::StandardAssembler(const AssemblyOptions& options)
    : options_(options)
{
}

StandardAssembler::~StandardAssembler() = default;

StandardAssembler::StandardAssembler(StandardAssembler&& other) noexcept = default;

StandardAssembler& StandardAssembler::operator=(StandardAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void StandardAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    row_dof_map_ = &dof_map;
    col_dof_map_ = &dof_map;
    row_dof_offset_ = 0;
    col_dof_offset_ = 0;
}

void StandardAssembler::setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset)
{
    row_dof_map_ = &dof_map;
    row_dof_offset_ = row_offset;
}

void StandardAssembler::setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset)
{
    col_dof_map_ = &dof_map;
    col_dof_offset_ = col_offset;
}

void StandardAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    dof_handler_ = &dof_handler;
    row_dof_map_ = &dof_handler.getDofMap();
    col_dof_map_ = row_dof_map_;
    row_dof_offset_ = 0;
    col_dof_offset_ = 0;
}

void StandardAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    constraints_ = constraints;

    if (constraints_ && constraints_->isClosed()) {
        constraint_distributor_ = std::make_unique<constraints::ConstraintDistributor>(*constraints_);
    } else {
        constraint_distributor_.reset();
    }
}

void StandardAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    sparsity_ = sparsity;
}

void StandardAssembler::setOptions(const AssemblyOptions& options)
{
    options_ = options;
}

void StandardAssembler::setCurrentSolution(std::span<const Real> solution)
{
    current_solution_ = solution;
    current_solution_view_ = nullptr;
}

void StandardAssembler::setCurrentSolutionView(const GlobalSystemView* solution_view)
{
    current_solution_view_ = solution_view;
}

void StandardAssembler::setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields)
{
    field_solution_access_.assign(fields.begin(), fields.end());
}

void StandardAssembler::setPreviousSolution(std::span<const Real> solution)
{
    setPreviousSolutionK(1, solution);
}

void StandardAssembler::setPreviousSolution2(std::span<const Real> solution)
{
    setPreviousSolutionK(2, solution);
}

void StandardAssembler::setPreviousSolutionView(const GlobalSystemView* solution_view)
{
    setPreviousSolutionViewK(1, solution_view);
}

void StandardAssembler::setPreviousSolution2View(const GlobalSystemView* solution_view)
{
    setPreviousSolutionViewK(2, solution_view);
}

void StandardAssembler::setPreviousSolutionK(int k, std::span<const Real> solution)
{
    FE_THROW_IF(k <= 0, FEException, "StandardAssembler::setPreviousSolutionK: k must be >= 1");
    if (previous_solutions_.size() < static_cast<std::size_t>(k)) {
        previous_solutions_.resize(static_cast<std::size_t>(k));
    }
    previous_solutions_[static_cast<std::size_t>(k - 1)] = solution;

    if (previous_solution_views_.size() < static_cast<std::size_t>(k)) {
        previous_solution_views_.resize(static_cast<std::size_t>(k), nullptr);
    }
    previous_solution_views_[static_cast<std::size_t>(k - 1)] = nullptr;
}

void StandardAssembler::setPreviousSolutionViewK(int k, const GlobalSystemView* solution_view)
{
    FE_THROW_IF(k <= 0, FEException, "StandardAssembler::setPreviousSolutionViewK: k must be >= 1");
    if (previous_solution_views_.size() < static_cast<std::size_t>(k)) {
        previous_solution_views_.resize(static_cast<std::size_t>(k), nullptr);
    }
    previous_solution_views_[static_cast<std::size_t>(k - 1)] = solution_view;

    if (previous_solutions_.size() < static_cast<std::size_t>(k)) {
        previous_solutions_.resize(static_cast<std::size_t>(k));
    }
}

void StandardAssembler::setTimeIntegrationContext(const TimeIntegrationContext* ctx)
{
    time_integration_ = ctx;
}

void StandardAssembler::setTime(Real time)
{
    time_ = time;
}

void StandardAssembler::setTimeStep(Real dt)
{
    dt_ = dt;
}

void StandardAssembler::setRealParameterGetter(
    const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept
{
    get_real_param_ = get_real_param;
}

void StandardAssembler::setParameterGetter(
    const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept
{
    get_param_ = get_param;
}

void StandardAssembler::setUserData(const void* user_data) noexcept
{
    user_data_ = user_data;
}

void StandardAssembler::setJITConstants(std::span<const Real> constants) noexcept
{
    jit_constants_ = constants;
}

void StandardAssembler::setCoupledValues(std::span<const Real> integrals,
                                        std::span<const Real> aux_state) noexcept
{
    coupled_integrals_ = integrals;
    coupled_aux_state_ = aux_state;
}

void StandardAssembler::setMaterialStateProvider(IMaterialStateProvider* provider) noexcept
{
    material_state_provider_ = provider;
}

const AssemblyOptions& StandardAssembler::getOptions() const noexcept
{
    return options_;
}

bool StandardAssembler::isConfigured() const noexcept
{
    return row_dof_map_ != nullptr;
}

// ============================================================================
// Lifecycle
// ============================================================================

void StandardAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("StandardAssembler::initialize: assembler not configured");
    }

    // Reserve working storage based on DOF map
    const auto max_row_dofs = row_dof_map_->getMaxDofsPerCell();
    const auto max_col_dofs = col_dof_map_ ? col_dof_map_->getMaxDofsPerCell() : max_row_dofs;
    const auto max_dofs = std::max(max_row_dofs, max_col_dofs);
    const auto max_dofs_size = static_cast<std::size_t>(max_dofs);

    row_dofs_.reserve(max_dofs_size);
    col_dofs_.reserve(max_dofs_size);
    scratch_rows_.reserve(max_dofs_size);
    scratch_cols_.reserve(max_dofs_size);
    scratch_matrix_.reserve(max_dofs_size * max_dofs_size);
    scratch_vector_.reserve(max_dofs_size);

    // Reserve context storage (estimate quadrature points)
    const LocalIndex est_qpts = 27;  // Typical for 3D Q2
    // Dimension is mesh-dependent; this is overwritten in the assembly entrypoints
    // using IMeshAccess::dimension().
    context_.reserve(max_dofs, est_qpts, 3);

    initialized_ = true;
}

void StandardAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    // End assembly phase and trigger finalization
    if (matrix_view) {
        matrix_view->endAssemblyPhase();
        matrix_view->finalizeAssembly();
    }

    if (vector_view && vector_view != matrix_view) {
        vector_view->endAssemblyPhase();
        vector_view->finalizeAssembly();
    }
}

void StandardAssembler::reset()
{
    context_.clear();
    row_dofs_.clear();
    col_dofs_.clear();
    current_solution_ = {};
    previous_solutions_.clear();
    local_solution_coeffs_.clear();
    local_prev_solution_coeffs_.clear();
    field_solution_access_.clear();
    time_integration_ = nullptr;
    initialized_ = false;
}

// ============================================================================
// Matrix Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, nullptr, true, false);
}

// ============================================================================
// Vector Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return assembleCellsCore(mesh, space, space, kernel,
                             nullptr, &vector_view, false, true);
}

// ============================================================================
// Combined Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, &vector_view, true, true);
}

// ============================================================================
// Face Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    return assembleBoundaryFaces(mesh, boundary_marker, space, space, kernel, matrix_view, vector_view);
}

AssemblyResult StandardAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (!kernel.hasBoundaryFace()) {
        return result;  // Nothing to do
    }

    // Begin assembly phase
    if (matrix_view) matrix_view->beginAssemblyPhase();
    if (vector_view && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();
    const auto field_requirements = kernel.fieldRequirements();
    const bool need_field_solutions = !field_requirements.empty();
    const bool need_solution =
        hasFlag(required_data, RequiredData::SolutionCoefficients) ||
        hasFlag(required_data, RequiredData::SolutionValues) ||
        hasFlag(required_data, RequiredData::SolutionGradients) ||
        hasFlag(required_data, RequiredData::SolutionHessians) ||
        hasFlag(required_data, RequiredData::SolutionLaplacians);
    const bool need_material_state =
        hasFlag(required_data, RequiredData::MaterialState);
    const auto material_state_spec = kernel.materialStateSpec();

    if (need_material_state) {
        FE_THROW_IF(material_state_provider_ == nullptr, FEException,
                    "StandardAssembler::assembleBoundaryFaces: kernel requires material state but no material state provider was set");
        FE_THROW_IF(material_state_spec.bytes_per_qpt == 0u, FEException,
                    "StandardAssembler::assembleBoundaryFaces: kernel requires material state but materialStateSpec().bytes_per_qpt == 0");
    }

    FE_CHECK_NOT_NULL(row_dof_map_, "StandardAssembler::assembleBoundaryFaces: row_dof_map");
    if (!col_dof_map_) {
        col_dof_map_ = row_dof_map_;
        col_dof_offset_ = row_dof_offset_;
    }

    // Ensure AssemblyContext uses the mesh spatial dimension (2D/3D).
    context_.reserve(std::max(row_dof_map_->getMaxDofsPerCell(),
                              col_dof_map_->getMaxDofsPerCell()),
                     /*max_qpts=*/27, mesh.dimension());

    const bool owned_rows_only = (options_.ghost_policy == GhostPolicy::OwnedRowsOnly);
    std::optional<OwnedRowOnlyView> owned_row_matrix;
    std::optional<OwnedRowOnlyView> owned_row_vector;
    GlobalSystemView* insert_matrix_view = matrix_view;
    GlobalSystemView* insert_vector_view = vector_view;
    if (owned_rows_only) {
        if (matrix_view != nullptr) {
            owned_row_matrix.emplace(*matrix_view, *row_dof_map_, row_dof_offset_);
            insert_matrix_view = &*owned_row_matrix;
        }
        if (vector_view != nullptr) {
            if (vector_view == matrix_view && owned_row_matrix) {
                insert_vector_view = insert_matrix_view;
            } else {
                owned_row_vector.emplace(*vector_view, *row_dof_map_, row_dof_offset_);
                insert_vector_view = &*owned_row_vector;
            }
        }
    }

    // Iterate over boundary faces with given marker
    mesh.forEachBoundaryFace(boundary_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            if (!owned_rows_only && !mesh.isOwnedCell(cell_id)) {
                return;
            }
            // Get cell DOFs (rows/cols may come from different maps)
            auto row_cell_dofs = row_dof_map_->getCellDofs(cell_id);
            auto col_cell_dofs = col_dof_map_->getCellDofs(cell_id);

            row_dofs_.resize(row_cell_dofs.size());
            for (std::size_t i = 0; i < row_cell_dofs.size(); ++i) {
                row_dofs_[i] = row_cell_dofs[i] + row_dof_offset_;
            }

            col_dofs_.resize(col_cell_dofs.size());
            for (std::size_t j = 0; j < col_cell_dofs.size(); ++j) {
                col_dofs_[j] = col_cell_dofs[j] + col_dof_offset_;
            }

            // Prepare context for face
            LocalIndex local_face_id = mesh.getLocalFaceIndex(face_id, cell_id);
            prepareContextFace(context_, mesh, face_id, cell_id, local_face_id, test_space, trial_space,
                               required_data, ContextType::BoundaryFace);
            context_.setMaterialState(nullptr, nullptr, 0u, 0u);
            context_.setTimeIntegrationContext(time_integration_);
            context_.setTime(time_);
            context_.setTimeStep(dt_);
            context_.setRealParameterGetter(get_real_param_);
            context_.setParameterGetter(get_param_);
            context_.setUserData(user_data_);
            context_.setJITConstants(jit_constants_);
            context_.setCoupledValues(coupled_integrals_, coupled_aux_state_);
            context_.clearAllPreviousSolutionData();
            context_.setBoundaryMarker(boundary_marker);

	            if (need_solution) {
	                FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
	                            "StandardAssembler::assembleBoundaryFaces: kernel requires solution but no solution was set");
	                local_solution_coeffs_.resize(col_dofs_.size());
	                if (current_solution_view_ != nullptr) {
	                    for (std::size_t i = 0; i < col_dofs_.size(); ++i) {
	                        const auto dof = col_dofs_[i];
	                        if (dof < 0) {
	                            FE_THROW(FEException, "StandardAssembler::assembleBoundaryFaces: negative DOF index");
	                        }
	                        local_solution_coeffs_[i] = current_solution_view_->getVectorEntry(dof);
	                    }
	                } else {
	                    for (std::size_t i = 0; i < col_dofs_.size(); ++i) {
	                        const auto dof = col_dofs_[i];
	                        FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= current_solution_.size(), FEException,
	                                    "StandardAssembler::assembleBoundaryFaces: solution vector too small for DOF " +
	                                        std::to_string(dof));
	                        local_solution_coeffs_[i] = current_solution_[static_cast<std::size_t>(dof)];
	                    }
	                }
	                if (context_.trialUsesVectorBasis()) {
	                    applyVectorBasisGlobalToLocal(mesh, cell_id, trial_space,
	                                                  std::span<Real>(local_solution_coeffs_));
	                }
	                context_.setSolutionCoefficients(local_solution_coeffs_);

	                if (time_integration_ != nullptr) {
	                    const int required = requiredHistoryStates(time_integration_);
	                    if (required > 0) {
	                        FE_THROW_IF(previous_solutions_.size() < static_cast<std::size_t>(required), FEException,
	                                    "StandardAssembler::assembleBoundaryFaces: time integration requires " +
	                                        std::to_string(required) + " history states, but only " +
	                                        std::to_string(previous_solutions_.size()) + " were provided");
	                        if (local_prev_solution_coeffs_.size() < static_cast<std::size_t>(required)) {
	                            local_prev_solution_coeffs_.resize(static_cast<std::size_t>(required));
	                        }
	                        for (int k = 1; k <= required; ++k) {
	                            const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	                            const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                                        ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                                        : nullptr;
	                            FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                        "StandardAssembler::assembleBoundaryFaces: previous solution (k=" +
	                                            std::to_string(k) + ") not set");
	                            auto& local_prev = local_prev_solution_coeffs_[static_cast<std::size_t>(k - 1)];
	                            local_prev.resize(col_dofs_.size());
	                            for (std::size_t i = 0; i < col_dofs_.size(); ++i) {
	                                const auto dof = col_dofs_[i];
	                                if (dof < 0) {
	                                    FE_THROW(FEException, "StandardAssembler::assembleBoundaryFaces: negative DOF index (prev)");
	                                }
	                                if (prev_view != nullptr) {
	                                    local_prev[i] = prev_view->getVectorEntry(dof);
	                                } else {
	                                    FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                                "StandardAssembler::assembleBoundaryFaces: previous solution vector too small for DOF " +
	                                                    std::to_string(dof));
	                                    local_prev[i] = prev[static_cast<std::size_t>(dof)];
	                                }
	                            }
	                            if (context_.trialUsesVectorBasis()) {
	                                applyVectorBasisGlobalToLocal(mesh, cell_id, trial_space,
	                                                              std::span<Real>(local_prev));
	                            }
                            context_.setPreviousSolutionCoefficientsK(k, local_prev);
                        }
                    }
                }
            }

            if (need_field_solutions) {
                populateFieldSolutionData(context_, mesh, cell_id, field_requirements);
            }

            if (need_material_state) {
                auto view = material_state_provider_->getBoundaryFaceState(kernel, face_id, context_.numQuadraturePoints());
                FE_THROW_IF(!view, FEException,
                            "StandardAssembler::assembleBoundaryFaces: material state provider returned null storage");
                FE_THROW_IF(view.bytes_per_qpt != material_state_spec.bytes_per_qpt, FEException,
                            "StandardAssembler::assembleBoundaryFaces: material state bytes_per_qpt mismatch");
                FE_THROW_IF(view.stride_bytes < view.bytes_per_qpt, FEException,
                            "StandardAssembler::assembleBoundaryFaces: invalid material state stride");
                context_.setMaterialState(view.data_old, view.data_work, view.bytes_per_qpt, view.stride_bytes, view.alignment);
            }

            // Compute local contributions
            kernel_output_.clear();
            kernel.computeBoundaryFace(context_, boundary_marker, kernel_output_);

            if (context_.testUsesVectorBasis() || context_.trialUsesVectorBasis()) {
                applyVectorBasisOutputOrientation(mesh, cell_id, test_space, cell_id, trial_space, kernel_output_);
            }

            // Insert into global system
            if (options_.use_constraints && constraint_distributor_) {
                insertLocalConstrained(kernel_output_, row_dofs_, col_dofs_,
                                       insert_matrix_view, insert_vector_view);
            } else {
                insertLocal(kernel_output_, row_dofs_, col_dofs_,
                            insert_matrix_view, insert_vector_view);
            }

            result.boundary_faces_assembled++;
        });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

AssemblyResult StandardAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (!kernel.hasInteriorFace()) {
        return result;
    }

    matrix_view.beginAssemblyPhase();
    if (vector_view && vector_view != &matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();
    const auto field_requirements = kernel.fieldRequirements();
    const bool need_field_solutions = !field_requirements.empty();
    const bool need_solution =
        hasFlag(required_data, RequiredData::SolutionCoefficients) ||
        hasFlag(required_data, RequiredData::SolutionValues) ||
        hasFlag(required_data, RequiredData::SolutionGradients) ||
        hasFlag(required_data, RequiredData::SolutionHessians) ||
        hasFlag(required_data, RequiredData::SolutionLaplacians);
    const bool need_material_state =
        hasFlag(required_data, RequiredData::MaterialState);
    const auto material_state_spec = kernel.materialStateSpec();

    if (need_material_state) {
        FE_THROW_IF(material_state_provider_ == nullptr, FEException,
                    "StandardAssembler::assembleInteriorFaces: kernel requires material state but no material state provider was set");
        FE_THROW_IF(material_state_spec.bytes_per_qpt == 0u, FEException,
                    "StandardAssembler::assembleInteriorFaces: kernel requires material state but materialStateSpec().bytes_per_qpt == 0");
    }

    FE_CHECK_NOT_NULL(row_dof_map_, "StandardAssembler::assembleInteriorFaces: row_dof_map");
    if (!col_dof_map_) {
        col_dof_map_ = row_dof_map_;
        col_dof_offset_ = row_dof_offset_;
    }

    // Ensure AssemblyContext uses the mesh spatial dimension (2D/3D).
    context_.reserve(std::max(row_dof_map_->getMaxDofsPerCell(),
                              col_dof_map_->getMaxDofsPerCell()),
                     /*max_qpts=*/27, mesh.dimension());

    const bool owned_rows_only = (options_.ghost_policy == GhostPolicy::OwnedRowsOnly);
    std::optional<OwnedRowOnlyView> owned_row_matrix;
    std::optional<OwnedRowOnlyView> owned_row_vector;
    GlobalSystemView* insert_matrix_view = &matrix_view;
    GlobalSystemView* insert_vector_view = vector_view;
    if (owned_rows_only) {
        owned_row_matrix.emplace(matrix_view, *row_dof_map_, row_dof_offset_);
        insert_matrix_view = &*owned_row_matrix;
        if (vector_view != nullptr) {
            if (vector_view == &matrix_view) {
                insert_vector_view = insert_matrix_view;
            } else {
                owned_row_vector.emplace(*vector_view, *row_dof_map_, row_dof_offset_);
                insert_vector_view = &*owned_row_vector;
            }
        }
    }

    const auto should_process = [&](GlobalIndex cell_minus, GlobalIndex cell_plus) -> bool {
        if (owned_rows_only) {
            return mesh.isOwnedCell(cell_minus) || mesh.isOwnedCell(cell_plus);
        }
        return mesh.isOwnedCell(cell_minus);
    };

    // Create second context for the "plus" side
    AssemblyContext context_plus;
    const auto max_row_dofs = row_dof_map_->getMaxDofsPerCell();
    const auto max_col_dofs = col_dof_map_->getMaxDofsPerCell();
    context_plus.reserve(std::max(max_row_dofs, max_col_dofs), 27, mesh.dimension());

    // Kernel outputs for DG face terms
    KernelOutput output_minus, output_plus, coupling_mp, coupling_pm;

    // Scratch for DOFs
    std::vector<GlobalIndex> minus_row_dofs, plus_row_dofs;
    std::vector<GlobalIndex> minus_col_dofs, plus_col_dofs;
    std::vector<Real> plus_solution_coeffs;
    std::vector<std::vector<Real>> plus_prev_solution_coeffs;
    std::vector<GlobalIndex> cell_nodes_minus;
    std::vector<GlobalIndex> cell_nodes_plus;

    mesh.forEachInteriorFace(
        [&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
            if (!should_process(cell_minus, cell_plus)) {
                return;
            }
            // Get DOFs for both cells (rows/cols may differ)
            auto minus_row_local = row_dof_map_->getCellDofs(cell_minus);
            auto plus_row_local = row_dof_map_->getCellDofs(cell_plus);
            auto minus_col_local = col_dof_map_->getCellDofs(cell_minus);
            auto plus_col_local = col_dof_map_->getCellDofs(cell_plus);

            minus_row_dofs.resize(minus_row_local.size());
            for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
                minus_row_dofs[i] = minus_row_local[i] + row_dof_offset_;
            }
            plus_row_dofs.resize(plus_row_local.size());
            for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
                plus_row_dofs[i] = plus_row_local[i] + row_dof_offset_;
            }

            minus_col_dofs.resize(minus_col_local.size());
            for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
                minus_col_dofs[j] = minus_col_local[j] + col_dof_offset_;
            }
            plus_col_dofs.resize(plus_col_local.size());
            for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
                plus_col_dofs[j] = plus_col_local[j] + col_dof_offset_;
            }

            // Prepare contexts for both sides
            LocalIndex local_face_minus = mesh.getLocalFaceIndex(face_id, cell_minus);
            LocalIndex local_face_plus = mesh.getLocalFaceIndex(face_id, cell_plus);

            prepareContextFace(context_, mesh, face_id, cell_minus, local_face_minus, test_space, trial_space,
                               required_data, ContextType::InteriorFace);
            context_.setMaterialState(nullptr, nullptr, 0u, 0u);
            context_.setTimeIntegrationContext(time_integration_);
            context_.setTime(time_);
            context_.setTimeStep(dt_);
            context_.setRealParameterGetter(get_real_param_);
            context_.setParameterGetter(get_param_);
            context_.setUserData(user_data_);
            context_.setJITConstants(jit_constants_);
            context_.setCoupledValues(coupled_integrals_, coupled_aux_state_);
            context_.clearAllPreviousSolutionData();

            std::array<LocalIndex, 4> align_plus_storage{};
            std::span<const LocalIndex> align_plus{};
            const ElementType cell_type_minus = mesh.getCellType(cell_minus);
            const ElementType cell_type_plus = mesh.getCellType(cell_plus);
            if (cell_type_minus == cell_type_plus) {
                elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type_minus);
                const auto& face_nodes_minus = ref.face_nodes(static_cast<std::size_t>(local_face_minus));
                const auto& face_nodes_plus = ref.face_nodes(static_cast<std::size_t>(local_face_plus));
                if (face_nodes_minus.size() == face_nodes_plus.size() &&
                    (face_nodes_minus.size() == 2 || face_nodes_minus.size() == 3)) {
                    mesh.getCellNodes(cell_minus, cell_nodes_minus);
                    mesh.getCellNodes(cell_plus, cell_nodes_plus);

                    for (std::size_t j = 0; j < face_nodes_plus.size(); ++j) {
                        const GlobalIndex global_plus = cell_nodes_plus.at(static_cast<std::size_t>(face_nodes_plus[j]));
                        std::size_t i_match = face_nodes_minus.size();
                        for (std::size_t i = 0; i < face_nodes_minus.size(); ++i) {
                            const GlobalIndex global_minus = cell_nodes_minus.at(static_cast<std::size_t>(face_nodes_minus[i]));
                            if (global_minus == global_plus) {
                                i_match = i;
                                break;
                            }
                        }
                        align_plus_storage[j] = static_cast<LocalIndex>(i_match);
                    }

                    bool ok = true;
                    for (std::size_t j = 0; j < face_nodes_plus.size(); ++j) {
                        if (static_cast<std::size_t>(align_plus_storage[j]) >= face_nodes_minus.size()) {
                            ok = false;
                            break;
                        }
                    }

                    if (ok) {
                        align_plus = std::span<const LocalIndex>(
                            align_plus_storage.data(),
                            face_nodes_plus.size());
                    }
                }
            }

            prepareContextFace(context_plus, mesh, face_id, cell_plus, local_face_plus, test_space, trial_space,
                               required_data, ContextType::InteriorFace, align_plus);
            context_plus.setMaterialState(nullptr, nullptr, 0u, 0u);
            context_plus.setTimeIntegrationContext(time_integration_);
            context_plus.setTime(time_);
            context_plus.setTimeStep(dt_);
            context_plus.setRealParameterGetter(get_real_param_);
            context_plus.setParameterGetter(get_param_);
            context_plus.setUserData(user_data_);
            context_plus.setJITConstants(jit_constants_);
            context_plus.setCoupledValues(coupled_integrals_, coupled_aux_state_);
            context_plus.clearAllPreviousSolutionData();

	            if (need_solution) {
	                FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
	                            "StandardAssembler::assembleInteriorFaces: kernel requires solution but no solution was set");
	
	                local_solution_coeffs_.resize(minus_col_dofs.size());
	                if (current_solution_view_ != nullptr) {
	                    for (std::size_t i = 0; i < minus_col_dofs.size(); ++i) {
	                        const auto dof = minus_col_dofs[i];
	                        if (dof < 0) {
	                            FE_THROW(FEException, "StandardAssembler::assembleInteriorFaces: negative DOF index");
	                        }
	                        local_solution_coeffs_[i] = current_solution_view_->getVectorEntry(dof);
	                    }
	                } else {
	                    for (std::size_t i = 0; i < minus_col_dofs.size(); ++i) {
	                        const auto dof = minus_col_dofs[i];
	                        FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= current_solution_.size(), FEException,
	                                    "StandardAssembler::assembleInteriorFaces: solution vector too small for DOF " +
	                                        std::to_string(dof));
	                        local_solution_coeffs_[i] = current_solution_[static_cast<std::size_t>(dof)];
	                    }
	                }
	                if (context_.trialUsesVectorBasis()) {
	                    applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
	                                                  std::span<Real>(local_solution_coeffs_));
	                }
	                context_.setSolutionCoefficients(local_solution_coeffs_);
	
	                plus_solution_coeffs.resize(plus_col_dofs.size());
	                if (current_solution_view_ != nullptr) {
	                    for (std::size_t i = 0; i < plus_col_dofs.size(); ++i) {
	                        const auto dof = plus_col_dofs[i];
	                        if (dof < 0) {
	                            FE_THROW(FEException, "StandardAssembler::assembleInteriorFaces: negative DOF index");
	                        }
	                        plus_solution_coeffs[i] = current_solution_view_->getVectorEntry(dof);
	                    }
	                } else {
	                    for (std::size_t i = 0; i < plus_col_dofs.size(); ++i) {
	                        const auto dof = plus_col_dofs[i];
	                        FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= current_solution_.size(), FEException,
	                                    "StandardAssembler::assembleInteriorFaces: solution vector too small for DOF " +
	                                        std::to_string(dof));
	                        plus_solution_coeffs[i] = current_solution_[static_cast<std::size_t>(dof)];
	                    }
	                }
	                if (context_plus.trialUsesVectorBasis()) {
	                    applyVectorBasisGlobalToLocal(mesh, cell_plus, trial_space,
	                                                  std::span<Real>(plus_solution_coeffs));
	                }
	                context_plus.setSolutionCoefficients(plus_solution_coeffs);

                if (time_integration_ != nullptr) {
                    const int required = requiredHistoryStates(time_integration_);
                    if (required > 0) {
                        FE_THROW_IF(previous_solutions_.size() < static_cast<std::size_t>(required), FEException,
                                    "StandardAssembler::assembleInteriorFaces: time integration requires " +
                                        std::to_string(required) + " history states, but only " +
                                        std::to_string(previous_solutions_.size()) + " were provided");
                        if (local_prev_solution_coeffs_.size() < static_cast<std::size_t>(required)) {
                            local_prev_solution_coeffs_.resize(static_cast<std::size_t>(required));
                        }
                        if (plus_prev_solution_coeffs.size() < static_cast<std::size_t>(required)) {
                            plus_prev_solution_coeffs.resize(static_cast<std::size_t>(required));
                        }

	                        for (int k = 1; k <= required; ++k) {
	                            const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	                            const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                                        ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                                        : nullptr;
	                            FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                        "StandardAssembler::assembleInteriorFaces: previous solution (k=" +
	                                            std::to_string(k) + ") not set");

	                            auto& local_prev_minus = local_prev_solution_coeffs_[static_cast<std::size_t>(k - 1)];
	                            local_prev_minus.resize(minus_col_dofs.size());
	                            for (std::size_t i = 0; i < minus_col_dofs.size(); ++i) {
	                                const auto dof = minus_col_dofs[i];
	                                if (dof < 0) {
	                                    FE_THROW(FEException,
	                                             "StandardAssembler::assembleInteriorFaces: negative DOF index (prev)");
	                                }
	                                if (prev_view != nullptr) {
	                                    local_prev_minus[i] = prev_view->getVectorEntry(dof);
	                                } else {
	                                    FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                                "StandardAssembler::assembleInteriorFaces: previous solution vector too small for DOF " +
	                                                    std::to_string(dof));
	                                    local_prev_minus[i] = prev[static_cast<std::size_t>(dof)];
	                                }
	                            }
	                            if (context_.trialUsesVectorBasis()) {
	                                applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
	                                                              std::span<Real>(local_prev_minus));
	                            }
                            context_.setPreviousSolutionCoefficientsK(k, local_prev_minus);

	                            auto& local_prev_plus = plus_prev_solution_coeffs[static_cast<std::size_t>(k - 1)];
	                            local_prev_plus.resize(plus_col_dofs.size());
	                            for (std::size_t i = 0; i < plus_col_dofs.size(); ++i) {
	                                const auto dof = plus_col_dofs[i];
	                                if (dof < 0) {
	                                    FE_THROW(FEException,
	                                             "StandardAssembler::assembleInteriorFaces: negative DOF index (prev)");
	                                }
	                                if (prev_view != nullptr) {
	                                    local_prev_plus[i] = prev_view->getVectorEntry(dof);
	                                } else {
	                                    FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                                "StandardAssembler::assembleInteriorFaces: previous solution vector too small for DOF " +
	                                                    std::to_string(dof));
	                                    local_prev_plus[i] = prev[static_cast<std::size_t>(dof)];
	                                }
	                            }
	                            if (context_plus.trialUsesVectorBasis()) {
	                                applyVectorBasisGlobalToLocal(mesh, cell_plus, trial_space,
	                                                              std::span<Real>(local_prev_plus));
	                            }
                            context_plus.setPreviousSolutionCoefficientsK(k, local_prev_plus);
                        }
                    }
                }
            }

            if (need_field_solutions) {
                populateFieldSolutionData(context_, mesh, cell_minus, field_requirements);
                populateFieldSolutionData(context_plus, mesh, cell_plus, field_requirements);
            }

            if (need_material_state) {
                FE_THROW_IF(context_plus.numQuadraturePoints() != context_.numQuadraturePoints(), FEException,
                            "StandardAssembler::assembleInteriorFaces: mismatched quadrature point counts for interior face state binding");

                auto view = material_state_provider_->getInteriorFaceState(kernel, face_id, context_.numQuadraturePoints());
                FE_THROW_IF(!view, FEException,
                            "StandardAssembler::assembleInteriorFaces: material state provider returned null storage");
                FE_THROW_IF(view.bytes_per_qpt != material_state_spec.bytes_per_qpt, FEException,
                            "StandardAssembler::assembleInteriorFaces: material state bytes_per_qpt mismatch");
                FE_THROW_IF(view.stride_bytes < view.bytes_per_qpt, FEException,
                            "StandardAssembler::assembleInteriorFaces: invalid material state stride");

                context_.setMaterialState(view.data_old, view.data_work, view.bytes_per_qpt, view.stride_bytes, view.alignment);
                context_plus.setMaterialState(view.data_old, view.data_work, view.bytes_per_qpt, view.stride_bytes, view.alignment);
            }

            // Compute DG face contributions
            output_minus.clear();
            output_plus.clear();
            coupling_mp.clear();
            coupling_pm.clear();

            kernel.computeInteriorFace(context_, context_plus,
                                       output_minus, output_plus,
                                       coupling_mp, coupling_pm);

            if (output_minus.has_matrix || output_minus.has_vector) {
                applyVectorBasisOutputOrientation(mesh, cell_minus, test_space, cell_minus, trial_space, output_minus);
            }
            if (output_plus.has_matrix || output_plus.has_vector) {
                applyVectorBasisOutputOrientation(mesh, cell_plus, test_space, cell_plus, trial_space, output_plus);
            }
            if (coupling_mp.has_matrix) {
                applyVectorBasisOutputOrientation(mesh, cell_minus, test_space, cell_plus, trial_space, coupling_mp);
            }
            if (coupling_pm.has_matrix) {
                applyVectorBasisOutputOrientation(mesh, cell_plus, test_space, cell_minus, trial_space, coupling_pm);
            }

            // Insert contributions (4 blocks for DG)
            // Self-coupling: minus-minus
            if (output_minus.has_matrix || output_minus.has_vector) {
                insertLocal(output_minus, minus_row_dofs, minus_col_dofs, insert_matrix_view, insert_vector_view);
            }

            // Self-coupling: plus-plus
            if (output_plus.has_matrix || output_plus.has_vector) {
                insertLocal(output_plus, plus_row_dofs, plus_col_dofs, insert_matrix_view, insert_vector_view);
            }

            // Cross-coupling: minus-plus (minus rows, plus cols)
            if (coupling_mp.has_matrix) {
                insert_matrix_view->addMatrixEntries(minus_row_dofs, plus_col_dofs,
                                                     coupling_mp.local_matrix);
            }

            // Cross-coupling: plus-minus (plus rows, minus cols)
            if (coupling_pm.has_matrix) {
                insert_matrix_view->addMatrixEntries(plus_row_dofs, minus_col_dofs,
                                                     coupling_pm.local_matrix);
            }

            result.interior_faces_assembled++;
        });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
AssemblyResult StandardAssembler::assembleInterfaceFaces(
    const IMeshAccess& mesh,
    const svmp::InterfaceMesh& interface_mesh,
    int interface_marker,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (!kernel.hasInterfaceFace()) {
        return result;
    }

    matrix_view.beginAssemblyPhase();
    if (vector_view && vector_view != &matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();
    const auto field_requirements = kernel.fieldRequirements();
    const bool need_field_solutions = !field_requirements.empty();
    const bool need_solution =
        hasFlag(required_data, RequiredData::SolutionCoefficients) ||
        hasFlag(required_data, RequiredData::SolutionValues) ||
        hasFlag(required_data, RequiredData::SolutionGradients) ||
        hasFlag(required_data, RequiredData::SolutionHessians) ||
        hasFlag(required_data, RequiredData::SolutionLaplacians);
    const bool need_material_state =
        hasFlag(required_data, RequiredData::MaterialState);
    const auto material_state_spec = kernel.materialStateSpec();

    if (need_material_state) {
        FE_THROW_IF(material_state_provider_ == nullptr, FEException,
                    "StandardAssembler::assembleInterfaceFaces: kernel requires material state but no material state provider was set");
        FE_THROW_IF(material_state_spec.bytes_per_qpt == 0u, FEException,
                    "StandardAssembler::assembleInterfaceFaces: kernel requires material state but materialStateSpec().bytes_per_qpt == 0");
    }

    FE_CHECK_NOT_NULL(row_dof_map_, "StandardAssembler::assembleInterfaceFaces: row_dof_map");
    if (!col_dof_map_) {
        col_dof_map_ = row_dof_map_;
        col_dof_offset_ = row_dof_offset_;
    }

    // Ensure AssemblyContext uses the mesh spatial dimension (2D/3D).
    context_.reserve(std::max(row_dof_map_->getMaxDofsPerCell(),
                              col_dof_map_->getMaxDofsPerCell()),
                     /*max_qpts=*/27, mesh.dimension());

    // Create second context for the "plus" side
    AssemblyContext context_plus;
    const auto max_row_dofs = row_dof_map_->getMaxDofsPerCell();
    const auto max_col_dofs = col_dof_map_->getMaxDofsPerCell();
    context_plus.reserve(std::max(max_row_dofs, max_col_dofs), 27, mesh.dimension());

    // Kernel outputs for interface face terms
    KernelOutput output_minus, output_plus, coupling_mp, coupling_pm;

    // Scratch for DOFs
    std::vector<GlobalIndex> minus_row_dofs, plus_row_dofs;
    std::vector<GlobalIndex> minus_col_dofs, plus_col_dofs;
    std::vector<Real> plus_solution_coeffs;
    std::vector<std::vector<Real>> plus_prev_solution_coeffs;
    std::vector<GlobalIndex> cell_nodes_minus;
    std::vector<GlobalIndex> cell_nodes_plus;

    const bool owned_rows_only = (options_.ghost_policy == GhostPolicy::OwnedRowsOnly);
    const auto should_process = [&](GlobalIndex cell_minus, GlobalIndex cell_plus) -> bool {
        if (owned_rows_only) {
            return mesh.isOwnedCell(cell_minus) || mesh.isOwnedCell(cell_plus);
        }
        return mesh.isOwnedCell(cell_minus);
    };

    std::optional<OwnedRowOnlyView> owned_row_matrix;
    std::optional<OwnedRowOnlyView> owned_row_vector;
    GlobalSystemView* insert_matrix_view = &matrix_view;
    GlobalSystemView* insert_vector_view = vector_view;
    if (owned_rows_only) {
        owned_row_matrix.emplace(matrix_view, *row_dof_map_, row_dof_offset_);
        insert_matrix_view = &*owned_row_matrix;
        if (vector_view != nullptr) {
            if (vector_view == &matrix_view) {
                insert_vector_view = insert_matrix_view;
            } else {
                owned_row_vector.emplace(*vector_view, *row_dof_map_, row_dof_offset_);
                insert_vector_view = &*owned_row_vector;
            }
        }
    }

    for (std::size_t local_iface = 0; local_iface < interface_mesh.n_faces(); ++local_iface) {
        const auto iface = static_cast<svmp::index_t>(local_iface);
        const GlobalIndex face_id = static_cast<GlobalIndex>(interface_mesh.volume_face(iface));
        const GlobalIndex cell_minus = static_cast<GlobalIndex>(interface_mesh.volume_cell_minus(iface));
        const GlobalIndex cell_plus = static_cast<GlobalIndex>(interface_mesh.volume_cell_plus(iface));

        FE_THROW_IF(cell_minus == INVALID_GLOBAL_INDEX || cell_plus == INVALID_GLOBAL_INDEX, FEException,
                    "StandardAssembler::assembleInterfaceFaces: interface face requires two parent cells");

        if (!should_process(cell_minus, cell_plus)) {
            continue;
        }

        const LocalIndex local_face_minus = static_cast<LocalIndex>(interface_mesh.local_face_in_cell_minus(iface));
        const LocalIndex local_face_plus = static_cast<LocalIndex>(interface_mesh.local_face_in_cell_plus(iface));
        FE_THROW_IF(local_face_minus < 0 || local_face_plus < 0, FEException,
                    "StandardAssembler::assembleInterfaceFaces: invalid local face index in InterfaceMesh");

        // Get DOFs for both cells (rows/cols may differ)
        auto minus_row_local = row_dof_map_->getCellDofs(cell_minus);
        auto plus_row_local = row_dof_map_->getCellDofs(cell_plus);
        auto minus_col_local = col_dof_map_->getCellDofs(cell_minus);
        auto plus_col_local = col_dof_map_->getCellDofs(cell_plus);

        minus_row_dofs.resize(minus_row_local.size());
        for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
            minus_row_dofs[i] = minus_row_local[i] + row_dof_offset_;
        }
        plus_row_dofs.resize(plus_row_local.size());
        for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
            plus_row_dofs[i] = plus_row_local[i] + row_dof_offset_;
        }

        minus_col_dofs.resize(minus_col_local.size());
        for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
            minus_col_dofs[j] = minus_col_local[j] + col_dof_offset_;
        }
        plus_col_dofs.resize(plus_col_local.size());
        for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
            plus_col_dofs[j] = plus_col_local[j] + col_dof_offset_;
        }

        // Prepare contexts for both sides
        prepareContextFace(context_, mesh, face_id, cell_minus, local_face_minus, test_space, trial_space,
                           required_data, ContextType::InteriorFace);
        context_.setMaterialState(nullptr, nullptr, 0u, 0u);
        context_.setTimeIntegrationContext(time_integration_);
        context_.setTime(time_);
        context_.setTimeStep(dt_);
        context_.setRealParameterGetter(get_real_param_);
        context_.setParameterGetter(get_param_);
        context_.setUserData(user_data_);
        context_.setJITConstants(jit_constants_);
        context_.setCoupledValues(coupled_integrals_, coupled_aux_state_);
        context_.clearAllPreviousSolutionData();

        std::array<LocalIndex, 4> align_plus_storage{};
        std::span<const LocalIndex> align_plus{};
        const ElementType cell_type_minus = mesh.getCellType(cell_minus);
        const ElementType cell_type_plus = mesh.getCellType(cell_plus);
        if (cell_type_minus == cell_type_plus) {
            elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type_minus);
            const auto& face_nodes_minus = ref.face_nodes(static_cast<std::size_t>(local_face_minus));
            const auto& face_nodes_plus = ref.face_nodes(static_cast<std::size_t>(local_face_plus));
            if (face_nodes_minus.size() == face_nodes_plus.size() &&
                (face_nodes_minus.size() == 2 || face_nodes_minus.size() == 3)) {
                mesh.getCellNodes(cell_minus, cell_nodes_minus);
                mesh.getCellNodes(cell_plus, cell_nodes_plus);

                for (std::size_t j = 0; j < face_nodes_plus.size(); ++j) {
                    const GlobalIndex global_plus = cell_nodes_plus.at(static_cast<std::size_t>(face_nodes_plus[j]));
                    std::size_t i_match = face_nodes_minus.size();
                    for (std::size_t i = 0; i < face_nodes_minus.size(); ++i) {
                        const GlobalIndex global_minus = cell_nodes_minus.at(static_cast<std::size_t>(face_nodes_minus[i]));
                        if (global_minus == global_plus) {
                            i_match = i;
                            break;
                        }
                    }
                    align_plus_storage[j] = static_cast<LocalIndex>(i_match);
                }

                bool ok = true;
                for (std::size_t j = 0; j < face_nodes_plus.size(); ++j) {
                    if (static_cast<std::size_t>(align_plus_storage[j]) >= face_nodes_minus.size()) {
                        ok = false;
                        break;
                    }
                }

                if (ok) {
                    align_plus = std::span<const LocalIndex>(
                        align_plus_storage.data(),
                        face_nodes_plus.size());
                }
            }
        }

        prepareContextFace(context_plus, mesh, face_id, cell_plus, local_face_plus, test_space, trial_space,
                           required_data, ContextType::InteriorFace, align_plus);
        context_plus.setMaterialState(nullptr, nullptr, 0u, 0u);
        context_plus.setTimeIntegrationContext(time_integration_);
        context_plus.setTime(time_);
        context_plus.setTimeStep(dt_);
        context_plus.setRealParameterGetter(get_real_param_);
        context_plus.setParameterGetter(get_param_);
        context_plus.setUserData(user_data_);
        context_plus.setJITConstants(jit_constants_);
        context_plus.setCoupledValues(coupled_integrals_, coupled_aux_state_);
        context_plus.clearAllPreviousSolutionData();

        if (need_solution) {
            FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
                        "StandardAssembler::assembleInterfaceFaces: kernel requires solution but no solution was set");

            local_solution_coeffs_.resize(minus_col_dofs.size());
            if (current_solution_view_ != nullptr) {
                for (std::size_t i = 0; i < minus_col_dofs.size(); ++i) {
                    const auto dof = minus_col_dofs[i];
                    if (dof < 0) {
                        FE_THROW(FEException, "StandardAssembler::assembleInterfaceFaces: negative DOF index");
                    }
                    local_solution_coeffs_[i] = current_solution_view_->getVectorEntry(dof);
                }
            } else {
                for (std::size_t i = 0; i < minus_col_dofs.size(); ++i) {
                    const auto dof = minus_col_dofs[i];
                    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= current_solution_.size(), FEException,
                                "StandardAssembler::assembleInterfaceFaces: solution vector too small for DOF " +
                                    std::to_string(dof));
                    local_solution_coeffs_[i] = current_solution_[static_cast<std::size_t>(dof)];
                }
            }
            if (context_.trialUsesVectorBasis()) {
                applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
                                              std::span<Real>(local_solution_coeffs_));
            }
            context_.setSolutionCoefficients(local_solution_coeffs_);

            plus_solution_coeffs.resize(plus_col_dofs.size());
            if (current_solution_view_ != nullptr) {
                for (std::size_t i = 0; i < plus_col_dofs.size(); ++i) {
                    const auto dof = plus_col_dofs[i];
                    if (dof < 0) {
                        FE_THROW(FEException, "StandardAssembler::assembleInterfaceFaces: negative DOF index");
                    }
                    plus_solution_coeffs[i] = current_solution_view_->getVectorEntry(dof);
                }
            } else {
                for (std::size_t i = 0; i < plus_col_dofs.size(); ++i) {
                    const auto dof = plus_col_dofs[i];
                    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= current_solution_.size(), FEException,
                                "StandardAssembler::assembleInterfaceFaces: solution vector too small for DOF " +
                                    std::to_string(dof));
                    plus_solution_coeffs[i] = current_solution_[static_cast<std::size_t>(dof)];
                }
            }
            if (context_plus.trialUsesVectorBasis()) {
                applyVectorBasisGlobalToLocal(mesh, cell_plus, trial_space,
                                              std::span<Real>(plus_solution_coeffs));
            }
            context_plus.setSolutionCoefficients(plus_solution_coeffs);

            if (time_integration_ != nullptr) {
                const int required = requiredHistoryStates(time_integration_);
                if (required > 0) {
                    FE_THROW_IF(previous_solutions_.size() < static_cast<std::size_t>(required), FEException,
                                "StandardAssembler::assembleInterfaceFaces: time integration requires " +
                                    std::to_string(required) + " history states, but only " +
                                    std::to_string(previous_solutions_.size()) + " were provided");
                    if (local_prev_solution_coeffs_.size() < static_cast<std::size_t>(required)) {
                        local_prev_solution_coeffs_.resize(static_cast<std::size_t>(required));
                    }
                    if (plus_prev_solution_coeffs.size() < static_cast<std::size_t>(required)) {
                        plus_prev_solution_coeffs.resize(static_cast<std::size_t>(required));
                    }

	                    for (int k = 1; k <= required; ++k) {
	                        const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	                        const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                                    ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                                    : nullptr;
	                        FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                    "StandardAssembler::assembleInterfaceFaces: previous solution (k=" +
	                                        std::to_string(k) + ") not set");

	                        auto& local_prev_minus = local_prev_solution_coeffs_[static_cast<std::size_t>(k - 1)];
	                        local_prev_minus.resize(minus_col_dofs.size());
	                        for (std::size_t i = 0; i < minus_col_dofs.size(); ++i) {
	                            const auto dof = minus_col_dofs[i];
	                            if (dof < 0) {
	                                FE_THROW(FEException,
	                                         "StandardAssembler::assembleInterfaceFaces: negative DOF index (prev)");
	                            }
	                            if (prev_view != nullptr) {
	                                local_prev_minus[i] = prev_view->getVectorEntry(dof);
	                            } else {
	                                FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                            "StandardAssembler::assembleInterfaceFaces: previous solution vector too small for DOF " +
	                                                std::to_string(dof));
	                                local_prev_minus[i] = prev[static_cast<std::size_t>(dof)];
	                            }
	                        }
	                        if (context_.trialUsesVectorBasis()) {
	                            applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
	                                                          std::span<Real>(local_prev_minus));
	                        }
                        context_.setPreviousSolutionCoefficientsK(k, local_prev_minus);

	                        auto& local_prev_plus = plus_prev_solution_coeffs[static_cast<std::size_t>(k - 1)];
	                        local_prev_plus.resize(plus_col_dofs.size());
	                        for (std::size_t i = 0; i < plus_col_dofs.size(); ++i) {
	                            const auto dof = plus_col_dofs[i];
	                            if (dof < 0) {
	                                FE_THROW(FEException,
	                                         "StandardAssembler::assembleInterfaceFaces: negative DOF index (prev)");
	                            }
	                            if (prev_view != nullptr) {
	                                local_prev_plus[i] = prev_view->getVectorEntry(dof);
	                            } else {
	                                FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                            "StandardAssembler::assembleInterfaceFaces: previous solution vector too small for DOF " +
	                                                std::to_string(dof));
	                                local_prev_plus[i] = prev[static_cast<std::size_t>(dof)];
	                            }
	                        }
	                        if (context_plus.trialUsesVectorBasis()) {
	                            applyVectorBasisGlobalToLocal(mesh, cell_plus, trial_space,
	                                                          std::span<Real>(local_prev_plus));
	                        }
                        context_plus.setPreviousSolutionCoefficientsK(k, local_prev_plus);
                    }
                }
            }
        }

        if (need_field_solutions) {
            populateFieldSolutionData(context_, mesh, cell_minus, field_requirements);
            populateFieldSolutionData(context_plus, mesh, cell_plus, field_requirements);
        }

        if (need_material_state) {
            FE_THROW_IF(context_plus.numQuadraturePoints() != context_.numQuadraturePoints(), FEException,
                        "StandardAssembler::assembleInterfaceFaces: mismatched quadrature point counts for face state binding");

            // Reuse interior-face state storage for interface faces (subset of faces).
            auto view = material_state_provider_->getInteriorFaceState(kernel, face_id, context_.numQuadraturePoints());
            FE_THROW_IF(!view, FEException,
                        "StandardAssembler::assembleInterfaceFaces: material state provider returned null storage");
            FE_THROW_IF(view.bytes_per_qpt != material_state_spec.bytes_per_qpt, FEException,
                        "StandardAssembler::assembleInterfaceFaces: material state bytes_per_qpt mismatch");
            FE_THROW_IF(view.stride_bytes < view.bytes_per_qpt, FEException,
                        "StandardAssembler::assembleInterfaceFaces: invalid material state stride");

            context_.setMaterialState(view.data_old, view.data_work, view.bytes_per_qpt, view.stride_bytes, view.alignment);
            context_plus.setMaterialState(view.data_old, view.data_work, view.bytes_per_qpt, view.stride_bytes, view.alignment);
        }

        // Compute interface face contributions
        output_minus.clear();
        output_plus.clear();
        coupling_mp.clear();
        coupling_pm.clear();

        kernel.computeInterfaceFace(context_, context_plus, interface_marker,
                                   output_minus, output_plus,
                                   coupling_mp, coupling_pm);

        if (output_minus.has_matrix || output_minus.has_vector) {
            applyVectorBasisOutputOrientation(mesh, cell_minus, test_space, cell_minus, trial_space, output_minus);
        }
        if (output_plus.has_matrix || output_plus.has_vector) {
            applyVectorBasisOutputOrientation(mesh, cell_plus, test_space, cell_plus, trial_space, output_plus);
        }
        if (coupling_mp.has_matrix) {
            applyVectorBasisOutputOrientation(mesh, cell_minus, test_space, cell_plus, trial_space, coupling_mp);
        }
        if (coupling_pm.has_matrix) {
            applyVectorBasisOutputOrientation(mesh, cell_plus, test_space, cell_minus, trial_space, coupling_pm);
        }

        // Insert contributions (4 blocks, DG-style)
        if (output_minus.has_matrix || output_minus.has_vector) {
            insertLocal(output_minus, minus_row_dofs, minus_col_dofs, insert_matrix_view, insert_vector_view);
        }
        if (output_plus.has_matrix || output_plus.has_vector) {
            insertLocal(output_plus, plus_row_dofs, plus_col_dofs, insert_matrix_view, insert_vector_view);
        }
        if (coupling_mp.has_matrix) {
            insert_matrix_view->addMatrixEntries(minus_row_dofs, plus_col_dofs,
                                                 coupling_mp.local_matrix);
        }
        if (coupling_pm.has_matrix) {
            insert_matrix_view->addMatrixEntries(plus_row_dofs, minus_col_dofs,
                                                 coupling_pm.local_matrix);
        }

        result.interface_faces_assembled++;
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}
#endif

// ============================================================================
// Internal Implementation
// ============================================================================

AssemblyResult StandardAssembler::assembleCellsCore(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view,
    bool assemble_matrix,
    bool assemble_vector)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    // Begin assembly phase
    if (matrix_view && assemble_matrix) {
        matrix_view->beginAssemblyPhase();
    }
    if (vector_view && assemble_vector && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();
    const auto field_requirements = kernel.fieldRequirements();
    const bool need_field_solutions = !field_requirements.empty();
    const bool need_solution =
        hasFlag(required_data, RequiredData::SolutionCoefficients) ||
        hasFlag(required_data, RequiredData::SolutionValues) ||
        hasFlag(required_data, RequiredData::SolutionGradients) ||
        hasFlag(required_data, RequiredData::SolutionHessians) ||
        hasFlag(required_data, RequiredData::SolutionLaplacians);
    const bool need_material_state =
        hasFlag(required_data, RequiredData::MaterialState);
    const auto material_state_spec = kernel.materialStateSpec();

    if (need_material_state) {
        FE_THROW_IF(material_state_provider_ == nullptr, FEException,
                    "StandardAssembler::assembleCellsCore: kernel requires material state but no material state provider was set");
        FE_THROW_IF(material_state_spec.bytes_per_qpt == 0, FEException,
                    "StandardAssembler::assembleCellsCore: kernel requires material state but materialStateSpec().bytes_per_qpt == 0");
    }

    FE_CHECK_NOT_NULL(row_dof_map_, "StandardAssembler::assembleCellsCore: row_dof_map");
    if (!col_dof_map_) {
        col_dof_map_ = row_dof_map_;
        col_dof_offset_ = row_dof_offset_;
    }

    // Ensure AssemblyContext uses the mesh spatial dimension (2D/3D). This affects
    // the runtime shapes of geometry tensors (J, Jinv, normals, etc.) used by Forms.
    context_.reserve(std::max(row_dof_map_->getMaxDofsPerCell(),
                              col_dof_map_->getMaxDofsPerCell()),
                     /*max_qpts=*/27, mesh.dimension());

    const bool owned_rows_only = (options_.ghost_policy == GhostPolicy::OwnedRowsOnly);
    std::optional<OwnedRowOnlyView> owned_row_matrix;
    std::optional<OwnedRowOnlyView> owned_row_vector;
    GlobalSystemView* insert_matrix_view = matrix_view;
    GlobalSystemView* insert_vector_view = vector_view;
    if (owned_rows_only) {
        if (matrix_view != nullptr) {
            owned_row_matrix.emplace(*matrix_view, *row_dof_map_, row_dof_offset_);
            insert_matrix_view = &*owned_row_matrix;
        }
        if (vector_view != nullptr) {
            if (vector_view == matrix_view && owned_row_matrix) {
                insert_vector_view = insert_matrix_view;
            } else {
                owned_row_vector.emplace(*vector_view, *row_dof_map_, row_dof_offset_);
                insert_vector_view = &*owned_row_vector;
            }
        }
    }

    auto for_each_cell = [&](auto&& callback) {
        if (owned_rows_only) {
            mesh.forEachCell(std::forward<decltype(callback)>(callback));
        } else {
            // Default behavior for distributed meshes: assemble each cell exactly once to avoid
            // double-counting contributions from ghost layers.
            mesh.forEachOwnedCell(std::forward<decltype(callback)>(callback));
        }
    };

    // Iterate over cells
    for_each_cell([&](GlobalIndex cell_id) {
        // Get element DOFs (rows/cols may differ)
        auto row_local = row_dof_map_->getCellDofs(cell_id);
        auto col_local = col_dof_map_->getCellDofs(cell_id);

        row_dofs_.resize(row_local.size());
        for (std::size_t i = 0; i < row_local.size(); ++i) {
            row_dofs_[i] = row_local[i] + row_dof_offset_;
        }

        col_dofs_.resize(col_local.size());
        for (std::size_t j = 0; j < col_local.size(); ++j) {
            col_dofs_[j] = col_local[j] + col_dof_offset_;
        }

        // Prepare assembly context
        prepareContext(context_, mesh, cell_id, test_space, trial_space, required_data);
        context_.setMaterialState(nullptr, nullptr, 0u, 0u);
        context_.setTimeIntegrationContext(time_integration_);
        context_.setTime(time_);
        context_.setTimeStep(dt_);
        context_.setRealParameterGetter(get_real_param_);
        context_.setParameterGetter(get_param_);
        context_.setUserData(user_data_);
        context_.setJITConstants(jit_constants_);
        context_.setCoupledValues(coupled_integrals_, coupled_aux_state_);
        context_.clearAllPreviousSolutionData();
        FE_THROW_IF(row_dofs_.size() != static_cast<std::size_t>(context_.numTestDofs()), FEException,
                    "StandardAssembler::assembleCellsCore: row DOF count does not match test space element DOFs");
        FE_THROW_IF(col_dofs_.size() != static_cast<std::size_t>(context_.numTrialDofs()), FEException,
                    "StandardAssembler::assembleCellsCore: column DOF count does not match trial space element DOFs");

	        if (need_solution) {
	            FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
	                        "StandardAssembler::assembleCellsCore: kernel requires solution but no solution was set");
	            local_solution_coeffs_.resize(col_dofs_.size());
	            if (current_solution_view_ != nullptr) {
	                for (std::size_t i = 0; i < col_dofs_.size(); ++i) {
	                    const auto dof = col_dofs_[i];
	                    if (dof < 0) {
	                        FE_THROW(FEException, "StandardAssembler::assembleCellsCore: negative DOF index");
	                    }
	                    local_solution_coeffs_[i] = current_solution_view_->getVectorEntry(dof);
	                }
	            } else {
	                for (std::size_t i = 0; i < col_dofs_.size(); ++i) {
	                    const auto dof = col_dofs_[i];
	                    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= current_solution_.size(), FEException,
	                                "StandardAssembler::assembleCellsCore: solution vector too small for DOF " +
	                                    std::to_string(dof));
	                    local_solution_coeffs_[i] = current_solution_[static_cast<std::size_t>(dof)];
	                }
	            }
	            if (context_.trialUsesVectorBasis()) {
	                applyVectorBasisGlobalToLocal(mesh, cell_id, trial_space,
	                                              std::span<Real>(local_solution_coeffs_));
	            }
	            context_.setSolutionCoefficients(local_solution_coeffs_);

            if (time_integration_ != nullptr) {
                const int required = requiredHistoryStates(time_integration_);
                if (required > 0) {
                    FE_THROW_IF(previous_solutions_.size() < static_cast<std::size_t>(required), FEException,
                                "StandardAssembler::assembleCellsCore: time integration requires " +
                                    std::to_string(required) + " history states, but only " +
                                    std::to_string(previous_solutions_.size()) + " were provided");
                    if (local_prev_solution_coeffs_.size() < static_cast<std::size_t>(required)) {
                        local_prev_solution_coeffs_.resize(static_cast<std::size_t>(required));
                    }
	                    for (int k = 1; k <= required; ++k) {
	                        const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	                        const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                                    ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                                    : nullptr;
	                        FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                    "StandardAssembler::assembleCellsCore: previous solution (k=" +
	                                        std::to_string(k) + ") not set");
	                        auto& local_prev = local_prev_solution_coeffs_[static_cast<std::size_t>(k - 1)];
	                        local_prev.resize(col_dofs_.size());
	                        for (std::size_t i = 0; i < col_dofs_.size(); ++i) {
	                            const auto dof = col_dofs_[i];
	                            if (dof < 0) {
	                                FE_THROW(FEException,
	                                         "StandardAssembler::assembleCellsCore: negative DOF index (prev)");
	                            }
	                            if (prev_view != nullptr) {
	                                local_prev[i] = prev_view->getVectorEntry(dof);
	                            } else {
	                                FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                            "StandardAssembler::assembleCellsCore: previous solution vector too small for DOF " +
	                                                std::to_string(dof));
	                                local_prev[i] = prev[static_cast<std::size_t>(dof)];
	                            }
	                        }
	                        if (context_.trialUsesVectorBasis()) {
	                            applyVectorBasisGlobalToLocal(mesh, cell_id, trial_space,
	                                                          std::span<Real>(local_prev));
	                        }
                        context_.setPreviousSolutionCoefficientsK(k, local_prev);
                    }
                }
            }
        }

        if (need_field_solutions) {
            populateFieldSolutionData(context_, mesh, cell_id, field_requirements);
        }

        if (need_material_state) {
            auto view = material_state_provider_->getCellState(kernel, cell_id, context_.numQuadraturePoints());
            FE_THROW_IF(!view, FEException,
                        "StandardAssembler::assembleCellsCore: material state provider returned null storage");
            FE_THROW_IF(view.bytes_per_qpt != material_state_spec.bytes_per_qpt, FEException,
                        "StandardAssembler::assembleCellsCore: material state bytes_per_qpt mismatch");
            FE_THROW_IF(view.stride_bytes < view.bytes_per_qpt, FEException,
                        "StandardAssembler::assembleCellsCore: invalid material state stride");
            context_.setMaterialState(view.data_old, view.data_work, view.bytes_per_qpt, view.stride_bytes, view.alignment);
        }

        // Compute local matrix/vector via kernel
        kernel_output_.clear();
        kernel.computeCell(context_, kernel_output_);

        if (context_.testUsesVectorBasis() || context_.trialUsesVectorBasis()) {
            applyVectorBasisOutputOrientation(mesh, cell_id, test_space, cell_id, trial_space, kernel_output_);
        }

        // Insert into global system
        if (options_.use_constraints && constraint_distributor_) {
            insertLocalConstrained(kernel_output_, row_dofs_, col_dofs_,
                                   assemble_matrix ? insert_matrix_view : nullptr,
                                   assemble_vector ? insert_vector_view : nullptr);
        } else {
            insertLocal(kernel_output_, row_dofs_, col_dofs_,
                        assemble_matrix ? insert_matrix_view : nullptr,
                        assemble_vector ? insert_vector_view : nullptr);
        }

        result.elements_assembled++;

        if (kernel_output_.has_matrix) {
            result.matrix_entries_inserted +=
                static_cast<GlobalIndex>(row_dofs_.size() * col_dofs_.size());
        }
        if (kernel_output_.has_vector) {
            result.vector_entries_inserted +=
                static_cast<GlobalIndex>(row_dofs_.size());
        }
    });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

void StandardAssembler::prepareContext(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data)
{
    // 1. Get element type from mesh
    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    // 2. Get elements for test and trial spaces
    const auto& test_element = getElement(test_space, cell_id, cell_type);
    const auto& trial_element = getElement(trial_space, cell_id, cell_type);

    // 3. Get quadrature rule from the element
    auto quad_rule = test_element.quadrature();
    if (!quad_rule) {
        // Fall back to factory-created quadrature if element doesn't provide one
        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            test_element.polynomial_order(), false);
        quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
    }

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());
    const auto n_test_dofs = static_cast<LocalIndex>(test_space.dofs_per_element());
    const auto n_trial_dofs = static_cast<LocalIndex>(trial_space.dofs_per_element());
    const auto n_test_scalar_dofs = static_cast<LocalIndex>(test_element.num_dofs());
    const auto n_trial_scalar_dofs = static_cast<LocalIndex>(trial_element.num_dofs());
    const bool test_is_product = (test_space.space_type() == spaces::SpaceType::Product);
    const bool trial_is_product = (trial_space.space_type() == spaces::SpaceType::Product);
    if (test_is_product) {
        FE_CHECK_ARG(test_space.field_type() == FieldType::Vector,
                     "StandardAssembler::prepareContext: ProductSpace test space must be vector-valued");
        FE_CHECK_ARG(test_space.value_dimension() > 0,
                     "StandardAssembler::prepareContext: invalid test space value dimension");
        FE_CHECK_ARG(n_test_dofs ==
                         static_cast<LocalIndex>(
                             n_test_scalar_dofs * static_cast<LocalIndex>(test_space.value_dimension())),
                     "StandardAssembler::prepareContext: test ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_test_dofs == n_test_scalar_dofs,
                     "StandardAssembler::prepareContext: non-Product test space DOF count mismatch");
    }
    if (trial_is_product) {
        FE_CHECK_ARG(trial_space.field_type() == FieldType::Vector,
                     "StandardAssembler::prepareContext: ProductSpace trial space must be vector-valued");
        FE_CHECK_ARG(trial_space.value_dimension() > 0,
                     "StandardAssembler::prepareContext: invalid trial space value dimension");
        FE_CHECK_ARG(n_trial_dofs ==
                         static_cast<LocalIndex>(
                             n_trial_scalar_dofs * static_cast<LocalIndex>(trial_space.value_dimension())),
                     "StandardAssembler::prepareContext: trial ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_trial_dofs == n_trial_scalar_dofs,
                     "StandardAssembler::prepareContext: non-Product trial space DOF count mismatch");
    }
    const bool need_basis_hessians = hasFlag(required_data, RequiredData::BasisHessians);
    const bool need_basis_curls = hasFlag(required_data, RequiredData::BasisCurls);
    const bool need_basis_divergences = hasFlag(required_data, RequiredData::BasisDivergences);

    const auto& test_basis = test_element.basis();
    const auto& trial_basis = trial_element.basis();
    const bool test_is_vector_basis = test_basis.is_vector_valued();
    const bool trial_is_vector_basis = trial_basis.is_vector_valued();

    const auto validate_vector_basis_requirements =
        [&](const spaces::FunctionSpace& space, bool is_vector_basis, const char* which) {
            if (!is_vector_basis) {
                return;
            }

            const auto continuity = space.continuity();
            FE_THROW_IF(continuity != Continuity::H_curl && continuity != Continuity::H_div, FEException,
                        std::string("StandardAssembler::prepareContext: ") + which +
                            " space uses a vector-valued basis but is not H(curl)/H(div)");

            if (continuity == Continuity::H_curl) {
                FE_THROW_IF(need_basis_divergences, FEException,
                            std::string("StandardAssembler::prepareContext: BasisDivergences requested for ") + which +
                                " H(curl) space");
            } else if (continuity == Continuity::H_div) {
                FE_THROW_IF(need_basis_curls, FEException,
                            std::string("StandardAssembler::prepareContext: BasisCurls requested for ") + which +
                                " H(div) space");
            }
        };

    validate_vector_basis_requirements(test_space, test_is_vector_basis, "test");
    validate_vector_basis_requirements(trial_space, trial_is_vector_basis, "trial");

    // 4. Get cell node coordinates from mesh
    mesh.getCellCoordinates(cell_id, cell_coords_);
    const auto n_nodes = cell_coords_.size();

    // Convert to math::Vector format for geometry mapping
    std::vector<math::Vector<Real, 3>> node_coords(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            cell_coords_[i][0], cell_coords_[i][1], cell_coords_[i][2]};
    }

    // 5. Create geometry mapping
    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = defaultGeometryOrder(cell_type);
    map_request.use_affine = (map_request.geometry_order <= 1);

    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    // 6. Resize scratch storage
    scratch_quad_points_.resize(n_qpts);
    scratch_quad_weights_.resize(n_qpts);
    scratch_phys_points_.resize(n_qpts);
    scratch_jacobians_.resize(n_qpts);
    scratch_inv_jacobians_.resize(n_qpts);
    scratch_jac_dets_.resize(n_qpts);
    scratch_integration_weights_.resize(n_qpts);

    const auto test_basis_size = static_cast<std::size_t>(n_test_dofs * n_qpts);
    const auto trial_basis_size = static_cast<std::size_t>(n_trial_dofs * n_qpts);
    const bool need_test_vector_values =
        test_is_vector_basis &&
        (hasFlag(required_data, RequiredData::BasisValues) ||
         hasFlag(required_data, RequiredData::SolutionValues) ||
         required_data == RequiredData::None);
    const bool need_trial_vector_values =
        trial_is_vector_basis &&
        (hasFlag(required_data, RequiredData::BasisValues) ||
         hasFlag(required_data, RequiredData::SolutionValues) ||
         required_data == RequiredData::None);

    if (test_is_vector_basis) {
        scratch_basis_values_.clear();
        scratch_ref_gradients_.clear();
        scratch_phys_gradients_.clear();
        scratch_ref_hessians_.clear();
        scratch_phys_hessians_.clear();

        if (need_test_vector_values) {
            scratch_basis_vector_values_.resize(test_basis_size);
        } else {
            scratch_basis_vector_values_.clear();
        }
        if (need_basis_curls) {
            scratch_basis_curls_.resize(test_basis_size);
        } else {
            scratch_basis_curls_.clear();
        }
        if (need_basis_divergences) {
            scratch_basis_divergences_.resize(test_basis_size);
        } else {
            scratch_basis_divergences_.clear();
        }
    } else {
        scratch_basis_vector_values_.clear();
        scratch_basis_curls_.clear();
        scratch_basis_divergences_.clear();

        scratch_basis_values_.resize(test_basis_size);
        scratch_ref_gradients_.resize(test_basis_size);
        scratch_phys_gradients_.resize(test_basis_size);
        if (need_basis_hessians) {
            scratch_ref_hessians_.resize(test_basis_size);
            scratch_phys_hessians_.resize(test_basis_size);
        } else {
            scratch_ref_hessians_.clear();
            scratch_phys_hessians_.clear();
        }
    }

    // Storage for trial if different from test
    std::vector<Real> trial_basis_values;
    std::vector<AssemblyContext::Vector3D> trial_ref_gradients;
    std::vector<AssemblyContext::Vector3D> trial_phys_gradients;
    std::vector<AssemblyContext::Vector3D> trial_basis_vector_values;
    std::vector<AssemblyContext::Vector3D> trial_basis_curls;
    std::vector<Real> trial_basis_divergences;
    std::vector<AssemblyContext::Matrix3x3> trial_ref_hessians;
    std::vector<AssemblyContext::Matrix3x3> trial_phys_hessians;

    if (&test_space != &trial_space) {
        if (trial_is_vector_basis) {
            trial_basis_values.clear();
            trial_ref_gradients.clear();
            trial_phys_gradients.clear();
            trial_ref_hessians.clear();
            trial_phys_hessians.clear();

            if (need_trial_vector_values) {
                trial_basis_vector_values.resize(trial_basis_size);
            }
            if (need_basis_curls) {
                trial_basis_curls.resize(trial_basis_size);
            }
            if (need_basis_divergences) {
                trial_basis_divergences.resize(trial_basis_size);
            }
        } else {
            trial_basis_vector_values.clear();
            trial_basis_curls.clear();
            trial_basis_divergences.clear();

            trial_basis_values.resize(trial_basis_size);
            trial_ref_gradients.resize(trial_basis_size);
            trial_phys_gradients.resize(trial_basis_size);
            if (need_basis_hessians) {
                trial_ref_hessians.resize(trial_basis_size);
                trial_phys_hessians.resize(trial_basis_size);
            } else {
                trial_ref_hessians.clear();
                trial_phys_hessians.clear();
            }
        }
    }

    // 7. Copy quadrature data and compute physical points and Jacobians
    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        // Copy reference quadrature point
        const auto& qpt = quad_points[q];
        scratch_quad_points_[q] = {qpt[0], qpt[1], qpt[2]};
        scratch_quad_weights_[q] = quad_weights[q];

        // Map to physical space
        const math::Vector<Real, 3> xi{qpt[0], qpt[1], qpt[2]};
        const auto x_phys = mapping->map_to_physical(xi);
        scratch_phys_points_[q] = {x_phys[0], x_phys[1], x_phys[2]};

        // Compute Jacobian
        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        // Store as arrays
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch_jacobians_[q][i][j] = J(i, j);
                scratch_inv_jacobians_[q][i][j] = J_inv(i, j);
            }
        }
        scratch_jac_dets_[q] = det_J;

        // Integration weight = quadrature weight * |det(J)|
        scratch_integration_weights_[q] = quad_weights[q] * std::abs(det_J);
    }

    // 8. Evaluate basis functions at quadrature points
    std::vector<Real> scalar_values_at_pt;
    std::vector<basis::Gradient> scalar_gradients_at_pt;
    std::vector<basis::Hessian> scalar_hessians_at_pt;

    std::vector<math::Vector<Real, 3>> vec_values_at_pt;
    std::vector<math::Vector<Real, 3>> vec_curls_at_pt;
    std::vector<Real> vec_divs_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch_quad_points_[q][0],
            scratch_quad_points_[q][1],
            scratch_quad_points_[q][2]};

        const auto& J = scratch_jacobians_[q];
        const auto& J_inv = scratch_inv_jacobians_[q];
        const Real det_J = scratch_jac_dets_[q];

        std::array<AssemblyContext::Matrix3x3, 3> d2xi_dx2{};
        if (need_basis_hessians) {
            const auto map_hess = mapping->mapping_hessian(xi);
            for (int a = 0; a < dim; ++a) {
                for (int i = 0; i < dim; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        Real sum = 0.0;
                        for (int m = 0; m < dim; ++m) {
                            for (int p = 0; p < dim; ++p) {
                                for (int r = 0; r < dim; ++r) {
                                    sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(m)] *
                                           map_hess[static_cast<std::size_t>(m)](
                                               static_cast<std::size_t>(p), static_cast<std::size_t>(r)) *
                                           J_inv[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)] *
                                           J_inv[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)];
                                }
                            }
                        }
                        d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = -sum;
                    }
                }
            }
        }

        if (test_is_vector_basis) {
            if (need_test_vector_values) {
                test_basis.evaluate_vector_values(xi, vec_values_at_pt);
            }
            if (need_basis_curls) {
                test_basis.evaluate_curl(xi, vec_curls_at_pt);
            }
            if (need_basis_divergences) {
                test_basis.evaluate_divergence(xi, vec_divs_at_pt);
            }

            const auto cont = test_space.continuity();
            for (LocalIndex i = 0; i < n_test_dofs; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);

                if (need_test_vector_values) {
                    const auto& vref = vec_values_at_pt[static_cast<std::size_t>(i)];
                    AssemblyContext::Vector3D vphys{0.0, 0.0, 0.0};
                    if (cont == Continuity::H_curl) {
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                vphys[static_cast<std::size_t>(r)] += J_inv[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] *
                                                                      vref[static_cast<std::size_t>(c)];
                            }
                        }
                    } else { // H_div
                        const Real inv_det = Real(1) / det_J;
                        for (int r = 0; r < 3; ++r) {
                            Real sum = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                       vref[static_cast<std::size_t>(c)];
                            }
                            vphys[static_cast<std::size_t>(r)] = inv_det * sum;
                        }
                    }
                    scratch_basis_vector_values_[idx] = vphys;
                }

                if (need_basis_curls) {
                    const auto& cref = vec_curls_at_pt[static_cast<std::size_t>(i)];
                    AssemblyContext::Vector3D cphys{0.0, 0.0, 0.0};
                    const Real inv_det = Real(1) / det_J;
                    for (int r = 0; r < 3; ++r) {
                        Real sum = 0.0;
                        for (int c = 0; c < 3; ++c) {
                            sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                   cref[static_cast<std::size_t>(c)];
                        }
                        cphys[static_cast<std::size_t>(r)] = inv_det * sum;
                    }
                    scratch_basis_curls_[idx] = cphys;
                }

                if (need_basis_divergences) {
                    scratch_basis_divergences_[idx] =
                        vec_divs_at_pt[static_cast<std::size_t>(i)] / det_J;
                }
            }
        } else {
            test_basis.evaluate_values(xi, scalar_values_at_pt);
            test_basis.evaluate_gradients(xi, scalar_gradients_at_pt);
            if (need_basis_hessians) {
                test_basis.evaluate_hessians(xi, scalar_hessians_at_pt);
            }

            for (LocalIndex i = 0; i < n_test_dofs; ++i) {
                const LocalIndex si = test_is_product ? static_cast<LocalIndex>(i % n_test_scalar_dofs) : i;
                const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
                scratch_basis_values_[idx] = scalar_values_at_pt[static_cast<std::size_t>(si)];
                scratch_ref_gradients_[idx] = {
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][0],
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][1],
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][2]};

                const auto& grad_ref = scratch_ref_gradients_[idx];
                AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                    }
                }
                scratch_phys_gradients_[idx] = grad_phys;

                if (need_basis_hessians) {
                    AssemblyContext::Matrix3x3 H_ref{};
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                scalar_hessians_at_pt[static_cast<std::size_t>(si)](static_cast<std::size_t>(r),
                                                                                     static_cast<std::size_t>(c));
                        }
                    }
                    scratch_ref_hessians_[idx] = H_ref;

                    AssemblyContext::Matrix3x3 H_phys{};
                    for (int r = 0; r < dim; ++r) {
                        for (int c = 0; c < dim; ++c) {
                            Real sum = 0.0;
                            for (int a = 0; a < dim; ++a) {
                                for (int b = 0; b < dim; ++b) {
                                    sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                           H_ref[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                           J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                }
                            }
                            for (int a = 0; a < dim; ++a) {
                                sum += grad_ref[static_cast<std::size_t>(a)] *
                                       d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            }
                            H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                        }
                    }
                    scratch_phys_hessians_[idx] = H_phys;
                }
            }
        }

        if (&test_space != &trial_space) {
            if (trial_is_vector_basis) {
                if (need_trial_vector_values) {
                    trial_basis.evaluate_vector_values(xi, vec_values_at_pt);
                }
                if (need_basis_curls) {
                    trial_basis.evaluate_curl(xi, vec_curls_at_pt);
                }
                if (need_basis_divergences) {
                    trial_basis.evaluate_divergence(xi, vec_divs_at_pt);
                }

                const auto cont = trial_space.continuity();
                for (LocalIndex j = 0; j < n_trial_dofs; ++j) {
                    const std::size_t idx = static_cast<std::size_t>(j * n_qpts + q);

                    if (need_trial_vector_values) {
                        const auto& vref = vec_values_at_pt[static_cast<std::size_t>(j)];
                        AssemblyContext::Vector3D vphys{0.0, 0.0, 0.0};
                        if (cont == Continuity::H_curl) {
                            for (int r = 0; r < dim; ++r) {
                                for (int c = 0; c < dim; ++c) {
                                    vphys[static_cast<std::size_t>(r)] += J_inv[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] *
                                                                          vref[static_cast<std::size_t>(c)];
                                }
                            }
                        } else { // H_div
                            const Real inv_det = Real(1) / det_J;
                            for (int r = 0; r < 3; ++r) {
                                Real sum = 0.0;
                                for (int c = 0; c < 3; ++c) {
                                    sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                           vref[static_cast<std::size_t>(c)];
                                }
                                vphys[static_cast<std::size_t>(r)] = inv_det * sum;
                            }
                        }
                        trial_basis_vector_values[idx] = vphys;
                    }

                    if (need_basis_curls) {
                        const auto& cref = vec_curls_at_pt[static_cast<std::size_t>(j)];
                        AssemblyContext::Vector3D cphys{0.0, 0.0, 0.0};
                        const Real inv_det = Real(1) / det_J;
                        for (int r = 0; r < 3; ++r) {
                            Real sum = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                       cref[static_cast<std::size_t>(c)];
                            }
                            cphys[static_cast<std::size_t>(r)] = inv_det * sum;
                        }
                        trial_basis_curls[idx] = cphys;
                    }

                    if (need_basis_divergences) {
                        trial_basis_divergences[idx] =
                            vec_divs_at_pt[static_cast<std::size_t>(j)] / det_J;
                    }
                }
            } else {
                trial_basis.evaluate_values(xi, scalar_values_at_pt);
                trial_basis.evaluate_gradients(xi, scalar_gradients_at_pt);
                if (need_basis_hessians) {
                    trial_basis.evaluate_hessians(xi, scalar_hessians_at_pt);
                }

                for (LocalIndex j = 0; j < n_trial_dofs; ++j) {
                    const LocalIndex sj = trial_is_product ? static_cast<LocalIndex>(j % n_trial_scalar_dofs) : j;
                    const std::size_t idx = static_cast<std::size_t>(j * n_qpts + q);
                    trial_basis_values[idx] = scalar_values_at_pt[static_cast<std::size_t>(sj)];
                    trial_ref_gradients[idx] = {
                        scalar_gradients_at_pt[static_cast<std::size_t>(sj)][0],
                        scalar_gradients_at_pt[static_cast<std::size_t>(sj)][1],
                        scalar_gradients_at_pt[static_cast<std::size_t>(sj)][2]};

                    const auto& grad_ref = trial_ref_gradients[idx];
                    AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

                    for (int d1 = 0; d1 < dim; ++d1) {
                        for (int d2 = 0; d2 < dim; ++d2) {
                            grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                        }
                    }
                    trial_phys_gradients[idx] = grad_phys;

                    if (need_basis_hessians) {
                        AssemblyContext::Matrix3x3 H_ref{};
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                    scalar_hessians_at_pt[static_cast<std::size_t>(sj)](static_cast<std::size_t>(r),
                                                                                         static_cast<std::size_t>(c));
                            }
                        }
                        trial_ref_hessians[idx] = H_ref;

                        AssemblyContext::Matrix3x3 H_phys{};
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                Real sum = 0.0;
                                for (int a = 0; a < dim; ++a) {
                                    for (int b = 0; b < dim; ++b) {
                                        sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                               H_ref[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                               J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                    }
                                }
                                for (int a = 0; a < dim; ++a) {
                                    sum += grad_ref[static_cast<std::size_t>(a)] *
                                           d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                }
                                H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                            }
                        }
                        trial_phys_hessians[idx] = H_phys;
                    }
                }
            }
        }
    }

    // 9. Configure context with basic info
    context.configure(cell_id, test_space, trial_space, required_data);
    context.setCellDomainId(mesh.getCellDomainId(cell_id));

    // 10. Set all computed data into context
    context.setQuadratureData(scratch_quad_points_, scratch_quad_weights_);
    context.setPhysicalPoints(scratch_phys_points_);
    context.setJacobianData(scratch_jacobians_, scratch_inv_jacobians_, scratch_jac_dets_);
    context.setIntegrationWeights(scratch_integration_weights_);

    // Basis data
    if (test_is_vector_basis) {
        context.setTestVectorBasisValues(n_test_dofs,
                                         need_test_vector_values
                                             ? std::span<const AssemblyContext::Vector3D>(scratch_basis_vector_values_)
                                             : std::span<const AssemblyContext::Vector3D>{});
        if (need_basis_curls) {
            context.setTestBasisCurls(n_test_dofs, std::span<const AssemblyContext::Vector3D>(scratch_basis_curls_));
        }
        if (need_basis_divergences) {
            context.setTestBasisDivergences(n_test_dofs, std::span<const Real>(scratch_basis_divergences_));
        }
    } else {
        context.setTestBasisData(n_test_dofs, scratch_basis_values_, scratch_ref_gradients_);
        if (need_basis_hessians) {
            context.setTestBasisHessians(n_test_dofs, scratch_ref_hessians_);
        }
    }

    if (&test_space != &trial_space) {
        if (trial_is_vector_basis) {
            context.setTrialVectorBasisValues(n_trial_dofs,
                                              need_trial_vector_values
                                                  ? std::span<const AssemblyContext::Vector3D>(trial_basis_vector_values)
                                                  : std::span<const AssemblyContext::Vector3D>{});
            if (need_basis_curls) {
                context.setTrialBasisCurls(n_trial_dofs, std::span<const AssemblyContext::Vector3D>(trial_basis_curls));
            }
            if (need_basis_divergences) {
                context.setTrialBasisDivergences(n_trial_dofs, std::span<const Real>(trial_basis_divergences));
            }
        } else {
            context.setTrialBasisData(n_trial_dofs, trial_basis_values, trial_ref_gradients);
            if (need_basis_hessians) {
                context.setTrialBasisHessians(n_trial_dofs, trial_ref_hessians);
            }
        }
    }

    // Physical gradients/Hessians (scalar bases only). Vector-basis spaces use curl/div instead.
    const auto test_phys = test_is_vector_basis
                               ? std::span<const AssemblyContext::Vector3D>{}
                               : std::span<const AssemblyContext::Vector3D>(scratch_phys_gradients_);
    std::span<const AssemblyContext::Vector3D> trial_phys{};
    if (&test_space != &trial_space) {
        trial_phys = trial_is_vector_basis ? std::span<const AssemblyContext::Vector3D>{}
                                           : std::span<const AssemblyContext::Vector3D>(trial_phys_gradients);
    } else {
        trial_phys = test_phys;
    }
    context.setPhysicalGradients(test_phys, trial_phys);

    if (need_basis_hessians) {
        const auto test_H = test_is_vector_basis
                                ? std::span<const AssemblyContext::Matrix3x3>{}
                                : std::span<const AssemblyContext::Matrix3x3>(scratch_phys_hessians_);
        std::span<const AssemblyContext::Matrix3x3> trial_H{};
        if (&test_space != &trial_space) {
            trial_H = trial_is_vector_basis ? std::span<const AssemblyContext::Matrix3x3>{}
                                            : std::span<const AssemblyContext::Matrix3x3>(trial_phys_hessians);
        } else {
            trial_H = test_H;
        }
        context.setPhysicalHessians(test_H, trial_H);
    }

    if (hasFlag(required_data, RequiredData::EntityMeasures)) {
        Real cell_volume = 0.0;
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            cell_volume += scratch_integration_weights_[static_cast<std::size_t>(q)];
        }

        Real h = 0.0;
        for (std::size_t a = 0; a < n_nodes; ++a) {
            for (std::size_t b = a + 1; b < n_nodes; ++b) {
                const Real dx = node_coords[a][0] - node_coords[b][0];
                const Real dy = node_coords[a][1] - node_coords[b][1];
                const Real dz = node_coords[a][2] - node_coords[b][2];
                const Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > h) h = dist;
            }
        }

        context.setEntityMeasures(h, cell_volume, /*facet_area=*/0.0);
    }
}

void StandardAssembler::prepareContextFace(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex face_id,
    GlobalIndex cell_id,
    LocalIndex local_face_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    ContextType type,
    std::span<const LocalIndex> align_facet_to_reference)
{
    // 1. Get element type from mesh
    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    // 2. Get element for test and trial spaces
    const auto& test_element = getElement(test_space, cell_id, cell_type);
    const auto& trial_element = getElement(trial_space, cell_id, cell_type);

    // 3. Determine face element type from reference topology
    elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face_id));

    ElementType face_type = ElementType::Unknown;
    switch (face_nodes.size()) {
        case 2:
            face_type = ElementType::Line2;
            break;
        case 3:
            face_type = ElementType::Triangle3;
            break;
        case 4:
            face_type = ElementType::Quad4;
            break;
        default:
            throw std::runtime_error("StandardAssembler::prepareContextFace: unsupported face topology");
    }

    // 4. Create a face quadrature rule
    const int quad_order = quadrature::QuadratureFactory::recommended_order(
        std::max(test_element.polynomial_order(), trial_element.polynomial_order()), false);
    auto quad_rule = quadrature::QuadratureFactory::create(face_type, quad_order);

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());
    const auto n_test_dofs = static_cast<LocalIndex>(test_space.dofs_per_element());
    const auto n_trial_dofs = static_cast<LocalIndex>(trial_space.dofs_per_element());
    const auto n_test_scalar_dofs = static_cast<LocalIndex>(test_element.num_dofs());
    const auto n_trial_scalar_dofs = static_cast<LocalIndex>(trial_element.num_dofs());
    const bool test_is_product = (test_space.space_type() == spaces::SpaceType::Product);
    const bool trial_is_product = (trial_space.space_type() == spaces::SpaceType::Product);
    if (test_is_product) {
        FE_CHECK_ARG(test_space.field_type() == FieldType::Vector,
                     "StandardAssembler::prepareContextFace: ProductSpace test space must be vector-valued");
        FE_CHECK_ARG(test_space.value_dimension() > 0,
                     "StandardAssembler::prepareContextFace: invalid test space value dimension");
        FE_CHECK_ARG(n_test_dofs ==
                         static_cast<LocalIndex>(
                             n_test_scalar_dofs * static_cast<LocalIndex>(test_space.value_dimension())),
                     "StandardAssembler::prepareContextFace: test ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_test_dofs == n_test_scalar_dofs,
                     "StandardAssembler::prepareContextFace: non-Product test space DOF count mismatch");
    }
    if (trial_is_product) {
        FE_CHECK_ARG(trial_space.field_type() == FieldType::Vector,
                     "StandardAssembler::prepareContextFace: ProductSpace trial space must be vector-valued");
        FE_CHECK_ARG(trial_space.value_dimension() > 0,
                     "StandardAssembler::prepareContextFace: invalid trial space value dimension");
        FE_CHECK_ARG(n_trial_dofs ==
                         static_cast<LocalIndex>(
                             n_trial_scalar_dofs * static_cast<LocalIndex>(trial_space.value_dimension())),
                     "StandardAssembler::prepareContextFace: trial ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_trial_dofs == n_trial_scalar_dofs,
                     "StandardAssembler::prepareContextFace: non-Product trial space DOF count mismatch");
    }
    const bool need_basis_hessians = hasFlag(required_data, RequiredData::BasisHessians);
    const bool need_basis_curls = hasFlag(required_data, RequiredData::BasisCurls);
    const bool need_basis_divergences = hasFlag(required_data, RequiredData::BasisDivergences);

    const auto& test_basis = test_element.basis();
    const auto& trial_basis = trial_element.basis();
    const bool test_is_vector_basis = test_basis.is_vector_valued();
    const bool trial_is_vector_basis = trial_basis.is_vector_valued();

    const bool need_test_vector_values =
        test_is_vector_basis &&
        (hasFlag(required_data, RequiredData::BasisValues) ||
         hasFlag(required_data, RequiredData::SolutionValues) ||
         required_data == RequiredData::None);
    const bool need_trial_vector_values =
        trial_is_vector_basis &&
        (hasFlag(required_data, RequiredData::BasisValues) ||
         hasFlag(required_data, RequiredData::SolutionValues) ||
         required_data == RequiredData::None);

    // 5. Get cell node coordinates from mesh
    mesh.getCellCoordinates(cell_id, cell_coords_);
    const auto n_nodes = cell_coords_.size();

    // Convert to math::Vector format
    std::vector<math::Vector<Real, 3>> node_coords(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            cell_coords_[i][0], cell_coords_[i][1], cell_coords_[i][2]};
    }

    // 6. Create geometry mapping
    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = defaultGeometryOrder(cell_type);
    map_request.use_affine = (map_request.geometry_order <= 1);

    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    // 7. Resize scratch storage
    scratch_quad_points_.resize(n_qpts);
    scratch_quad_weights_.resize(n_qpts);
    scratch_phys_points_.resize(n_qpts);
    scratch_jacobians_.resize(n_qpts);
    scratch_inv_jacobians_.resize(n_qpts);
    scratch_jac_dets_.resize(n_qpts);
    scratch_integration_weights_.resize(n_qpts);
    scratch_normals_.resize(n_qpts);

    const auto test_basis_size = static_cast<std::size_t>(n_test_dofs * n_qpts);
    const auto trial_basis_size = static_cast<std::size_t>(n_trial_dofs * n_qpts);
    if (test_is_vector_basis) {
        scratch_basis_values_.clear();
        scratch_ref_gradients_.clear();
        scratch_phys_gradients_.clear();
        scratch_ref_hessians_.clear();
        scratch_phys_hessians_.clear();

        if (need_test_vector_values) {
            scratch_basis_vector_values_.resize(test_basis_size);
        } else {
            scratch_basis_vector_values_.clear();
        }
        if (need_basis_curls) {
            scratch_basis_curls_.resize(test_basis_size);
        } else {
            scratch_basis_curls_.clear();
        }
        if (need_basis_divergences) {
            scratch_basis_divergences_.resize(test_basis_size);
        } else {
            scratch_basis_divergences_.clear();
        }
    } else {
        scratch_basis_vector_values_.clear();
        scratch_basis_curls_.clear();
        scratch_basis_divergences_.clear();

        scratch_basis_values_.resize(test_basis_size);
        scratch_ref_gradients_.resize(test_basis_size);
        scratch_phys_gradients_.resize(test_basis_size);
        if (need_basis_hessians) {
            scratch_ref_hessians_.resize(test_basis_size);
            scratch_phys_hessians_.resize(test_basis_size);
        } else {
            scratch_ref_hessians_.clear();
            scratch_phys_hessians_.clear();
        }
    }

    std::vector<Real> trial_basis_values;
    std::vector<AssemblyContext::Vector3D> trial_ref_gradients;
    std::vector<AssemblyContext::Vector3D> trial_phys_gradients;
    std::vector<AssemblyContext::Vector3D> trial_basis_vector_values;
    std::vector<AssemblyContext::Vector3D> trial_basis_curls;
    std::vector<Real> trial_basis_divergences;
    std::vector<AssemblyContext::Matrix3x3> trial_ref_hessians;
    std::vector<AssemblyContext::Matrix3x3> trial_phys_hessians;

    if (&test_space != &trial_space) {
        if (trial_is_vector_basis) {
            trial_basis_values.clear();
            trial_ref_gradients.clear();
            trial_phys_gradients.clear();
            trial_ref_hessians.clear();
            trial_phys_hessians.clear();

            if (need_trial_vector_values) {
                trial_basis_vector_values.resize(trial_basis_size);
            }
            if (need_basis_curls) {
                trial_basis_curls.resize(trial_basis_size);
            }
            if (need_basis_divergences) {
                trial_basis_divergences.resize(trial_basis_size);
            }
        } else {
            trial_basis_vector_values.clear();
            trial_basis_curls.clear();
            trial_basis_divergences.clear();

            trial_basis_values.resize(trial_basis_size);
            trial_ref_gradients.resize(trial_basis_size);
            trial_phys_gradients.resize(trial_basis_size);
            if (need_basis_hessians) {
                trial_ref_hessians.resize(trial_basis_size);
                trial_phys_hessians.resize(trial_basis_size);
            } else {
                trial_ref_hessians.clear();
                trial_phys_hessians.clear();
            }
        }
    }

    // 8. Map face quadrature points to element reference coordinates and compute normals/weights
    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    const AssemblyContext::Vector3D n_ref = computeFaceNormal(local_face_id, cell_type, dim);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& qpt = quad_points[q];
        scratch_quad_weights_[q] = quad_weights[q];

        // Convert quadrature point to facet-local coordinates expected by ElementTransform
        math::Vector<Real, 3> facet_coords{};
        if (face_type == ElementType::Line2) {
            // Line quadrature is on [-1,1]; facet parameterization uses t in [0,1]
            Real t = (qpt[0] + Real(1)) * Real(0.5);
            if (!align_facet_to_reference.empty() && align_facet_to_reference.size() == 2) {
                const Real w_ref0 = Real(1) - t;
                const Real w_ref1 = t;
                const std::array<Real, 2> w_ref{w_ref0, w_ref1};
                std::array<Real, 2> w_local{0.0, 0.0};
                for (std::size_t j = 0; j < 2; ++j) {
                    const auto src = static_cast<std::size_t>(align_facet_to_reference[j]);
                    w_local[j] = w_ref[src];
                }
                t = w_local[1];
            }
            facet_coords = math::Vector<Real, 3>{t, Real(0), Real(0)};
        } else if (face_type == ElementType::Quad4) {
            // Quad quadrature is on [-1,1]^2; facet parameterization uses (s,t) in [0,1]^2
            facet_coords = math::Vector<Real, 3>{
                (qpt[0] + Real(1)) * Real(0.5),
                (qpt[1] + Real(1)) * Real(0.5),
                Real(0)};
        } else {
            // Triangle quadrature uses reference simplex coordinates (0<=x,y, x+y<=1)
            const Real x = qpt[0];
            const Real y = qpt[1];
            facet_coords = math::Vector<Real, 3>{x, y, Real(0)};

            // For interior faces, the plus-side element may have a different local face
            // vertex ordering. The weak-form evaluation assumes that q is the same
            // physical point on both sides, so we optionally permute barycentric weights
            // to align this face parameterization to a reference orientation.
            if (!align_facet_to_reference.empty()) {
                const Real w_ref0 = Real(1) - x - y;
                const Real w_ref1 = x;
                const Real w_ref2 = y;
                const std::array<Real, 3> w_ref{w_ref0, w_ref1, w_ref2};

                if (align_facet_to_reference.size() == 3) {
                    std::array<Real, 3> w_local{0.0, 0.0, 0.0};
                    for (std::size_t j = 0; j < 3; ++j) {
                        const auto src = static_cast<std::size_t>(align_facet_to_reference[j]);
                        w_local[j] = w_ref[src];
                    }
                    // Convert back to (x,y) for the local face ordering.
                    facet_coords = math::Vector<Real, 3>{w_local[1], w_local[2], Real(0)};
                }
            }
        }

        // Map to the cell reference coordinates on the requested face
        const math::Vector<Real, 3> xi = elements::ElementTransform::facet_to_reference(
            cell_type, static_cast<int>(local_face_id), facet_coords);

        scratch_quad_points_[q] = {xi[0], xi[1], xi[2]};

        // Compute physical point and mapping Jacobians
        const auto x_phys = mapping->map_to_physical(xi);
        scratch_phys_points_[q] = {x_phys[0], x_phys[1], x_phys[2]};

        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch_jacobians_[q][i][j] = J(i, j);
                scratch_inv_jacobians_[q][i][j] = J_inv(i, j);
            }
        }
        scratch_jac_dets_[q] = det_J;

        Real surface_measure;
        AssemblyContext::Vector3D n_phys;
        computeSurfaceMeasureAndNormal(n_ref, scratch_inv_jacobians_[q], det_J, dim,
                                       surface_measure, n_phys);

        scratch_integration_weights_[q] = quad_weights[q] * surface_measure;
        scratch_normals_[q] = n_phys;
    }

    // 9. Evaluate basis functions at face quadrature points
    std::vector<Real> scalar_values_at_pt;
    std::vector<basis::Gradient> scalar_gradients_at_pt;
    std::vector<basis::Hessian> scalar_hessians_at_pt;

    std::vector<math::Vector<Real, 3>> vec_values_at_pt;
    std::vector<math::Vector<Real, 3>> vec_curls_at_pt;
    std::vector<Real> vec_divs_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch_quad_points_[q][0],
            scratch_quad_points_[q][1],
            scratch_quad_points_[q][2]};

        const auto& J = scratch_jacobians_[q];
        const auto& J_inv = scratch_inv_jacobians_[q];
        const Real det_J = scratch_jac_dets_[q];

        std::array<AssemblyContext::Matrix3x3, 3> d2xi_dx2{};
        if (need_basis_hessians) {
            const auto map_hess = mapping->mapping_hessian(xi);
            for (int a = 0; a < dim; ++a) {
                for (int i = 0; i < dim; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        Real sum = 0.0;
                        for (int m = 0; m < dim; ++m) {
                            for (int p = 0; p < dim; ++p) {
                                for (int r = 0; r < dim; ++r) {
                                    sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(m)] *
                                           map_hess[static_cast<std::size_t>(m)](
                                               static_cast<std::size_t>(p), static_cast<std::size_t>(r)) *
                                           J_inv[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)] *
                                           J_inv[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)];
                                }
                            }
                        }
                        d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = -sum;
                    }
                }
            }
        }

        if (test_is_vector_basis) {
            if (need_test_vector_values) {
                test_basis.evaluate_vector_values(xi, vec_values_at_pt);
            }
            if (need_basis_curls) {
                test_basis.evaluate_curl(xi, vec_curls_at_pt);
            }
            if (need_basis_divergences) {
                test_basis.evaluate_divergence(xi, vec_divs_at_pt);
            }

            const auto cont = test_space.continuity();
            for (LocalIndex i = 0; i < n_test_dofs; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);

                if (need_test_vector_values) {
                    const auto& vref = vec_values_at_pt[static_cast<std::size_t>(i)];
                    AssemblyContext::Vector3D vphys{0.0, 0.0, 0.0};
                    if (cont == Continuity::H_curl) {
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                vphys[static_cast<std::size_t>(r)] +=
                                    J_inv[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] *
                                    vref[static_cast<std::size_t>(c)];
                            }
                        }
                    } else { // H_div
                        const Real inv_det = Real(1) / det_J;
                        for (int r = 0; r < 3; ++r) {
                            Real sum = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                       vref[static_cast<std::size_t>(c)];
                            }
                            vphys[static_cast<std::size_t>(r)] = inv_det * sum;
                        }
                    }
                    scratch_basis_vector_values_[idx] = vphys;
                }

                if (need_basis_curls) {
                    const auto& cref = vec_curls_at_pt[static_cast<std::size_t>(i)];
                    AssemblyContext::Vector3D cphys{0.0, 0.0, 0.0};
                    const Real inv_det = Real(1) / det_J;
                    for (int r = 0; r < 3; ++r) {
                        Real sum = 0.0;
                        for (int c = 0; c < 3; ++c) {
                            sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                   cref[static_cast<std::size_t>(c)];
                        }
                        cphys[static_cast<std::size_t>(r)] = inv_det * sum;
                    }
                    scratch_basis_curls_[idx] = cphys;
                }

                if (need_basis_divergences) {
                    scratch_basis_divergences_[idx] =
                        vec_divs_at_pt[static_cast<std::size_t>(i)] / det_J;
                }
            }
        } else {
            test_basis.evaluate_values(xi, scalar_values_at_pt);
            test_basis.evaluate_gradients(xi, scalar_gradients_at_pt);
            if (need_basis_hessians) {
                test_basis.evaluate_hessians(xi, scalar_hessians_at_pt);
            }

            for (LocalIndex i = 0; i < n_test_dofs; ++i) {
                const LocalIndex si = test_is_product ? static_cast<LocalIndex>(i % n_test_scalar_dofs) : i;
                const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
                scratch_basis_values_[idx] = scalar_values_at_pt[static_cast<std::size_t>(si)];
                scratch_ref_gradients_[idx] = {
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][0],
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][1],
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][2]};

                const auto& grad_ref = scratch_ref_gradients_[idx];
                AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                    }
                }
                scratch_phys_gradients_[idx] = grad_phys;

                if (need_basis_hessians) {
                    AssemblyContext::Matrix3x3 H_ref{};
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                scalar_hessians_at_pt[static_cast<std::size_t>(si)](static_cast<std::size_t>(r),
                                                                                     static_cast<std::size_t>(c));
                        }
                    }
                    scratch_ref_hessians_[idx] = H_ref;

                    AssemblyContext::Matrix3x3 H_phys{};
                    for (int r = 0; r < dim; ++r) {
                        for (int c = 0; c < dim; ++c) {
                            Real sum = 0.0;
                            for (int a = 0; a < dim; ++a) {
                                for (int b = 0; b < dim; ++b) {
                                    sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                           H_ref[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                           J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                }
                            }
                            for (int a = 0; a < dim; ++a) {
                                sum += grad_ref[static_cast<std::size_t>(a)] *
                                       d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            }
                            H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                        }
                    }
                    scratch_phys_hessians_[idx] = H_phys;
                }
            }
        }

        if (&test_space != &trial_space) {
            if (trial_is_vector_basis) {
                if (need_trial_vector_values) {
                    trial_basis.evaluate_vector_values(xi, vec_values_at_pt);
                }
                if (need_basis_curls) {
                    trial_basis.evaluate_curl(xi, vec_curls_at_pt);
                }
                if (need_basis_divergences) {
                    trial_basis.evaluate_divergence(xi, vec_divs_at_pt);
                }

                const auto cont = trial_space.continuity();
                for (LocalIndex j = 0; j < n_trial_dofs; ++j) {
                    const std::size_t idx = static_cast<std::size_t>(j * n_qpts + q);

                    if (need_trial_vector_values) {
                        const auto& vref = vec_values_at_pt[static_cast<std::size_t>(j)];
                        AssemblyContext::Vector3D vphys{0.0, 0.0, 0.0};
                        if (cont == Continuity::H_curl) {
                            for (int r = 0; r < dim; ++r) {
                                for (int c = 0; c < dim; ++c) {
                                    vphys[static_cast<std::size_t>(r)] +=
                                        J_inv[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] *
                                        vref[static_cast<std::size_t>(c)];
                                }
                            }
                        } else { // H_div
                            const Real inv_det = Real(1) / det_J;
                            for (int r = 0; r < 3; ++r) {
                                Real sum = 0.0;
                                for (int c = 0; c < 3; ++c) {
                                    sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                           vref[static_cast<std::size_t>(c)];
                                }
                                vphys[static_cast<std::size_t>(r)] = inv_det * sum;
                            }
                        }
                        trial_basis_vector_values[idx] = vphys;
                    }

                    if (need_basis_curls) {
                        const auto& cref = vec_curls_at_pt[static_cast<std::size_t>(j)];
                        AssemblyContext::Vector3D cphys{0.0, 0.0, 0.0};
                        const Real inv_det = Real(1) / det_J;
                        for (int r = 0; r < 3; ++r) {
                            Real sum = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                       cref[static_cast<std::size_t>(c)];
                            }
                            cphys[static_cast<std::size_t>(r)] = inv_det * sum;
                        }
                        trial_basis_curls[idx] = cphys;
                    }

                    if (need_basis_divergences) {
                        trial_basis_divergences[idx] =
                            vec_divs_at_pt[static_cast<std::size_t>(j)] / det_J;
                    }
                }
            } else {
                trial_basis.evaluate_values(xi, scalar_values_at_pt);
                trial_basis.evaluate_gradients(xi, scalar_gradients_at_pt);
                if (need_basis_hessians) {
                    trial_basis.evaluate_hessians(xi, scalar_hessians_at_pt);
                }

                for (LocalIndex j = 0; j < n_trial_dofs; ++j) {
                    const LocalIndex sj = trial_is_product ? static_cast<LocalIndex>(j % n_trial_scalar_dofs) : j;
                    const std::size_t idx = static_cast<std::size_t>(j * n_qpts + q);
                    trial_basis_values[idx] = scalar_values_at_pt[static_cast<std::size_t>(sj)];
                    trial_ref_gradients[idx] = {
                        scalar_gradients_at_pt[static_cast<std::size_t>(sj)][0],
                        scalar_gradients_at_pt[static_cast<std::size_t>(sj)][1],
                        scalar_gradients_at_pt[static_cast<std::size_t>(sj)][2]};

                    const auto& grad_ref = trial_ref_gradients[idx];
                    AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

                    for (int d1 = 0; d1 < dim; ++d1) {
                        for (int d2 = 0; d2 < dim; ++d2) {
                            grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                        }
                    }
                    trial_phys_gradients[idx] = grad_phys;

                    if (need_basis_hessians) {
                        AssemblyContext::Matrix3x3 H_ref{};
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                    scalar_hessians_at_pt[static_cast<std::size_t>(sj)](static_cast<std::size_t>(r),
                                                                                         static_cast<std::size_t>(c));
                            }
                        }
                        trial_ref_hessians[idx] = H_ref;

                        AssemblyContext::Matrix3x3 H_phys{};
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                Real sum = 0.0;
                                for (int a = 0; a < dim; ++a) {
                                    for (int b = 0; b < dim; ++b) {
                                        sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                               H_ref[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                               J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                    }
                                }
                                for (int a = 0; a < dim; ++a) {
                                    sum += grad_ref[static_cast<std::size_t>(a)] *
                                           d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                }
                                H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                            }
                        }
                        trial_phys_hessians[idx] = H_phys;
                    }
                }
            }
        }
    }

    // 10. Configure face context and set computed data
    context.configureFace(face_id, cell_id, local_face_id, test_space, trial_space, required_data, type);
    context.setCellDomainId(mesh.getCellDomainId(cell_id));
    context.setQuadratureData(scratch_quad_points_, scratch_quad_weights_);
    context.setPhysicalPoints(scratch_phys_points_);
    context.setJacobianData(scratch_jacobians_, scratch_inv_jacobians_, scratch_jac_dets_);
    context.setIntegrationWeights(scratch_integration_weights_);

    // Basis data
    if (test_is_vector_basis) {
        context.setTestVectorBasisValues(n_test_dofs,
                                         need_test_vector_values
                                             ? std::span<const AssemblyContext::Vector3D>(scratch_basis_vector_values_)
                                             : std::span<const AssemblyContext::Vector3D>{});
        if (need_basis_curls) {
            context.setTestBasisCurls(n_test_dofs, std::span<const AssemblyContext::Vector3D>(scratch_basis_curls_));
        }
        if (need_basis_divergences) {
            context.setTestBasisDivergences(n_test_dofs, std::span<const Real>(scratch_basis_divergences_));
        }
    } else {
        context.setTestBasisData(n_test_dofs, scratch_basis_values_, scratch_ref_gradients_);
        if (need_basis_hessians) {
            context.setTestBasisHessians(n_test_dofs, scratch_ref_hessians_);
        }
    }

    if (&test_space != &trial_space) {
        if (trial_is_vector_basis) {
            context.setTrialVectorBasisValues(n_trial_dofs,
                                              need_trial_vector_values
                                                  ? std::span<const AssemblyContext::Vector3D>(trial_basis_vector_values)
                                                  : std::span<const AssemblyContext::Vector3D>{});
            if (need_basis_curls) {
                context.setTrialBasisCurls(n_trial_dofs, std::span<const AssemblyContext::Vector3D>(trial_basis_curls));
            }
            if (need_basis_divergences) {
                context.setTrialBasisDivergences(n_trial_dofs, std::span<const Real>(trial_basis_divergences));
            }
        } else {
            context.setTrialBasisData(n_trial_dofs, trial_basis_values, trial_ref_gradients);
            if (need_basis_hessians) {
                context.setTrialBasisHessians(n_trial_dofs, trial_ref_hessians);
            }
        }
    }

    const auto test_phys = test_is_vector_basis
                               ? std::span<const AssemblyContext::Vector3D>{}
                               : std::span<const AssemblyContext::Vector3D>(scratch_phys_gradients_);
    std::span<const AssemblyContext::Vector3D> trial_phys{};
    if (&test_space != &trial_space) {
        trial_phys = trial_is_vector_basis ? std::span<const AssemblyContext::Vector3D>{}
                                           : std::span<const AssemblyContext::Vector3D>(trial_phys_gradients);
    } else {
        trial_phys = test_phys;
    }
    context.setPhysicalGradients(test_phys, trial_phys);
    context.setNormals(scratch_normals_);

    if (need_basis_hessians) {
        const auto test_H = test_is_vector_basis
                                ? std::span<const AssemblyContext::Matrix3x3>{}
                                : std::span<const AssemblyContext::Matrix3x3>(scratch_phys_hessians_);
        std::span<const AssemblyContext::Matrix3x3> trial_H{};
        if (&test_space != &trial_space) {
            trial_H = trial_is_vector_basis ? std::span<const AssemblyContext::Matrix3x3>{}
                                            : std::span<const AssemblyContext::Matrix3x3>(trial_phys_hessians);
        } else {
            trial_H = test_H;
        }
        context.setPhysicalHessians(test_H, trial_H);
    }

    if (hasFlag(required_data, RequiredData::EntityMeasures)) {
        Real facet_area = 0.0;
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            facet_area += scratch_integration_weights_[static_cast<std::size_t>(q)];
        }

        Real h = 0.0;
        for (std::size_t a = 0; a < n_nodes; ++a) {
            for (std::size_t b = a + 1; b < n_nodes; ++b) {
                const Real dx = node_coords[a][0] - node_coords[b][0];
                const Real dy = node_coords[a][1] - node_coords[b][1];
                const Real dz = node_coords[a][2] - node_coords[b][2];
                const Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > h) h = dist;
            }
        }

        context.setEntityMeasures(h, /*cell_volume=*/0.0, facet_area);
    }
}

AssemblyContext::Vector3D StandardAssembler::computeFaceNormal(
    LocalIndex local_face_id,
    ElementType cell_type,
    int dim) const
{
    (void)dim;
    auto n = elements::ElementTransform::reference_facet_normal(
        cell_type, static_cast<int>(local_face_id));
    // NOTE: Face quadrature points/weights are defined on canonical face
    // reference domains (e.g., Triangle3 has area 0.5). For some reference
    // element facets (e.g., the oblique tetra face), the mapping from the
    // canonical face to the element-reference face carries a constant metric
    // factor. Scaling the reference normal by that factor yields correct
    // surface measures via the cofactor (det(J) * J^{-T}) formula.
    if ((cell_type == ElementType::Tetra4 || cell_type == ElementType::Tetra10) && local_face_id == 2) {
        const Real scale = std::sqrt(Real(3));
        n[0] *= scale;
        n[1] *= scale;
        n[2] *= scale;
    }
    return {n[0], n[1], n[2]};
}

void StandardAssembler::computeSurfaceMeasureAndNormal(
    const AssemblyContext::Vector3D& n_ref,
    const AssemblyContext::Matrix3x3& J_inv,
    Real det_J,
    int dim,
    Real& surface_measure,
    AssemblyContext::Vector3D& n_phys) const
{
    // Compute the transformation J^{-T} * n_ref.
    //
    // Mathematical derivation:
    // For a mapping x = F(xi) from reference to physical coordinates, the
    // Jacobian is J = dx/dxi. The transformation of area elements is:
    //
    //   dS_phys = ||cof(J) * n_ref|| * dS_ref
    //
    // where cof(J) is the cofactor matrix of J. Using the identity
    // cof(J) = det(J) * J^{-T}, we have:
    //
    //   dS_phys = ||det(J) * J^{-T} * n_ref|| * dS_ref
    //           = |det(J)| * ||J^{-T} * n_ref|| * dS_ref
    //
    // The physical normal direction (unnormalized) is given by:
    //   n_phys_unnorm = J^{-T} * n_ref = (J^{-1})^T * n_ref
    //
    // To apply J^{-T} = (J^{-1})^T to a vector v:
    //   (J^{-T} * v)_i = sum_k J^{-1}_{ki} * v_k
    //
    // This is the transpose action: column i of J^{-1} dotted with v.

    // Compute J^{-T} * n_ref
    AssemblyContext::Vector3D Jit_n = {0.0, 0.0, 0.0};
    for (int i = 0; i < dim; ++i) {
        for (int k = 0; k < dim; ++k) {
            // J^{-T}_{ik} = J^{-1}_{ki}
            Jit_n[i] += J_inv[k][i] * n_ref[k];
        }
    }

    // Compute the norm of J^{-T} * n_ref
    Real norm_Jit_n = 0.0;
    for (int i = 0; i < dim; ++i) {
        norm_Jit_n += Jit_n[i] * Jit_n[i];
    }
    norm_Jit_n = std::sqrt(norm_Jit_n);

    // Surface measure = ||J^{-T} * n_ref|| * |det(J)|
    surface_measure = norm_Jit_n * std::abs(det_J);

    // Physical unit normal = normalize(J^{-T} * n_ref)
    constexpr Real tol = 1e-14;
    if (norm_Jit_n > tol) {
        n_phys[0] = Jit_n[0] / norm_Jit_n;
        n_phys[1] = Jit_n[1] / norm_Jit_n;
        n_phys[2] = Jit_n[2] / norm_Jit_n;
    } else {
        // Degenerate case: fall back to reference normal
        // This should not happen for valid meshes
        n_phys = n_ref;
    }
}

const StandardAssembler::FaceTransform& StandardAssembler::getFaceTransform(
    BasisType basis_type,
    Continuity continuity,
    ElementType face_type,
    int poly_order,
    const spaces::OrientationManager::FaceOrientation& orientation,
    LocalIndex expected_size)
{
    FaceTransformKey key{};
    key.basis_type = basis_type;
    key.continuity = continuity;
    key.face_type = face_type;
    key.poly_order = poly_order;
    key.sign = orientation.sign;
    key.n_verts = static_cast<std::uint8_t>(std::min<std::size_t>(orientation.vertex_perm.size(), 4u));
    for (std::size_t i = 0; i < static_cast<std::size_t>(key.n_verts); ++i) {
        key.vertex_perm[i] = orientation.vertex_perm[i];
    }

    auto it = face_transform_cache_.find(key);
    if (it != face_transform_cache_.end()) {
        FE_THROW_IF(it->second.n != expected_size, FEException,
                    "StandardAssembler::getFaceTransform: cached face transform size mismatch");
        return it->second;
    }

    FaceTransform tf{};
    tf.n = expected_size;
    const auto nn = static_cast<std::size_t>(std::max<LocalIndex>(0, expected_size));
    if (expected_size <= 0) {
        auto [ins, inserted] = face_transform_cache_.emplace(key, std::move(tf));
        (void)inserted;
        return ins->second;
    }

    FE_THROW_IF(key.n_verts == 0u, FEException,
                "StandardAssembler::getFaceTransform: missing face vertex permutation");

    std::vector<Real> O(nn * nn, 0.0);
    std::vector<Real> work;
    std::vector<Real> inv;

    scratch_orient_in_.assign(nn, 0.0);

    for (LocalIndex col = 0; col < expected_size; ++col) {
        std::fill(scratch_orient_in_.begin(), scratch_orient_in_.end(), 0.0);
        scratch_orient_in_[static_cast<std::size_t>(col)] = 1.0;

        std::vector<Real> oriented;
        if (continuity == Continuity::H_div) {
            if (face_type == ElementType::Triangle3) {
                oriented = spaces::OrientationManager::orient_triangle_face_dofs(
                    scratch_orient_in_, orientation, poly_order);
            } else if (face_type == ElementType::Quad4) {
                oriented = spaces::OrientationManager::orient_quad_face_dofs(
                    scratch_orient_in_, orientation, poly_order);
            } else {
                FE_THROW(FEException, "StandardAssembler::getFaceTransform: unsupported face type for H(div)");
            }
        } else if (continuity == Continuity::H_curl) {
            if (face_type == ElementType::Triangle3) {
                oriented = spaces::OrientationManager::orient_hcurl_triangle_face_dofs(
                    scratch_orient_in_, orientation, poly_order);
            } else if (face_type == ElementType::Quad4) {
                oriented = spaces::OrientationManager::orient_hcurl_quad_face_dofs(
                    scratch_orient_in_, orientation, poly_order);
            } else {
                FE_THROW(FEException, "StandardAssembler::getFaceTransform: unsupported face type for H(curl)");
            }
        } else {
            FE_THROW(FEException, "StandardAssembler::getFaceTransform: unsupported continuity");
        }

        FE_THROW_IF(oriented.size() != nn, FEException,
                    "StandardAssembler::getFaceTransform: oriented face DOF vector size mismatch");

        for (std::size_t row = 0; row < nn; ++row) {
            O[row * nn + static_cast<std::size_t>(col)] = oriented[row];
        }
    }

    const bool ok = invertDenseMatrix(O, expected_size, inv, work);
    FE_THROW_IF(!ok, FEException, "StandardAssembler::getFaceTransform: singular face orientation matrix");
    tf.P = std::move(inv);

    tf.PT.resize(nn * nn, 0.0);
    for (std::size_t i = 0; i < nn; ++i) {
        for (std::size_t j = 0; j < nn; ++j) {
            tf.PT[i * nn + j] = tf.P[j * nn + i];
        }
    }

    auto [ins, inserted] = face_transform_cache_.emplace(key, std::move(tf));
    (void)inserted;
    return ins->second;
}

void StandardAssembler::applyVectorBasisGlobalToLocal(
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const spaces::FunctionSpace& space,
    std::span<Real> coeffs)
{
    if (coeffs.empty()) {
        return;
    }
    const auto continuity = space.continuity();
    if (continuity != Continuity::H_curl && continuity != Continuity::H_div) {
        return;
    }

    // Standalone/unit-test assembly may provide only a DofMap. For a single-cell mesh, the
    // canonical orientation is sufficient and we can treat the orientation transform as identity.
    if (dof_handler_ == nullptr || !dof_handler_->hasCellOrientations()) {
        FE_THROW_IF(mesh.numCells() > 1, FEException,
                    "StandardAssembler: H(curl)/H(div) coefficient orientation requires a DofHandler with cell orientations");
        return;
    }

    const ElementType cell_type = mesh.getCellType(cell_id);
    const auto& element = getElement(space, cell_id, cell_type);
    const auto& basis = element.basis();
    FE_THROW_IF(!basis.is_vector_valued(), FEException,
                "StandardAssembler::applyVectorBasisGlobalToLocal: expected vector-valued basis for H(curl)/H(div) space");

    const auto* vbf = dynamic_cast<const basis::VectorBasisFunction*>(&basis);
    FE_THROW_IF(vbf == nullptr, FEException,
                "StandardAssembler::applyVectorBasisGlobalToLocal: vector-valued basis is not a VectorBasisFunction");

    const auto associations = vbf->dof_associations();
    FE_THROW_IF(associations.size() != coeffs.size(), FEException,
                "StandardAssembler::applyVectorBasisGlobalToLocal: DOF association size mismatch");

    const auto edge_orients = dof_handler_->cellEdgeOrientations(cell_id);
    const auto face_orients = dof_handler_->cellFaceOrientations(cell_id);

    bool needs_edges = false;
    bool needs_faces = false;
    for (const auto& a : associations) {
        if (a.entity_type == basis::DofEntity::Edge) needs_edges = true;
        if (a.entity_type == basis::DofEntity::Face) needs_faces = true;
    }

    if (needs_edges) {
        FE_THROW_IF(edge_orients.empty(), FEException,
                    "StandardAssembler::applyVectorBasisGlobalToLocal: missing cell edge orientations");
    }
    if (needs_faces) {
        FE_THROW_IF(face_orients.empty(), FEException,
                    "StandardAssembler::applyVectorBasisGlobalToLocal: missing cell face orientations");
    }

    // Edge DOFs: apply sign based on edge orientation. (Any needed reordering of edge modes is
    // already handled by DofMap canonical edge ordering.)
    if (needs_edges) {
        for (LocalIndex i = 0; i < static_cast<LocalIndex>(associations.size()); ++i) {
            const auto& a = associations[static_cast<std::size_t>(i)];
            if (a.entity_type != basis::DofEntity::Edge) continue;
            const int e = a.entity_id;
            FE_THROW_IF(e < 0 || static_cast<std::size_t>(e) >= edge_orients.size(), FEException,
                        "StandardAssembler::applyVectorBasisGlobalToLocal: edge id out of range in DOF associations");
            const int sign = edge_orients[static_cast<std::size_t>(e)];
            if (sign < 0) {
                coeffs[static_cast<std::size_t>(i)] = -coeffs[static_cast<std::size_t>(i)];
            }
        }
    }

    // Face DOFs: apply dense (possibly mixing) transform P = O^{-1}.
    if (needs_faces) {
        const elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
        const auto n_faces = static_cast<int>(ref.num_faces());
        FE_THROW_IF(n_faces <= 0, FEException,
                    "StandardAssembler::applyVectorBasisGlobalToLocal: vector basis has face DOFs but cell has no faces");
        FE_THROW_IF(face_orients.size() < static_cast<std::size_t>(n_faces), FEException,
                    "StandardAssembler::applyVectorBasisGlobalToLocal: face orientation span too small for cell");

        std::vector<std::vector<std::pair<int, LocalIndex>>> face_pairs(static_cast<std::size_t>(n_faces));
        for (LocalIndex i = 0; i < static_cast<LocalIndex>(associations.size()); ++i) {
            const auto& a = associations[static_cast<std::size_t>(i)];
            if (a.entity_type != basis::DofEntity::Face) continue;
            const int f = a.entity_id;
            FE_THROW_IF(f < 0 || f >= n_faces, FEException,
                        "StandardAssembler::applyVectorBasisGlobalToLocal: face id out of range in DOF associations");
            face_pairs[static_cast<std::size_t>(f)].push_back({a.moment_index, i});
        }

        for (int f = 0; f < n_faces; ++f) {
            auto& pairs = face_pairs[static_cast<std::size_t>(f)];
            if (pairs.empty()) continue;

            std::sort(pairs.begin(), pairs.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

            const LocalIndex n_face_dofs = static_cast<LocalIndex>(pairs.size());
            const auto& fn = ref.face_nodes(static_cast<std::size_t>(f));
            ElementType face_type = ElementType::Unknown;
            if (fn.size() == 3u) {
                face_type = ElementType::Triangle3;
            } else if (fn.size() == 4u) {
                face_type = ElementType::Quad4;
            } else {
                FE_THROW(FEException, "StandardAssembler::applyVectorBasisGlobalToLocal: unsupported face topology");
            }

            const auto& orient = face_orients[static_cast<std::size_t>(f)];
            const auto& tf = getFaceTransform(basis.basis_type(), continuity, face_type,
                                              element.polynomial_order(), orient, n_face_dofs);

            const auto nn = static_cast<std::size_t>(n_face_dofs);
            scratch_orient_in_.assign(nn, 0.0);
            scratch_orient_out_.assign(nn, 0.0);

            for (std::size_t k = 0; k < nn; ++k) {
                const auto dof = pairs[k].second;
                scratch_orient_in_[k] = coeffs[static_cast<std::size_t>(dof)];
            }

            for (std::size_t r = 0; r < nn; ++r) {
                Real sum = 0.0;
                const std::size_t base = r * nn;
                for (std::size_t c = 0; c < nn; ++c) {
                    sum += tf.P[base + c] * scratch_orient_in_[c];
                }
                scratch_orient_out_[r] = sum;
            }

            for (std::size_t k = 0; k < nn; ++k) {
                const auto dof = pairs[k].second;
                coeffs[static_cast<std::size_t>(dof)] = scratch_orient_out_[k];
            }
        }
    }
}

void StandardAssembler::applyVectorBasisOutputOrientation(
    const IMeshAccess& mesh,
    GlobalIndex test_cell_id,
    const spaces::FunctionSpace& test_space,
    GlobalIndex trial_cell_id,
    const spaces::FunctionSpace& trial_space,
    KernelOutput& output)
{
    const bool need_test = (test_space.continuity() == Continuity::H_curl || test_space.continuity() == Continuity::H_div);
    const bool need_trial = (trial_space.continuity() == Continuity::H_curl || trial_space.continuity() == Continuity::H_div);
    if (!need_test && !need_trial) {
        return;
    }

    // Standalone/unit-test assembly may provide only a DofMap. For a single-cell mesh, the
    // canonical orientation is sufficient and we can treat the output transform as identity.
    if (dof_handler_ == nullptr || !dof_handler_->hasCellOrientations()) {
        FE_THROW_IF(mesh.numCells() > 1, FEException,
                    "StandardAssembler: H(curl)/H(div) output orientation requires a DofHandler with cell orientations");
        return;
    }

    struct Blocks {
        std::vector<std::pair<LocalIndex, int>> edge_dofs; // (dof_index, sign)
        struct FaceBlock {
            std::vector<LocalIndex> dofs; // in moment-index order
            const FaceTransform* tf{nullptr};
        };
        std::vector<FaceBlock> faces;
    };

    const auto build_blocks = [&](GlobalIndex cell_id,
                                  const spaces::FunctionSpace& space,
                                  LocalIndex expected_dofs) -> std::optional<Blocks> {
        const auto continuity = space.continuity();
        if (continuity != Continuity::H_curl && continuity != Continuity::H_div) {
            return std::nullopt;
        }
        FE_THROW_IF(expected_dofs < 0, FEException,
                    "StandardAssembler::applyVectorBasisOutputOrientation: negative expected DOF count");

        const ElementType cell_type = mesh.getCellType(cell_id);
        const auto& element = getElement(space, cell_id, cell_type);
        const auto& basis = element.basis();
        FE_THROW_IF(!basis.is_vector_valued(), FEException,
                    "StandardAssembler::applyVectorBasisOutputOrientation: expected vector-valued basis for H(curl)/H(div) space");

        const auto* vbf = dynamic_cast<const basis::VectorBasisFunction*>(&basis);
        FE_THROW_IF(vbf == nullptr, FEException,
                    "StandardAssembler::applyVectorBasisOutputOrientation: vector-valued basis is not a VectorBasisFunction");

        const auto associations = vbf->dof_associations();
        FE_THROW_IF(static_cast<LocalIndex>(associations.size()) != expected_dofs, FEException,
                    "StandardAssembler::applyVectorBasisOutputOrientation: DOF association size mismatch");

        Blocks blocks;

        const auto edge_orients = dof_handler_->cellEdgeOrientations(cell_id);
        const auto face_orients = dof_handler_->cellFaceOrientations(cell_id);

        bool needs_edges = false;
        bool needs_faces = false;
        for (const auto& a : associations) {
            if (a.entity_type == basis::DofEntity::Edge) needs_edges = true;
            if (a.entity_type == basis::DofEntity::Face) needs_faces = true;
        }
        if (needs_edges) {
            FE_THROW_IF(edge_orients.empty(), FEException,
                        "StandardAssembler::applyVectorBasisOutputOrientation: missing cell edge orientations");
        }
        if (needs_faces) {
            FE_THROW_IF(face_orients.empty(), FEException,
                        "StandardAssembler::applyVectorBasisOutputOrientation: missing cell face orientations");
        }

        if (needs_edges) {
            for (LocalIndex i = 0; i < expected_dofs; ++i) {
                const auto& a = associations[static_cast<std::size_t>(i)];
                if (a.entity_type != basis::DofEntity::Edge) continue;
                const int e = a.entity_id;
                FE_THROW_IF(e < 0 || static_cast<std::size_t>(e) >= edge_orients.size(), FEException,
                            "StandardAssembler::applyVectorBasisOutputOrientation: edge id out of range in DOF associations");
                blocks.edge_dofs.push_back({i, edge_orients[static_cast<std::size_t>(e)]});
            }
        }

        if (needs_faces) {
            const elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
            const auto n_faces = static_cast<int>(ref.num_faces());
            FE_THROW_IF(n_faces <= 0, FEException,
                        "StandardAssembler::applyVectorBasisOutputOrientation: vector basis has face DOFs but cell has no faces");
            FE_THROW_IF(face_orients.size() < static_cast<std::size_t>(n_faces), FEException,
                        "StandardAssembler::applyVectorBasisOutputOrientation: face orientation span too small for cell");

            std::vector<std::vector<std::pair<int, LocalIndex>>> face_pairs(static_cast<std::size_t>(n_faces));
            for (LocalIndex i = 0; i < expected_dofs; ++i) {
                const auto& a = associations[static_cast<std::size_t>(i)];
                if (a.entity_type != basis::DofEntity::Face) continue;
                const int f = a.entity_id;
                FE_THROW_IF(f < 0 || f >= n_faces, FEException,
                            "StandardAssembler::applyVectorBasisOutputOrientation: face id out of range in DOF associations");
                face_pairs[static_cast<std::size_t>(f)].push_back({a.moment_index, i});
            }

            blocks.faces.reserve(static_cast<std::size_t>(n_faces));
            for (int f = 0; f < n_faces; ++f) {
                auto& pairs = face_pairs[static_cast<std::size_t>(f)];
                if (pairs.empty()) continue;

                std::sort(pairs.begin(), pairs.end(),
                          [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

                const LocalIndex n_face_dofs = static_cast<LocalIndex>(pairs.size());
                const auto& fn = ref.face_nodes(static_cast<std::size_t>(f));
                ElementType face_type = ElementType::Unknown;
                if (fn.size() == 3u) {
                    face_type = ElementType::Triangle3;
                } else if (fn.size() == 4u) {
                    face_type = ElementType::Quad4;
                } else {
                    FE_THROW(FEException, "StandardAssembler::applyVectorBasisOutputOrientation: unsupported face topology");
                }

                const auto& orient = face_orients[static_cast<std::size_t>(f)];
                const auto& tf = getFaceTransform(basis.basis_type(), continuity, face_type,
                                                  element.polynomial_order(), orient, n_face_dofs);

                Blocks::FaceBlock blk;
                blk.dofs.reserve(static_cast<std::size_t>(n_face_dofs));
                for (const auto& pr : pairs) {
                    blk.dofs.push_back(pr.second);
                }
                blk.tf = &tf;
                blocks.faces.push_back(std::move(blk));
            }
        }

        return blocks;
    };

    const auto test_blocks = need_test ? build_blocks(test_cell_id, test_space, output.n_test_dofs) : std::nullopt;
    const auto trial_blocks = need_trial ? build_blocks(trial_cell_id, trial_space, output.n_trial_dofs) : std::nullopt;

    // Transform vector: r_g = P_test^T r_l
    if (output.has_vector && test_blocks) {
        auto vec = std::span<Real>(output.local_vector);

        for (const auto& [idx, sign] : test_blocks->edge_dofs) {
            if (sign < 0) {
                vec[static_cast<std::size_t>(idx)] = -vec[static_cast<std::size_t>(idx)];
            }
        }

        for (const auto& face : test_blocks->faces) {
            const LocalIndex n = face.tf ? face.tf->n : 0;
            if (n <= 0) continue;
            const auto nn = static_cast<std::size_t>(n);
            FE_THROW_IF(face.dofs.size() != nn, FEException,
                        "StandardAssembler::applyVectorBasisOutputOrientation: face block size mismatch (vector)");

            scratch_orient_in_.assign(nn, 0.0);
            scratch_orient_out_.assign(nn, 0.0);

            for (std::size_t i = 0; i < nn; ++i) {
                scratch_orient_in_[i] = vec[static_cast<std::size_t>(face.dofs[i])];
            }

            for (std::size_t r = 0; r < nn; ++r) {
                Real sum = 0.0;
                const std::size_t base = r * nn;
                for (std::size_t c = 0; c < nn; ++c) {
                    sum += face.tf->PT[base + c] * scratch_orient_in_[c];
                }
                scratch_orient_out_[r] = sum;
            }

            for (std::size_t i = 0; i < nn; ++i) {
                vec[static_cast<std::size_t>(face.dofs[i])] = scratch_orient_out_[i];
            }
        }
    }

    if (!output.has_matrix) {
        return;
    }

    const auto n_test = static_cast<std::size_t>(std::max<LocalIndex>(0, output.n_test_dofs));
    const auto n_trial = static_cast<std::size_t>(std::max<LocalIndex>(0, output.n_trial_dofs));
    FE_THROW_IF(output.local_matrix.size() != n_test * n_trial, FEException,
                "StandardAssembler::applyVectorBasisOutputOrientation: local_matrix size mismatch");

    // Right multiply: A <- A * P_trial
    if (trial_blocks) {
        // Edge scaling (diagonal)
        for (const auto& [idx, sign] : trial_blocks->edge_dofs) {
            if (sign >= 0) continue;
            const std::size_t j = static_cast<std::size_t>(idx);
            for (std::size_t i = 0; i < n_test; ++i) {
                output.local_matrix[i * n_trial + j] = -output.local_matrix[i * n_trial + j];
            }
        }

        for (const auto& face : trial_blocks->faces) {
            const LocalIndex n = face.tf ? face.tf->n : 0;
            if (n <= 0) continue;
            const auto nn = static_cast<std::size_t>(n);
            FE_THROW_IF(face.dofs.size() != nn, FEException,
                        "StandardAssembler::applyVectorBasisOutputOrientation: face block size mismatch (trial)");

            scratch_orient_in_.assign(nn, 0.0);
            scratch_orient_out_.assign(nn, 0.0);

            for (std::size_t i = 0; i < n_test; ++i) {
                for (std::size_t l = 0; l < nn; ++l) {
                    scratch_orient_in_[l] = output.local_matrix[i * n_trial + static_cast<std::size_t>(face.dofs[l])];
                }
                for (std::size_t k = 0; k < nn; ++k) {
                    Real sum = 0.0;
                    for (std::size_t l = 0; l < nn; ++l) {
                        sum += scratch_orient_in_[l] * face.tf->P[l * nn + k];
                    }
                    scratch_orient_out_[k] = sum;
                }
                for (std::size_t k = 0; k < nn; ++k) {
                    output.local_matrix[i * n_trial + static_cast<std::size_t>(face.dofs[k])] = scratch_orient_out_[k];
                }
            }
        }
    }

    // Left multiply: A <- P_test^T * A
    if (test_blocks) {
        // Edge scaling (diagonal)
        for (const auto& [idx, sign] : test_blocks->edge_dofs) {
            if (sign >= 0) continue;
            const std::size_t i = static_cast<std::size_t>(idx);
            const std::size_t base = i * n_trial;
            for (std::size_t j = 0; j < n_trial; ++j) {
                output.local_matrix[base + j] = -output.local_matrix[base + j];
            }
        }

        for (const auto& face : test_blocks->faces) {
            const LocalIndex n = face.tf ? face.tf->n : 0;
            if (n <= 0) continue;
            const auto nn = static_cast<std::size_t>(n);
            FE_THROW_IF(face.dofs.size() != nn, FEException,
                        "StandardAssembler::applyVectorBasisOutputOrientation: face block size mismatch (test)");

            scratch_orient_in_.assign(nn, 0.0);
            scratch_orient_out_.assign(nn, 0.0);

            for (std::size_t j = 0; j < n_trial; ++j) {
                for (std::size_t l = 0; l < nn; ++l) {
                    scratch_orient_in_[l] =
                        output.local_matrix[static_cast<std::size_t>(face.dofs[l]) * n_trial + j];
                }
                for (std::size_t k = 0; k < nn; ++k) {
                    Real sum = 0.0;
                    const std::size_t base = k * nn;
                    for (std::size_t l = 0; l < nn; ++l) {
                        sum += face.tf->PT[base + l] * scratch_orient_in_[l];
                    }
                    scratch_orient_out_[k] = sum;
                }
                for (std::size_t k = 0; k < nn; ++k) {
                    output.local_matrix[static_cast<std::size_t>(face.dofs[k]) * n_trial + j] = scratch_orient_out_[k];
                }
            }
        }
    }
}

const FieldSolutionAccess* StandardAssembler::findFieldSolutionAccess(FieldId field) const noexcept
{
    for (const auto& rec : field_solution_access_) {
        if (rec.field == field) {
            return &rec;
        }
    }
    return nullptr;
}

void StandardAssembler::populateFieldSolutionData(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const std::vector<FieldRequirement>& requirements)
{
    context.clearFieldSolutionData();
    if (requirements.empty()) {
        return;
    }

    FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
	                "StandardAssembler::populateFieldSolutionData: no current solution vector was set");

    int required_history = 0;
    if (time_integration_ != nullptr) {
        if (time_integration_->dt1) {
            required_history = std::max(required_history, time_integration_->dt1->requiredHistoryStates());
        }
        if (time_integration_->dt2) {
            required_history = std::max(required_history, time_integration_->dt2->requiredHistoryStates());
        }
    }

	    if (required_history > 0) {
	        FE_THROW_IF(static_cast<int>(previous_solutions_.size()) < required_history, FEException,
	                    "StandardAssembler::populateFieldSolutionData: insufficient solution history (need " +
	                        std::to_string(required_history) + ")");
	        for (int k = 1; k <= required_history; ++k) {
	            const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	            const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                        ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                        : nullptr;
	            FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                        "StandardAssembler::populateFieldSolutionData: previous solution state " + std::to_string(k) +
	                            " not set");
	        }
	    }

    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();
    const auto qpts = context.quadraturePoints();

    std::vector<Real> values_at_pt;
    std::vector<basis::Gradient> gradients_at_pt;
    std::vector<basis::Hessian> hessians_at_pt;
    std::vector<Real> local_coeffs;

    std::vector<Real> scalar_values;
    std::vector<AssemblyContext::Vector3D> scalar_gradients;
    std::vector<AssemblyContext::Matrix3x3> scalar_hessians;
    std::vector<Real> scalar_laplacians;

    std::vector<AssemblyContext::Vector3D> vector_values;
    std::vector<AssemblyContext::Matrix3x3> vector_jacobians;
    std::vector<AssemblyContext::Matrix3x3> vector_component_hessians;
    std::vector<Real> vector_component_laplacians;

    for (const auto& req : requirements) {
        FE_THROW_IF(req.field == INVALID_FIELD_ID, FEException,
                    "StandardAssembler::populateFieldSolutionData: kernel requested an invalid FieldId");

        const auto* access = findFieldSolutionAccess(req.field);
        FE_THROW_IF(access == nullptr, FEException,
                    "StandardAssembler::populateFieldSolutionData: no FieldSolutionAccess was provided for field " +
                        std::to_string(req.field));
        FE_CHECK_NOT_NULL(access->space, "StandardAssembler::populateFieldSolutionData: field space");
        FE_CHECK_NOT_NULL(access->dof_map, "StandardAssembler::populateFieldSolutionData: field dof_map");

        const auto& space = *access->space;
        const auto& element = getElement(space, cell_id, cell_type);
        const auto& basis = element.basis();

        const bool is_product = (space.space_type() == spaces::SpaceType::Product);
        const auto n_qpts = context.numQuadraturePoints();
        const auto n_dofs = static_cast<LocalIndex>(space.dofs_per_element());
        const auto n_scalar_dofs = static_cast<LocalIndex>(element.num_dofs());

        const bool want_values = hasFlag(req.required, RequiredData::SolutionValues) || (req.required == RequiredData::None);
        const bool want_gradients = hasFlag(req.required, RequiredData::SolutionGradients);
        const bool want_hessians = hasFlag(req.required, RequiredData::SolutionHessians);
        const bool want_laplacians = hasFlag(req.required, RequiredData::SolutionLaplacians);
        const bool need_gradients = want_gradients;
        const bool need_hessians = want_hessians || want_laplacians;

        auto cell_dofs = access->dof_map->getCellDofs(cell_id);
        FE_THROW_IF(cell_dofs.size() != static_cast<std::size_t>(n_dofs), FEException,
                    "StandardAssembler::populateFieldSolutionData: field DOF count does not match its space DOFs");

	        local_coeffs.resize(cell_dofs.size());
	        for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
	            const auto dof = cell_dofs[i] + access->dof_offset;
	            if (dof < 0) {
	                FE_THROW(FEException, "StandardAssembler::populateFieldSolutionData: negative DOF index");
	            }
	            if (current_solution_view_ != nullptr) {
	                local_coeffs[i] = current_solution_view_->getVectorEntry(dof);
	            } else {
	                FE_THROW_IF(static_cast<std::size_t>(dof) >= current_solution_.size(), FEException,
	                            "StandardAssembler::populateFieldSolutionData: solution vector too small for DOF " +
	                                std::to_string(dof));
	                local_coeffs[i] = current_solution_[static_cast<std::size_t>(dof)];
	            }
	        }

        if (space.field_type() == FieldType::Scalar) {
            FE_THROW_IF(is_product, FEException,
                        "StandardAssembler::populateFieldSolutionData: ProductSpace cannot be scalar-valued");
            FE_THROW_IF(n_dofs != n_scalar_dofs, FEException,
                        "StandardAssembler::populateFieldSolutionData: non-Product scalar space DOF count mismatch");

            scalar_values.assign(static_cast<std::size_t>(n_qpts), 0.0);
            if (need_gradients) {
                scalar_gradients.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Vector3D{0.0, 0.0, 0.0});
            } else {
                scalar_gradients.clear();
            }
            if (need_hessians) {
                scalar_hessians.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Matrix3x3{});
            } else {
                scalar_hessians.clear();
            }
            if (want_laplacians) {
                scalar_laplacians.assign(static_cast<std::size_t>(n_qpts), 0.0);
            } else {
                scalar_laplacians.clear();
            }

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                               qpts[static_cast<std::size_t>(q)][1],
                                               qpts[static_cast<std::size_t>(q)][2]};

                basis.evaluate_values(xi, values_at_pt);
                if (need_gradients) {
                    basis.evaluate_gradients(xi, gradients_at_pt);
                }
                if (need_hessians) {
                    basis.evaluate_hessians(xi, hessians_at_pt);
                }

                const auto J_inv = context.inverseJacobian(q);
                Real val = 0.0;
                AssemblyContext::Vector3D grad = {0.0, 0.0, 0.0};
                AssemblyContext::Matrix3x3 H{};

                for (LocalIndex j = 0; j < n_dofs; ++j) {
                    const Real coef = local_coeffs[static_cast<std::size_t>(j)];
                    val += coef * values_at_pt[static_cast<std::size_t>(j)];

                    if (need_gradients) {
                        const auto& gref = gradients_at_pt[static_cast<std::size_t>(j)];
                        AssemblyContext::Vector3D gphys = {0.0, 0.0, 0.0};
                        for (int d1 = 0; d1 < dim; ++d1) {
                            for (int d2 = 0; d2 < dim; ++d2) {
                                gphys[d1] += J_inv[static_cast<std::size_t>(d2)][static_cast<std::size_t>(d1)] * gref[static_cast<std::size_t>(d2)];
                            }
                        }
                        grad[0] += coef * gphys[0];
                        grad[1] += coef * gphys[1];
                        grad[2] += coef * gphys[2];
                    }

                    if (need_hessians) {
                        AssemblyContext::Matrix3x3 H_ref{};
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                    hessians_at_pt[static_cast<std::size_t>(j)](static_cast<std::size_t>(r),
                                                                                static_cast<std::size_t>(c));
                            }
                        }

                        AssemblyContext::Matrix3x3 H_phys{};
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                Real sum = 0.0;
                                for (int a = 0; a < dim; ++a) {
                                    for (int b = 0; b < dim; ++b) {
                                        sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                               H_ref[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                               J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                    }
                                }
                                H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                            }
                        }

                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                                    coef * H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            }
                        }
                    }
                }

                scalar_values[static_cast<std::size_t>(q)] = val;
                if (need_gradients) {
                    scalar_gradients[static_cast<std::size_t>(q)] = grad;
                }
                if (need_hessians) {
                    scalar_hessians[static_cast<std::size_t>(q)] = H;
                    if (want_laplacians) {
                        Real lap = 0.0;
                        for (int d = 0; d < dim; ++d) {
                            lap += H[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)];
                        }
                        scalar_laplacians[static_cast<std::size_t>(q)] = lap;
                    }
                }
            }

            context.setFieldSolutionScalar(req.field,
                                           want_values ? std::span<const Real>(scalar_values) : std::span<const Real>{},
                                           want_gradients ? std::span<const AssemblyContext::Vector3D>(scalar_gradients) : std::span<const AssemblyContext::Vector3D>{},
                                           want_hessians ? std::span<const AssemblyContext::Matrix3x3>(scalar_hessians) : std::span<const AssemblyContext::Matrix3x3>{},
                                           want_laplacians ? std::span<const Real>(scalar_laplacians) : std::span<const Real>{});

            // Populate previous field values if a transient context requires history.
	            if (required_history > 0) {
	                for (int k = 1; k <= required_history; ++k) {
	                    const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	                    const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                                ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                                : nullptr;
	                    FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                "StandardAssembler::populateFieldSolutionData: previous solution data missing");

	                    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
	                        const auto dof = cell_dofs[i] + access->dof_offset;
	                        FE_THROW_IF(dof < 0, FEException,
	                                    "StandardAssembler::populateFieldSolutionData: negative DOF index (prev)");
	                        if (prev_view != nullptr) {
	                            local_coeffs[i] = prev_view->getVectorEntry(dof);
	                        } else {
	                            FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                        "StandardAssembler::populateFieldSolutionData: previous solution vector too small for DOF " +
	                                            std::to_string(dof));
	                            local_coeffs[i] = prev[static_cast<std::size_t>(dof)];
	                        }
	                    }

                    // Values only (dt() needs just value history).
                    scalar_values.assign(static_cast<std::size_t>(n_qpts), 0.0);
                    for (LocalIndex q = 0; q < n_qpts; ++q) {
                        const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                       qpts[static_cast<std::size_t>(q)][1],
                                                       qpts[static_cast<std::size_t>(q)][2]};
                        basis.evaluate_values(xi, values_at_pt);
                        Real val = 0.0;
                        for (LocalIndex j = 0; j < n_dofs; ++j) {
                            const Real coef = local_coeffs[static_cast<std::size_t>(j)];
                            val += coef * values_at_pt[static_cast<std::size_t>(j)];
                        }
                        scalar_values[static_cast<std::size_t>(q)] = val;
                    }
                    context.setFieldPreviousSolutionScalarK(req.field, k, std::span<const Real>(scalar_values));
                }
            }
            continue;
        }

        if (space.field_type() == FieldType::Vector) {
            const int vd = space.value_dimension();
            FE_THROW_IF(vd <= 0 || vd > 3, FEException,
                        "StandardAssembler::populateFieldSolutionData: vector space value_dimension must be 1..3");

            // Vector-basis spaces (H(curl)/H(div)) are non-Product vector-valued spaces. We currently support
            // values only for these coefficient fields (no gradients/Hessians/Laplacians).
            if (!is_product) {
                FE_THROW_IF(!basis.is_vector_valued(), FEException,
                            "StandardAssembler::populateFieldSolutionData: non-Product vector field is not a vector-basis space");
                const auto cont = space.continuity();
                FE_THROW_IF(cont != Continuity::H_curl && cont != Continuity::H_div, FEException,
                            "StandardAssembler::populateFieldSolutionData: non-Product vector field is not H(curl)/H(div)");
                FE_THROW_IF(need_gradients || need_hessians, FEException,
                            "StandardAssembler::populateFieldSolutionData: SolutionGradients/Hessians/Laplacians are not supported for vector-basis fields");
                FE_THROW_IF(n_dofs != n_scalar_dofs, FEException,
                            "StandardAssembler::populateFieldSolutionData: vector-basis DOF count mismatch");

                applyVectorBasisGlobalToLocal(mesh, cell_id, space, std::span<Real>(local_coeffs));

                vector_values.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Vector3D{0.0, 0.0, 0.0});
                vector_jacobians.clear();
                vector_component_hessians.clear();
                vector_component_laplacians.clear();

                std::vector<math::Vector<Real, 3>> vec_values_at_pt;
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                   qpts[static_cast<std::size_t>(q)][1],
                                                   qpts[static_cast<std::size_t>(q)][2]};
                    basis.evaluate_vector_values(xi, vec_values_at_pt);

                    const auto J = context.jacobian(q);
                    const auto J_inv = context.inverseJacobian(q);
                    const Real det_J = context.jacobianDet(q);

                    AssemblyContext::Vector3D u{0.0, 0.0, 0.0};
                    for (LocalIndex j = 0; j < n_dofs; ++j) {
                        const Real coef = local_coeffs[static_cast<std::size_t>(j)];
                        const auto& vref = vec_values_at_pt[static_cast<std::size_t>(j)];
                        AssemblyContext::Vector3D vphys{0.0, 0.0, 0.0};
                        if (cont == Continuity::H_curl) {
                            for (int r = 0; r < dim; ++r) {
                                for (int c = 0; c < dim; ++c) {
                                    vphys[static_cast<std::size_t>(r)] +=
                                        J_inv[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] *
                                        vref[static_cast<std::size_t>(c)];
                                }
                            }
                        } else { // H_div
                            const Real inv_det = Real(1) / det_J;
                            for (int r = 0; r < dim; ++r) {
                                Real sum = 0.0;
                                for (int c = 0; c < dim; ++c) {
                                    sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                           vref[static_cast<std::size_t>(c)];
                                }
                                vphys[static_cast<std::size_t>(r)] = inv_det * sum;
                            }
                        }

                        u[0] += coef * vphys[0];
                        u[1] += coef * vphys[1];
                        u[2] += coef * vphys[2];
                    }

                    vector_values[static_cast<std::size_t>(q)] = u;
                }

                context.setFieldSolutionVector(req.field, vd,
                                               want_values ? std::span<const AssemblyContext::Vector3D>(vector_values)
                                                           : std::span<const AssemblyContext::Vector3D>{});

	                if (required_history > 0) {
	                    for (int k = 1; k <= required_history; ++k) {
	                        const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	                        const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                                    ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                                    : nullptr;
	                        FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                    "StandardAssembler::populateFieldSolutionData: previous solution data missing");

	                        for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
	                            const auto dof = cell_dofs[i] + access->dof_offset;
	                            FE_THROW_IF(dof < 0, FEException,
	                                        "StandardAssembler::populateFieldSolutionData: negative DOF index (prev)");
	                            if (prev_view != nullptr) {
	                                local_coeffs[i] = prev_view->getVectorEntry(dof);
	                            } else {
	                                FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                            "StandardAssembler::populateFieldSolutionData: previous solution vector too small for DOF " +
	                                                std::to_string(dof));
	                                local_coeffs[i] = prev[static_cast<std::size_t>(dof)];
	                            }
	                        }

                        applyVectorBasisGlobalToLocal(mesh, cell_id, space, std::span<Real>(local_coeffs));

                        vector_values.assign(static_cast<std::size_t>(n_qpts),
                                             AssemblyContext::Vector3D{0.0, 0.0, 0.0});

                        for (LocalIndex q = 0; q < n_qpts; ++q) {
                            const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                           qpts[static_cast<std::size_t>(q)][1],
                                                           qpts[static_cast<std::size_t>(q)][2]};
                            basis.evaluate_vector_values(xi, vec_values_at_pt);

                            const auto J = context.jacobian(q);
                            const auto J_inv = context.inverseJacobian(q);
                            const Real det_J = context.jacobianDet(q);

                            AssemblyContext::Vector3D u{0.0, 0.0, 0.0};
                            for (LocalIndex j = 0; j < n_dofs; ++j) {
                                const Real coef = local_coeffs[static_cast<std::size_t>(j)];
                                const auto& vref = vec_values_at_pt[static_cast<std::size_t>(j)];
                                AssemblyContext::Vector3D vphys{0.0, 0.0, 0.0};
                                if (cont == Continuity::H_curl) {
                                    for (int r = 0; r < dim; ++r) {
                                        for (int c = 0; c < dim; ++c) {
                                            vphys[static_cast<std::size_t>(r)] +=
                                                J_inv[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] *
                                                vref[static_cast<std::size_t>(c)];
                                        }
                                    }
                                } else { // H_div
                                    const Real inv_det = Real(1) / det_J;
                                    for (int r = 0; r < dim; ++r) {
                                        Real sum = 0.0;
                                        for (int c = 0; c < dim; ++c) {
                                            sum += J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                                                   vref[static_cast<std::size_t>(c)];
                                        }
                                        vphys[static_cast<std::size_t>(r)] = inv_det * sum;
                                    }
                                }

                                u[0] += coef * vphys[0];
                                u[1] += coef * vphys[1];
                                u[2] += coef * vphys[2];
                            }

                            vector_values[static_cast<std::size_t>(q)] = u;
                        }

                        context.setFieldPreviousSolutionVectorK(req.field, k, vd,
                                                               std::span<const AssemblyContext::Vector3D>(vector_values));
                    }
                }

                continue;
            }

            FE_THROW_IF(n_dofs != static_cast<LocalIndex>(n_scalar_dofs * static_cast<LocalIndex>(vd)), FEException,
                        "StandardAssembler::populateFieldSolutionData: ProductSpace DOF count mismatch");

            const LocalIndex dofs_per_component = static_cast<LocalIndex>(n_dofs / static_cast<LocalIndex>(vd));

            vector_values.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Vector3D{0.0, 0.0, 0.0});
            if (need_gradients) {
                vector_jacobians.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Matrix3x3{});
            } else {
                vector_jacobians.clear();
            }
            if (need_hessians) {
                vector_component_hessians.assign(static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(vd),
                                                 AssemblyContext::Matrix3x3{});
            } else {
                vector_component_hessians.clear();
            }
            if (want_laplacians) {
                vector_component_laplacians.assign(static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(vd), 0.0);
            } else {
                vector_component_laplacians.clear();
            }

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                               qpts[static_cast<std::size_t>(q)][1],
                                               qpts[static_cast<std::size_t>(q)][2]};

                basis.evaluate_values(xi, values_at_pt);
                if (need_gradients) {
                    basis.evaluate_gradients(xi, gradients_at_pt);
                }
                if (need_hessians) {
                    basis.evaluate_hessians(xi, hessians_at_pt);
                }

                const auto J_inv = context.inverseJacobian(q);
                AssemblyContext::Matrix3x3 J{};

                const auto q_base = static_cast<std::size_t>(q) * static_cast<std::size_t>(vd);
                for (int comp = 0; comp < vd; ++comp) {
                    Real val_c = 0.0;
                    AssemblyContext::Matrix3x3 H{};

                    const LocalIndex base = static_cast<LocalIndex>(comp) * dofs_per_component;
                    for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                        const LocalIndex jj = base + j;
                        const LocalIndex sj = static_cast<LocalIndex>(jj % n_scalar_dofs);
                        const Real coef = local_coeffs[static_cast<std::size_t>(jj)];
                        val_c += coef * values_at_pt[static_cast<std::size_t>(sj)];

                        if (need_gradients) {
                            const auto& gref = gradients_at_pt[static_cast<std::size_t>(sj)];
                            AssemblyContext::Vector3D gphys = {0.0, 0.0, 0.0};
                            for (int d1 = 0; d1 < dim; ++d1) {
                                for (int d2 = 0; d2 < dim; ++d2) {
                                    gphys[d1] += J_inv[static_cast<std::size_t>(d2)][static_cast<std::size_t>(d1)] * gref[static_cast<std::size_t>(d2)];
                                }
                            }
                            J[static_cast<std::size_t>(comp)][0] += coef * gphys[0];
                            J[static_cast<std::size_t>(comp)][1] += coef * gphys[1];
                            J[static_cast<std::size_t>(comp)][2] += coef * gphys[2];
                        }

                        if (need_hessians) {
                            AssemblyContext::Matrix3x3 H_ref{};
                            for (int r = 0; r < 3; ++r) {
                                for (int c = 0; c < 3; ++c) {
                                    H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                        hessians_at_pt[static_cast<std::size_t>(sj)](static_cast<std::size_t>(r),
                                                                                     static_cast<std::size_t>(c));
                                }
                            }

                            AssemblyContext::Matrix3x3 H_phys{};
                            for (int r = 0; r < dim; ++r) {
                                for (int c = 0; c < dim; ++c) {
                                    Real sum = 0.0;
                                    for (int a = 0; a < dim; ++a) {
                                        for (int b = 0; b < dim; ++b) {
                                            sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                                   H_ref[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                                   J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                        }
                                    }
                                    H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                                }
                            }

                            for (int r = 0; r < 3; ++r) {
                                for (int c = 0; c < 3; ++c) {
                                    H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                                        coef * H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                }
                            }
                        }
                    }

                    vector_values[static_cast<std::size_t>(q)][static_cast<std::size_t>(comp)] = val_c;
                    if (need_hessians) {
                        const auto idx = q_base + static_cast<std::size_t>(comp);
                        vector_component_hessians[idx] = H;
                        if (want_laplacians) {
                            Real lap = 0.0;
                            for (int d = 0; d < dim; ++d) {
                                lap += H[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)];
                            }
                            vector_component_laplacians[idx] = lap;
                        }
                    }
                }

                if (need_gradients) {
                    vector_jacobians[static_cast<std::size_t>(q)] = J;
                }
            }

            context.setFieldSolutionVector(req.field, vd,
                                           want_values ? std::span<const AssemblyContext::Vector3D>(vector_values) : std::span<const AssemblyContext::Vector3D>{},
                                           want_gradients ? std::span<const AssemblyContext::Matrix3x3>(vector_jacobians) : std::span<const AssemblyContext::Matrix3x3>{},
                                           want_hessians ? std::span<const AssemblyContext::Matrix3x3>(vector_component_hessians) : std::span<const AssemblyContext::Matrix3x3>{},
                                           want_laplacians ? std::span<const Real>(vector_component_laplacians) : std::span<const Real>{});

            // Populate previous field values if a transient context requires history.
	            if (required_history > 0) {
	                for (int k = 1; k <= required_history; ++k) {
	                    const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
	                    const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
	                                                ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
	                                                : nullptr;
	                    FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                "StandardAssembler::populateFieldSolutionData: previous solution data missing");

	                    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
	                        const auto dof = cell_dofs[i] + access->dof_offset;
	                        FE_THROW_IF(dof < 0, FEException,
	                                    "StandardAssembler::populateFieldSolutionData: negative DOF index (prev)");
	                        if (prev_view != nullptr) {
	                            local_coeffs[i] = prev_view->getVectorEntry(dof);
	                        } else {
	                            FE_THROW_IF(static_cast<std::size_t>(dof) >= prev.size(), FEException,
	                                        "StandardAssembler::populateFieldSolutionData: previous solution vector too small for DOF " +
	                                            std::to_string(dof));
	                            local_coeffs[i] = prev[static_cast<std::size_t>(dof)];
	                        }
	                    }

                    // Values only (dt() needs just value history).
                    vector_values.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Vector3D{0.0, 0.0, 0.0});
                    for (LocalIndex q = 0; q < n_qpts; ++q) {
                        const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                       qpts[static_cast<std::size_t>(q)][1],
                                                       qpts[static_cast<std::size_t>(q)][2]};
                        basis.evaluate_values(xi, values_at_pt);

                        AssemblyContext::Vector3D u_prev = {0.0, 0.0, 0.0};
                        for (int c = 0; c < vd; ++c) {
                            Real val_c = 0.0;
                            const LocalIndex base = static_cast<LocalIndex>(c) * dofs_per_component;
                            for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                                const LocalIndex jj = base + j;
                                val_c += local_coeffs[static_cast<std::size_t>(jj)] *
                                         values_at_pt[static_cast<std::size_t>(j)];
                            }
                            u_prev[static_cast<std::size_t>(c)] = val_c;
                        }
                        vector_values[static_cast<std::size_t>(q)] = u_prev;
                    }

                    context.setFieldPreviousSolutionVectorK(req.field, k, vd,
                                                            std::span<const AssemblyContext::Vector3D>(vector_values));
                }
            }
            continue;
        }

        throw FEException("StandardAssembler::populateFieldSolutionData: unsupported field type",
                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    }
}

void StandardAssembler::insertLocal(
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (options_.check_finite_values) {
        auto check_finite = [](std::span<const Real> values, const char* what) {
            for (Real v : values) {
                if (!std::isfinite(v)) {
                    throw std::runtime_error(
                        std::string("StandardAssembler: ") + what + " contains NaN/Inf");
                }
            }
        };

        if (output.has_matrix) {
            check_finite(output.local_matrix, "local matrix");
        }
        if (output.has_vector) {
            check_finite(output.local_vector, "local vector");
        }
    }

    // Insert matrix entries
    if (matrix_view && output.has_matrix) {
        matrix_view->addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
    }

    // Insert vector entries
    if (vector_view && output.has_vector) {
        vector_view->addVectorEntries(row_dofs, output.local_vector);
    }
}

void StandardAssembler::insertLocalConstrained(
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (options_.check_finite_values) {
        auto check_finite = [](std::span<const Real> values, const char* what) {
            for (Real v : values) {
                if (!std::isfinite(v)) {
                    throw std::runtime_error(
                        std::string("StandardAssembler: ") + what + " contains NaN/Inf");
                }
            }
        };

        if (output.has_matrix) {
            check_finite(output.local_matrix, "local matrix");
        }
        if (output.has_vector) {
            check_finite(output.local_vector, "local vector");
        }
    }

    // Check if any DOFs are constrained
    if (!constraints_->hasConstrainedDofs(row_dofs) &&
        !constraints_->hasConstrainedDofs(col_dofs)) {
        // No constraints - direct insertion
        insertLocal(output, row_dofs, col_dofs, matrix_view, vector_view);
        return;
    }

    // Use ConstraintDistributor for constrained assembly
    // This expands constrained DOFs to their masters and distributes contributions

    // For matrix
    if (matrix_view && output.has_matrix && constraint_distributor_) {
        // Create matrix ops adapter
        class MatrixOpsAdapter : public constraints::IMatrixOperations {
        public:
            explicit MatrixOpsAdapter(GlobalSystemView& view) : view_(view) {}

            void addValues(std::span<const GlobalIndex> rows,
                           std::span<const GlobalIndex> cols,
                           std::span<const double> values) override {
                view_.addMatrixEntries(rows, cols, values);
            }

            void addValue(GlobalIndex row, GlobalIndex col, double value) override {
                view_.addMatrixEntry(row, col, value);
            }

            void setDiagonal(GlobalIndex row, double value) override {
                view_.setDiagonal(row, value);
            }

            [[nodiscard]] GlobalIndex numRows() const override { return view_.numRows(); }
            [[nodiscard]] GlobalIndex numCols() const override { return view_.numCols(); }

        private:
            GlobalSystemView& view_;
        };

        MatrixOpsAdapter matrix_ops(*matrix_view);

        // Also need vector ops if vector_view is provided
        if (vector_view && output.has_vector) {
            class VectorOpsAdapter : public constraints::IVectorOperations {
            public:
                explicit VectorOpsAdapter(GlobalSystemView& view) : view_(view) {}

                void addValues(std::span<const GlobalIndex> indices,
                               std::span<const double> values) override {
                    view_.addVectorEntries(indices, values);
                }

                void addValue(GlobalIndex index, double value) override {
                    view_.addVectorEntry(index, value);
                }

                void setValue(GlobalIndex index, double value) override {
                    view_.addVectorEntry(index, value, AddMode::Insert);
                }

                [[nodiscard]] double getValue(GlobalIndex index) const override {
                    return view_.getVectorEntry(index);
                }

                [[nodiscard]] GlobalIndex size() const override {
                    return view_.numRows();
                }

            private:
                GlobalSystemView& view_;
            };

            VectorOpsAdapter vector_ops(*vector_view);

            constraint_distributor_->distributeLocalToGlobal(
                output.local_matrix, output.local_vector,
                row_dofs, col_dofs, matrix_ops, vector_ops);
        } else {
            constraint_distributor_->distributeMatrixToGlobal(
                output.local_matrix, row_dofs, col_dofs, matrix_ops);
        }
    } else if (vector_view && output.has_vector && constraint_distributor_) {
        // Vector-only with constraints
        class VectorOpsAdapter : public constraints::IVectorOperations {
        public:
            explicit VectorOpsAdapter(GlobalSystemView& view) : view_(view) {}

            void addValues(std::span<const GlobalIndex> indices,
                           std::span<const double> values) override {
                view_.addVectorEntries(indices, values);
            }

            void addValue(GlobalIndex index, double value) override {
                view_.addVectorEntry(index, value);
            }

            void setValue(GlobalIndex index, double value) override {
                view_.addVectorEntry(index, value, AddMode::Insert);
            }

            [[nodiscard]] double getValue(GlobalIndex index) const override {
                return view_.getVectorEntry(index);
            }

            [[nodiscard]] GlobalIndex size() const override {
                return view_.numRows();
            }

        private:
            GlobalSystemView& view_;
        };

        VectorOpsAdapter vector_ops(*vector_view);
        constraint_distributor_->distributeRhsToGlobal(
            output.local_vector, row_dofs, vector_ops);
    }
}

const elements::Element& StandardAssembler::getElement(
    const spaces::FunctionSpace& space,
    GlobalIndex cell_id,
    ElementType cell_type) const
{
    return space.getElement(cell_type, cell_id);
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Assembler> createStandardAssembler()
{
    return std::make_unique<StandardAssembler>();
}

std::unique_ptr<Assembler> createStandardAssembler(const AssemblyOptions& options)
{
    return std::make_unique<StandardAssembler>(options);
}

std::unique_ptr<Assembler> createAssembler(ThreadingStrategy strategy)
{
    switch (strategy) {
        case ThreadingStrategy::Sequential:
            return createStandardAssembler();
        case ThreadingStrategy::Colored:
        case ThreadingStrategy::WorkStream:
        case ThreadingStrategy::Atomic:
            // These would return specialized assemblers
            // For now, fall back to standard
            return createStandardAssembler();
        default:
            return createStandardAssembler();
    }
}

std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options)
{
    return createAssembler(options.threading);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
