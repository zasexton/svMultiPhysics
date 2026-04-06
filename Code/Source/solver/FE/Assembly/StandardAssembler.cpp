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
#include "Geometry/GeometryFrameUtils.h"
#include "Basis/BasisFunction.h"
#include "Basis/BasisCache.h"
#include "Basis/VectorBasis.h"
#include "Assembly/JIT/KernelArgs.h"
#include "Core/KernelTrace.h"
#include "Coloring.h"
#include "Forms/FormKernels.h"
#include "Forms/MixedBlockKernelSet.h"
#include "Forms/MonolithicCellKernel.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Math/Vector.h"
#include "Math/Matrix.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/InterfaceMesh.h"
#endif

#include <atomic>
#include <chrono>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdio>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

namespace {

/// Runtime assembly timing control: set SVMP_ASSEMBLY_TIMING=1 to enable.
/// When disabled (default), TP() returns 0.0 with a well-predicted branch.
[[nodiscard]] inline bool assemblyTimingEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_ASSEMBLY_TIMING");
        return env && std::string_view(env) != "0";
    }();
    return enabled;
}

[[nodiscard]] inline double assemblyTimeNow() noexcept {
    if (!assemblyTimingEnabled()) return 0.0;
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[nodiscard]] inline bool monolithicCompiledDispatchEnabled() noexcept
{
    return core::envEnabled("SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH") &&
           !core::envEnabled("SVMP_FE_DISABLE_MONOLITHIC_COMPILED_DISPATCH");
}

[[nodiscard]] inline bool monolithicCompiledCompareEnabled() noexcept
{
    return core::envEnabled("SVMP_FE_COMPARE_MONOLITHIC_COMPILED");
}

[[nodiscard]] inline bool dirichletFastPathEnabled() noexcept
{
    return !core::envEnabled("SVMP_FE_DISABLE_DIRICHLET_FAST_PATH");
}

[[nodiscard]] inline Real monolithicCompiledCompareTolerance() noexcept
{
    static const Real tol = [] {
        const char* env = std::getenv("SVMP_FE_COMPARE_MONOLITHIC_TOL");
        if (env == nullptr || env[0] == '\0') {
            return Real(1e-11);
        }
        char* end = nullptr;
        const double value = std::strtod(env, &end);
        if (end == env || !std::isfinite(value) || value < 0.0) {
            return Real(1e-11);
        }
        return static_cast<Real>(value);
    }();
    return tol;
}

[[nodiscard]] inline int monolithicCompiledCompareMaxCells() noexcept
{
    static const int max_cells = [] {
        const char* env = std::getenv("SVMP_FE_COMPARE_MONOLITHIC_MAX_CELLS");
        if (env == nullptr || env[0] == '\0') {
            return 1;
        }
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end == env || value <= 0) {
            return 1;
        }
        if (value > std::numeric_limits<int>::max()) {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(value);
    }();
    return max_cells;
}

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

[[nodiscard]] math::Vector<Real, 3> cross3(const math::Vector<Real, 3>& a,
                                           const math::Vector<Real, 3>& b) noexcept
{
    return math::Vector<Real, 3>{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

[[nodiscard]] Real norm3(const math::Vector<Real, 3>& v) noexcept
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

[[nodiscard]] Real canonicalFaceJacobianToReference(
    ElementType face_type,
    std::span<const math::Vector<Real, 3>> ref_face_coords,
    const math::Vector<Real, 3>& facet_coords)
{
    // Convert canonical face quadrature weights to the element-reference facet measure.
    //
    // Face quadrature rules are defined on canonical domains:
    //   - Line2:    s in [-1, 1]
    //   - Quad4:    (xi,eta) in [-1,1]^2
    //   - Triangle: (x,y) on reference simplex (area 0.5)
    //
    // ElementTransform::facet_to_reference() expects facet-local parameters:
    //   - edges:    t in [0, 1]
    //   - quad:     (s,t) in [0,1]^2
    //   - triangle: (x,y) on reference simplex
    //
    // This function returns |dX_ref/du| where u is the canonical quadrature coordinate,
    // so that dS_ref = jac * du. Multiply quadrature weights by this factor.
    switch (face_type) {
        case ElementType::Line2: {
            FE_THROW_IF(ref_face_coords.size() < 2, FEException,
                        "canonicalFaceJacobianToReference(Line2): missing vertices");
            const math::Vector<Real, 3> dx = ref_face_coords[1] - ref_face_coords[0];
            // t = (s+1)/2 => dt/ds = 1/2
            return Real(0.5) * norm3(dx);
        }
        case ElementType::Triangle3: {
            FE_THROW_IF(ref_face_coords.size() < 3, FEException,
                        "canonicalFaceJacobianToReference(Triangle3): missing vertices");
            (void)facet_coords;
            const math::Vector<Real, 3> e1 = ref_face_coords[1] - ref_face_coords[0];
            const math::Vector<Real, 3> e2 = ref_face_coords[2] - ref_face_coords[0];
            // xi(x,y) = v0 + x*(v1-v0) + y*(v2-v0) => jac = |e1 x e2|
            return norm3(cross3(e1, e2));
        }
        case ElementType::Quad4: {
            FE_THROW_IF(ref_face_coords.size() < 4, FEException,
                        "canonicalFaceJacobianToReference(Quad4): missing vertices");
            const Real s = facet_coords[0];
            const Real t = facet_coords[1];
            // X_ref(s,t) is bilinear on [0,1]^2; canonical quad weights are on [-1,1]^2.
            // (s,t) = ((xi+1)/2, (eta+1)/2) => dxi deta = 4 ds dt, so:
            // |dX/dxi x dX/deta| = 0.25 * |dX/ds x dX/dt|
            math::Vector<Real, 3> dXds{};
            math::Vector<Real, 3> dXdt{};
            for (std::size_t i = 0; i < 3; ++i) {
                dXds[i] = (Real(1) - t) * (ref_face_coords[1][i] - ref_face_coords[0][i]) +
                          t * (ref_face_coords[2][i] - ref_face_coords[3][i]);
                dXdt[i] = (Real(1) - s) * (ref_face_coords[3][i] - ref_face_coords[0][i]) +
                          s * (ref_face_coords[2][i] - ref_face_coords[1][i]);
            }
            return Real(0.25) * norm3(cross3(dXds, dXdt));
        }
        default:
            break;
    }
    return Real(1);
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
    for (const auto& s : ctx->dt_extra) {
        if (s) {
            required = std::max(required, s->requiredHistoryStates());
        }
    }
    return required;
}

struct ResolvedVectorGatherCache {
    const void* layout_handle{nullptr};
    std::vector<GlobalIndex> dofs{};
    std::vector<GlobalIndex> resolved{};
};

void gatherVectorCoefficients(
    std::span<const GlobalIndex> dofs,
    const GlobalSystemView* view,
    std::span<const Real> raw_values,
    std::vector<Real>& out,
    ResolvedVectorGatherCache* cache,
    const char* error_prefix,
    bool validate_negative_dofs,
    std::span<const GlobalIndex> pre_resolved = {})
{
    out.resize(dofs.size());

    if (validate_negative_dofs) {
        for (const auto dof : dofs) {
            FE_THROW_IF(dof < 0, FEException, std::string(error_prefix) + ": negative DOF index");
        }
    }

    if (view != nullptr) {
        if (!pre_resolved.empty()) {
            FE_THROW_IF(pre_resolved.size() != dofs.size(), FEException,
                        std::string(error_prefix) + ": resolved vector-entry size mismatch");
            view->getVectorEntriesResolved(pre_resolved, std::span<Real>(out));
            return;
        }
        if (cache != nullptr) {
            const void* layout_handle = view->vectorLayoutHandle();
            const bool cache_hit =
                cache->layout_handle == layout_handle &&
                cache->dofs.size() == dofs.size() &&
                std::equal(cache->dofs.begin(), cache->dofs.end(), dofs.begin());

            if (!cache_hit) {
                cache->layout_handle = layout_handle;
                cache->dofs.assign(dofs.begin(), dofs.end());
                cache->resolved.resize(dofs.size());
                view->resolveVectorEntries(dofs, cache->resolved);
            }

            view->getVectorEntriesResolved(cache->resolved, std::span<Real>(out));
        } else {
            view->getVectorEntries(dofs, std::span<Real>(out));
        }
        return;
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        const auto dof = dofs[i];
        FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= raw_values.size(), FEException,
                    std::string(error_prefix) + ": solution vector too small for DOF " +
                        std::to_string(dof));
        out[i] = raw_values[static_cast<std::size_t>(dof)];
    }
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

    [[nodiscard]] const void* matrixLayoutHandle() const noexcept override
    {
        return base_ ? base_->matrixLayoutHandle() : nullptr;
    }

    void resolveMatrixEntries(std::span<const GlobalIndex> row_dofs,
                              std::span<const GlobalIndex> col_dofs,
                              std::span<GlobalIndex> resolved) const override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        FE_THROW_IF(resolved.size() != row_dofs.size() * col_dofs.size(), FEException,
                    "OwnedRowOnlyView::resolveMatrixEntries: size mismatch");

        bool all_owned = true;
        for (const auto row : row_dofs) {
            if (!isOwnedRow(row)) {
                all_owned = false;
                break;
            }
        }
        if (all_owned) {
            base_->resolveMatrixEntries(row_dofs, col_dofs, resolved);
            return;
        }

        std::fill(resolved.begin(), resolved.end(), INVALID_GLOBAL_INDEX);
        std::vector<GlobalIndex> owned_rows;
        owned_rows.reserve(row_dofs.size());
        std::vector<std::size_t> owned_row_idx;
        owned_row_idx.reserve(row_dofs.size());
        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            const auto row = row_dofs[i];
            if (!isOwnedRow(row)) {
                continue;
            }
            owned_rows.push_back(row);
            owned_row_idx.push_back(i);
        }
        if (owned_rows.empty()) {
            return;
        }

        std::vector<GlobalIndex> owned_resolved(owned_rows.size() * col_dofs.size(),
                                                INVALID_GLOBAL_INDEX);
        base_->resolveMatrixEntries(std::span<const GlobalIndex>(owned_rows),
                                    col_dofs,
                                    std::span<GlobalIndex>(owned_resolved));
        for (std::size_t oi = 0; oi < owned_row_idx.size(); ++oi) {
            const std::size_t row_i = owned_row_idx[oi];
            const std::size_t src_base = oi * col_dofs.size();
            const std::size_t dst_base = row_i * col_dofs.size();
            std::copy_n(owned_resolved.data() + src_base,
                        static_cast<std::ptrdiff_t>(col_dofs.size()),
                        resolved.data() + dst_base);
        }
    }

    void addMatrixEntriesResolved(std::span<const GlobalIndex> row_dofs,
                                  std::span<const GlobalIndex> col_dofs,
                                  std::span<const GlobalIndex> resolved,
                                  std::span<const Real> local_matrix,
                                  AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "OwnedRowOnlyView::base");
        FE_THROW_IF(local_matrix.size() != row_dofs.size() * col_dofs.size() ||
                        resolved.size() != local_matrix.size(),
                    FEException,
                    "OwnedRowOnlyView::addMatrixEntriesResolved: size mismatch");

        bool all_owned = true;
        for (const auto row : row_dofs) {
            if (!isOwnedRow(row)) {
                all_owned = false;
                break;
            }
        }
        if (all_owned) {
            base_->addMatrixEntriesResolved(row_dofs, col_dofs, resolved, local_matrix, mode);
            return;
        }

        owned_rows_.clear();
        owned_values_.clear();
        owned_rows_.reserve(row_dofs.size());
        owned_values_.reserve(local_matrix.size());
        std::vector<GlobalIndex> owned_resolved;
        owned_resolved.reserve(local_matrix.size());

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            const auto row = row_dofs[i];
            if (!isOwnedRow(row)) {
                continue;
            }
            owned_rows_.push_back(row);
            const std::size_t base_idx = i * col_dofs.size();
            owned_values_.insert(owned_values_.end(),
                                 local_matrix.begin() + static_cast<std::ptrdiff_t>(base_idx),
                                 local_matrix.begin() + static_cast<std::ptrdiff_t>(base_idx + col_dofs.size()));
            owned_resolved.insert(owned_resolved.end(),
                                  resolved.begin() + static_cast<std::ptrdiff_t>(base_idx),
                                  resolved.begin() + static_cast<std::ptrdiff_t>(base_idx + col_dofs.size()));
        }

        if (!owned_rows_.empty()) {
            base_->addMatrixEntriesResolved(std::span<const GlobalIndex>(owned_rows_),
                                            col_dofs,
                                            std::span<const GlobalIndex>(owned_resolved),
                                            std::span<const Real>(owned_values_),
                                            mode);
        }
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

// Resolve dynamic kernel type once (outside element/face loops) so the compiler
// can devirtualize hot `compute*` calls for common final kernels.
template <typename F>
decltype(auto) withDevirtualizedKernel(AssemblyKernel& kernel, F&& fn)
{
    if (auto* mass_kernel = dynamic_cast<MassKernel*>(&kernel); mass_kernel != nullptr) {
        return std::forward<F>(fn)(*mass_kernel);
    }
    if (auto* stiffness_kernel = dynamic_cast<StiffnessKernel*>(&kernel); stiffness_kernel != nullptr) {
        return std::forward<F>(fn)(*stiffness_kernel);
    }
    if (auto* source_kernel = dynamic_cast<SourceKernel*>(&kernel); source_kernel != nullptr) {
        return std::forward<F>(fn)(*source_kernel);
    }
    if (auto* poisson_kernel = dynamic_cast<PoissonKernel*>(&kernel); poisson_kernel != nullptr) {
        return std::forward<F>(fn)(*poisson_kernel);
    }
    if (auto* composite_kernel = dynamic_cast<CompositeKernel*>(&kernel); composite_kernel != nullptr) {
        return std::forward<F>(fn)(*composite_kernel);
    }

    if (auto* form_kernel = dynamic_cast<forms::FormKernel*>(&kernel); form_kernel != nullptr) {
        return std::forward<F>(fn)(*form_kernel);
    }
    if (auto* linear_kernel = dynamic_cast<forms::LinearFormKernel*>(&kernel); linear_kernel != nullptr) {
        return std::forward<F>(fn)(*linear_kernel);
    }
    if (auto* nonlinear_kernel = dynamic_cast<forms::NonlinearFormKernel*>(&kernel); nonlinear_kernel != nullptr) {
        return std::forward<F>(fn)(*nonlinear_kernel);
    }
    if (auto* symbolic_nonlinear_kernel = dynamic_cast<forms::SymbolicNonlinearFormKernel*>(&kernel);
        symbolic_nonlinear_kernel != nullptr) {
        return std::forward<F>(fn)(*symbolic_nonlinear_kernel);
    }
    if (auto* coupled_sensitivity_kernel = dynamic_cast<forms::CoupledResidualSensitivityKernel*>(&kernel);
        coupled_sensitivity_kernel != nullptr) {
        return std::forward<F>(fn)(*coupled_sensitivity_kernel);
    }
    if (auto* boundary_gradient_kernel = dynamic_cast<forms::BoundaryFunctionalGradientKernel*>(&kernel);
        boundary_gradient_kernel != nullptr) {
        return std::forward<F>(fn)(*boundary_gradient_kernel);
    }
    if (auto* jit_kernel = dynamic_cast<forms::jit::JITKernelWrapper*>(&kernel); jit_kernel != nullptr) {
        return std::forward<F>(fn)(*jit_kernel);
    }

    return std::forward<F>(fn)(kernel);
}

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
    cell_dof_tables_.clear();
    cell_resolved_vector_tables_.clear();
    cell_resolved_matrix_tables_.clear();
    field_access_plans_.clear();
}

void StandardAssembler::setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset)
{
    row_dof_map_ = &dof_map;
    row_dof_offset_ = row_offset;
    // NOTE: Do NOT clear cell_dof_tables_, cell_resolved_*_tables_, or
    // field_access_plans_ here.  These caches are keyed by (dof_map_ptr, offset)
    // and remain valid when the assembler's "default" DOF maps change.
    // Clearing them forces expensive rebuilds (0.2s per call for resolved
    // matrix tables on 75K-cell meshes).
}

void StandardAssembler::setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset)
{
    col_dof_map_ = &dof_map;
    col_dof_offset_ = col_offset;
    // See setRowDofMap note above.
}

void StandardAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    dof_handler_ = &dof_handler;
    row_dof_map_ = &dof_handler.getDofMap();
    col_dof_map_ = row_dof_map_;
    row_dof_offset_ = 0;
    col_dof_offset_ = 0;
    cell_dof_tables_.clear();
    cell_resolved_vector_tables_.clear();
    cell_resolved_matrix_tables_.clear();
    field_access_plans_.clear();
}

void StandardAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    constraints_ = constraints;
    cell_constrained_flags_valid_ = false;
    cached_field_recipes_valid_ = false;

    if (constraints_ && constraints_->isClosed()) {
        constraint_distributor_ = std::make_unique<constraints::ConstraintDistributor>(*constraints_);
    } else {
        constraint_distributor_.reset();
    }
}

void StandardAssembler::setSuppressConstraintInhomogeneity(bool suppress)
{
    suppress_constraint_inhomogeneity_ = suppress;
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
    cell_resolved_vector_tables_.clear();
}

void StandardAssembler::setCurrentSolutionView(const GlobalSystemView* solution_view)
{
    if (current_solution_view_ != solution_view) {
        cell_resolved_vector_tables_.clear();
    }
    current_solution_view_ = solution_view;
}

void StandardAssembler::setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields)
{
    // Check if the access list actually changed before clearing expensive caches.
    // This function is called on every assembleOperator call; clearing forces
    // 0.2s+ table rebuilds for 75K-cell meshes.
    bool changed = (fields.size() != field_solution_access_.size());
    if (!changed) {
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (fields[i].field != field_solution_access_[i].field ||
                fields[i].space != field_solution_access_[i].space ||
                fields[i].dof_map != field_solution_access_[i].dof_map ||
                fields[i].dof_offset != field_solution_access_[i].dof_offset) {
                changed = true;
                break;
            }
        }
    }

    field_solution_access_.assign(fields.begin(), fields.end());

    if (changed) {
        // Only clear plans (which store CellDofTable* pointers that could be
        // invalidated).  DOF tables and resolved tables are keyed by DofMap
        // pointers and remain valid.
        field_access_plans_.clear();
    }
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
    cell_resolved_vector_tables_.clear();
}

void StandardAssembler::setPreviousSolutionViewK(int k, const GlobalSystemView* solution_view)
{
    FE_THROW_IF(k <= 0, FEException, "StandardAssembler::setPreviousSolutionViewK: k must be >= 1");
    if (previous_solution_views_.size() < static_cast<std::size_t>(k)) {
        previous_solution_views_.resize(static_cast<std::size_t>(k), nullptr);
    }
    const bool view_changed =
        (previous_solution_views_[static_cast<std::size_t>(k - 1)] != solution_view);
    previous_solution_views_[static_cast<std::size_t>(k - 1)] = solution_view;

    if (previous_solutions_.size() < static_cast<std::size_t>(k)) {
        previous_solutions_.resize(static_cast<std::size_t>(k));
    }
    if (view_changed) {
        cell_resolved_vector_tables_.clear();
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
    auxiliary_inputs_ = integrals;
    auxiliary_state_ = aux_state;
}

void StandardAssembler::setAuxiliaryValues(std::span<const Real> inputs,
                                            std::span<const Real> state,
                                            std::span<const Real> outputs) noexcept
{
    auxiliary_inputs_ = inputs;
    auxiliary_state_ = state;
    auxiliary_outputs_ = outputs;
    coupled_integrals_ = inputs;
    coupled_aux_state_ = state;
}

void StandardAssembler::setAuxiliaryOutputBindings(
    std::span<const AuxiliaryOutputBinding> bindings) noexcept
{
    auxiliary_output_bindings_ = bindings;
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

    // Pre-allocate field solution storage to avoid per-cell heap allocations.
    // Use conservative estimates: 8 fields, 27 qpts, 3D vector values.
    context_.preAllocateFieldSolutionData(/*max_fields=*/8, /*max_qpts=*/est_qpts, /*max_value_dim=*/3);

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
    cached_mapping_.reset();
    cached_mapping_type_ = ElementType::Unknown;
    cached_mapping_order_ = -1;
    cached_mapping_affine_ = false;
    cached_geom_h_ = 0.0;
    cached_geom_volume_ = 0.0;
    cached_cell_dof_mesh_ = nullptr;
    cached_cell_dof_count_ = 0;
    cell_dof_tables_.clear();
    cell_resolved_vector_tables_.clear();
    cell_resolved_matrix_tables_.clear();
    field_access_plans_.clear();
    cell_constrained_flags_valid_ = false;
    initialized_ = false;
}

void StandardAssembler::ensureCellDofTables(const IMeshAccess& mesh)
{
    const auto n_cells = mesh.numCells();
    if (cached_cell_dof_mesh_ != &mesh || cached_cell_dof_count_ != n_cells) {
        cached_cell_dof_mesh_ = &mesh;
        cached_cell_dof_count_ = n_cells;
        cell_dof_tables_.clear();
        cell_resolved_vector_tables_.clear();
        cell_resolved_matrix_tables_.clear();
        field_access_plans_.clear();
        cell_constrained_flags_valid_ = false;
    }
}

const StandardAssembler::CellDofTable& StandardAssembler::getCellDofTable(
    const IMeshAccess& mesh,
    const dofs::DofMap* dof_map,
    GlobalIndex dof_offset
)
{
    FE_CHECK_NOT_NULL(dof_map, "StandardAssembler::getCellDofTable: dof_map");
    ensureCellDofTables(mesh);

    for (const auto& table : cell_dof_tables_) {
        if (table.dof_map == dof_map && table.dof_offset == dof_offset) {
            return table;
        }
    }

    // NOTE: Do NOT clear cell_resolved_*_tables_ here.  Resolved tables are
    // keyed by (layout_handle, dof_map_ptr, offset) and remain valid when new
    // DOF tables are added.  Clearing forces expensive rebuilds.
    // field_access_plans_ may reference CellDofTable pointers; clear if needed.
    field_access_plans_.clear();
    auto& table = cell_dof_tables_.emplace_back();
    table.dof_map = dof_map;
    table.dof_offset = dof_offset;
    table.cell_offsets.resize(static_cast<std::size_t>(std::max<GlobalIndex>(0, mesh.numCells())) + 1u, 0);

    std::size_t total_dofs = 0;
    for (GlobalIndex cell_id = 0; cell_id < mesh.numCells(); ++cell_id) {
        table.cell_offsets[static_cast<std::size_t>(cell_id)] = static_cast<GlobalIndex>(total_dofs);
        total_dofs += dof_map->getCellDofs(cell_id).size();
    }
    table.cell_offsets.back() = static_cast<GlobalIndex>(total_dofs);
    table.dofs.resize(total_dofs);

    for (GlobalIndex cell_id = 0; cell_id < mesh.numCells(); ++cell_id) {
        const auto src = dof_map->getCellDofs(cell_id);
        const auto begin = static_cast<std::size_t>(
            table.cell_offsets[static_cast<std::size_t>(cell_id)]);
        for (std::size_t i = 0; i < src.size(); ++i) {
            table.dofs[begin + i] = src[i] + dof_offset;
        }
    }

    return table;
}

std::span<const GlobalIndex> StandardAssembler::getCellDofsCached(
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const dofs::DofMap* dof_map,
    GlobalIndex dof_offset)
{
    const auto& table = getCellDofTable(mesh, dof_map, dof_offset);
    FE_THROW_IF(cell_id < 0 || cell_id >= cached_cell_dof_count_, FEException,
                "StandardAssembler::getCellDofsCached: cell_id out of range");
    const auto begin = static_cast<std::size_t>(table.cell_offsets[static_cast<std::size_t>(cell_id)]);
    const auto end = static_cast<std::size_t>(table.cell_offsets[static_cast<std::size_t>(cell_id) + 1u]);
    return std::span<const GlobalIndex>(table.dofs.data() + begin, end - begin);
}

std::span<const GlobalIndex> StandardAssembler::getCellDofsFromTable(
    const CellDofTable& table,
    GlobalIndex cell_id) const
{
    FE_THROW_IF(cell_id < 0 || cell_id >= cached_cell_dof_count_, FEException,
                "StandardAssembler::getCellDofsFromTable: cell_id out of range");
    const auto begin = static_cast<std::size_t>(table.cell_offsets[static_cast<std::size_t>(cell_id)]);
    const auto end = static_cast<std::size_t>(table.cell_offsets[static_cast<std::size_t>(cell_id) + 1u]);
    return std::span<const GlobalIndex>(table.dofs.data() + begin, end - begin);
}

void StandardAssembler::ensureResolvedVectorTable(
    const IMeshAccess& mesh,
    const dofs::DofMap* dof_map,
    GlobalIndex dof_offset,
    const GlobalSystemView* view)
{
    if (view == nullptr || dof_map == nullptr) {
        return;
    }

    const void* layout_handle = view->vectorLayoutHandle();
    if (layout_handle == nullptr) {
        return;
    }

    for (const auto& table : cell_resolved_vector_tables_) {
        if (table.layout_handle == layout_handle &&
            table.dof_map == dof_map &&
            table.dof_offset == dof_offset) {
            return;
        }
    }

    const auto& dof_table = getCellDofTable(mesh, dof_map, dof_offset);
    auto& table = cell_resolved_vector_tables_.emplace_back();
    table.layout_handle = layout_handle;
    table.dof_map = dof_map;
    table.dof_offset = dof_offset;
    table.resolved.resize(dof_table.dofs.size());

    for (GlobalIndex cell_id = 0; cell_id < cached_cell_dof_count_; ++cell_id) {
        const auto begin = static_cast<std::size_t>(
            dof_table.cell_offsets[static_cast<std::size_t>(cell_id)]);
        const auto end = static_cast<std::size_t>(
            dof_table.cell_offsets[static_cast<std::size_t>(cell_id) + 1u]);
        view->resolveVectorEntries(
            std::span<const GlobalIndex>(dof_table.dofs.data() + begin, end - begin),
            std::span<GlobalIndex>(table.resolved.data() + begin, end - begin));
    }
}

void StandardAssembler::ensureResolvedVectorTables(const IMeshAccess& mesh)
{
    ensureCellDofTables(mesh);
    if (current_solution_view_ != nullptr) {
        for (const auto& table : cell_dof_tables_) {
            ensureResolvedVectorTable(mesh, table.dof_map, table.dof_offset, current_solution_view_);
        }
    }
}

void StandardAssembler::ensureResolvedMatrixTable(
    const IMeshAccess& mesh,
    const dofs::DofMap* row_dof_map,
    GlobalIndex row_dof_offset,
    const dofs::DofMap* col_dof_map,
    GlobalIndex col_dof_offset,
    const GlobalSystemView* view)
{
    if (view == nullptr || row_dof_map == nullptr || col_dof_map == nullptr) {
        return;
    }

    const void* layout_handle = view->matrixLayoutHandle();
    if (layout_handle == nullptr) {
        return;
    }

    for (const auto& table : cell_resolved_matrix_tables_) {
        if (table.layout_handle == layout_handle &&
            table.row_dof_map == row_dof_map &&
            table.row_dof_offset == row_dof_offset &&
            table.col_dof_map == col_dof_map &&
            table.col_dof_offset == col_dof_offset) {
            return;
        }
    }

    const auto& row_table = getCellDofTable(mesh, row_dof_map, row_dof_offset);
    const auto& col_table = getCellDofTable(mesh, col_dof_map, col_dof_offset);

    auto& table = cell_resolved_matrix_tables_.emplace_back();
    table.layout_handle = layout_handle;
    table.row_dof_map = row_dof_map;
    table.row_dof_offset = row_dof_offset;
    table.col_dof_map = col_dof_map;
    table.col_dof_offset = col_dof_offset;
    table.cell_offsets.resize(static_cast<std::size_t>(cached_cell_dof_count_) + 1u, 0);

    std::size_t total_entries = 0;
    for (GlobalIndex cell_id = 0; cell_id < cached_cell_dof_count_; ++cell_id) {
        const auto row_dofs = getCellDofsFromTable(row_table, cell_id);
        const auto col_dofs = getCellDofsFromTable(col_table, cell_id);
        table.cell_offsets[static_cast<std::size_t>(cell_id)] = static_cast<GlobalIndex>(total_entries);
        total_entries += row_dofs.size() * col_dofs.size();
    }
    table.cell_offsets.back() = static_cast<GlobalIndex>(total_entries);
    table.resolved.resize(total_entries);

    for (GlobalIndex cell_id = 0; cell_id < cached_cell_dof_count_; ++cell_id) {
        const auto row_dofs = getCellDofsFromTable(row_table, cell_id);
        const auto col_dofs = getCellDofsFromTable(col_table, cell_id);
        const auto begin = static_cast<std::size_t>(
            table.cell_offsets[static_cast<std::size_t>(cell_id)]);
        view->resolveMatrixEntries(
            row_dofs,
            col_dofs,
            std::span<GlobalIndex>(table.resolved.data() + begin,
                                   row_dofs.size() * col_dofs.size()));
    }
}

void StandardAssembler::ensureCellConstrainedFlags(const IMeshAccess& mesh)
{
    if (cell_constrained_flags_valid_) return;
    if (!constraints_ || !options_.use_constraints || !constraint_distributor_) {
        cell_constrained_flags_valid_ = true;
        return;
    }

    ensureCellDofTables(mesh);
    const auto n_cells = cached_cell_dof_count_;
    cell_constrained_flags_.resize(static_cast<std::size_t>(n_cells), 0);

    for (GlobalIndex cell_id = 0; cell_id < n_cells; ++cell_id) {
        bool any_constrained = false;
        for (const auto& table : cell_dof_tables_) {
            const auto dofs = getCellDofsFromTable(table, cell_id);
            if (constraints_->hasConstrainedDofs(dofs)) {
                any_constrained = true;
                break;
            }
        }
        cell_constrained_flags_[static_cast<std::size_t>(cell_id)] =
            any_constrained ? static_cast<std::uint8_t>(1u) : static_cast<std::uint8_t>(0u);
    }
    cell_constrained_flags_valid_ = true;
}

std::span<const GlobalIndex> StandardAssembler::getResolvedCellVectorEntries(
    GlobalIndex cell_id,
    const dofs::DofMap* dof_map,
    GlobalIndex dof_offset,
    const GlobalSystemView* view) const
{
    if (view == nullptr || dof_map == nullptr || cell_id < 0 || cell_id >= cached_cell_dof_count_) {
        return {};
    }

    const void* layout_handle = view->vectorLayoutHandle();
    if (layout_handle == nullptr) {
        return {};
    }

    const CellDofTable* dof_table = nullptr;
    for (const auto& table : cell_dof_tables_) {
        if (table.dof_map == dof_map && table.dof_offset == dof_offset) {
            dof_table = &table;
            break;
        }
    }
    if (dof_table == nullptr) {
        return {};
    }

    for (const auto& table : cell_resolved_vector_tables_) {
        if (table.layout_handle == layout_handle &&
            table.dof_map == dof_map &&
            table.dof_offset == dof_offset) {
            const auto begin = static_cast<std::size_t>(
                dof_table->cell_offsets[static_cast<std::size_t>(cell_id)]);
            const auto end = static_cast<std::size_t>(
                dof_table->cell_offsets[static_cast<std::size_t>(cell_id) + 1u]);
            return std::span<const GlobalIndex>(table.resolved.data() + begin, end - begin);
        }
    }

    return {};
}

std::span<const GlobalIndex> StandardAssembler::getResolvedCellMatrixEntries(
    GlobalIndex cell_id,
    const dofs::DofMap* row_dof_map,
    GlobalIndex row_dof_offset,
    const dofs::DofMap* col_dof_map,
    GlobalIndex col_dof_offset,
    const GlobalSystemView* view) const
{
    if (view == nullptr || row_dof_map == nullptr || col_dof_map == nullptr ||
        cell_id < 0 || cell_id >= cached_cell_dof_count_) {
        return {};
    }

    const void* layout_handle = view->matrixLayoutHandle();
    if (layout_handle == nullptr) {
        return {};
    }

    for (const auto& table : cell_resolved_matrix_tables_) {
        if (table.layout_handle == layout_handle &&
            table.row_dof_map == row_dof_map &&
            table.row_dof_offset == row_dof_offset &&
            table.col_dof_map == col_dof_map &&
            table.col_dof_offset == col_dof_offset) {
            const auto begin = static_cast<std::size_t>(
                table.cell_offsets[static_cast<std::size_t>(cell_id)]);
            const auto end = static_cast<std::size_t>(
                table.cell_offsets[static_cast<std::size_t>(cell_id) + 1u]);
            return std::span<const GlobalIndex>(table.resolved.data() + begin, end - begin);
        }
    }

    return {};
}

void StandardAssembler::ensureFieldAccessPlans(const IMeshAccess& mesh)
{
    ensureCellDofTables(mesh);
    if (field_access_plans_.size() == field_solution_access_.size()) {
        return;
    }

    field_access_plans_.clear();
    field_access_plans_.reserve(field_solution_access_.size());
    for (const auto& access : field_solution_access_) {
        FE_CHECK_NOT_NULL(access.space, "StandardAssembler::ensureFieldAccessPlans: field space");
        FE_CHECK_NOT_NULL(access.dof_map, "StandardAssembler::ensureFieldAccessPlans: field dof_map");
        field_access_plans_.push_back(FieldAccessPlan{
            access.field,
            access.space,
            access.dof_map,
            access.dof_offset,
            &getCellDofTable(mesh, access.dof_map, access.dof_offset),
            access.space->field_type(),
            access.space->space_type() == spaces::SpaceType::Product,
            access.space->value_dimension(),
        });
    }
}

const StandardAssembler::FieldAccessPlan* StandardAssembler::findFieldAccessPlan(FieldId field) const noexcept
{
    for (const auto& plan : field_access_plans_) {
        if (plan.field == field) {
            return &plan;
        }
    }
    return nullptr;
}

void StandardAssembler::gatherCellVectorCoefficients(
    GlobalIndex cell_id,
    const dofs::DofMap* dof_map,
    GlobalIndex dof_offset,
    std::span<const GlobalIndex> dofs,
    const GlobalSystemView* view,
    std::span<const Real> raw_values,
    std::vector<Real>& out,
    const char* error_prefix,
    bool validate_negative_dofs)
{
    auto resolved = getResolvedCellVectorEntries(cell_id, dof_map, dof_offset, view);
    if (resolved.empty() && view != nullptr && cached_cell_dof_mesh_ != nullptr &&
        dof_map != nullptr) {
        ensureResolvedVectorTable(*cached_cell_dof_mesh_, dof_map, dof_offset, view);
        resolved = getResolvedCellVectorEntries(cell_id, dof_map, dof_offset, view);
    }
    gatherVectorCoefficients(dofs, view, raw_values, out, nullptr,
                             error_prefix, validate_negative_dofs, resolved);
}

std::span<const Real> StandardAssembler::gatherCachedCellVectorCoefficients(
    std::deque<CellCoefficientCacheEntry>& cache,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const dofs::DofMap* dof_map,
    GlobalIndex dof_offset,
    const spaces::FunctionSpace* space,
    std::span<const GlobalIndex> dofs,
    int history_index,
    bool localized_vector_basis,
    const char* error_prefix)
{
    FE_THROW_IF(history_index < 0, InvalidArgumentException,
                "gatherCachedCellVectorCoefficients: history index must be non-negative");

    auto find_cached =
        [&](bool localized) -> CellCoefficientCacheEntry* {
            for (auto& entry : cache) {
                if (entry.dof_map == dof_map &&
                    entry.dof_offset == dof_offset &&
                    entry.space == space &&
                    entry.history_index == history_index &&
                    entry.localized_vector_basis == localized) {
                    return &entry;
                }
            }
            return nullptr;
        };

    if (auto* existing = find_cached(localized_vector_basis)) {
        return std::span<const Real>(existing->coeffs);
    }

    if (localized_vector_basis) {
        if (auto* raw_entry = find_cached(false)) {
            auto& localized_entry = cache.emplace_back();
            localized_entry.dof_map = dof_map;
            localized_entry.dof_offset = dof_offset;
            localized_entry.space = space;
            localized_entry.history_index = history_index;
            localized_entry.localized_vector_basis = true;
            localized_entry.coeffs = raw_entry->coeffs;
            FE_CHECK_NOT_NULL(space, "gatherCachedCellVectorCoefficients: vector-basis space");
            applyVectorBasisGlobalToLocal(
                mesh, cell_id, *space, std::span<Real>(localized_entry.coeffs));
            return std::span<const Real>(localized_entry.coeffs);
        }
    }

    const GlobalSystemView* source_view = current_solution_view_;
    std::span<const Real> source_values = current_solution_;
    if (history_index > 0) {
        const auto idx = static_cast<std::size_t>(history_index - 1);
        FE_THROW_IF(idx >= previous_solutions_.size(), FEException,
                    "gatherCachedCellVectorCoefficients: requested history state is not available");
        source_view =
            (idx < previous_solution_views_.size()) ? previous_solution_views_[idx] : nullptr;
        source_values = previous_solutions_[idx];
        FE_THROW_IF(source_view == nullptr && source_values.empty(), FEException,
                    "gatherCachedCellVectorCoefficients: previous solution source is not available");
    } else {
        FE_THROW_IF(source_view == nullptr && source_values.empty(), FEException,
                    "gatherCachedCellVectorCoefficients: current solution source is not available");
    }

    auto& entry = cache.emplace_back();
    entry.dof_map = dof_map;
    entry.dof_offset = dof_offset;
    entry.space = space;
    entry.history_index = history_index;
    entry.localized_vector_basis = localized_vector_basis;
    gatherCellVectorCoefficients(
        cell_id, dof_map, dof_offset, dofs, source_view, source_values,
        entry.coeffs, error_prefix, false);
    if (localized_vector_basis) {
        FE_CHECK_NOT_NULL(space, "gatherCachedCellVectorCoefficients: localized vector-basis space");
        applyVectorBasisGlobalToLocal(
            mesh, cell_id, *space, std::span<Real>(entry.coeffs));
    }
    return std::span<const Real>(entry.coeffs);
}

void StandardAssembler::insertLocalForCell(
    GlobalIndex cell_id,
    const dofs::DofMap* row_dof_map,
    GlobalIndex row_dof_offset,
    const dofs::DofMap* col_dof_map,
    GlobalIndex col_dof_offset,
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (options_.use_constraints && constraint_distributor_ && constraints_ != nullptr) {
        // Use pre-computed per-cell constrained flags when available (avoids
        // per-call hasConstrainedDofs hash-map/bitset lookups). Falls back to
        // the original per-DOF check if flags haven't been built yet.
        const bool is_constrained =
            (cell_constrained_flags_valid_ &&
             cell_id >= 0 &&
             static_cast<std::size_t>(cell_id) < cell_constrained_flags_.size())
            ? (cell_constrained_flags_[static_cast<std::size_t>(cell_id)] != 0)
            : (constraints_->hasConstrainedDofs(row_dofs) ||
               constraints_->hasConstrainedDofs(col_dofs));
        if (is_constrained) {
            insertLocalConstrained(output, row_dofs, col_dofs, matrix_view, vector_view);
            return;
        }
    }

    // Use pre-resolved CSR slots when available (avoids hash probes per insertion).
    const auto resolved = getResolvedCellMatrixEntries(
        cell_id, row_dof_map, row_dof_offset, col_dof_map, col_dof_offset, matrix_view);
    const auto resolved_vector = getResolvedCellVectorEntries(
        cell_id, row_dof_map, row_dof_offset, vector_view);
    insertLocal(output, row_dofs, col_dofs, matrix_view, vector_view, resolved, resolved_vector);
}

void StandardAssembler::resizeCombinedInsertScratch(std::size_t batch_size, int combined_n)
{
    const auto mat_stride = static_cast<std::size_t>(combined_n) *
                            static_cast<std::size_t>(combined_n);
    const auto vec_stride = static_cast<std::size_t>(combined_n);
    scratch_fused_matrices_.resize(batch_size * mat_stride);
    scratch_fused_vectors_.resize(batch_size * vec_stride);
    scratch_fused_dofs_.resize(batch_size * vec_stride);
}

void StandardAssembler::zeroCombinedInsertScratch(std::size_t active, int combined_n)
{
    const auto mat_stride = static_cast<std::size_t>(combined_n) *
                            static_cast<std::size_t>(combined_n);
    const auto vec_stride = static_cast<std::size_t>(combined_n);
    for (std::size_t slot = 0; slot < active; ++slot) {
        std::fill_n(scratch_fused_matrices_.data() + slot * mat_stride,
                    mat_stride, Real(0));
        std::fill_n(scratch_fused_vectors_.data() + slot * vec_stride,
                    vec_stride, Real(0));
    }
}

void StandardAssembler::scatterCombinedInsertBlockOutput(
    std::size_t slot,
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    const CombinedInsertBlockInfo& info,
    int total_comps,
    int combined_n,
    bool want_matrix,
    bool want_vector)
{
    const auto mat_stride = static_cast<std::size_t>(combined_n) *
                            static_cast<std::size_t>(combined_n);
    const auto vec_stride = static_cast<std::size_t>(combined_n);
    Real* combined_matrix = scratch_fused_matrices_.data() + slot * mat_stride;
    Real* combined_vector = scratch_fused_vectors_.data() + slot * vec_stride;
    GlobalIndex* combined_dofs = scratch_fused_dofs_.data() + slot * vec_stride;

    const int n_rows = static_cast<int>(row_dofs.size());
    const int n_cols = static_cast<int>(col_dofs.size());

    if (output.has_matrix && want_matrix) {
        const Real* src = output.local_matrix.data();
        for (int i = 0; i < n_rows; ++i) {
            const int combined_i = (i / info.row_comps) * total_comps +
                                   info.row_comp_start + (i % info.row_comps);
            for (int j = 0; j < n_cols; ++j) {
                const int combined_j = (j / info.col_comps) * total_comps +
                                       info.col_comp_start + (j % info.col_comps);
                combined_matrix[static_cast<std::size_t>(combined_i) *
                                static_cast<std::size_t>(combined_n) +
                                static_cast<std::size_t>(combined_j)] +=
                    src[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_cols) +
                        static_cast<std::size_t>(j)];
            }
        }
    }

    if (output.has_vector && want_vector) {
        const Real* src = output.local_vector.data();
        for (int i = 0; i < n_rows; ++i) {
            const int combined_i = (i / info.row_comps) * total_comps +
                                   info.row_comp_start + (i % info.row_comps);
            combined_vector[static_cast<std::size_t>(combined_i)] +=
                src[static_cast<std::size_t>(i)];
        }
    }

    for (int i = 0; i < n_rows; ++i) {
        const int combined_i = (i / info.row_comps) * total_comps +
                               info.row_comp_start + (i % info.row_comps);
        combined_dofs[static_cast<std::size_t>(combined_i)] =
            row_dofs[static_cast<std::size_t>(i)];
    }
    for (int j = 0; j < n_cols; ++j) {
        const int combined_j = (j / info.col_comps) * total_comps +
                               info.col_comp_start + (j % info.col_comps);
        combined_dofs[static_cast<std::size_t>(combined_j)] =
            col_dofs[static_cast<std::size_t>(j)];
    }
}

void StandardAssembler::flushCombinedInsertBatch(
    std::span<const GlobalIndex> batch_cell_ids,
    int combined_n,
    const CombinedInsertTarget& target)
{
    const auto mat_stride = static_cast<std::size_t>(combined_n) *
                            static_cast<std::size_t>(combined_n);
    const auto vec_stride = static_cast<std::size_t>(combined_n);
    const bool use_constrained =
        options_.use_constraints && constraint_distributor_ != nullptr;

    KernelOutput fused_out;
    if (use_constrained) {
        fused_out.local_matrix.resize(mat_stride);
        fused_out.local_vector.resize(vec_stride);
    }

    for (std::size_t slot = 0; slot < batch_cell_ids.size(); ++slot) {
        auto dofs_span = std::span<const GlobalIndex>(
            scratch_fused_dofs_.data() + slot * vec_stride, vec_stride);
        auto mat_span = std::span<const Real>(
            scratch_fused_matrices_.data() + slot * mat_stride, mat_stride);
        auto vec_span = std::span<const Real>(
            scratch_fused_vectors_.data() + slot * vec_stride, vec_stride);

        bool skip_direct_insert = false;
        if (use_constrained && constraints_->hasConstrainedDofs(dofs_span)) {
            bool all_dirichlet = true;
            for (std::size_t d = 0; d < vec_stride; ++d) {
                if (constraints_->isConstrained(dofs_span[d])) {
                    auto cv = constraints_->getConstraint(dofs_span[d]);
                    if (cv && !cv->isDirichlet()) {
                        all_dirichlet = false;
                        break;
                    }
                }
            }

            if (all_dirichlet && dirichletFastPathEnabled()) {
                Real* mat_ptr = scratch_fused_matrices_.data() + slot * mat_stride;
                Real* vec_ptr = scratch_fused_vectors_.data() + slot * vec_stride;
                for (std::size_t d = 0; d < vec_stride; ++d) {
                    if (!constraints_->isConstrained(dofs_span[d])) {
                        continue;
                    }
                    for (std::size_t j = 0; j < vec_stride; ++j) {
                        mat_ptr[d * vec_stride + j] = 0.0;
                    }
                    for (std::size_t i = 0; i < vec_stride; ++i) {
                        mat_ptr[i * vec_stride + d] = 0.0;
                    }
                    mat_ptr[d * vec_stride + d] = 1.0;
                    vec_ptr[d] = 0.0;
                }
                mat_span = std::span<const Real>(mat_ptr, mat_stride);
                vec_span = std::span<const Real>(vec_ptr, vec_stride);
            } else {
                fused_out.has_matrix = target.assemble_matrix;
                fused_out.has_vector = target.assemble_vector;
                if (target.assemble_matrix) {
                    std::memcpy(fused_out.local_matrix.data(),
                                mat_span.data(), mat_stride * sizeof(Real));
                }
                if (target.assemble_vector) {
                    std::memcpy(fused_out.local_vector.data(),
                                vec_span.data(), vec_stride * sizeof(Real));
                }
                insertLocalConstrained(
                    fused_out, dofs_span, dofs_span,
                    target.matrix_view, target.vector_view);
                skip_direct_insert = true;
            }
        }

        if (skip_direct_insert) {
            continue;
        }

        if (target.assemble_matrix && target.matrix_view) {
            const auto cell_id = batch_cell_ids[slot];
            if (!scratch_fused_resolved_.empty() &&
                cell_id >= 0 &&
                static_cast<std::size_t>(cell_id) <
                    scratch_fused_resolved_offsets_.size() - 1u) {
                const auto offset = static_cast<std::size_t>(
                    scratch_fused_resolved_offsets_[static_cast<std::size_t>(cell_id)]);
                const auto end_offset = static_cast<std::size_t>(
                    scratch_fused_resolved_offsets_[static_cast<std::size_t>(cell_id) + 1u]);
                auto resolved_span = std::span<const GlobalIndex>(
                    scratch_fused_resolved_.data() + offset, end_offset - offset);
                target.matrix_view->addMatrixEntriesResolved(
                    dofs_span, dofs_span, resolved_span, mat_span,
                    assembly::AddMode::Add);
            } else {
                target.matrix_view->addMatrixEntries(
                    dofs_span, dofs_span, mat_span,
                    assembly::AddMode::Add);
            }
        }
        if (target.assemble_vector && target.vector_view) {
            target.vector_view->addVectorEntries(
                dofs_span, vec_span, assembly::AddMode::Add);
        }
    }
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
    ensureCellDofTables(mesh);

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
    (void)getCellDofTable(mesh, row_dof_map_, row_dof_offset_);
    (void)getCellDofTable(mesh, col_dof_map_, col_dof_offset_);
    for (const auto& access : field_solution_access_) {
        if (access.dof_map != nullptr) {
            (void)getCellDofTable(mesh, access.dof_map, access.dof_offset);
        }
    }
    ensureFieldAccessPlans(mesh);
    ensureResolvedVectorTables(mesh);

    // Ensure AssemblyContext uses the mesh spatial dimension (2D/3D).
    const LocalIndex max_dofs =
        std::max(row_dof_map_->getMaxDofsPerCell(),
                 col_dof_map_->getMaxDofsPerCell());
    constexpr LocalIndex max_qpts = 27;
    context_.reserve(max_dofs, max_qpts, mesh.dimension());

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

    auto assemble_boundary_faces_with_kernel = [&](auto& kernel_impl) {
        // Iterate over boundary faces with given marker
        mesh.forEachBoundaryFace(boundary_marker,
            [&](GlobalIndex face_id, GlobalIndex cell_id) {
                if (!owned_rows_only && !mesh.isOwnedCell(cell_id)) {
                    return;
                }
                // Get cell DOFs (rows/cols may come from different maps)
                const auto row_dofs =
                    getCellDofsCached(mesh, cell_id, row_dof_map_, row_dof_offset_);
                const auto col_dofs =
                    getCellDofsCached(mesh, cell_id, col_dof_map_, col_dof_offset_);

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
                context_.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
                context_.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
                context_.clearAllPreviousSolutionData();
                context_.setBoundaryMarker(boundary_marker);

		            if (need_solution) {
		                FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
		                            "StandardAssembler::assembleBoundaryFaces: kernel requires solution but no solution was set");
                        ResolvedVectorGatherCache resolved_cache;
		                local_solution_coeffs_.resize(col_dofs.size());
                        gatherVectorCoefficients(col_dofs, current_solution_view_, current_solution_,
                                                 local_solution_coeffs_, &resolved_cache,
                                                 "StandardAssembler::assembleBoundaryFaces", true);
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
                                    gatherVectorCoefficients(col_dofs, prev_view, prev, local_prev,
                                                             &resolved_cache,
                                                             "StandardAssembler::assembleBoundaryFaces", true);
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
                kernel_impl.computeBoundaryFace(context_, boundary_marker, kernel_output_);

                if (context_.testUsesVectorBasis() || context_.trialUsesVectorBasis()) {
                    applyVectorBasisOutputOrientation(mesh, cell_id, test_space, cell_id, trial_space, kernel_output_);
                }

                // Insert into global system
                if (options_.use_constraints && constraint_distributor_) {
                    insertLocalConstrained(kernel_output_, row_dofs, col_dofs,
                                           insert_matrix_view, insert_vector_view);
                } else {
                    insertLocal(kernel_output_, row_dofs, col_dofs,
                                insert_matrix_view, insert_vector_view);
                }

                result.boundary_faces_assembled++;
            });
    };

    withDevirtualizedKernel(kernel, [&](auto& kernel_impl) {
        assemble_boundary_faces_with_kernel(kernel_impl);
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
    ensureCellDofTables(mesh);

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
    (void)getCellDofTable(mesh, row_dof_map_, row_dof_offset_);
    (void)getCellDofTable(mesh, col_dof_map_, col_dof_offset_);
    for (const auto& access : field_solution_access_) {
        if (access.dof_map != nullptr) {
            (void)getCellDofTable(mesh, access.dof_map, access.dof_offset);
        }
    }
    ensureFieldAccessPlans(mesh);
    ensureResolvedVectorTables(mesh);

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
    std::span<const GlobalIndex> minus_row_dofs;
    std::span<const GlobalIndex> plus_row_dofs;
    std::span<const GlobalIndex> minus_col_dofs;
    std::span<const GlobalIndex> plus_col_dofs;
    std::vector<Real> plus_solution_coeffs;
    std::vector<std::vector<Real>> plus_prev_solution_coeffs;
    std::vector<GlobalIndex> cell_nodes_minus;
    std::vector<GlobalIndex> cell_nodes_plus;

    withDevirtualizedKernel(kernel, [&](auto& kernel_impl) {
        mesh.forEachInteriorFace(
            [&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
            if (!should_process(cell_minus, cell_plus)) {
                return;
            }
            // Get DOFs for both cells (rows/cols may differ)
            minus_row_dofs = getCellDofsCached(mesh, cell_minus, row_dof_map_, row_dof_offset_);
            plus_row_dofs = getCellDofsCached(mesh, cell_plus, row_dof_map_, row_dof_offset_);
            minus_col_dofs = getCellDofsCached(mesh, cell_minus, col_dof_map_, col_dof_offset_);
            plus_col_dofs = getCellDofsCached(mesh, cell_plus, col_dof_map_, col_dof_offset_);

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
            context_.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
            context_.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
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
            context_plus.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
            context_plus.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
            context_plus.clearAllPreviousSolutionData();

		            if (need_solution) {
		                FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
		                            "StandardAssembler::assembleInteriorFaces: kernel requires solution but no solution was set");
                        ResolvedVectorGatherCache minus_resolved_cache;
                        ResolvedVectorGatherCache plus_resolved_cache;
		
		                local_solution_coeffs_.resize(minus_col_dofs.size());
                        gatherVectorCoefficients(minus_col_dofs, current_solution_view_, current_solution_,
                                                 local_solution_coeffs_, &minus_resolved_cache,
                                                 "StandardAssembler::assembleInteriorFaces", true);
		                if (context_.trialUsesVectorBasis()) {
		                    applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
		                                                  std::span<Real>(local_solution_coeffs_));
	                }
	                context_.setSolutionCoefficients(local_solution_coeffs_);
		
		                plus_solution_coeffs.resize(plus_col_dofs.size());
                        gatherVectorCoefficients(plus_col_dofs, current_solution_view_, current_solution_,
                                                 plus_solution_coeffs, &plus_resolved_cache,
                                                 "StandardAssembler::assembleInteriorFaces", true);
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
                                    gatherVectorCoefficients(minus_col_dofs, prev_view, prev,
                                                             local_prev_minus, &minus_resolved_cache,
                                                             "StandardAssembler::assembleInteriorFaces", true);
		                            if (context_.trialUsesVectorBasis()) {
		                                applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
		                                                              std::span<Real>(local_prev_minus));
	                            }
                            context_.setPreviousSolutionCoefficientsK(k, local_prev_minus);

		                            auto& local_prev_plus = plus_prev_solution_coeffs[static_cast<std::size_t>(k - 1)];
                                    gatherVectorCoefficients(plus_col_dofs, prev_view, prev,
                                                             local_prev_plus, &plus_resolved_cache,
                                                             "StandardAssembler::assembleInteriorFaces", true);
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

            kernel_impl.computeInteriorFace(context_, context_plus,
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
    ensureCellDofTables(mesh);

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
    (void)getCellDofTable(mesh, row_dof_map_, row_dof_offset_);
    (void)getCellDofTable(mesh, col_dof_map_, col_dof_offset_);
    for (const auto& access : field_solution_access_) {
        if (access.dof_map != nullptr) {
            (void)getCellDofTable(mesh, access.dof_map, access.dof_offset);
        }
    }
    ensureFieldAccessPlans(mesh);
    ensureResolvedVectorTables(mesh);

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
    std::span<const GlobalIndex> minus_row_dofs;
    std::span<const GlobalIndex> plus_row_dofs;
    std::span<const GlobalIndex> minus_col_dofs;
    std::span<const GlobalIndex> plus_col_dofs;
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

    withDevirtualizedKernel(kernel, [&](auto& kernel_impl) {
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
        minus_row_dofs = getCellDofsCached(mesh, cell_minus, row_dof_map_, row_dof_offset_);
        plus_row_dofs = getCellDofsCached(mesh, cell_plus, row_dof_map_, row_dof_offset_);
        minus_col_dofs = getCellDofsCached(mesh, cell_minus, col_dof_map_, col_dof_offset_);
        plus_col_dofs = getCellDofsCached(mesh, cell_plus, col_dof_map_, col_dof_offset_);

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
        context_.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
        context_.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
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
        context_plus.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
        context_plus.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
        context_plus.clearAllPreviousSolutionData();

	        if (need_solution) {
	            FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
	                        "StandardAssembler::assembleInterfaceFaces: kernel requires solution but no solution was set");
            ResolvedVectorGatherCache minus_resolved_cache;
            ResolvedVectorGatherCache plus_resolved_cache;

	            local_solution_coeffs_.resize(minus_col_dofs.size());
                gatherVectorCoefficients(minus_col_dofs, current_solution_view_, current_solution_,
                                         local_solution_coeffs_, &minus_resolved_cache,
                                         "StandardAssembler::assembleInterfaceFaces", true);
	            if (context_.trialUsesVectorBasis()) {
	                applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
	                                              std::span<Real>(local_solution_coeffs_));
            }
            context_.setSolutionCoefficients(local_solution_coeffs_);

	            plus_solution_coeffs.resize(plus_col_dofs.size());
                gatherVectorCoefficients(plus_col_dofs, current_solution_view_, current_solution_,
                                         plus_solution_coeffs, &plus_resolved_cache,
                                         "StandardAssembler::assembleInterfaceFaces", true);
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
                                gatherVectorCoefficients(minus_col_dofs, prev_view, prev,
                                                         local_prev_minus, &minus_resolved_cache,
                                                         "StandardAssembler::assembleInterfaceFaces", true);
		                        if (context_.trialUsesVectorBasis()) {
		                            applyVectorBasisGlobalToLocal(mesh, cell_minus, trial_space,
		                                                          std::span<Real>(local_prev_minus));
	                        }
                        context_.setPreviousSolutionCoefficientsK(k, local_prev_minus);

		                        auto& local_prev_plus = plus_prev_solution_coeffs[static_cast<std::size_t>(k - 1)];
                                gatherVectorCoefficients(plus_col_dofs, prev_view, prev,
                                                         local_prev_plus, &plus_resolved_cache,
                                                         "StandardAssembler::assembleInterfaceFaces", true);
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

        kernel_impl.computeInterfaceFace(context_, context_plus, interface_marker,
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
    });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}
#endif

// ============================================================================
// Internal Implementation
// ============================================================================

#ifdef SVMP_FE_ASSEMBLY_TIMING
// File-scope accumulators for prepareContext sub-phase timing
static thread_local double g_pc_setup = 0.0;
static thread_local double g_pc_mapping = 0.0;
static thread_local double g_pc_resize = 0.0;
static thread_local double g_pc_jacobian = 0.0;
static thread_local double g_pc_basis = 0.0;
static thread_local double g_pc_ctx_config = 0.0;
static thread_local int g_pc_call_count = 0;
#endif

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
    ensureCellDofTables(mesh);

    // Begin assembly phase
    if (matrix_view && assemble_matrix && matrix_view->getPhase() == AssemblyPhase::NotStarted) {
        matrix_view->beginAssemblyPhase();
    }
    if (vector_view && assemble_vector && vector_view != matrix_view &&
        vector_view->getPhase() == AssemblyPhase::NotStarted) {
        vector_view->beginAssemblyPhase();
    }

    if (!kernel.hasCell()) {
        auto end_time = std::chrono::steady_clock::now();
        result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return result;
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
    (void)getCellDofTable(mesh, row_dof_map_, row_dof_offset_);
    (void)getCellDofTable(mesh, col_dof_map_, col_dof_offset_);
    for (const auto& access : field_solution_access_) {
        if (access.dof_map != nullptr) {
            (void)getCellDofTable(mesh, access.dof_map, access.dof_offset);
        }
    }
    ensureFieldAccessPlans(mesh);
    ensureResolvedVectorTables(mesh);
    if (assemble_matrix && matrix_view &&
        matrix_view->insertionCapabilities().resolved_matrix_entries) {
        ensureResolvedMatrixTable(mesh, row_dof_map_, row_dof_offset_,
                                  col_dof_map_, col_dof_offset_, matrix_view);
    }
    ensureCellConstrainedFlags(mesh);

    const LocalIndex max_dofs =
        std::max(row_dof_map_->getMaxDofsPerCell(), col_dof_map_->getMaxDofsPerCell());
    constexpr LocalIndex max_qpts = 27;

    // Ensure AssemblyContext uses the mesh spatial dimension (2D/3D). This affects
    // the runtime shapes of geometry tensors (J, Jinv, normals, etc.) used by Forms.
    context_.reserve(max_dofs, max_qpts, mesh.dimension());

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

    std::vector<GlobalIndex> cell_ids;
    for_each_cell([&](GlobalIndex cell_id) {
        cell_ids.push_back(cell_id);
    });

    const std::size_t requested_batch_size =
        (options_.use_batching && options_.batch_size > 1)
            ? static_cast<std::size_t>(options_.batch_size)
            : 1u;

    // Sub-phase timing accumulators for prepareContext breakdown
    double tp_sub_dofmap = 0.0, tp_sub_prepare_ctx = 0.0, tp_sub_solution = 0.0;
    double tp_sub_field_sol = 0.0, tp_sub_material = 0.0, tp_sub_setters = 0.0;
    auto TP_SUB = assemblyTimeNow;

    auto prepare_cell_data = [&](GlobalIndex cell_id,
                                 AssemblyContext& ctx,
                                 std::span<const GlobalIndex>& row_dofs,
                                 std::span<const GlobalIndex>& col_dofs) {
        double tp_s0 = TP_SUB();
        row_dofs = getCellDofsCached(mesh, cell_id, row_dof_map_, row_dof_offset_);
        col_dofs = getCellDofsCached(mesh, cell_id, col_dof_map_, col_dof_offset_);
        tp_sub_dofmap += TP_SUB() - tp_s0;

        tp_s0 = TP_SUB();
        prepareContext(ctx, mesh, cell_id, test_space, trial_space, required_data);
        tp_sub_prepare_ctx += TP_SUB() - tp_s0;

        tp_s0 = TP_SUB();
        ctx.setMaterialState(nullptr, nullptr, 0u, 0u);
        ctx.setTimeIntegrationContext(time_integration_);
        ctx.setTime(time_);
        ctx.setTimeStep(dt_);
        ctx.setRealParameterGetter(get_real_param_);
        ctx.setParameterGetter(get_param_);
        ctx.setUserData(user_data_);
        ctx.setJITConstants(jit_constants_);
        ctx.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
        ctx.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
        ctx.clearAllPreviousSolutionData();
        tp_sub_setters += TP_SUB() - tp_s0;

        FE_THROW_IF(row_dofs.size() != static_cast<std::size_t>(ctx.numTestDofs()), FEException,
                    "StandardAssembler::assembleCellsCore: row DOF count does not match test space element DOFs");
        FE_THROW_IF(col_dofs.size() != static_cast<std::size_t>(ctx.numTrialDofs()), FEException,
                    "StandardAssembler::assembleCellsCore: column DOF count does not match trial space element DOFs");

	        tp_s0 = TP_SUB();
	        if (need_solution) {
	            FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
	                        "StandardAssembler::assembleCellsCore: kernel requires solution but no solution was set");
	            local_solution_coeffs_.resize(col_dofs.size());
                gatherCellVectorCoefficients(cell_id, col_dof_map_, col_dof_offset_,
                                             col_dofs, current_solution_view_,
                                             current_solution_, local_solution_coeffs_,
                                             "StandardAssembler::assembleCellsCore", true);
	            if (ctx.trialUsesVectorBasis()) {
	                applyVectorBasisGlobalToLocal(mesh, cell_id, trial_space,
	                                              std::span<Real>(local_solution_coeffs_));
            }
            ctx.setSolutionCoefficients(local_solution_coeffs_);

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
                            gatherCellVectorCoefficients(cell_id, col_dof_map_, col_dof_offset_,
                                                         col_dofs, prev_view, prev, local_prev,
                                                         "StandardAssembler::assembleCellsCore", true);
	                        if (ctx.trialUsesVectorBasis()) {
	                            applyVectorBasisGlobalToLocal(mesh, cell_id, trial_space,
	                                                          std::span<Real>(local_prev));
                        }
                        ctx.setPreviousSolutionCoefficientsK(k, local_prev);
                    }
                }
            }
        }

        tp_sub_solution += TP_SUB() - tp_s0;

        tp_s0 = TP_SUB();
        if (need_field_solutions) {
            populateFieldSolutionData(ctx, mesh, cell_id, field_requirements);
        }
        tp_sub_field_sol += TP_SUB() - tp_s0;

        tp_s0 = TP_SUB();
        if (need_material_state) {
            auto view = material_state_provider_->getCellState(kernel, cell_id, ctx.numQuadraturePoints());
            FE_THROW_IF(!view, FEException,
                        "StandardAssembler::assembleCellsCore: material state provider returned null storage");
            FE_THROW_IF(view.bytes_per_qpt != material_state_spec.bytes_per_qpt, FEException,
                        "StandardAssembler::assembleCellsCore: material state bytes_per_qpt mismatch");
            FE_THROW_IF(view.stride_bytes < view.bytes_per_qpt, FEException,
                        "StandardAssembler::assembleCellsCore: invalid material state stride");
            ctx.setMaterialState(view.data_old, view.data_work, view.bytes_per_qpt, view.stride_bytes, view.alignment);
        }
        tp_sub_material += TP_SUB() - tp_s0;
    };

    auto insert_cell_output = [&](GlobalIndex cell_id,
                                  AssemblyContext& ctx,
                                  std::span<const GlobalIndex> row_dofs,
                                  std::span<const GlobalIndex> col_dofs,
                                  KernelOutput& output) {
        if (ctx.testUsesVectorBasis() || ctx.trialUsesVectorBasis()) {
            applyVectorBasisOutputOrientation(mesh, cell_id, test_space, cell_id, trial_space, output);
        }

        insertLocalForCell(cell_id, row_dof_map_, row_dof_offset_,
                           col_dof_map_, col_dof_offset_,
                           output, row_dofs, col_dofs,
                           assemble_matrix ? insert_matrix_view : nullptr,
                           assemble_vector ? insert_vector_view : nullptr);

        result.elements_assembled++;
        if (output.has_matrix) {
            result.matrix_entries_inserted +=
                static_cast<GlobalIndex>(row_dofs.size() * col_dofs.size());
        }
        if (output.has_vector) {
            result.vector_entries_inserted += static_cast<GlobalIndex>(row_dofs.size());
        }
    };

    auto assemble_cells_with_kernel = [&](auto& kernel_impl) {
        const bool use_batch_path =
            (requested_batch_size > 1u) && kernel_impl.supportsCellBatch();

        if (!use_batch_path) {
            double tp_prepare = 0.0, tp_kernel = 0.0, tp_insert = 0.0;
            auto TP = assemblyTimeNow;
            double tp0;
            std::span<const GlobalIndex> row_dofs;
            std::span<const GlobalIndex> col_dofs;

            for (const auto cell_id : cell_ids) {
                tp0 = TP();
                prepare_cell_data(cell_id, context_, row_dofs, col_dofs);
                tp_prepare += TP() - tp0;

                tp0 = TP();
                kernel_output_.clear();
                kernel_impl.computeCell(context_, kernel_output_);
                tp_kernel += TP() - tp0;

                tp0 = TP();
                insert_cell_output(cell_id, context_, row_dofs, col_dofs, kernel_output_);
                tp_insert += TP() - tp0;
            }

#ifdef SVMP_FE_ASSEMBLY_TIMING
            if (assemblyTimingEnabled())
            {
                int rank = 0;
#if FE_HAS_MPI
                int mpi_init = 0;
                MPI_Initialized(&mpi_init);
                if (mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
                if (rank == 0) {
                    const double total = tp_prepare + tp_kernel + tp_insert;
                    if (total > 1e-7) {
                        std::fprintf(stderr,
                            "    --- cellLoop TIMING (rank 0, %zu cells) ---\n"
                            "      Total:            %9.6f s\n"
                            "      prepareContext:    %9.6f s  (%5.1f%%)\n"
                            "        dofMap lookup:   %9.6f s  (%5.1f%%)\n"
                            "        geom+basis:      %9.6f s  (%5.1f%%)\n"
                            "        ctx setters:     %9.6f s  (%5.1f%%)\n"
                            "        solution gather: %9.6f s  (%5.1f%%)\n"
                            "        field solutions: %9.6f s  (%5.1f%%)\n"
                            "        material state:  %9.6f s  (%5.1f%%)\n"
                            "      computeCell:      %9.6f s  (%5.1f%%)\n"
                            "      insertLocal:      %9.6f s  (%5.1f%%)\n"
                            "    ------------------------------------\n",
                            cell_ids.size(),
                            total,
                            tp_prepare, 100.0 * tp_prepare / total,
                            tp_sub_dofmap,       100.0 * tp_sub_dofmap       / total,
                            tp_sub_prepare_ctx,  100.0 * tp_sub_prepare_ctx  / total,
                            tp_sub_setters,      100.0 * tp_sub_setters      / total,
                            tp_sub_solution,     100.0 * tp_sub_solution     / total,
                            tp_sub_field_sol,    100.0 * tp_sub_field_sol    / total,
                            tp_sub_material,     100.0 * tp_sub_material     / total,
                            tp_kernel,  100.0 * tp_kernel  / total,
                            tp_insert,  100.0 * tp_insert  / total);
                    }
                }
            }
#endif
            return;
        }

        std::vector<AssemblyContext> batch_contexts(requested_batch_size);
        for (auto& ctx : batch_contexts) {
            ctx.reserve(max_dofs, max_qpts, mesh.dimension());
        }
        std::vector<std::span<const GlobalIndex>> batch_row_dofs(requested_batch_size);
        std::vector<std::span<const GlobalIndex>> batch_col_dofs(requested_batch_size);
        std::vector<KernelOutput> batch_outputs(requested_batch_size);
        std::vector<const AssemblyContext*> batch_context_ptrs(requested_batch_size, nullptr);

        double tp_prepare = 0.0, tp_kernel = 0.0, tp_insert = 0.0;
        auto TP = assemblyTimeNow;

        auto assemble_batch_range = [&](std::span<const GlobalIndex> grouped_cell_ids) {
            for (std::size_t begin = 0; begin < grouped_cell_ids.size(); begin += requested_batch_size) {
                const std::size_t active = std::min(requested_batch_size, grouped_cell_ids.size() - begin);

                double tp0 = TP();
                for (std::size_t slot = 0; slot < active; ++slot) {
                    const auto cell_id = grouped_cell_ids[begin + slot];
                    prepare_cell_data(cell_id,
                                      batch_contexts[slot],
                                      batch_row_dofs[slot],
                                      batch_col_dofs[slot]);
                    batch_outputs[slot].clear();
                    batch_context_ptrs[slot] = &batch_contexts[slot];
                }
                tp_prepare += TP() - tp0;

                tp0 = TP();
                kernel_impl.computeCellBatch(
                    std::span<const AssemblyContext* const>(batch_context_ptrs.data(), active),
                    std::span<KernelOutput>(batch_outputs.data(), active));
                tp_kernel += TP() - tp0;

                tp0 = TP();
                for (std::size_t slot = 0; slot < active; ++slot) {
                    const auto cell_id = grouped_cell_ids[begin + slot];
                    insert_cell_output(cell_id,
                                       batch_contexts[slot],
                                       batch_row_dofs[slot],
                                       batch_col_dofs[slot],
                                       batch_outputs[slot]);
                }
                tp_insert += TP() - tp0;
            }
        };

        const bool allow_topology_reorder =
            !options_.stable_insertion_order && (options_.default_mode == AssemblyMode::Add);

        if (!allow_topology_reorder) {
            // Preserve traversal order: only form homogeneous batches over
            // contiguous runs of identical element topology.
            std::size_t run_begin = 0u;
            while (run_begin < cell_ids.size()) {
                const auto run_type = mesh.getCellType(cell_ids[run_begin]);
                std::size_t run_end = run_begin + 1u;
                while (run_end < cell_ids.size() && mesh.getCellType(cell_ids[run_end]) == run_type) {
                    ++run_end;
                }
                assemble_batch_range(std::span<const GlobalIndex>(cell_ids.data() + run_begin,
                                                                  run_end - run_begin));
                run_begin = run_end;
            }
        } else {
            // Higher-throughput path: globally group by topology when stable
            // insertion order is explicitly relaxed.
            std::vector<std::pair<ElementType, std::vector<GlobalIndex>>> topology_groups;
            for (const auto cell_id : cell_ids) {
                const auto cell_type = mesh.getCellType(cell_id);
                auto it = std::find_if(topology_groups.begin(), topology_groups.end(),
                                       [&](const auto& group) { return group.first == cell_type; });
                if (it == topology_groups.end()) {
                    topology_groups.emplace_back(cell_type, std::vector<GlobalIndex>{});
                    it = topology_groups.end() - 1;
                }
                it->second.push_back(cell_id);
            }

            for (const auto& group : topology_groups) {
                assemble_batch_range(std::span<const GlobalIndex>(group.second));
            }
        }

#ifdef SVMP_FE_ASSEMBLY_TIMING
        if (assemblyTimingEnabled())
        // Print batch path timing
        {
            int rank = 0;
#if FE_HAS_MPI
            int mpi_init = 0;
            MPI_Initialized(&mpi_init);
            if (mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
            if (rank == 0) {
                const double total = tp_prepare + tp_kernel + tp_insert;
                if (total > 1e-7) {
                    const double pc_total = g_pc_setup + g_pc_mapping + g_pc_resize
                                          + g_pc_jacobian + g_pc_basis + g_pc_ctx_config;
                    (void)pc_total;
                    std::fprintf(stderr,
                        "    --- cellLoop TIMING (rank 0, %zu cells, batch=%zu) ---\n"
                        "      Total:             %9.6f s\n"
                        "      prepare_cell_data: %9.6f s  (%5.1f%%)\n"
                        "        dofMap lookup:   %9.6f s  (%5.1f%%)\n"
                        "        prepareContext:  %9.6f s  (%5.1f%%)\n"
                        "          elem+quad:     %9.6f s  (%5.1f%%)\n"
                        "          mapping create:%9.6f s  (%5.1f%%)\n"
                        "          scratch resize:%9.6f s  (%5.1f%%)\n"
                        "          jacobian QP:   %9.6f s  (%5.1f%%)\n"
                        "          basis eval QP: %9.6f s  (%5.1f%%)\n"
                        "          ctx config:    %9.6f s  (%5.1f%%)\n"
                        "        ctx setters:     %9.6f s  (%5.1f%%)\n"
                        "        solution gather: %9.6f s  (%5.1f%%)\n"
                        "        field solutions: %9.6f s  (%5.1f%%)\n"
                        "        material state:  %9.6f s  (%5.1f%%)\n"
                        "      computeCellBatch: %9.6f s  (%5.1f%%)\n"
                        "      insertLocal:      %9.6f s  (%5.1f%%)\n"
                        "    ------------------------------------\n",
                        cell_ids.size(), requested_batch_size,
                        total,
                        tp_prepare, 100.0 * tp_prepare / total,
                        tp_sub_dofmap,       100.0 * tp_sub_dofmap       / total,
                        tp_sub_prepare_ctx,  100.0 * tp_sub_prepare_ctx  / total,
                        g_pc_setup,          100.0 * g_pc_setup          / total,
                        g_pc_mapping,        100.0 * g_pc_mapping        / total,
                        g_pc_resize,         100.0 * g_pc_resize         / total,
                        g_pc_jacobian,       100.0 * g_pc_jacobian       / total,
                        g_pc_basis,          100.0 * g_pc_basis          / total,
                        g_pc_ctx_config,     100.0 * g_pc_ctx_config     / total,
                        tp_sub_setters,      100.0 * tp_sub_setters      / total,
                        tp_sub_solution,     100.0 * tp_sub_solution     / total,
                        tp_sub_field_sol,    100.0 * tp_sub_field_sol    / total,
                        tp_sub_material,     100.0 * tp_sub_material     / total,
                        tp_kernel,  100.0 * tp_kernel  / total,
                        tp_insert,  100.0 * tp_insert  / total);
                }
            }
            // Reset prepareContext accumulators for next call
            g_pc_setup = g_pc_mapping = g_pc_resize = 0.0;
            g_pc_jacobian = g_pc_basis = g_pc_ctx_config = 0.0;
            g_pc_call_count = 0;
        }
#endif
    };

    withDevirtualizedKernel(kernel, [&](auto& kernel_impl) {
        assemble_cells_with_kernel(kernel_impl);
    });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

std::shared_ptr<const quadrature::QuadratureRule> StandardAssembler::resolveQuadratureRule(
    const spaces::FunctionSpace& test_space,
    GlobalIndex cell_id,
    ElementType cell_type) const
{
    const auto& test_element = getElement(test_space, cell_id, cell_type);

    // For P1 Tet4 elements, use position-based rules that match the legacy
    // solver (4 QPs instead of 27 from tensor-product Duffy transform).
    // Do NOT override Tri3: its Duffy rule already gives 4 QPs, and the
    // NS-VMS stabilization terms (nonlinear τ) need the extra point for
    // accurate integration compared to the 3-point position-based rule.
    const int basis_order = test_element.polynomial_order();
    if (basis_order <= 1 && cell_type == ElementType::Tetra4) {
        const auto default_mod = quadrature::QuadratureFactory::default_legacy_modifier(cell_type);
        return quadrature::QuadratureFactory::create_legacy_compatible(
            cell_type, default_mod);
    }

    auto quad_rule = test_element.quadrature();
    if (!quad_rule) {
        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            basis_order, false);
        quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
    }
    return quad_rule;
}

void StandardAssembler::ensureColoring(const IMeshAccess& mesh,
                                       std::span<const dofs::DofMap* const> extra_dof_maps)
{
    // Check if coloring is already valid for this mesh
    if (coloring_valid_ && coloring_mesh_ == &mesh) {
        return;
    }

    // Build element connectivity graph from all DOF maps.
    // For coupled multi-field assembly, different blocks use different DOF maps
    // (e.g., velocity vs pressure). The coloring must ensure no two same-color
    // elements share a DOF in ANY map.
    ElementGraph graph;
    if (extra_dof_maps.empty()) {
        graph.build(mesh, *row_dof_map_);
    } else {
        std::vector<const dofs::DofMap*> all_maps;
        all_maps.reserve(1 + extra_dof_maps.size());
        all_maps.push_back(row_dof_map_);
        for (const auto* dm : extra_dof_maps) {
            if (dm != nullptr && dm != row_dof_map_) {
                // Deduplicate: only add if not already present
                bool found = false;
                for (const auto* existing : all_maps) {
                    if (existing == dm) { found = true; break; }
                }
                if (!found) all_maps.push_back(dm);
            }
        }
        graph.build(mesh, all_maps);
    }

    // Color the graph using greedy algorithm
    coloring_num_colors_ = colorGraph(graph, ColoringAlgorithm::Greedy, coloring_colors_);

    // Verify coloring in debug builds
    FE_THROW_IF(!verifyColoring(graph, coloring_colors_), FEException,
                "ensureColoring: graph coloring verification failed");

    // Build per-color cell lists
    coloring_cells_by_color_.clear();
    coloring_cells_by_color_.resize(coloring_num_colors_);
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        const int color = coloring_colors_[cell_id];
        coloring_cells_by_color_[color].push_back(cell_id);
    });

    // Mark cache as valid
    coloring_valid_ = true;
    coloring_mesh_ = &mesh;
}

void StandardAssembler::prepareGeometry(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const quadrature::QuadratureRule& quad_rule)
{
    auto PC_TP = assemblyTimeNow;
    double pc_t0 = PC_TP();

    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    // Get cell node coordinates — use flat table when available (Tier 2/3),
    // bypassing virtual dispatch through IMeshAccess::getCellCoordinates.
    if (flat_cell_coords_.valid && flat_cell_coords_.mesh == &mesh &&
        cell_id >= 0 &&
        static_cast<std::size_t>(cell_id) < flat_cell_coords_.coords.size() /
            (static_cast<std::size_t>(flat_cell_coords_.nodes_per_cell) * 3u)) {
        const auto npc = static_cast<std::size_t>(flat_cell_coords_.nodes_per_cell);
        const auto base = static_cast<std::size_t>(cell_id) * npc * 3u;
        cell_coords_.resize(npc);
        scratch_node_coords_.resize(npc);
        for (std::size_t i = 0; i < npc; ++i) {
            const auto x = flat_cell_coords_.coords[base + i * 3u + 0u];
            const auto y = flat_cell_coords_.coords[base + i * 3u + 1u];
            const auto z = flat_cell_coords_.coords[base + i * 3u + 2u];
            cell_coords_[i] = {x, y, z};
            scratch_node_coords_[i] = math::Vector<Real, 3>{x, y, z};
        }
    } else {
        mesh.getCellCoordinates(cell_id, cell_coords_);
        scratch_node_coords_.resize(cell_coords_.size());
        for (std::size_t i = 0; i < cell_coords_.size(); ++i) {
            scratch_node_coords_[i] = math::Vector<Real, 3>{
                cell_coords_[i][0], cell_coords_[i][1], cell_coords_[i][2]};
        }
    }
    const auto n_nodes = cell_coords_.size();

#ifdef SVMP_FE_ASSEMBLY_TIMING
    g_pc_setup += PC_TP() - pc_t0;
#endif

    // Create or reuse geometry mapping
#ifdef SVMP_FE_ASSEMBLY_TIMING
    pc_t0 = PC_TP();
#endif
    const int geom_order = defaultGeometryOrder(cell_type);
    const bool use_affine = (geom_order <= 1);

    if (cell_type != cached_mapping_type_ ||
        geom_order != cached_mapping_order_ ||
        use_affine != cached_mapping_affine_ ||
        !cached_mapping_) {
        geometry::MappingRequest map_request;
        map_request.element_type = cell_type;
        map_request.geometry_order = geom_order;
        map_request.use_affine = use_affine;
        cached_mapping_ = geometry::MappingFactory::create(map_request, scratch_node_coords_);
        cached_mapping_type_ = cell_type;
        cached_mapping_order_ = geom_order;
        cached_mapping_affine_ = use_affine;
        // Invalidate cached BasisCacheEntry pointers — will be re-looked-up below.
        cached_geom_bcache_ = nullptr;
        cached_test_bcache_ = nullptr;
        cached_trial_bcache_ = nullptr;
        cached_field_bcache_.clear();
    } else {
        cached_mapping_->resetNodes(scratch_node_coords_);
    }
    const auto& mapping = cached_mapping_;
#ifdef SVMP_FE_ASSEMBLY_TIMING
    g_pc_mapping += PC_TP() - pc_t0;
#endif

    // Prepare context arena geometry storage (zero-copy into arena)
#ifdef SVMP_FE_ASSEMBLY_TIMING
    pc_t0 = PC_TP();
#endif
    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    context.prepareGeometryStorage(n_qpts);
    auto ctx_quad_pts = context.quadPointsWritable();
    auto ctx_quad_wts = context.quadWeightsWritable();
    auto ctx_phys_pts = context.physicalPointsWritable();
    auto ctx_jacs = context.jacobiansWritable();
    auto ctx_inv_jacs = context.inverseJacobiansWritable();
    auto ctx_jac_dets = context.jacobianDetsWritable();
    auto ctx_int_wts = context.integrationWeightsWritable();
#ifdef SVMP_FE_ASSEMBLY_TIMING
    g_pc_resize += PC_TP() - pc_t0;
#endif

    // Compute quadrature data, physical points, and Jacobians
#ifdef SVMP_FE_ASSEMBLY_TIMING
    pc_t0 = PC_TP();
#endif
    const auto& quad_points = quad_rule.points();
    const auto& quad_weights = quad_rule.weights();

    if (!cached_geom_bcache_) {
        cached_geom_bcache_ = &basis::BasisCache::instance().get_or_compute(
            mapping->geometryBasis(), quad_rule, /*gradients=*/true, /*hessians=*/false);
    }
    const auto& geom_bcache = *cached_geom_bcache_;
    const auto& geom_nodes = mapping->nodes();
    const auto n_geom_dofs = geom_bcache.num_dofs;

    if (mapping->isAffine()) {
        // Affine element: Jacobian is constant across all QPs.
        math::Matrix<Real, 3, 3> J{};
        for (std::size_t a = 0; a < n_geom_dofs; ++a) {
            const auto& grad_a = geom_bcache.gradients[0][a];
            for (int j = 0; j < dim; ++j) {
                for (int i = 0; i < 3; ++i) {
                    J(i, j) += geom_nodes[a][i] * grad_a[j];
                }
            }
        }

        // Frame completion for embedded geometry (dim < 3)
        if (dim == 1) {
            const math::Vector<Real, 3> t{J(0, 0), J(1, 0), J(2, 0)};
            math::Vector<Real, 3> n1{}, n2{};
            geometry::detail::complete_curve_frame(t, n1, n2);
            J(0, 1) = n1[0]; J(1, 1) = n1[1]; J(2, 1) = n1[2];
            J(0, 2) = n2[0]; J(1, 2) = n2[1]; J(2, 2) = n2[2];
        } else if (dim == 2) {
            const math::Vector<Real, 3> tu{J(0, 0), J(1, 0), J(2, 0)};
            const math::Vector<Real, 3> tv{J(0, 1), J(1, 1), J(2, 1)};
            const auto nv = tu.cross(tv);
            const Real nv_norm = nv.norm();
            if (nv_norm < geometry::detail::kDegenerateTol) {
                J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
            } else {
                const auto nv_unit = nv / nv_norm;
                J(0, 2) = nv_unit[0]; J(1, 2) = nv_unit[1]; J(2, 2) = nv_unit[2];
            }
        }

        const auto J_inv = J.inverse();
        const Real det_J = J.determinant();
        const Real abs_det_J = std::abs(det_J);

        AssemblyContext::Matrix3x3 J_arr{}, J_inv_arr{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                J_arr[i][j] = J(i, j);
                J_inv_arr[i][j] = J_inv(i, j);
            }

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const auto& qpt = quad_points[q];
            ctx_quad_pts[q] = {qpt[0], qpt[1], qpt[2]};
            ctx_quad_wts[q] = quad_weights[q];

            const auto qidx = static_cast<std::size_t>(q);
            Real x0 = 0, x1 = 0, x2 = 0;
            for (std::size_t a = 0; a < n_geom_dofs; ++a) {
                const Real N_a = geom_bcache.scalarValue(a, qidx);
                x0 += geom_nodes[a][0] * N_a;
                x1 += geom_nodes[a][1] * N_a;
                x2 += geom_nodes[a][2] * N_a;
            }
            ctx_phys_pts[q] = {x0, x1, x2};

            ctx_jacs[q] = J_arr;
            ctx_inv_jacs[q] = J_inv_arr;
            ctx_jac_dets[q] = det_J;
            ctx_int_wts[q] = quad_weights[q] * abs_det_J;
        }
    } else {
        // Non-affine element: compute J per QP from cached geometry basis gradients.
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const auto& qpt = quad_points[q];
            ctx_quad_pts[q] = {qpt[0], qpt[1], qpt[2]};
            ctx_quad_wts[q] = quad_weights[q];

            const auto qidx = static_cast<std::size_t>(q);

            Real x0 = 0, x1 = 0, x2 = 0;
            for (std::size_t a = 0; a < n_geom_dofs; ++a) {
                const Real N_a = geom_bcache.scalarValue(a, qidx);
                x0 += geom_nodes[a][0] * N_a;
                x1 += geom_nodes[a][1] * N_a;
                x2 += geom_nodes[a][2] * N_a;
            }
            ctx_phys_pts[q] = {x0, x1, x2};

            math::Matrix<Real, 3, 3> J{};
            for (std::size_t a = 0; a < n_geom_dofs; ++a) {
                const auto& grad_a = geom_bcache.gradients[qidx][a];
                for (int j = 0; j < dim; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        J(i, j) += geom_nodes[a][i] * grad_a[j];
                    }
                }
            }

            if (dim == 1) {
                const math::Vector<Real, 3> t{J(0, 0), J(1, 0), J(2, 0)};
                math::Vector<Real, 3> n1{}, n2{};
                geometry::detail::complete_curve_frame(t, n1, n2);
                J(0, 1) = n1[0]; J(1, 1) = n1[1]; J(2, 1) = n1[2];
                J(0, 2) = n2[0]; J(1, 2) = n2[1]; J(2, 2) = n2[2];
            } else if (dim == 2) {
                const math::Vector<Real, 3> tu{J(0, 0), J(1, 0), J(2, 0)};
                const math::Vector<Real, 3> tv{J(0, 1), J(1, 1), J(2, 1)};
                const auto nv = tu.cross(tv);
                const Real nv_norm = nv.norm();
                if (nv_norm < geometry::detail::kDegenerateTol) {
                    J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
                } else {
                    const auto nv_unit = nv / nv_norm;
                    J(0, 2) = nv_unit[0]; J(1, 2) = nv_unit[1]; J(2, 2) = nv_unit[2];
                }
            }

            const auto J_inv = J.inverse();
            const Real det_J = J.determinant();

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    ctx_jacs[q][i][j] = J(i, j);
                    ctx_inv_jacs[q][i][j] = J_inv(i, j);
                }
            ctx_jac_dets[q] = det_J;
            ctx_int_wts[q] = quad_weights[q] * std::abs(det_J);
        }
    }

    // Precompute entity measures (h = element diameter, volume = sum of integration
    // weights).  These are cached in member variables and saved per batch slot in the
    // coupled assembly path, allowing the prepareBasis fast path to use them directly
    // without accessing mapping->nodes().  For affine elements this eliminates the
    // need to restore scratch_node_coords_ / resetNodes() between blocks.
    {
        Real cell_volume = 0.0;
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            cell_volume += ctx_int_wts[q];
        }
        Real h = 0.0;
        for (std::size_t a = 0; a < n_nodes; ++a) {
            for (std::size_t b = a + 1; b < n_nodes; ++b) {
                const Real dx = scratch_node_coords_[a][0] - scratch_node_coords_[b][0];
                const Real dy = scratch_node_coords_[a][1] - scratch_node_coords_[b][1];
                const Real dz = scratch_node_coords_[a][2] - scratch_node_coords_[b][2];
                const Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > h) h = dist;
            }
        }
        cached_geom_h_ = h;
        cached_geom_volume_ = cell_volume;
    }

    context.markGeometryDirty();

#ifdef SVMP_FE_ASSEMBLY_TIMING
    g_pc_jacobian += PC_TP() - pc_t0;
#endif

    // Cache quad_rule for use in populateFieldSolutionData BasisCache lookups
    cached_quad_rule_ = std::shared_ptr<const quadrature::QuadratureRule>(
        std::shared_ptr<const quadrature::QuadratureRule>{}, &quad_rule);
}

void StandardAssembler::prepareGeometry(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const quadrature::QuadratureRule& quad_rule,
    GeometryWorkspace& ws)
{
    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    mesh.getCellCoordinates(cell_id, ws.cell_coords);
    const auto n_nodes = ws.cell_coords.size();

    ws.node_coords.resize(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        ws.node_coords[i] = math::Vector<Real, 3>{
            ws.cell_coords[i][0], ws.cell_coords[i][1], ws.cell_coords[i][2]};
    }

    const int geom_order = defaultGeometryOrder(cell_type);
    const bool use_affine = (geom_order <= 1);

    if (cell_type != ws.mapping_type ||
        geom_order != ws.mapping_order ||
        use_affine != ws.mapping_affine ||
        !ws.mapping) {
        geometry::MappingRequest map_request;
        map_request.element_type = cell_type;
        map_request.geometry_order = geom_order;
        map_request.use_affine = use_affine;
        ws.mapping = geometry::MappingFactory::create(map_request, ws.node_coords);
        ws.mapping_type = cell_type;
        ws.mapping_order = geom_order;
        ws.mapping_affine = use_affine;
        ws.geom_bcache = nullptr;
    } else {
        ws.mapping->resetNodes(ws.node_coords);
    }
    const auto& mapping = ws.mapping;

    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    context.prepareGeometryStorage(n_qpts);
    auto ctx_quad_pts = context.quadPointsWritable();
    auto ctx_quad_wts = context.quadWeightsWritable();
    auto ctx_phys_pts = context.physicalPointsWritable();
    auto ctx_jacs = context.jacobiansWritable();
    auto ctx_inv_jacs = context.inverseJacobiansWritable();
    auto ctx_jac_dets = context.jacobianDetsWritable();
    auto ctx_int_wts = context.integrationWeightsWritable();

    const auto& quad_points = quad_rule.points();
    const auto& quad_weights = quad_rule.weights();

    if (!ws.geom_bcache) {
        ws.geom_bcache = &basis::BasisCache::instance().get_or_compute(
            mapping->geometryBasis(), quad_rule, /*gradients=*/true, /*hessians=*/false);
    }
    const auto& geom_bcache = *ws.geom_bcache;
    const auto& geom_nodes = mapping->nodes();
    const auto n_geom_dofs = geom_bcache.num_dofs;

    if (mapping->isAffine()) {
        math::Matrix<Real, 3, 3> J{};
        for (std::size_t a = 0; a < n_geom_dofs; ++a) {
            const auto& grad_a = geom_bcache.gradients[0][a];
            for (int j = 0; j < dim; ++j)
                for (int i = 0; i < 3; ++i)
                    J(i, j) += geom_nodes[a][i] * grad_a[j];
        }

        if (dim == 1) {
            const math::Vector<Real, 3> t{J(0, 0), J(1, 0), J(2, 0)};
            math::Vector<Real, 3> n1{}, n2{};
            geometry::detail::complete_curve_frame(t, n1, n2);
            J(0, 1) = n1[0]; J(1, 1) = n1[1]; J(2, 1) = n1[2];
            J(0, 2) = n2[0]; J(1, 2) = n2[1]; J(2, 2) = n2[2];
        } else if (dim == 2) {
            const math::Vector<Real, 3> tu{J(0, 0), J(1, 0), J(2, 0)};
            const math::Vector<Real, 3> tv{J(0, 1), J(1, 1), J(2, 1)};
            const auto nv = tu.cross(tv);
            const Real nv_norm = nv.norm();
            if (nv_norm < geometry::detail::kDegenerateTol) {
                J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
            } else {
                const auto nv_unit = nv / nv_norm;
                J(0, 2) = nv_unit[0]; J(1, 2) = nv_unit[1]; J(2, 2) = nv_unit[2];
            }
        }

        const auto J_inv = J.inverse();
        const Real det_J = J.determinant();
        const Real abs_det_J = std::abs(det_J);

        AssemblyContext::Matrix3x3 J_arr{}, J_inv_arr{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                J_arr[i][j] = J(i, j);
                J_inv_arr[i][j] = J_inv(i, j);
            }

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const auto& qpt = quad_points[q];
            ctx_quad_pts[q] = {qpt[0], qpt[1], qpt[2]};
            ctx_quad_wts[q] = quad_weights[q];
            const auto qidx = static_cast<std::size_t>(q);
            Real x0 = 0, x1 = 0, x2 = 0;
            for (std::size_t a = 0; a < n_geom_dofs; ++a) {
                const Real N_a = geom_bcache.scalarValue(a, qidx);
                x0 += geom_nodes[a][0] * N_a;
                x1 += geom_nodes[a][1] * N_a;
                x2 += geom_nodes[a][2] * N_a;
            }
            ctx_phys_pts[q] = {x0, x1, x2};
            ctx_jacs[q] = J_arr;
            ctx_inv_jacs[q] = J_inv_arr;
            ctx_jac_dets[q] = det_J;
            ctx_int_wts[q] = quad_weights[q] * abs_det_J;
        }
    } else {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const auto& qpt = quad_points[q];
            ctx_quad_pts[q] = {qpt[0], qpt[1], qpt[2]};
            ctx_quad_wts[q] = quad_weights[q];
            const auto qidx = static_cast<std::size_t>(q);
            Real x0 = 0, x1 = 0, x2 = 0;
            for (std::size_t a = 0; a < n_geom_dofs; ++a) {
                const Real N_a = geom_bcache.scalarValue(a, qidx);
                x0 += geom_nodes[a][0] * N_a;
                x1 += geom_nodes[a][1] * N_a;
                x2 += geom_nodes[a][2] * N_a;
            }
            ctx_phys_pts[q] = {x0, x1, x2};
            math::Matrix<Real, 3, 3> J{};
            for (std::size_t a = 0; a < n_geom_dofs; ++a) {
                const auto& grad_a = geom_bcache.gradients[qidx][a];
                for (int j = 0; j < dim; ++j)
                    for (int i = 0; i < 3; ++i)
                        J(i, j) += geom_nodes[a][i] * grad_a[j];
            }
            if (dim == 1) {
                const math::Vector<Real, 3> t{J(0, 0), J(1, 0), J(2, 0)};
                math::Vector<Real, 3> n1{}, n2{};
                geometry::detail::complete_curve_frame(t, n1, n2);
                J(0, 1) = n1[0]; J(1, 1) = n1[1]; J(2, 1) = n1[2];
                J(0, 2) = n2[0]; J(1, 2) = n2[1]; J(2, 2) = n2[2];
            } else if (dim == 2) {
                const math::Vector<Real, 3> tu{J(0, 0), J(1, 0), J(2, 0)};
                const math::Vector<Real, 3> tv{J(0, 1), J(1, 1), J(2, 1)};
                const auto nv = tu.cross(tv);
                const Real nv_norm = nv.norm();
                if (nv_norm < geometry::detail::kDegenerateTol) {
                    J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
                } else {
                    const auto nv_unit = nv / nv_norm;
                    J(0, 2) = nv_unit[0]; J(1, 2) = nv_unit[1]; J(2, 2) = nv_unit[2];
                }
            }
            const auto J_inv = J.inverse();
            const Real det_J = J.determinant();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    ctx_jacs[q][i][j] = J(i, j);
                    ctx_inv_jacs[q][i][j] = J_inv(i, j);
                }
            ctx_jac_dets[q] = det_J;
            ctx_int_wts[q] = quad_weights[q] * std::abs(det_J);
        }
    }

    // Entity measures
    {
        Real cell_volume = 0.0;
        for (LocalIndex q = 0; q < n_qpts; ++q)
            cell_volume += ctx_int_wts[q];
        Real h = 0.0;
        for (std::size_t a = 0; a < n_nodes; ++a)
            for (std::size_t b = a + 1; b < n_nodes; ++b) {
                const Real dx = ws.node_coords[a][0] - ws.node_coords[b][0];
                const Real dy = ws.node_coords[a][1] - ws.node_coords[b][1];
                const Real dz = ws.node_coords[a][2] - ws.node_coords[b][2];
                const Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > h) h = dist;
            }
        ws.geom_h = h;
        ws.geom_volume = cell_volume;
    }

    context.markGeometryDirty();
}

void StandardAssembler::prepareBasis(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    const quadrature::QuadratureRule& quad_rule)
{
    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    const auto& test_element = getElement(test_space, cell_id, cell_type);
    const auto& trial_element = getElement(trial_space, cell_id, cell_type);

    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    const auto n_test_dofs = static_cast<LocalIndex>(test_space.dofs_per_element());
    const auto n_trial_dofs = static_cast<LocalIndex>(trial_space.dofs_per_element());
    const auto n_test_scalar_dofs = static_cast<LocalIndex>(test_element.num_dofs());
    const auto n_trial_scalar_dofs = static_cast<LocalIndex>(trial_element.num_dofs());
    const bool test_is_product = (test_space.space_type() == spaces::SpaceType::Product);
    const bool trial_is_product = (trial_space.space_type() == spaces::SpaceType::Product);
    if (test_is_product) {
        FE_CHECK_ARG(test_space.field_type() == FieldType::Vector,
                     "StandardAssembler::prepareBasis: ProductSpace test space must be vector-valued");
        FE_CHECK_ARG(test_space.value_dimension() > 0,
                     "StandardAssembler::prepareBasis: invalid test space value dimension");
        FE_CHECK_ARG(n_test_dofs ==
                         static_cast<LocalIndex>(
                             n_test_scalar_dofs * static_cast<LocalIndex>(test_space.value_dimension())),
                     "StandardAssembler::prepareBasis: test ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_test_dofs == n_test_scalar_dofs,
                     "StandardAssembler::prepareBasis: non-Product test space DOF count mismatch");
    }
    if (trial_is_product) {
        FE_CHECK_ARG(trial_space.field_type() == FieldType::Vector,
                     "StandardAssembler::prepareBasis: ProductSpace trial space must be vector-valued");
        FE_CHECK_ARG(trial_space.value_dimension() > 0,
                     "StandardAssembler::prepareBasis: invalid trial space value dimension");
        FE_CHECK_ARG(n_trial_dofs ==
                         static_cast<LocalIndex>(
                             n_trial_scalar_dofs * static_cast<LocalIndex>(trial_space.value_dimension())),
                     "StandardAssembler::prepareBasis: trial ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_trial_dofs == n_trial_scalar_dofs,
                     "StandardAssembler::prepareBasis: non-Product trial space DOF count mismatch");
    }
    const bool need_basis_hessians = hasFlag(required_data, RequiredData::BasisHessians);
    const bool need_basis_curls = hasFlag(required_data, RequiredData::BasisCurls);
    const bool need_basis_divergences = hasFlag(required_data, RequiredData::BasisDivergences);

    // Invalidate cached BasisCacheEntry pointers when quad rule or hessian requirement changes.
    if (&quad_rule != cached_quad_rule_ptr_ || need_basis_hessians != cached_need_hessians_) {
        cached_geom_bcache_ = nullptr;
        cached_test_bcache_ = nullptr;
        cached_trial_bcache_ = nullptr;
        cached_field_bcache_.clear();
        cached_quad_rule_ptr_ = &quad_rule;
        cached_need_hessians_ = need_basis_hessians;
        cached_qpt_test_valid_ = false;
        cached_qpt_trial_valid_ = false;
        cached_qpt_major_valid_ = false;
    }

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
                        std::string("StandardAssembler::prepareBasis: ") + which +
                            " space uses a vector-valued basis but is not H(curl)/H(div)");

            if (continuity == Continuity::H_curl) {
                FE_THROW_IF(need_basis_divergences, FEException,
                            std::string("StandardAssembler::prepareBasis: BasisDivergences requested for ") + which +
                                " H(curl) space");
            } else if (continuity == Continuity::H_div) {
                FE_THROW_IF(need_basis_curls, FEException,
                            std::string("StandardAssembler::prepareBasis: BasisCurls requested for ") + which +
                                " H(div) space");
            }
        };

    validate_vector_basis_requirements(test_space, test_is_vector_basis, "test");
    validate_vector_basis_requirements(trial_space, trial_is_vector_basis, "trial");

    // ---- Fast path: skip BasisCache reads when topology matches previous cell ----
    const bool same_space = (&test_space == &trial_space);
    const bool spaces_match_cached =
        (same_space && cached_basis_same_space_) ||
        (!same_space && !cached_basis_same_space_ &&
         &test_space == cached_basis_test_space_ptr_ &&
         &trial_space == cached_basis_trial_space_ptr_ &&
         n_trial_dofs == cached_basis_n_trial_dofs_ &&
         !trial_is_vector_basis);
    if (basis_scratch_valid_ &&
        spaces_match_cached &&
        !test_is_vector_basis &&
        cell_type == cached_basis_cell_type_ &&
        n_test_dofs == cached_basis_n_test_dofs_ &&
        n_qpts == cached_basis_n_qpts_ &&
        (!need_basis_hessians || cached_basis_has_hessians_) &&
        !need_basis_curls &&
        !need_basis_divergences)
    {
        // Configure context metadata first (sets trial_is_test_, n_test_dofs_,
        // n_trial_dofs_, etc. — required for batch contexts that were not
        // configured by the slow path).  Clears arena arrays (size=0, no dealloc)
        // so direct-write transforms below can safely write into the arena.
        if (active_coupled_block_meta_ != nullptr) {
            // Coupled path: use pre-computed metadata to avoid virtual calls.
            context.configureForCoupledBlock(
                cell_id, mesh.getCellDomainId(cell_id),
                *active_coupled_block_meta_);
        } else {
            context.configure(cell_id, test_space, trial_space, required_data);
            context.setCellDomainId(mesh.getCellDomainId(cell_id));
        }

        // Recompute physical gradients + hessians from cached ref data + new J_inv.
        // Write directly to context arena to avoid scratch→context memcpy.
        const auto ctx_inv_jacs_r = context.inverseJacobians();

        auto transformGradientsDirect = [&](LocalIndex n_dofs,
                                            const std::vector<AssemblyContext::Vector3D>& ref_grads,
                                            AssemblyContext::Vector3D* __restrict__ out,
                                            const AssemblyContext::Matrix3x3& J_inv_const) {
            for (LocalIndex i = 0; i < n_dofs; ++i) {
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const std::size_t ref_idx = static_cast<std::size_t>(i * n_qpts + q);
                    const std::size_t phys_idx = static_cast<std::size_t>(q * n_dofs + i);
                    const auto& J_inv = cached_mapping_affine_ ? J_inv_const : ctx_inv_jacs_r[q];
                    const auto& gr = ref_grads[ref_idx];
                    AssemblyContext::Vector3D gp = {0.0, 0.0, 0.0};
                    if (dim == 3) {
                        gp[0] = J_inv[0][0]*gr[0] + J_inv[1][0]*gr[1] + J_inv[2][0]*gr[2];
                        gp[1] = J_inv[0][1]*gr[0] + J_inv[1][1]*gr[1] + J_inv[2][1]*gr[2];
                        gp[2] = J_inv[0][2]*gr[0] + J_inv[1][2]*gr[1] + J_inv[2][2]*gr[2];
                    } else if (dim == 2) {
                        gp[0] = J_inv[0][0]*gr[0] + J_inv[1][0]*gr[1];
                        gp[1] = J_inv[0][1]*gr[0] + J_inv[1][1]*gr[1];
                    } else {
                        gp[0] = J_inv[0][0]*gr[0];
                    }
                    out[phys_idx] = gp;
                }
            }
        };

        auto transformHessiansAffineDirect = [&](LocalIndex n_dofs,
                                                  const std::vector<AssemblyContext::Matrix3x3>& ref_hess,
                                                  AssemblyContext::Matrix3x3* __restrict__ out,
                                                  const AssemblyContext::Matrix3x3& J_inv) {
            for (LocalIndex i = 0; i < n_dofs; ++i) {
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const std::size_t ref_idx = static_cast<std::size_t>(i * n_qpts + q);
                    const std::size_t qpt_idx = static_cast<std::size_t>(q * n_dofs + i);
                    const auto& Hr = ref_hess[ref_idx];
                    AssemblyContext::Matrix3x3 Hp{};
                    for (int r = 0; r < dim; ++r) {
                        for (int c = 0; c < dim; ++c) {
                            Real s = 0.0;
                            for (int a = 0; a < dim; ++a)
                                for (int b = 0; b < dim; ++b)
                                    s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                         Hr[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                         J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                            Hp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = s;
                        }
                    }
                    out[qpt_idx] = Hp;
                }
            }
        };

        const auto& J_inv0 = ctx_inv_jacs_r[0];
        const auto test_count = static_cast<std::size_t>(n_test_dofs * n_qpts);

        // Test side: transform directly into context arena
        auto* test_phys_ptr = context.testPhysGradientsWritePtr(test_count);
        transformGradientsDirect(n_test_dofs, scratch_ref_gradients_, test_phys_ptr, J_inv0);

        if (need_basis_hessians) {
            auto* test_hess_ptr = context.testPhysHessiansWritePtr(test_count);
            if (cached_mapping_affine_) {
                transformHessiansAffineDirect(n_test_dofs, scratch_ref_hessians_, test_hess_ptr, J_inv0);
            } else {
                const auto& mapping = cached_mapping_;
                const auto ctx_quad_pts_r = context.quadraturePoints();
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const auto& J_inv = ctx_inv_jacs_r[q];
                    const auto& qp = ctx_quad_pts_r[q];
                    const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};
                    std::array<AssemblyContext::Matrix3x3, 3> d2xi_dx2{};
                    const auto map_hess = mapping->mapping_hessian(xi);
                    for (int a = 0; a < dim; ++a)
                        for (int ii = 0; ii < dim; ++ii)
                            for (int jj = 0; jj < dim; ++jj) {
                                Real s = 0.0;
                                for (int m = 0; m < dim; ++m)
                                    for (int p = 0; p < dim; ++p)
                                        for (int rr = 0; rr < dim; ++rr)
                                            s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(m)] *
                                                 map_hess[static_cast<std::size_t>(m)](
                                                     static_cast<std::size_t>(p), static_cast<std::size_t>(rr)) *
                                                 J_inv[static_cast<std::size_t>(p)][static_cast<std::size_t>(ii)] *
                                                 J_inv[static_cast<std::size_t>(rr)][static_cast<std::size_t>(jj)];
                                d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(ii)][static_cast<std::size_t>(jj)] = -s;
                            }
                    for (LocalIndex i = 0; i < n_test_dofs; ++i) {
                        const std::size_t ref_idx = static_cast<std::size_t>(i * n_qpts + q);
                        const std::size_t qpt_idx = static_cast<std::size_t>(q * n_test_dofs + i);
                        const auto& Hr = scratch_ref_hessians_[ref_idx];
                        const auto& gr = scratch_ref_gradients_[ref_idx];
                        AssemblyContext::Matrix3x3 Hp{};
                        for (int r = 0; r < dim; ++r)
                            for (int c = 0; c < dim; ++c) {
                                Real s = 0.0;
                                for (int a = 0; a < dim; ++a)
                                    for (int b = 0; b < dim; ++b)
                                        s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                             Hr[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                             J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                for (int a = 0; a < dim; ++a)
                                    s += gr[static_cast<std::size_t>(a)] *
                                         d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                Hp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = s;
                            }
                        test_hess_ptr[qpt_idx] = Hp;
                    }
                }
            }
        }

        // Trial side: transform directly into context arena if different from test
        if (!same_space) {
            const auto trial_count = static_cast<std::size_t>(n_trial_dofs * n_qpts);
            auto* trial_phys_ptr = context.trialPhysGradientsWritePtr(trial_count);
            transformGradientsDirect(n_trial_dofs, scratch_trial_ref_gradients_, trial_phys_ptr, J_inv0);

            if (need_basis_hessians && !scratch_trial_ref_hessians_.empty()) {
                auto* trial_hess_ptr = context.trialPhysHessiansWritePtr(trial_count);
                if (cached_mapping_affine_) {
                    transformHessiansAffineDirect(n_trial_dofs, scratch_trial_ref_hessians_, trial_hess_ptr, J_inv0);
                } else {
                    // Non-affine trial hessians: same d2xi_dx2 logic as test
                    const auto& mapping = cached_mapping_;
                    const auto ctx_quad_pts_r = context.quadraturePoints();
                    for (LocalIndex q = 0; q < n_qpts; ++q) {
                        const auto& J_inv = ctx_inv_jacs_r[q];
                        const auto& qp = ctx_quad_pts_r[q];
                        const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};
                        std::array<AssemblyContext::Matrix3x3, 3> d2xi_dx2{};
                        const auto map_hess = mapping->mapping_hessian(xi);
                        for (int a = 0; a < dim; ++a)
                            for (int ii = 0; ii < dim; ++ii)
                                for (int jj = 0; jj < dim; ++jj) {
                                    Real s = 0.0;
                                    for (int m = 0; m < dim; ++m)
                                        for (int p = 0; p < dim; ++p)
                                            for (int rr = 0; rr < dim; ++rr)
                                                s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(m)] *
                                                     map_hess[static_cast<std::size_t>(m)](
                                                         static_cast<std::size_t>(p), static_cast<std::size_t>(rr)) *
                                                     J_inv[static_cast<std::size_t>(p)][static_cast<std::size_t>(ii)] *
                                                     J_inv[static_cast<std::size_t>(rr)][static_cast<std::size_t>(jj)];
                                    d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(ii)][static_cast<std::size_t>(jj)] = -s;
                                }
                        for (LocalIndex i = 0; i < n_trial_dofs; ++i) {
                            const std::size_t ref_idx = static_cast<std::size_t>(i * n_qpts + q);
                            const std::size_t qpt_idx = static_cast<std::size_t>(q * n_trial_dofs + i);
                            const auto& Hr = scratch_trial_ref_hessians_[ref_idx];
                            const auto& gr = scratch_trial_ref_gradients_[ref_idx];
                            AssemblyContext::Matrix3x3 Hp{};
                            for (int r = 0; r < dim; ++r)
                                for (int c = 0; c < dim; ++c) {
                                    Real s = 0.0;
                                    for (int a = 0; a < dim; ++a)
                                        for (int b = 0; b < dim; ++b)
                                            s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                                 Hr[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                                 J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                    for (int a = 0; a < dim; ++a)
                                        s += gr[static_cast<std::size_t>(a)] *
                                             d2xi_dx2[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                    Hp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = s;
                                }
                            trial_hess_ptr[qpt_idx] = Hp;
                        }
                    }
                }
            }
        }

        // Re-set basis values from qpt-major cache (memcpy, no transpose).
        // Skip ref gradients/hessians — JIT kernels only read physical data.
        // Physical gradients/hessians were written directly above.
        if (cached_qpt_major_valid_) {
            context.setTestBasisValuesOnlyQptMajor(n_test_dofs, cached_qpt_test_values_);
            if (!same_space && !cached_qpt_major_same_space_) {
                context.setTrialBasisValuesOnlyQptMajor(n_trial_dofs, cached_qpt_trial_values_);
            }
        } else {
            // Fallback: transpose as before (first call before slow path populates cache)
            context.setTestBasisData(n_test_dofs, scratch_basis_values_, scratch_ref_gradients_);
            if (need_basis_hessians) {
                context.setTestBasisHessians(n_test_dofs, scratch_ref_hessians_);
            }
            if (!same_space) {
                context.setTrialBasisData(n_trial_dofs, scratch_trial_basis_values_, scratch_trial_ref_gradients_);
                if (need_basis_hessians) {
                    context.setTrialBasisHessians(n_trial_dofs, scratch_trial_ref_hessians_);
                }
            }
        }

        if (hasFlag(required_data, RequiredData::EntityMeasures)) {
            context.setEntityMeasures(cached_geom_h_, cached_geom_volume_, 0.0);
        }

        return;
    }
    // ---- End fast path ----

    // Resize basis scratch storage
    auto PC_TP = assemblyTimeNow;
    double pc_t0 = PC_TP();
    const auto test_basis_size = static_cast<std::size_t>(n_test_dofs * n_qpts);

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

    // Storage for trial if different from test — use member scratch for fast-path caching
    std::vector<AssemblyContext::Vector3D> trial_basis_vector_values;
    std::vector<AssemblyContext::Vector3D> trial_basis_curls;
    std::vector<Real> trial_basis_divergences;
    auto& trial_basis_values = scratch_trial_basis_values_;
    auto& trial_ref_gradients = scratch_trial_ref_gradients_;
    auto& trial_phys_gradients = scratch_trial_phys_gradients_;
    auto& trial_ref_hessians = scratch_trial_ref_hessians_;
    auto& trial_phys_hessians = scratch_trial_phys_hessians_;

    if (&test_space != &trial_space) {
        const auto trial_basis_size = static_cast<std::size_t>(n_trial_dofs * n_qpts);
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

#ifdef SVMP_FE_ASSEMBLY_TIMING
    g_pc_resize += PC_TP() - pc_t0;
#endif

    // Evaluate basis functions at quadrature points
#ifdef SVMP_FE_ASSEMBLY_TIMING
    pc_t0 = PC_TP();
#endif

    if (!test_is_vector_basis && !cached_test_bcache_) {
        cached_test_bcache_ = &basis::BasisCache::instance().get_or_compute(
            test_basis, quad_rule, true, need_basis_hessians);
    }
    if (&test_space != &trial_space && !trial_is_vector_basis && !cached_trial_bcache_) {
        cached_trial_bcache_ = &basis::BasisCache::instance().get_or_compute(
            trial_basis, quad_rule, true, need_basis_hessians);
    }
    const basis::BasisCacheEntry* test_bcache = cached_test_bcache_;
    const basis::BasisCacheEntry* trial_bcache = cached_trial_bcache_;

    const auto& mapping = cached_mapping_;

    auto& vec_values_at_pt = scratch_vec_values_at_pt_;
    auto& vec_curls_at_pt = scratch_vec_curls_at_pt_;
    auto& vec_divs_at_pt = scratch_vec_divs_at_pt_;

    // Read geometry from context arena (populated by prepareGeometry)
    const auto ctx_quad_pts_r = context.quadraturePoints();
    const auto ctx_jacs_r = context.jacobians();
    const auto ctx_inv_jacs_r = context.inverseJacobians();
    const auto ctx_jac_dets_r = context.jacobianDets();

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& qp = ctx_quad_pts_r[q];
        const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};

        const auto& J = ctx_jacs_r[q];
        const auto& J_inv = ctx_inv_jacs_r[q];
        const Real det_J = ctx_jac_dets_r[q];

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
            // Scalar test basis: read from BasisCache instead of per-QP evaluation
            const auto qsz = static_cast<std::size_t>(q);

            for (LocalIndex i = 0; i < n_test_dofs; ++i) {
                const LocalIndex si = test_is_product ? static_cast<LocalIndex>(i % n_test_scalar_dofs) : i;
                const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
                const std::size_t idx_phys = static_cast<std::size_t>(q * n_test_dofs + i);
                const auto sisz = static_cast<std::size_t>(si);

                scratch_basis_values_[idx] = test_bcache->scalarValue(sisz, qsz);

                const auto& g = test_bcache->gradients[qsz][sisz];
                scratch_ref_gradients_[idx] = {g[0], g[1], g[2]};

                const auto& grad_ref = scratch_ref_gradients_[idx];
                AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};
                if (dim == 3) {
                    grad_phys[0] = J_inv[0][0] * grad_ref[0] + J_inv[1][0] * grad_ref[1] + J_inv[2][0] * grad_ref[2];
                    grad_phys[1] = J_inv[0][1] * grad_ref[0] + J_inv[1][1] * grad_ref[1] + J_inv[2][1] * grad_ref[2];
                    grad_phys[2] = J_inv[0][2] * grad_ref[0] + J_inv[1][2] * grad_ref[1] + J_inv[2][2] * grad_ref[2];
                } else if (dim == 2) {
                    grad_phys[0] = J_inv[0][0] * grad_ref[0] + J_inv[1][0] * grad_ref[1];
                    grad_phys[1] = J_inv[0][1] * grad_ref[0] + J_inv[1][1] * grad_ref[1];
                } else if (dim == 1) {
                    grad_phys[0] = J_inv[0][0] * grad_ref[0];
                }
                scratch_phys_gradients_[idx_phys] = grad_phys;

                if (need_basis_hessians) {
                    const auto& hess = test_bcache->hessians[qsz][sisz];
                    AssemblyContext::Matrix3x3 H_ref{};
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                hess(static_cast<std::size_t>(r), static_cast<std::size_t>(c));
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
                // Scalar trial basis: read from BasisCache
                const auto qsz = static_cast<std::size_t>(q);

                for (LocalIndex j = 0; j < n_trial_dofs; ++j) {
                    const LocalIndex sj = trial_is_product ? static_cast<LocalIndex>(j % n_trial_scalar_dofs) : j;
                    const std::size_t idx = static_cast<std::size_t>(j * n_qpts + q);
                    const std::size_t idx_phys = static_cast<std::size_t>(q * n_trial_dofs + j);
                    const auto sjsz = static_cast<std::size_t>(sj);

                    trial_basis_values[idx] = trial_bcache->scalarValue(sjsz, qsz);

                    const auto& g = trial_bcache->gradients[qsz][sjsz];
                    trial_ref_gradients[idx] = {g[0], g[1], g[2]};

                    const auto& grad_ref = trial_ref_gradients[idx];
                    AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

                    for (int d1 = 0; d1 < dim; ++d1) {
                        for (int d2 = 0; d2 < dim; ++d2) {
                            grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                        }
                    }
                    trial_phys_gradients[idx_phys] = grad_phys;

                    if (need_basis_hessians) {
                        const auto& hess = trial_bcache->hessians[qsz][sjsz];
                        AssemblyContext::Matrix3x3 H_ref{};
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                    hess(static_cast<std::size_t>(r), static_cast<std::size_t>(c));
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

#ifdef SVMP_FE_ASSEMBLY_TIMING
    g_pc_basis += PC_TP() - pc_t0;
#endif

    // Configure context with basic info (geometry already in arena from prepareGeometry)
#ifdef SVMP_FE_ASSEMBLY_TIMING
    pc_t0 = PC_TP();
#endif
    context.configure(cell_id, test_space, trial_space, required_data);
    context.setCellDomainId(mesh.getCellDomainId(cell_id));

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
        // Use entity measures precomputed in prepareGeometry.
        context.setEntityMeasures(cached_geom_h_, cached_geom_volume_, /*facet_area=*/0.0);
    }

    // Update basis topology cache for fast-path reuse on next cell
    if (!test_is_vector_basis && (!trial_is_vector_basis || &test_space == &trial_space)) {
        basis_scratch_valid_ = true;
        cached_basis_cell_type_ = cell_type;
        cached_basis_n_test_dofs_ = n_test_dofs;
        cached_basis_n_trial_dofs_ = n_trial_dofs;
        cached_basis_n_qpts_ = n_qpts;
        cached_basis_same_space_ = (&test_space == &trial_space);
        cached_basis_has_hessians_ = need_basis_hessians;
        cached_basis_test_space_ptr_ = &test_space;
        cached_basis_trial_space_ptr_ = &trial_space;

        // Populate qpt-major cache from dof-major scratch (one transpose per block).
        // All subsequent fast-path cells use this cache via memcpy setters.
        {
            const auto nd_test = static_cast<std::size_t>(n_test_dofs);
            const auto nq = static_cast<std::size_t>(n_qpts);
            cached_qpt_test_values_.resize(nd_test * nq);
            cached_qpt_test_ref_grads_.resize(nd_test * nq);
            for (std::size_t i = 0; i < nd_test; ++i)
                for (std::size_t q = 0; q < nq; ++q) {
                    cached_qpt_test_values_[q * nd_test + i] = scratch_basis_values_[i * nq + q];
                    cached_qpt_test_ref_grads_[q * nd_test + i] = scratch_ref_gradients_[i * nq + q];
                }
            if (need_basis_hessians && !scratch_ref_hessians_.empty()) {
                cached_qpt_test_ref_hess_.resize(nd_test * nq);
                for (std::size_t i = 0; i < nd_test; ++i)
                    for (std::size_t q = 0; q < nq; ++q)
                        cached_qpt_test_ref_hess_[q * nd_test + i] = scratch_ref_hessians_[i * nq + q];
            } else {
                cached_qpt_test_ref_hess_.clear();
            }

            if (&test_space != &trial_space) {
                const auto nd_trial = static_cast<std::size_t>(n_trial_dofs);
                cached_qpt_trial_values_.resize(nd_trial * nq);
                cached_qpt_trial_ref_grads_.resize(nd_trial * nq);
                for (std::size_t i = 0; i < nd_trial; ++i)
                    for (std::size_t q = 0; q < nq; ++q) {
                        cached_qpt_trial_values_[q * nd_trial + i] = scratch_trial_basis_values_[i * nq + q];
                        cached_qpt_trial_ref_grads_[q * nd_trial + i] = scratch_trial_ref_gradients_[i * nq + q];
                    }
                if (need_basis_hessians && !scratch_trial_ref_hessians_.empty()) {
                    cached_qpt_trial_ref_hess_.resize(nd_trial * nq);
                    for (std::size_t i = 0; i < nd_trial; ++i)
                        for (std::size_t q = 0; q < nq; ++q)
                            cached_qpt_trial_ref_hess_[q * nd_trial + i] = scratch_trial_ref_hessians_[i * nq + q];
                } else {
                    cached_qpt_trial_ref_hess_.clear();
                }
            }

            cached_qpt_major_valid_ = true;
            cached_qpt_major_same_space_ = (&test_space == &trial_space);
            cached_qpt_major_has_hessians_ = need_basis_hessians;
        }

        cached_qpt_test_valid_ = false;
        cached_qpt_trial_valid_ = false;
    } else {
        basis_scratch_valid_ = false;
        cached_qpt_test_valid_ = false;
        cached_qpt_trial_valid_ = false;
        cached_qpt_major_valid_ = false;
    }

#ifdef SVMP_FE_ASSEMBLY_TIMING
    g_pc_ctx_config += PC_TP() - pc_t0;
    g_pc_call_count++;
#endif
}

void StandardAssembler::prepareContext(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data)
{
    const ElementType cell_type = mesh.getCellType(cell_id);
    auto quad_rule = resolveQuadratureRule(test_space, cell_id, cell_type);
    prepareGeometry(context, mesh, cell_id, *quad_rule);
    prepareBasis(context, mesh, cell_id, test_space, trial_space, required_data, *quad_rule);
    // Override the non-owning alias from prepareBasis with the actual shared_ptr
    // so that cached_quad_rule_ survives after this function returns.
    cached_quad_rule_ = std::move(quad_rule);
}

// ============================================================================
// Fused Multi-Term Cell Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleCellsFused(
    const IMeshAccess& mesh,
    std::span<const FusedCellTerm> terms)
{
    AssemblyResult result;
    if (terms.empty()) {
        return result;
    }

    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }
    ensureCellDofTables(mesh);

    // Begin assembly phase on all views
    for (const auto& t : terms) {
        if (t.matrix_view && t.assemble_matrix &&
            t.matrix_view->getPhase() == AssemblyPhase::NotStarted) {
            t.matrix_view->beginAssemblyPhase();
        }
        if (t.vector_view && t.assemble_vector &&
            t.vector_view != t.matrix_view &&
            t.vector_view->getPhase() == AssemblyPhase::NotStarted) {
            t.vector_view->beginAssemblyPhase();
        }
    }

    // Verify all terms have non-null kernels and spaces
    for (std::size_t ti = 0; ti < terms.size(); ++ti) {
        const auto& t = terms[ti];
        FE_CHECK_NOT_NULL(t.kernel, "assembleCellsFused: kernel");
        FE_CHECK_NOT_NULL(t.test_space, "assembleCellsFused: test_space");
        FE_CHECK_NOT_NULL(t.trial_space, "assembleCellsFused: trial_space");
        FE_CHECK_NOT_NULL(t.row_dof_map, "assembleCellsFused: row_dof_map");
        FE_CHECK_NOT_NULL(t.col_dof_map, "assembleCellsFused: col_dof_map");
        if (!t.kernel->hasCell()) continue;
        (void)getCellDofTable(mesh, t.row_dof_map, t.row_dof_offset);
        (void)getCellDofTable(mesh, t.col_dof_map, t.col_dof_offset);
    }
    for (const auto& access : field_solution_access_) {
        if (access.dof_map != nullptr) {
            (void)getCellDofTable(mesh, access.dof_map, access.dof_offset);
        }
    }
    ensureFieldAccessPlans(mesh);
    ensureResolvedVectorTables(mesh);
    ensureFlatCellCoords(mesh);

    // Pre-resolve matrix CSR slots for all terms that assemble matrices.
    // Tables persist across Newton iterations; subsequent insertions become
    // flat scatter (no hash probes).
    for (const auto& t : terms) {
        if (t.assemble_matrix && t.matrix_view &&
            t.matrix_view->insertionCapabilities().resolved_matrix_entries) {
            ensureResolvedMatrixTable(mesh, t.row_dof_map, t.row_dof_offset,
                                      t.col_dof_map, t.col_dof_offset,
                                      t.matrix_view);
        }
    }

    // Pre-compute per-cell constrained flags (built once, persists across
    // Newton iterations). Avoids per-cell hasConstrainedDofs lookups in
    // insertLocalForCell and the colored parallel insertion path.
    ensureCellConstrainedFlags(mesh);

    // Check if any term actually has work to do
    bool any_has_cell = false;
    for (const auto& t : terms) {
        if (t.kernel->hasCell()) {
            any_has_cell = true;
            break;
        }
    }
    if (!any_has_cell) {
        auto end_time = std::chrono::steady_clock::now();
        result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return result;
    }

    // Resolve quadrature rule from first active term's test element.
    // Verify all terms use the same quad rule; fall back to sequential if mismatched.
    std::shared_ptr<const quadrature::QuadratureRule> fused_quad_rule;
    {
        const auto& first_t = terms[0];
        // We need a sample cell to resolve the quad rule. Use cell 0 if available.
        const auto n_cells = mesh.numOwnedCells();
        if (n_cells == 0) {
            auto end_time = std::chrono::steady_clock::now();
            result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
            return result;
        }
        // Get the first owned cell
        GlobalIndex first_cell = -1;
        mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
            if (first_cell < 0) first_cell = cell_id;
        });
        const auto cell_type = mesh.getCellType(first_cell);
        fused_quad_rule = resolveQuadratureRule(*first_t.test_space, first_cell, cell_type);

        // Verify all terms produce the same quad rule
        for (std::size_t ti = 1; ti < terms.size(); ++ti) {
            auto other_qr = resolveQuadratureRule(*terms[ti].test_space, first_cell, cell_type);
            if (other_qr->num_points() != fused_quad_rule->num_points()) {
                // Quad rule mismatch — fall back to default sequential path
                return Assembler::assembleCellsFused(mesh, terms);
            }
        }
    }

    // Pre-compute per-term data requirements
    struct TermData {
        RequiredData required_data{RequiredData::None};
        std::vector<FieldRequirement> field_requirements;
        bool need_solution{false};
        bool need_field_solutions{false};
        bool need_material_state{false};
        MaterialStateSpec material_state_spec{};
    };
    std::vector<TermData> term_data(terms.size());
    for (std::size_t ti = 0; ti < terms.size(); ++ti) {
        auto& td = term_data[ti];
        td.required_data = terms[ti].kernel->getRequiredData();
        td.field_requirements = terms[ti].kernel->fieldRequirements();
        td.need_field_solutions = !td.field_requirements.empty();
        td.need_solution =
            hasFlag(td.required_data, RequiredData::SolutionCoefficients) ||
            hasFlag(td.required_data, RequiredData::SolutionValues) ||
            hasFlag(td.required_data, RequiredData::SolutionGradients) ||
            hasFlag(td.required_data, RequiredData::SolutionHessians) ||
            hasFlag(td.required_data, RequiredData::SolutionLaplacians);
        td.need_material_state = hasFlag(td.required_data, RequiredData::MaterialState);
        td.material_state_spec = terms[ti].kernel->materialStateSpec();
    }

    // Compute union of field requirements across all terms
    std::vector<FieldRequirement> union_field_reqs;
    for (const auto& td : term_data) {
        for (const auto& fr : td.field_requirements) {
            bool found = false;
            for (auto& ufr : union_field_reqs) {
                if (ufr.field == fr.field) {
                    ufr.required = ufr.required | fr.required;
                    found = true;
                    break;
                }
            }
            if (!found) {
                union_field_reqs.push_back(fr);
            }
        }
    }
    const bool any_need_field_solutions = !union_field_reqs.empty();
    if (any_need_field_solutions) {
        ensureFieldRecipes(mesh, union_field_reqs);
    }

    // Determine max DOFs for context reservation
    LocalIndex max_dofs = 0;
    for (const auto& t : terms) {
        max_dofs = std::max(max_dofs, t.row_dof_map->getMaxDofsPerCell());
        max_dofs = std::max(max_dofs, t.col_dof_map->getMaxDofsPerCell());
    }
    constexpr LocalIndex max_qpts = 27;
    context_.reserve(max_dofs, max_qpts, mesh.dimension());

    // Determine owned-rows-only policy
    const bool owned_rows_only = (options_.ghost_policy == GhostPolicy::OwnedRowsOnly);

    // Build cell ID list
    std::vector<GlobalIndex> cell_ids;
    if (owned_rows_only) {
        mesh.forEachCell([&](GlobalIndex cell_id) { cell_ids.push_back(cell_id); });
    } else {
        mesh.forEachOwnedCell([&](GlobalIndex cell_id) { cell_ids.push_back(cell_id); });
    }

    // Per-term scratch: DOF vectors and OwnedRowOnlyViews
    struct TermScratch {
        std::span<const GlobalIndex> row_dofs{};
        std::span<const GlobalIndex> col_dofs{};
        std::optional<OwnedRowOnlyView> owned_matrix_view;
        std::optional<OwnedRowOnlyView> owned_vector_view;
        GlobalSystemView* insert_matrix{nullptr};
        GlobalSystemView* insert_vector{nullptr};
    };
    std::vector<TermScratch> term_scratch(terms.size());
    for (std::size_t ti = 0; ti < terms.size(); ++ti) {
        auto& ts = term_scratch[ti];
        const auto& t = terms[ti];
        ts.insert_matrix = t.matrix_view;
        ts.insert_vector = t.vector_view;
        if (owned_rows_only) {
            if (t.matrix_view && t.assemble_matrix) {
                ts.owned_matrix_view.emplace(*t.matrix_view, *t.row_dof_map, t.row_dof_offset);
                ts.insert_matrix = &*ts.owned_matrix_view;
            }
            if (t.vector_view && t.assemble_vector) {
                if (t.vector_view == t.matrix_view && ts.owned_matrix_view) {
                    ts.insert_vector = ts.insert_matrix;
                } else {
                    ts.owned_vector_view.emplace(*t.vector_view, *t.row_dof_map, t.row_dof_offset);
                    ts.insert_vector = &*ts.owned_vector_view;
                }
            }
        }
    }

    const forms::MonolithicCellKernel* monolithic_kernel = nullptr;
    if (terms.size() == 1 &&
        terms[0].kernel->semanticKernelKind() == SemanticKernelKind::MonolithicCell) {
        monolithic_kernel = dynamic_cast<const forms::MonolithicCellKernel*>(terms[0].kernel);
    }

    if (monolithic_kernel && monolithic_kernel->isResolved()) {
        const auto& parent_term = terms[0];
        const auto& parent_td = term_data[0];
        const auto matrix_caps = parent_term.matrix_view
            ? parent_term.matrix_view->insertionCapabilities()
            : InsertionCapabilities{};
        const auto vector_caps = parent_term.vector_view
            ? parent_term.vector_view->insertionCapabilities()
            : InsertionCapabilities{};

        struct BlockInsertViews {
            std::optional<OwnedRowOnlyView> owned_matrix_view;
            std::optional<OwnedRowOnlyView> owned_vector_view;
            GlobalSystemView* insert_matrix{nullptr};
            GlobalSystemView* insert_vector{nullptr};
        };

        struct BlockWorkspace {
            AssemblyContext ctx;
            std::vector<Real> solution_coeffs;
            std::vector<std::vector<Real>> previous_solution_coeffs;
            KernelOutput output;
            std::span<const GlobalIndex> row_dofs{};
            std::span<const GlobalIndex> col_dofs{};
        };

        const std::size_t n_blocks = monolithic_kernel->numBlocks();
        std::vector<BlockInsertViews> block_inserts(n_blocks);
        std::vector<BlockWorkspace> block_workspaces(n_blocks);
        std::vector<assembly::jit::CoupledBlockView> block_views(n_blocks);
        std::vector<assembly::jit::CoupledCellKernelArgsV1> element_args(1);

        cached_coupled_block_meta_.resize(n_blocks);
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            const auto& bs = monolithic_kernel->blockSpec(bi);
            cached_coupled_block_meta_[bi] =
                AssemblyContext::makeCoupledBlockMetadata(
                    *bs.test_space, *bs.trial_space,
                    bs.fallback_kernel ? bs.fallback_kernel->getRequiredData()
                                       : RequiredData::Standard);
            block_workspaces[bi].ctx.reserve(max_dofs, max_qpts, mesh.dimension());

            auto& insert = block_inserts[bi];
            insert.insert_matrix = parent_term.matrix_view;
            insert.insert_vector = parent_term.vector_view;
            if (owned_rows_only) {
                if (parent_term.matrix_view && parent_term.assemble_matrix) {
                    insert.owned_matrix_view.emplace(
                        *parent_term.matrix_view, *bs.row_dof_map, bs.row_dof_offset);
                    insert.insert_matrix = &*insert.owned_matrix_view;
                }
                if (parent_term.vector_view && parent_term.assemble_vector) {
                    if (parent_term.vector_view == parent_term.matrix_view &&
                        insert.owned_matrix_view) {
                        insert.insert_vector = insert.insert_matrix;
                    } else {
                        insert.owned_vector_view.emplace(
                            *parent_term.vector_view, *bs.row_dof_map, bs.row_dof_offset);
                        insert.insert_vector = &*insert.owned_vector_view;
                    }
                }
            }

            if (parent_term.assemble_matrix && parent_term.matrix_view &&
                matrix_caps.resolved_matrix_entries && bs.want_matrix) {
                ensureResolvedMatrixTable(mesh, bs.row_dof_map, bs.row_dof_offset,
                                          bs.col_dof_map, bs.col_dof_offset,
                                          parent_term.matrix_view);
            }
            if (parent_term.assemble_vector && parent_term.vector_view &&
                vector_caps.resolved_vector_entries && bs.want_vector) {
                ensureResolvedVectorTable(mesh, bs.row_dof_map, bs.row_dof_offset,
                                          parent_term.vector_view);
            }
        }

        bool use_coupled_scalar_cache = false;
        LocalIndex coupled_n_scalar = 0;
        if (n_blocks >= 2 && !cell_ids.empty()) {
            use_coupled_scalar_cache = true;
            const auto first_cell = cell_ids.front();
            const auto first_ct = mesh.getCellType(first_cell);
            const auto& first_bs = monolithic_kernel->blockSpec(0);
            const auto& first_el = getElement(*first_bs.test_space, first_cell, first_ct);
            coupled_n_scalar = static_cast<LocalIndex>(first_el.num_dofs());
            const auto* first_basis_ptr = &first_el.basis();

            for (std::size_t bi = 0; bi < n_blocks && use_coupled_scalar_cache; ++bi) {
                const auto& bs = monolithic_kernel->blockSpec(bi);
                const auto& te = getElement(*bs.test_space, first_cell, first_ct);
                const auto& tre = getElement(*bs.trial_space, first_cell, first_ct);
                if (te.basis().is_vector_valued() || tre.basis().is_vector_valued()) {
                    use_coupled_scalar_cache = false;
                } else if (&te.basis() != first_basis_ptr || &tre.basis() != first_basis_ptr) {
                    if (static_cast<LocalIndex>(te.num_dofs()) != coupled_n_scalar ||
                        static_cast<LocalIndex>(tre.num_dofs()) != coupled_n_scalar) {
                        use_coupled_scalar_cache = false;
                    }
                }
            }

            if (use_coupled_scalar_cache) {
                if (coupled_slot_phys_cache_.empty()) {
                    coupled_slot_phys_cache_.resize(1);
                }

                const auto n_qpts = static_cast<LocalIndex>(fused_quad_rule->num_points());
                const auto ns = static_cast<std::size_t>(coupled_n_scalar);
                const auto nq = static_cast<std::size_t>(n_qpts);
                const bool need_hess = std::any_of(
                    cached_coupled_block_meta_.begin(), cached_coupled_block_meta_.end(),
                    [](const auto& m) {
                        return hasFlag(m.required_data, RequiredData::BasisHessians);
                    });

                if (!coupled_scalar_ref_valid_ ||
                    coupled_scalar_n_dofs_ != coupled_n_scalar ||
                    coupled_scalar_n_qpts_ != n_qpts ||
                    (need_hess && !coupled_scalar_has_hessians_)) {
                    const auto& bcache = basis::BasisCache::instance().get_or_compute(
                        first_el.basis(), *fused_quad_rule, true, need_hess);

                    coupled_scalar_ref_grads_.resize(ns * nq);
                    coupled_scalar_ref_hess_.resize(need_hess ? ns * nq : 0);
                    coupled_scalar_basis_values_.resize(nq * ns);

                    for (std::size_t q = 0; q < nq; ++q) {
                        for (std::size_t si = 0; si < ns; ++si) {
                            const auto ref_idx = si * nq + q;
                            coupled_scalar_ref_grads_[ref_idx] = {
                                bcache.gradients[q][si][0],
                                bcache.gradients[q][si][1],
                                bcache.gradients[q][si][2]
                            };
                            if (need_hess) {
                                AssemblyContext::Matrix3x3 Hr{};
                                for (int r = 0; r < 3; ++r) {
                                    for (int c = 0; c < 3; ++c) {
                                        Hr[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                            bcache.hessians[q][si](static_cast<std::size_t>(r),
                                                                   static_cast<std::size_t>(c));
                                    }
                                }
                                coupled_scalar_ref_hess_[ref_idx] = Hr;
                            }
                            coupled_scalar_basis_values_[q * ns + si] =
                                bcache.scalarValue(si, q);
                        }
                    }

                    coupled_space_qpt_caches_.clear();
                    std::vector<LocalIndex> unique_dof_counts;
                    for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                        const auto& bs = monolithic_kernel->blockSpec(bi);
                        const auto td = static_cast<LocalIndex>(bs.test_space->dofs_per_element());
                        bool found = false;
                        for (auto d : unique_dof_counts) {
                            if (d == td) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            unique_dof_counts.push_back(td);
                        }
                        if (bs.trial_space && bs.trial_space != bs.test_space) {
                            const auto trd = static_cast<LocalIndex>(bs.trial_space->dofs_per_element());
                            found = false;
                            for (auto d : unique_dof_counts) {
                                if (d == trd) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                unique_dof_counts.push_back(trd);
                            }
                        }
                    }

                    for (const auto n_dofs : unique_dof_counts) {
                        const auto nd = static_cast<std::size_t>(n_dofs);
                        CoupledSpaceQptCache cache;
                        cache.n_dofs = n_dofs;
                        cache.qpt_values.resize(nq * nd);
                        for (std::size_t q = 0; q < nq; ++q) {
                            for (std::size_t i = 0; i < nd; ++i) {
                                cache.qpt_values[q * nd + i] =
                                    coupled_scalar_basis_values_[q * ns + (i % ns)];
                            }
                        }
                        coupled_space_qpt_caches_.push_back(std::move(cache));
                    }

                    coupled_scalar_n_dofs_ = coupled_n_scalar;
                    coupled_scalar_n_qpts_ = n_qpts;
                    coupled_scalar_has_hessians_ = need_hess;
                    coupled_scalar_ref_valid_ = true;
                }
            }
        }

        using CoupledKernelFn = void(*)(void*);
        CoupledKernelFn compiled_fn = nullptr;
        monolithic_kernel->ensureCompiled();
        if (monolithicCompiledDispatchEnabled() && monolithic_kernel->hasCompiledDispatch()) {
            compiled_fn = reinterpret_cast<CoupledKernelFn>(monolithic_kernel->compiledCellAddress());
        } else if (!monolithicCompiledDispatchEnabled() &&
                   core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
            core::kernelTraceLog(
                core::KernelTraceChannel::Selection,
                "StandardAssembler::assembleCellsFused: monolithic compiled dispatch disabled; "
                "set SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1 to opt in");
        } else if (core::kernelTraceEnabled(core::KernelTraceChannel::Selection) &&
                   !monolithic_kernel->compileMessage().empty()) {
            core::kernelTraceLog(
                core::KernelTraceChannel::Selection,
                "StandardAssembler::assembleCellsFused: monolithic JIT fallback: " +
                    monolithic_kernel->compileMessage());
        }
        if ((!parent_term.assemble_matrix || !parent_term.assemble_vector) &&
            core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
            core::kernelTraceLog(
                core::KernelTraceChannel::Selection,
                compiled_fn != nullptr
                    ? "StandardAssembler::assembleCellsFused: monolithic compiled dispatch reused for one-sided request"
                    : "StandardAssembler::assembleCellsFused: monolithic exact fallback for one-sided request");
        }

        const bool run_compiled_dispatch =
            (compiled_fn != nullptr) && parent_term.assemble_matrix;
        const bool compiled_matrix_only_dispatch =
            run_compiled_dispatch && parent_term.assemble_vector;

        AssemblyContext shared_ctx;
        shared_ctx.reserve(max_dofs, max_qpts, mesh.dimension());
        std::deque<CellCoefficientCacheEntry> cell_coefficient_cache;
        std::deque<CellFieldEvaluationCacheEntry> cell_field_eval_cache;
        int compared_monolithic_cells = 0;
        double tp_m_geom = 0.0, tp_m_shared_field = 0.0, tp_m_basis = 0.0;
        double tp_m_block_field = 0.0, tp_m_dof = 0.0, tp_m_sol = 0.0;
        double tp_m_kernel = 0.0, tp_m_insert = 0.0;
        auto TP = assemblyTimeNow;

        std::vector<bool> monolithic_block_needs_solution(n_blocks, false);
        std::vector<bool> monolithic_block_use_coeffs_only(n_blocks, false);
        bool monolithic_any_need_solution = false;
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            const auto& bs = monolithic_kernel->blockSpec(bi);
            const auto block_required_data =
                bs.fallback_kernel ? bs.fallback_kernel->getRequiredData() : RequiredData::None;
            const bool need_solution =
                hasFlag(block_required_data, RequiredData::SolutionCoefficients) ||
                hasFlag(block_required_data, RequiredData::SolutionValues) ||
                hasFlag(block_required_data, RequiredData::SolutionGradients) ||
                hasFlag(block_required_data, RequiredData::SolutionHessians) ||
                hasFlag(block_required_data, RequiredData::SolutionLaplacians);
            monolithic_block_needs_solution[bi] = need_solution;
            monolithic_any_need_solution = monolithic_any_need_solution || need_solution;

            auto* jit = dynamic_cast<forms::jit::JITKernelWrapper*>(bs.fallback_kernel.get());
            if (jit != nullptr) {
                jit->ensureCompiled();
                monolithic_block_use_coeffs_only[bi] = jit->isJITReady();
            }
        }

        const int monolithic_required_history =
            (time_integration_ != nullptr) ? requiredHistoryStates(time_integration_) : 0;

        std::vector<std::vector<FieldId>> monolithic_block_field_ids(n_blocks);
        std::vector<bool> monolithic_block_copy_all_fields(n_blocks, false);
        std::vector<FieldId> monolithic_union_field_ids;
        monolithic_union_field_ids.reserve(union_field_reqs.size());
        for (const auto& req : union_field_reqs) {
            monolithic_union_field_ids.push_back(req.field);
        }
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            const auto& bs = monolithic_kernel->blockSpec(bi);
            if (!bs.fallback_kernel) {
                continue;
            }

            auto& field_ids = monolithic_block_field_ids[bi];
            for (const auto& req : bs.fallback_kernel->fieldRequirements()) {
                const auto it = std::find(field_ids.begin(), field_ids.end(), req.field);
                if (it == field_ids.end()) {
                    field_ids.push_back(req.field);
                }
            }
            monolithic_block_copy_all_fields[bi] =
                !field_ids.empty() &&
                field_ids.size() == monolithic_union_field_ids.size() &&
                std::equal(field_ids.begin(),
                           field_ids.end(),
                           monolithic_union_field_ids.begin(),
                           monolithic_union_field_ids.end());
        }

        const auto setCommonContextState = [&](AssemblyContext& ctx) {
            ctx.setMaterialState(nullptr, nullptr, 0u, 0u);
            ctx.setTimeIntegrationContext(time_integration_);
            ctx.setTime(time_);
            ctx.setTimeStep(dt_);
            ctx.setRealParameterGetter(get_real_param_);
            ctx.setParameterGetter(get_param_);
            ctx.setUserData(user_data_);
            ctx.setJITConstants(jit_constants_);
            ctx.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
            ctx.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
            ctx.clearAllPreviousSolutionData();
        };

        const auto applyBlockFieldCopy =
            [&](AssemblyContext& dst,
                const AssemblyContext& shared,
                std::size_t bi,
                GlobalIndex cell_id,
                std::deque<CellCoefficientCacheEntry>* coefficient_cache,
                std::deque<CellFieldEvaluationCacheEntry>* field_eval_cache) {
                const auto& field_ids = monolithic_block_field_ids[bi];
                if (field_ids.empty()) {
                    dst.clearFieldSolutionData();
                    return;
                }
                if (monolithic_block_copy_all_fields[bi]) {
                    dst.copyFieldSolutionDataFrom(shared);
                    return;
                }
                if (dst.copyFieldSolutionDataSubsetFrom(shared, std::span<const FieldId>(field_ids))) {
                    return;
                }

                const auto& bs = monolithic_kernel->blockSpec(bi);
                const auto field_reqs = bs.fallback_kernel->fieldRequirements();
                if (!field_reqs.empty()) {
                    populateFieldSolutionDataFast(
                        dst, mesh, cell_id, field_reqs, coefficient_cache, field_eval_cache);
                } else {
                    dst.clearFieldSolutionData();
                }
        };

        if (monolithic_any_need_solution) {
            FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
                        "assembleCellsFused: kernel requires solution but no solution was set");
            if (monolithic_required_history > 0) {
                FE_THROW_IF(previous_solutions_.size() <
                                static_cast<std::size_t>(monolithic_required_history),
                            FEException,
                            "assembleCellsFused: time integration requires " +
                                std::to_string(monolithic_required_history) +
                                " history states");
            }
        }

        const std::size_t monolithic_batch_size =
            (options_.use_batching && options_.batch_size > 1)
                ? static_cast<std::size_t>(options_.batch_size)
                : 1u;

        if (monolithic_batch_size > 1u) {
            struct BatchBlockWorkspace {
                AssemblyContext ctx;
                std::span<const GlobalIndex> row_dofs{};
                std::span<const GlobalIndex> col_dofs{};
            };

            struct DofGroupSlotCache {
                std::span<const GlobalIndex> dofs{};
                bool have_dofs{false};
            };

            struct TrialGroupSlotCache {
                std::span<const GlobalIndex> dofs{};
                std::span<const Real> solution_coeffs{};
                std::vector<std::span<const Real>> previous_solution_coeffs;
                bool have_dofs{false};
                bool gathered{false};
            };

            const auto old_sz = scratch_batch_contexts_.size();
            if (old_sz < monolithic_batch_size) {
                scratch_batch_contexts_.resize(monolithic_batch_size);
                for (std::size_t i = old_sz; i < monolithic_batch_size; ++i) {
                    scratch_batch_contexts_[i].reserve(max_dofs, max_qpts, mesh.dimension());
                }
            }
            if (max_dofs > scratch_batch_reserved_dofs_ ||
                max_qpts > scratch_batch_reserved_qpts_ ||
                mesh.dimension() != scratch_batch_reserved_dim_) {
                for (auto& batch_ctx : scratch_batch_contexts_) {
                    batch_ctx.reserve(max_dofs, max_qpts, mesh.dimension());
                }
                scratch_batch_reserved_dofs_ = max_dofs;
                scratch_batch_reserved_qpts_ = max_qpts;
                scratch_batch_reserved_dim_ = mesh.dimension();
            }

            scratch_batch_outputs_.resize(monolithic_batch_size);
            scratch_batch_context_ptrs_.assign(monolithic_batch_size, nullptr);
            scratch_saved_node_coords_.resize(monolithic_batch_size);
            if (coupled_slot_phys_cache_.size() < monolithic_batch_size) {
                coupled_slot_phys_cache_.resize(monolithic_batch_size);
            }

            auto& shared_contexts = scratch_batch_contexts_;
            auto& batch_outputs = scratch_batch_outputs_;
            auto& batch_context_ptrs = scratch_batch_context_ptrs_;
            auto& saved_node_coords = scratch_saved_node_coords_;
            std::vector<std::deque<CellCoefficientCacheEntry>> slot_coefficient_caches(
                monolithic_batch_size);
            std::vector<std::deque<CellFieldEvaluationCacheEntry>> slot_field_eval_caches(
                monolithic_batch_size);

            std::vector<BatchBlockWorkspace> batch_block_workspaces(monolithic_batch_size);
            for (auto& workspace : batch_block_workspaces) {
                workspace.ctx.reserve(max_dofs, max_qpts, mesh.dimension());
            }

            std::vector<int> row_group_of(n_blocks, -1);
            int n_row_groups = 0;
            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                if (row_group_of[bi] >= 0) {
                    continue;
                }
                const auto& bs_i = monolithic_kernel->blockSpec(bi);
                row_group_of[bi] = n_row_groups;
                for (std::size_t bj = bi + 1; bj < n_blocks; ++bj) {
                    if (row_group_of[bj] >= 0) {
                        continue;
                    }
                    const auto& bs_j = monolithic_kernel->blockSpec(bj);
                    if (bs_i.row_dof_map == bs_j.row_dof_map &&
                        bs_i.row_dof_offset == bs_j.row_dof_offset &&
                        bs_i.test_space == bs_j.test_space) {
                        row_group_of[bj] = n_row_groups;
                    }
                }
                ++n_row_groups;
            }

            std::vector<int> trial_group_of(n_blocks, -1);
            int n_trial_groups = 0;
            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                if (trial_group_of[bi] >= 0) {
                    continue;
                }
                const auto& bs_i = monolithic_kernel->blockSpec(bi);
                trial_group_of[bi] = n_trial_groups;
                for (std::size_t bj = bi + 1; bj < n_blocks; ++bj) {
                    if (trial_group_of[bj] >= 0) {
                        continue;
                    }
                    const auto& bs_j = monolithic_kernel->blockSpec(bj);
                    if (bs_i.col_dof_map == bs_j.col_dof_map &&
                        bs_i.col_dof_offset == bs_j.col_dof_offset &&
                        bs_i.trial_space == bs_j.trial_space) {
                        trial_group_of[bj] = n_trial_groups;
                    }
                }
                ++n_trial_groups;
            }
            std::vector<DofGroupSlotCache> row_group_cache(
                static_cast<std::size_t>(n_row_groups) * monolithic_batch_size);
            std::vector<TrialGroupSlotCache> tg_cache(
                static_cast<std::size_t>(n_trial_groups) * monolithic_batch_size);

            bool use_fused_insert = false;
            int fused_combined_n = 0;
            int fused_total_comps = 0;
            int fused_n_nodes = 0;
            std::vector<CombinedInsertBlockInfo> fused_info(n_blocks);

            if (!owned_rows_only && parent_term.assemble_matrix && n_blocks >= 2) {
                bool all_active = true;
                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                    const auto& bs = monolithic_kernel->blockSpec(bi);
                    if (!bs.fallback_kernel || !bs.fallback_kernel->hasCell()) {
                        all_active = false;
                        break;
                    }
                    if (!(parent_term.assemble_matrix && bs.want_matrix) &&
                        !(parent_term.assemble_vector && bs.want_vector)) {
                        all_active = false;
                        break;
                    }
                }

                if (all_active) {
                    struct DofSideInfo {
                        const dofs::DofMap* dof_map{nullptr};
                        GlobalIndex dof_offset{0};
                        int comps_per_node{0};
                        int comp_start{0};
                    };
                    std::vector<DofSideInfo> dof_sides;

                    auto register_side =
                        [&](const dofs::DofMap* dof_map, GlobalIndex dof_offset, int value_dim) {
                            for (const auto& side : dof_sides) {
                                if (side.dof_map == dof_map && side.dof_offset == dof_offset) {
                                    return;
                                }
                            }
                            dof_sides.push_back(DofSideInfo{
                                .dof_map = dof_map,
                                .dof_offset = dof_offset,
                                .comps_per_node = value_dim,
                                .comp_start = 0,
                            });
                        };

                    for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                        const auto& bs = monolithic_kernel->blockSpec(bi);
                        register_side(bs.row_dof_map, bs.row_dof_offset,
                                      bs.test_space->value_dimension());
                        register_side(bs.col_dof_map, bs.col_dof_offset,
                                      bs.trial_space->value_dimension());
                    }

                    for (auto& side : dof_sides) {
                        side.comp_start = fused_total_comps;
                        fused_total_comps += side.comps_per_node;
                    }

                    if (!dof_sides.empty() && !cell_ids.empty()) {
                        const auto sample = getCellDofsCached(
                            mesh, cell_ids.front(),
                            dof_sides.front().dof_map, dof_sides.front().dof_offset);
                        if (dof_sides.front().comps_per_node > 0 &&
                            static_cast<int>(sample.size()) % dof_sides.front().comps_per_node == 0) {
                            fused_n_nodes =
                                static_cast<int>(sample.size()) / dof_sides.front().comps_per_node;
                            fused_combined_n = fused_n_nodes * fused_total_comps;
                        }
                    }

                    if (fused_combined_n > 0 && fused_total_comps > 0) {
                        auto find_side =
                            [&](const dofs::DofMap* dof_map, GlobalIndex dof_offset)
                                -> const DofSideInfo& {
                            for (const auto& side : dof_sides) {
                                if (side.dof_map == dof_map && side.dof_offset == dof_offset) {
                                    return side;
                                }
                            }
                            return dof_sides.front();
                        };

                        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                            const auto& bs = monolithic_kernel->blockSpec(bi);
                            auto& fi = fused_info[bi];
                            const auto& row_side = find_side(bs.row_dof_map, bs.row_dof_offset);
                            const auto& col_side = find_side(bs.col_dof_map, bs.col_dof_offset);
                            fi.row_comp_start = row_side.comp_start;
                            fi.col_comp_start = col_side.comp_start;
                            fi.row_comps = row_side.comps_per_node;
                            fi.col_comps = col_side.comps_per_node;
                        }

                        resizeCombinedInsertScratch(monolithic_batch_size, fused_combined_n);
                        use_fused_insert = true;
                    }
                }
            }

            if (use_fused_insert && parent_term.assemble_matrix && parent_term.matrix_view &&
                parent_term.matrix_view->insertionCapabilities().contiguous_combined_matrix_insert &&
                parent_term.matrix_view->insertionCapabilities().resolved_matrix_entries) {
                const auto cn = static_cast<std::size_t>(fused_combined_n);
                const auto n_cells = mesh.numCells();
                const auto entries_per_cell = cn * cn;
                const auto total_entries =
                    static_cast<std::size_t>(n_cells) * entries_per_cell;

                const bool need_rebuild =
                    scratch_fused_resolved_offsets_.size() != static_cast<std::size_t>(n_cells) + 1u ||
                    scratch_fused_resolved_.size() != total_entries;

                if (need_rebuild && cn > 0 && n_cells > 0) {
                    scratch_fused_resolved_offsets_.resize(
                        static_cast<std::size_t>(n_cells) + 1u);
                    scratch_fused_resolved_.resize(total_entries);

                    std::vector<GlobalIndex> combined_dofs(cn);
                    for (GlobalIndex cell_id = 0; cell_id < n_cells; ++cell_id) {
                        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                            const auto& bs = monolithic_kernel->blockSpec(bi);
                            const auto& fi = fused_info[bi];

                            const auto rd = getCellDofsCached(
                                mesh, cell_id, bs.row_dof_map, bs.row_dof_offset);
                            const int n_br = static_cast<int>(rd.size());
                            for (int i = 0; i < n_br; ++i) {
                                const int ci = (i / fi.row_comps) * fused_total_comps +
                                               fi.row_comp_start + (i % fi.row_comps);
                                combined_dofs[static_cast<std::size_t>(ci)] =
                                    rd[static_cast<std::size_t>(i)];
                            }

                            const auto cd = getCellDofsCached(
                                mesh, cell_id, bs.col_dof_map, bs.col_dof_offset);
                            const int n_bc = static_cast<int>(cd.size());
                            for (int j = 0; j < n_bc; ++j) {
                                const int cj = (j / fi.col_comps) * fused_total_comps +
                                               fi.col_comp_start + (j % fi.col_comps);
                                combined_dofs[static_cast<std::size_t>(cj)] =
                                    cd[static_cast<std::size_t>(j)];
                            }
                        }

                        const auto offset =
                            static_cast<std::size_t>(cell_id) * entries_per_cell;
                        scratch_fused_resolved_offsets_[static_cast<std::size_t>(cell_id)] =
                            static_cast<GlobalIndex>(offset);
                        parent_term.matrix_view->resolveMatrixEntries(
                            std::span<const GlobalIndex>(combined_dofs),
                            std::span<const GlobalIndex>(combined_dofs),
                            std::span<GlobalIndex>(
                                scratch_fused_resolved_.data() + offset,
                                entries_per_cell));
                    }
                    scratch_fused_resolved_offsets_[static_cast<std::size_t>(n_cells)] =
                        static_cast<GlobalIndex>(total_entries);
                }
            }

            const auto prepareBatchOutput =
                [](KernelOutput& output,
                   LocalIndex n_test,
                   LocalIndex n_trial,
                   bool want_matrix,
                   bool want_vector) {
                    if (output.n_test_dofs != n_test ||
                        output.n_trial_dofs != n_trial ||
                        output.has_matrix != want_matrix ||
                        output.has_vector != want_vector) {
                        output.reserveNoZero(n_test, n_trial, want_matrix, want_vector);
                    }
                    output.n_test_dofs = n_test;
                    output.n_trial_dofs = n_trial;
                    output.has_matrix = want_matrix;
                    output.has_vector = want_vector;
                    output.clear();
                };

            const auto restorePreparedGeometry = [&](std::size_t slot) {
                cached_geom_h_ = saved_node_coords[slot].entity_h;
                cached_geom_volume_ = saved_node_coords[slot].entity_volume;
                if (!cached_mapping_affine_) {
                    scratch_node_coords_ = saved_node_coords[slot].node_coords;
                    cached_mapping_->resetNodes(scratch_node_coords_);
                }
            };

            std::size_t run_begin = 0u;
            while (run_begin < cell_ids.size()) {
                const auto run_type = mesh.getCellType(cell_ids[run_begin]);
                std::size_t run_end = run_begin + 1u;
                while (run_end < cell_ids.size() && mesh.getCellType(cell_ids[run_end]) == run_type) {
                    ++run_end;
                }

                for (std::size_t begin = run_begin; begin < run_end; begin += monolithic_batch_size) {
                    const std::size_t active = std::min(monolithic_batch_size, run_end - begin);
                    for (auto& cache : row_group_cache) {
                        cache.have_dofs = false;
                        cache.dofs = {};
                    }
                    for (auto& cache : tg_cache) {
                        cache.have_dofs = false;
                        cache.dofs = {};
                        cache.gathered = false;
                        cache.solution_coeffs = {};
                        cache.previous_solution_coeffs.clear();
                    }
                    for (std::size_t slot = 0; slot < active; ++slot) {
                        slot_coefficient_caches[slot].clear();
                        slot_coefficient_caches[slot].resize(0);
                        slot_field_eval_caches[slot].clear();
                        slot_field_eval_caches[slot].resize(0);
                    }

                    if (use_fused_insert) {
                        zeroCombinedInsertScratch(active, fused_combined_n);
                    }

                    for (std::size_t slot = 0; slot < active; ++slot) {
                        const auto cell_id = cell_ids[begin + slot];
                        auto& shared = shared_contexts[slot];

                        double tp0 = TP();
                        prepareGeometry(shared, mesh, cell_id, *fused_quad_rule);
                        tp_m_geom += TP() - tp0;

                        saved_node_coords[slot].node_coords = scratch_node_coords_;
                        saved_node_coords[slot].entity_h = cached_geom_h_;
                        saved_node_coords[slot].entity_volume = cached_geom_volume_;

                        setCommonContextState(shared);
                        if (any_need_field_solutions) {
                            tp0 = TP();
                            populateFieldSolutionDataFast(
                                shared, mesh, cell_id, union_field_reqs,
                                &slot_coefficient_caches[slot],
                                &slot_field_eval_caches[slot]);
                            tp_m_shared_field += TP() - tp0;
                        }
                    }

                    const bool use_batch_basis =
                        use_coupled_scalar_cache && cached_mapping_affine_;
                    if (use_batch_basis) {
                        const auto nq = coupled_scalar_n_qpts_;
                        const auto ns = coupled_scalar_n_dofs_;
                        const bool need_hess = coupled_scalar_has_hessians_;
                        const int dim = mesh.dimension();

                        for (std::size_t slot = 0; slot < active; ++slot) {
                            const auto& ctx = shared_contexts[slot];
                            const auto& J_inv = ctx.inverseJacobians().front();
                            auto& slotc = coupled_slot_phys_cache_[slot];

                            for (LocalIndex si = 0; si < ns; ++si) {
                                for (LocalIndex q = 0; q < nq; ++q) {
                                    const auto ref_idx = static_cast<std::size_t>(si * nq + q);
                                    const auto& gr = coupled_scalar_ref_grads_[ref_idx];
                                    auto& gp = slotc.phys_grads[q * ns + si];
                                    if (dim == 3) {
                                        gp[0] = J_inv[0][0] * gr[0] + J_inv[1][0] * gr[1] + J_inv[2][0] * gr[2];
                                        gp[1] = J_inv[0][1] * gr[0] + J_inv[1][1] * gr[1] + J_inv[2][1] * gr[2];
                                        gp[2] = J_inv[0][2] * gr[0] + J_inv[1][2] * gr[1] + J_inv[2][2] * gr[2];
                                    } else if (dim == 2) {
                                        gp[0] = J_inv[0][0] * gr[0] + J_inv[1][0] * gr[1];
                                        gp[1] = J_inv[0][1] * gr[0] + J_inv[1][1] * gr[1];
                                        gp[2] = 0.0;
                                    } else {
                                        gp[0] = J_inv[0][0] * gr[0];
                                        gp[1] = 0.0;
                                        gp[2] = 0.0;
                                    }
                                }
                            }

                            if (need_hess) {
                                for (LocalIndex si = 0; si < ns; ++si) {
                                    for (LocalIndex q = 0; q < nq; ++q) {
                                        const auto ref_idx = static_cast<std::size_t>(si * nq + q);
                                        const auto& Hr = coupled_scalar_ref_hess_[ref_idx];
                                        auto& Hp = slotc.phys_hess[q * ns + si];
                                        for (int r = 0; r < dim; ++r) {
                                            for (int c = 0; c < dim; ++c) {
                                                Real s = 0.0;
                                                for (int a = 0; a < dim; ++a) {
                                                    for (int b = 0; b < dim; ++b) {
                                                        s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                                             Hr[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                                             J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                                    }
                                                }
                                                Hp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = s;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (std::size_t slot = 0; slot < active; ++slot) {
                        batch_block_workspaces[slot].ctx.copyGeometryDataFrom(shared_contexts[slot]);
                    }

                    const auto prepareMonolithicBatchBlock =
                        [&](std::size_t bi,
                            std::size_t slot,
                            AssemblyContext& ctx,
                            std::span<const GlobalIndex>& row_dofs,
                            std::span<const GlobalIndex>& col_dofs,
                            KernelOutput& output,
                            bool want_matrix,
                            bool want_vector) {
                            const auto cell_id = cell_ids[begin + slot];
                            const auto& bs = monolithic_kernel->blockSpec(bi);
                            auto& shared = shared_contexts[slot];
                            const bool use_expansion_this_block =
                                use_batch_basis || (use_coupled_scalar_cache && bi > 0);
                            const bool preloaded_geometry_ctx =
                                (&ctx == &batch_block_workspaces[slot].ctx);

                            double tp0 = TP();
                            if (!preloaded_geometry_ctx) {
                                ctx.copyGeometryDataFrom(shared);
                            }
                            if (use_expansion_this_block) {
                                const auto& meta = cached_coupled_block_meta_[bi];
                                ctx.configureForCoupledBlock(
                                    cell_id, mesh.getCellDomainId(cell_id), meta);

                                const auto n_test = meta.n_test_dofs;
                                const auto n_trial = meta.n_trial_dofs;
                                const auto nq = coupled_scalar_n_qpts_;
                                const auto ns = coupled_scalar_n_dofs_;
                                const bool same_sp = meta.trial_is_test;
                                const bool need_hess =
                                    hasFlag(meta.required_data, RequiredData::BasisHessians);
                                const auto& slotc = coupled_slot_phys_cache_[slot];

                                const auto test_count = static_cast<std::size_t>(n_test * nq);
                                auto* tg = ctx.testPhysGradientsWritePtr(test_count);
                                for (LocalIndex q = 0; q < nq; ++q) {
                                    for (LocalIndex i = 0; i < n_test; ++i) {
                                        const auto si =
                                            static_cast<LocalIndex>(i % static_cast<LocalIndex>(ns));
                                        tg[static_cast<std::size_t>(q * n_test + i)] =
                                            slotc.phys_grads[q * ns + si];
                                    }
                                }

                                if (need_hess) {
                                    auto* th = ctx.testPhysHessiansWritePtr(test_count);
                                    for (LocalIndex q = 0; q < nq; ++q) {
                                        for (LocalIndex i = 0; i < n_test; ++i) {
                                            const auto si =
                                                static_cast<LocalIndex>(i % static_cast<LocalIndex>(ns));
                                            th[static_cast<std::size_t>(q * n_test + i)] =
                                                slotc.phys_hess[q * ns + si];
                                        }
                                    }
                                }

                                if (!same_sp) {
                                    const auto trial_count = static_cast<std::size_t>(n_trial * nq);
                                    auto* trg = ctx.trialPhysGradientsWritePtr(trial_count);
                                    for (LocalIndex q = 0; q < nq; ++q) {
                                        for (LocalIndex j = 0; j < n_trial; ++j) {
                                            const auto sj =
                                                static_cast<LocalIndex>(j % static_cast<LocalIndex>(ns));
                                            trg[static_cast<std::size_t>(q * n_trial + j)] =
                                                slotc.phys_grads[q * ns + sj];
                                        }
                                    }
                                    if (need_hess) {
                                        auto* trh = ctx.trialPhysHessiansWritePtr(trial_count);
                                        for (LocalIndex q = 0; q < nq; ++q) {
                                            for (LocalIndex j = 0; j < n_trial; ++j) {
                                                const auto sj =
                                                    static_cast<LocalIndex>(j % static_cast<LocalIndex>(ns));
                                                trh[static_cast<std::size_t>(q * n_trial + j)] =
                                                    slotc.phys_hess[q * ns + sj];
                                            }
                                        }
                                    }
                                }

                                if (const auto* tc = findCoupledQptCache(n_test)) {
                                    ctx.setTestBasisValuesOnlyQptMajor(n_test, *tc);
                                }
                                if (!same_sp) {
                                    if (const auto* trc = findCoupledQptCache(n_trial)) {
                                        ctx.setTrialBasisValuesOnlyQptMajor(n_trial, *trc);
                                    }
                                }

                                if (hasFlag(meta.required_data, RequiredData::EntityMeasures)) {
                                    ctx.setEntityMeasures(
                                        shared.cellDiameter(), shared.cellVolume(), 0.0);
                                }
                            } else {
                                restorePreparedGeometry(slot);
                                const auto* saved_coupled_meta = active_coupled_block_meta_;
                                active_coupled_block_meta_ = &cached_coupled_block_meta_[bi];
                                try {
                                    prepareBasis(ctx, mesh, cell_id, *bs.test_space, *bs.trial_space,
                                                 bs.fallback_kernel->getRequiredData(),
                                                 *fused_quad_rule);
                                } catch (...) {
                                    active_coupled_block_meta_ = saved_coupled_meta;
                                    throw;
                                }
                                active_coupled_block_meta_ = saved_coupled_meta;

                                if (use_coupled_scalar_cache && bi == 0) {
                                    const auto nq = coupled_scalar_n_qpts_;
                                    const auto ns = coupled_scalar_n_dofs_;
                                    const auto n_test = static_cast<LocalIndex>(
                                        bs.test_space->dofs_per_element());
                                    auto& slotc = coupled_slot_phys_cache_[slot];
                                    const auto tg_raw = ctx.testPhysicalGradientsRaw();
                                    for (LocalIndex q = 0; q < nq; ++q) {
                                        for (LocalIndex si = 0; si < ns; ++si) {
                                            slotc.phys_grads[q * ns + si] =
                                                tg_raw[static_cast<std::size_t>(q * n_test + si)];
                                        }
                                    }

                                    if (coupled_scalar_has_hessians_) {
                                        const auto th_raw = ctx.testPhysicalHessiansRaw();
                                        for (LocalIndex q = 0; q < nq; ++q) {
                                            for (LocalIndex si = 0; si < ns; ++si) {
                                                slotc.phys_hess[q * ns + si] =
                                                    th_raw[static_cast<std::size_t>(q * n_test + si)];
                                            }
                                        }
                                    }
                                }
                            }
                            tp_m_basis += TP() - tp0;

                            setCommonContextState(ctx);

                            tp0 = TP();
                            applyBlockFieldCopy(
                                ctx,
                                shared,
                                bi,
                                cell_id,
                                &slot_coefficient_caches[slot],
                                &slot_field_eval_caches[slot]);
                            tp_m_block_field += TP() - tp0;

                            const auto rg_index =
                                static_cast<std::size_t>(row_group_of[bi]) * monolithic_batch_size + slot;
                            auto& row_cache = row_group_cache[rg_index];
                            const auto tg_index =
                                static_cast<std::size_t>(trial_group_of[bi]) * monolithic_batch_size + slot;
                            auto& group_cache = tg_cache[tg_index];

                            tp0 = TP();
                            if (!row_cache.have_dofs) {
                                row_cache.dofs = getCellDofsCached(
                                    mesh, cell_id, bs.row_dof_map, bs.row_dof_offset);
                                row_cache.have_dofs = true;
                            }
                            row_dofs = row_cache.dofs;
                            if (!group_cache.have_dofs) {
                                group_cache.dofs = getCellDofsCached(
                                    mesh, cell_id, bs.col_dof_map, bs.col_dof_offset);
                                group_cache.have_dofs = true;
                            }
                            col_dofs = group_cache.dofs;
                            tp_m_dof += TP() - tp0;

                            if (monolithic_block_needs_solution[bi]) {
                                tp0 = TP();
                                if (!group_cache.gathered) {
                                    group_cache.solution_coeffs =
                                        gatherCachedCellVectorCoefficients(
                                            slot_coefficient_caches[slot],
                                            mesh,
                                            cell_id,
                                            bs.col_dof_map,
                                            bs.col_dof_offset,
                                            bs.trial_space,
                                            group_cache.dofs,
                                            /*history_index=*/0,
                                            ctx.trialUsesVectorBasis(),
                                            "assembleCellsFused");

                                    if (monolithic_required_history > 0) {
                                        group_cache.previous_solution_coeffs.resize(
                                            static_cast<std::size_t>(monolithic_required_history));
                                        for (int k = 1; k <= monolithic_required_history; ++k) {
                                            group_cache.previous_solution_coeffs[
                                                static_cast<std::size_t>(k - 1)] =
                                                gatherCachedCellVectorCoefficients(
                                                    slot_coefficient_caches[slot],
                                                    mesh,
                                                    cell_id,
                                                    bs.col_dof_map,
                                                    bs.col_dof_offset,
                                                    bs.trial_space,
                                                    group_cache.dofs,
                                                    k,
                                                    ctx.trialUsesVectorBasis(),
                                                    "assembleCellsFused");
                                        }
                                    }
                                    group_cache.gathered = true;
                                }

                                if (monolithic_block_use_coeffs_only[bi]) {
                                    ctx.setSolutionCoefficientsOnly(group_cache.solution_coeffs);
                                } else {
                                    ctx.setSolutionCoefficients(group_cache.solution_coeffs);
                                }
                                for (int k = 1; k <= monolithic_required_history; ++k) {
                                    if (monolithic_block_use_coeffs_only[bi]) {
                                        ctx.setPreviousSolutionCoefficientsOnlyK(
                                            k,
                                            group_cache.previous_solution_coeffs[
                                                static_cast<std::size_t>(k - 1)]);
                                    } else {
                                        ctx.setPreviousSolutionCoefficientsK(
                                            k,
                                            group_cache.previous_solution_coeffs[
                                                static_cast<std::size_t>(k - 1)]);
                                    }
                                }
                                tp_m_sol += TP() - tp0;
                            }

                            prepareBatchOutput(
                                output,
                                static_cast<LocalIndex>(row_dofs.size()),
                                static_cast<LocalIndex>(col_dofs.size()),
                                want_matrix,
                                want_vector);
                        };

                    if (run_compiled_dispatch) {
                        struct CompiledBlockWorkspace {
                            AssemblyContext ctx;
                            KernelOutput output;
                            std::span<const GlobalIndex> row_dofs{};
                            std::span<const GlobalIndex> col_dofs{};
                        };

                        std::vector<CompiledBlockWorkspace> compiled_workspaces(active * n_blocks);
                        for (auto& workspace : compiled_workspaces) {
                            workspace.ctx.reserve(max_dofs, max_qpts, mesh.dimension());
                        }
                        std::vector<assembly::jit::CoupledBlockView> compiled_block_views(active * n_blocks);
                        std::vector<assembly::jit::CoupledCellKernelArgsV1> compiled_element_args(active);
                        std::vector<KernelOutput> compiled_vector_outputs(active);

                        auto compiled_workspace =
                            [&](std::size_t slot, std::size_t bi) -> CompiledBlockWorkspace& {
                                return compiled_workspaces[slot * n_blocks + bi];
                            };

                        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                            const auto& bs = monolithic_kernel->blockSpec(bi);
                            const bool block_want_matrix =
                                bs.want_matrix && parent_term.assemble_matrix;
                            const bool block_want_vector =
                                bs.want_vector && parent_term.assemble_vector;
                            if (!bs.fallback_kernel || (!block_want_matrix && !block_want_vector)) {
                                continue;
                            }

                            const bool compiled_want_vector =
                                block_want_vector && !compiled_matrix_only_dispatch;
                            for (std::size_t slot = 0; slot < active; ++slot) {
                                auto& workspace = compiled_workspace(slot, bi);
                                prepareMonolithicBatchBlock(
                                    bi,
                                    slot,
                                    workspace.ctx,
                                    workspace.row_dofs,
                                    workspace.col_dofs,
                                    workspace.output,
                                    block_want_matrix,
                                    compiled_want_vector);
                                auto view = assembly::jit::packCoupledBlockView(
                                    workspace.ctx, workspace.output);
                                if (compiled_matrix_only_dispatch) {
                                    view.element_vector = nullptr;
                                }
                                compiled_block_views[slot * n_blocks + bi] = view;
                            }
                        }

                        double tp0 = TP();
                        for (std::size_t slot = 0; slot < active; ++slot) {
                            compiled_element_args[slot] =
                                assembly::jit::packCoupledCellKernelArgsV1(
                                    shared_contexts[slot],
                                    std::span<const assembly::jit::CoupledBlockView>(
                                        compiled_block_views.data() + slot * n_blocks,
                                        n_blocks));
                        }
                        assembly::jit::CoupledCellKernelBatchArgsV1 batch_args;
                        batch_args.abi_version = assembly::jit::kCoupledCellKernelABIV1;
                        batch_args.batch_size = static_cast<std::uint32_t>(active);
                        batch_args.num_blocks = static_cast<std::uint32_t>(n_blocks);
                        batch_args.elements = compiled_element_args.data();
                        compiled_fn(reinterpret_cast<void*>(&batch_args));
                        tp_m_kernel += TP() - tp0;

                        if (monolithicCompiledCompareEnabled() &&
                            compared_monolithic_cells < monolithicCompiledCompareMaxCells()) {
                            const Real tol = monolithicCompiledCompareTolerance();
                            const std::size_t compare_active = std::min<std::size_t>(
                                active,
                                static_cast<std::size_t>(
                                    monolithicCompiledCompareMaxCells() - compared_monolithic_cells));
                            for (std::size_t slot = 0; slot < compare_active; ++slot) {
                                ++compared_monolithic_cells;
                                const auto cell_id = cell_ids[begin + slot];
                                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                                    const auto& bs = monolithic_kernel->blockSpec(bi);
                                    auto& workspace = compiled_workspace(slot, bi);
                                    if (!bs.fallback_kernel || workspace.output.local_matrix.empty()) {
                                        continue;
                                    }

                                    assembly::KernelOutput exact_output;
                                    exact_output.n_test_dofs = workspace.output.n_test_dofs;
                                    exact_output.n_trial_dofs = workspace.output.n_trial_dofs;
                                    exact_output.has_matrix = true;
                                    exact_output.has_vector = false;
                                    exact_output.local_matrix.assign(
                                        workspace.output.local_matrix.size(), Real(0));

                                    try {
                                        bs.fallback_kernel->computeCell(workspace.ctx, exact_output);
                                    } catch (const std::exception& e) {
                                        std::ostringstream oss;
                                        oss << "StandardAssembler::assembleCellsFused: compiled-vs-fallback compare failed"
                                            << " during exact block evaluation"
                                            << " cell=" << cell_id
                                            << " block=" << bi
                                            << " test_field=" << bs.test_field
                                            << " trial_field=" << bs.trial_field
                                            << " kernel='" << bs.fallback_kernel->name() << "'"
                                            << " what=" << e.what();
                                        throw FEException(oss.str(), __FILE__, __LINE__, __func__);
                                    }

                                    Real max_matrix_diff = 0.0;
                                    std::size_t max_matrix_idx = 0;
                                    for (std::size_t idx = 0; idx < workspace.output.local_matrix.size(); ++idx) {
                                        const Real diff = std::abs(
                                            workspace.output.local_matrix[idx] - exact_output.local_matrix[idx]);
                                        if (diff > max_matrix_diff) {
                                            max_matrix_diff = diff;
                                            max_matrix_idx = idx;
                                        }
                                    }

                                    if (max_matrix_diff > tol) {
                                        std::ostringstream oss;
                                        oss.setf(std::ios::scientific);
                                        oss.precision(16);
                                        oss << "StandardAssembler::assembleCellsFused: monolithic compiled dispatch mismatch"
                                            << " cell=" << cell_id
                                            << " block=" << bi
                                            << " test_field=" << bs.test_field
                                            << " trial_field=" << bs.trial_field
                                            << " kernel='" << bs.fallback_kernel->name() << "'"
                                            << " matrix_max_diff=" << max_matrix_diff
                                            << " matrix_idx=" << max_matrix_idx
                                            << " compiled=" << workspace.output.local_matrix[max_matrix_idx]
                                            << " exact=" << exact_output.local_matrix[max_matrix_idx]
                                            << " vector_max_diff=0.0000000000000000e+00";
                                        throw FEException(oss.str(), __FILE__, __LINE__, __func__);
                                    }
                                }
                            }
                        }

                        if (compiled_matrix_only_dispatch) {
                            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                                const auto& bs = monolithic_kernel->blockSpec(bi);
                                const bool block_want_vector =
                                    bs.want_vector && parent_term.assemble_vector;
                                if (!bs.fallback_kernel || !block_want_vector) {
                                    continue;
                                }

                                for (std::size_t slot = 0; slot < active; ++slot) {
                                    auto& workspace = compiled_workspace(slot, bi);
                                    prepareBatchOutput(
                                        compiled_vector_outputs[slot],
                                        workspace.output.n_test_dofs,
                                        workspace.output.n_trial_dofs,
                                        /*want_matrix=*/false,
                                        /*want_vector=*/true);
                                    batch_context_ptrs[slot] = &workspace.ctx;
                                }

                                tp0 = TP();
                                bs.fallback_kernel->computeCellBatch(
                                    std::span<const AssemblyContext* const>(batch_context_ptrs.data(), active),
                                    std::span<KernelOutput>(compiled_vector_outputs.data(), active));
                                tp_m_kernel += TP() - tp0;

                                for (std::size_t slot = 0; slot < active; ++slot) {
                                    auto& workspace = compiled_workspace(slot, bi);
                                    workspace.output.local_vector =
                                        std::move(compiled_vector_outputs[slot].local_vector);
                                    workspace.output.has_vector = !workspace.output.local_vector.empty();
                                }
                            }
                        }

                        tp0 = TP();
                        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                            const auto& bs = monolithic_kernel->blockSpec(bi);
                            const bool block_want_matrix =
                                bs.want_matrix && parent_term.assemble_matrix;
                            const bool block_want_vector =
                                bs.want_vector && parent_term.assemble_vector;
                            if (!bs.fallback_kernel || (!block_want_matrix && !block_want_vector)) {
                                continue;
                            }

                            for (std::size_t slot = 0; slot < active; ++slot) {
                                const auto cell_id = cell_ids[begin + slot];
                                auto& workspace = compiled_workspace(slot, bi);
                                auto& output = workspace.output;
                                output.has_matrix = block_want_matrix && !output.local_matrix.empty();
                                output.has_vector = block_want_vector && !output.local_vector.empty();
                                if (!output.has_matrix && !output.has_vector) {
                                    continue;
                                }
                                if (workspace.ctx.testUsesVectorBasis() || workspace.ctx.trialUsesVectorBasis()) {
                                    applyVectorBasisOutputOrientation(
                                        mesh, cell_id, *bs.test_space, cell_id, *bs.trial_space, output);
                                }

                                if (use_fused_insert) {
                                    scatterCombinedInsertBlockOutput(
                                        slot, output,
                                        workspace.row_dofs, workspace.col_dofs,
                                        fused_info[bi], fused_total_comps, fused_combined_n,
                                        block_want_matrix, block_want_vector);
                                } else {
                                    auto& insert = block_inserts[bi];
                                    insertLocalForCell(
                                        cell_id,
                                        bs.row_dof_map, bs.row_dof_offset,
                                        bs.col_dof_map, bs.col_dof_offset,
                                        output,
                                        workspace.row_dofs, workspace.col_dofs,
                                        insert.insert_matrix, insert.insert_vector);
                                }

                                if (output.has_matrix) {
                                    result.matrix_entries_inserted += static_cast<GlobalIndex>(
                                        workspace.row_dofs.size() * workspace.col_dofs.size());
                                }
                                if (output.has_vector) {
                                    result.vector_entries_inserted +=
                                        static_cast<GlobalIndex>(workspace.row_dofs.size());
                                }
                            }
                        }
                        tp_m_insert += TP() - tp0;
                    } else {
                        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                            const auto& bs = monolithic_kernel->blockSpec(bi);
                            const bool block_want_matrix =
                                bs.want_matrix && parent_term.assemble_matrix;
                            const bool block_want_vector =
                                bs.want_vector && parent_term.assemble_vector;
                            if (!bs.fallback_kernel || (!block_want_matrix && !block_want_vector)) {
                                continue;
                            }

                            for (std::size_t slot = 0; slot < active; ++slot) {
                                auto& workspace = batch_block_workspaces[slot];
                                auto& output = batch_outputs[slot];
                                prepareMonolithicBatchBlock(
                                    bi,
                                    slot,
                                    workspace.ctx,
                                    workspace.row_dofs,
                                    workspace.col_dofs,
                                    output,
                                    block_want_matrix,
                                    block_want_vector);
                                batch_context_ptrs[slot] = &workspace.ctx;
                            }

                            double tp0 = TP();
                            bs.fallback_kernel->computeCellBatch(
                                std::span<const AssemblyContext* const>(batch_context_ptrs.data(), active),
                                std::span<KernelOutput>(batch_outputs.data(), active));
                            tp_m_kernel += TP() - tp0;

                            tp0 = TP();
                            for (std::size_t slot = 0; slot < active; ++slot) {
                                const auto cell_id = cell_ids[begin + slot];
                                auto& workspace = batch_block_workspaces[slot];
                                auto& output = batch_outputs[slot];
                                if (!output.has_matrix && !output.has_vector) {
                                    continue;
                                }
                                if (workspace.ctx.testUsesVectorBasis() || workspace.ctx.trialUsesVectorBasis()) {
                                    applyVectorBasisOutputOrientation(
                                        mesh, cell_id, *bs.test_space, cell_id, *bs.trial_space, output);
                                }

                                if (use_fused_insert) {
                                    scatterCombinedInsertBlockOutput(
                                        slot, output,
                                        workspace.row_dofs, workspace.col_dofs,
                                        fused_info[bi], fused_total_comps, fused_combined_n,
                                        block_want_matrix, block_want_vector);
                                } else {
                                    auto& insert = block_inserts[bi];
                                    insertLocalForCell(
                                        cell_id,
                                        bs.row_dof_map, bs.row_dof_offset,
                                        bs.col_dof_map, bs.col_dof_offset,
                                        output,
                                        workspace.row_dofs, workspace.col_dofs,
                                        insert.insert_matrix, insert.insert_vector);
                                }

                                if (output.has_matrix) {
                                    result.matrix_entries_inserted += static_cast<GlobalIndex>(
                                        workspace.row_dofs.size() * workspace.col_dofs.size());
                                }
                                if (output.has_vector) {
                                    result.vector_entries_inserted +=
                                        static_cast<GlobalIndex>(workspace.row_dofs.size());
                                }
                            }
                            tp_m_insert += TP() - tp0;
                        }
                    }

                    if (use_fused_insert) {
                        double tp0 = TP();
                        flushCombinedInsertBatch(
                            std::span<const GlobalIndex>(cell_ids.data() + begin, active),
                            fused_combined_n,
                            CombinedInsertTarget{
                                .matrix_view = parent_term.matrix_view,
                                .vector_view = parent_term.vector_view,
                                .assemble_matrix = parent_term.assemble_matrix,
                                .assemble_vector = parent_term.assemble_vector,
                            });
                        tp_m_insert += TP() - tp0;
                    }

                    result.elements_assembled += static_cast<GlobalIndex>(active);
                }

                run_begin = run_end;
            }

            auto end_time = std::chrono::steady_clock::now();
            result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
            if (assemblyTimingEnabled()) {
                const double total = tp_m_geom + tp_m_shared_field + tp_m_basis +
                                     tp_m_block_field + tp_m_dof + tp_m_sol +
                                     tp_m_kernel + tp_m_insert;
                if (total > 0.0) {
                    std::fprintf(
                        stdout,
                        "    --- monolithic cellLoop TIMING (rank 0, %zu cells, %zu blocks, batch=%zu) ---\n"
                        "      geometry:        %9.6f s  (%5.1f%%)\n"
                        "      shared fields:   %9.6f s  (%5.1f%%)\n"
                        "      prepareBasis:    %9.6f s  (%5.1f%%)\n"
                        "      block fields:    %9.6f s  (%5.1f%%)\n"
                        "      dof lookup:      %9.6f s  (%5.1f%%)\n"
                        "      solution gather: %9.6f s  (%5.1f%%)\n"
                        "      kernel:          %9.6f s  (%5.1f%%)\n"
                        "      insert:          %9.6f s  (%5.1f%%)\n"
                        "      TOTAL:           %9.6f s\n"
                        "    -----------------------------------------------\n",
                        static_cast<std::size_t>(cell_ids.size()),
                        n_blocks,
                        monolithic_batch_size,
                        tp_m_geom,         100.0 * tp_m_geom / total,
                        tp_m_shared_field, 100.0 * tp_m_shared_field / total,
                        tp_m_basis,        100.0 * tp_m_basis / total,
                        tp_m_block_field,  100.0 * tp_m_block_field / total,
                        tp_m_dof,          100.0 * tp_m_dof / total,
                        tp_m_sol,          100.0 * tp_m_sol / total,
                        tp_m_kernel,       100.0 * tp_m_kernel / total,
                        tp_m_insert,       100.0 * tp_m_insert / total,
                        total);
                }
            }
            return result;
        }

        std::vector<int> row_group_of(n_blocks, -1);
        int n_row_groups = 0;
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            if (row_group_of[bi] >= 0) {
                continue;
            }
            const auto& bs_i = monolithic_kernel->blockSpec(bi);
            row_group_of[bi] = n_row_groups;
            for (std::size_t bj = bi + 1; bj < n_blocks; ++bj) {
                if (row_group_of[bj] >= 0) {
                    continue;
                }
                const auto& bs_j = monolithic_kernel->blockSpec(bj);
                if (bs_i.row_dof_map == bs_j.row_dof_map &&
                    bs_i.row_dof_offset == bs_j.row_dof_offset &&
                    bs_i.test_space == bs_j.test_space) {
                    row_group_of[bj] = n_row_groups;
                }
            }
            ++n_row_groups;
        }

        std::vector<int> trial_group_of(n_blocks, -1);
        int n_trial_groups = 0;
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            if (trial_group_of[bi] >= 0) {
                continue;
            }
            const auto& bs_i = monolithic_kernel->blockSpec(bi);
            trial_group_of[bi] = n_trial_groups;
            for (std::size_t bj = bi + 1; bj < n_blocks; ++bj) {
                if (trial_group_of[bj] >= 0) {
                    continue;
                }
                const auto& bs_j = monolithic_kernel->blockSpec(bj);
                if (bs_i.col_dof_map == bs_j.col_dof_map &&
                    bs_i.col_dof_offset == bs_j.col_dof_offset &&
                    bs_i.trial_space == bs_j.trial_space) {
                    trial_group_of[bj] = n_trial_groups;
                }
            }
            ++n_trial_groups;
        }

        struct CellDofGroupCache {
            std::span<const GlobalIndex> dofs{};
            bool have_dofs{false};
        };

        struct CellTrialGroupCache {
            std::span<const GlobalIndex> dofs{};
            std::span<const Real> solution_coeffs{};
            std::vector<std::span<const Real>> previous_solution_coeffs;
            bool have_dofs{false};
            bool gathered{false};
        };

        std::vector<CellDofGroupCache> cell_row_group_cache(
            static_cast<std::size_t>(n_row_groups));
        std::vector<CellTrialGroupCache> cell_trial_group_cache(
            static_cast<std::size_t>(n_trial_groups));

        for (const auto cell_id : cell_ids) {
            cell_coefficient_cache.clear();
            cell_field_eval_cache.clear();
            for (auto& cache : cell_row_group_cache) {
                cache.have_dofs = false;
                cache.dofs = {};
            }
            for (auto& cache : cell_trial_group_cache) {
                cache.have_dofs = false;
                cache.dofs = {};
                cache.solution_coeffs = {};
                cache.previous_solution_coeffs.clear();
                cache.gathered = false;
            }
            double tp0 = TP();
            prepareGeometry(shared_ctx, mesh, cell_id, *fused_quad_rule);
            tp_m_geom += TP() - tp0;
            setCommonContextState(shared_ctx);
            if (any_need_field_solutions) {
                tp0 = TP();
                populateFieldSolutionDataFast(
                    shared_ctx, mesh, cell_id, union_field_reqs, &cell_coefficient_cache,
                    &cell_field_eval_cache);
                tp_m_shared_field += TP() - tp0;
            }

            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                const auto& bs = monolithic_kernel->blockSpec(bi);
                auto& workspace = block_workspaces[bi];
                const bool use_expansion_this_block =
                    use_coupled_scalar_cache && bi > 0;

                tp0 = TP();
                workspace.ctx.copyGeometryDataFrom(shared_ctx);
                if (use_expansion_this_block) {
                    const auto& meta = cached_coupled_block_meta_[bi];
                    workspace.ctx.configureForCoupledBlock(
                        cell_id, mesh.getCellDomainId(cell_id), meta);

                    const auto n_test = meta.n_test_dofs;
                    const auto n_trial = meta.n_trial_dofs;
                    const auto nq = coupled_scalar_n_qpts_;
                    const auto ns = coupled_scalar_n_dofs_;
                    const bool same_sp = meta.trial_is_test;
                    const bool need_hess =
                        hasFlag(meta.required_data, RequiredData::BasisHessians);
                    const auto& slotc = coupled_slot_phys_cache_[0];

                    const auto test_count = static_cast<std::size_t>(n_test * nq);
                    auto* tg = workspace.ctx.testPhysGradientsWritePtr(test_count);
                    for (LocalIndex q = 0; q < nq; ++q) {
                        for (LocalIndex i = 0; i < n_test; ++i) {
                            const auto si =
                                static_cast<LocalIndex>(i % static_cast<LocalIndex>(ns));
                            tg[static_cast<std::size_t>(q * n_test + i)] =
                                slotc.phys_grads[q * ns + si];
                        }
                    }

                    if (need_hess) {
                        auto* th = workspace.ctx.testPhysHessiansWritePtr(test_count);
                        for (LocalIndex q = 0; q < nq; ++q) {
                            for (LocalIndex i = 0; i < n_test; ++i) {
                                const auto si =
                                    static_cast<LocalIndex>(i % static_cast<LocalIndex>(ns));
                                th[static_cast<std::size_t>(q * n_test + i)] =
                                    slotc.phys_hess[q * ns + si];
                            }
                        }
                    }

                    if (!same_sp) {
                        const auto trial_count = static_cast<std::size_t>(n_trial * nq);
                        auto* trg = workspace.ctx.trialPhysGradientsWritePtr(trial_count);
                        for (LocalIndex q = 0; q < nq; ++q) {
                            for (LocalIndex j = 0; j < n_trial; ++j) {
                                const auto sj =
                                    static_cast<LocalIndex>(j % static_cast<LocalIndex>(ns));
                                trg[static_cast<std::size_t>(q * n_trial + j)] =
                                    slotc.phys_grads[q * ns + sj];
                            }
                        }
                        if (need_hess) {
                            auto* trh = workspace.ctx.trialPhysHessiansWritePtr(trial_count);
                            for (LocalIndex q = 0; q < nq; ++q) {
                                for (LocalIndex j = 0; j < n_trial; ++j) {
                                    const auto sj =
                                        static_cast<LocalIndex>(j % static_cast<LocalIndex>(ns));
                                    trh[static_cast<std::size_t>(q * n_trial + j)] =
                                        slotc.phys_hess[q * ns + sj];
                                }
                            }
                        }
                    }

                    if (const auto* tc = findCoupledQptCache(n_test)) {
                        workspace.ctx.setTestBasisValuesOnlyQptMajor(n_test, *tc);
                    }
                    if (!same_sp) {
                        if (const auto* trc = findCoupledQptCache(n_trial)) {
                            workspace.ctx.setTrialBasisValuesOnlyQptMajor(n_trial, *trc);
                        }
                    }

                    if (hasFlag(meta.required_data, RequiredData::EntityMeasures)) {
                        workspace.ctx.setEntityMeasures(cached_geom_h_, cached_geom_volume_, 0.0);
                    }
                } else {
                    const auto* saved_coupled_meta = active_coupled_block_meta_;
                    active_coupled_block_meta_ = &cached_coupled_block_meta_[bi];
                    try {
                        prepareBasis(workspace.ctx, mesh, cell_id, *bs.test_space, *bs.trial_space,
                                     bs.fallback_kernel ? bs.fallback_kernel->getRequiredData()
                                                        : RequiredData::Standard,
                                     *fused_quad_rule);
                    } catch (...) {
                        active_coupled_block_meta_ = saved_coupled_meta;
                        throw;
                    }
                    active_coupled_block_meta_ = saved_coupled_meta;

                    if (use_coupled_scalar_cache && bi == 0) {
                        const auto nq = coupled_scalar_n_qpts_;
                        const auto ns = coupled_scalar_n_dofs_;
                        const auto n_test = static_cast<LocalIndex>(
                            bs.test_space->dofs_per_element());
                        auto& slotc = coupled_slot_phys_cache_[0];
                        const auto tg_raw = workspace.ctx.testPhysicalGradientsRaw();
                        for (LocalIndex q = 0; q < nq; ++q) {
                            for (LocalIndex si = 0; si < ns; ++si) {
                                slotc.phys_grads[q * ns + si] =
                                    tg_raw[static_cast<std::size_t>(q * n_test + si)];
                            }
                        }

                        if (coupled_scalar_has_hessians_) {
                            const auto th_raw = workspace.ctx.testPhysicalHessiansRaw();
                            for (LocalIndex q = 0; q < nq; ++q) {
                                for (LocalIndex si = 0; si < ns; ++si) {
                                    slotc.phys_hess[q * ns + si] =
                                        th_raw[static_cast<std::size_t>(q * n_test + si)];
                                }
                            }
                        }
                    }
                }
                tp_m_basis += TP() - tp0;
                setCommonContextState(workspace.ctx);
                tp0 = TP();
                applyBlockFieldCopy(
                    workspace.ctx,
                    shared_ctx,
                    bi,
                    cell_id,
                    &cell_coefficient_cache,
                    &cell_field_eval_cache);
                tp_m_block_field += TP() - tp0;

                const auto rg_index = static_cast<std::size_t>(row_group_of[bi]);
                auto& row_cache = cell_row_group_cache[rg_index];
                const auto tg_index = static_cast<std::size_t>(trial_group_of[bi]);
                auto& trial_cache = cell_trial_group_cache[tg_index];

                tp0 = TP();
                if (!row_cache.have_dofs) {
                    row_cache.dofs = getCellDofsCached(
                        mesh, cell_id, bs.row_dof_map, bs.row_dof_offset);
                    row_cache.have_dofs = true;
                }
                workspace.row_dofs = row_cache.dofs;
                if (!trial_cache.have_dofs) {
                    trial_cache.dofs = getCellDofsCached(
                        mesh, cell_id, bs.col_dof_map, bs.col_dof_offset);
                    trial_cache.have_dofs = true;
                }
                workspace.col_dofs = trial_cache.dofs;
                tp_m_dof += TP() - tp0;

                if (bs.fallback_kernel &&
                    (hasFlag(bs.fallback_kernel->getRequiredData(), RequiredData::SolutionCoefficients) ||
                     hasFlag(bs.fallback_kernel->getRequiredData(), RequiredData::SolutionValues) ||
                     hasFlag(bs.fallback_kernel->getRequiredData(), RequiredData::SolutionGradients) ||
                     hasFlag(bs.fallback_kernel->getRequiredData(), RequiredData::SolutionHessians) ||
                     hasFlag(bs.fallback_kernel->getRequiredData(), RequiredData::SolutionLaplacians))) {
                    tp0 = TP();
                    FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
                                "assembleCellsFused: kernel requires solution but no solution was set");
                    if (!trial_cache.gathered) {
                        trial_cache.solution_coeffs =
                            gatherCachedCellVectorCoefficients(
                                cell_coefficient_cache,
                                mesh,
                                cell_id,
                                bs.col_dof_map,
                                bs.col_dof_offset,
                                bs.trial_space,
                                trial_cache.dofs,
                                /*history_index=*/0,
                                workspace.ctx.trialUsesVectorBasis(),
                                "assembleCellsFused");

                        const int required_history = monolithic_required_history;
                        if (required_history > 0) {
                            trial_cache.previous_solution_coeffs.resize(
                                static_cast<std::size_t>(required_history));
                            for (int k = 1; k <= required_history; ++k) {
                                trial_cache.previous_solution_coeffs[
                                    static_cast<std::size_t>(k - 1)] =
                                    gatherCachedCellVectorCoefficients(
                                        cell_coefficient_cache,
                                        mesh,
                                        cell_id,
                                        bs.col_dof_map,
                                        bs.col_dof_offset,
                                        bs.trial_space,
                                        trial_cache.dofs,
                                        k,
                                        workspace.ctx.trialUsesVectorBasis(),
                                        "assembleCellsFused");
                            }
                        }
                        trial_cache.gathered = true;
                    }

                    if (monolithic_block_use_coeffs_only[bi]) {
                        workspace.ctx.setSolutionCoefficientsOnly(trial_cache.solution_coeffs);
                    } else {
                        workspace.ctx.setSolutionCoefficients(trial_cache.solution_coeffs);
                    }

                    const int required_history = monolithic_required_history;
                    if (required_history > 0) {
                        FE_THROW_IF(previous_solutions_.size() < static_cast<std::size_t>(required_history), FEException,
                                    "assembleCellsFused: time integration requires " +
                                        std::to_string(required_history) + " history states");
                        for (int k = 1; k <= required_history; ++k) {
                            if (monolithic_block_use_coeffs_only[bi]) {
                                workspace.ctx.setPreviousSolutionCoefficientsOnlyK(
                                    k,
                                    trial_cache.previous_solution_coeffs[
                                        static_cast<std::size_t>(k - 1)]);
                            } else {
                                workspace.ctx.setPreviousSolutionCoefficientsK(
                                    k,
                                    trial_cache.previous_solution_coeffs[
                                        static_cast<std::size_t>(k - 1)]);
                            }
                        }
                    }
                    tp_m_sol += TP() - tp0;
                }

                workspace.output.n_test_dofs = static_cast<LocalIndex>(workspace.row_dofs.size());
                workspace.output.n_trial_dofs = static_cast<LocalIndex>(workspace.col_dofs.size());
                workspace.output.local_matrix.clear();
                workspace.output.local_vector.clear();
                workspace.output.has_matrix = false;
                workspace.output.has_vector = false;
                const bool compute_matrix = bs.want_matrix && parent_term.assemble_matrix;
                const bool compute_vector = bs.want_vector && parent_term.assemble_vector;
                if (compute_matrix) {
                    workspace.output.local_matrix.assign(
                        static_cast<std::size_t>(workspace.row_dofs.size() * workspace.col_dofs.size()), Real(0));
                    workspace.output.has_matrix = true;
                }
                if (compute_vector) {
                    workspace.output.local_vector.assign(
                        static_cast<std::size_t>(workspace.row_dofs.size()), Real(0));
                    workspace.output.has_vector = true;
                }

                const auto test_basis = workspace.ctx.testBasisValuesRaw();
                const auto trial_basis = workspace.ctx.trialBasisValuesRaw();
                const auto test_grads = workspace.ctx.testPhysicalGradientsRaw();
                const auto trial_grads = workspace.ctx.trialPhysicalGradientsRaw();
                const auto test_hess = workspace.ctx.testPhysicalHessiansRaw();
                const auto trial_hess = workspace.ctx.trialPhysicalHessiansRaw();

                auto flatten_vec3 = [](std::span<const AssemblyContext::Vector3D> data) -> const Real* {
                    return data.empty() ? nullptr : &data.front()[0];
                };
                auto flatten_mat3 = [](std::span<const AssemblyContext::Matrix3x3> data) -> const Real* {
                    return data.empty() ? nullptr : &data.front()[0][0];
                };

                block_views[bi] = assembly::jit::CoupledBlockView{
                    .test_basis_values = test_basis.empty() ? nullptr : test_basis.data(),
                    .test_phys_gradients_xyz = flatten_vec3(test_grads),
                    .trial_basis_values = trial_basis.empty() ? nullptr : trial_basis.data(),
                    .trial_phys_gradients_xyz = flatten_vec3(trial_grads),
                    .test_phys_hessians = flatten_mat3(test_hess),
                    .trial_phys_hessians = flatten_mat3(trial_hess),
                    .n_test_dofs = static_cast<std::uint32_t>(workspace.row_dofs.size()),
                    .n_trial_dofs = static_cast<std::uint32_t>(workspace.col_dofs.size()),
                    .test_value_dim = static_cast<std::uint32_t>(std::max(1, bs.test_space->value_dimension())),
                    .trial_value_dim = static_cast<std::uint32_t>(std::max(1, bs.trial_space->value_dimension())),
                    .test_uses_vector_basis = workspace.ctx.testUsesVectorBasis() ? 1u : 0u,
                    .trial_uses_vector_basis = workspace.ctx.trialUsesVectorBasis() ? 1u : 0u,
                    .solution_coefficients = workspace.ctx.solutionCoefficients().empty()
                        ? nullptr
                        : workspace.ctx.solutionCoefficients().data(),
                    .num_previous_solutions =
                        static_cast<std::uint32_t>(std::min<std::size_t>(
                            workspace.ctx.previousSolutionHistoryCount(),
                            assembly::jit::kMaxPreviousSolutionsV6)),
                    .previous_solution_coefficients = {},
                    .element_matrix = workspace.output.local_matrix.empty() ? nullptr : workspace.output.local_matrix.data(),
                    .element_vector = (compiled_matrix_only_dispatch || workspace.output.local_vector.empty())
                        ? nullptr
                        : workspace.output.local_vector.data(),
                };

                for (std::size_t k = 0; k < static_cast<std::size_t>(block_views[bi].num_previous_solutions); ++k) {
                    const auto coeffs = workspace.ctx.previousSolutionCoefficientsRaw(static_cast<int>(k + 1));
                    block_views[bi].previous_solution_coefficients[k] =
                        coeffs.empty() ? nullptr : coeffs.data();
                }
            }

            const auto compute_exact_block_output =
                [&](const forms::MonolithicCellKernel::BlockSpec& bs,
                    BlockWorkspace& workspace,
                    bool want_matrix,
                    bool want_vector) {
                    assembly::KernelOutput exact_output;
                    if (!bs.fallback_kernel || (!want_matrix && !want_vector)) {
                        return exact_output;
                    }

                    exact_output.n_test_dofs = workspace.output.n_test_dofs;
                    exact_output.n_trial_dofs = workspace.output.n_trial_dofs;
                    exact_output.has_matrix = want_matrix;
                    exact_output.has_vector = want_vector;
                    if (exact_output.has_matrix) {
                        exact_output.local_matrix.assign(workspace.output.local_matrix.size(), Real(0));
                    }
                    if (exact_output.has_vector) {
                        exact_output.local_vector.assign(workspace.output.local_vector.size(), Real(0));
                    }

                    try {
                        bs.fallback_kernel->computeCell(workspace.ctx, exact_output);
                    } catch (...) {
                        throw;
                    }
                    return exact_output;
                };

            tp0 = TP();
            if (run_compiled_dispatch) {
                element_args[0] = assembly::jit::packCoupledCellKernelArgsV1(
                    shared_ctx,
                    std::span<const assembly::jit::CoupledBlockView>(block_views));
                assembly::jit::CoupledCellKernelBatchArgsV1 batch_args;
                batch_args.abi_version = assembly::jit::kCoupledCellKernelABIV1;
                batch_args.batch_size = 1u;
                batch_args.num_blocks = static_cast<std::uint32_t>(n_blocks);
                batch_args.elements = element_args.data();
                compiled_fn(reinterpret_cast<void*>(&batch_args));
            } else {
                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                    const auto& bs = monolithic_kernel->blockSpec(bi);
                    auto& workspace = block_workspaces[bi];
                    if (!bs.fallback_kernel || (!workspace.output.has_matrix && !workspace.output.has_vector)) {
                        continue;
                    }
                    try {
                        bs.fallback_kernel->computeCell(workspace.ctx, workspace.output);
                    } catch (const std::exception& e) {
                        if (core::kernelTraceEnabled(core::KernelTraceChannel::Assembly)) {
                            std::ostringstream oss;
                            oss << "StandardAssembler::assembleCellsFused: monolithic fallback block=" << bi
                                << " test_field=" << bs.test_field
                                << " trial_field=" << bs.trial_field
                                << " kernel='" << bs.fallback_kernel->name() << "'"
                                << " what=" << e.what();
                            core::kernelTraceLog(core::KernelTraceChannel::Assembly, oss.str());
                        }
                        throw;
                    } catch (...) {
                        if (core::kernelTraceEnabled(core::KernelTraceChannel::Assembly)) {
                            std::ostringstream oss;
                            oss << "StandardAssembler::assembleCellsFused: monolithic fallback block=" << bi
                                << " test_field=" << bs.test_field
                                << " trial_field=" << bs.trial_field
                                << " kernel='" << bs.fallback_kernel->name() << "'"
                                << " threw unknown exception";
                            core::kernelTraceLog(core::KernelTraceChannel::Assembly, oss.str());
                        }
                        throw;
                    }
                }
            }

            if (run_compiled_dispatch &&
                monolithicCompiledCompareEnabled() &&
                compared_monolithic_cells < monolithicCompiledCompareMaxCells()) {
                ++compared_monolithic_cells;
                const Real tol = monolithicCompiledCompareTolerance();
                const bool compare_vector = !compiled_matrix_only_dispatch;

                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                    const auto& bs = monolithic_kernel->blockSpec(bi);
                    auto& workspace = block_workspaces[bi];
                    if (!bs.fallback_kernel ||
                        (workspace.output.local_matrix.empty() && workspace.output.local_vector.empty())) {
                        continue;
                    }

                    assembly::KernelOutput exact_output;
                    try {
                        exact_output = compute_exact_block_output(
                            bs,
                            workspace,
                            !workspace.output.local_matrix.empty(),
                            compare_vector && !workspace.output.local_vector.empty());
                    } catch (const std::exception& e) {
                        std::ostringstream oss;
                        oss << "StandardAssembler::assembleCellsFused: compiled-vs-fallback compare failed"
                            << " during exact block evaluation"
                            << " cell=" << cell_id
                            << " block=" << bi
                            << " test_field=" << bs.test_field
                            << " trial_field=" << bs.trial_field
                            << " kernel='" << bs.fallback_kernel->name() << "'"
                            << " what=" << e.what();
                        throw FEException(oss.str(), __FILE__, __LINE__, __func__);
                    }

                    Real max_matrix_diff = 0.0;
                    std::size_t max_matrix_idx = 0;
                    for (std::size_t idx = 0; idx < workspace.output.local_matrix.size(); ++idx) {
                        const Real diff = std::abs(workspace.output.local_matrix[idx] - exact_output.local_matrix[idx]);
                        if (diff > max_matrix_diff) {
                            max_matrix_diff = diff;
                            max_matrix_idx = idx;
                        }
                    }

                    Real max_vector_diff = 0.0;
                    std::size_t max_vector_idx = 0;
                    if (compare_vector) {
                        for (std::size_t idx = 0; idx < workspace.output.local_vector.size(); ++idx) {
                            const Real diff = std::abs(workspace.output.local_vector[idx] - exact_output.local_vector[idx]);
                            if (diff > max_vector_diff) {
                                max_vector_diff = diff;
                                max_vector_idx = idx;
                            }
                        }
                    }

                    if (max_matrix_diff > tol || max_vector_diff > tol) {
                        std::ostringstream oss;
                        oss.setf(std::ios::scientific);
                        oss.precision(16);
                        oss << "StandardAssembler::assembleCellsFused: monolithic compiled dispatch mismatch"
                            << " cell=" << cell_id
                            << " block=" << bi
                            << " test_field=" << bs.test_field
                            << " trial_field=" << bs.trial_field
                            << " kernel='" << bs.fallback_kernel->name() << "'"
                            << " matrix_max_diff=" << max_matrix_diff;
                        if (!workspace.output.local_matrix.empty()) {
                            oss << " matrix_idx=" << max_matrix_idx
                                << " compiled=" << workspace.output.local_matrix[max_matrix_idx]
                                << " exact=" << exact_output.local_matrix[max_matrix_idx];
                        }
                        oss << " vector_max_diff=" << max_vector_diff;
                        if (compare_vector && !workspace.output.local_vector.empty()) {
                            oss << " vector_idx=" << max_vector_idx
                                << " compiled=" << workspace.output.local_vector[max_vector_idx]
                                << " exact=" << exact_output.local_vector[max_vector_idx];
                        }
                        throw FEException(oss.str(), __FILE__, __LINE__, __func__);
                    }
                }
            }

            if (run_compiled_dispatch && parent_term.assemble_vector) {
                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                    const auto& bs = monolithic_kernel->blockSpec(bi);
                    auto& workspace = block_workspaces[bi];
                    if (!bs.fallback_kernel || workspace.output.local_vector.empty()) {
                        continue;
                    }

                    try {
                        auto exact_output = compute_exact_block_output(
                            bs,
                            workspace,
                            /*want_matrix=*/false,
                            /*want_vector=*/true);
                        workspace.output.local_vector = std::move(exact_output.local_vector);
                    } catch (const std::exception& e) {
                        std::ostringstream oss;
                        oss << "StandardAssembler::assembleCellsFused: monolithic exact residual fallback failed"
                            << " cell=" << cell_id
                            << " block=" << bi
                            << " test_field=" << bs.test_field
                            << " trial_field=" << bs.trial_field
                            << " kernel='" << bs.fallback_kernel->name() << "'"
                            << " what=" << e.what();
                        throw FEException(oss.str(), __FILE__, __LINE__, __func__);
                    }
                }
            }
            tp_m_kernel += TP() - tp0;

            tp0 = TP();
            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                const auto& bs = monolithic_kernel->blockSpec(bi);
                auto& workspace = block_workspaces[bi];
                workspace.output.has_matrix =
                    parent_term.assemble_matrix && bs.want_matrix && !workspace.output.local_matrix.empty();
                workspace.output.has_vector =
                    parent_term.assemble_vector && bs.want_vector && !workspace.output.local_vector.empty();
                if (!workspace.output.has_matrix && !workspace.output.has_vector) {
                    continue;
                }
                if (workspace.ctx.testUsesVectorBasis() || workspace.ctx.trialUsesVectorBasis()) {
                    applyVectorBasisOutputOrientation(
                        mesh, cell_id, *bs.test_space, cell_id, *bs.trial_space, workspace.output);
                }

                const bool cell_constrained =
                    options_.use_constraints && constraint_distributor_ && constraints_ &&
                    (constraints_->hasConstrainedDofs(workspace.row_dofs) ||
                     constraints_->hasConstrainedDofs(workspace.col_dofs));

                const auto resolved_matrix =
                    (parent_term.assemble_matrix && parent_term.matrix_view &&
                     matrix_caps.resolved_matrix_entries && bs.want_matrix)
                        ? getResolvedCellMatrixEntries(cell_id, bs.row_dof_map, bs.row_dof_offset,
                                                       bs.col_dof_map, bs.col_dof_offset,
                                                       parent_term.matrix_view)
                        : std::span<const GlobalIndex>{};
                const auto resolved_vector =
                    (parent_term.assemble_vector && parent_term.vector_view &&
                     vector_caps.resolved_vector_entries && bs.want_vector)
                        ? getResolvedCellVectorEntries(cell_id, bs.row_dof_map, bs.row_dof_offset,
                                                       parent_term.vector_view)
                        : std::span<const GlobalIndex>{};

                auto& insert = block_inserts[bi];
                if (cell_constrained) {
                    insertLocalConstrained(workspace.output, workspace.row_dofs, workspace.col_dofs,
                                           insert.insert_matrix, insert.insert_vector);
                } else {
                    insertLocal(workspace.output, workspace.row_dofs, workspace.col_dofs,
                                insert.insert_matrix, insert.insert_vector,
                                resolved_matrix, resolved_vector);
                }

                if (workspace.output.has_matrix) {
                    result.matrix_entries_inserted +=
                        static_cast<GlobalIndex>(workspace.row_dofs.size() * workspace.col_dofs.size());
                }
                if (workspace.output.has_vector) {
                    result.vector_entries_inserted += static_cast<GlobalIndex>(workspace.row_dofs.size());
                }
            }
            tp_m_insert += TP() - tp0;

            result.elements_assembled += 1;
        }

        auto end_time = std::chrono::steady_clock::now();
        result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        if (assemblyTimingEnabled()) {
            const double total = tp_m_geom + tp_m_shared_field + tp_m_basis +
                                 tp_m_block_field + tp_m_dof + tp_m_sol +
                                 tp_m_kernel + tp_m_insert;
            if (total > 0.0) {
                std::fprintf(
                    stdout,
                    "    --- monolithic cellLoop TIMING (rank 0, %zu cells, %zu blocks%s) ---\n"
                    "      geometry:        %9.6f s  (%5.1f%%)\n"
                    "      shared fields:   %9.6f s  (%5.1f%%)\n"
                    "      prepareBasis:    %9.6f s  (%5.1f%%)\n"
                    "      block fields:    %9.6f s  (%5.1f%%)\n"
                    "      dof lookup:      %9.6f s  (%5.1f%%)\n"
                    "      solution gather: %9.6f s  (%5.1f%%)\n"
                    "      kernel:          %9.6f s  (%5.1f%%)\n"
                    "      insert:          %9.6f s  (%5.1f%%)\n"
                    "      TOTAL:           %9.6f s\n"
                    "    -----------------------------------------------\n",
                    static_cast<std::size_t>(cell_ids.size()),
                    n_blocks,
                    run_compiled_dispatch ? ", compiled" : "",
                    tp_m_geom,         100.0 * tp_m_geom / total,
                    tp_m_shared_field, 100.0 * tp_m_shared_field / total,
                    tp_m_basis,        100.0 * tp_m_basis / total,
                    tp_m_block_field,  100.0 * tp_m_block_field / total,
                    tp_m_dof,          100.0 * tp_m_dof / total,
                    tp_m_sol,          100.0 * tp_m_sol / total,
                    tp_m_kernel,       100.0 * tp_m_kernel / total,
                    tp_m_insert,       100.0 * tp_m_insert / total,
                    total);
            }
        }
        return result;
    }

    // ========================================================================
    // Check if fused+batched path is viable
    // ========================================================================
    std::size_t requested_batch_size =
        (options_.use_batching && options_.batch_size > 1)
            ? static_cast<std::size_t>(options_.batch_size) : 1u;
    bool all_support_batch = (requested_batch_size > 1);
    if (all_support_batch) {
        for (const auto& t : terms) {
            if (t.kernel->hasCell() && !t.kernel->supportsCellBatch()) {
                all_support_batch = false;
                break;
            }
        }
    }

    if (all_support_batch) {
    // ========================================================================
    // FUSED + BATCHED PATH: share geometry across terms, batch kernel calls
    // ========================================================================
    const std::size_t B = requested_batch_size;

    // Grow scratch batch storage on demand; reserve() is a no-op once sized.
    {
        const auto old_sz = scratch_batch_contexts_.size();
        if (old_sz < B) {
            scratch_batch_contexts_.resize(B);
            for (std::size_t i = old_sz; i < B; ++i)
                scratch_batch_contexts_[i].reserve(max_dofs, max_qpts, mesh.dimension());
        }
        if (max_dofs > scratch_batch_reserved_dofs_ ||
            max_qpts > scratch_batch_reserved_qpts_ ||
            mesh.dimension() != scratch_batch_reserved_dim_) {
            for (auto& bctx : scratch_batch_contexts_)
                bctx.reserve(max_dofs, max_qpts, mesh.dimension());
            scratch_batch_reserved_dofs_ = max_dofs;
            scratch_batch_reserved_qpts_ = max_qpts;
            scratch_batch_reserved_dim_ = mesh.dimension();
        }
    }
    scratch_batch_outputs_.resize(B);
    scratch_batch_context_ptrs_.assign(B, nullptr);
    scratch_saved_node_coords_.resize(B);
    if (scratch_batch_dofs_.size() < terms.size() ||
        (scratch_batch_dofs_.size() > 0 && scratch_batch_dofs_[0].size() < B)) {
        scratch_batch_dofs_.assign(terms.size(), std::vector<SlotDofs>(B));
    }
    scratch_batch_sol_coeffs_.resize(B);
    scratch_batch_prev_sol_coeffs_.resize(B);

    auto& batch_contexts = scratch_batch_contexts_;
    auto& batch_outputs = scratch_batch_outputs_;
    auto& batch_context_ptrs = scratch_batch_context_ptrs_;
    auto& saved_node_coords = scratch_saved_node_coords_;
    auto& batch_dofs = scratch_batch_dofs_;
    auto& batch_sol_coeffs = scratch_batch_sol_coeffs_;
    auto& batch_prev_sol_coeffs = scratch_batch_prev_sol_coeffs_;

    double tp_fb_geom = 0.0, tp_fb_save = 0.0, tp_fb_restore = 0.0;
    double tp_fb_basis = 0.0, tp_fb_field = 0.0, tp_fb_dof = 0.0;
    double tp_fb_sol = 0.0, tp_fb_setters = 0.0;
    double tp_fb_kernel = 0.0, tp_fb_insert = 0.0, tp_fb_snap = 0.0, tp_fb_scatter = 0.0;
    auto TP = assemblyTimeNow;

    // Helper: gather solution coefficients for a term/slot
    auto gather_solution = [&](std::size_t ti, std::size_t slot,
                               GlobalIndex cell_id, AssemblyContext& ctx,
                               std::span<const GlobalIndex> col_dofs) {
        const auto& t = terms[ti];
        const auto& td = term_data[ti];
        if (!td.need_solution) return;
        FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
                    "assembleCellsFused: kernel requires solution but no solution was set");
        auto& sol = batch_sol_coeffs[slot];
        gatherCellVectorCoefficients(cell_id, t.col_dof_map, t.col_dof_offset,
                                     col_dofs, current_solution_view_,
                                     current_solution_, sol,
                                     "assembleCellsFused", false);
        if (ctx.trialUsesVectorBasis()) {
            applyVectorBasisGlobalToLocal(mesh, cell_id, *t.trial_space, std::span<Real>(sol));
        }
        ctx.setSolutionCoefficients(sol);
        if (time_integration_ != nullptr) {
            const int required = requiredHistoryStates(time_integration_);
            if (required > 0) {
                FE_THROW_IF(previous_solutions_.size() < static_cast<std::size_t>(required), FEException,
                            "assembleCellsFused: time integration requires " + std::to_string(required) + " history states");
                auto& psc = batch_prev_sol_coeffs[slot];
                if (psc.size() < static_cast<std::size_t>(required))
                    psc.resize(static_cast<std::size_t>(required));
                for (int k = 1; k <= required; ++k) {
                    const auto& pdata = previous_solutions_[static_cast<std::size_t>(k - 1)];
                    const auto* pview = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
                                            ? previous_solution_views_[static_cast<std::size_t>(k - 1)] : nullptr;
                    FE_THROW_IF(pdata.empty() && pview == nullptr, FEException,
                                "assembleCellsFused: previous solution (k=" + std::to_string(k) + ") not set");
                    auto& lp = psc[static_cast<std::size_t>(k - 1)];
                    gatherCellVectorCoefficients(cell_id, t.col_dof_map, t.col_dof_offset,
                                                 col_dofs, pview, pdata, lp,
                                                 "assembleCellsFused", false);
                    if (ctx.trialUsesVectorBasis())
                        applyVectorBasisGlobalToLocal(mesh, cell_id, *t.trial_space, std::span<Real>(lp));
                    ctx.setPreviousSolutionCoefficientsK(k, lp);
                }
            }
        }
    };

    // Helper: insert batch outputs for a term
    auto insert_batch_outputs = [&](std::size_t ti, std::size_t active,
                                    std::span<const GlobalIndex> gids, std::size_t begin) {
        const auto& t = terms[ti];
        auto& ts = term_scratch[ti];
        for (std::size_t slot = 0; slot < active; ++slot) {
            const auto cid = gids[begin + slot];
            auto& ctx = batch_contexts[slot];
            auto& output = batch_outputs[slot];
            if (ctx.testUsesVectorBasis() || ctx.trialUsesVectorBasis())
                applyVectorBasisOutputOrientation(mesh, cid, *t.test_space, cid, *t.trial_space, output);
            const auto& rd = batch_dofs[ti][slot].row_dofs;
            const auto& cd = batch_dofs[ti][slot].col_dofs;
            insertLocalForCell(cid, t.row_dof_map, t.row_dof_offset,
                               t.col_dof_map, t.col_dof_offset,
                               output, rd, cd,
                               t.assemble_matrix ? ts.insert_matrix : nullptr,
                               t.assemble_vector ? ts.insert_vector : nullptr);
            result.elements_assembled++;
            if (output.has_matrix)
                result.matrix_entries_inserted += static_cast<GlobalIndex>(rd.size() * cd.size());
            if (output.has_vector)
                result.vector_entries_inserted += static_cast<GlobalIndex>(rd.size());
        }
    };

    // ========================================================================
    // Explicit mixed-block dispatch: one semantic kernel owns an exact set of
    // independent cell blocks, while optional colocation remains an internal
    // optimization of that semantic kernel.
    // ========================================================================
    const forms::MixedBlockKernelSet* mixed_block_kernel = nullptr;
    if (terms.size() == 1 &&
        terms[0].kernel->semanticKernelKind() == SemanticKernelKind::MixedBlockSet) {
        mixed_block_kernel = dynamic_cast<const forms::MixedBlockKernelSet*>(terms[0].kernel);
    }

    // Pre-build DOF tables for all mixed-block specs so the batch loop
    // can use getCellDofsCached as a read-only lookup (thread-safe).
    // Also re-resolve vector tables for the newly added DOF maps.
    if (mixed_block_kernel && mixed_block_kernel->isResolved()) {
        const auto n_blk = mixed_block_kernel->numBlocks();
        for (std::size_t bi = 0; bi < n_blk; ++bi) {
            const auto& bs = mixed_block_kernel->blockSpec(bi);
            if (bs.row_dof_map)
                (void)getCellDofTable(mesh, bs.row_dof_map, bs.row_dof_offset);
            if (bs.col_dof_map)
                (void)getCellDofTable(mesh, bs.col_dof_map, bs.col_dof_offset);
        }
        ensureResolvedVectorTables(mesh);
    }

    // Mixed-block batch range processor (prepare geometry once, then walk the
    // exact block set without redundant geometry save/restore).
    auto assemble_mixed_block_batch_range = [&](std::span<const GlobalIndex> gids) {
        if (!mixed_block_kernel || !mixed_block_kernel->isResolved()) return;
        const auto& parent_term = terms[0];
        const auto& parent_td = term_data[0];
        const std::size_t n_blocks = mixed_block_kernel->numBlocks();

        // Ensure batch_dofs has enough room for all blocks
        if (scratch_batch_dofs_.size() < n_blocks ||
            (scratch_batch_dofs_.size() > 0 && scratch_batch_dofs_[0].size() < B)) {
            scratch_batch_dofs_.assign(n_blocks, std::vector<SlotDofs>(B));
        }

        // Full batch size B for all paths.
        const std::size_t B_eff = B;

        // --- Compute trial space groups (once per assembly call) ---
        // trial_group_of[bi] = group index for block bi.
        // Blocks share a group iff col_dof_map ptr AND col_dof_offset match.
        // E.g., in a 2-field system: blocks with the same trial field share a group.
        struct TrialGroupSlotCache {
            std::span<const GlobalIndex> col_dofs{};
            std::vector<Real> sol_coeffs;
            std::vector<std::vector<Real>> prev_sol_coeffs;
            bool uses_vector_basis{false};
            bool gathered{false};
        };

        std::vector<int> trial_group_of(n_blocks, -1);
        int n_trial_groups = 0;
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            if (trial_group_of[bi] >= 0) continue;
            const auto& bs_i = mixed_block_kernel->blockSpec(bi);
            trial_group_of[bi] = n_trial_groups;
            for (std::size_t bj = bi + 1; bj < n_blocks; ++bj) {
                if (trial_group_of[bj] >= 0) continue;
                const auto& bs_j = mixed_block_kernel->blockSpec(bj);
                if (bs_i.col_dof_map == bs_j.col_dof_map &&
                    bs_i.col_dof_offset == bs_j.col_dof_offset) {
                    trial_group_of[bj] = n_trial_groups;
                }
            }
            ++n_trial_groups;
        }

        // Per-group, per-slot cache.  Indexed as [group * B_eff + slot].
        // Allocated once and reused across batches; gathered flags are reset per batch.
        const auto n_tg = static_cast<std::size_t>(n_trial_groups);
        std::vector<TrialGroupSlotCache> tg_cache(n_tg * B_eff);

        // --- Fused insertion setup (once per assembly call) ---
        // When all blocks are active and unconstrained, combine per-block
        // outputs into a single combined matrix/vector per cell.  This makes
        // DOFs form complete node runs (run == fsils_dof), hitting the FSILS
        // addMatrixEntries fast path and reducing blockBase hash lookups by ~4x.
        bool use_fused_insert = false;
        int fused_combined_n = 0;
        int fused_total_comps = 0;
        int fused_n_nodes = 0;
        std::vector<CombinedInsertBlockInfo> fused_info(n_blocks);

        if (parent_term.assemble_matrix && n_blocks >= 2) {
            // Check all blocks are active
            bool all_active = true;
            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                const auto& bs = mixed_block_kernel->blockSpec(bi);
                if (!bs.fallback_kernel || !bs.fallback_kernel->hasCell()) {
                    all_active = false; break;
                }
                if (!(parent_term.assemble_matrix && bs.want_matrix) &&
                    !(parent_term.assemble_vector && bs.want_vector)) {
                    all_active = false; break;
                }
            }

            if (all_active) {
                // Discover all unique (dof_map, dof_offset) pairs from both
                // row and col sides.  Assign contiguous component ranges.
                struct DofSideInfo {
                    const dofs::DofMap* dof_map;
                    GlobalIndex dof_offset;
                    int comps_per_node;
                    int comp_start;
                };
                std::vector<DofSideInfo> dof_sides;

                auto register_side = [&](const dofs::DofMap* dm, GlobalIndex off, int dim) {
                    for (const auto& s : dof_sides) {
                        if (s.dof_map == dm && s.dof_offset == off) return;
                    }
                    dof_sides.push_back({dm, off, dim, -1});
                };

                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                    const auto& bs = mixed_block_kernel->blockSpec(bi);
                    register_side(bs.row_dof_map, bs.row_dof_offset,
                                  bs.test_space->value_dimension());
                    register_side(bs.col_dof_map, bs.col_dof_offset,
                                  bs.trial_space->value_dimension());
                }

                int total_comps = 0;
                for (auto& s : dof_sides) {
                    s.comp_start = total_comps;
                    total_comps += s.comps_per_node;
                }
                fused_total_comps = total_comps;

                // Determine n_nodes from first side's sample DOFs
                if (!dof_sides.empty() && !gids.empty()) {
                    auto sample = getCellDofsCached(mesh, gids[0],
                                                    dof_sides[0].dof_map, dof_sides[0].dof_offset);
                    if (dof_sides[0].comps_per_node > 0 &&
                        static_cast<int>(sample.size()) % dof_sides[0].comps_per_node == 0) {
                        fused_n_nodes = static_cast<int>(sample.size()) / dof_sides[0].comps_per_node;
                        fused_combined_n = fused_n_nodes * fused_total_comps;
                    }
                }

                if (fused_combined_n > 0 && fused_total_comps > 0) {
                    // Build per-block scatter info
                    auto find_side = [&](const dofs::DofMap* dm, GlobalIndex off) -> const DofSideInfo& {
                        for (const auto& s : dof_sides) {
                            if (s.dof_map == dm && s.dof_offset == off) return s;
                        }
                        return dof_sides[0]; // should never reach
                    };

                    for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                        const auto& bs = mixed_block_kernel->blockSpec(bi);
                        auto& fi = fused_info[bi];
                        const auto& rs = find_side(bs.row_dof_map, bs.row_dof_offset);
                        const auto& cs = find_side(bs.col_dof_map, bs.col_dof_offset);
                        fi.row_comp_start = rs.comp_start;
                        fi.col_comp_start = cs.comp_start;
                        fi.row_comps = rs.comps_per_node;
                        fi.col_comps = cs.comps_per_node;
                    }

                    // Allocate scratch
                    resizeCombinedInsertScratch(B, fused_combined_n);
                    use_fused_insert = true;
                }
            }
        }

        // Check if all block kernels support coefficients-only mode (JIT compiled).
        // When true, skip the expensive QP solution value/gradient computation in
        // setSolutionCoefficients — the JIT kernel reads raw coefficients directly.
        bool use_coeffs_only = false;
        if (mixed_block_kernel) {
            use_coeffs_only = true;
            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                auto* jit = dynamic_cast<forms::jit::JITKernelWrapper*>(
                    mixed_block_kernel->blockSpec(bi).fallback_kernel.get());
                if (!jit) {
                    use_coeffs_only = false;
                    break;
                }
                jit->ensureCompiled();
                if (!jit->isJITReady()) {
                    use_coeffs_only = false;
                    break;
                }
            }
        }

        // Pre-compute mixed-block metadata to avoid virtual calls in fast path.
        cached_coupled_block_meta_.resize(n_blocks);
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            const auto& bs = mixed_block_kernel->blockSpec(bi);
            cached_coupled_block_meta_[bi] =
                AssemblyContext::makeCoupledBlockMetadata(
                    *bs.test_space, *bs.trial_space,
                    bs.fallback_kernel->getRequiredData());
        }

        // Pre-resolve CSR matrix insertion slots for all mixed blocks.
        // The resolved slot tables persist across Newton iterations (same mesh
        // topology → same DOF numbering → same CSR layout). On subsequent
        // assembly calls, insertions become flat scatter (no hash probes).
        bool use_resolved_insert = false;
        if (parent_term.assemble_matrix && parent_term.matrix_view &&
            parent_term.matrix_view->insertionCapabilities().resolved_matrix_entries) {
            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                const auto& bs = mixed_block_kernel->blockSpec(bi);
                if (bs.want_matrix) {
                    ensureResolvedMatrixTable(mesh, bs.row_dof_map, bs.row_dof_offset,
                                              bs.col_dof_map, bs.col_dof_offset,
                                              parent_term.matrix_view);
                }
            }
            use_resolved_insert = true;
        }

        // Pre-resolve vector insertion slots for the output vector view.
        // This is critical for thread-safe parallel assembly — bypasses the
        // resolution cache in FsilsVector::resolveEntriesCached which is not
        // thread-safe for concurrent access.
        bool use_resolved_vector_insert = false;
        if (parent_term.assemble_vector && parent_term.vector_view &&
            parent_term.vector_view->insertionCapabilities().resolved_vector_entries) {
            for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                const auto& bs = mixed_block_kernel->blockSpec(bi);
                if (bs.want_vector) {
                    ensureResolvedVectorTable(mesh, bs.row_dof_map, bs.row_dof_offset,
                                              parent_term.vector_view);
                }
            }
            use_resolved_vector_insert = true;
        }

        // Build fused-insert resolved slot table.
        // The fused-insert path combines all per-block DOFs into an interleaved
        // combined DOF list per cell, then inserts one combined matrix.
        // Pre-resolving the CSR slots for the combined DOF list eliminates
        // hash probes in the hot insertion loop.
        if (use_fused_insert && parent_term.assemble_matrix && parent_term.matrix_view &&
            parent_term.matrix_view->insertionCapabilities().contiguous_combined_matrix_insert &&
            parent_term.matrix_view->insertionCapabilities().resolved_matrix_entries) {
            const auto cn = static_cast<std::size_t>(fused_combined_n);
            const auto n_cells = mesh.numCells();
            if (scratch_fused_resolved_.empty() && cn > 0 && n_cells > 0) {
                // Build combined DOF list for each cell, then resolve.
                scratch_fused_resolved_offsets_.resize(static_cast<std::size_t>(n_cells) + 1u);
                const std::size_t entries_per_cell = cn * cn;
                const std::size_t total_entries = static_cast<std::size_t>(n_cells) * entries_per_cell;
                scratch_fused_resolved_.resize(total_entries);
                scratch_fused_resolved_offsets_[0] = 0;

                std::vector<GlobalIndex> combined_dofs(cn);
                for (GlobalIndex cell_id = 0; cell_id < n_cells; ++cell_id) {
                    // Build combined DOF list for this cell
                    for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                        const auto& bs = mixed_block_kernel->blockSpec(bi);
                        const auto& fi = fused_info[bi];
                        const auto rd = getCellDofsCached(mesh, cell_id, bs.row_dof_map, bs.row_dof_offset);
                        const int n_br = static_cast<int>(rd.size());
                        const int tc = fused_total_comps;
                        for (int i = 0; i < n_br; ++i) {
                            const int ci = (i / fi.row_comps) * tc +
                                           fi.row_comp_start + (i % fi.row_comps);
                            combined_dofs[static_cast<std::size_t>(ci)] = rd[static_cast<std::size_t>(i)];
                        }
                        const auto cd = getCellDofsCached(mesh, cell_id, bs.col_dof_map, bs.col_dof_offset);
                        const int n_bc = static_cast<int>(cd.size());
                        for (int j = 0; j < n_bc; ++j) {
                            const int cj = (j / fi.col_comps) * tc +
                                           fi.col_comp_start + (j % fi.col_comps);
                            combined_dofs[static_cast<std::size_t>(cj)] = cd[static_cast<std::size_t>(j)];
                        }
                    }

                    const std::size_t offset = static_cast<std::size_t>(cell_id) * entries_per_cell;
                    scratch_fused_resolved_offsets_[static_cast<std::size_t>(cell_id)] =
                        static_cast<GlobalIndex>(offset);
                    parent_term.matrix_view->resolveMatrixEntries(
                        std::span<const GlobalIndex>(combined_dofs),
                        std::span<const GlobalIndex>(combined_dofs),
                        std::span<GlobalIndex>(scratch_fused_resolved_.data() + offset, entries_per_cell));
                }
                scratch_fused_resolved_offsets_[static_cast<std::size_t>(n_cells)] =
                    static_cast<GlobalIndex>(total_entries);
            }
        }

        // ---------------------------------------------------------------
        // Coupled scalar basis cache: detect if all blocks share the same
        // scalar element basis (e.g. P1 Lagrange on Tet4).  When true,
        // blocks 1-3 can skip the full prepareBasis slow-path fallback
        // on block transitions by expanding cached scalar physical data.
        // ---------------------------------------------------------------
        bool use_coupled_scalar_cache = false;
        LocalIndex coupled_n_scalar = 0;
        int coupled_dim = mesh.dimension();
        if (n_blocks >= 2 && !gids.empty()) {
            use_coupled_scalar_cache = true;
            const auto first_ct = mesh.getCellType(gids[0]);
            const auto& first_bs = mixed_block_kernel->blockSpec(0);
            const auto& first_el = getElement(*first_bs.test_space, gids[0], first_ct);
            coupled_n_scalar = static_cast<LocalIndex>(first_el.num_dofs());
            const auto* first_basis_ptr = &first_el.basis();

            for (std::size_t bi = 0; bi < n_blocks && use_coupled_scalar_cache; ++bi) {
                const auto& bs = mixed_block_kernel->blockSpec(bi);
                const auto& te = getElement(*bs.test_space, gids[0], first_ct);
                const auto& tre = getElement(*bs.trial_space, gids[0], first_ct);
                if (te.basis().is_vector_valued() || tre.basis().is_vector_valued()) {
                    use_coupled_scalar_cache = false;
                } else if (&te.basis() != first_basis_ptr || &tre.basis() != first_basis_ptr) {
                    // Different scalar basis functions — can't share cache.
                    // But for equal-order elements (P1/P1), they will match.
                    if (static_cast<LocalIndex>(te.num_dofs()) != coupled_n_scalar ||
                        static_cast<LocalIndex>(tre.num_dofs()) != coupled_n_scalar) {
                        use_coupled_scalar_cache = false;
                    }
                }
            }

            if (use_coupled_scalar_cache) {
                // Allocate per-slot phys cache
                if (coupled_slot_phys_cache_.size() < B)
                    coupled_slot_phys_cache_.resize(B);

                // Populate scalar ref data if not already cached
                const auto n_qpts = static_cast<LocalIndex>(fused_quad_rule->num_points());
                const auto ns = static_cast<std::size_t>(coupled_n_scalar);
                const auto nq = static_cast<std::size_t>(n_qpts);
                const bool need_hess = std::any_of(
                    cached_coupled_block_meta_.begin(), cached_coupled_block_meta_.end(),
                    [](const auto& m) {
                        return hasFlag(m.required_data, RequiredData::BasisHessians);
                    });

                if (!coupled_scalar_ref_valid_ ||
                    coupled_scalar_n_dofs_ != coupled_n_scalar ||
                    coupled_scalar_n_qpts_ != n_qpts ||
                    (need_hess && !coupled_scalar_has_hessians_))
                {
                    // Populate from BasisCache
                    const auto& basis_fn = first_el.basis();
                    const auto& bcache = basis::BasisCache::instance().get_or_compute(
                        basis_fn, *fused_quad_rule, true, need_hess);

                    coupled_scalar_ref_grads_.resize(ns * nq);
                    coupled_scalar_ref_hess_.resize(need_hess ? ns * nq : 0);
                    coupled_scalar_basis_values_.resize(nq * ns);

                    for (std::size_t q = 0; q < nq; ++q) {
                        for (std::size_t si = 0; si < ns; ++si) {
                            const auto ref_idx = si * nq + q;
                            coupled_scalar_ref_grads_[ref_idx] = {
                                bcache.gradients[q][si][0],
                                bcache.gradients[q][si][1],
                                bcache.gradients[q][si][2]
                            };
                            if (need_hess) {
                                AssemblyContext::Matrix3x3 Hr{};
                                for (int r = 0; r < 3; ++r)
                                    for (int c = 0; c < 3; ++c)
                                        Hr[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                            bcache.hessians[q][si](static_cast<std::size_t>(r),
                                                                    static_cast<std::size_t>(c));
                                coupled_scalar_ref_hess_[ref_idx] = Hr;
                            }
                            coupled_scalar_basis_values_[q * ns + si] =
                                bcache.scalarValue(si, q);
                        }
                    }

                    // Build qpt-major basis value caches for each unique DOF count
                    // across all coupled blocks. Scalar basis values are replicated
                    // for ProductSpace fields (e.g., 3-component velocity = 3*scalar DOFs).
                    coupled_space_qpt_caches_.clear();
                    {
                        // Collect unique DOF counts from all coupled blocks.
                        std::vector<LocalIndex> unique_dof_counts;
                        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                            const auto& bs = mixed_block_kernel->blockSpec(bi);
                            const auto td = static_cast<LocalIndex>(bs.test_space->dofs_per_element());
                            bool found = false;
                            for (auto d : unique_dof_counts) { if (d == td) { found = true; break; } }
                            if (!found) unique_dof_counts.push_back(td);
                            if (bs.trial_space && bs.trial_space != bs.test_space) {
                                const auto trd = static_cast<LocalIndex>(bs.trial_space->dofs_per_element());
                                found = false;
                                for (auto d : unique_dof_counts) { if (d == trd) { found = true; break; } }
                                if (!found) unique_dof_counts.push_back(trd);
                            }
                        }

                        for (const auto n_dofs : unique_dof_counts) {
                            const auto nd = static_cast<std::size_t>(n_dofs);
                            CoupledSpaceQptCache cache;
                            cache.n_dofs = n_dofs;
                            cache.qpt_values.resize(nq * nd);
                            for (std::size_t q = 0; q < nq; ++q)
                                for (std::size_t i = 0; i < nd; ++i)
                                    cache.qpt_values[q * nd + i] =
                                        coupled_scalar_basis_values_[q * ns + (i % ns)];
                            coupled_space_qpt_caches_.push_back(std::move(cache));
                        }
                    }

                    coupled_scalar_n_dofs_ = coupled_n_scalar;
                    coupled_scalar_n_qpts_ = n_qpts;
                    coupled_scalar_has_hessians_ = need_hess;
                    coupled_scalar_ref_valid_ = true;
                }
            }
        }

        // =================================================================
        // COLORED PARALLEL ASSEMBLY PATH
        // =================================================================
        // When conditions are met, use graph coloring to partition elements
        // so that same-colored elements share no DOFs.  This makes CSR
        // insertion race-free within each color, enabling OMP parallelism
        // across cells within a color.
        //
        // Conditions:
        //  - coupled_kernel + use_coupled_scalar_cache (all blocks share basis)
        //  - affine elements (determined from first cell's geometry order)
        //  - OMP available with > 1 thread
        //  - enough cells to amortize coloring/OMP overhead
        //  - not using monolithic/pairwise JIT (those have own parallelism)
        //  - not using constraints (insertLocalConstrained is not thread-safe)
        // =================================================================
        const bool all_affine = !gids.empty() &&
            (defaultGeometryOrder(mesh.getCellType(gids[0])) <= 1);
        const bool use_batch_basis_early =
            use_coupled_scalar_cache && all_affine;

        int max_omp_threads = 1;
#ifdef _OPENMP
        max_omp_threads = omp_get_max_threads();
#endif
        const bool has_active_constraints =
            options_.use_constraints && constraint_distributor_ && constraints_;
        const bool use_colored_parallel =
            use_batch_basis_early &&
            max_omp_threads > 1 &&
            gids.size() >= 256 &&
            !has_active_constraints;

        if (use_colored_parallel) {
            // --- Build per-color cell lists from gids ---
            // Collect all block DOF maps for union-connectivity coloring.
            std::vector<const dofs::DofMap*> block_dof_maps;
            if (mixed_block_kernel) {
                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                    const auto& bs = mixed_block_kernel->blockSpec(bi);
                    if (bs.row_dof_map) block_dof_maps.push_back(bs.row_dof_map);
                    if (bs.col_dof_map) block_dof_maps.push_back(bs.col_dof_map);
                }
            }
            ensureColoring(mesh, block_dof_maps);

            // Diagnostic: can be enabled via SVMP_ASSEMBLY_TIMING=1
            if (assemblyTimingEnabled()) {
                static bool diag_printed = false;
                if (!diag_printed) {
                    std::fprintf(stderr, "[COLORED_PARALLEL] active: %d colors, %zu cells, %d threads\n",
                        coloring_num_colors_, gids.size(), max_omp_threads);
                    diag_printed = true;
                }
            }
            std::vector<std::vector<GlobalIndex>> color_cells(
                static_cast<std::size_t>(coloring_num_colors_));
            for (const auto cid : gids) {
                const int c = coloring_colors_[static_cast<std::size_t>(cid)];
                color_cells[static_cast<std::size_t>(c)].push_back(cid);
            }

            // Pre-populate field access plans + basis cache (serial, before parallel)
            if (any_need_field_solutions)
                ensureFieldAccessPlans(mesh);

            // Shared read-only state aliases
            const auto& ref_grads = coupled_scalar_ref_grads_;
            const auto& ref_hess = coupled_scalar_ref_hess_;
            const auto nq = coupled_scalar_n_qpts_;
            const auto ns = coupled_scalar_n_dofs_;
            const bool need_hess = coupled_scalar_has_hessians_;
            const int dim = mesh.dimension();

            const auto& parent_term = terms[0];
            const auto& parent_td = term_data[0];

            // Thread-safe result accumulators
            std::atomic<GlobalIndex> atomic_elements_assembled{0};
            std::atomic<GlobalIndex> atomic_matrix_entries{0};
            std::atomic<GlobalIndex> atomic_vector_entries{0};

            // Thread-safety invariants: resolved insertion tables MUST be
            // pre-built before entering the parallel region. Without them,
            // the fallback paths call FsilsVector::resolveEntriesCached()
            // and FsilsMatrix hash probes which have shared mutable state.
            FE_THROW_IF(parent_term.assemble_matrix && !use_resolved_insert, FEException,
                        "Colored parallel assembly requires pre-resolved matrix insertion tables");
            FE_THROW_IF(parent_term.assemble_vector && !use_resolved_vector_insert, FEException,
                        "Colored parallel assembly requires pre-resolved vector insertion tables");

            // Single persistent thread team for all colors.
            // The implicit barrier at end of each 'omp for' ensures
            // color-to-color synchronization without fork/join overhead.
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                // Per-thread state — allocated once, reused across colors
                AssemblyContext thread_ctx;
                thread_ctx.reserve(max_dofs, max_qpts, dim);
                GeometryWorkspace geom_ws;
                FieldSolutionWorkspace field_ws;
                CoupledSlotPhysCache slot_phys;
                KernelOutput thread_output;

                // Per-trial-group solution cache for this thread
                struct ThreadTrialCache {
                    std::span<const GlobalIndex> col_dofs{};
                    std::vector<Real> sol_coeffs;
                    std::vector<std::vector<Real>> prev_sol_coeffs;
                    bool uses_vector_basis{false};
                    bool gathered{false};
                };
                std::vector<ThreadTrialCache> thread_tg(
                    static_cast<std::size_t>(n_trial_groups));

                // Per-thread counters
                GlobalIndex local_elems = 0;
                GlobalIndex local_mat_entries = 0;
                GlobalIndex local_vec_entries = 0;

            for (std::size_t ci = 0; ci < color_cells.size(); ++ci) {
                const auto& cells_this_color = color_cells[ci];
                // NOTE: no 'continue' on empty — all threads must reach omp for

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (std::size_t ci_cell = 0;
                         ci_cell < cells_this_color.size(); ++ci_cell) {
                        const auto cell_id = cells_this_color[ci_cell];

                        // --- 1. Prepare geometry (thread-safe workspace version) ---
                        prepareGeometry(thread_ctx, mesh, cell_id,
                                        *fused_quad_rule, geom_ws);

                        // --- 2. Set context properties ---
                        thread_ctx.setMaterialState(nullptr, nullptr, 0u, 0u);
                        thread_ctx.setTimeIntegrationContext(time_integration_);
                        thread_ctx.setTime(time_);
                        thread_ctx.setTimeStep(dt_);
                        thread_ctx.setRealParameterGetter(get_real_param_);
                        thread_ctx.setParameterGetter(get_param_);
                        thread_ctx.setUserData(user_data_);
                        thread_ctx.setJITConstants(jit_constants_);
                        thread_ctx.setAuxiliaryValues(auxiliary_inputs_,
                                                    auxiliary_state_, auxiliary_outputs_);
                        thread_ctx.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
                        thread_ctx.clearAllPreviousSolutionData();

                        // --- 3. Field solutions (thread-safe workspace version) ---
                        if (any_need_field_solutions)
                            populateFieldSolutionData(thread_ctx, mesh, cell_id,
                                                      union_field_reqs, field_ws);

                        // --- 4. Gradient transform (ref → physical) ---
                        {
                            const auto ctx_inv_jacs = thread_ctx.inverseJacobians();
                            const auto& J_inv = ctx_inv_jacs[0]; // affine

                            for (LocalIndex si = 0; si < ns; ++si) {
                                for (LocalIndex q = 0; q < nq; ++q) {
                                    const auto ri = static_cast<std::size_t>(
                                        si * nq + q);
                                    const auto& gr = ref_grads[ri];
                                    auto& gp = slot_phys.phys_grads[q * ns + si];
                                    if (dim == 3) {
                                        gp[0] = J_inv[0][0]*gr[0] + J_inv[1][0]*gr[1] + J_inv[2][0]*gr[2];
                                        gp[1] = J_inv[0][1]*gr[0] + J_inv[1][1]*gr[1] + J_inv[2][1]*gr[2];
                                        gp[2] = J_inv[0][2]*gr[0] + J_inv[1][2]*gr[1] + J_inv[2][2]*gr[2];
                                    } else if (dim == 2) {
                                        gp[0] = J_inv[0][0]*gr[0] + J_inv[1][0]*gr[1];
                                        gp[1] = J_inv[0][1]*gr[0] + J_inv[1][1]*gr[1];
                                        gp[2] = 0.0;
                                    } else {
                                        gp[0] = J_inv[0][0]*gr[0];
                                        gp[1] = 0.0;
                                        gp[2] = 0.0;
                                    }
                                }
                            }

                            if (need_hess) {
                                for (LocalIndex si = 0; si < ns; ++si) {
                                    for (LocalIndex q = 0; q < nq; ++q) {
                                        const auto ri = static_cast<std::size_t>(
                                            si * nq + q);
                                        const auto& Hr = ref_hess[ri];
                                        auto& Hp = slot_phys.phys_hess[q * ns + si];
                                        for (int r = 0; r < dim; ++r)
                                            for (int c_d = 0; c_d < dim; ++c_d) {
                                                Real s = 0.0;
                                                for (int a = 0; a < dim; ++a)
                                                    for (int b = 0; b < dim; ++b)
                                                        s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                                             Hr[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                                             J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c_d)];
                                                Hp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c_d)] = s;
                                            }
                                    }
                                }
                            }
                        }

                        // --- 5. Per-block: expand, DOF, sol, kernel, insert ---
                        // Reset trial group gathered flags
                        for (auto& tg : thread_tg) tg.gathered = false;

                        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                            const auto& bs = mixed_block_kernel->blockSpec(bi);
                            if (!bs.fallback_kernel || !bs.fallback_kernel->hasCell())
                                continue;
                            const bool block_want_matrix =
                                parent_term.assemble_matrix && bs.want_matrix;
                            const bool block_want_vector =
                                parent_term.assemble_vector && bs.want_vector;
                            if (!block_want_matrix && !block_want_vector) continue;

                            const int tg_idx = trial_group_of[bi];

                            // Expand coupled scalar cache → context
                            const auto& meta = cached_coupled_block_meta_[bi];
                            thread_ctx.configureForCoupledBlock(
                                cell_id, mesh.getCellDomainId(cell_id), meta);

                            const auto n_test = meta.n_test_dofs;
                            const auto n_trial = meta.n_trial_dofs;
                            const bool same_sp = meta.trial_is_test;
                            const bool blk_need_hess = hasFlag(
                                meta.required_data, RequiredData::BasisHessians);

                            // Test gradients
                            const auto test_count =
                                static_cast<std::size_t>(n_test * nq);
                            auto* tg_ptr = thread_ctx.testPhysGradientsWritePtr(
                                test_count);
                            for (LocalIndex q = 0; q < nq; ++q)
                                for (LocalIndex i = 0; i < n_test; ++i) {
                                    const auto si = static_cast<LocalIndex>(
                                        i % static_cast<LocalIndex>(ns));
                                    tg_ptr[static_cast<std::size_t>(q * n_test + i)] =
                                        slot_phys.phys_grads[q * ns + si];
                                }

                            if (blk_need_hess) {
                                auto* th = thread_ctx.testPhysHessiansWritePtr(
                                    test_count);
                                for (LocalIndex q = 0; q < nq; ++q)
                                    for (LocalIndex i = 0; i < n_test; ++i) {
                                        const auto si = static_cast<LocalIndex>(
                                            i % static_cast<LocalIndex>(ns));
                                        th[static_cast<std::size_t>(q * n_test + i)] =
                                            slot_phys.phys_hess[q * ns + si];
                                    }
                            }

                            // Trial side
                            if (!same_sp) {
                                const auto trial_count =
                                    static_cast<std::size_t>(n_trial * nq);
                                auto* trg = thread_ctx.trialPhysGradientsWritePtr(
                                    trial_count);
                                for (LocalIndex q = 0; q < nq; ++q)
                                    for (LocalIndex j = 0; j < n_trial; ++j) {
                                        const auto sj = static_cast<LocalIndex>(
                                            j % static_cast<LocalIndex>(ns));
                                        trg[static_cast<std::size_t>(q * n_trial + j)] =
                                            slot_phys.phys_grads[q * ns + sj];
                                    }
                                if (blk_need_hess) {
                                    auto* trh =
                                        thread_ctx.trialPhysHessiansWritePtr(
                                            trial_count);
                                    for (LocalIndex q = 0; q < nq; ++q)
                                        for (LocalIndex j = 0; j < n_trial; ++j) {
                                            const auto sj = static_cast<LocalIndex>(
                                                j % static_cast<LocalIndex>(ns));
                                            trh[static_cast<std::size_t>(
                                                    q * n_trial + j)] =
                                                slot_phys.phys_hess[q * ns + sj];
                                        }
                                }
                            }

                            // Basis values from pre-built qpt-major cache (generic N-space lookup)
                            if (const auto* tc = findCoupledQptCache(n_test)) {
                                thread_ctx.setTestBasisValuesOnlyQptMajor(n_test, *tc);
                            }
                            if (!same_sp) {
                                if (const auto* trc = findCoupledQptCache(n_trial)) {
                                    thread_ctx.setTrialBasisValuesOnlyQptMajor(n_trial, *trc);
                                }
                            }

                            if (hasFlag(meta.required_data,
                                        RequiredData::EntityMeasures))
                                thread_ctx.setEntityMeasures(
                                    geom_ws.geom_h, geom_ws.geom_volume, 0.0);

                            // Row DOFs
                            auto row_dofs = getCellDofsCached(
                                mesh, cell_id, bs.row_dof_map, bs.row_dof_offset);

                            // Column DOFs + solution gather (trial group dedup)
                            auto& gc = thread_tg[static_cast<std::size_t>(tg_idx)];
                            if (!gc.gathered) {
                                gc.col_dofs = getCellDofsCached(
                                    mesh, cell_id, bs.col_dof_map,
                                    bs.col_dof_offset);
                                if (parent_td.need_solution) {
                                    gatherCellVectorCoefficients(
                                        cell_id, bs.col_dof_map,
                                        bs.col_dof_offset,
                                        gc.col_dofs,
                                        current_solution_view_,
                                        current_solution_,
                                        gc.sol_coeffs,
                                        "assembleCellsFused", false);
                                    gc.uses_vector_basis =
                                        thread_ctx.trialUsesVectorBasis();
                                    if (gc.uses_vector_basis)
                                        applyVectorBasisGlobalToLocal(
                                            mesh, cell_id, *bs.trial_space,
                                            std::span<Real>(gc.sol_coeffs));
                                    if (use_coeffs_only)
                                        thread_ctx.setSolutionCoefficientsOnly(
                                            std::span<const Real>(gc.sol_coeffs));
                                    else
                                        thread_ctx.setSolutionCoefficients(
                                            std::span<const Real>(gc.sol_coeffs));

                                    if (time_integration_ != nullptr) {
                                        const int req_hist =
                                            requiredHistoryStates(
                                                time_integration_);
                                        if (req_hist > 0) {
                                            gc.prev_sol_coeffs.resize(
                                                static_cast<std::size_t>(
                                                    req_hist));
                                            for (int k = 1; k <= req_hist; ++k) {
                                                const auto& pdata =
                                                    previous_solutions_[
                                                        static_cast<std::size_t>(
                                                            k - 1)];
                                                const auto* pview =
                                                    (static_cast<std::size_t>(
                                                         k - 1) <
                                                     previous_solution_views_
                                                         .size())
                                                        ? previous_solution_views_
                                                              [static_cast<
                                                                  std::size_t>(
                                                                  k - 1)]
                                                        : nullptr;
                                                auto& lp =
                                                    gc.prev_sol_coeffs
                                                        [static_cast<std::size_t>(
                                                            k - 1)];
                                                gatherCellVectorCoefficients(
                                                    cell_id, bs.col_dof_map,
                                                    bs.col_dof_offset,
                                                    gc.col_dofs,
                                                    pview, pdata, lp,
                                                    "assembleCellsFused", false);
                                                if (gc.uses_vector_basis)
                                                    applyVectorBasisGlobalToLocal(
                                                        mesh, cell_id,
                                                        *bs.trial_space,
                                                        std::span<Real>(lp));
                                                if (use_coeffs_only)
                                                    thread_ctx
                                                        .setPreviousSolutionCoefficientsOnlyK(
                                                            k, lp);
                                                else
                                                    thread_ctx
                                                        .setPreviousSolutionCoefficientsK(
                                                            k, lp);
                                            }
                                        }
                                    }
                                }
                                gc.gathered = true;
                            } else {
                                // Reuse cached trial-group data
                                if (parent_td.need_solution) {
                                    if (use_coeffs_only) {
                                        thread_ctx.setSolutionCoefficientsOnly(
                                            std::span<const Real>(
                                                gc.sol_coeffs));
                                        for (std::size_t k = 0;
                                             k < gc.prev_sol_coeffs.size();
                                             ++k)
                                            thread_ctx
                                                .setPreviousSolutionCoefficientsOnlyK(
                                                    static_cast<int>(k + 1),
                                                    gc.prev_sol_coeffs[k]);
                                    } else {
                                        thread_ctx.setSolutionCoefficients(
                                            std::span<const Real>(
                                                gc.sol_coeffs));
                                        for (std::size_t k = 0;
                                             k < gc.prev_sol_coeffs.size();
                                             ++k)
                                            thread_ctx
                                                .setPreviousSolutionCoefficientsK(
                                                    static_cast<int>(k + 1),
                                                    gc.prev_sol_coeffs[k]);
                                    }
                                }
                            }
                            auto col_dofs = gc.col_dofs;

                            // Kernel compute (batch of 1)
                            thread_output.clear();
                            const AssemblyContext* ctx_ptr = &thread_ctx;
                            bs.fallback_kernel->computeCellBatch(
                                std::span<const AssemblyContext* const>(
                                    &ctx_ptr, 1),
                                std::span<KernelOutput>(&thread_output, 1));

                            if (thread_ctx.testUsesVectorBasis() ||
                                thread_ctx.trialUsesVectorBasis())
                                applyVectorBasisOutputOrientation(
                                    mesh, cell_id, *bs.test_space,
                                    cell_id, *bs.trial_space, thread_output);

                            thread_output.has_matrix =
                                block_want_matrix &&
                                !thread_output.local_matrix.empty();
                            thread_output.has_vector =
                                block_want_vector &&
                                !thread_output.local_vector.empty();

                            // Insert (race-free for unconstrained: same-color
                            // cells share no DOFs; constrained cells use mutex)
                            GlobalSystemView* ins_mat =
                                block_want_matrix ? parent_term.matrix_view
                                                  : nullptr;
                            GlobalSystemView* ins_vec =
                                block_want_vector ? parent_term.vector_view
                                                  : nullptr;
                            const bool cell_constrained =
                                options_.use_constraints &&
                                constraint_distributor_ &&
                                constraints_ &&
                                (cell_constrained_flags_valid_ &&
                                 static_cast<std::size_t>(cell_id) < cell_constrained_flags_.size()
                                    ? (cell_constrained_flags_[static_cast<std::size_t>(cell_id)] != 0)
                                    : (constraints_->hasConstrainedDofs(row_dofs) ||
                                       constraints_->hasConstrainedDofs(col_dofs)));
                            // Look up pre-resolved vector entries for thread-safe insertion
                            const auto resolved_vec =
                                use_resolved_vector_insert
                                    ? getResolvedCellVectorEntries(
                                          cell_id, bs.row_dof_map,
                                          bs.row_dof_offset,
                                          parent_term.vector_view)
                                    : std::span<const GlobalIndex>{};

                            if (cell_constrained) {
                                // Constrained cells write to master DOFs that
                                // may overlap with other elements. Serialize.
                                #pragma omp critical(constrained_insert)
                                insertLocalConstrained(thread_output, row_dofs,
                                    col_dofs, ins_mat, ins_vec);
                            } else if (use_resolved_insert && block_want_matrix) {
                                const auto resolved =
                                    getResolvedCellMatrixEntries(
                                        cell_id, bs.row_dof_map,
                                        bs.row_dof_offset,
                                        bs.col_dof_map, bs.col_dof_offset,
                                        parent_term.matrix_view);
                                insertLocal(thread_output, row_dofs, col_dofs,
                                            ins_mat, ins_vec, resolved,
                                            resolved_vec);
                            } else {
                                insertLocal(thread_output, row_dofs, col_dofs,
                                            ins_mat, ins_vec, {},
                                            resolved_vec);
                            }

                            local_elems++;
                            if (thread_output.has_matrix)
                                local_mat_entries += static_cast<GlobalIndex>(
                                    row_dofs.size() * col_dofs.size());
                            if (thread_output.has_vector)
                                local_vec_entries += static_cast<GlobalIndex>(
                                    row_dofs.size());
                        } // end per-block loop
                    } // end per-cell loop
            } // end per-color loop

                // Accumulate per-thread counters
                atomic_elements_assembled.fetch_add(
                    local_elems, std::memory_order_relaxed);
                atomic_matrix_entries.fetch_add(
                    local_mat_entries, std::memory_order_relaxed);
                atomic_vector_entries.fetch_add(
                    local_vec_entries, std::memory_order_relaxed);
            } // end omp parallel

            result.elements_assembled += atomic_elements_assembled.load();
            result.matrix_entries_inserted += atomic_matrix_entries.load();
            result.vector_entries_inserted += atomic_vector_entries.load();

        } else {
        // =================================================================
        // SERIAL BATCH PATH (original)
        // =================================================================

        for (std::size_t begin = 0; begin < gids.size(); begin += B_eff) {
            const std::size_t active = std::min(B_eff, gids.size() - begin);

            // Reset trial group gathered flags for this batch
            for (std::size_t gi = 0; gi < n_tg * B_eff; ++gi)
                tg_cache[gi].gathered = false;

            // Zero-initialize fused combined matrices/vectors for this batch
            if (use_fused_insert) {
                zeroCombinedInsertScratch(active, fused_combined_n);
            }

            // === Prepare geometry + context for all cells in batch (ONCE) ===
            for (std::size_t slot = 0; slot < active; ++slot) {
                const auto cell_id = gids[begin + slot];
                auto& ctx = batch_contexts[slot];

                double tp0 = TP();
                prepareGeometry(ctx, mesh, cell_id, *fused_quad_rule);
                tp_fb_geom += TP() - tp0;

                tp0 = TP();
                saved_node_coords[slot].node_coords = scratch_node_coords_;
                saved_node_coords[slot].entity_h = cached_geom_h_;
                saved_node_coords[slot].entity_volume = cached_geom_volume_;
                tp_fb_save += TP() - tp0;

                tp0 = TP();
                ctx.setMaterialState(nullptr, nullptr, 0u, 0u);
                ctx.setTimeIntegrationContext(time_integration_);
                ctx.setTime(time_);
                ctx.setTimeStep(dt_);
                ctx.setRealParameterGetter(get_real_param_);
                ctx.setParameterGetter(get_param_);
                ctx.setUserData(user_data_);
                ctx.setJITConstants(jit_constants_);
                ctx.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
                ctx.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
                ctx.clearAllPreviousSolutionData();
                tp_fb_setters += TP() - tp0;
            }

            // === Batch field solutions + gradient transform ===
            // When coupled scalar cache is active on affine elements, we can
            // compute field solutions and scalar physical gradients/hessians
            // for all slots BEFORE the block loop, eliminating per-cell
            // prepareBasis calls entirely.
            const bool use_batch_basis =
                use_coupled_scalar_cache && cached_mapping_affine_;

            if (use_batch_basis) {
                // Phase A: field solutions (serial — uses member scratch)
                if (any_need_field_solutions) {
                    for (std::size_t slot = 0; slot < active; ++slot) {
                        const auto cell_id = gids[begin + slot];
                        auto& ctx = batch_contexts[slot];
                        double tp0 = TP();
                        populateFieldSolutionDataFast(ctx, mesh, cell_id, union_field_reqs);
                        tp_fb_field += TP() - tp0;
                    }
                }

                // Phase B: batch gradient transform — compute scalar physical
                // gradients/hessians from cached ref data + per-slot J_inv.
                // Replaces 75K per-cell prepareBasis calls with a tight loop.
                {
                    const auto nq = coupled_scalar_n_qpts_;
                    const auto ns = coupled_scalar_n_dofs_;
                    const bool need_hess = coupled_scalar_has_hessians_;
                    const int dim = mesh.dimension();

                    for (std::size_t slot = 0; slot < active; ++slot) {
                        auto& ctx = batch_contexts[slot];
                        const auto ctx_inv_jacs = ctx.inverseJacobians();
                        const auto& J_inv = ctx_inv_jacs[0]; // constant for affine

                        auto& slotc = coupled_slot_phys_cache_[slot];

                        // Transform ref gradients → physical gradients
                        // grad_phys = J_inv^T * grad_ref
                        for (LocalIndex si = 0; si < ns; ++si) {
                            for (LocalIndex q = 0; q < nq; ++q) {
                                const auto ref_idx = static_cast<std::size_t>(si * nq + q);
                                const auto& gr = coupled_scalar_ref_grads_[ref_idx];
                                auto& gp = slotc.phys_grads[q * ns + si];
                                if (dim == 3) {
                                    gp[0] = J_inv[0][0]*gr[0] + J_inv[1][0]*gr[1] + J_inv[2][0]*gr[2];
                                    gp[1] = J_inv[0][1]*gr[0] + J_inv[1][1]*gr[1] + J_inv[2][1]*gr[2];
                                    gp[2] = J_inv[0][2]*gr[0] + J_inv[1][2]*gr[1] + J_inv[2][2]*gr[2];
                                } else if (dim == 2) {
                                    gp[0] = J_inv[0][0]*gr[0] + J_inv[1][0]*gr[1];
                                    gp[1] = J_inv[0][1]*gr[0] + J_inv[1][1]*gr[1];
                                    gp[2] = 0.0;
                                } else {
                                    gp[0] = J_inv[0][0]*gr[0];
                                    gp[1] = 0.0;
                                    gp[2] = 0.0;
                                }
                            }
                        }

                        // Transform ref hessians → physical hessians (affine only)
                        // H_phys = J_inv^T * H_ref * J_inv
                        if (need_hess) {
                            for (LocalIndex si = 0; si < ns; ++si) {
                                for (LocalIndex q = 0; q < nq; ++q) {
                                    const auto ref_idx = static_cast<std::size_t>(si * nq + q);
                                    const auto& Hr = coupled_scalar_ref_hess_[ref_idx];
                                    auto& Hp = slotc.phys_hess[q * ns + si];
                                    for (int r = 0; r < dim; ++r)
                                        for (int c = 0; c < dim; ++c) {
                                            Real s = 0.0;
                                            for (int a = 0; a < dim; ++a)
                                                for (int b = 0; b < dim; ++b)
                                                    s += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                                         Hr[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                                         J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                            Hp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = s;
                                        }
                                }
                            }
                        }
                    }
                }
            }


                for (std::size_t bi = 0; bi < n_blocks; ++bi) {
                    const auto& bs = mixed_block_kernel->blockSpec(bi);
                    if (!bs.fallback_kernel || !bs.fallback_kernel->hasCell()) continue;
                    const bool block_want_matrix = parent_term.assemble_matrix && bs.want_matrix;
                    const bool block_want_vector = parent_term.assemble_vector && bs.want_vector;
                    if (!block_want_matrix && !block_want_vector) continue;

                    const int tg = trial_group_of[bi];

                    // Use pre-computed metadata for this block to skip virtual calls.
                    active_coupled_block_meta_ = &cached_coupled_block_meta_[bi];

                    // When use_batch_basis is active, ALL blocks use the expansion
                    // path (physical gradients were pre-computed in the batch
                    // gradient transform above).  Otherwise, only blocks 1+ use it.
                    const bool use_expansion_this_block =
                        use_batch_basis || (use_coupled_scalar_cache && bi > 0);

                    // When all conditions are met, the expansion slot loop can be
                    // parallelized: each slot writes only to independent per-slot
                    // data (ctx, batch_dofs, tg_cache, output).  Requires:
                    //  - Expansion path active (no prepareBasis needed)
                    //  - Affine elements (no mapping resetNodes needed)
                    //  - bi > 0 (block 0 needs serial populateFieldSolutionData
                    //    when !use_batch_basis; when use_batch_basis, field sols
                    //    are already done)
                    const bool omp_slot_parallel =
                        use_expansion_this_block && cached_mapping_affine_ &&
                        (use_batch_basis || bi > 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_slot_parallel && active >= 4)
#endif
                    for (std::size_t slot = 0; slot < active; ++slot) {
                        const auto cell_id = gids[begin + slot];
                        auto& ctx = batch_contexts[slot];

                        // Per-slot entity measures from saved geometry.
                        const double slot_h = saved_node_coords[slot].entity_h;
                        const double slot_vol = saved_node_coords[slot].entity_volume;

                        // tp0 used for all sub-phase timing; guarded for parallel safety.
                        double tp0 = 0;

                        if (!omp_slot_parallel) {
                            tp0 = TP();
                            // Restore cached entity measures (serial only — member vars not thread-safe)
                            cached_geom_h_ = slot_h;
                            cached_geom_volume_ = slot_vol;
                            if (!cached_mapping_affine_) {
                                scratch_node_coords_ = saved_node_coords[slot].node_coords;
                                cached_mapping_->resetNodes(scratch_node_coords_);
                            }
                            tp_fb_restore += TP() - tp0;
                        }

                        tp0 = TP();
                        if (use_expansion_this_block) {
                            // ===================================================
                            // COUPLED SCALAR FAST PATH: expand from per-slot
                            // cached scalar physical gradients/hessians.
                            // Avoids full prepareBasis (no slow-path fallback,
                            // no redundant gradient/hessian transforms).
                            // ===================================================
                            const auto& meta = cached_coupled_block_meta_[bi];
                            ctx.configureForCoupledBlock(
                                cell_id, mesh.getCellDomainId(cell_id), meta);

                            const auto n_test = meta.n_test_dofs;
                            const auto n_trial = meta.n_trial_dofs;
                            const auto nq = coupled_scalar_n_qpts_;
                            const auto ns = coupled_scalar_n_dofs_;
                            const bool same_sp = meta.trial_is_test;
                            const bool need_hess = hasFlag(
                                meta.required_data, RequiredData::BasisHessians);

                            const auto& slotc = coupled_slot_phys_cache_[slot];

                            // Test gradients: expand scalar phys → DOF layout
                            const auto test_count = static_cast<std::size_t>(n_test * nq);
                            auto* tg = ctx.testPhysGradientsWritePtr(test_count);
                            for (LocalIndex q = 0; q < nq; ++q)
                                for (LocalIndex i = 0; i < n_test; ++i) {
                                    const auto si = static_cast<LocalIndex>(
                                        i % static_cast<LocalIndex>(ns));
                                    tg[static_cast<std::size_t>(q * n_test + i)] =
                                        slotc.phys_grads[q * ns + si];
                                }

                            // Test hessians
                            if (need_hess) {
                                auto* th = ctx.testPhysHessiansWritePtr(test_count);
                                for (LocalIndex q = 0; q < nq; ++q)
                                    for (LocalIndex i = 0; i < n_test; ++i) {
                                        const auto si = static_cast<LocalIndex>(
                                            i % static_cast<LocalIndex>(ns));
                                        th[static_cast<std::size_t>(q * n_test + i)] =
                                            slotc.phys_hess[q * ns + si];
                                    }
                            }

                            // Trial side
                            if (!same_sp) {
                                const auto trial_count = static_cast<std::size_t>(n_trial * nq);
                                auto* trg = ctx.trialPhysGradientsWritePtr(trial_count);
                                for (LocalIndex q = 0; q < nq; ++q)
                                    for (LocalIndex j = 0; j < n_trial; ++j) {
                                        const auto sj = static_cast<LocalIndex>(
                                            j % static_cast<LocalIndex>(ns));
                                        trg[static_cast<std::size_t>(q * n_trial + j)] =
                                            slotc.phys_grads[q * ns + sj];
                                    }
                                if (need_hess) {
                                    auto* trh = ctx.trialPhysHessiansWritePtr(trial_count);
                                    for (LocalIndex q = 0; q < nq; ++q)
                                        for (LocalIndex j = 0; j < n_trial; ++j) {
                                            const auto sj = static_cast<LocalIndex>(
                                                j % static_cast<LocalIndex>(ns));
                                            trh[static_cast<std::size_t>(q * n_trial + j)] =
                                                slotc.phys_hess[q * ns + sj];
                                        }
                                }
                            }

                            // Basis values from pre-built qpt-major cache (generic N-space lookup)
                            if (const auto* tc = findCoupledQptCache(n_test)) {
                                ctx.setTestBasisValuesOnlyQptMajor(n_test, *tc);
                            }
                            if (!same_sp) {
                                if (const auto* trc = findCoupledQptCache(n_trial)) {
                                    ctx.setTrialBasisValuesOnlyQptMajor(n_trial, *trc);
                                }
                            }

                            if (hasFlag(meta.required_data, RequiredData::EntityMeasures))
                                ctx.setEntityMeasures(slot_h, slot_vol, 0.0);

                        } else {
                            prepareBasis(ctx, mesh, cell_id, *bs.test_space, *bs.trial_space,
                                         bs.fallback_kernel->getRequiredData(), *fused_quad_rule);

                            // After block 0's prepareBasis, cache scalar physical data
                            // so subsequent blocks can use the expansion path.
                            if (use_coupled_scalar_cache && bi == 0) {
                                const auto nq = coupled_scalar_n_qpts_;
                                const auto ns = coupled_scalar_n_dofs_;
                                const auto n_test = static_cast<LocalIndex>(
                                    bs.test_space->dofs_per_element());
                                const bool need_hess = hasFlag(
                                    bs.fallback_kernel->getRequiredData(),
                                    RequiredData::BasisHessians);

                                auto& slotc = coupled_slot_phys_cache_[slot];
                                const auto tg_raw = ctx.testPhysicalGradientsRaw();
                                for (LocalIndex q = 0; q < nq; ++q)
                                    for (LocalIndex si = 0; si < ns; ++si)
                                        slotc.phys_grads[q * ns + si] =
                                            tg_raw[static_cast<std::size_t>(q * n_test + si)];

                                if (need_hess) {
                                    const auto th_raw = ctx.testPhysicalHessiansRaw();
                                    for (LocalIndex q = 0; q < nq; ++q)
                                        for (LocalIndex si = 0; si < ns; ++si)
                                            slotc.phys_hess[q * ns + si] =
                                                th_raw[static_cast<std::size_t>(q * n_test + si)];
                                }

                            }
                        }
                        if (!omp_slot_parallel) tp_fb_basis += TP() - tp0;

                        if (bi == 0 && any_need_field_solutions && !use_batch_basis) {
                            tp0 = TP();
                            populateFieldSolutionData(ctx, mesh, cell_id, union_field_reqs);
                            tp_fb_field += TP() - tp0;
                        }

                        // Row DOFs are always per-block (different test spaces)
                        if (!omp_slot_parallel) tp0 = TP();
                        {
                            auto& rd = batch_dofs[bi][slot].row_dofs;
                            rd = getCellDofsCached(mesh, cell_id, bs.row_dof_map, bs.row_dof_offset);
                        }

                        // Column DOFs + solution gather: deduplicate by trial group
                        auto& gc = tg_cache[static_cast<std::size_t>(tg) * B_eff + slot];
                        if (!gc.gathered) {
                            // First block in this trial group for this slot: do full gather
                            gc.col_dofs = getCellDofsCached(mesh, cell_id, bs.col_dof_map, bs.col_dof_offset);
                            batch_dofs[bi][slot].col_dofs = gc.col_dofs;
                            if (!omp_slot_parallel) tp_fb_dof += TP() - tp0;

                            if (!omp_slot_parallel) tp0 = TP();
                            if (parent_td.need_solution) {
                                const auto& cd = gc.col_dofs;
                                gatherCellVectorCoefficients(cell_id, bs.col_dof_map,
                                                             bs.col_dof_offset,
                                                             std::span<const GlobalIndex>(cd),
                                                             current_solution_view_,
                                                             current_solution_,
                                                             gc.sol_coeffs,
                                                             "assembleCellsFused", false);
                                gc.uses_vector_basis = ctx.trialUsesVectorBasis();
                                if (gc.uses_vector_basis)
                                    applyVectorBasisGlobalToLocal(mesh, cell_id, *bs.trial_space,
                                                                  std::span<Real>(gc.sol_coeffs));
                                if (use_coeffs_only)
                                    ctx.setSolutionCoefficientsOnly(
                                        std::span<const Real>(gc.sol_coeffs));
                                else
                                    ctx.setSolutionCoefficients(
                                        std::span<const Real>(gc.sol_coeffs));

                                if (time_integration_ != nullptr) {
                                    const int required =
                                        requiredHistoryStates(time_integration_);
                                    if (required > 0) {
                                        gc.prev_sol_coeffs.resize(
                                            static_cast<std::size_t>(required));
                                        for (int k = 1; k <= required; ++k) {
                                            const auto& pdata =
                                                previous_solutions_[static_cast<std::size_t>(k - 1)];
                                            const auto* pview =
                                                (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
                                                    ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
                                                    : nullptr;
                                            auto& lp = gc.prev_sol_coeffs[
                                                static_cast<std::size_t>(k - 1)];
                                            gatherCellVectorCoefficients(cell_id, bs.col_dof_map,
                                                                         bs.col_dof_offset,
                                                                         std::span<const GlobalIndex>(cd),
                                                                         pview, pdata, lp,
                                                                         "assembleCellsFused", false);
                                            if (gc.uses_vector_basis)
                                                applyVectorBasisGlobalToLocal(
                                                    mesh, cell_id, *bs.trial_space,
                                                    std::span<Real>(lp));
                                            if (use_coeffs_only)
                                                ctx.setPreviousSolutionCoefficientsOnlyK(k, lp);
                                            else
                                                ctx.setPreviousSolutionCoefficientsK(k, lp);
                                        }
                                    }
                                }
                            }
                            gc.gathered = true;
                        } else {
                            // Reuse cached col_dofs and solution coefficients
                            batch_dofs[bi][slot].col_dofs = gc.col_dofs;
                            if (!omp_slot_parallel) tp_fb_dof += TP() - tp0;

                            if (!omp_slot_parallel) tp0 = TP();
                            if (parent_td.need_solution) {
                                if (use_coeffs_only) {
                                    ctx.setSolutionCoefficientsOnly(
                                        std::span<const Real>(gc.sol_coeffs));
                                    for (std::size_t k = 0; k < gc.prev_sol_coeffs.size(); ++k) {
                                        ctx.setPreviousSolutionCoefficientsOnlyK(
                                            static_cast<int>(k + 1), gc.prev_sol_coeffs[k]);
                                    }
                                } else {
                                    ctx.setSolutionCoefficients(
                                        std::span<const Real>(gc.sol_coeffs));
                                    for (std::size_t k = 0; k < gc.prev_sol_coeffs.size(); ++k) {
                                        ctx.setPreviousSolutionCoefficientsK(
                                            static_cast<int>(k + 1), gc.prev_sol_coeffs[k]);
                                    }
                                }
                            }
                        }
                        if (!omp_slot_parallel) tp_fb_sol += TP() - tp0;

                        batch_outputs[slot].clear();
                        batch_context_ptrs[slot] = &ctx;
                    }

                    active_coupled_block_meta_ = nullptr;

                    double tp0 = TP();
                    bs.fallback_kernel->computeCellBatch(
                        std::span<const AssemblyContext* const>(batch_context_ptrs.data(), active),
                        std::span<KernelOutput>(batch_outputs.data(), active));
                    tp_fb_kernel += TP() - tp0;

                    double tp0_scatter = TP();
                    for (std::size_t slot = 0; slot < active; ++slot) {
                        const auto cid = gids[begin + slot];
                        auto& ctx = batch_contexts[slot];
                        auto& output = batch_outputs[slot];
                        if (ctx.testUsesVectorBasis() || ctx.trialUsesVectorBasis())
                            applyVectorBasisOutputOrientation(mesh, cid, *bs.test_space,
                                                              cid, *bs.trial_space, output);

                        if (use_fused_insert) {
                            scatterCombinedInsertBlockOutput(
                                slot, output,
                                batch_dofs[bi][slot].row_dofs,
                                batch_dofs[bi][slot].col_dofs,
                                fused_info[bi], fused_total_comps, fused_combined_n,
                                block_want_matrix, block_want_vector);
                        } else {
                            // Per-block insertion (with optional pre-resolved CSR slots)
                            const auto& rd = batch_dofs[bi][slot].row_dofs;
                            const auto& cd = batch_dofs[bi][slot].col_dofs;
                            GlobalSystemView* ins_mat = block_want_matrix ? parent_term.matrix_view : nullptr;
                            GlobalSystemView* ins_vec = block_want_vector ? parent_term.vector_view : nullptr;
                            if (options_.use_constraints && constraint_distributor_) {
                                insertLocalConstrained(output, rd, cd, ins_mat, ins_vec);
                            } else if (use_resolved_insert && block_want_matrix) {
                                const auto resolved = getResolvedCellMatrixEntries(
                                    cid, bs.row_dof_map, bs.row_dof_offset,
                                    bs.col_dof_map, bs.col_dof_offset,
                                    parent_term.matrix_view);
                                insertLocal(output, rd, cd, ins_mat, ins_vec, resolved);
                            } else {
                                insertLocal(output, rd, cd, ins_mat, ins_vec);
                            }
                        }

                        result.elements_assembled++;
                        if (output.has_matrix)
                            result.matrix_entries_inserted += static_cast<GlobalIndex>(
                                batch_dofs[bi][slot].row_dofs.size() * batch_dofs[bi][slot].col_dofs.size());
                        if (output.has_vector)
                            result.vector_entries_inserted += static_cast<GlobalIndex>(
                                batch_dofs[bi][slot].row_dofs.size());
                    }
                    tp_fb_scatter += TP() - tp0_scatter;
                }

                // === Fused combined insertion (after all blocks) ===
                if (use_fused_insert) {
                    double tp0 = TP();
                    flushCombinedInsertBatch(
                        gids.subspan(begin, active),
                        fused_combined_n,
                        CombinedInsertTarget{
                            .matrix_view = parent_term.matrix_view,
                            .vector_view = parent_term.vector_view,
                            .assemble_matrix = parent_term.assemble_matrix,
                            .assemble_vector = parent_term.assemble_vector,
                        });
                    tp_fb_insert += TP() - tp0;
                }
        }
        } // end serial batch else-branch
    };

    // Main fused-batch range processor
    auto assemble_fused_batch_range = [&](std::span<const GlobalIndex> gids) {
        for (std::size_t begin = 0; begin < gids.size(); begin += B) {
            const std::size_t active = std::min(B, gids.size() - begin);

            // === PHASE 1: Geometry + field sols + term[0] basis ===
            for (std::size_t slot = 0; slot < active; ++slot) {
                const auto cell_id = gids[begin + slot];
                auto& ctx = batch_contexts[slot];

                double tp0 = TP();
                prepareGeometry(ctx, mesh, cell_id, *fused_quad_rule);
                tp_fb_geom += TP() - tp0;

                tp0 = TP();
                saved_node_coords[slot].node_coords = scratch_node_coords_;
                saved_node_coords[slot].entity_h = cached_geom_h_;
                saved_node_coords[slot].entity_volume = cached_geom_volume_;
                tp_fb_save += TP() - tp0;

                tp0 = TP();
                prepareBasis(ctx, mesh, cell_id, *terms[0].test_space, *terms[0].trial_space,
                             term_data[0].required_data, *fused_quad_rule);
                tp_fb_basis += TP() - tp0;

                tp0 = TP();
                ctx.setMaterialState(nullptr, nullptr, 0u, 0u);
                ctx.setTimeIntegrationContext(time_integration_);
                ctx.setTime(time_);
                ctx.setTimeStep(dt_);
                ctx.setRealParameterGetter(get_real_param_);
                ctx.setParameterGetter(get_param_);
                ctx.setUserData(user_data_);
                ctx.setJITConstants(jit_constants_);
                ctx.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
                ctx.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
                ctx.clearAllPreviousSolutionData();
                tp_fb_setters += TP() - tp0;

                if (any_need_field_solutions) {
                    tp0 = TP();
                    populateFieldSolutionData(ctx, mesh, cell_id, union_field_reqs);
                    tp_fb_field += TP() - tp0;
                }

                tp0 = TP();
                {
                    const auto& t0 = terms[0];
                    auto& rd = batch_dofs[0][slot].row_dofs;
                    auto& cd = batch_dofs[0][slot].col_dofs;
                    rd = getCellDofsCached(mesh, cell_id, t0.row_dof_map, t0.row_dof_offset);
                    cd = getCellDofsCached(mesh, cell_id, t0.col_dof_map, t0.col_dof_offset);
                }
                tp_fb_dof += TP() - tp0;

                tp0 = TP();
                gather_solution(0, slot, cell_id, ctx, batch_dofs[0][slot].col_dofs);
                tp_fb_sol += TP() - tp0;

                if (term_data[0].need_material_state) {
                    FE_THROW_IF(material_state_provider_ == nullptr, FEException,
                                "assembleCellsFused: kernel requires material state but no provider was set");
                    auto view = material_state_provider_->getCellState(*terms[0].kernel, cell_id,
                                                                       ctx.numQuadraturePoints());
                    FE_THROW_IF(!view, FEException, "assembleCellsFused: material state provider returned null");
                    ctx.setMaterialState(view.data_old, view.data_work,
                                         view.bytes_per_qpt, view.stride_bytes, view.alignment);
                }

                batch_outputs[slot].clear();
                batch_context_ptrs[slot] = &ctx;
            }

            // Batch-compute term[0]
            if (terms[0].kernel->hasCell() && (terms[0].assemble_matrix || terms[0].assemble_vector)) {
                double tp0 = TP();
                terms[0].kernel->computeCellBatch(
                    std::span<const AssemblyContext* const>(batch_context_ptrs.data(), active),
                    std::span<KernelOutput>(batch_outputs.data(), active));
                tp_fb_kernel += TP() - tp0;
                tp0 = TP();
                insert_batch_outputs(0, active, gids, begin);
                tp_fb_insert += TP() - tp0;
            }

            // === PHASE 2: Remaining terms (restore geometry, re-prepare basis) ===
            for (std::size_t ti = 1; ti < terms.size(); ++ti) {
                const auto& t = terms[ti];
                const auto& td = term_data[ti];
                if (!t.kernel->hasCell()) continue;
                if (!t.assemble_matrix && !t.assemble_vector) continue;

                for (std::size_t slot = 0; slot < active; ++slot) {
                    const auto cell_id = gids[begin + slot];
                    auto& ctx = batch_contexts[slot];

                    double tp0 = TP();
                    {
                        // Restore cached entity measures for this slot.
                        // For affine elements, skip coordinate restore + resetNodes
                        // since the prepareBasis fast path uses cached entity measures
                        // and never accesses mapping nodes.
                        cached_geom_h_ = saved_node_coords[slot].entity_h;
                        cached_geom_volume_ = saved_node_coords[slot].entity_volume;
                        if (!cached_mapping_affine_) {
                            scratch_node_coords_ = saved_node_coords[slot].node_coords;
                            cached_mapping_->resetNodes(scratch_node_coords_);
                        }
                    }
                    tp_fb_restore += TP() - tp0;

                    tp0 = TP();
                    prepareBasis(ctx, mesh, cell_id, *t.test_space, *t.trial_space,
                                 td.required_data, *fused_quad_rule);
                    tp_fb_basis += TP() - tp0;

                    tp0 = TP();
                    {
                        auto& rd = batch_dofs[ti][slot].row_dofs;
                        auto& cd = batch_dofs[ti][slot].col_dofs;
                        rd = getCellDofsCached(mesh, cell_id, t.row_dof_map, t.row_dof_offset);
                        cd = getCellDofsCached(mesh, cell_id, t.col_dof_map, t.col_dof_offset);
                    }
                    tp_fb_dof += TP() - tp0;

                    tp0 = TP();
                    gather_solution(ti, slot, cell_id, ctx, batch_dofs[ti][slot].col_dofs);
                    tp_fb_sol += TP() - tp0;

                    if (td.need_material_state) {
                        FE_THROW_IF(material_state_provider_ == nullptr, FEException,
                                    "assembleCellsFused: kernel requires material state but no provider was set");
                        auto view = material_state_provider_->getCellState(*t.kernel, cell_id,
                                                                           ctx.numQuadraturePoints());
                        FE_THROW_IF(!view, FEException, "assembleCellsFused: material state provider returned null");
                        ctx.setMaterialState(view.data_old, view.data_work,
                                             view.bytes_per_qpt, view.stride_bytes, view.alignment);
                    }

                    batch_outputs[slot].clear();
                    batch_context_ptrs[slot] = &ctx;
                }

                double tp0 = TP();
                t.kernel->computeCellBatch(
                    std::span<const AssemblyContext* const>(batch_context_ptrs.data(), active),
                    std::span<KernelOutput>(batch_outputs.data(), active));
                tp_fb_kernel += TP() - tp0;

                tp0 = TP();
                insert_batch_outputs(ti, active, gids, begin);
                tp_fb_insert += TP() - tp0;
            }
        }

    };

    // Topology grouping (homogeneous batches)
    const bool allow_topology_reorder =
        !options_.stable_insertion_order && (options_.default_mode == AssemblyMode::Add);
    if (!allow_topology_reorder) {
        std::size_t run_begin = 0u;
        while (run_begin < cell_ids.size()) {
            const auto run_type = mesh.getCellType(cell_ids[run_begin]);
            std::size_t run_end = run_begin + 1u;
            while (run_end < cell_ids.size() && mesh.getCellType(cell_ids[run_end]) == run_type)
                ++run_end;
            const auto batch_span = std::span<const GlobalIndex>(
                cell_ids.data() + run_begin, run_end - run_begin);
            if (mixed_block_kernel)
                assemble_mixed_block_batch_range(batch_span);
            else
                assemble_fused_batch_range(batch_span);
            run_begin = run_end;
        }
    } else {
        std::vector<std::pair<ElementType, std::vector<GlobalIndex>>> topo_groups;
        for (const auto cid : cell_ids) {
            const auto ct = mesh.getCellType(cid);
            auto it = std::find_if(topo_groups.begin(), topo_groups.end(),
                                   [&](const auto& g) { return g.first == ct; });
            if (it == topo_groups.end()) {
                topo_groups.emplace_back(ct, std::vector<GlobalIndex>{});
                it = topo_groups.end() - 1;
            }
            it->second.push_back(cid);
        }
        for (const auto& g : topo_groups) {
            if (mixed_block_kernel)
                assemble_mixed_block_batch_range(std::span<const GlobalIndex>(g.second));
            else
                assemble_fused_batch_range(std::span<const GlobalIndex>(g.second));
        }
    }

#ifdef SVMP_FE_ASSEMBLY_TIMING
    if (assemblyTimingEnabled())
    // Fused+batch timing output
    {
        int rank = 0;
#if FE_HAS_MPI
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        if (mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        if (rank == 0) {
            const double total = tp_fb_geom + tp_fb_save + tp_fb_restore + tp_fb_basis +
                                 tp_fb_field + tp_fb_dof + tp_fb_sol + tp_fb_setters +
                                 tp_fb_kernel + tp_fb_insert + tp_fb_snap;
            if (total > 1e-7) {
                std::fprintf(stderr,
                    "    --- cellLoop FUSED+BATCH TIMING (rank 0, %zu cells, %zu terms, batch=%zu%s) ---\n"
                    "      Total:           %9.6f s\n"
                    "      geometry:        %9.6f s  (%5.1f%%)\n"
                    "      save geom:       %9.6f s  (%5.1f%%)\n"
                    "      restore geom:    %9.6f s  (%5.1f%%)\n"
                    "      prepareBasis:    %9.6f s  (%5.1f%%)\n"
                    "      field solutions: %9.6f s  (%5.1f%%)\n"
                    "      dof lookup:      %9.6f s  (%5.1f%%)\n"
                    "      sol gather:      %9.6f s  (%5.1f%%)\n"
                    "      ctx setters:     %9.6f s  (%5.1f%%)\n"
                    "      snap+pack:       %9.6f s  (%5.1f%%)\n"
                    "      kernel (batch):  %9.6f s  (%5.1f%%)\n"
                    "      scatter:         %9.6f s  (%5.1f%%)\n"
                    "      insert:          %9.6f s  (%5.1f%%)\n"
                    "    ------------------------------------\n",
                    cell_ids.size(), terms.size(), requested_batch_size,
                    mixed_block_kernel ? " MIXED_BLOCK" : "",
                    total,
                    tp_fb_geom, 100.0 * tp_fb_geom / total,
                    tp_fb_save, 100.0 * tp_fb_save / total,
                    tp_fb_restore, 100.0 * tp_fb_restore / total,
                    tp_fb_basis, 100.0 * tp_fb_basis / total,
                    tp_fb_field, 100.0 * tp_fb_field / total,
                    tp_fb_dof, 100.0 * tp_fb_dof / total,
                    tp_fb_sol, 100.0 * tp_fb_sol / total,
                    tp_fb_setters, 100.0 * tp_fb_setters / total,
                    tp_fb_snap, 100.0 * tp_fb_snap / total,
                    tp_fb_kernel, 100.0 * tp_fb_kernel / total,
                    tp_fb_scatter, 100.0 * tp_fb_scatter / total,
                    tp_fb_insert, 100.0 * tp_fb_insert / total);
            }
        }
    }
#endif

    } else {
    // ========================================================================
    // FUSED PER-CELL PATH (fallback when kernels don't support batch)
    // ========================================================================

    // Timing
    double tp_geometry = 0.0, tp_term_loop = 0.0;
    double tp_fused_dof = 0.0, tp_fused_basis = 0.0, tp_fused_sol = 0.0;
    double tp_fused_kernel = 0.0, tp_fused_insert = 0.0, tp_fused_field = 0.0;
    auto TP = assemblyTimeNow;

    // ========================================================================
    // Main fused cell loop
    // ========================================================================
    for (const auto cell_id : cell_ids) {
        double tp0 = TP();

        // 1. Prepare geometry ONCE for this cell
        prepareGeometry(context_, mesh, cell_id, *fused_quad_rule);

        // 1b. Configure context with first term's spaces and set geometry data
        //     so that populateFieldSolutionData can access QP/geometry info.
        //     prepareBasis will reconfigure per-term later (FunctionSpace configure
        //     variant does NOT clear field_solution_data).
        {
            const auto& ft0 = terms[0];
            context_.configure(cell_id, *ft0.test_space, *ft0.trial_space,
                               term_data[0].required_data);
            context_.setCellDomainId(mesh.getCellDomainId(cell_id));
            // Geometry already in arena from prepareGeometry — no copy needed
        }

        // 2. Set context global state ONCE
        context_.setMaterialState(nullptr, nullptr, 0u, 0u);
        context_.setTimeIntegrationContext(time_integration_);
        context_.setTime(time_);
        context_.setTimeStep(dt_);
        context_.setRealParameterGetter(get_real_param_);
        context_.setParameterGetter(get_param_);
        context_.setUserData(user_data_);
        context_.setJITConstants(jit_constants_);
        context_.setAuxiliaryValues(auxiliary_inputs_, auxiliary_state_, auxiliary_outputs_);
        context_.setAuxiliaryOutputBindings(auxiliary_output_bindings_);
        context_.clearAllPreviousSolutionData();

        // 3. Populate union of field solution data ONCE
        if (any_need_field_solutions) {
            double tp_f0 = TP();
            populateFieldSolutionData(context_, mesh, cell_id, union_field_reqs);
            tp_fused_field += TP() - tp_f0;
        }

        tp_geometry += TP() - tp0;

        // 4. Loop over terms
        tp0 = TP();
        for (std::size_t ti = 0; ti < terms.size(); ++ti) {
            const auto& t = terms[ti];
            auto& td = term_data[ti];
            auto& ts = term_scratch[ti];

            if (!t.kernel->hasCell()) continue;
            if (!t.assemble_matrix && !t.assemble_vector) continue;

            // a. DOF map lookup
            double tp_a = TP();
            ts.row_dofs = getCellDofsCached(mesh, cell_id, t.row_dof_map, t.row_dof_offset);
            ts.col_dofs = getCellDofsCached(mesh, cell_id, t.col_dof_map, t.col_dof_offset);
            tp_fused_dof += TP() - tp_a;

            // b. Prepare basis for this term's test/trial spaces
            tp_a = TP();
            prepareBasis(context_, mesh, cell_id, *t.test_space, *t.trial_space,
                         td.required_data, *fused_quad_rule);
            tp_fused_basis += TP() - tp_a;

	            // c. Solution coefficient gather for this term's trial DOFs
	            tp_a = TP();
	            if (td.need_solution) {
	                FE_THROW_IF(current_solution_view_ == nullptr && current_solution_.empty(), FEException,
	                            "assembleCellsFused: kernel requires solution but no solution was set");
	                local_solution_coeffs_.resize(ts.col_dofs.size());
                    gatherCellVectorCoefficients(cell_id, t.col_dof_map, t.col_dof_offset,
                                                 ts.col_dofs, current_solution_view_,
                                                 current_solution_, local_solution_coeffs_,
                                                 "assembleCellsFused", true);
	                if (context_.trialUsesVectorBasis()) {
	                    applyVectorBasisGlobalToLocal(mesh, cell_id, *t.trial_space,
	                                                  std::span<Real>(local_solution_coeffs_));
                }
                context_.setSolutionCoefficients(local_solution_coeffs_);

                // d. Previous solution gather
                if (time_integration_ != nullptr) {
                    const int required = requiredHistoryStates(time_integration_);
                    if (required > 0) {
                        FE_THROW_IF(previous_solutions_.size() < static_cast<std::size_t>(required), FEException,
                                    "assembleCellsFused: time integration requires " +
                                        std::to_string(required) + " history states");
                        if (local_prev_solution_coeffs_.size() < static_cast<std::size_t>(required)) {
                            local_prev_solution_coeffs_.resize(static_cast<std::size_t>(required));
                        }
                        for (int k = 1; k <= required; ++k) {
                            const auto& prev = previous_solutions_[static_cast<std::size_t>(k - 1)];
                            const auto* prev_view = (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
                                                        ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
                                                        : nullptr;
	                            FE_THROW_IF(prev.empty() && prev_view == nullptr, FEException,
	                                        "assembleCellsFused: previous solution (k=" +
	                                            std::to_string(k) + ") not set");
	                            auto& local_prev = local_prev_solution_coeffs_[static_cast<std::size_t>(k - 1)];
                                gatherCellVectorCoefficients(cell_id, t.col_dof_map,
                                                             t.col_dof_offset,
                                                             ts.col_dofs, prev_view, prev,
                                                             local_prev,
                                                             "assembleCellsFused", true);
	                            if (context_.trialUsesVectorBasis()) {
	                                applyVectorBasisGlobalToLocal(mesh, cell_id, *t.trial_space,
	                                                              std::span<Real>(local_prev));
                            }
                            context_.setPreviousSolutionCoefficientsK(k, local_prev);
                        }
                    }
                }
            }

            tp_fused_sol += TP() - tp_a;

            // e. Material state binding
            if (td.need_material_state) {
                FE_THROW_IF(material_state_provider_ == nullptr, FEException,
                            "assembleCellsFused: kernel requires material state but no provider was set");
                auto view = material_state_provider_->getCellState(*t.kernel, cell_id,
                                                                    context_.numQuadraturePoints());
                FE_THROW_IF(!view, FEException, "assembleCellsFused: material state provider returned null");
                context_.setMaterialState(view.data_old, view.data_work,
                                          view.bytes_per_qpt, view.stride_bytes, view.alignment);
            }

            // f. Compute kernel
            tp_a = TP();
            kernel_output_.clear();
            t.kernel->computeCell(context_, kernel_output_);
            tp_fused_kernel += TP() - tp_a;

            // g. Apply orientation transforms
            if (context_.testUsesVectorBasis() || context_.trialUsesVectorBasis()) {
                applyVectorBasisOutputOrientation(mesh, cell_id, *t.test_space,
                                                  cell_id, *t.trial_space, kernel_output_);
            }

            // h. Insert into global system
            tp_a = TP();
            insertLocalForCell(cell_id, t.row_dof_map, t.row_dof_offset,
                               t.col_dof_map, t.col_dof_offset,
                               kernel_output_, ts.row_dofs, ts.col_dofs,
                               t.assemble_matrix ? ts.insert_matrix : nullptr,
                               t.assemble_vector ? ts.insert_vector : nullptr);

            tp_fused_insert += TP() - tp_a;

            result.elements_assembled++;
            if (kernel_output_.has_matrix) {
                result.matrix_entries_inserted +=
                    static_cast<GlobalIndex>(ts.row_dofs.size() * ts.col_dofs.size());
            }
            if (kernel_output_.has_vector) {
                result.vector_entries_inserted += static_cast<GlobalIndex>(ts.row_dofs.size());
            }
        }
        tp_term_loop += TP() - tp0;
    }

#ifdef SVMP_FE_ASSEMBLY_TIMING
    if (assemblyTimingEnabled())
    // Print fused timing
    {
        int rank = 0;
#if FE_HAS_MPI
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        if (mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        if (rank == 0) {
            const double total = tp_geometry + tp_term_loop;
            if (total > 1e-7) {
                std::fprintf(stderr,
                    "    --- cellLoop FUSED TIMING (rank 0, %zu cells, %zu terms) ---\n"
                    "      Total:          %9.6f s\n"
                    "      geometry+cfg:   %9.6f s  (%5.1f%%)\n"
                    "        field sols:   %9.6f s  (%5.1f%%)\n"
                    "      term loop:      %9.6f s  (%5.1f%%)\n"
                    "        dof lookup:   %9.6f s  (%5.1f%%)\n"
                    "        prepareBasis: %9.6f s  (%5.1f%%)\n"
                    "        sol gather:   %9.6f s  (%5.1f%%)\n"
                    "        kernel:       %9.6f s  (%5.1f%%)\n"
                    "        insert:       %9.6f s  (%5.1f%%)\n"
                    "    ------------------------------------\n",
                    cell_ids.size(), terms.size(),
                    total,
                    tp_geometry, 100.0 * tp_geometry / total,
                    tp_fused_field, 100.0 * tp_fused_field / total,
                    tp_term_loop, 100.0 * tp_term_loop / total,
                    tp_fused_dof, 100.0 * tp_fused_dof / total,
                    tp_fused_basis, 100.0 * tp_fused_basis / total,
                    tp_fused_sol, 100.0 * tp_fused_sol / total,
                    tp_fused_kernel, 100.0 * tp_fused_kernel / total,
                    tp_fused_insert, 100.0 * tp_fused_insert / total);
            }
        }
    }
#endif

    } // end if/else fused path selection

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
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
    // Face assembly reuses cell-assembly scratch arrays (scratch_ref_gradients_, etc.).
    // Invalidate the cell-basis fast-path cache so the next prepareBasis call
    // falls through to the slow path and repopulates the scratch arrays.
    basis_scratch_valid_ = false;

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

    // Representative interior point for outward-normal consistency checks.
    AssemblyContext::Vector3D cell_center{0.0, 0.0, 0.0};
    if (!cell_coords_.empty()) {
        for (const auto& xc : cell_coords_) {
            cell_center[0] += xc[0];
            cell_center[1] += xc[1];
            cell_center[2] += xc[2];
        }
        const Real inv_n = Real(1.0) / static_cast<Real>(cell_coords_.size());
        cell_center[0] *= inv_n;
        cell_center[1] *= inv_n;
        cell_center[2] *= inv_n;
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

    auto [vtx, ref_face_coords] =
        elements::ElementTransform::facet_vertices(cell_type, static_cast<int>(local_face_id));
    (void)vtx;

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

        // Ensure the physical unit normal points outward from the cell interior.
        // For some meshes, facet vertex ordering can be inconsistent even when det(J) > 0; rely on a
        // geometric check against an interior point (legacy gnnb-style).
        {
            const Real dx = cell_center[0] - x_phys[0];
            const Real dy = cell_center[1] - x_phys[1];
            const Real dz = cell_center[2] - x_phys[2];
            const Real dot = dx * n_phys[0] + dy * n_phys[1] + dz * n_phys[2];
            if (dot > Real(0.0)) {
                n_phys[0] = -n_phys[0];
                n_phys[1] = -n_phys[1];
                n_phys[2] = -n_phys[2];
            }
        }

        // Convert canonical face weights to element-reference facet measure, then map to physical.
        Real w = quad_weights[q] *
                 canonicalFaceJacobianToReference(face_type,
                                                  std::span<const math::Vector<Real, 3>>(ref_face_coords),
                                                  facet_coords);

        scratch_integration_weights_[q] = w * surface_measure;
        scratch_normals_[q] = n_phys;
    }

    // 9. Evaluate basis functions at face quadrature points
    auto& scalar_values_at_pt = scratch_scalar_values_at_pt_;
    auto& scalar_gradients_at_pt = scratch_scalar_gradients_at_pt_;
    auto& scalar_hessians_at_pt = scratch_scalar_hessians_at_pt_;

    auto& vec_values_at_pt = scratch_vec_values_at_pt_;
    auto& vec_curls_at_pt = scratch_vec_curls_at_pt_;
    auto& vec_divs_at_pt = scratch_vec_divs_at_pt_;

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
                const std::size_t idx_phys = static_cast<std::size_t>(q * n_test_dofs + i);
                scratch_basis_values_[idx] = scalar_values_at_pt[static_cast<std::size_t>(si)];
                scratch_ref_gradients_[idx] = {
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][0],
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][1],
                    scalar_gradients_at_pt[static_cast<std::size_t>(si)][2]};

                const auto& grad_ref = scratch_ref_gradients_[idx];
                AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};
                if (dim == 3) {
                    grad_phys[0] = J_inv[0][0] * grad_ref[0] + J_inv[1][0] * grad_ref[1] + J_inv[2][0] * grad_ref[2];
                    grad_phys[1] = J_inv[0][1] * grad_ref[0] + J_inv[1][1] * grad_ref[1] + J_inv[2][1] * grad_ref[2];
                    grad_phys[2] = J_inv[0][2] * grad_ref[0] + J_inv[1][2] * grad_ref[1] + J_inv[2][2] * grad_ref[2];
                } else if (dim == 2) {
                    grad_phys[0] = J_inv[0][0] * grad_ref[0] + J_inv[1][0] * grad_ref[1];
                    grad_phys[1] = J_inv[0][1] * grad_ref[0] + J_inv[1][1] * grad_ref[1];
                } else if (dim == 1) {
                    grad_phys[0] = J_inv[0][0] * grad_ref[0];
                }
                scratch_phys_gradients_[idx_phys] = grad_phys;

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
                    const std::size_t idx_phys = static_cast<std::size_t>(q * n_trial_dofs + j);
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
                    trial_phys_gradients[idx_phys] = grad_phys;

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

    // For orientation-reversing mappings (det(J) < 0), the outward normal must be flipped.
    // The cofactor identity uses cof(J) = det(J) * J^{-T}, so the sign of det(J) is part
    // of the correct normal direction even though we use |det(J)| for surface measures.
    if (det_J < 0.0) {
        for (int i = 0; i < dim; ++i) {
            Jit_n[i] = -Jit_n[i];
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
    ensureFieldAccessPlans(mesh);

    // Suspend JIT field table rebuilds — each setter would rebuild redundantly.
    // One rebuild at the end of this function suffices.
    context.suspendJITFieldTableRebuild();

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
        for (const auto& s : time_integration_->dt_extra) {
            if (s) {
                required_history = std::max(required_history, s->requiredHistoryStates());
            }
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

    auto& values_at_pt = scratch_scalar_values_at_pt_;
    auto& gradients_at_pt = scratch_scalar_gradients_at_pt_;
    auto& hessians_at_pt = scratch_scalar_hessians_at_pt_;
    auto& local_coeffs = scratch_field_local_coeffs_;

    auto& scalar_values = scratch_fsd_scalar_values_;
    auto& scalar_gradients = scratch_fsd_scalar_gradients_;
    auto& scalar_hessians = scratch_fsd_scalar_hessians_;
    auto& scalar_laplacians = scratch_fsd_scalar_laplacians_;

    auto& vector_values = scratch_fsd_vector_values_;
    auto& vector_jacobians = scratch_fsd_vector_jacobians_;
    auto& vector_component_hessians = scratch_fsd_vector_component_hessians_;
    auto& vector_component_laplacians = scratch_fsd_vector_component_laplacians_;

    for (const auto& req : requirements) {
        FE_THROW_IF(req.field == INVALID_FIELD_ID, FEException,
                    "StandardAssembler::populateFieldSolutionData: kernel requested an invalid FieldId");

        const auto* access = findFieldAccessPlan(req.field);
        FE_THROW_IF(access == nullptr, FEException,
                    "StandardAssembler::populateFieldSolutionData: no FieldSolutionAccess was provided for field " +
                        std::to_string(req.field));
        FE_CHECK_NOT_NULL(access->space, "StandardAssembler::populateFieldSolutionData: field space");
        FE_CHECK_NOT_NULL(access->dof_map, "StandardAssembler::populateFieldSolutionData: field dof_map");
        FE_CHECK_NOT_NULL(access->dof_table, "StandardAssembler::populateFieldSolutionData: field dof table");

        const auto& space = *access->space;
        const auto& element = getElement(space, cell_id, cell_type);
        const auto& basis = element.basis();

        const bool is_product = access->is_product;
        const auto n_qpts = context.numQuadraturePoints();
        const auto n_dofs = static_cast<LocalIndex>(space.dofs_per_element());
        const auto n_scalar_dofs = static_cast<LocalIndex>(element.num_dofs());

        const bool want_values = hasFlag(req.required, RequiredData::SolutionValues) || (req.required == RequiredData::None);
        const bool want_gradients = hasFlag(req.required, RequiredData::SolutionGradients);
        const bool want_hessians = hasFlag(req.required, RequiredData::SolutionHessians);
        const bool want_laplacians = hasFlag(req.required, RequiredData::SolutionLaplacians);
        const bool need_gradients = want_gradients;
        const bool need_hessians = want_hessians || want_laplacians;

	        const auto cell_dofs = getCellDofsFromTable(*access->dof_table, cell_id);
	        FE_THROW_IF(cell_dofs.size() != static_cast<std::size_t>(n_dofs), FEException,
	                    "StandardAssembler::populateFieldSolutionData: field DOF count does not match its space DOFs");

	        local_coeffs.resize(cell_dofs.size());
            gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                         cell_dofs, current_solution_view_,
                                         current_solution_, local_coeffs,
                                         "StandardAssembler::populateFieldSolutionData", false);

        if (access->field_type == FieldType::Scalar) {
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

            // Use BasisCache for field basis evaluations when quad rule is available.
            // Look up in flat cache first to avoid repeated mutex+hash per cell.
            const basis::BasisCacheEntry* field_bcache = nullptr;
            if (cached_quad_rule_ &&
                static_cast<LocalIndex>(cached_quad_rule_->num_points()) == n_qpts) {
                for (const auto& fc : cached_field_bcache_) {
                    if (fc.basis == &basis && fc.gradients == need_gradients && fc.hessians == need_hessians) {
                        field_bcache = fc.entry;
                        break;
                    }
                }
                if (!field_bcache) {
                    field_bcache = &basis::BasisCache::instance().get_or_compute(
                        basis, *cached_quad_rule_, need_gradients, need_hessians);
                    cached_field_bcache_.push_back({&basis, need_gradients, need_hessians, field_bcache});
                }
            }

            // Get inverse Jacobians span once (avoid per-QP accessor overhead).
            const auto ctx_inv_jacs_span = context.inverseJacobians();

            // P1 affine fast path: for linear basis functions on affine
            // elements, ref-space gradients are constant across all QPs
            // (same basis gradients at every point).  Detect this by comparing
            // basis gradients at QP 0 vs QP 1.  When true:
            //  - gradient: compute gref_sum + J_inv transform at QP 0, copy to rest
            //  - hessians: zero for P1 (constant gradient → zero curvature)
            const bool p1_affine_grad = cached_mapping_affine_ && need_gradients &&
                                        field_bcache && n_qpts >= 2 && n_dofs > 0 &&
                                        field_bcache->gradients.size() >= 2 &&
                                        [&]() -> bool {
                                            for (LocalIndex j = 0; j < n_dofs; ++j) {
                                                const auto& g0 = field_bcache->gradients[0][static_cast<std::size_t>(j)];
                                                const auto& g1 = field_bcache->gradients[1][static_cast<std::size_t>(j)];
                                                for (int d = 0; d < dim; ++d) {
                                                    if (g0[static_cast<std::size_t>(d)] != g1[static_cast<std::size_t>(d)])
                                                        return false;
                                                }
                                            }
                                            return true;
                                        }();

            AssemblyContext::Vector3D p1_const_grad{};
            bool p1_grad_ready = false;

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                if (!field_bcache) {
                    const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                   qpts[static_cast<std::size_t>(q)][1],
                                                   qpts[static_cast<std::size_t>(q)][2]};
                    basis.evaluate_values(xi, values_at_pt);
                    if (need_gradients && !(p1_affine_grad && p1_grad_ready)) {
                        basis.evaluate_gradients(xi, gradients_at_pt);
                    }
                    if (need_hessians && !p1_affine_grad) {
                        basis.evaluate_hessians(xi, hessians_at_pt);
                    }
                }

                // For affine elements, J_inv is the same at all QPs.
                const auto& J_inv = ctx_inv_jacs_span[cached_mapping_affine_ ? 0 : static_cast<std::size_t>(q)];

                // Values always vary per QP (different N_j at each point)
                Real val = 0.0;

                // Accumulate in reference space first, then apply J_inv
                // transform once.  This saves (n_dofs-1) matrix-vector
                // products per QP for gradients and (n_dofs-1) matrix
                // triple products for hessians.
                AssemblyContext::Vector3D gref_sum = {0.0, 0.0, 0.0};
                AssemblyContext::Matrix3x3 H_ref_sum{};

                const bool do_grad_accum = need_gradients && !(p1_affine_grad && p1_grad_ready);
                const bool do_hess_accum = need_hessians && !p1_affine_grad;

                for (LocalIndex j = 0; j < n_dofs; ++j) {
                    const Real coef = local_coeffs[static_cast<std::size_t>(j)];
                    const Real basis_val = field_bcache
                        ? field_bcache->scalarValue(static_cast<std::size_t>(j), static_cast<std::size_t>(q))
                        : values_at_pt[static_cast<std::size_t>(j)];
                    val += coef * basis_val;

                    if (do_grad_accum) {
                        const auto& gref = field_bcache
                            ? field_bcache->gradients[static_cast<std::size_t>(q)][static_cast<std::size_t>(j)]
                            : gradients_at_pt[static_cast<std::size_t>(j)];
                        for (int d = 0; d < dim; ++d) {
                            gref_sum[d] += coef * gref[static_cast<std::size_t>(d)];
                        }
                    }

                    if (do_hess_accum) {
                        const auto& hess_j = field_bcache
                            ? field_bcache->hessians[static_cast<std::size_t>(q)][static_cast<std::size_t>(j)]
                            : hessians_at_pt[static_cast<std::size_t>(j)];
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                H_ref_sum[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                                    coef * hess_j(static_cast<std::size_t>(r),
                                                   static_cast<std::size_t>(c));
                            }
                        }
                    }
                }

                scalar_values[static_cast<std::size_t>(q)] = val;

                if (need_gradients) {
                    if (p1_affine_grad && p1_grad_ready) {
                        // Reuse constant gradient from QP 0
                        scalar_gradients[static_cast<std::size_t>(q)] = p1_const_grad;
                    } else {
                        // Single J_inv^T * gref_sum transform
                        AssemblyContext::Vector3D grad = {0.0, 0.0, 0.0};
                        for (int d1 = 0; d1 < dim; ++d1) {
                            for (int d2 = 0; d2 < dim; ++d2) {
                                grad[d1] += J_inv[static_cast<std::size_t>(d2)][static_cast<std::size_t>(d1)] *
                                            gref_sum[d2];
                            }
                        }
                        scalar_gradients[static_cast<std::size_t>(q)] = grad;
                        if (p1_affine_grad) {
                            p1_const_grad = grad;
                            p1_grad_ready = true;
                        }
                    }
                }

                if (need_hessians) {
                    if (p1_affine_grad) {
                        // P1 affine: hessians are zero (constant gradient)
                        scalar_hessians[static_cast<std::size_t>(q)] = AssemblyContext::Matrix3x3{};
                        if (want_laplacians) {
                            scalar_laplacians[static_cast<std::size_t>(q)] = 0.0;
                        }
                    } else {
                        // Single J_inv^T * H_ref_sum * J_inv transform
                        AssemblyContext::Matrix3x3 H{};
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                Real sum = 0.0;
                                for (int a = 0; a < dim; ++a) {
                                    for (int b = 0; b < dim; ++b) {
                                        sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                               H_ref_sum[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                               J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                    }
                                }
                                H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                            }
                        }
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

                        gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                                     cell_dofs, prev_view, prev,
                                                     local_coeffs,
                                                     "StandardAssembler::populateFieldSolutionData", false);

                    // Values only (dt() needs just value history).
                    scalar_values.assign(static_cast<std::size_t>(n_qpts), 0.0);
                    for (LocalIndex q = 0; q < n_qpts; ++q) {
                        if (field_bcache) {
                            values_at_pt.resize(static_cast<std::size_t>(n_scalar_dofs));
                            for (LocalIndex j = 0; j < n_scalar_dofs; ++j) {
                                values_at_pt[static_cast<std::size_t>(j)] =
                                    field_bcache->scalarValue(static_cast<std::size_t>(j),
                                                             static_cast<std::size_t>(q));
                            }
                        } else {
                            const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                           qpts[static_cast<std::size_t>(q)][1],
                                                           qpts[static_cast<std::size_t>(q)][2]};
                            basis.evaluate_values(xi, values_at_pt);
                        }
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

        if (access->field_type == FieldType::Vector) {
            const int vd = access->value_dimension;
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

                auto& vec_values_at_pt = scratch_vec_values_at_pt_;
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

                            gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                                         cell_dofs, prev_view, prev,
                                                         local_coeffs,
                                                         "StandardAssembler::populateFieldSolutionData", false);

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

            // Use BasisCache for ProductSpace field basis evaluations.
            // Look up in flat cache first to avoid repeated mutex+hash per cell.
            const basis::BasisCacheEntry* field_bcache_ps = nullptr;
            if (cached_quad_rule_ &&
                static_cast<LocalIndex>(cached_quad_rule_->num_points()) == n_qpts) {
                for (const auto& fc : cached_field_bcache_) {
                    if (fc.basis == &basis && fc.gradients == need_gradients && fc.hessians == need_hessians) {
                        field_bcache_ps = fc.entry;
                        break;
                    }
                }
                if (!field_bcache_ps) {
                    field_bcache_ps = &basis::BasisCache::instance().get_or_compute(
                        basis, *cached_quad_rule_, need_gradients, need_hessians);
                    cached_field_bcache_.push_back({&basis, need_gradients, need_hessians, field_bcache_ps});
                }
            }

            // Get inverse Jacobians span once (avoid per-QP accessor overhead).
            const auto ctx_inv_jacs_ps = context.inverseJacobians();

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                if (!field_bcache_ps) {
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
                }

                const auto& J_inv = ctx_inv_jacs_ps[cached_mapping_affine_ ? 0 : static_cast<std::size_t>(q)];
                AssemblyContext::Matrix3x3 J{};

                const auto q_base = static_cast<std::size_t>(q) * static_cast<std::size_t>(vd);
                for (int comp = 0; comp < vd; ++comp) {
                    Real val_c = 0.0;
                    // Accumulate in reference space, then transform once.
                    AssemblyContext::Vector3D gref_sum_c = {0.0, 0.0, 0.0};
                    AssemblyContext::Matrix3x3 H_ref_sum_c{};

                    const LocalIndex base = static_cast<LocalIndex>(comp) * dofs_per_component;
                    for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                        const LocalIndex jj = base + j;
                        const LocalIndex sj = static_cast<LocalIndex>(jj % n_scalar_dofs);
                        const Real coef = local_coeffs[static_cast<std::size_t>(jj)];
                        const Real basis_val = field_bcache_ps
                            ? field_bcache_ps->scalarValue(static_cast<std::size_t>(sj), static_cast<std::size_t>(q))
                            : values_at_pt[static_cast<std::size_t>(sj)];
                        val_c += coef * basis_val;

                        if (need_gradients) {
                            const auto& gref = field_bcache_ps
                                ? field_bcache_ps->gradients[static_cast<std::size_t>(q)][static_cast<std::size_t>(sj)]
                                : gradients_at_pt[static_cast<std::size_t>(sj)];
                            for (int d = 0; d < dim; ++d) {
                                gref_sum_c[d] += coef * gref[static_cast<std::size_t>(d)];
                            }
                        }

                        if (need_hessians) {
                            const auto& hess_sj = field_bcache_ps
                                ? field_bcache_ps->hessians[static_cast<std::size_t>(q)][static_cast<std::size_t>(sj)]
                                : hessians_at_pt[static_cast<std::size_t>(sj)];
                            for (int r = 0; r < dim; ++r) {
                                for (int c = 0; c < dim; ++c) {
                                    H_ref_sum_c[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                                        coef * hess_sj(static_cast<std::size_t>(r),
                                                        static_cast<std::size_t>(c));
                                }
                            }
                        }
                    }

                    vector_values[static_cast<std::size_t>(q)][static_cast<std::size_t>(comp)] = val_c;

                    if (need_gradients) {
                        // Single J_inv^T * gref_sum transform per component
                        for (int d1 = 0; d1 < dim; ++d1) {
                            Real sum = 0.0;
                            for (int d2 = 0; d2 < dim; ++d2) {
                                sum += J_inv[static_cast<std::size_t>(d2)][static_cast<std::size_t>(d1)] *
                                       gref_sum_c[d2];
                            }
                            J[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d1)] += sum;
                        }
                    }

                    if (need_hessians) {
                        AssemblyContext::Matrix3x3 H{};
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                Real sum = 0.0;
                                for (int a = 0; a < dim; ++a) {
                                    for (int b = 0; b < dim; ++b) {
                                        sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                               H_ref_sum_c[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                               J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                    }
                                }
                                H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                            }
                        }
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

                        gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                                     cell_dofs, prev_view, prev,
                                                     local_coeffs,
                                                     "StandardAssembler::populateFieldSolutionData", false);

                    // Values only (dt() needs just value history).
                    vector_values.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Vector3D{0.0, 0.0, 0.0});
                    for (LocalIndex q = 0; q < n_qpts; ++q) {
                        if (field_bcache_ps) {
                            values_at_pt.resize(static_cast<std::size_t>(n_scalar_dofs));
                            for (LocalIndex j = 0; j < n_scalar_dofs; ++j) {
                                values_at_pt[static_cast<std::size_t>(j)] =
                                    field_bcache_ps->scalarValue(static_cast<std::size_t>(j),
                                                                static_cast<std::size_t>(q));
                            }
                        } else {
                            const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                           qpts[static_cast<std::size_t>(q)][1],
                                                           qpts[static_cast<std::size_t>(q)][2]};
                            basis.evaluate_values(xi, values_at_pt);
                        }

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
	    // Resume JIT field table rebuild (one rebuild instead of per-setter).
	    context.resumeJITFieldTableRebuild();
	}

void StandardAssembler::populateFieldSolutionData(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const std::vector<FieldRequirement>& requirements,
    FieldSolutionWorkspace& ws)
{
    context.clearFieldSolutionData();
    if (requirements.empty()) {
        return;
    }
    // NOTE: ensureFieldAccessPlans() must be called before the parallel region.
    // Do NOT call it here — it mutates shared state.

    // Suspend JIT field table rebuilds — each setter would rebuild redundantly.
    // One rebuild at the end of this function suffices.
    context.suspendJITFieldTableRebuild();

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
        for (const auto& s : time_integration_->dt_extra) {
            if (s) {
                required_history = std::max(required_history, s->requiredHistoryStates());
            }
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

    auto& values_at_pt = ws.scalar_values_at_pt;
    auto& gradients_at_pt = ws.scalar_gradients_at_pt;
    auto& hessians_at_pt = ws.scalar_hessians_at_pt;
    auto& local_coeffs = ws.field_local_coeffs;

    auto& scalar_values = ws.fsd_scalar_values;
    auto& scalar_gradients = ws.fsd_scalar_gradients;
    auto& scalar_hessians = ws.fsd_scalar_hessians;
    auto& scalar_laplacians = ws.fsd_scalar_laplacians;

    auto& vector_values = ws.fsd_vector_values;
    auto& vector_jacobians = ws.fsd_vector_jacobians;
    auto& vector_component_hessians = ws.fsd_vector_comp_hessians;
    auto& vector_component_laplacians = ws.fsd_vector_comp_laplacians;

    for (const auto& req : requirements) {
        FE_THROW_IF(req.field == INVALID_FIELD_ID, FEException,
                    "StandardAssembler::populateFieldSolutionData: kernel requested an invalid FieldId");

        const auto* access = findFieldAccessPlan(req.field);
        FE_THROW_IF(access == nullptr, FEException,
                    "StandardAssembler::populateFieldSolutionData: no FieldSolutionAccess was provided for field " +
                        std::to_string(req.field));
        FE_CHECK_NOT_NULL(access->space, "StandardAssembler::populateFieldSolutionData: field space");
        FE_CHECK_NOT_NULL(access->dof_map, "StandardAssembler::populateFieldSolutionData: field dof_map");
        FE_CHECK_NOT_NULL(access->dof_table, "StandardAssembler::populateFieldSolutionData: field dof table");

        const auto& space = *access->space;
        const auto& element = getElement(space, cell_id, cell_type);
        const auto& basis = element.basis();

        const bool is_product = access->is_product;
        const auto n_qpts = context.numQuadraturePoints();
        const auto n_dofs = static_cast<LocalIndex>(space.dofs_per_element());
        const auto n_scalar_dofs = static_cast<LocalIndex>(element.num_dofs());

        const bool want_values = hasFlag(req.required, RequiredData::SolutionValues) || (req.required == RequiredData::None);
        const bool want_gradients = hasFlag(req.required, RequiredData::SolutionGradients);
        const bool want_hessians = hasFlag(req.required, RequiredData::SolutionHessians);
        const bool want_laplacians = hasFlag(req.required, RequiredData::SolutionLaplacians);
        const bool need_gradients = want_gradients;
        const bool need_hessians = want_hessians || want_laplacians;

	        const auto cell_dofs = getCellDofsFromTable(*access->dof_table, cell_id);
	        FE_THROW_IF(cell_dofs.size() != static_cast<std::size_t>(n_dofs), FEException,
	                    "StandardAssembler::populateFieldSolutionData: field DOF count does not match its space DOFs");

	        local_coeffs.resize(cell_dofs.size());
            gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                         cell_dofs, current_solution_view_,
                                         current_solution_, local_coeffs,
                                         "StandardAssembler::populateFieldSolutionData", false);

        if (access->field_type == FieldType::Scalar) {
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

            // Use BasisCache for field basis evaluations when quad rule is available.
            // Read-only lookup in pre-populated cache; do NOT insert on miss
            // (this overload may run from multiple threads).
            const basis::BasisCacheEntry* field_bcache = nullptr;
            if (cached_quad_rule_ &&
                static_cast<LocalIndex>(cached_quad_rule_->num_points()) == n_qpts) {
                for (const auto& fc : cached_field_bcache_) {
                    if (fc.basis == &basis && fc.gradients == need_gradients && fc.hessians == need_hessians) {
                        field_bcache = fc.entry;
                        break;
                    }
                }
                // Cache miss: query the global thread-safe BasisCache singleton
                // but do NOT insert into cached_field_bcache_ (shared state).
                if (!field_bcache) {
                    field_bcache = &basis::BasisCache::instance().get_or_compute(
                        basis, *cached_quad_rule_, need_gradients, need_hessians);
                }
            }

            // Get inverse Jacobians span once (avoid per-QP accessor overhead).
            const auto ctx_inv_jacs_span = context.inverseJacobians();

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                if (!field_bcache) {
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
                }

                // For affine elements, J_inv is the same at all QPs.
                const auto& J_inv = ctx_inv_jacs_span[cached_mapping_affine_ ? 0 : static_cast<std::size_t>(q)];
                Real val = 0.0;

                // Accumulate in reference space first, then apply J_inv
                // transform once.  This saves (n_dofs-1) matrix-vector
                // products per QP for gradients and (n_dofs-1) matrix
                // triple products for hessians.
                AssemblyContext::Vector3D gref_sum = {0.0, 0.0, 0.0};
                AssemblyContext::Matrix3x3 H_ref_sum{};

                for (LocalIndex j = 0; j < n_dofs; ++j) {
                    const Real coef = local_coeffs[static_cast<std::size_t>(j)];
                    const Real basis_val = field_bcache
                        ? field_bcache->scalarValue(static_cast<std::size_t>(j), static_cast<std::size_t>(q))
                        : values_at_pt[static_cast<std::size_t>(j)];
                    val += coef * basis_val;

                    if (need_gradients) {
                        const auto& gref = field_bcache
                            ? field_bcache->gradients[static_cast<std::size_t>(q)][static_cast<std::size_t>(j)]
                            : gradients_at_pt[static_cast<std::size_t>(j)];
                        for (int d = 0; d < dim; ++d) {
                            gref_sum[d] += coef * gref[static_cast<std::size_t>(d)];
                        }
                    }

                    if (need_hessians) {
                        const auto& hess_j = field_bcache
                            ? field_bcache->hessians[static_cast<std::size_t>(q)][static_cast<std::size_t>(j)]
                            : hessians_at_pt[static_cast<std::size_t>(j)];
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                H_ref_sum[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                                    coef * hess_j(static_cast<std::size_t>(r),
                                                   static_cast<std::size_t>(c));
                            }
                        }
                    }
                }

                scalar_values[static_cast<std::size_t>(q)] = val;

                if (need_gradients) {
                    // Single J_inv^T * gref_sum transform
                    AssemblyContext::Vector3D grad = {0.0, 0.0, 0.0};
                    for (int d1 = 0; d1 < dim; ++d1) {
                        for (int d2 = 0; d2 < dim; ++d2) {
                            grad[d1] += J_inv[static_cast<std::size_t>(d2)][static_cast<std::size_t>(d1)] *
                                        gref_sum[d2];
                        }
                    }
                    scalar_gradients[static_cast<std::size_t>(q)] = grad;
                }

                if (need_hessians) {
                    // Single J_inv^T * H_ref_sum * J_inv transform
                    AssemblyContext::Matrix3x3 H{};
                    for (int r = 0; r < dim; ++r) {
                        for (int c = 0; c < dim; ++c) {
                            Real sum = 0.0;
                            for (int a = 0; a < dim; ++a) {
                                for (int b = 0; b < dim; ++b) {
                                    sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                           H_ref_sum[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                           J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                }
                            }
                            H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                        }
                    }
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

                        gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                                     cell_dofs, prev_view, prev,
                                                     local_coeffs,
                                                     "StandardAssembler::populateFieldSolutionData", false);

                    // Values only (dt() needs just value history).
                    scalar_values.assign(static_cast<std::size_t>(n_qpts), 0.0);
                    for (LocalIndex q = 0; q < n_qpts; ++q) {
                        if (field_bcache) {
                            values_at_pt.resize(static_cast<std::size_t>(n_scalar_dofs));
                            for (LocalIndex j = 0; j < n_scalar_dofs; ++j) {
                                values_at_pt[static_cast<std::size_t>(j)] =
                                    field_bcache->scalarValue(static_cast<std::size_t>(j),
                                                             static_cast<std::size_t>(q));
                            }
                        } else {
                            const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                           qpts[static_cast<std::size_t>(q)][1],
                                                           qpts[static_cast<std::size_t>(q)][2]};
                            basis.evaluate_values(xi, values_at_pt);
                        }
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

        if (access->field_type == FieldType::Vector) {
            const int vd = access->value_dimension;
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

                auto& vec_values_at_pt = ws.vec_values_at_pt;
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

                            gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                                         cell_dofs, prev_view, prev,
                                                         local_coeffs,
                                                         "StandardAssembler::populateFieldSolutionData", false);

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

            // Use BasisCache for ProductSpace field basis evaluations.
            // Read-only lookup in pre-populated cache; do NOT insert on miss.
            const basis::BasisCacheEntry* field_bcache_ps = nullptr;
            if (cached_quad_rule_ &&
                static_cast<LocalIndex>(cached_quad_rule_->num_points()) == n_qpts) {
                for (const auto& fc : cached_field_bcache_) {
                    if (fc.basis == &basis && fc.gradients == need_gradients && fc.hessians == need_hessians) {
                        field_bcache_ps = fc.entry;
                        break;
                    }
                }
                // Cache miss: query the global thread-safe BasisCache singleton
                // but do NOT insert into cached_field_bcache_ (shared state).
                if (!field_bcache_ps) {
                    field_bcache_ps = &basis::BasisCache::instance().get_or_compute(
                        basis, *cached_quad_rule_, need_gradients, need_hessians);
                }
            }

            // Get inverse Jacobians span once (avoid per-QP accessor overhead).
            const auto ctx_inv_jacs_ps = context.inverseJacobians();

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                if (!field_bcache_ps) {
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
                }

                const auto& J_inv = ctx_inv_jacs_ps[cached_mapping_affine_ ? 0 : static_cast<std::size_t>(q)];
                AssemblyContext::Matrix3x3 J{};

                const auto q_base = static_cast<std::size_t>(q) * static_cast<std::size_t>(vd);
                for (int comp = 0; comp < vd; ++comp) {
                    Real val_c = 0.0;
                    // Accumulate in reference space, then transform once.
                    AssemblyContext::Vector3D gref_sum_c = {0.0, 0.0, 0.0};
                    AssemblyContext::Matrix3x3 H_ref_sum_c{};

                    const LocalIndex base = static_cast<LocalIndex>(comp) * dofs_per_component;
                    for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                        const LocalIndex jj = base + j;
                        const LocalIndex sj = static_cast<LocalIndex>(jj % n_scalar_dofs);
                        const Real coef = local_coeffs[static_cast<std::size_t>(jj)];
                        const Real basis_val = field_bcache_ps
                            ? field_bcache_ps->scalarValue(static_cast<std::size_t>(sj), static_cast<std::size_t>(q))
                            : values_at_pt[static_cast<std::size_t>(sj)];
                        val_c += coef * basis_val;

                        if (need_gradients) {
                            const auto& gref = field_bcache_ps
                                ? field_bcache_ps->gradients[static_cast<std::size_t>(q)][static_cast<std::size_t>(sj)]
                                : gradients_at_pt[static_cast<std::size_t>(sj)];
                            for (int d = 0; d < dim; ++d) {
                                gref_sum_c[d] += coef * gref[static_cast<std::size_t>(d)];
                            }
                        }

                        if (need_hessians) {
                            const auto& hess_sj = field_bcache_ps
                                ? field_bcache_ps->hessians[static_cast<std::size_t>(q)][static_cast<std::size_t>(sj)]
                                : hessians_at_pt[static_cast<std::size_t>(sj)];
                            for (int r = 0; r < dim; ++r) {
                                for (int c = 0; c < dim; ++c) {
                                    H_ref_sum_c[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                                        coef * hess_sj(static_cast<std::size_t>(r),
                                                        static_cast<std::size_t>(c));
                                }
                            }
                        }
                    }

                    vector_values[static_cast<std::size_t>(q)][static_cast<std::size_t>(comp)] = val_c;

                    if (need_gradients) {
                        // Single J_inv^T * gref_sum transform per component
                        for (int d1 = 0; d1 < dim; ++d1) {
                            Real sum = 0.0;
                            for (int d2 = 0; d2 < dim; ++d2) {
                                sum += J_inv[static_cast<std::size_t>(d2)][static_cast<std::size_t>(d1)] *
                                       gref_sum_c[d2];
                            }
                            J[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d1)] += sum;
                        }
                    }

                    if (need_hessians) {
                        AssemblyContext::Matrix3x3 H{};
                        for (int r = 0; r < dim; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                Real sum = 0.0;
                                for (int a = 0; a < dim; ++a) {
                                    for (int b = 0; b < dim; ++b) {
                                        sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                               H_ref_sum_c[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                               J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                                    }
                                }
                                H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                            }
                        }
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

                        gatherCellVectorCoefficients(cell_id, access->dof_map, access->dof_offset,
                                                     cell_dofs, prev_view, prev,
                                                     local_coeffs,
                                                     "StandardAssembler::populateFieldSolutionData", false);

                    // Values only (dt() needs just value history).
                    vector_values.assign(static_cast<std::size_t>(n_qpts), AssemblyContext::Vector3D{0.0, 0.0, 0.0});
                    for (LocalIndex q = 0; q < n_qpts; ++q) {
                        if (field_bcache_ps) {
                            values_at_pt.resize(static_cast<std::size_t>(n_scalar_dofs));
                            for (LocalIndex j = 0; j < n_scalar_dofs; ++j) {
                                values_at_pt[static_cast<std::size_t>(j)] =
                                    field_bcache_ps->scalarValue(static_cast<std::size_t>(j),
                                                                static_cast<std::size_t>(q));
                            }
                        } else {
                            const math::Vector<Real, 3> xi{qpts[static_cast<std::size_t>(q)][0],
                                                           qpts[static_cast<std::size_t>(q)][1],
                                                           qpts[static_cast<std::size_t>(q)][2]};
                            basis.evaluate_values(xi, values_at_pt);
                        }

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
	    // Resume JIT field table rebuild (one rebuild instead of per-setter).
	    context.resumeJITFieldTableRebuild();
	}

void StandardAssembler::insertLocal(
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view,
    std::span<const GlobalIndex> resolved_matrix_entries,
    std::span<const GlobalIndex> resolved_vector_entries)
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
        if (!resolved_matrix_entries.empty()) {
            matrix_view->addMatrixEntriesResolved(row_dofs, col_dofs,
                                                  resolved_matrix_entries,
                                                  output.local_matrix);
        } else {
            matrix_view->addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
        }
    }

    // Insert vector entries
    if (vector_view && output.has_vector) {
        if (!resolved_vector_entries.empty()) {
            vector_view->addVectorEntriesResolved(row_dofs, resolved_vector_entries,
                                                  output.local_vector);
        } else {
            vector_view->addVectorEntries(row_dofs, output.local_vector);
        }
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

    // NOTE: The redundant hasConstrainedDofs check that was here has been
    // removed. All callers (insertLocalForCell, colored parallel path) already
    // verify constraint status before calling this method, either via the
    // pre-computed cell_constrained_flags_ or via direct hasConstrainedDofs.

    // Use ConstraintDistributor for constrained assembly.
    // Buffered adapters collect expanded entries; flush eliminates per-entry
    // virtual dispatch overhead. Vector batching via addVectorEntries reduces
    // resolveEntriesCached calls from per-DOF to per-cell.

    // For matrix
    if (matrix_view && output.has_matrix && constraint_distributor_) {
        // Buffered matrix adapter: collects (row, col, value) triples,
        // passes setDiagonal through immediately.
        class BufferedMatrixOps : public constraints::IMatrixOperations {
        public:
            BufferedMatrixOps(GlobalSystemView& view,
                              std::vector<GlobalIndex>& rows,
                              std::vector<GlobalIndex>& cols,
                              std::vector<Real>& vals)
                : view_(view), rows_(rows), cols_(cols), vals_(vals)
            {
                rows_.clear();
                cols_.clear();
                vals_.clear();
            }

            void addValues(std::span<const GlobalIndex> rows,
                           std::span<const GlobalIndex> cols,
                           std::span<const double> values) override {
                // Dense block: expand to per-entry triples
                const auto nr = rows.size();
                const auto nc = cols.size();
                for (std::size_t i = 0; i < nr; ++i) {
                    for (std::size_t j = 0; j < nc; ++j) {
                        const auto v = values[i * nc + j];
                        if (v != 0.0) {
                            rows_.push_back(rows[i]);
                            cols_.push_back(cols[j]);
                            vals_.push_back(v);
                        }
                    }
                }
            }

            void addValue(GlobalIndex row, GlobalIndex col, double value) override {
                rows_.push_back(row);
                cols_.push_back(col);
                vals_.push_back(value);
            }

            void setDiagonal(GlobalIndex row, double value) override {
                // Dirichlet diagonal: pass through immediately (rare, ~3% of DOFs)
                view_.setDiagonal(row, value);
            }

            [[nodiscard]] GlobalIndex numRows() const override { return view_.numRows(); }
            [[nodiscard]] GlobalIndex numCols() const override { return view_.numCols(); }

            void flush() {
                // Flush individual scattered entries (constraint expansion
                // produces scattered master DOF entries, not dense blocks).
                for (std::size_t k = 0; k < rows_.size(); ++k) {
                    view_.addMatrixEntry(rows_[k], cols_[k], vals_[k]);
                }
            }

        private:
            GlobalSystemView& view_;
            std::vector<GlobalIndex>& rows_;
            std::vector<GlobalIndex>& cols_;
            std::vector<Real>& vals_;
        };

        // Buffered vector adapter: collects (index, value) pairs for addValue,
        // passes setValue/getValue through immediately.
        class BufferedVectorOps : public constraints::IVectorOperations {
        public:
            BufferedVectorOps(GlobalSystemView& view,
                              std::vector<GlobalIndex>& dofs,
                              std::vector<Real>& vals)
                : view_(view), dofs_(dofs), vals_(vals)
            {
                dofs_.clear();
                vals_.clear();
            }

            void addValues(std::span<const GlobalIndex> indices,
                           std::span<const double> values) override {
                for (std::size_t i = 0; i < indices.size(); ++i) {
                    dofs_.push_back(indices[i]);
                    vals_.push_back(values[i]);
                }
            }

            void addValue(GlobalIndex index, double value) override {
                dofs_.push_back(index);
                vals_.push_back(value);
            }

            void setValue(GlobalIndex index, double value) override {
                // Dirichlet enforcement: pass through immediately
                view_.addVectorEntry(index, value, AddMode::Insert);
            }

            [[nodiscard]] double getValue(GlobalIndex index) const override {
                return view_.getVectorEntry(index);
            }

            [[nodiscard]] GlobalIndex size() const override {
                return view_.numRows();
            }

            void flush() {
                if (!dofs_.empty()) {
                    view_.addVectorEntries(dofs_, vals_);
                }
            }

        private:
            GlobalSystemView& view_;
            std::vector<GlobalIndex>& dofs_;
            std::vector<Real>& vals_;
        };

        BufferedMatrixOps matrix_ops(*matrix_view,
                                     scratch_expanded_rows_,
                                     scratch_expanded_cols_,
                                     scratch_expanded_matrix_vals_);

        // When both matrix and vector are present, the distribution strategy depends
        // on whether we suppress the Dirichlet inhomogeneity correction:
        //  - Linear solves (suppress=false): Use joint distributeLocalToGlobal which
        //    applies the -K*g Dirichlet inhomogeneity correction to the RHS.
        //  - Newton solves (suppress=true):  Distribute independently because the
        //    residual R(u) is already evaluated at the constrained state and the
        //    -K*g correction would double-count the inhomogeneity.
        if (vector_view && output.has_vector && !suppress_constraint_inhomogeneity_) {
            BufferedVectorOps vector_ops(*vector_view,
                                         scratch_expanded_vec_dofs_,
                                         scratch_expanded_vec_vals_);
            constraint_distributor_->distributeLocalToGlobal(
                output.local_matrix, output.local_vector,
                row_dofs, col_dofs, matrix_ops, vector_ops);
            matrix_ops.flush();
            vector_ops.flush();
        } else {
            constraint_distributor_->distributeMatrixToGlobal(
                output.local_matrix, row_dofs, col_dofs, matrix_ops);
            matrix_ops.flush();
        }
    }

    // Vector-only distribution: either no matrix present, or suppress mode is active
    // (in which case matrix was already distributed above independently).
    if (vector_view && output.has_vector && constraint_distributor_ &&
        !(matrix_view && output.has_matrix && !suppress_constraint_inhomogeneity_)) {
        class BufferedVectorOps : public constraints::IVectorOperations {
        public:
            BufferedVectorOps(GlobalSystemView& view,
                              std::vector<GlobalIndex>& dofs,
                              std::vector<Real>& vals)
                : view_(view), dofs_(dofs), vals_(vals)
            {
                dofs_.clear();
                vals_.clear();
            }

            void addValues(std::span<const GlobalIndex> indices,
                           std::span<const double> values) override {
                for (std::size_t i = 0; i < indices.size(); ++i) {
                    dofs_.push_back(indices[i]);
                    vals_.push_back(values[i]);
                }
            }

            void addValue(GlobalIndex index, double value) override {
                dofs_.push_back(index);
                vals_.push_back(value);
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

            void flush() {
                if (!dofs_.empty()) {
                    view_.addVectorEntries(dofs_, vals_);
                }
            }

        private:
            GlobalSystemView& view_;
            std::vector<GlobalIndex>& dofs_;
            std::vector<Real>& vals_;
        };

        BufferedVectorOps vector_ops(*vector_view,
                                     scratch_expanded_vec_dofs_,
                                     scratch_expanded_vec_vals_);
        constraint_distributor_->distributeRhsToGlobal(
            output.local_vector, row_dofs, vector_ops);
        vector_ops.flush();
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

// ============================================================================
// Tier 3: Flat Cell Data Table
// ============================================================================

void StandardAssembler::ensureFlatCellCoords(const IMeshAccess& mesh) {
    if (flat_cell_coords_.valid && flat_cell_coords_.mesh == &mesh) {
        return;
    }
    const int dim = mesh.dimension();
    const auto n_cells = mesh.numCells();
    if (n_cells == 0) {
        flat_cell_coords_.valid = false;
        return;
    }

    // Determine nodes per cell from first cell type
    GlobalIndex first_cell = -1;
    mesh.forEachCell([&](GlobalIndex cid) {
        if (first_cell < 0) first_cell = cid;
    });
    if (first_cell < 0) {
        flat_cell_coords_.valid = false;
        return;
    }

    std::vector<std::array<Real, 3>> tmp_coords;
    mesh.getCellCoordinates(first_cell, tmp_coords);
    const int nodes_per_cell = static_cast<int>(tmp_coords.size());

    // Allocate flat array: n_cells * nodes_per_cell * 3
    flat_cell_coords_.coords.resize(
        static_cast<std::size_t>(n_cells) *
        static_cast<std::size_t>(nodes_per_cell) * 3u);
    flat_cell_coords_.dim = dim;
    flat_cell_coords_.nodes_per_cell = nodes_per_cell;

    // Fill from mesh
    mesh.forEachCell([&](GlobalIndex cid) {
        mesh.getCellCoordinates(cid, tmp_coords);
        const auto base = static_cast<std::size_t>(cid) *
                          static_cast<std::size_t>(nodes_per_cell) * 3u;
        for (int n = 0; n < nodes_per_cell; ++n) {
            flat_cell_coords_.coords[base + static_cast<std::size_t>(n) * 3u + 0u] = tmp_coords[static_cast<std::size_t>(n)][0];
            flat_cell_coords_.coords[base + static_cast<std::size_t>(n) * 3u + 1u] = tmp_coords[static_cast<std::size_t>(n)][1];
            flat_cell_coords_.coords[base + static_cast<std::size_t>(n) * 3u + 2u] = tmp_coords[static_cast<std::size_t>(n)][2];
        }
    });

    flat_cell_coords_.mesh = &mesh;
    flat_cell_coords_.valid = true;
}

// ============================================================================
// Tier 1: Cached Field Evaluation Recipes
// ============================================================================

void StandardAssembler::ensureFieldRecipes(
    const IMeshAccess& mesh,
    const std::vector<FieldRequirement>& requirements)
{
    if (cached_field_recipes_valid_ && !requirements.empty()) {
        // Check if recipes match current requirements
        bool match = (cached_field_recipes_.size() == requirements.size());
        if (match) {
            for (std::size_t i = 0; i < requirements.size(); ++i) {
                if (cached_field_recipes_[i].field_id != requirements[i].field) {
                    match = false;
                    break;
                }
            }
        }
        if (match) return;
    }

    ensureFieldAccessPlans(mesh);
    cached_field_recipes_.clear();
    cached_field_recipes_.reserve(requirements.size());

    for (const auto& req : requirements) {
        CachedFieldRecipe recipe;
        recipe.field_id = req.field;
        recipe.access = findFieldAccessPlan(req.field);
        if (!recipe.access || !recipe.access->space || !recipe.access->dof_map) {
            cached_field_recipes_.push_back(recipe);
            continue;
        }

        const auto& space = *recipe.access->space;
        recipe.is_product = recipe.access->is_product;
        recipe.field_type = recipe.access->field_type;
        recipe.value_dim = recipe.access->value_dimension;
        recipe.n_dofs = static_cast<LocalIndex>(space.dofs_per_element());

        // Get element for first cell to determine scalar DOF count
        GlobalIndex first_cell = -1;
        mesh.forEachCell([&](GlobalIndex cid) {
            if (first_cell < 0) first_cell = cid;
        });
        if (first_cell >= 0) {
            const ElementType cell_type = mesh.getCellType(first_cell);
            const auto& element = getElement(space, first_cell, cell_type);
            recipe.n_scalar_dofs = static_cast<LocalIndex>(element.num_dofs());
        }

        recipe.want_values = hasFlag(req.required, RequiredData::SolutionValues) ||
                             (req.required == RequiredData::None);
        recipe.want_gradients = hasFlag(req.required, RequiredData::SolutionGradients);
        recipe.want_hessians = hasFlag(req.required, RequiredData::SolutionHessians);
        recipe.want_laplacians = hasFlag(req.required, RequiredData::SolutionLaplacians);

        // Pre-cache basis cache entry
        if (cached_quad_rule_) {
            const bool need_grads = recipe.want_gradients;
            const bool need_hess = recipe.want_hessians || recipe.want_laplacians;
            if (first_cell >= 0) {
                const ElementType cell_type = mesh.getCellType(first_cell);
                const auto& element = getElement(space, first_cell, cell_type);
                recipe.bcache = &basis::BasisCache::instance().get_or_compute(
                    element.basis(), *cached_quad_rule_, need_grads, need_hess);
            }
        }

        cached_field_recipes_.push_back(recipe);
    }

    cached_field_recipes_valid_ = true;
}

void StandardAssembler::populateFieldSolutionDataFast(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const std::vector<FieldRequirement>& requirements,
    std::deque<CellCoefficientCacheEntry>* coefficient_cache,
    std::deque<CellFieldEvaluationCacheEntry>* field_eval_cache)
{
    // Use cached recipes to skip per-field lookups.
    // Falls back to the original path if recipes aren't valid.
    if (!cached_field_recipes_valid_) {
        populateFieldSolutionData(context, mesh, cell_id, requirements);
        return;
    }

    context.clearFieldSolutionData();
    if (requirements.empty()) return;

    std::vector<const CachedFieldRecipe*> matched_recipes;
    matched_recipes.reserve(requirements.size());
    for (const auto& req : requirements) {
        const auto it = std::find_if(
            cached_field_recipes_.begin(), cached_field_recipes_.end(),
            [&](const CachedFieldRecipe& recipe) { return recipe.field_id == req.field; });
        if (it == cached_field_recipes_.end() || !it->access || !it->access->space) {
            populateFieldSolutionData(context, mesh, cell_id, requirements);
            return;
        }
        matched_recipes.push_back(&*it);
    }

    context.suspendJITFieldTableRebuild();

    const auto n_qpts = context.numQuadraturePoints();
    const auto ctx_inv_jacs = context.inverseJacobians();

    auto& scalar_values = scratch_fsd_scalar_values_;
    auto& scalar_gradients = scratch_fsd_scalar_gradients_;
    auto& vector_values = scratch_fsd_vector_values_;
    auto& vector_jacobians = scratch_fsd_vector_jacobians_;
    const auto bindCachedCurrentField =
        [&](const CellFieldEvaluationCacheEntry& entry,
            const FieldRequirement& req,
            bool want_values,
            bool want_gradients) {
            if (entry.field_type == FieldType::Scalar) {
                context.setFieldSolutionScalar(
                    req.field,
                    want_values ? std::span<const Real>(entry.scalar_values) : std::span<const Real>{},
                    want_gradients ? std::span<const AssemblyContext::Vector3D>(entry.scalar_gradients)
                                   : std::span<const AssemblyContext::Vector3D>{});
            } else {
                context.setFieldSolutionVector(
                    req.field,
                    entry.value_dim,
                    want_values ? std::span<const AssemblyContext::Vector3D>(entry.vector_values)
                                : std::span<const AssemblyContext::Vector3D>{},
                    want_gradients ? std::span<const AssemblyContext::Matrix3x3>(entry.vector_jacobians)
                                   : std::span<const AssemblyContext::Matrix3x3>{});
            }
        };
    const auto bindCachedPreviousField =
        [&](const CellFieldEvaluationCacheEntry& entry,
            const FieldRequirement& req,
            int history_index) {
            if (entry.field_type == FieldType::Scalar) {
                context.setFieldPreviousSolutionScalarK(
                    req.field, history_index, std::span<const Real>(entry.scalar_values));
            } else {
                context.setFieldPreviousSolutionVectorK(
                    req.field,
                    history_index,
                    entry.value_dim,
                    std::span<const AssemblyContext::Vector3D>(entry.vector_values));
            }
        };
    const auto findCachedFieldEvaluation =
        [&](FieldId field_id, int history_index) -> CellFieldEvaluationCacheEntry* {
            if (field_eval_cache == nullptr) {
                return nullptr;
            }
            for (auto& entry : *field_eval_cache) {
                if (entry.field_id == field_id &&
                    entry.cell_id == cell_id &&
                    entry.history_index == history_index) {
                    return &entry;
                }
            }
            return nullptr;
        };

    for (std::size_t ri = 0; ri < requirements.size(); ++ri) {
        const auto& recipe = *matched_recipes[ri];
        const auto& req = requirements[ri];
        const bool want_values =
            hasFlag(req.required, RequiredData::SolutionValues) ||
            (req.required == RequiredData::None);
        const bool want_gradients = hasFlag(req.required, RequiredData::SolutionGradients);
        const bool want_hessians = hasFlag(req.required, RequiredData::SolutionHessians);
        const bool want_laplacians = hasFlag(req.required, RequiredData::SolutionLaplacians);

        if (!recipe.access || !recipe.access->space ||
            (want_gradients && !recipe.want_gradients) ||
            (want_hessians && !recipe.want_hessians) ||
            (want_laplacians && !recipe.want_laplacians) ||
            want_hessians || want_laplacians) {
            populateFieldSolutionData(context, mesh, cell_id, requirements);
            context.resumeJITFieldTableRebuild();
            return;
        }

        if (auto* cached_eval = findCachedFieldEvaluation(req.field, /*history_index=*/0)) {
            const bool compatible =
                cached_eval->field_type == recipe.field_type &&
                cached_eval->value_dim == recipe.value_dim &&
                (!want_values || cached_eval->has_values) &&
                (!want_gradients || cached_eval->has_gradients);
            if (compatible) {
                bindCachedCurrentField(*cached_eval, req, want_values, want_gradients);
                continue;
            }
        }

        const auto n_dofs = recipe.n_dofs;
        const auto n_scalar = recipe.n_scalar_dofs;

        // Gather DOF coefficients (direct, no virtual dispatch)
        const auto cell_dofs = getCellDofsFromTable(*recipe.access->dof_table, cell_id);
        const auto local_coeffs =
            (coefficient_cache != nullptr)
                ? gatherCachedCellVectorCoefficients(
                      *coefficient_cache,
                      mesh,
                      cell_id,
                      recipe.access->dof_map,
                      recipe.access->dof_offset,
                      recipe.access->space,
                      cell_dofs,
                      /*history_index=*/0,
                      /*localized_vector_basis=*/false,
                      "populateFieldSolutionDataFast")
                : [&]() -> std::span<const Real> {
                      auto& coeff_scratch = scratch_field_local_coeffs_;
                      coeff_scratch.resize(cell_dofs.size());
                      gatherCellVectorCoefficients(
                          cell_id, recipe.access->dof_map,
                          recipe.access->dof_offset,
                          cell_dofs, current_solution_view_,
                          current_solution_, coeff_scratch,
                          "populateFieldSolutionDataFast", false);
                      return std::span<const Real>(coeff_scratch);
                  }();

        // Use cached basis cache entry (no hash lookup)
        const auto* field_bcache = recipe.bcache;

        if (recipe.field_type == FieldType::Scalar && !recipe.is_product) {
            // Scalar field: values + optional gradients
            if (want_values) {
                scalar_values.assign(static_cast<std::size_t>(n_qpts), 0.0);
            } else {
                scalar_values.clear();
            }
            if (want_gradients) {
                scalar_gradients.assign(static_cast<std::size_t>(n_qpts),
                                        AssemblyContext::Vector3D{0.0, 0.0, 0.0});
            } else {
                scalar_gradients.clear();
            }

            if (field_bcache && !field_bcache->scalar_values.empty()) {
                // Fast path: use cached reference basis
                // Accumulate in reference space, then transform
                // scalar_values layout: [dof * n_qpts + qp]
                // gradients layout: [dof][qp] (vector of vectors)
                const auto& sv = field_bcache->scalar_values;
                const auto& sg = field_bcache->gradients;
                const bool have_grads = want_gradients && !sg.empty();

                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    Real val_sum = 0.0;
                    Real gref0 = 0.0, gref1 = 0.0, gref2 = 0.0;
                    for (LocalIndex i = 0; i < n_scalar; ++i) {
                        const auto c = local_coeffs[static_cast<std::size_t>(i)];
                        val_sum += c * sv[static_cast<std::size_t>(i) *
                                         static_cast<std::size_t>(n_qpts) +
                                         static_cast<std::size_t>(q)];
                        if (have_grads) {
                            const auto& gi = sg[static_cast<std::size_t>(i)]
                                               [static_cast<std::size_t>(q)];
                            gref0 += c * gi[0];
                            gref1 += c * gi[1];
                            gref2 += c * gi[2];
                        }
                    }
                    if (want_values) {
                        scalar_values[static_cast<std::size_t>(q)] = val_sum;
                    }
                    if (have_grads) {
                        const auto& Ji = ctx_inv_jacs[static_cast<std::size_t>(q)];
                        auto& g = scalar_gradients[static_cast<std::size_t>(q)];
                        g[0] = Ji[0][0] * gref0 + Ji[1][0] * gref1 + Ji[2][0] * gref2;
                        g[1] = Ji[0][1] * gref0 + Ji[1][1] * gref1 + Ji[2][1] * gref2;
                        g[2] = Ji[0][2] * gref0 + Ji[1][2] * gref1 + Ji[2][2] * gref2;
                    }
                }
            } else {
                // No basis cache: fall back to original
                populateFieldSolutionData(context, mesh, cell_id, requirements);
                context.resumeJITFieldTableRebuild();
                return;
            }

            if (field_eval_cache != nullptr) {
                auto* cached_eval = findCachedFieldEvaluation(req.field, /*history_index=*/0);
                if (cached_eval == nullptr) {
                    field_eval_cache->emplace_back();
                    cached_eval = &field_eval_cache->back();
                }
                cached_eval->field_id = req.field;
                cached_eval->cell_id = cell_id;
                cached_eval->history_index = 0;
                cached_eval->field_type = FieldType::Scalar;
                cached_eval->value_dim = 1;
                cached_eval->has_values = want_values;
                cached_eval->has_gradients = want_gradients;
                cached_eval->scalar_values.assign(scalar_values.begin(), scalar_values.end());
                cached_eval->scalar_gradients.assign(
                    scalar_gradients.begin(), scalar_gradients.end());
                cached_eval->vector_values.clear();
                cached_eval->vector_jacobians.clear();
                bindCachedCurrentField(*cached_eval, req, want_values, want_gradients);
            } else {
                context.setFieldSolutionScalar(
                    req.field,
                    scalar_values,
                    want_gradients ? std::span<const AssemblyContext::Vector3D>(scalar_gradients)
                                   : std::span<const AssemblyContext::Vector3D>{});
            }

        } else if (recipe.is_product) {
            // Vector (ProductSpace) field: values + optional jacobians
            const int vdim = recipe.value_dim;
            if (want_values) {
                vector_values.assign(static_cast<std::size_t>(n_qpts),
                                     AssemblyContext::Vector3D{0.0, 0.0, 0.0});
            } else {
                vector_values.clear();
            }
            if (want_gradients) {
                vector_jacobians.assign(static_cast<std::size_t>(n_qpts),
                                        AssemblyContext::Matrix3x3{});
            } else {
                vector_jacobians.clear();
            }

            if (field_bcache && !field_bcache->scalar_values.empty()) {
                const auto& sv = field_bcache->scalar_values;
                const auto& sg = field_bcache->gradients;
                const bool have_grads = want_gradients && !sg.empty();

                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    AssemblyContext::Vector3D v_sum{0.0, 0.0, 0.0};
                    AssemblyContext::Matrix3x3 jac_ref{};

                    for (LocalIndex si = 0; si < n_scalar; ++si) {
                        const auto phi = sv[static_cast<std::size_t>(si) *
                                           static_cast<std::size_t>(n_qpts) +
                                           static_cast<std::size_t>(q)];
                        Real gref0 = 0, gref1 = 0, gref2 = 0;
                        if (have_grads) {
                            const auto& gi = sg[static_cast<std::size_t>(si)]
                                               [static_cast<std::size_t>(q)];
                            gref0 = gi[0];
                            gref1 = gi[1];
                            gref2 = gi[2];
                        }

                        for (int comp = 0; comp < vdim; ++comp) {
                            const auto c = local_coeffs[
                                static_cast<std::size_t>(comp) *
                                static_cast<std::size_t>(n_scalar) +
                                static_cast<std::size_t>(si)];
                            v_sum[comp] += c * phi;
                            if (have_grads) {
                                jac_ref[comp][0] += c * gref0;
                                jac_ref[comp][1] += c * gref1;
                                jac_ref[comp][2] += c * gref2;
                            }
                        }
                    }

                    if (want_values) {
                        vector_values[static_cast<std::size_t>(q)] = v_sum;
                    }
                    if (have_grads) {
                        const auto& Ji = ctx_inv_jacs[static_cast<std::size_t>(q)];
                        auto& jac = vector_jacobians[static_cast<std::size_t>(q)];
                        for (int comp = 0; comp < vdim; ++comp) {
                            for (int d = 0; d < 3; ++d) {
                                jac[comp][d] = jac_ref[comp][0] * Ji[0][d]
                                             + jac_ref[comp][1] * Ji[1][d]
                                             + jac_ref[comp][2] * Ji[2][d];
                            }
                        }
                    }
                }
            } else {
                populateFieldSolutionData(context, mesh, cell_id, requirements);
                context.resumeJITFieldTableRebuild();
                return;
            }

            if (field_eval_cache != nullptr) {
                auto* cached_eval = findCachedFieldEvaluation(req.field, /*history_index=*/0);
                if (cached_eval == nullptr) {
                    field_eval_cache->emplace_back();
                    cached_eval = &field_eval_cache->back();
                }
                cached_eval->field_id = req.field;
                cached_eval->cell_id = cell_id;
                cached_eval->history_index = 0;
                cached_eval->field_type = recipe.field_type;
                cached_eval->value_dim = recipe.value_dim;
                cached_eval->has_values = want_values;
                cached_eval->has_gradients = want_gradients;
                cached_eval->vector_values.assign(vector_values.begin(), vector_values.end());
                cached_eval->vector_jacobians.assign(
                    vector_jacobians.begin(), vector_jacobians.end());
                cached_eval->scalar_values.clear();
                cached_eval->scalar_gradients.clear();
                bindCachedCurrentField(*cached_eval, req, want_values, want_gradients);
            } else {
                context.setFieldSolutionVector(
                    req.field,
                    recipe.value_dim,
                    vector_values,
                    want_gradients ? std::span<const AssemblyContext::Matrix3x3>(vector_jacobians)
                                   : std::span<const AssemblyContext::Matrix3x3>{});
            }

        } else {
            // Unsupported field type: fall back
            populateFieldSolutionData(context, mesh, cell_id, requirements);
            context.resumeJITFieldTableRebuild();
            return;
        }
    }

    // Previous solutions (time integration): evaluate previous-step field
    // values at QPs and bind via setFieldPreviousSolutionScalarK / VectorK.
    if (time_integration_ != nullptr) {
        int required_history = 0;
        if (time_integration_->dt1)
            required_history = std::max(required_history,
                                        time_integration_->dt1->requiredHistoryStates());
        if (time_integration_->dt2)
            required_history = std::max(required_history,
                                        time_integration_->dt2->requiredHistoryStates());
        for (const auto& s : time_integration_->dt_extra)
            if (s) required_history = std::max(required_history,
                                               s->requiredHistoryStates());

        if (required_history > 0) {
            auto& prev_coeffs = scratch_field_local_coeffs_;  // reuse scratch
            auto& prev_scalar_vals = scratch_fsd_scalar_values_;
            auto& prev_vec_vals = scratch_fsd_vector_values_;

            for (int k = 1; k <= required_history; ++k) {
                const auto& prev_sol = previous_solutions_[static_cast<std::size_t>(k - 1)];
                const auto* prev_view =
                    (static_cast<std::size_t>(k - 1) < previous_solution_views_.size())
                        ? previous_solution_views_[static_cast<std::size_t>(k - 1)]
                        : nullptr;

                for (std::size_t ri = 0; ri < requirements.size(); ++ri) {
                    const auto& recipe = *matched_recipes[ri];
                    if (!recipe.access || !recipe.bcache) continue;

                    if (auto* cached_eval =
                            findCachedFieldEvaluation(requirements[ri].field, k)) {
                        const bool compatible =
                            cached_eval->field_type == recipe.field_type &&
                            cached_eval->value_dim == recipe.value_dim &&
                            cached_eval->has_values;
                        if (compatible) {
                            bindCachedPreviousField(*cached_eval, requirements[ri], k);
                            continue;
                        }
                    }

                    const auto n_scalar = recipe.n_scalar_dofs;
                    const auto cell_dofs = getCellDofsFromTable(
                        *recipe.access->dof_table, cell_id);
                    const auto prev_coeff_span =
                        (coefficient_cache != nullptr)
                            ? gatherCachedCellVectorCoefficients(
                                  *coefficient_cache,
                                  mesh,
                                  cell_id,
                                  recipe.access->dof_map,
                                  recipe.access->dof_offset,
                                  recipe.access->space,
                                  cell_dofs,
                                  k,
                                  /*localized_vector_basis=*/false,
                                  "populateFieldSolutionDataFast:prev")
                            : [&]() -> std::span<const Real> {
                                  prev_coeffs.resize(cell_dofs.size());
                                  gatherCellVectorCoefficients(
                                      cell_id, recipe.access->dof_map,
                                      recipe.access->dof_offset, cell_dofs,
                                      prev_view, prev_sol, prev_coeffs,
                                      "populateFieldSolutionDataFast:prev", false);
                                  return std::span<const Real>(prev_coeffs);
                              }();

                    const auto& sv = recipe.bcache->scalar_values;

                    if (recipe.field_type == FieldType::Scalar && !recipe.is_product) {
                        prev_scalar_vals.resize(static_cast<std::size_t>(n_qpts));
                        for (LocalIndex q = 0; q < n_qpts; ++q) {
                            Real val = 0.0;
                            for (LocalIndex i = 0; i < n_scalar; ++i) {
                                val += prev_coeff_span[static_cast<std::size_t>(i)] *
                                       sv[static_cast<std::size_t>(i) *
                                          static_cast<std::size_t>(n_qpts) +
                                          static_cast<std::size_t>(q)];
                            }
                            prev_scalar_vals[static_cast<std::size_t>(q)] = val;
                        }
                        if (field_eval_cache != nullptr) {
                            auto* cached_eval =
                                findCachedFieldEvaluation(requirements[ri].field, k);
                            if (cached_eval == nullptr) {
                                field_eval_cache->emplace_back();
                                cached_eval = &field_eval_cache->back();
                            }
                            cached_eval->field_id = requirements[ri].field;
                            cached_eval->cell_id = cell_id;
                            cached_eval->history_index = k;
                            cached_eval->field_type = FieldType::Scalar;
                            cached_eval->value_dim = 1;
                            cached_eval->has_values = true;
                            cached_eval->has_gradients = false;
                            cached_eval->scalar_values.assign(
                                prev_scalar_vals.begin(), prev_scalar_vals.end());
                            cached_eval->scalar_gradients.clear();
                            cached_eval->vector_values.clear();
                            cached_eval->vector_jacobians.clear();
                            bindCachedPreviousField(*cached_eval, requirements[ri], k);
                        } else {
                            context.setFieldPreviousSolutionScalarK(
                                requirements[ri].field, k, prev_scalar_vals);
                        }
                    } else if (recipe.is_product) {
                        prev_vec_vals.resize(static_cast<std::size_t>(n_qpts));
                        for (LocalIndex q = 0; q < n_qpts; ++q) {
                            AssemblyContext::Vector3D v{0.0, 0.0, 0.0};
                            for (LocalIndex si = 0; si < n_scalar; ++si) {
                                const auto phi =
                                    sv[static_cast<std::size_t>(si) *
                                       static_cast<std::size_t>(n_qpts) +
                                       static_cast<std::size_t>(q)];
                                for (int c = 0; c < recipe.value_dim; ++c) {
                                    v[c] += prev_coeff_span[
                                        static_cast<std::size_t>(c) *
                                        static_cast<std::size_t>(n_scalar) +
                                        static_cast<std::size_t>(si)] * phi;
                                }
                            }
                            prev_vec_vals[static_cast<std::size_t>(q)] = v;
                        }
                        if (field_eval_cache != nullptr) {
                            auto* cached_eval =
                                findCachedFieldEvaluation(requirements[ri].field, k);
                            if (cached_eval == nullptr) {
                                field_eval_cache->emplace_back();
                                cached_eval = &field_eval_cache->back();
                            }
                            cached_eval->field_id = requirements[ri].field;
                            cached_eval->cell_id = cell_id;
                            cached_eval->history_index = k;
                            cached_eval->field_type = recipe.field_type;
                            cached_eval->value_dim = recipe.value_dim;
                            cached_eval->has_values = true;
                            cached_eval->has_gradients = false;
                            cached_eval->vector_values.assign(
                                prev_vec_vals.begin(), prev_vec_vals.end());
                            cached_eval->vector_jacobians.clear();
                            cached_eval->scalar_values.clear();
                            cached_eval->scalar_gradients.clear();
                            bindCachedPreviousField(*cached_eval, requirements[ri], k);
                        } else {
                            context.setFieldPreviousSolutionVectorK(
                                requirements[ri].field, k, recipe.value_dim,
                                prev_vec_vals);
                        }
                    }
                }
            }
        }
    }

    context.resumeJITFieldTableRebuild();
}

} // namespace assembly
} // namespace FE
} // namespace svmp
