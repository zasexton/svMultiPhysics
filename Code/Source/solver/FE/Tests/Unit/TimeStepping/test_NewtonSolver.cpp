/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryModelDSL.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Assembly/CutIntegrationContext.h"
#include "Constraints/SystemConstraint.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/StepController.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"
#include "TimeStepping/VSVO_BDF_Controller.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ts_test = svmp::FE::timestepping::test;

namespace {

class RejectFirstConvergedAttemptController final
    : public svmp::FE::timestepping::StepController {
public:
    [[nodiscard]] int maxRetries() const noexcept override { return 1; }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onAccepted(
        const svmp::FE::timestepping::StepAttemptInfo&) override
    {
        svmp::FE::timestepping::StepDecision decision;
        decision.accept = true;
        return decision;
    }

    [[nodiscard]] svmp::FE::timestepping::StepDecision
    onRejected(
        const svmp::FE::timestepping::StepAttemptInfo& info,
        svmp::FE::timestepping::StepRejectReason) override
    {
        svmp::FE::timestepping::StepDecision decision;
        decision.accept = false;
        decision.retry = info.attempt_index == 0;
        decision.next_dt = decision.retry ? info.dt : 0.0;
        decision.message = decision.retry
            ? "retry converged candidate"
            : "retry budget exhausted";
        return decision;
    }
};

// Test-only contract wrapper for mesh fixtures whose coordinates,
// connectivity, ownership, numbering, fields, and labels are immutable for
// their entire lifetime.  This makes the revision-domain promise explicit
// without teaching the generic lightweight mesh fixtures to claim tracking.
class ImmutableRevisionTrackedMeshAccess final
    : public svmp::FE::assembly::IMeshAccess {
public:
    explicit ImmutableRevisionTrackedMeshAccess(
        std::shared_ptr<svmp::FE::assembly::IMeshAccess> mesh)
        : mesh_(std::move(mesh))
    {
        if (!mesh_) {
            throw std::runtime_error(
                "ImmutableRevisionTrackedMeshAccess: mesh is null");
        }
    }

    [[nodiscard]] svmp::FE::GlobalIndex numCells() const override
    {
        return mesh_->numCells();
    }
    [[nodiscard]] svmp::FE::GlobalIndex numOwnedCells() const override
    {
        return mesh_->numOwnedCells();
    }
    [[nodiscard]] svmp::FE::GlobalIndex numVertices() const override
    {
        return mesh_->numVertices();
    }
    [[nodiscard]] svmp::FE::GlobalIndex numOwnedVertices() const override
    {
        return mesh_->numOwnedVertices();
    }
    [[nodiscard]] svmp::FE::GlobalIndex numBoundaryFaces() const override
    {
        return mesh_->numBoundaryFaces();
    }
    [[nodiscard]] svmp::FE::GlobalIndex numInteriorFaces() const override
    {
        return mesh_->numInteriorFaces();
    }
    [[nodiscard]] int dimension() const override
    {
        return mesh_->dimension();
    }

    [[nodiscard]] bool revisionTrackingAvailable() const override
    {
        return true;
    }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t numberingRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t labelRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t activeConfigurationEpoch() const override
    {
        return 0u;
    }
    [[nodiscard]] std::uint64_t coordinateConfigurationKey() const override
    {
        return 1u;
    }
    [[nodiscard]] bool cellIdsAreDense() const override
    {
        return mesh_->cellIdsAreDense();
    }

    [[nodiscard]] bool isOwnedCell(
        svmp::FE::GlobalIndex cell_id) const override
    {
        return mesh_->isOwnedCell(cell_id);
    }
    [[nodiscard]] svmp::FE::ElementType getCellType(
        svmp::FE::GlobalIndex cell_id) const override
    {
        return mesh_->getCellType(cell_id);
    }
    [[nodiscard]] int getCellGeometryOrder(
        svmp::FE::GlobalIndex cell_id) const override
    {
        return mesh_->getCellGeometryOrder(cell_id);
    }
    [[nodiscard]] int getCellDomainId(
        svmp::FE::GlobalIndex cell_id) const override
    {
        return mesh_->getCellDomainId(cell_id);
    }
    void getCellNodes(
        svmp::FE::GlobalIndex cell_id,
        std::vector<svmp::FE::GlobalIndex>& nodes) const override
    {
        mesh_->getCellNodes(cell_id, nodes);
    }
    [[nodiscard]] std::array<svmp::FE::Real, 3> getNodeCoordinates(
        svmp::FE::GlobalIndex node_id) const override
    {
        return mesh_->getNodeCoordinates(node_id);
    }
    void getCellCoordinates(
        svmp::FE::GlobalIndex cell_id,
        std::vector<std::array<svmp::FE::Real, 3>>& coordinates) const override
    {
        mesh_->getCellCoordinates(cell_id, coordinates);
    }
    [[nodiscard]] bool supportsCoordinateFrame(
        svmp::FE::assembly::CoordinateFrame frame) const override
    {
        return mesh_->supportsCoordinateFrame(frame);
    }
    void getCellCoordinates(
        svmp::FE::GlobalIndex cell_id,
        svmp::FE::assembly::CoordinateFrame frame,
        std::vector<std::array<svmp::FE::Real, 3>>& coordinates) const override
    {
        mesh_->getCellCoordinates(cell_id, frame, coordinates);
    }
    [[nodiscard]] svmp::FE::LocalIndex getLocalFaceIndex(
        svmp::FE::GlobalIndex face_id,
        svmp::FE::GlobalIndex cell_id) const override
    {
        return mesh_->getLocalFaceIndex(face_id, cell_id);
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        svmp::FE::GlobalIndex face_id) const override
    {
        return mesh_->getBoundaryFaceMarker(face_id);
    }
    [[nodiscard]] std::pair<svmp::FE::GlobalIndex, svmp::FE::GlobalIndex>
    getInteriorFaceCells(svmp::FE::GlobalIndex face_id) const override
    {
        return mesh_->getInteriorFaceCells(face_id);
    }
    void forEachCell(
        std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        mesh_->forEachCell(std::move(callback));
    }
    void forEachOwnedCell(
        std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        mesh_->forEachOwnedCell(std::move(callback));
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(svmp::FE::GlobalIndex, svmp::FE::GlobalIndex)>
            callback) const override
    {
        mesh_->forEachBoundaryFace(marker, std::move(callback));
    }
    void forEachInteriorFace(
        std::function<void(svmp::FE::GlobalIndex,
                           svmp::FE::GlobalIndex,
                           svmp::FE::GlobalIndex)> callback) const override
    {
        mesh_->forEachInteriorFace(std::move(callback));
    }

private:
    std::shared_ptr<const svmp::FE::assembly::IMeshAccess> mesh_{};
};

[[nodiscard]] svmp::FE::dofs::MeshTopologyInfo
twoTetraSharedFaceTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = 2;
    topology.n_vertices = 5;
    topology.dim = 3;
    topology.cell2vertex_offsets = {0, 4, 8};
    topology.cell2vertex_data = {0, 1, 2, 3, 1, 2, 3, 4};
    topology.vertex_gids = {0, 1, 2, 3, 4};
    topology.cell_gids = {0, 1};
    topology.cell_owner_ranks = {0, 0};
    return topology;
}

struct KernelCallCounts {
    int total{0};
    int matrix_only{0};
    int vector_only{0};
    int matrix_and_vector{0};
};

struct FreeSurfaceConservativeBalanceKernelCounts {
    std::array<KernelCallCounts, 6> operators{};
    KernelCallCounts pressure_representability_pair{};
};

class CountingKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    CountingKernel(std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner, KernelCallCounts* counts)
        : inner_(std::move(inner))
        , counts_(counts)
    {
        if (!inner_) {
            throw std::runtime_error("CountingKernel: inner is null");
        }
        if (!counts_) {
            throw std::runtime_error("CountingKernel: counts is null");
        }
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        return inner_->getRequiredData();
    }

    [[nodiscard]] std::vector<svmp::FE::assembly::FieldRequirement> fieldRequirements() const override
    {
        return inner_->fieldRequirements();
    }

    [[nodiscard]] svmp::FE::assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return inner_->materialStateSpec();
    }

    [[nodiscard]] std::vector<svmp::FE::params::Spec> parameterSpecs() const override
    {
        return inner_->parameterSpecs();
    }

    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override
    {
        return inner_->maxTemporalDerivativeOrder();
    }

    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override
    {
        return inner_->hasStateIndependentMatrix();
    }

    [[nodiscard]] bool hasCell() const noexcept override { return inner_->hasCell(); }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return inner_->hasBoundaryFace(); }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return inner_->hasInteriorFace(); }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        const bool want_matrix = output.has_matrix;
        const bool want_vector = output.has_vector;
        inner_->computeCell(ctx, output);
        const bool did_matrix = want_matrix || !output.local_matrix.empty();
        const bool did_vector = want_vector || !output.local_vector.empty();
        counts_->total += 1;
        if (did_matrix && did_vector) {
            counts_->matrix_and_vector += 1;
        } else if (did_matrix) {
            counts_->matrix_only += 1;
        } else if (did_vector) {
            counts_->vector_only += 1;
        }
    }

    void computeBoundaryFace(const svmp::FE::assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             svmp::FE::assembly::KernelOutput& output) override
    {
        inner_->computeBoundaryFace(ctx, boundary_marker, output);
    }

    void computeInteriorFace(const svmp::FE::assembly::AssemblyContext& ctx_minus,
                             const svmp::FE::assembly::AssemblyContext& ctx_plus,
                             svmp::FE::assembly::KernelOutput& output_minus,
                             svmp::FE::assembly::KernelOutput& output_plus,
                             svmp::FE::assembly::KernelOutput& coupling_minus_plus,
                             svmp::FE::assembly::KernelOutput& coupling_plus_minus) override
    {
        inner_->computeInteriorFace(ctx_minus,
                                    ctx_plus,
                                    output_minus,
                                    output_plus,
                                    coupling_minus_plus,
                                    coupling_plus_minus);
    }

    [[nodiscard]] std::string name() const override
    {
        return "Counting(" + inner_->name() + ")";
    }

private:
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner_{};
    KernelCallCounts* counts_{nullptr};
};

class ReplacementOnlyMatrix final
    : public svmp::FE::backends::GenericMatrix {
public:
    explicit ReplacementOnlyMatrix(
        std::unique_ptr<svmp::FE::backends::GenericMatrix> inner)
        : inner_(std::move(inner))
    {
        if (!inner_) {
            throw std::runtime_error(
                "ReplacementOnlyMatrix requires an inner matrix");
        }
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind()
        const noexcept override
    {
        return inner_->backendKind();
    }
    [[nodiscard]] svmp::FE::GlobalIndex numRows() const noexcept override
    {
        return inner_->numRows();
    }
    [[nodiscard]] svmp::FE::GlobalIndex numCols() const noexcept override
    {
        return inner_->numCols();
    }
    void zero() override { inner_->zero(); }
    void finalizeAssembly() override { inner_->finalizeAssembly(); }
    void mult(const svmp::FE::backends::GenericVector& x,
              svmp::FE::backends::GenericVector& y) const override
    {
        inner_->mult(x, y);
    }
    void multAdd(const svmp::FE::backends::GenericVector& x,
                 svmp::FE::backends::GenericVector& y) const override
    {
        inner_->multAdd(x, y);
    }
    [[nodiscard]] std::unique_ptr<svmp::FE::assembly::GlobalSystemView>
    createAssemblyView() override
    {
        return inner_->createAssemblyView();
    }
    [[nodiscard]] svmp::FE::Real getEntry(
        svmp::FE::GlobalIndex row,
        svmp::FE::GlobalIndex column) const override
    {
        return inner_->getEntry(row, column);
    }

    // Model a backend with immutable matrix storage. Newton must replace the
    // matrix, then rebuild every vector tied to that matrix's refreshed map.
    bool reinitFromPattern(
        const svmp::FE::sparsity::SparsityPattern&) override
    {
        return false;
    }

private:
    std::unique_ptr<svmp::FE::backends::GenericMatrix> inner_{};
};

class ReplacementOnlyMatrixFactory final
    : public svmp::FE::backends::BackendFactory {
public:
    ReplacementOnlyMatrixFactory()
        : inner_(svmp::FE::backends::BackendFactory::create(
              svmp::FE::backends::BackendKind::Eigen))
    {
        if (!inner_) {
            throw std::runtime_error(
                "ReplacementOnlyMatrixFactory requires Eigen");
        }
    }

    using BackendFactory::createMatrix;
    using BackendFactory::createVector;

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind()
        const noexcept override
    {
        return inner_->backendKind();
    }
    [[nodiscard]] std::unique_ptr<svmp::FE::backends::GenericMatrix>
    createMatrix(
        const svmp::FE::sparsity::SparsityPattern& pattern) const override
    {
        return std::make_unique<ReplacementOnlyMatrix>(
            inner_->createMatrix(pattern));
    }
    [[nodiscard]] std::unique_ptr<svmp::FE::backends::GenericVector>
    createVector(svmp::FE::GlobalIndex size) const override
    {
        return inner_->createVector(size);
    }
    [[nodiscard]] std::unique_ptr<svmp::FE::backends::LinearSolver>
    createLinearSolver(
        const svmp::FE::backends::SolverOptions& options) const override
    {
        return inner_->createLinearSolver(options);
    }

private:
    std::unique_ptr<svmp::FE::backends::BackendFactory> inner_{};
};

class AffineScalarCellKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    explicit AffineScalarCellKernel(svmp::FE::Real target)
        : target_(target)
    {
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        using svmp::FE::assembly::RequiredData;
        return RequiredData::IntegrationWeights |
               RequiredData::BasisValues |
               RequiredData::SolutionCoefficients;
    }

    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override { return true; }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        const auto n_test = ctx.numTestDofs();
        const auto n_trial = ctx.numTrialDofs();
        bool want_matrix = output.has_matrix || !output.local_matrix.empty();
        bool want_vector = output.has_vector || !output.local_vector.empty();
        if (!want_matrix && !want_vector) {
            want_matrix = true;
            want_vector = true;
        }
        output.reserve(n_test, n_trial, want_matrix, want_vector);
        output.clear();

        const auto coeffs = ctx.solutionCoefficients();
        if (want_vector && coeffs.size() < static_cast<std::size_t>(n_trial)) {
            throw std::runtime_error("AffineScalarCellKernel: missing solution coefficients");
        }

        for (svmp::FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            const auto w = ctx.integrationWeight(q);
            svmp::FE::Real uh = 0.0;
            if (want_vector) {
                for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                    uh += ctx.trialBasisValue(j, q) * coeffs[static_cast<std::size_t>(j)];
                }
            }
            for (svmp::FE::LocalIndex i = 0; i < n_test; ++i) {
                const auto vi = ctx.basisValue(i, q);
                if (want_vector) {
                    output.vectorEntry(i) += w * vi * (uh - target_);
                }
                if (want_matrix) {
                    for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                        output.matrixEntry(i, j) += w * vi * ctx.trialBasisValue(j, q);
                    }
                }
            }
        }
    }

    [[nodiscard]] std::string name() const override { return "AffineScalarCellKernel"; }

private:
    svmp::FE::Real target_{0.0};
};

class DiagonalCellMatrixKernel final
    : public svmp::FE::assembly::AssemblyKernel {
public:
    explicit DiagonalCellMatrixKernel(
        std::array<svmp::FE::Real, 4> diagonal)
        : diagonal_(std::move(diagonal))
    {
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData()
        const override
    {
        return svmp::FE::assembly::RequiredData::None;
    }

    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override
    {
        return true;
    }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        const auto n_test = ctx.numTestDofs();
        const auto n_trial = ctx.numTrialDofs();
        if (n_test != static_cast<svmp::FE::LocalIndex>(diagonal_.size()) ||
            n_trial != static_cast<svmp::FE::LocalIndex>(diagonal_.size())) {
            throw std::runtime_error(
                "DiagonalCellMatrixKernel requires four test and trial DOFs");
        }

        bool want_matrix = output.has_matrix || !output.local_matrix.empty();
        const bool want_vector =
            output.has_vector || !output.local_vector.empty();
        if (!want_matrix && !want_vector) {
            want_matrix = true;
        }
        output.reserve(n_test, n_trial, want_matrix, want_vector);
        output.clear();
        if (want_matrix) {
            for (svmp::FE::LocalIndex i = 0; i < n_test; ++i) {
                output.matrixEntry(i, i) =
                    diagonal_[static_cast<std::size_t>(i)];
            }
        }
    }

    [[nodiscard]] std::string name() const override
    {
        return "DiagonalCellMatrixKernel";
    }

private:
    std::array<svmp::FE::Real, 4> diagonal_{};
};

class RefreshedGeometryMeasureKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    RefreshedGeometryMeasureKernel(const double* measure, svmp::FE::Real target)
        : measure_(measure)
        , target_(target)
    {
        if (!measure_) {
            throw std::runtime_error("RefreshedGeometryMeasureKernel: measure is null");
        }
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        using svmp::FE::assembly::RequiredData;
        return RequiredData::IntegrationWeights |
               RequiredData::BasisValues |
               RequiredData::SolutionCoefficients;
    }

    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override { return false; }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        const auto n_test = ctx.numTestDofs();
        const auto n_trial = ctx.numTrialDofs();
        bool want_matrix = output.has_matrix || !output.local_matrix.empty();
        bool want_vector = output.has_vector || !output.local_vector.empty();
        if (!want_matrix && !want_vector) {
            want_matrix = true;
            want_vector = true;
        }
        output.reserve(n_test, n_trial, want_matrix, want_vector);
        output.clear();

        const auto coeffs = ctx.solutionCoefficients();
        if (want_vector && coeffs.size() < static_cast<std::size_t>(n_trial)) {
            throw std::runtime_error("RefreshedGeometryMeasureKernel: missing solution coefficients");
        }

        const auto measure = static_cast<svmp::FE::Real>(*measure_);
        for (svmp::FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            const auto w = ctx.integrationWeight(q);
            svmp::FE::Real uh = 0.0;
            if (want_vector) {
                for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                    uh += ctx.trialBasisValue(j, q) *
                          coeffs[static_cast<std::size_t>(j)];
                }
            }
            for (svmp::FE::LocalIndex i = 0; i < n_test; ++i) {
                const auto vi = ctx.basisValue(i, q);
                if (want_vector) {
                    output.vectorEntry(i) += w * vi * measure * (uh - target_);
                }
                if (want_matrix) {
                    for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                        output.matrixEntry(i, j) +=
                            w * vi * measure * ctx.trialBasisValue(j, q);
                    }
                }
            }
        }
    }

    [[nodiscard]] std::string name() const override
    {
        return "RefreshedGeometryMeasureKernel";
    }

private:
    const double* measure_{nullptr};
    svmp::FE::Real target_{0.0};
};

class RefreshedGeometryRootKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    RefreshedGeometryRootKernel(const double* measure, svmp::FE::Real target)
        : measure_(measure)
        , target_(target)
    {
        if (!measure_) {
            throw std::runtime_error("RefreshedGeometryRootKernel: measure is null");
        }
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        using svmp::FE::assembly::RequiredData;
        return RequiredData::IntegrationWeights |
               RequiredData::BasisValues |
               RequiredData::SolutionCoefficients;
    }

    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override
    {
        return false;
    }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        const auto n_test = ctx.numTestDofs();
        const auto n_trial = ctx.numTrialDofs();
        bool want_matrix = output.has_matrix || !output.local_matrix.empty();
        bool want_vector = output.has_vector || !output.local_vector.empty();
        if (!want_matrix && !want_vector) {
            want_matrix = true;
            want_vector = true;
        }
        output.reserve(n_test, n_trial, want_matrix, want_vector);
        output.clear();

        const auto coeffs = ctx.solutionCoefficients();
        if (want_vector &&
            coeffs.size() < static_cast<std::size_t>(n_trial)) {
            throw std::runtime_error(
                "RefreshedGeometryRootKernel: missing solution coefficients");
        }

        const auto measure = static_cast<svmp::FE::Real>(*measure_);
        for (svmp::FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            const auto w = ctx.integrationWeight(q);
            svmp::FE::Real uh = 0.0;
            if (want_vector) {
                for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                    uh += ctx.trialBasisValue(j, q) *
                          coeffs[static_cast<std::size_t>(j)];
                }
            }
            for (svmp::FE::LocalIndex i = 0; i < n_test; ++i) {
                const auto vi = ctx.basisValue(i, q);
                if (want_vector) {
                    output.vectorEntry(i) +=
                        w * vi * (measure * uh - target_);
                }
                if (want_matrix) {
                    for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                        output.matrixEntry(i, j) +=
                            w * vi * measure *
                            ctx.trialBasisValue(j, q);
                    }
                }
            }
        }
    }

    [[nodiscard]] std::string name() const override
    {
        return "RefreshedGeometryRootKernel";
    }

private:
    const double* measure_{nullptr};
    svmp::FE::Real target_{0.0};
};

class ScalingLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    ScalingLinearSolver(svmp::FE::backends::LinearSolver& inner, double scale)
        : inner_(inner)
        , scale_(scale)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        auto rep = inner_.solve(A, x, b);
        x.scale(static_cast<svmp::FE::Real>(scale_));
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    double scale_{1.0};
};

class SelectedDofScalingLinearSolver final
    : public svmp::FE::backends::LinearSolver {
public:
    SelectedDofScalingLinearSolver(
        svmp::FE::backends::LinearSolver& inner,
        svmp::FE::GlobalIndex selected_dof,
        double scale)
        : inner_(inner)
        , selected_dof_(selected_dof)
        , scale_(scale)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(
        const svmp::FE::backends::GenericMatrix& A,
        svmp::FE::backends::GenericVector& x,
        const svmp::FE::backends::GenericVector& b) override
    {
        auto rep = inner_.solve(A, x, b);
        auto values = x.localSpan();
        if (selected_dof_ < 0 ||
            static_cast<std::size_t>(selected_dof_) >= values.size()) {
            throw std::runtime_error(
                "SelectedDofScalingLinearSolver: selected DOF is not local");
        }
        values[static_cast<std::size_t>(selected_dof_)] *=
            static_cast<svmp::FE::Real>(scale_);
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    svmp::FE::GlobalIndex selected_dof_{0};
    double scale_{1.0};
};

class AlwaysFailLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit AlwaysFailLinearSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix&,
                                                          svmp::FE::backends::GenericVector&,
                                                          const svmp::FE::backends::GenericVector&) override
    {
        svmp::FE::backends::SolverReport rep;
        rep.converged = false;
        rep.iterations = 0;
        rep.message = "intentional test failure";
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class ScriptedFiniteInexactLinearSolver final
    : public svmp::FE::backends::LinearSolver {
public:
    struct Result {
        int iterations{1};
        svmp::FE::Real initial_residual_norm{1.0};
        svmp::FE::Real final_residual_norm{0.5};
        svmp::FE::Real relative_residual{0.5};
        bool zero_correction{false};
    };

    ScriptedFiniteInexactLinearSolver(
        svmp::FE::backends::LinearSolver& inner,
        Result result)
        : inner_(inner)
        , result_(result)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(
        const svmp::FE::backends::GenericMatrix& A,
        svmp::FE::backends::GenericVector& x,
        const svmp::FE::backends::GenericVector& b) override
    {
        ++solve_calls;
        auto report = inner_.solve(A, x, b);
        if (result_.zero_correction) {
            x.zero();
        }
        report.converged = false;
        report.numerical_breakdown = false;
        report.iterations = result_.iterations;
        report.initial_residual_norm = result_.initial_residual_norm;
        report.final_residual_norm = result_.final_residual_norm;
        report.relative_residual = result_.relative_residual;
        report.message = "scripted finite inexact result";
        return report;
    }

    int solve_calls{0};

private:
    svmp::FE::backends::LinearSolver& inner_;
    Result result_{};
};

class InvalidNumericalLinearSolver final
    : public svmp::FE::backends::LinearSolver {
public:
    enum class Mode {
        NonfiniteReport,
        NonfiniteCorrection
    };

    InvalidNumericalLinearSolver(
        svmp::FE::backends::LinearSolver& inner,
        Mode mode)
        : inner_(inner)
        , mode_(mode)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(
        const svmp::FE::backends::GenericMatrix&,
        svmp::FE::backends::GenericVector& x,
        const svmp::FE::backends::GenericVector&) override
    {
        ++solve_calls;
        x.zero();

        svmp::FE::backends::SolverReport rep;
        rep.iterations = 1;
        rep.initial_residual_norm = 1.0;
        if (mode_ == Mode::NonfiniteReport) {
            rep.converged = false;
            rep.final_residual_norm =
                std::numeric_limits<svmp::FE::Real>::infinity();
            rep.relative_residual =
                std::numeric_limits<svmp::FE::Real>::infinity();
            rep.message = "injected nonfinite report";
            return rep;
        }

        rep.converged = true;
        rep.final_residual_norm = 0.0;
        rep.relative_residual = 0.0;
        rep.message = "injected converged report with nonfinite correction";
        auto values = x.localSpan();
        if (!values.empty()) {
            values.front() =
                std::numeric_limits<svmp::FE::Real>::quiet_NaN();
        }
        return rep;
    }

    int solve_calls{0};

private:
    svmp::FE::backends::LinearSolver& inner_;
    Mode mode_{Mode::NonfiniteReport};
};

class OverflowingBorderedRecoveryLinearSolver final
    : public svmp::FE::backends::LinearSolver {
public:
    OverflowingBorderedRecoveryLinearSolver(
        svmp::FE::backends::LinearSolver& inner,
        svmp::FE::systems::FESystem& system)
        : inner_(inner)
        , system_(system)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(
        const svmp::FE::backends::GenericMatrix&,
        svmp::FE::backends::GenericVector& x,
        const svmp::FE::backends::GenericVector&) override
    {
        ++solve_calls;
        x.zero();

        const auto finite_huge = static_cast<svmp::FE::Real>(2.0) *
                                 std::sqrt(std::numeric_limits<svmp::FE::Real>::max());
        if (solve_calls == 1) {
            // The bordered structure has already selected the explicit
            // recovery path.  Replace its numerical values with an otherwise
            // valid finite system.  The scripted K^{-1}B columns and auxiliary
            // deltas below are each finite, but their product is not.
            auto& bordered = system_.borderedCoupling();
            if (!bordered.active || bordered.n_aux <= 0 ||
                bordered.n_field_dofs != static_cast<std::size_t>(x.size())) {
                throw std::runtime_error(
                    "OverflowingBorderedRecoveryLinearSolver: missing bordered system");
            }
            const auto na = static_cast<std::size_t>(bordered.n_aux);
            const auto nf = bordered.n_field_dofs;
            bordered.B.assign(nf * na, static_cast<svmp::FE::Real>(1.0));
            bordered.Ct.assign(na * nf, static_cast<svmp::FE::Real>(0.0));
            bordered.D.assign(na * na, static_cast<svmp::FE::Real>(0.0));
            bordered.g.assign(na, finite_huge);
            for (std::size_t i = 0; i < na; ++i) {
                bordered.D[i * na + i] = static_cast<svmp::FE::Real>(1.0);
            }
            bordered.direct_coupling_records.clear();
        } else {
            std::fill(x.localSpan().begin(), x.localSpan().end(), finite_huge);
        }

        svmp::FE::backends::SolverReport report;
        report.converged = true;
        report.iterations = 1;
        report.initial_residual_norm = 1.0;
        report.final_residual_norm = 0.0;
        report.relative_residual = 0.0;
        report.message = "scripted finite bordered recovery solve";
        return report;
    }

    int solve_calls{0};

private:
    svmp::FE::backends::LinearSolver& inner_;
    svmp::FE::systems::FESystem& system_;
};

class RecordingEffectiveTimeStepSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit RecordingEffectiveTimeStepSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        return inner_.solve(A, x, b);
    }

    void setEffectiveTimeStep(double dt_eff) override
    {
        effective_time_steps.push_back(dt_eff);
        inner_.setEffectiveTimeStep(dt_eff);
    }

    std::vector<double> effective_time_steps{};

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class FailOnceThenSolveRecordingMatrixSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit FailOnceThenSolveRecordingMatrixSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        ++solve_calls;
        observed_diagonals.push_back(static_cast<double>(A.getEntry(0, 0)));
        if (solve_calls == 1) {
            svmp::FE::backends::SolverReport rep;
            rep.converged = false;
            rep.iterations = 0;
            rep.message = "intentional first failure for PTC retry test";
            return rep;
        }
        return inner_.solve(A, x, b);
    }

    void setEffectiveTimeStep(double dt_eff) override
    {
        inner_.setEffectiveTimeStep(dt_eff);
    }

    int solve_calls{0};
    std::vector<double> observed_diagonals{};

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class RecordingRankOneSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit RecordingRankOneSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        return inner_.solve(A, x, b);
    }

    void setRankOneUpdates(std::span<const svmp::FE::backends::RankOneUpdate> updates) override
    {
        last_updates.assign(updates.begin(), updates.end());
        if (!updates.empty()) {
            saw_nonempty_rank_one_updates = true;
        }
    }

    void setReducedFieldUpdates(
        std::span<const svmp::FE::backends::ReducedFieldUpdate> updates) override
    {
        last_reduced_updates.assign(updates.begin(), updates.end());
        if (!updates.empty()) {
            saw_nonempty_reduced_updates = true;
        }
    }

    [[nodiscard]] bool supportsNativeRankOneUpdates() const noexcept override
    {
        return true;
    }

    [[nodiscard]] bool supportsNativeReducedFieldUpdates() const noexcept override
    {
        return true;
    }

    std::vector<svmp::FE::backends::RankOneUpdate> last_updates{};
    std::vector<svmp::FE::backends::ReducedFieldUpdate> last_reduced_updates{};
    bool saw_nonempty_rank_one_updates{false};
    bool saw_nonempty_reduced_updates{false};

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class RecordingSolveOptionsSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit RecordingSolveOptionsSolver(svmp::FE::backends::LinearSolver& inner,
                                         bool native_rank_one_support = false)
        : inner_(inner)
        , native_rank_one_support_(native_rank_one_support)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        last_set_options = options;
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        saw_solve = true;
        options_seen_in_solve = inner_.getOptions();
        return inner_.solve(A, x, b);
    }

    void setRankOneUpdates(std::span<const svmp::FE::backends::RankOneUpdate> updates) override
    {
        last_updates.assign(updates.begin(), updates.end());
        inner_.setRankOneUpdates(updates);
    }

    [[nodiscard]] bool supportsNativeRankOneUpdates() const noexcept override
    {
        return native_rank_one_support_;
    }

    bool saw_solve{false};
    std::optional<svmp::FE::backends::SolverOptions> last_set_options{};
    std::optional<svmp::FE::backends::SolverOptions> options_seen_in_solve{};
    std::vector<svmp::FE::backends::RankOneUpdate> last_updates{};

private:
    svmp::FE::backends::LinearSolver& inner_;
    bool native_rank_one_support_{false};
};

class ForceResidualReportLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    ForceResidualReportLinearSolver(svmp::FE::backends::LinearSolver& inner,
                                    double initial_residual_norm,
                                    double final_residual_norm,
                                    int forced_miss_calls = 1,
                                    bool native_rank_one_support = false)
        : inner_(inner)
        , initial_residual_norm_(initial_residual_norm)
        , final_residual_norm_(final_residual_norm)
        , forced_miss_calls_(forced_miss_calls)
        , native_rank_one_support_(native_rank_one_support)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        auto rep = inner_.solve(A, x, b);
        if (forced_miss_calls_ <= 0) {
            return rep;
        }
        --forced_miss_calls_;
        rep.converged = false;
        rep.initial_residual_norm = initial_residual_norm_;
        rep.final_residual_norm = final_residual_norm_;
        rep.relative_residual =
            final_residual_norm_ / std::max(initial_residual_norm_, 1e-30);
        rep.message = "synthetic strict-coupled miss";
        return rep;
    }

    void setRankOneUpdates(std::span<const svmp::FE::backends::RankOneUpdate> updates) override
    {
        inner_.setRankOneUpdates(updates);
    }

    [[nodiscard]] bool supportsNativeRankOneUpdates() const noexcept override
    {
        return native_rank_one_support_;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    double initial_residual_norm_{1.0};
    double final_residual_norm_{1.0};
    int forced_miss_calls_{1};
    bool native_rank_one_support_{false};
};

class ScopedEnvVar final {
public:
    ScopedEnvVar(const char* key, const char* value)
        : key_(key)
    {
        const char* prior = std::getenv(key_);
        if (prior != nullptr) {
            had_prior_ = true;
            prior_value_ = prior;
        }
        ::setenv(key_, value, 1);
    }

    ~ScopedEnvVar()
    {
        if (had_prior_) {
            ::setenv(key_, prior_value_.c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* key_{nullptr};
    bool had_prior_{false};
    std::string prior_value_{};
};

struct SharedFieldMpcState {
    bool reverse{false};
    double weight{1.0};
    double inhomogeneity{0.0};
};

class TimeDependentScalarDirichlet final
    : public svmp::FE::constraints::ISystemConstraint {
public:
    explicit TimeDependentScalarDirichlet(svmp::FE::FieldId field)
        : field_(field)
    {
    }

    void apply(
        const svmp::FE::systems::FESystem& system,
        svmp::FE::constraints::AffineConstraints& constraints) override
    {
        dof_ = system.fieldDofOffset(field_);
        constraints.addLine(dof_);
        constraints.setInhomogeneity(dof_, 0.0);
    }

    bool updateValues(
        const svmp::FE::systems::FESystem&,
        svmp::FE::constraints::AffineConstraints& constraints,
        double time,
        double) override
    {
        constraints.updateInhomogeneity(
            dof_, static_cast<svmp::FE::Real>(time));
        return true;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return true;
    }

    [[nodiscard]] svmp::FE::systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        return {};
    }

private:
    svmp::FE::FieldId field_{svmp::FE::INVALID_FIELD_ID};
    svmp::FE::GlobalIndex dof_{-1};
};

class SharedStateFieldMpc final
    : public svmp::FE::constraints::ISystemConstraint {
public:
    SharedStateFieldMpc(svmp::FE::FieldId first_field,
                        svmp::FE::FieldId second_field,
                        std::shared_ptr<SharedFieldMpcState> state)
        : first_field_(first_field)
        , second_field_(second_field)
        , state_(std::move(state))
    {
        if (!state_) {
            throw std::runtime_error(
                "SharedStateFieldMpc requires shared state");
        }
    }

    void apply(
        const svmp::FE::systems::FESystem& system,
        svmp::FE::constraints::AffineConstraints& constraints) override
    {
        const auto slave_field =
            state_->reverse ? second_field_ : first_field_;
        const auto master_field =
            state_->reverse ? first_field_ : second_field_;
        const auto slave = system.fieldDofOffset(slave_field);
        const auto master = system.fieldDofOffset(master_field);
        if (slave < 0 || master < 0 ||
            system.fieldDofHandler(slave_field).getNumDofs() != 1 ||
            system.fieldDofHandler(master_field).getNumDofs() != 1) {
            throw std::runtime_error(
                "SharedStateFieldMpc requires one DOF per field");
        }
        constraints.addLine(slave);
        constraints.addEntry(slave, master, state_->weight);
        constraints.setInhomogeneity(slave, state_->inhomogeneity);
    }

    bool updateValues(
        const svmp::FE::systems::FESystem&,
        svmp::FE::constraints::AffineConstraints&,
        double,
        double) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return false;
    }

    [[nodiscard]] svmp::FE::systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        return {};
    }

private:
    svmp::FE::FieldId first_field_{svmp::FE::INVALID_FIELD_ID};
    svmp::FE::FieldId second_field_{svmp::FE::INVALID_FIELD_ID};
    std::shared_ptr<SharedFieldMpcState> state_{};
};

struct CrossCellPressureMpcState {
    bool enabled{false};
};

class CrossCellPressureMpc final
    : public svmp::FE::constraints::ISystemConstraint {
public:
    CrossCellPressureMpc(
        svmp::FE::FieldId pressure_field,
        std::shared_ptr<CrossCellPressureMpcState> state)
        : pressure_field_(pressure_field)
        , state_(std::move(state))
    {
        if (!state_) {
            throw std::runtime_error(
                "CrossCellPressureMpc requires shared state");
        }
    }

    void apply(
        const svmp::FE::systems::FESystem& system,
        svmp::FE::constraints::AffineConstraints& constraints) override
    {
        if (!state_->enabled) {
            return;
        }
        const auto count =
            system.fieldDofHandler(pressure_field_).getNumDofs();
        if (count != 2) {
            throw std::runtime_error(
                "CrossCellPressureMpc requires two pressure DOFs");
        }
        const auto first = system.fieldDofOffset(pressure_field_);
        constraints.addLine(first);
        constraints.addEntry(first, first + 1, 1.0);
    }

    bool updateValues(
        const svmp::FE::systems::FESystem&,
        svmp::FE::constraints::AffineConstraints&,
        double,
        double) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return false;
    }

    [[nodiscard]] svmp::FE::systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        return {};
    }

private:
    svmp::FE::FieldId pressure_field_{svmp::FE::INVALID_FIELD_ID};
    std::shared_ptr<CrossCellPressureMpcState> state_{};
};

struct FixedPressureDirichletState {
    svmp::FE::Real value{0.0};
};

class FixedPressureDirichlet final
    : public svmp::FE::constraints::ISystemConstraint {
public:
    FixedPressureDirichlet(
        svmp::FE::FieldId pressure_field,
        std::shared_ptr<FixedPressureDirichletState> state)
        : pressure_field_(pressure_field)
        , state_(std::move(state))
    {
        if (!state_) {
            throw std::runtime_error(
                "FixedPressureDirichlet requires shared state");
        }
    }

    void apply(
        const svmp::FE::systems::FESystem& system,
        svmp::FE::constraints::AffineConstraints& constraints) override
    {
        if (system.fieldDofHandler(pressure_field_).getNumDofs() != 1) {
            throw std::runtime_error(
                "FixedPressureDirichlet requires one pressure DOF");
        }
        const auto pressure_dof =
            system.fieldDofOffset(pressure_field_);
        constraints.addLine(pressure_dof);
        constraints.setInhomogeneity(pressure_dof, state_->value);
    }

    bool updateValues(
        const svmp::FE::systems::FESystem&,
        svmp::FE::constraints::AffineConstraints&,
        double,
        double) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return false;
    }

    [[nodiscard]] svmp::FE::systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        return {};
    }

private:
    svmp::FE::FieldId pressure_field_{svmp::FE::INVALID_FIELD_ID};
    std::shared_ptr<FixedPressureDirichletState> state_{};
};

struct ScalarProblem {
    std::shared_ptr<svmp::FE::forms::test::SingleTetraMeshAccess> mesh{};
    std::shared_ptr<svmp::FE::spaces::L2Space> space{};
    std::unique_ptr<svmp::FE::systems::FESystem> sys{};
    svmp::FE::FieldId u_field{std::numeric_limits<svmp::FE::FieldId>::max()};
    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> integrator{};
    std::unique_ptr<svmp::FE::systems::TransientSystem> transient{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    std::unique_ptr<svmp::FE::backends::LinearSolver> linear{};
    svmp::FE::timestepping::TimeHistory history{};
};

struct TwoFieldProblem {
    std::shared_ptr<svmp::FE::forms::test::SingleTetraMeshAccess> mesh{};
    std::shared_ptr<svmp::FE::spaces::L2Space> space{};
    std::unique_ptr<svmp::FE::systems::FESystem> sys{};
    svmp::FE::FieldId selected_field{svmp::FE::INVALID_FIELD_ID};
    svmp::FE::FieldId dominant_field{svmp::FE::INVALID_FIELD_ID};
    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> integrator{};
    std::unique_ptr<svmp::FE::systems::TransientSystem> transient{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    std::unique_ptr<svmp::FE::backends::LinearSolver> linear{};
    svmp::FE::timestepping::TimeHistory history{};
};

struct PressureRepresentabilityProblem {
    std::shared_ptr<svmp::FE::assembly::IMeshAccess> mesh{};
    std::shared_ptr<svmp::FE::spaces::L2Space> space{};
    std::unique_ptr<svmp::FE::systems::FESystem> sys{};
    svmp::FE::FieldId velocity_field{svmp::FE::INVALID_FIELD_ID};
    svmp::FE::FieldId pressure_field{svmp::FE::INVALID_FIELD_ID};
    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> integrator{};
    std::unique_ptr<svmp::FE::systems::TransientSystem> transient{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    std::unique_ptr<svmp::FE::backends::LinearSolver> linear{};
    svmp::FE::timestepping::TimeHistory history{};
};

struct DirectCouplingProblem {
    std::shared_ptr<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess> mesh{};
    std::shared_ptr<svmp::FE::spaces::H1Space> space{};
    std::unique_ptr<svmp::FE::systems::FESystem> sys{};
    svmp::FE::FieldId u_field{std::numeric_limits<svmp::FE::FieldId>::max()};
    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> integrator{};
    std::unique_ptr<svmp::FE::systems::TransientSystem> transient{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    std::unique_ptr<svmp::FE::backends::LinearSolver> linear{};
    svmp::FE::timestepping::TimeHistory history{};
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::shared_ptr<svmp::Mesh> makeNativeSingleTetraMesh()
{
    auto base = std::make_shared<svmp::MeshBase>(3);
    const std::vector<svmp::real_t> reference_coordinates{
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const std::vector<svmp::offset_t> offsets{0, 4};
    const std::vector<svmp::index_t> connectivity{0, 1, 2, 3};
    const std::vector<svmp::CellShape> shapes{
        {svmp::CellFamily::Tetra, 4, 1},
    };
    base->build_from_arrays(
        3, reference_coordinates, offsets, connectivity, shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}
#endif

template <typename BuildForm>
[[nodiscard]] ScalarProblem makeScalarProblem(BuildForm build_form,
                                              double dt,
                                              const std::vector<svmp::FE::Real>& u0,
                                              KernelCallCounts* counts = nullptr,
                                              FreeSurfaceConservativeBalanceKernelCounts*
                                                  balance_counts = nullptr)
{
    ScalarProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*p.space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*p.space, "v");
    const auto form = build_form(u, v);

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto base_kernel =
        std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> kernel = base_kernel;
    if (counts != nullptr) {
        kernel = std::make_shared<CountingKernel>(kernel, counts);
    }
    p.sys->addCellKernel("op", p.u_field, p.u_field, kernel);

    if (balance_counts != nullptr) {
        constexpr std::array<const char*, 6> diagnostic_ops{
            "equations_diagnostic_ns_free_surface_pressure_virtual_work",
            "equations_diagnostic_ns_free_surface_surface_energy_virtual_work",
            "equations_diagnostic_ns_free_surface_gravitational_potential_virtual_work",
            "equations_diagnostic_ns_free_surface_physical_potential_virtual_work",
            "equations_diagnostic_ns_free_surface_pressure_representability_load_virtual_work",
            "equations_diagnostic_ns_free_surface_conservative_balance",
        };
        const std::array<svmp::FE::forms::FormExpr, 6> diagnostic_forms{
            (svmp::FE::forms::FormExpr::constant(2.0) * u * v).dx(),
            (svmp::FE::forms::FormExpr::constant(-0.5) * u * v).dx(),
            (svmp::FE::forms::FormExpr::constant(0.0) * u * v).dx(),
            (svmp::FE::forms::FormExpr::constant(-0.5) * u * v).dx(),
            (svmp::FE::forms::FormExpr::constant(-0.5) * u * v).dx(),
            (svmp::FE::forms::FormExpr::constant(1.5) * u * v).dx(),
        };
        for (std::size_t i = 0; i < diagnostic_ops.size(); ++i) {
            p.sys->addOperator(diagnostic_ops[i]);
            auto diagnostic_ir = compiler.compileResidual(diagnostic_forms[i]);
            auto diagnostic_kernel =
                std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
                    std::move(diagnostic_ir),
                    svmp::FE::forms::ADMode::Forward);
            p.sys->addCellKernel(
                diagnostic_ops[i],
                p.u_field,
                p.u_field,
                std::make_shared<CountingKernel>(
                    std::move(diagnostic_kernel),
                    &balance_counts->operators[i]));
        }
    }

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(*p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error("ScalarProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("ScalarProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("ScalarProblem u0 size mismatch");
    }
    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] PressureRepresentabilityProblem
makePressureRepresentabilityProblem(
    FreeSurfaceConservativeBalanceKernelCounts& counts,
    bool two_cells = false,
    std::shared_ptr<CrossCellPressureMpcState> pressure_mpc_state = {},
    std::shared_ptr<FixedPressureDirichletState>
        pressure_dirichlet_state = {},
    svmp::FE::Real entry_velocity = svmp::FE::Real{1.0},
    std::optional<std::array<svmp::FE::Real, 4>>
        pressure_pair_diagonal = std::nullopt,
    bool track_immutable_mesh_revisions = false,
    bool install_symbolic_cut_volume_pair = false,
    svmp::FE::Real production_pressure_target = svmp::FE::Real{0.0},
    svmp::FE::Real entry_pressure_baseline = svmp::FE::Real{0.0},
    bool use_native_mesh = false,
    svmp::FE::Real surface_energy_scale = svmp::FE::Real{1.0},
    svmp::FE::Real gravitational_potential_scale = svmp::FE::Real{0.0},
    svmp::FE::Real prescribed_external_pressure_scale =
        svmp::FE::Real{0.0})
{
    using svmp::FE::forms::FormExpr;

    PressureRepresentabilityProblem p;
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(
        svmp::FE::ElementType::Tetra4,
        pressure_pair_diagonal.has_value() ? /*order=*/1 : /*order=*/0);
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (use_native_mesh) {
        if (two_cells) {
            throw std::runtime_error(
                "native pressure fixture supports one tetrahedron");
        }
        p.sys = std::make_unique<svmp::FE::systems::FESystem>(
            makeNativeSingleTetraMesh());
    } else
#else
    (void)use_native_mesh;
#endif
    {
        auto base_mesh = two_cells
                             ? std::static_pointer_cast<
                                   svmp::FE::assembly::IMeshAccess>(
                                   std::make_shared<
                                       svmp::FE::forms::test::
                                           TwoTetraSharedFaceMeshAccess>())
                             : std::static_pointer_cast<
                                   svmp::FE::assembly::IMeshAccess>(
                                   std::make_shared<
                                       svmp::FE::forms::test::
                                           SingleTetraMeshAccess>());
        p.mesh = track_immutable_mesh_revisions
                     ? std::static_pointer_cast<
                           svmp::FE::assembly::IMeshAccess>(
                           std::make_shared<
                               ImmutableRevisionTrackedMeshAccess>(
                               std::move(base_mesh)))
                     : std::move(base_mesh);
        p.sys =
            std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    }
    p.velocity_field = p.sys->addField(
        svmp::FE::systems::FieldSpec{
            .name = "synthetic_velocity",
            .space = p.space,
            .components = 1});
    p.pressure_field = p.sys->addField(
        svmp::FE::systems::FieldSpec{
            .name = "synthetic_pressure",
            .space = p.space,
            .components = 1});
    if (pressure_mpc_state) {
        p.sys->addSystemConstraint(
            std::make_unique<CrossCellPressureMpc>(
                p.pressure_field, pressure_mpc_state));
    }
    if (pressure_dirichlet_state) {
        p.sys->addSystemConstraint(
            std::make_unique<FixedPressureDirichlet>(
                p.pressure_field, pressure_dirichlet_state));
    }

    const auto u = FormExpr::trialFunction(*p.space, "u");
    const auto pressure = FormExpr::trialFunction(*p.space, "p");
    const auto v = FormExpr::testFunction(*p.space, "v");
    const auto q = FormExpr::testFunction(*p.space, "q");
    svmp::FE::forms::FormCompiler compiler;
    const auto make_kernel = [&](const FormExpr& form) {
        auto ir = compiler.compileResidual(form);
        return std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
            std::move(ir), svmp::FE::forms::ADMode::Forward);
    };

    // A nonsingular, zero-at-entry production system lets the Newton solve
    // stop after its initial residual while still executing all diagnostics.
    // The entry velocity is one, which also gives the diagnostic residual
    // forms a real TrialFunction dependency (the form compiler deliberately
    // rejects a standalone constant-times-test residual).
    p.sys->addOperator("op");
    p.sys->addCellKernel(
        "op",
        p.velocity_field,
        p.velocity_field,
        make_kernel(
            ((u - FormExpr::constant(entry_velocity)) * v).dx()));
    p.sys->addCellKernel(
        "op",
        p.pressure_field,
        p.pressure_field,
        make_kernel(
            ((pressure - FormExpr::constant(production_pressure_target)) * q)
                .dx()));

    constexpr std::array<const char*, 6> vector_ops{
        "equations_diagnostic_ns_free_surface_pressure_virtual_work",
        "equations_diagnostic_ns_free_surface_surface_energy_virtual_work",
        "equations_diagnostic_ns_free_surface_gravitational_potential_virtual_work",
        "equations_diagnostic_ns_free_surface_physical_potential_virtual_work",
        "equations_diagnostic_ns_free_surface_pressure_representability_load_virtual_work",
        "equations_diagnostic_ns_free_surface_conservative_balance",
    };
    // On the unit tetra, a constant P1 field satisfies int(v_i)=1/24.
    // The finite-precision LSQR regression therefore uses factors 12 and 24
    // so its physical-potential load has controlled unit-scale coefficients.
    const svmp::FE::Real pressure_scale =
        pressure_pair_diagonal.has_value() ? 12.0 : 0.5;
    const svmp::FE::Real scaled_surface_energy =
        (pressure_pair_diagonal.has_value() ? 24.0 : 2.0) *
        surface_energy_scale;
    const svmp::FE::Real scaled_gravitational_potential =
        (pressure_pair_diagonal.has_value() ? 24.0 : 2.0) *
        gravitational_potential_scale;
    const svmp::FE::Real scaled_prescribed_external_pressure =
        (pressure_pair_diagonal.has_value() ? 24.0 : 2.0) *
        prescribed_external_pressure_scale;
    const svmp::FE::Real scaled_physical_potential =
        scaled_surface_energy + scaled_gravitational_potential;
    const std::array<svmp::FE::Real, 6> vector_scales{
        pressure_scale + scaled_prescribed_external_pressure,
        scaled_surface_energy,
        scaled_gravitational_potential,
        scaled_physical_potential,
        scaled_prescribed_external_pressure + scaled_physical_potential,
        pressure_scale + scaled_prescribed_external_pressure +
            scaled_physical_potential};
    const std::array<FormExpr, 6> vector_forms{
        (FormExpr::constant(vector_scales[0]) * u * v).dx(),
        (FormExpr::constant(vector_scales[1]) * u * v).dx(),
        (FormExpr::constant(vector_scales[2]) * u * v).dx(),
        (FormExpr::constant(vector_scales[3]) * u * v).dx(),
        (FormExpr::constant(vector_scales[4]) * u * v).dx(),
        (FormExpr::constant(vector_scales[5]) * u * v).dx(),
    };
    for (std::size_t i = 0; i < vector_ops.size(); ++i) {
        p.sys->addOperator(vector_ops[i]);
        p.sys->addCellKernel(
            vector_ops[i],
            p.velocity_field,
            p.velocity_field,
            std::make_shared<CountingKernel>(
                make_kernel(vector_forms[i]), &counts.operators[i]));
    }

    constexpr const char* pair_op =
        "equations_diagnostic_ns_free_surface_pressure_representability_pair";
    p.sys->addOperator(pair_op);
    if (install_symbolic_cut_volume_pair) {
        // Use the real mixed symbolic installer and cut-volume dispatch used
        // by the production pressure/velocity adjoint-pair diagnostic.  The
        // scalar mass analogue keeps this Newton cache fixture compact; the
        // FormsInstaller regression separately covers the vector-divergence
        // expression exactly.
        constexpr int marker = 812;
        const auto velocity_state = FormExpr::stateField(
            p.velocity_field, *p.space, "u");
        const auto pressure_state = FormExpr::stateField(
            p.pressure_field, *p.space, "p");
        const auto velocity_test = FormExpr::testFunction(
            p.velocity_field, *p.space, "v");
        const auto pressure_test = FormExpr::testFunction(
            p.pressure_field, *p.space, "q");
        const auto pair_residual =
            (-pressure_state * velocity_test)
                .dCutVolume(
                    marker,
                    svmp::FE::forms::CutVolumeSide::Negative) -
            (pressure_test * velocity_state)
                .dCutVolume(
                    marker,
                    svmp::FE::forms::CutVolumeSide::Negative);
        svmp::FE::systems::FormInstallOptions install;
        install.compiler_options.use_symbolic_tangent = true;
        install.compiler_options.jit.enable = false;
        (void)svmp::FE::systems::installFormulation(
            *p.sys,
            pair_op,
            {p.velocity_field, p.pressure_field},
            pair_residual,
            install);

        auto cut_context =
            std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
        svmp::FE::geometry::CutQuadratureRule rule;
        rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
        rule.side = svmp::FE::geometry::CutIntegrationSide::Negative;
        rule.parent_measure = svmp::FE::Real{1.0} / svmp::FE::Real{6.0};
        rule.measure = svmp::FE::Real{1.0} / svmp::FE::Real{12.0};
        rule.volume_fraction = svmp::FE::Real{0.5};
        rule.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
        rule.provenance.parent_entity = 0;
        rule.provenance.marker = marker;
        rule.points.push_back(svmp::FE::geometry::CutQuadraturePoint{
            .point = {{svmp::FE::Real{0.25},
                       svmp::FE::Real{0.25},
                       svmp::FE::Real{0.25}}},
            .weight = rule.measure,
        });
        svmp::FE::assembly::CutCellAssemblyMetadata metadata;
        metadata.parent_entity = 0;
        metadata.side = svmp::FE::geometry::CutIntegrationSide::Negative;
        metadata.volume_fraction = rule.volume_fraction;
        cut_context->addGeneratedVolumeRule(marker, metadata, rule);
        p.sys->setCutIntegrationContext(std::move(cut_context));
    } else if (pressure_pair_diagonal.has_value()) {
        // A symmetric [0,G;G^T,0] pair with prescribed singular values gives
        // a deterministic ill-conditioned LSQR problem without forming G^T G.
        p.sys->addCellKernel(
            pair_op,
            p.velocity_field,
            p.pressure_field,
            std::make_shared<CountingKernel>(
                std::make_shared<DiagonalCellMatrixKernel>(
                    *pressure_pair_diagonal),
                &counts.pressure_representability_pair));
        p.sys->addCellKernel(
            pair_op,
            p.pressure_field,
            p.velocity_field,
            std::make_shared<CountingKernel>(
                std::make_shared<DiagonalCellMatrixKernel>(
                    *pressure_pair_diagonal),
                &counts.pressure_representability_pair));
    } else {
        // Synthetic analogue of -p div(v) - q div(u).  Both off-diagonal mass
        // blocks are -M; the exact least-squares pressure for f=2 M is p=2.
        p.sys->addCellKernel(
            pair_op,
            p.velocity_field,
            p.pressure_field,
            std::make_shared<CountingKernel>(
                make_kernel((-pressure * v).dx()),
                &counts.pressure_representability_pair));
        p.sys->addCellKernel(
            pair_op,
            p.pressure_field,
            p.velocity_field,
            std::make_shared<CountingKernel>(
                make_kernel((-u * q).dx()),
                &counts.pressure_representability_pair));
    }

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override =
        two_cells ? twoTetraSharedFaceTopology()
                  : ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(
        *p.sys, p.integrator);
    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error(
            "PressureRepresentabilityProblem requires the Eigen backend");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error(
            "PressureRepresentabilityProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    p.history = svmp::FE::timestepping::TimeHistory::allocate(
        *p.factory, n_dofs);
    p.history.setDt(0.1);
    p.history.setPrevDt(0.1);
    std::vector<svmp::FE::Real> entry_state(
        static_cast<std::size_t>(n_dofs), svmp::FE::Real{0.0});
    const auto velocity_begin = p.sys->fieldDofOffset(p.velocity_field);
    const auto velocity_count =
        p.sys->fieldDofHandler(p.velocity_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < velocity_count; ++i) {
        entry_state[static_cast<std::size_t>(velocity_begin + i)] =
            entry_velocity;
    }
    const auto pressure_begin = p.sys->fieldDofOffset(p.pressure_field);
    const auto pressure_count =
        p.sys->fieldDofHandler(p.pressure_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        entry_state[static_cast<std::size_t>(pressure_begin + i)] =
            entry_pressure_baseline;
    }
    ts_test::setVectorByDof(p.history.uPrev(), entry_state);
    ts_test::setVectorByDof(p.history.uPrev2(), entry_state);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] ScalarProblem makeAffineScalarProblem(double target,
                                                    double dt,
                                                    const std::vector<svmp::FE::Real>& u0,
                                                    KernelCallCounts* counts,
                                                    bool add_time_dependent_dirichlet = false)
{
    ScalarProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> kernel =
        std::make_shared<AffineScalarCellKernel>(static_cast<svmp::FE::Real>(target));
    if (counts != nullptr) {
        kernel = std::make_shared<CountingKernel>(kernel, counts);
    }
    p.sys->addCellKernel("op", p.u_field, p.u_field, kernel);
    if (add_time_dependent_dirichlet) {
        p.sys->addSystemConstraint(
            std::make_unique<TimeDependentScalarDirichlet>(p.u_field));
    }

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(*p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error("ScalarProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("ScalarProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("ScalarProblem u0 size mismatch");
    }
    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] TwoFieldProblem makeTwoFieldAffineProblem(
    double selected_target,
    double dominant_target,
    double dt,
    std::shared_ptr<SharedFieldMpcState> constraint_state = {})
{
    TwoFieldProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(
        svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.selected_field = p.sys->addField(svmp::FE::systems::FieldSpec{
        .name = "selected", .space = p.space, .components = 1});
    p.dominant_field = p.sys->addField(svmp::FE::systems::FieldSpec{
        .name = "dominant", .space = p.space, .components = 1});
    p.sys->addOperator("op");
    p.sys->addCellKernel(
        "op",
        p.selected_field,
        p.selected_field,
        std::make_shared<AffineScalarCellKernel>(
            static_cast<svmp::FE::Real>(selected_target)));
    p.sys->addCellKernel(
        "op",
        p.dominant_field,
        p.dominant_field,
        std::make_shared<AffineScalarCellKernel>(
            static_cast<svmp::FE::Real>(dominant_target)));
    if (constraint_state) {
        p.sys->addSystemConstraint(
            std::make_unique<SharedStateFieldMpc>(
                p.selected_field,
                p.dominant_field,
                std::move(constraint_state)));
    }

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(
        *p.sys, p.integrator);
    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error(
            "TwoFieldProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error(
            "TwoFieldProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    p.history =
        svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    p.history.uPrev().zero();
    p.history.uPrev2().zero();
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] TwoFieldProblem makeTwoFieldHistorySensitiveMpcProblem(
    double dt,
    std::shared_ptr<SharedFieldMpcState> constraint_state)
{
    TwoFieldProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(
        svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.selected_field = p.sys->addField(svmp::FE::systems::FieldSpec{
        .name = "selected", .space = p.space, .components = 1});
    p.dominant_field = p.sys->addField(svmp::FE::systems::FieldSpec{
        .name = "dominant", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    const auto selected =
        svmp::FE::forms::FormExpr::trialFunction(*p.space, "selected");
    const auto selected_test =
        svmp::FE::forms::FormExpr::testFunction(*p.space, "selected_test");
    const auto dominant =
        svmp::FE::forms::FormExpr::trialFunction(*p.space, "dominant");
    const auto dominant_test =
        svmp::FE::forms::FormExpr::testFunction(*p.space, "dominant_test");
    const auto selected_form =
        ((svmp::FE::forms::dt(selected) + selected -
          svmp::FE::forms::FormExpr::constant(2.0)) *
         selected_test)
            .dx();
    const auto dominant_form =
        ((dominant - svmp::FE::forms::FormExpr::constant(1.0)) *
         dominant_test)
            .dx();

    svmp::FE::forms::FormCompiler compiler;
    auto selected_ir = compiler.compileResidual(selected_form);
    auto dominant_ir = compiler.compileResidual(dominant_form);
    p.sys->addCellKernel(
        "op",
        p.selected_field,
        p.selected_field,
        std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
            std::move(selected_ir), svmp::FE::forms::ADMode::Forward));
    p.sys->addCellKernel(
        "op",
        p.dominant_field,
        p.dominant_field,
        std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
            std::move(dominant_ir), svmp::FE::forms::ADMode::Forward));
    p.sys->addSystemConstraint(
        std::make_unique<SharedStateFieldMpc>(
            p.selected_field,
            p.dominant_field,
            std::move(constraint_state)));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(
        *p.sys, p.integrator);
    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error(
            "TwoFieldProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error(
            "TwoFieldProblem failed to create LinearSolver");
    }

    p.history = svmp::FE::timestepping::TimeHistory::allocate(
        *p.factory, p.sys->dofHandler().getNumDofs());
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    p.history.uPrev().zero();
    p.history.uPrev2().zero();
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] TwoFieldProblem makeTwoFieldRefreshedGeometryProblem(
    const double* refreshed_measure,
    double selected_initial,
    double selected_target,
    double dominant_target,
    double dt)
{
    if (refreshed_measure == nullptr) {
        throw std::runtime_error(
            "TwoFieldProblem refreshed measure must not be null");
    }

    TwoFieldProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(
        svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.selected_field = p.sys->addField(svmp::FE::systems::FieldSpec{
        .name = "selected", .space = p.space, .components = 1});
    p.dominant_field = p.sys->addField(svmp::FE::systems::FieldSpec{
        .name = "dominant", .space = p.space, .components = 1});
    p.sys->addOperator("op");
    p.sys->addCellKernel(
        "op",
        p.selected_field,
        p.selected_field,
        std::make_shared<RefreshedGeometryRootKernel>(
            refreshed_measure,
            static_cast<svmp::FE::Real>(selected_target)));
    p.sys->addCellKernel(
        "op",
        p.dominant_field,
        p.dominant_field,
        std::make_shared<AffineScalarCellKernel>(
            static_cast<svmp::FE::Real>(dominant_target)));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(
        *p.sys, p.integrator);
    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error(
            "TwoFieldProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error(
            "TwoFieldProblem failed to create LinearSolver");
    }

    const auto selected_dof = p.sys->fieldDofOffset(p.selected_field);
    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    p.history =
        svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    std::vector<svmp::FE::Real> initial_values(
        static_cast<std::size_t>(n_dofs), svmp::FE::Real{0.0});
    initial_values[static_cast<std::size_t>(selected_dof)] =
        static_cast<svmp::FE::Real>(selected_initial);
    ts_test::setVectorByDof(p.history.uPrev(), initial_values);
    ts_test::setVectorByDof(p.history.uPrev2(), initial_values);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] ScalarProblem makeRefreshedGeometryMeasureProblem(
    double target,
    double dt,
    const std::vector<svmp::FE::Real>& u0,
    const double* measure)
{
    ScalarProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(
        svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");
    p.sys->addOperator("diagnostic_probe");

    auto kernel = std::make_shared<RefreshedGeometryMeasureKernel>(
        measure, static_cast<svmp::FE::Real>(target));
    p.sys->addCellKernel("op", p.u_field, p.u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(
        *p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error(
            "ScalarProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("ScalarProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("ScalarProblem u0 size mismatch");
    }
    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] ScalarProblem makeRefreshedGeometryRootProblem(
    double target,
    double dt,
    const std::vector<svmp::FE::Real>& u0,
    const double* measure)
{
    ScalarProblem p;
    p.mesh =
        std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(
        svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(
        svmp::FE::systems::FieldSpec{
            .name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");
    p.sys->addCellKernel(
        "op",
        p.u_field,
        p.u_field,
        std::make_shared<RefreshedGeometryRootKernel>(
            measure, static_cast<svmp::FE::Real>(target)));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(
        *p.sys, p.integrator);
    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error(
            "ScalarProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error(
            "ScalarProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("ScalarProblem u0 size mismatch");
    }
    p.history =
        svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] double scalarFromDofVector(svmp::FE::backends::GenericVector& vec)
{
    const auto vals = ts_test::getVectorByDof(vec);
    if (vals.size() != 1u) {
        throw std::runtime_error("Expected scalar DOF vector");
    }
    return static_cast<double>(vals[0]);
}

[[nodiscard]] DirectCouplingProblem makeDirectCouplingProblem(double dt,
                                                              const std::vector<svmp::FE::Real>& u0,
                                                              std::optional<svmp::FE::systems::AuxiliaryBlockRole>
                                                                  solver_role = std::nullopt,
                                                              bool use_constraints_in_assembly = true)
{
    DirectCouplingProblem p;
    constexpr int marker = 6;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    p.space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, /*order=*/1);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    const auto u_disc = svmp::FE::forms::FormExpr::discreteField(p.u_field, *p.space, "u");
    auto Q = p.sys->boundaryIntegral(u_disc, marker);

    auto model = svmp::FE::systems::aux::model("newton_rank_one_snapshot",
        [](svmp::FE::systems::ModelFacade& m) {
            auto Q = m.input("Q");
            auto x1 = m.state("x1");
            auto x2 = m.state("x2");
            auto Rp = m.param("Rp");
            m << svmp::FE::systems::ddt(x1) == -x1;
            m << svmp::FE::systems::ddt(x2) == -x2 + Q;
            m << svmp::FE::systems::out("P_out") == x2 + Rp * Q;
        });

    auto deployment = svmp::FE::systems::use(model)
        .name("newton_rank_one_snapshot_inst")
        .global()
        .monolithic()
        .bind("Q", Q)
        .param("Rp", 3.0)
        .initialize({0.0, 0.0});
    if (solver_role.has_value()) {
        deployment.solverRole(*solver_role);
    }
    auto inst = p.sys->deploy(std::move(deployment));

    const auto u = svmp::FE::forms::FormExpr::stateField(p.u_field, *p.space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*p.space, "v");
    const auto residual =
        svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)).dx() -
        (inst.output("P_out") * v).ds(marker);
    (void)svmp::FE::systems::installFormulation(*p.sys, "op", {p.u_field}, residual);
    (void)svmp::FE::systems::installFormulation(
        *p.sys,
        "diagnostic_probe",
        {p.u_field},
        (u * v).dx());

    svmp::FE::systems::SetupOptions options;
    options.use_constraints_in_assembly = use_constraints_in_assembly;
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup(options, inputs);
    p.sys->finalizeAuxiliaryLayout();

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(*p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error("DirectCouplingProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("DirectCouplingProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("DirectCouplingProblem u0 size mismatch");
    }

    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

} // namespace

TEST(SystemAssembly,
     SuppressedAuxiliaryCouplingKeepsDiagnosticPureAndPreservesRecoveryState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "SystemAssembly isolation test requires the Eigen backend";
#endif
    const std::vector<svmp::FE::Real> coefficients{
        0.2, -0.4, 0.1, 0.7};
    auto problem = makeDirectCouplingProblem(
        /*dt=*/0.1,
        coefficients,
        /*solver_role=*/std::nullopt,
        /*use_constraints_in_assembly=*/false);

    svmp::FE::systems::SystemStateView state;
    state.time = 0.1;
    state.dt = 0.1;
    state.u = problem.history.u().localSpan();
    state.u_prev = problem.history.uPrev().localSpan();
    state.u_prev2 = problem.history.uPrev2().localSpan();
    problem.sys->beginTimeStep();

    const auto n_dofs = problem.sys->dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);
    svmp::FE::assembly::DenseMatrixView production_matrix(n_dofs);
    svmp::FE::assembly::DenseVectorView production_vector(n_dofs);
    svmp::FE::systems::AssemblyRequest production_request;
    production_request.op = "op";
    production_request.want_matrix = true;
    production_request.want_vector = true;
    production_request.is_nonlinear_iteration = true;
    const auto production = problem.sys->assemble(
        production_request,
        state,
        &production_matrix,
        &production_vector);
    ASSERT_TRUE(production.success) << production.error_message;

    const auto entry_auxiliary = problem.sys->checkpointAuxiliaryState();
    const auto entry_bordered = problem.sys->borderedCoupling();
    const std::vector<svmp::FE::backends::RankOneUpdate> entry_rank_one(
        problem.sys->lastRankOneUpdates().begin(),
        problem.sys->lastRankOneUpdates().end());
    const std::vector<svmp::FE::backends::ReducedFieldUpdate> entry_reduced(
        problem.sys->lastReducedFieldUpdates().begin(),
        problem.sys->lastReducedFieldUpdates().end());
    const std::vector<svmp::FE::Real> entry_local_condensed_rhs(
        problem.sys->lastLocalCondensedRhsShift().begin(),
        problem.sys->lastLocalCondensedRhsShift().end());
    const bool entry_has_local_condensed_recovery =
        problem.sys->hasLocalCondensedRecovery();
    ASSERT_FALSE(entry_auxiliary.empty());
    ASSERT_TRUE(entry_bordered.active);
    ASSERT_TRUE(!entry_rank_one.empty() || !entry_reduced.empty());

    svmp::FE::assembly::DenseMatrixView diagnostic_matrix(n_dofs);
    svmp::FE::assembly::DenseVectorView diagnostic_vector(n_dofs);
    svmp::FE::systems::AssemblyRequest diagnostic_request;
    diagnostic_request.op = "diagnostic_probe";
    diagnostic_request.want_matrix = true;
    diagnostic_request.want_vector = true;
    diagnostic_request.is_nonlinear_iteration = true;
    diagnostic_request.suppress_auxiliary_coupling_assembly = true;
    const auto diagnostic = problem.sys->assemble(
        diagnostic_request,
        state,
        &diagnostic_matrix,
        &diagnostic_vector);
    ASSERT_TRUE(diagnostic.success) << diagnostic.error_message;

    // P1 tetrahedron mass matrix: diagonal V/10=1/60 and off-diagonal
    // V/20=1/120 for V=1/6. The diagnostic contains only (u,v)_Omega;
    // any generalized auxiliary injection would violate these values.
    for (svmp::FE::GlobalIndex row = 0; row < n_dofs; ++row) {
        svmp::FE::Real expected_residual = 0.0;
        for (svmp::FE::GlobalIndex column = 0; column < n_dofs; ++column) {
            const auto expected_matrix =
                row == column ? svmp::FE::Real{1.0 / 60.0}
                              : svmp::FE::Real{1.0 / 120.0};
            EXPECT_NEAR(
                diagnostic_matrix.getMatrixEntry(row, column),
                expected_matrix,
                1.0e-13)
                << "row=" << row << " column=" << column;
            expected_residual +=
                expected_matrix * coefficients[static_cast<std::size_t>(column)];
        }
        EXPECT_NEAR(
            diagnostic_vector.getVectorEntry(row),
            expected_residual,
            1.0e-13)
            << "row=" << row;
    }

    EXPECT_EQ(problem.sys->checkpointAuxiliaryState(), entry_auxiliary);
    const auto& after_bordered = problem.sys->borderedCoupling();
    EXPECT_EQ(after_bordered.active, entry_bordered.active);
    EXPECT_EQ(after_bordered.globally_reduced,
              entry_bordered.globally_reduced);
    EXPECT_EQ(after_bordered.aux_self_terms_replicated,
              entry_bordered.aux_self_terms_replicated);
    EXPECT_EQ(after_bordered.n_aux, entry_bordered.n_aux);
    EXPECT_EQ(after_bordered.n_field_dofs,
              entry_bordered.n_field_dofs);
    EXPECT_EQ(after_bordered.D, entry_bordered.D);
    EXPECT_EQ(after_bordered.g, entry_bordered.g);
    EXPECT_EQ(after_bordered.B, entry_bordered.B);
    EXPECT_EQ(after_bordered.Ct, entry_bordered.Ct);
    EXPECT_EQ(after_bordered.dF_dxdot, entry_bordered.dF_dxdot);
    EXPECT_EQ(after_bordered.dF_dinputs, entry_bordered.dF_dinputs);
    EXPECT_EQ(after_bordered.dO_dx, entry_bordered.dO_dx);
    EXPECT_EQ(after_bordered.dO_dI, entry_bordered.dO_dI);
    ASSERT_EQ(after_bordered.aux_blocks.size(),
              entry_bordered.aux_blocks.size());
    for (std::size_t i = 0; i < after_bordered.aux_blocks.size(); ++i) {
        EXPECT_EQ(after_bordered.aux_blocks[i].name,
                  entry_bordered.aux_blocks[i].name);
        EXPECT_EQ(after_bordered.aux_blocks[i].dim,
                  entry_bordered.aux_blocks[i].dim);
    }
    ASSERT_EQ(after_bordered.direct_coupling_records.size(),
              entry_bordered.direct_coupling_records.size());
    for (std::size_t i = 0;
         i < after_bordered.direct_coupling_records.size();
         ++i) {
        const auto& after = after_bordered.direct_coupling_records[i];
        const auto& before = entry_bordered.direct_coupling_records[i];
        EXPECT_EQ(after.output_slot, before.output_slot);
        EXPECT_EQ(after.entity_index, before.entity_index);
        EXPECT_EQ(after.aux_local_indices, before.aux_local_indices);
        EXPECT_EQ(after.dF_dinputs, before.dF_dinputs);
        EXPECT_EQ(after.dO_dx, before.dO_dx);
        EXPECT_EQ(after.dO_dI, before.dO_dI);
        EXPECT_EQ(after.input_gradients, before.input_gradients);
        EXPECT_EQ(after.output_gradient, before.output_gradient);
    }

    const auto after_rank_one = problem.sys->lastRankOneUpdates();
    ASSERT_EQ(after_rank_one.size(), entry_rank_one.size());
    for (std::size_t i = 0; i < after_rank_one.size(); ++i) {
        EXPECT_EQ(after_rank_one[i].sigma, entry_rank_one[i].sigma);
        EXPECT_EQ(after_rank_one[i].v, entry_rank_one[i].v);
        EXPECT_EQ(after_rank_one[i].active_components,
                  entry_rank_one[i].active_components);
        EXPECT_EQ(after_rank_one[i].prefer_native_face,
                  entry_rank_one[i].prefer_native_face);
    }
    const auto after_reduced = problem.sys->lastReducedFieldUpdates();
    ASSERT_EQ(after_reduced.size(), entry_reduced.size());
    for (std::size_t i = 0; i < after_reduced.size(); ++i) {
        EXPECT_EQ(after_reduced[i].sigma, entry_reduced[i].sigma);
        EXPECT_EQ(after_reduced[i].left, entry_reduced[i].left);
        EXPECT_EQ(after_reduced[i].right, entry_reduced[i].right);
        EXPECT_EQ(after_reduced[i].active_components,
                  entry_reduced[i].active_components);
        EXPECT_EQ(after_reduced[i].grouped_coupling_id,
                  entry_reduced[i].grouped_coupling_id);
    }
    EXPECT_EQ(problem.sys->hasLocalCondensedRecovery(),
              entry_has_local_condensed_recovery);
    EXPECT_EQ(
        std::vector<svmp::FE::Real>(
            problem.sys->lastLocalCondensedRhsShift().begin(),
            problem.sys->lastLocalCondensedRhsShift().end()),
        entry_local_condensed_rhs);
}

TEST(NewtonSolverExternalStateFixedPoint,
     RequiresFreshlyRegeneratedResidualBeforeConvergence)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double generated_measure = 1.0;
    auto problem = makeRefreshedGeometryRootProblem(
        /*target=*/1.0,
        /*dt=*/0.1,
        /*u0=*/{2.0},
        &generated_measure);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    std::vector<double> refreshed_states;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 3;
    options.abs_tolerance = 1e-13;
    options.rel_tolerance = 1e-8;
    options.use_line_search = false;
    options.external_state_fixed_point.enabled = true;
    options.external_state_fixed_point.max_iterations = 5;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            EXPECT_TRUE(point == SyncPoint::OuterFixedPointState ||
                        point == SyncPoint::ProjectedOuterFixedPointState ||
                        point == SyncPoint::RestoredOuterFixedPointState);
            const auto u = static_cast<double>(state.u.front());
            // Frozen problem 1: u=1.  Regenerating at that state changes the
            // problem to 2u-1=0, whose consistent fixed point is u=0.5.
            generated_measure = u <= 1.0 ? 2.0 : 1.0;
            if (point == SyncPoint::OuterFixedPointState) {
                refreshed_states.push_back(u);
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_TRUE(report.converged);
    EXPECT_EQ(report.outer_iterations, 3);
    EXPECT_EQ(report.inner_iterations_total, 2);
    EXPECT_EQ(report.iterations, 2);
    EXPECT_NEAR(report.residual_norm, 0.0, 1e-13);
    EXPECT_NEAR(
        scalarFromDofVector(problem.history.u()), 0.5, 1e-13);
    ASSERT_EQ(refreshed_states.size(), 3u);
    EXPECT_NEAR(refreshed_states[0], 2.0, 1e-13);
    EXPECT_NEAR(refreshed_states[1], 1.0, 1e-13);
    EXPECT_NEAR(refreshed_states[2], 0.5, 1e-13);
}

TEST(NewtonSolverExternalStateFixedPoint,
     OuterIterationFailureRestoresSolutionHistoryAndRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double generated_measure = 1.0;
    auto problem = makeRefreshedGeometryRootProblem(
        /*target=*/1.0,
        /*dt=*/0.1,
        /*u0=*/{2.0},
        &generated_measure);
    problem.history.ensureSecondOrderState(*problem.factory);
    ts_test::setVectorByDof(problem.history.uDot(), {3.0});
    ts_test::setVectorByDof(problem.history.uDDot(), {-4.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool restored_generated_state = false;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 3;
    options.abs_tolerance = 1e-13;
    options.rel_tolerance = 1e-8;
    options.use_line_search = false;
    options.external_state_fixed_point.enabled = true;
    // One inner solve moves u from 2 to 1, but no refreshed zero-update
    // certificate is available within this deliberately insufficient budget.
    options.external_state_fixed_point.max_iterations = 1;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto u = static_cast<double>(state.u.front());
            generated_measure = u <= 1.0 ? 2.0 : 1.0;
            if (point == SyncPoint::OuterFixedPointState) {
                ts_test::setVectorByDof(problem.history.uPrev(), {7.0});
                ts_test::setVectorByDof(problem.history.uPrev2(), {-8.0});
                ts_test::setVectorByDof(problem.history.uDot(), {9.0});
                ts_test::setVectorByDof(problem.history.uDDot(), {-10.0});
            }
            if (point == SyncPoint::RestoredOuterFixedPointState) {
                restored_generated_state = true;
                EXPECT_NEAR(u, 2.0, 1e-13);
                EXPECT_DOUBLE_EQ(generated_measure, 1.0);
                EXPECT_NEAR(
                    scalarFromDofVector(problem.history.uPrev()),
                    2.0,
                    1e-13);
                EXPECT_NEAR(
                    scalarFromDofVector(problem.history.uPrev2()),
                    2.0,
                    1e-13);
                EXPECT_NEAR(
                    scalarFromDofVector(problem.history.uDot()),
                    3.0,
                    1e-13);
                EXPECT_NEAR(
                    scalarFromDofVector(problem.history.uDDot()),
                    -4.0,
                    1e-13);
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.outer_iterations, 1);
    EXPECT_EQ(report.inner_iterations_total, 1);
    EXPECT_TRUE(restored_generated_state);
    EXPECT_NEAR(
        scalarFromDofVector(problem.history.u()), 2.0, 1e-13);
    EXPECT_NEAR(
        scalarFromDofVector(problem.history.uPrev()), 2.0, 1e-13);
    EXPECT_NEAR(
        scalarFromDofVector(problem.history.uPrev2()), 2.0, 1e-13);
    EXPECT_NEAR(
        scalarFromDofVector(problem.history.uDot()), 3.0, 1e-13);
    EXPECT_NEAR(
        scalarFromDofVector(problem.history.uDDot()), -4.0, 1e-13);
}

TEST(NewtonSolverExternalStateFixedPoint,
     RefreshFailureRestoresAuxiliaryAndBorderedStateBeforeCallback)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(
        /*dt=*/0.1,
        /*u0=*/{0.0, 0.0, 0.0, 0.0});
    auto& block = problem.sys->auxiliaryStateManager().getBlock(
        "newton_rank_one_snapshot_inst");
    ASSERT_EQ(block.work().size(), 2u);
    block.work()[0] = 5.0;
    block.work()[1] = -6.0;
    const auto entry_auxiliary = problem.sys->checkpointAuxiliaryState();
    ASSERT_FALSE(entry_auxiliary.empty());

    auto& bordered = problem.sys->borderedCoupling();
    bordered.active = true;
    bordered.n_aux = 1;
    bordered.n_field_dofs = 4u;
    bordered.g = {7.0};
    const auto entry_bordered = bordered;

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool restored_callback_saw_entry_state = false;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.abs_tolerance = 1e-12;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.external_state_fixed_point.enabled = true;
    options.external_state_fixed_point.max_iterations = 2;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView&, SyncPoint point) {
            if (point == SyncPoint::OuterFixedPointState) {
                auto& trial_block =
                    problem.sys->auxiliaryStateManager().getBlock(
                        "newton_rank_one_snapshot_inst");
                trial_block.work()[0] = 90.0;
                trial_block.work()[1] = 91.0;
                auto& trial_bordered = problem.sys->borderedCoupling();
                trial_bordered.active = false;
                trial_bordered.g = {-99.0};
                throw std::runtime_error("synthetic outer refresh failure");
            }
            if (point == SyncPoint::RestoredOuterFixedPointState) {
                restored_callback_saw_entry_state =
                    problem.sys->checkpointAuxiliaryState() ==
                        entry_auxiliary &&
                    problem.sys->borderedCoupling().active ==
                        entry_bordered.active &&
                    problem.sys->borderedCoupling().n_aux ==
                        entry_bordered.n_aux &&
                    problem.sys->borderedCoupling().g == entry_bordered.g;
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    EXPECT_THROW(
        (void)newton.solveStep(
            *problem.transient,
            *problem.linear,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            workspace),
        std::runtime_error);
    EXPECT_TRUE(restored_callback_saw_entry_state);
    EXPECT_EQ(problem.sys->checkpointAuxiliaryState(), entry_auxiliary);
    EXPECT_EQ(problem.sys->borderedCoupling().active,
              entry_bordered.active);
    EXPECT_EQ(problem.sys->borderedCoupling().n_aux,
              entry_bordered.n_aux);
    EXPECT_EQ(problem.sys->borderedCoupling().g, entry_bordered.g);
}

TEST(NewtonSolverExternalStateFixedPoint,
     StageTimeConstraintIsEstablishedBeforeSnapshotAndRollback)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeAffineScalarProblem(
        /*target=*/2.0,
        /*dt=*/0.1,
        /*u0=*/{0.0},
        /*counts=*/nullptr,
        /*add_time_dependent_dirichlet=*/true);
    constexpr double stage_time = 0.075;

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool outer_saw_stage_value = false;
    bool restore_saw_stage_value = false;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.abs_tolerance = 1e-12;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.external_state_fixed_point.enabled = true;
    options.external_state_fixed_point.max_iterations = 2;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            if (point == SyncPoint::OuterFixedPointState) {
                outer_saw_stage_value =
                    std::abs(static_cast<double>(state.u.front()) -
                             stage_time) < 1e-13;
                throw std::runtime_error(
                    "synthetic stage-time outer failure");
            }
            if (point == SyncPoint::RestoredOuterFixedPointState) {
                restore_saw_stage_value =
                    std::abs(static_cast<double>(state.u.front()) -
                             stage_time) < 1e-13;
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    EXPECT_THROW(
        (void)newton.solveStep(
            *problem.transient,
            *problem.linear,
            stage_time,
            problem.history,
            workspace),
        std::runtime_error);
    EXPECT_TRUE(outer_saw_stage_value);
    EXPECT_TRUE(restore_saw_stage_value);
    EXPECT_NEAR(
        scalarFromDofVector(problem.history.u()), stage_time, 1e-13);
}

TEST(NewtonSolverExternalStateFixedPoint,
     ReallocatesJacobianAfterOuterConstraintSparsityChange)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldAffineProblem(
        /*selected_target=*/1.0,
        /*dominant_target=*/2.0,
        /*dt=*/0.1,
        constraint_state);
    ts_test::setVectorByDof(problem.history.uPrev(), {3.0, 4.0});
    ts_test::setVectorByDof(problem.history.uPrev2(), {-3.0, -4.0});
    problem.history.ensureSecondOrderState(*problem.factory);
    ts_test::setVectorByDof(problem.history.uDot(), {5.0, 6.0});
    ts_test::setVectorByDof(problem.history.uDDot(), {-5.0, -6.0});
    const auto entry_revision = problem.sys->sparsityPatternRevision();

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    int topology_flips = 0;
    bool projected_callback_saw_restored_entry_history = false;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 2;
    options.abs_tolerance = 1e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.assemble_both_when_possible = false;
    options.external_state_fixed_point.enabled = true;
    options.external_state_fixed_point.max_iterations = 4;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            if (point != SyncPoint::OuterFixedPointState &&
                point != SyncPoint::ProjectedOuterFixedPointState) {
                return;
            }
            ASSERT_EQ(state.u.size(), 2u);
            const bool desired_reverse =
                std::max(static_cast<double>(state.u[0]),
                         static_cast<double>(state.u[1])) > 0.5;
            if (desired_reverse != constraint_state->reverse) {
                constraint_state->reverse = desired_reverse;
                problem.sys->rebuildConstraintState();
                ++topology_flips;
            }
            if (point == SyncPoint::ProjectedOuterFixedPointState &&
                constraint_state->reverse) {
                const auto u_prev =
                    ts_test::getVectorByDof(problem.history.uPrev());
                const auto u_prev2 =
                    ts_test::getVectorByDof(problem.history.uPrev2());
                const auto u_dot =
                    ts_test::getVectorByDof(problem.history.uDot());
                const auto u_ddot =
                    ts_test::getVectorByDof(problem.history.uDDot());
                ASSERT_EQ(u_prev.size(), 2u);
                ASSERT_EQ(u_prev2.size(), 2u);
                ASSERT_EQ(u_dot.size(), 2u);
                ASSERT_EQ(u_ddot.size(), 2u);
                projected_callback_saw_restored_entry_history =
                    std::abs(static_cast<double>(u_prev[0]) - 3.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(u_prev[1]) - 3.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(u_prev2[0]) + 3.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(u_prev2[1]) + 3.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(u_dot[0]) - 5.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(u_dot[1]) - 5.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(u_ddot[0]) + 5.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(u_ddot[1]) + 5.0) <
                        1e-13;
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_TRUE(report.converged);
    EXPECT_EQ(report.outer_iterations, 2);
    EXPECT_EQ(report.inner_iterations_total, 1);
    EXPECT_EQ(topology_flips, 1);
    EXPECT_TRUE(projected_callback_saw_restored_entry_history);
    EXPECT_GT(problem.sys->sparsityPatternRevision(), entry_revision);
    EXPECT_EQ(workspace.sparsity_revision,
              problem.sys->sparsityPatternRevision());
    const auto values = ts_test::getVectorByDof(problem.history.u());
    ASSERT_EQ(values.size(), 2u);
    // The equality MPC leaves one admissible scalar.  With independent
    // targets 1 and 2, the constrained Galerkin root is their mean, 1.5;
    // reversing slave/master orientation must preserve that subspace/root.
    EXPECT_NEAR(values[0], 1.5, 1e-13);
    EXPECT_NEAR(values[1], 1.5, 1e-13);
}

TEST(NewtonSolverExternalStateFixedPoint,
     FirstGeneratedConstraintRefreshDefinesCanonicalRollbackEntry)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldAffineProblem(
        /*selected_target=*/1.0,
        /*dominant_target=*/2.0,
        /*dt=*/0.1,
        constraint_state);

    // The caller's candidate satisfies the stale/pre-refresh equality
    // selected=dominant.  The first generated-state refresh replaces that
    // relation with dominant=2*selected+1, whose projected canonical entry is
    // [1,3].  A failed outer solve must roll back to that synchronized entry,
    // not compare it against the stale equality fingerprint captured before
    // the first refresh.
    ts_test::setVectorByDof(problem.history.u(), {1.0, 1.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    int generated_constraint_installs = 0;
    int restored_outer_calls = 0;
    bool restored_outer_saw_canonical_entry = false;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 2;
    options.abs_tolerance = 1e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.assemble_both_when_possible = false;
    options.external_state_fixed_point.enabled = true;
    // The first frozen inner solve takes one Newton update.  With no second
    // outer refresh available, solveStep must return ordinary nonconvergence
    // after restoring the canonical synchronized entry.
    options.external_state_fixed_point.max_iterations = 1;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_EQ(state.u.size(), 2u);
            if (point == SyncPoint::RestoredOuterFixedPointState) {
                ++restored_outer_calls;
                restored_outer_saw_canonical_entry =
                    std::abs(static_cast<double>(state.u[0]) - 1.0) <
                        1e-13 &&
                    std::abs(static_cast<double>(state.u[1]) - 3.0) <
                        1e-13;
            }
            if (point != SyncPoint::OuterFixedPointState &&
                point != SyncPoint::RestoredOuterFixedPointState) {
                return;
            }
            if (!constraint_state->reverse ||
                constraint_state->weight != 2.0 ||
                constraint_state->inhomogeneity != 1.0) {
                constraint_state->reverse = true;
                constraint_state->weight = 2.0;
                constraint_state->inhomogeneity = 1.0;
                problem.sys->rebuildConstraintState();
                ++generated_constraint_installs;
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    std::optional<svmp::FE::timestepping::NewtonReport> report;
    EXPECT_NO_THROW(report.emplace(newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace)));
    ASSERT_TRUE(report.has_value());
    EXPECT_FALSE(report->converged);
    EXPECT_EQ(report->outer_iterations, 1);
    EXPECT_EQ(report->inner_iterations_total, 1);
    EXPECT_EQ(generated_constraint_installs, 1);
    EXPECT_EQ(restored_outer_calls, 1);
    EXPECT_TRUE(restored_outer_saw_canonical_entry);

    const auto values = ts_test::getVectorByDof(problem.history.u());
    ASSERT_EQ(values.size(), 2u);
    EXPECT_NEAR(values[0], 1.0, 1e-13);
    EXPECT_NEAR(values[1], 3.0, 1e-13);
}

TEST(NewtonSolverLineSearch, BacktracksWhenFullStepIncreasesResidual)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_shrink = 0.5;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/3.0);
    (void)newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    const double u_after = scalarFromDofVector(problem.history.u());
    EXPECT_NEAR(u_after, -0.5, 1e-13);
}

TEST(NewtonSolverLineSearch,
     RefreshedTrialMeritRejectsFrozenGeometryRoot)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    auto problem = makeRefreshedGeometryRootProblem(
        /*target=*/1.0,
        /*dt=*/0.1,
        /*u0=*/{2.0},
        &refreshed_measure);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct TrialSample {
        double u;
        double measure;
    };
    std::vector<TrialSample> trials;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = true;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 3;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 0.1;
    nopt.synchronize_state =
        [&refreshed_measure, &trials](
            const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto u = static_cast<double>(state.u.front());
            // The frozen Jacobian at u=2 proposes u=1.  On the actual
            // refreshed residual that state is not a root: the measure jump
            // makes |4*u-1| larger than the base |u-1|.  The half step at
            // u=1.5 retains measure one and is strictly reducing.
            refreshed_measure = u < 1.25 ? 4.0 : 1.0;
            if (point == SyncPoint::LineSearchTrialResidual) {
                trials.push_back(TrialSample{u, refreshed_measure});
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 1.5, 1e-13);
    ASSERT_GE(trials.size(), 2u);
    EXPECT_NEAR(trials[0].u, 1.0, 1e-13);
    EXPECT_DOUBLE_EQ(trials[0].measure, 4.0);
    EXPECT_NEAR(trials[1].u, 1.5, 1e-13);
    EXPECT_DOUBLE_EQ(trials[1].measure, 1.0);
}

TEST(NewtonSolverLineSearch,
     RestoresBaseConstraintTopologyBetweenRejectedAlphas)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldAffineProblem(
        /*selected_target=*/1.0,
        /*dominant_target=*/2.0,
        /*dt=*/0.1,
        constraint_state);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct TrialCandidate {
        double selected;
        double dominant;
    };
    std::vector<TrialCandidate> first_pass_trials;
    bool trial_constraint_active = false;
    bool saw_projected_trial_second_pass = false;
    bool saw_restore_between_alphas = false;
    int jacobian_sync_count = 0;
    const auto selected_dof = problem.sys->fieldDofOffset(
        problem.selected_field);
    const auto dominant_dof = problem.sys->fieldDofOffset(
        problem.dominant_field);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 2;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 2;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 0.5;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_NE(state.u_vector, nullptr);
            ASSERT_GT(state.u.size(),
                      static_cast<std::size_t>(
                          std::max(selected_dof, dominant_dof)));
            const auto selected = static_cast<double>(
                state.u[static_cast<std::size_t>(selected_dof)]);
            const auto dominant = static_cast<double>(
                state.u[static_cast<std::size_t>(dominant_dof)]);

            if (point == SyncPoint::JacobianAssembly) {
                ++jacobian_sync_count;
            }
            if (point == SyncPoint::RestoredNonlinearState) {
                saw_restore_between_alphas =
                    saw_restore_between_alphas || trial_constraint_active;
                constraint_state->reverse = false;
                constraint_state->weight = 1.0;
                constraint_state->inhomogeneity = 0.0;
                trial_constraint_active = false;
                problem.sys->rebuildConstraintState();
                return;
            }
            if (point != SyncPoint::LineSearchTrialResidual) {
                return;
            }

            if (trial_constraint_active) {
                // The callback is invoked again after Newton projects with
                // C_trial: dominant = selected + 100.  Residual-defining
                // derived data therefore sees the projected state.
                EXPECT_NEAR(dominant, selected + 100.0, 1e-12);
                saw_projected_trial_second_pass = true;
                return;
            }

            first_pass_trials.push_back(
                TrialCandidate{selected, dominant});
            if (first_pass_trials.size() == 1u) {
                // Force the full step onto a different slave topology with a
                // large offset, making its refreshed merit non-reducing.
                constraint_state->reverse = true;
                constraint_state->weight = 1.0;
                constraint_state->inhomogeneity = 100.0;
                trial_constraint_active = true;
                problem.sys->rebuildConstraintState();
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/3.0);
    (void)newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    ASSERT_GE(first_pass_trials.size(), 2u);
    EXPECT_TRUE(saw_projected_trial_second_pass);
    EXPECT_TRUE(saw_restore_between_alphas);
    EXPECT_NEAR(first_pass_trials[0].selected,
                first_pass_trials[0].dominant,
                1e-12);
    EXPECT_NEAR(first_pass_trials[1].selected,
                0.5 * first_pass_trials[0].selected,
                1e-12);
    EXPECT_NEAR(first_pass_trials[1].dominant,
                0.5 * first_pass_trials[0].dominant,
                1e-12);
    EXPECT_GE(jacobian_sync_count, 2);
}

TEST(NewtonSolverLineSearch,
     ProjectsChangedConstraintCoefficientsBeforeTrialResidual)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldAffineProblem(
        /*selected_target=*/1.0,
        /*dominant_target=*/2.0,
        /*dt=*/0.1,
        constraint_state);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    const auto selected_dof = problem.sys->fieldDofOffset(
        problem.selected_field);
    const auto dominant_dof = problem.sys->fieldDofOffset(
        problem.dominant_field);
    bool coefficient_trial_active = false;
    bool saw_projected_second_pass = false;
    bool saw_restored_base_coefficient = false;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            const auto selected = static_cast<double>(
                state.u[static_cast<std::size_t>(selected_dof)]);
            const auto dominant = static_cast<double>(
                state.u[static_cast<std::size_t>(dominant_dof)]);
            if (point == SyncPoint::RestoredNonlinearState) {
                constraint_state->weight = 1.0;
                coefficient_trial_active = false;
                problem.sys->rebuildConstraintState();
                saw_restored_base_coefficient = true;
                return;
            }
            if (point != SyncPoint::LineSearchTrialResidual) {
                return;
            }
            if (!coefficient_trial_active) {
                constraint_state->weight = 2.0;
                coefficient_trial_active = true;
                problem.sys->rebuildConstraintState();
                return;
            }
            EXPECT_NEAR(selected, 2.0 * dominant, 1e-12);
            saw_projected_second_pass = true;
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    (void)newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(saw_projected_second_pass);
    EXPECT_TRUE(saw_restored_base_coefficient);
    EXPECT_DOUBLE_EQ(constraint_state->weight, 1.0);
}

TEST(NewtonSolverLineSearch,
     FailsClosedWhenRejectedConstraintStateCannotBeRestored)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldAffineProblem(
        /*selected_target=*/1.0,
        /*dominant_target=*/2.0,
        /*dt=*/0.1,
        constraint_state);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool trial_constraint_active = false;
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView&,
            SyncPoint point) {
            if (point == SyncPoint::LineSearchTrialResidual &&
                !trial_constraint_active) {
                constraint_state->reverse = true;
                constraint_state->inhomogeneity = 100.0;
                trial_constraint_active = true;
                problem.sys->rebuildConstraintState();
            }
            // Deliberately leave the rejected trial constraint installed at
            // RestoredNonlinearState.  Newton must fail closed rather than
            // constructing another trial from contaminated state.
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    EXPECT_THROW(
        (void)newton.solveStep(
            *problem.transient,
            linear,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            ws),
        svmp::FE::systems::InvalidStateException);
}

TEST(NewtonSolverLineSearch,
     AcceptedTrialProjectsHistoryBeforeRefreshedResidual)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldAffineProblem(
        /*selected_target=*/2.0,
        /*dominant_target=*/1.0,
        /*dt=*/0.1,
        constraint_state);
    const auto selected_dof = problem.sys->fieldDofOffset(
        problem.selected_field);
    const auto dominant_dof = problem.sys->fieldDofOffset(
        problem.dominant_field);
    std::vector<svmp::FE::Real> base_state(
        static_cast<std::size_t>(problem.history.u().size()),
        svmp::FE::Real(0.0));
    base_state[static_cast<std::size_t>(selected_dof)] = 1.0;
    base_state[static_cast<std::size_t>(dominant_dof)] = 1.0;
    ts_test::setVectorByDof(problem.history.uPrev(), base_state);
    ts_test::setVectorByDof(problem.history.uPrev2(), base_state);
    problem.history.resetCurrentToPrevious();

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool trial_constraint_active = false;
    bool saw_refreshed_projected_history = false;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-13;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            if (point == SyncPoint::LineSearchTrialResidual &&
                !trial_constraint_active) {
                constraint_state->weight = 2.0;
                trial_constraint_active = true;
                problem.sys->rebuildConstraintState();
                return;
            }
            if (point == SyncPoint::ResidualAssembly &&
                trial_constraint_active) {
                ASSERT_GT(state.u_prev.size(),
                          static_cast<std::size_t>(std::max(
                              selected_dof, dominant_dof)));
                const auto selected_prev = static_cast<double>(
                    state.u_prev[static_cast<std::size_t>(selected_dof)]);
                const auto dominant_prev = static_cast<double>(
                    state.u_prev[static_cast<std::size_t>(dominant_dof)]);
                EXPECT_NEAR(selected_prev, 2.0 * dominant_prev, 1e-12);
                saw_refreshed_projected_history = true;
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Keep the raw current-state update at zero.  The trial becomes an exact
    // root only after its changed MPC coefficient projects both the current
    // vector and the transient history from selected=dominant to
    // selected=2*dominant.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/0.0);
    const auto rep = newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(rep.converged);
    EXPECT_TRUE(saw_refreshed_projected_history);
    const auto final_prev = problem.history.uPrevSpan();
    EXPECT_NEAR(
        static_cast<double>(
            final_prev[static_cast<std::size_t>(selected_dof)]),
        2.0 * static_cast<double>(
                  final_prev[static_cast<std::size_t>(dominant_dof)]),
        1e-12);
}

TEST(NewtonSolverLineSearch,
     TrialMeritUsesProjectedTransientHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldHistorySensitiveMpcProblem(
        /*dt=*/0.1, constraint_state);
    const auto selected_dof = problem.sys->fieldDofOffset(
        problem.selected_field);
    const auto dominant_dof = problem.sys->fieldDofOffset(
        problem.dominant_field);
    std::vector<svmp::FE::Real> base_state(
        static_cast<std::size_t>(problem.history.u().size()),
        svmp::FE::Real(0.0));
    base_state[static_cast<std::size_t>(selected_dof)] = 1.0;
    base_state[static_cast<std::size_t>(dominant_dof)] = 1.0;
    ts_test::setVectorByDof(problem.history.uPrev(), base_state);
    ts_test::setVectorByDof(problem.history.uPrev2(), base_state);
    problem.history.resetCurrentToPrevious();

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool changed_trial_constraint = false;
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-13;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView&,
            SyncPoint point) {
            if (point == SyncPoint::LineSearchTrialResidual &&
                !changed_trial_constraint) {
                constraint_state->weight = 2.0;
                changed_trial_constraint = true;
                problem.sys->rebuildConstraintState();
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // With u_prev left under the base selected=dominant relation, the dt term
    // at the projected candidate selected=2, dominant=1 is O(1/dt) and the
    // trial is rejected.  A coherent trial projects both u and u_prev under
    // selected=2*dominant, making dt(selected)=0 and the residual exactly zero.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/0.0);
    const auto rep = newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(rep.converged);
    EXPECT_TRUE(changed_trial_constraint);
    const auto final_state = problem.history.uSpan();
    const auto final_prev = problem.history.uPrevSpan();
    EXPECT_NEAR(
        static_cast<double>(
            final_state[static_cast<std::size_t>(selected_dof)]),
        2.0,
        1e-12);
    EXPECT_NEAR(
        static_cast<double>(
            final_state[static_cast<std::size_t>(dominant_dof)]),
        1.0,
        1e-12);
    EXPECT_NEAR(
        static_cast<double>(
            final_prev[static_cast<std::size_t>(selected_dof)]),
        2.0,
        1e-12);
    EXPECT_NEAR(
        static_cast<double>(
            final_prev[static_cast<std::size_t>(dominant_dof)]),
        1.0,
        1e-12);
}

TEST(NewtonSolverLineSearch,
     StepToleranceUsesProjectedConstraintState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldAffineProblem(
        /*selected_target=*/1.0,
        /*dominant_target=*/2.0,
        /*dt=*/0.1,
        constraint_state);
    const auto dominant_dof = problem.sys->fieldDofOffset(
        problem.dominant_field);
    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    // The nominal reduced-space Newton update is below this threshold, but
    // the accepted-state constraint projection adds an O(100) displacement.
    nopt.step_tolerance = 10.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = false;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            if (point != SyncPoint::AcceptedNonlinearState) {
                return;
            }
            const auto dominant = static_cast<double>(
                state.u[static_cast<std::size_t>(dominant_dof)]);
            if (dominant > 1.0 && !constraint_state->reverse) {
                constraint_state->reverse = true;
                constraint_state->inhomogeneity = 100.0;
                problem.sys->rebuildConstraintState();
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);
    EXPECT_FALSE(rep.converged);
    EXPECT_GT(problem.history.u().norm(), nopt.step_tolerance);
}

TEST(NewtonSolverLineSearch, RestoresOriginalWhenAlphaMinDoesNotReduce)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 5;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 0.6;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Force an overshoot so alpha=1 fails and even alpha_min still increases
    // the residual.  A clamped but non-reducing trial is not a valid
    // globalization step; restore the original iterate.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/4.0);
    (void)newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    const double u_after = scalarFromDofVector(problem.history.u());
    EXPECT_NEAR(u_after, 1.0, 1e-13);
}

TEST(NewtonSolverLineSearch, DoesNotConvergeFromRestoredZeroStep)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.75;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 1e-12;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Flip the Newton direction so the Armijo condition can never be satisfied. When the
    // line search cannot find a decreasing trial, the solver must reject the update.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    const auto rep = newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);
}

TEST(NewtonSolverLineSearch, SynchronizesTrialAndRestoredStates)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct SyncRecord {
        SyncPoint point;
        double u;
        std::uint64_t cut_topology_key;
        std::uint64_t value_revision;
    };
    std::vector<SyncRecord> sync_records;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.line_search_alpha_min = 1e-12;
    nopt.synchronize_state =
        [&sync_records](const svmp::FE::systems::SystemStateView& state,
                        SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto u = static_cast<double>(state.u.front());
            const auto cut_topology_key =
                u > 1.5 ? std::uint64_t{0x202u} : std::uint64_t{0x101u};
            const auto value_revision =
                state.u_vector != nullptr ? state.u_vector->valueRevision()
                                          : std::uint64_t{0};
            sync_records.push_back(
                SyncRecord{point, u, cut_topology_key, value_revision});
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    const auto rep = newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_FALSE(rep.converged);
    auto saw_trial_update = false;
    auto saw_zero_alpha_trial = false;
    auto saw_restored_state = false;
    auto saw_trial_cut_topology = false;
    auto saw_restored_state_cut_topology = false;
    std::optional<std::uint64_t> accepted_revision;
    std::optional<std::uint64_t> trial_revision;
    std::optional<std::uint64_t> restored_revision;
    for (const auto& rec : sync_records) {
        if (rec.point == SyncPoint::AcceptedNonlinearState &&
            std::abs(rec.u - 1.0) < 1e-13 &&
            !accepted_revision.has_value()) {
            accepted_revision = rec.value_revision;
        }
        if (rec.point == SyncPoint::LineSearchTrialResidual &&
            std::abs(rec.u - 2.0) < 1e-13) {
            saw_trial_update = true;
            saw_trial_cut_topology = rec.cut_topology_key == std::uint64_t{0x202u};
            trial_revision = rec.value_revision;
        }
        if (rec.point == SyncPoint::LineSearchTrialResidual &&
            std::abs(rec.u - 1.0) < 1e-13) {
            saw_zero_alpha_trial = true;
        }
        if (rec.point == SyncPoint::RestoredNonlinearState &&
            std::abs(rec.u - 1.0) < 1e-13) {
            saw_restored_state = true;
            saw_restored_state_cut_topology =
                rec.cut_topology_key == std::uint64_t{0x101u};
            restored_revision = rec.value_revision;
        }
    }
    EXPECT_TRUE(saw_trial_update);
    // A rejected update is restored directly; alpha=0 must not be opened as
    // another geometric trial after the base context has been reconstructed.
    EXPECT_FALSE(saw_zero_alpha_trial);
    EXPECT_TRUE(saw_restored_state);
    EXPECT_TRUE(saw_trial_cut_topology);
    EXPECT_TRUE(saw_restored_state_cut_topology);
    ASSERT_TRUE(accepted_revision.has_value());
    ASSERT_TRUE(trial_revision.has_value());
    ASSERT_TRUE(restored_revision.has_value());
    EXPECT_GT(*trial_revision, *accepted_revision);
    EXPECT_GT(*restored_revision, *trial_revision);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 1.0, 1e-13);
}

TEST(NewtonSolverLineSearch,
     ThrowingTrialRestoresOriginallyAbsentRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u,
           const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool allocated_rates_in_trial = false;
    bool restored_callback_saw_absent_rates = false;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.line_search_alpha_min = 1e-12;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView&, SyncPoint point) {
            if (point == SyncPoint::RestoredNonlinearState) {
                restored_callback_saw_absent_rates =
                    !problem.history.hasUDotState() &&
                    !problem.history.hasUDDotState();
                return;
            }
            if (point != SyncPoint::LineSearchTrialResidual ||
                allocated_rates_in_trial) {
                return;
            }

            EXPECT_FALSE(problem.history.hasUDotState());
            EXPECT_FALSE(problem.history.hasUDDotState());
            problem.history.ensureSecondOrderState(*problem.factory);
            ts_test::setVectorByDof(problem.history.uDot(), {17.0});
            ts_test::setVectorByDof(problem.history.uDDot(), {-23.0});
            allocated_rates_in_trial = true;
            throw std::runtime_error(
                "intentional line-search trial synchronization failure");
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);
    ASSERT_FALSE(problem.history.hasUDotState());
    ASSERT_FALSE(problem.history.hasUDDotState());

    // The scaled direction increases the residual. The callback failure is
    // treated as a rejected alpha and must roll every speculative rate-state
    // allocation back before RestoredNonlinearState is invoked.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    const auto rep = newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_TRUE(allocated_rates_in_trial);
    EXPECT_TRUE(restored_callback_saw_absent_rates);
    EXPECT_FALSE(problem.history.hasUDotState());
    EXPECT_FALSE(problem.history.hasUDDotState());
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 1.0, 1e-13);
}

TEST(NewtonSolverLineSearch, RestoresAcceptedHighOrderCutContextAfterRejectedTrial)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u,
           const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct CutContextState {
        SyncPoint point;
        double u;
        std::uint64_t topology_key;
        std::uint64_t request_policy_key;
    };

    constexpr std::uint64_t kAcceptedTopology = 0x5101u;
    constexpr std::uint64_t kTrialTopology = 0x5202u;
    constexpr std::uint64_t kHighOrderPolicy = 0x5a7e0001u;
    CutContextState installed{
        SyncPoint::AcceptedNonlinearState,
        1.0,
        kAcceptedTopology,
        kHighOrderPolicy};
    std::vector<CutContextState> sync_records;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.line_search_alpha_min = 1e-12;
    nopt.synchronize_state =
        [&installed, &sync_records](const svmp::FE::systems::SystemStateView& state,
                                    SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto u = static_cast<double>(state.u.front());
            const auto topology =
                u > 1.5 ? kTrialTopology : kAcceptedTopology;
            installed = CutContextState{point, u, topology, kHighOrderPolicy};
            sync_records.push_back(installed);
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    const auto rep = newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_FALSE(rep.converged);
    bool saw_rejected_trial_context = false;
    bool saw_zero_alpha_trial_context = false;
    bool saw_restored_context = false;
    bool saw_policy_change = false;
    for (const auto& rec : sync_records) {
        saw_policy_change =
            saw_policy_change || rec.request_policy_key != kHighOrderPolicy;
        if (rec.point == SyncPoint::LineSearchTrialResidual &&
            std::abs(rec.u - 2.0) < 1e-13) {
            saw_rejected_trial_context =
                rec.topology_key == kTrialTopology &&
                rec.request_policy_key == kHighOrderPolicy;
        }
        if (rec.point == SyncPoint::LineSearchTrialResidual &&
            std::abs(rec.u - 1.0) < 1e-13) {
            saw_zero_alpha_trial_context =
                rec.topology_key == kAcceptedTopology &&
                rec.request_policy_key == kHighOrderPolicy;
        }
        if (rec.point == SyncPoint::RestoredNonlinearState &&
            std::abs(rec.u - 1.0) < 1e-13) {
            saw_restored_context =
                rec.topology_key == kAcceptedTopology &&
                rec.request_policy_key == kHighOrderPolicy;
        }
    }
    EXPECT_TRUE(saw_rejected_trial_context);
    EXPECT_FALSE(saw_zero_alpha_trial_context);
    EXPECT_TRUE(saw_restored_context);
    EXPECT_FALSE(saw_policy_change);
    EXPECT_EQ(installed.topology_key, kAcceptedTopology);
    EXPECT_EQ(installed.request_policy_key, kHighOrderPolicy);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 1.0, 1e-13);
}

TEST(NewtonSolver, SynchronizesUpdatedCoupledGeometryBeforeResidualAssembly)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& phi,
           const svmp::FE::forms::FormExpr& v) {
            return ((phi - svmp::FE::forms::FormExpr::constant(2.0)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{0.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct SyncRecord {
        SyncPoint point;
        double phi;
        std::uint64_t topology_key;
    };
    std::vector<SyncRecord> sync_records;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 2;
    nopt.abs_tolerance = 1e-14;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.synchronize_state =
        [&sync_records](const svmp::FE::systems::SystemStateView& state,
                        SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto phi = static_cast<double>(state.u.front());
            const auto topology_key =
                phi > 1.0 ? std::uint64_t{0x220u} : std::uint64_t{0x110u};
            sync_records.push_back(SyncRecord{point, phi, topology_key});
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(rep.converged);
    bool saw_initial_residual_state = false;
    bool saw_updated_residual_state = false;
    for (const auto& rec : sync_records) {
        const bool residual_assembly =
            rec.point == SyncPoint::ResidualAssembly ||
            rec.point == SyncPoint::JacobianAndResidualAssembly;
        if (!residual_assembly) {
            continue;
        }
        if (std::abs(rec.phi) < 1e-13) {
            saw_initial_residual_state =
                rec.topology_key == std::uint64_t{0x110u};
        }
        if (std::abs(rec.phi - 2.0) < 1e-13) {
            saw_updated_residual_state =
                rec.topology_key == std::uint64_t{0x220u};
        }
    }

    EXPECT_TRUE(saw_initial_residual_state);
    EXPECT_TRUE(saw_updated_residual_state);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 2.0, 1e-13);
}

TEST(NewtonSolverLineSearch,
     ReassemblesResidualAfterAcceptedStateRefreshBeforeConvergence)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    auto problem = makeRefreshedGeometryRootProblem(
        /*target=*/1.0,
        /*dt=*/0.1,
        /*u0=*/{2.0},
        &refreshed_measure);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct SyncRecord {
        SyncPoint point;
        double u;
        double measure;
    };
    std::vector<SyncRecord> sync_records;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 3;
    nopt.abs_tolerance = 1e-13;
    nopt.rel_tolerance = 0.0;
    // Deliberately larger than the first update.  A solver that applies the
    // step test to the frozen trial residual would incorrectly stop at u=1.
    nopt.step_tolerance = 2.0;
    nopt.use_line_search = true;
    nopt.assemble_both_when_possible = true;
    nopt.accepted_state_sync_invalidates_residual = true;
    nopt.synchronize_state =
        [&refreshed_measure, &sync_records](
            const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto u = static_cast<double>(state.u.front());
            if (point == SyncPoint::AcceptedNonlinearState) {
                refreshed_measure = u < 1.5 ? 1.5 : 1.0;
            }
            sync_records.push_back(
                SyncRecord{point, u, refreshed_measure});
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(rep.iterations, 2);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 2.0 / 3.0, 1e-13);

    bool saw_refreshed_residual_at_stale_trial_root = false;
    for (const auto& rec : sync_records) {
        if (rec.point == SyncPoint::ResidualAssembly &&
            std::abs(rec.u - 1.0) < 1e-13 &&
            std::abs(rec.measure - 1.5) < 1e-13) {
            saw_refreshed_residual_at_stale_trial_root = true;
        }
    }
    EXPECT_TRUE(saw_refreshed_residual_at_stale_trial_root);
}

TEST(NewtonSolverLineSearch,
     RejectsCandidateThatOnlyPassesFrozenTrialMerit)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    auto problem = makeRefreshedGeometryRootProblem(
        /*target=*/1.0,
        /*dt=*/0.1,
        /*u0=*/{2.0},
        &refreshed_measure);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool saw_nonreducing_accepted_refresh = false;
    bool saw_restored_base_refresh = false;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.accepted_state_sync_invalidates_residual = true;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto u = static_cast<double>(state.u.front());
            if (point == SyncPoint::AcceptedNonlinearState) {
                refreshed_measure = u < 1.5 ? 2.0 : 1.0;
                if (std::abs(u - 1.0) < 1e-13) {
                    saw_nonreducing_accepted_refresh = true;
                }
            } else if (point == SyncPoint::RestoredNonlinearState) {
                refreshed_measure = u < 1.5 ? 2.0 : 1.0;
                if (std::abs(u - 2.0) < 1e-13 &&
                    std::abs(refreshed_measure - 1.0) < 1e-13) {
                    saw_restored_base_refresh = true;
                }
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_TRUE(saw_nonreducing_accepted_refresh);
    EXPECT_TRUE(saw_restored_base_refresh);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 2.0, 1e-13);
    EXPECT_NEAR(refreshed_measure, 1.0, 1e-13);
}

TEST(NewtonSolver,
     StepToleranceUsesResidualReassembledAfterAcceptedStateRefresh)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    auto problem = makeRefreshedGeometryRootProblem(
        /*target=*/1.0,
        /*dt=*/0.1,
        /*u0=*/{2.0},
        &refreshed_measure);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 2;
    nopt.abs_tolerance = 1e-13;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 2.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.accepted_state_sync_invalidates_residual = true;
    nopt.synchronize_state =
        [&refreshed_measure](
            const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            if (point == SyncPoint::AcceptedNonlinearState) {
                const auto u = static_cast<double>(state.u.front());
                refreshed_measure = u < 1.5 ? 2.0 : 1.0;
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(rep.iterations, 2);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 0.5, 1e-13);
    EXPECT_NEAR(rep.residual_norm, 0.0, 1e-13);
}

TEST(NewtonSolver, RefreshedGeometryJacobianCheckReportsQuasiNewtonMismatch)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    auto problem = makeRefreshedGeometryMeasureProblem(
        /*target=*/0.0,
        /*dt=*/0.1,
        /*u0=*/{1.0},
        &refreshed_measure);

    using svmp::FE::timestepping::JacobianCheckGeometryMode;
    using svmp::FE::timestepping::NewtonJacobianCheckDiagnostic;
    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;

    std::vector<NewtonJacobianCheckDiagnostic> diagnostics;
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.jacobian_check_geometry_mode =
        JacobianCheckGeometryMode::RefreshedGeometry;
    nopt.jacobian_check_geometry_tangent_policy =
        "RefreshedFrozenQuadrature";
    nopt.jacobian_check_relative_tolerance = 1e-8;
    nopt.synchronize_state =
        [&refreshed_measure](const svmp::FE::systems::SystemStateView& state,
                             SyncPoint) {
            if (!state.u.empty()) {
                refreshed_measure =
                    1.0 + 0.25 * static_cast<double>(state.u.front());
            }
        };
    nopt.jacobian_check_diagnostic =
        [&diagnostics](const NewtonJacobianCheckDiagnostic& diagnostic) {
            diagnostics.push_back(diagnostic);
        };

    ScopedEnvVar jac_check("SVMP_FE_JACOBIAN_CHECK", "1");
    ScopedEnvVar jac_it("SVMP_FE_JACOBIAN_CHECK_IT", "0");
    ScopedEnvVar jac_step("SVMP_FE_JACOBIAN_CHECK_STEP", "1e-7");
    ScopedEnvVar jac_scheme("SVMP_FE_JACOBIAN_CHECK_SCHEME", "forward");

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(rep.linear.converged);
    ASSERT_FALSE(diagnostics.empty());
    const auto& diagnostic = diagnostics.front();
    EXPECT_EQ(diagnostic.geometry_mode,
              JacobianCheckGeometryMode::RefreshedGeometry);
    EXPECT_EQ(diagnostic.geometry_tangent_policy,
              "RefreshedFrozenQuadrature");
    EXPECT_EQ(diagnostic.geometry_result,
              "expected_quasi_newton_geometry_mismatch");
    EXPECT_EQ(diagnostic.component_filter, "all");
    EXPECT_EQ(diagnostic.finite_difference_scheme, "forward");
    EXPECT_GT(diagnostic.relative_error,
              nopt.jacobian_check_relative_tolerance);
}

TEST(NewtonSolver,
     JacobianCheckRestoresTransientHistoryRatesAndStateDependentMpc)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldHistorySensitiveMpcProblem(
        /*dt=*/0.125, constraint_state);
    const auto selected_dof =
        problem.sys->fieldDofOffset(problem.selected_field);
    const auto dominant_dof =
        problem.sys->fieldDofOffset(problem.dominant_field);

    auto two_field_state = [&](double selected, double dominant) {
        std::vector<svmp::FE::Real> values(
            static_cast<std::size_t>(problem.history.u().size()),
            svmp::FE::Real(0.0));
        values[static_cast<std::size_t>(selected_dof)] =
            static_cast<svmp::FE::Real>(selected);
        values[static_cast<std::size_t>(dominant_dof)] =
            static_cast<svmp::FE::Real>(dominant);
        return values;
    };
    const auto current_base = two_field_state(1.0, 1.0);
    const auto previous_base = two_field_state(1.0, 1.0);
    const auto previous2_base = two_field_state(2.0, 2.0);
    const auto u_dot_base = two_field_state(3.0, 3.0);
    const auto u_ddot_base = two_field_state(4.0, 4.0);
    ts_test::setVectorByDof(problem.history.uPrev(), previous_base);
    ts_test::setVectorByDof(problem.history.uPrev2(), previous2_base);
    problem.history.resetCurrentToPrevious();
    problem.history.ensureSecondOrderState(*problem.factory);
    ts_test::setVectorByDof(problem.history.uDot(), u_dot_base);
    ts_test::setVectorByDof(problem.history.uDDot(), u_ddot_base);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool trial_constraint_active = false;
    bool saw_projected_probe = false;
    bool saw_restored_probe = false;
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            if (point == SyncPoint::RestoredNonlinearState) {
                if (trial_constraint_active) {
                    constraint_state->reverse = false;
                    constraint_state->weight = 1.0;
                    constraint_state->inhomogeneity = 0.0;
                    trial_constraint_active = false;
                    saw_restored_probe = true;
                    problem.sys->rebuildConstraintState();
                }
                return;
            }
            if (point != SyncPoint::ResidualAssembly) {
                return;
            }
            if (trial_constraint_active) {
                const auto selected = static_cast<double>(
                    state.u[static_cast<std::size_t>(selected_dof)]);
                const auto dominant = static_cast<double>(
                    state.u[static_cast<std::size_t>(dominant_dof)]);
                saw_projected_probe =
                    std::abs(dominant -
                             (2.0 * selected + 100.0)) < 1e-10;
                return;
            }

            const auto selected = static_cast<double>(
                state.u[static_cast<std::size_t>(selected_dof)]);
            const auto dominant = static_cast<double>(
                state.u[static_cast<std::size_t>(dominant_dof)]);
            if (std::abs(selected - 1.0) <= 1e-12 &&
                std::abs(dominant - 1.0) <= 1e-12) {
                return;
            }
            // Reverse the slave topology and alter both its homogeneous and
            // inhomogeneous maps.  Reprojecting this trial back with the base
            // MPC cannot recover any vector without an exact snapshot.
            constraint_state->reverse = true;
            constraint_state->weight = 2.0;
            constraint_state->inhomogeneity = 100.0;
            trial_constraint_active = true;
            problem.sys->rebuildConstraintState();
        };

    ScopedEnvVar jac_check("SVMP_FE_JACOBIAN_CHECK", "1");
    ScopedEnvVar jac_it("SVMP_FE_JACOBIAN_CHECK_IT", "0");
    ScopedEnvVar jac_step("SVMP_FE_JACOBIAN_CHECK_STEP", "1e-7");
    ScopedEnvVar jac_scheme("SVMP_FE_JACOBIAN_CHECK_SCHEME", "forward");

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Keep the real Newton update at zero so the post-solve current vector is
    // also an exact check of the diagnostic transaction.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/0.0);
    EXPECT_NO_THROW((void)newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws));

    EXPECT_TRUE(saw_projected_probe);
    EXPECT_TRUE(saw_restored_probe);
    EXPECT_FALSE(trial_constraint_active);
    EXPECT_FALSE(problem.sys->meshCoordinateTransactionActive());
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.u()), current_base);
    EXPECT_EQ(
        ts_test::getVectorByDof(problem.history.uPrev()), previous_base);
    EXPECT_EQ(
        ts_test::getVectorByDof(problem.history.uPrev2()), previous2_base);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uDot()), u_dot_base);
    EXPECT_EQ(
        ts_test::getVectorByDof(problem.history.uDDot()), u_ddot_base);
}

TEST(NewtonSolver,
     JacobianCheckRestoresStateBeforePropagatingProbeException)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto constraint_state = std::make_shared<SharedFieldMpcState>();
    auto problem = makeTwoFieldHistorySensitiveMpcProblem(
        /*dt=*/0.125, constraint_state);
    const auto selected_dof =
        problem.sys->fieldDofOffset(problem.selected_field);
    const auto dominant_dof =
        problem.sys->fieldDofOffset(problem.dominant_field);

    auto two_field_state = [&](double selected, double dominant) {
        std::vector<svmp::FE::Real> values(
            static_cast<std::size_t>(problem.history.u().size()),
            svmp::FE::Real(0.0));
        values[static_cast<std::size_t>(selected_dof)] =
            static_cast<svmp::FE::Real>(selected);
        values[static_cast<std::size_t>(dominant_dof)] =
            static_cast<svmp::FE::Real>(dominant);
        return values;
    };
    const auto current_base = two_field_state(1.0, 1.0);
    const auto previous_base = two_field_state(1.0, 1.0);
    const auto previous2_base = two_field_state(2.0, 2.0);
    ts_test::setVectorByDof(problem.history.uPrev(), previous_base);
    ts_test::setVectorByDof(problem.history.uPrev2(), previous2_base);
    problem.history.resetCurrentToPrevious();
    ASSERT_FALSE(problem.history.hasUDotState());
    ASSERT_FALSE(problem.history.hasUDDotState());

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool trial_constraint_active = false;
    bool saw_projected_probe = false;
    bool saw_restored_probe = false;
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            if (point == SyncPoint::RestoredNonlinearState) {
                if (trial_constraint_active) {
                    constraint_state->reverse = false;
                    constraint_state->weight = 1.0;
                    constraint_state->inhomogeneity = 0.0;
                    trial_constraint_active = false;
                    saw_restored_probe = true;
                    problem.sys->rebuildConstraintState();
                }
                return;
            }
            if (point != SyncPoint::ResidualAssembly) {
                return;
            }
            if (trial_constraint_active) {
                saw_projected_probe = true;
                // Allocation state is part of the transaction too: a failed
                // speculative callback must not leave first-step rate slots
                // installed on the live history.
                problem.history.ensureSecondOrderState(*problem.factory);
                throw std::runtime_error(
                    "intentional Jacobian-check probe failure");
            }

            const auto selected = static_cast<double>(
                state.u[static_cast<std::size_t>(selected_dof)]);
            const auto dominant = static_cast<double>(
                state.u[static_cast<std::size_t>(dominant_dof)]);
            if (std::abs(selected - 1.0) <= 1e-12 &&
                std::abs(dominant - 1.0) <= 1e-12) {
                return;
            }
            constraint_state->reverse = true;
            constraint_state->weight = 2.0;
            constraint_state->inhomogeneity = 100.0;
            trial_constraint_active = true;
            problem.sys->rebuildConstraintState();
        };

    ScopedEnvVar jac_check("SVMP_FE_JACOBIAN_CHECK", "1");
    ScopedEnvVar jac_it("SVMP_FE_JACOBIAN_CHECK_IT", "0");
    ScopedEnvVar jac_step("SVMP_FE_JACOBIAN_CHECK_STEP", "1e-7");
    ScopedEnvVar jac_scheme("SVMP_FE_JACOBIAN_CHECK_SCHEME", "forward");

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    EXPECT_THROW(
        (void)newton.solveStep(
            *problem.transient,
            *problem.linear,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            ws),
        std::runtime_error);

    EXPECT_TRUE(saw_projected_probe);
    EXPECT_TRUE(saw_restored_probe);
    EXPECT_FALSE(trial_constraint_active);
    EXPECT_FALSE(problem.sys->meshCoordinateTransactionActive());
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.u()), current_base);
    EXPECT_EQ(
        ts_test::getVectorByDof(problem.history.uPrev()), previous_base);
    EXPECT_EQ(
        ts_test::getVectorByDof(problem.history.uPrev2()), previous2_base);
    EXPECT_FALSE(problem.history.hasUDotState());
    EXPECT_FALSE(problem.history.hasUDDotState());
}

TEST(NewtonSolver, ReusesJacobianWhenRebuildPeriodGreaterThanOne)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 5;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = false;
    nopt.jacobian_rebuild_period = 3;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, nopt.max_iterations);

}

TEST(NewtonSolver, ReusesStateIndependentJacobianWhenRebuildPeriodIsOne)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeAffineScalarProblem(
        /*target=*/2.0,
        /*dt=*/0.1,
        /*u0=*/{0.0},
        &counts);

    EXPECT_TRUE(problem.sys->operatorMatrixStateIndependent("op"));

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 3;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.jacobian_rebuild_period = 1;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver under_relaxed(*problem.linear, /*scale=*/0.25);
    const auto rep = newton.solveStep(*problem.transient,
                                      under_relaxed,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, nopt.max_iterations);
    EXPECT_EQ(counts.matrix_and_vector, 1);
    EXPECT_EQ(counts.matrix_only, 0);
    EXPECT_GT(counts.vector_only, 0);
}

TEST(NewtonSolver, ScalesDtIncrementsByDtOrExplicitFactor)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;
    const std::vector<svmp::FE::Real> u0 = {1.0};

    auto problem = makeScalarProblem(
        [&](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            return (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();
        },
        dt,
        u0);

    const double u_exact = 1.0 / (1.0 + lambda * dt);
    const double du = 1.0 - u_exact;

    auto run_once = [&](bool scale_dt_increments, double dt_increment_scale) -> double {
        ts_test::setVectorByDof(problem.history.uPrev(), u0);
        ts_test::setVectorByDof(problem.history.uPrev2(), u0);
        problem.history.resetCurrentToPrevious();

        svmp::FE::timestepping::NewtonOptions nopt;
        nopt.residual_op = "op";
        nopt.jacobian_op = "op";
        nopt.max_iterations = 1;
        nopt.abs_tolerance = 0.0;
        nopt.rel_tolerance = 0.0;
        nopt.step_tolerance = 0.0;
        nopt.use_line_search = false;
        nopt.scale_dt_increments = scale_dt_increments;
        nopt.dt_increment_scale = dt_increment_scale;

        svmp::FE::timestepping::NewtonSolver newton(nopt);
        svmp::FE::timestepping::NewtonWorkspace ws;
        newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
        problem.history.repack(*problem.factory);

        (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
        return scalarFromDofVector(problem.history.u());
    };

    const double u_unscaled = run_once(/*scale_dt_increments=*/false, /*dt_increment_scale=*/0.0);
    EXPECT_NEAR(u_unscaled, u_exact, 1e-13);

    const double u_scaled_by_dt = run_once(/*scale_dt_increments=*/true, /*dt_increment_scale=*/0.0);
    EXPECT_NEAR(u_scaled_by_dt, 1.0 - dt * du, 1e-13);

    const double u_scaled_explicit = run_once(/*scale_dt_increments=*/true, /*dt_increment_scale=*/0.5);
    EXPECT_NEAR(u_scaled_explicit, 1.0 - 0.5 * du, 1e-13);
}

TEST(NewtonSolver, ExhibitsQuadraticConvergenceNearSolution)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const double u_star = std::sqrt(2.0);

    auto u_val = [&]() { return scalarFromDofVector(problem.history.u()); };
    auto err = [&]() { return std::abs(u_val() - u_star); };

    const double e0 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e1 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e2 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e3 = err();

    EXPECT_LT(e1, e0);
    EXPECT_LT(e2, e1);
    EXPECT_LT(e3, e2);

    constexpr double C = 0.6;
    EXPECT_LE(e2, C * e1 * e1);
    EXPECT_LE(e3, C * e2 * e2);
}

TEST(NewtonSolver, ModifiedNewtonConvergesMoreSlowlyThanFullNewton)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto make_history = [&](const svmp::FE::backends::BackendFactory& factory,
                            svmp::FE::GlobalIndex n_dofs,
                            const std::vector<svmp::FE::Real>& u0) {
        auto history = svmp::FE::timestepping::TimeHistory::allocate(factory, n_dofs);
        history.setDt(0.1);
        history.setPrevDt(0.1);
        ts_test::setVectorByDof(history.uPrev(), u0);
        ts_test::setVectorByDof(history.uPrev2(), u0);
        history.resetCurrentToPrevious();
        return history;
    };

    auto base_problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    const auto n_dofs = base_problem.sys->dofHandler().getNumDofs();
    const std::vector<svmp::FE::Real> u0 = {1.5};

    auto history_full = make_history(*base_problem.factory, n_dofs, u0);
    auto history_mod = make_history(*base_problem.factory, n_dofs, u0);

    svmp::FE::timestepping::NewtonOptions full;
    full.residual_op = "op";
    full.jacobian_op = "op";
    full.max_iterations = 3;
    full.abs_tolerance = 0.0;
    full.rel_tolerance = 0.0;
    full.step_tolerance = 0.0;
    full.use_line_search = false;
    full.jacobian_rebuild_period = 1;

    svmp::FE::timestepping::NewtonOptions mod = full;
    mod.jacobian_rebuild_period = 100;

    svmp::FE::timestepping::NewtonSolver newton_full(full);
    svmp::FE::timestepping::NewtonSolver newton_mod(mod);

    svmp::FE::timestepping::NewtonWorkspace ws_full;
    svmp::FE::timestepping::NewtonWorkspace ws_mod;
    newton_full.allocateWorkspace(*base_problem.sys, *base_problem.factory, ws_full);
    newton_mod.allocateWorkspace(*base_problem.sys, *base_problem.factory, ws_mod);
    history_full.repack(*base_problem.factory);
    history_mod.repack(*base_problem.factory);

    (void)newton_full.solveStep(*base_problem.transient, *base_problem.linear, /*solve_time=*/history_full.dt(), history_full, ws_full);
    (void)newton_mod.solveStep(*base_problem.transient, *base_problem.linear, /*solve_time=*/history_mod.dt(), history_mod, ws_mod);

    const double u_star = std::sqrt(2.0);
    const double err_full = std::abs(scalarFromDofVector(history_full.u()) - u_star);
    const double err_mod = std::abs(scalarFromDofVector(history_mod.u()) - u_star);

    EXPECT_LT(err_full, 1e-10);
    EXPECT_GT(err_mod, 1e-7);
}

TEST(NewtonSolver, ReportContainsResidualNormsWhenNotConverged)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-20;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 2;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);
    EXPECT_TRUE(std::isfinite(rep.residual_norm0));
    EXPECT_TRUE(std::isfinite(rep.residual_norm));
    EXPECT_GT(rep.residual_norm0, 0.0);
    EXPECT_GT(rep.residual_norm0, rep.residual_norm);
    EXPECT_TRUE(rep.linear.converged);
}

TEST(NewtonSolver,
     FreeSurfaceConservativeBalanceDiagnosticAssemblesAllThreeResidualOperators)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u,
           const svmp::FE::forms::FormExpr& v) {
            return ((u * u - svmp::FE::forms::FormExpr::constant(2.0)) * v)
                .dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5},
        /*counts=*/nullptr,
        &diagnostic_counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1.0e-20;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    (void)newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    const int expected_calls = diagnostic_counts.operators.front().total;
    EXPECT_GT(expected_calls, 0);
    for (const auto& counts : diagnostic_counts.operators) {
        EXPECT_EQ(counts.total, expected_calls);
        EXPECT_EQ(counts.vector_only, counts.total);
        EXPECT_EQ(counts.matrix_only, 0);
        EXPECT_EQ(counts.matrix_and_vector, 0);
    }
}

TEST(NewtonSolver,
     FreeSurfaceDiagnosticCachesRepresentabilityButPreservesAcceptedRefreshSample)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    const std::array<svmp::FE::Real, 4> pressure_pair_diagonal{
        1.0, 1.0, 1.0, 1.0};
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        pressure_pair_diagonal,
        /*track_immutable_mesh_revisions=*/true);
    // The immutable test mesh explicitly supplies a trustworthy background
    // revision domain; the cut context independently tracks generated cut
    // content, as it does in an unfitted production problem.
    problem.sys->setCutIntegrationContext(
        std::make_shared<svmp::FE::assembly::CutIntegrationContext>());

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = true;
    options.line_search_max_iterations = 1;
    options.accepted_state_sync_invalidates_residual = true;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    // Only pressure changes during the Newton update.  The synchronized cut
    // geometry, mixed pressure/velocity pair, and surface-energy load remain
    // exact, so the second distinct state and the accepted refresh can reuse
    // the LSQR result.  The accepted refresh must nevertheless assemble and
    // emit its own full diagnostic sample: equality of current u/time/geometry
    // cannot exclude dependencies on history, parameters, auxiliary inputs,
    // or user data.
    const auto n_dofs = problem.sys->dofHandler().getNumDofs();
    std::vector<svmp::FE::Real> current(
        static_cast<std::size_t>(n_dofs), svmp::FE::Real{0.0});
    const auto velocity_begin =
        problem.sys->fieldDofOffset(problem.velocity_field);
    const auto velocity_count =
        problem.sys->fieldDofHandler(problem.velocity_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < velocity_count; ++i) {
        current[static_cast<std::size_t>(velocity_begin + i)] = 1.0;
    }
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count =
        problem.sys->fieldDofHandler(problem.pressure_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        current[static_cast<std::size_t>(pressure_begin + i)] = 1.0;
    }
    ts_test::setVectorByDof(problem.history.u(), current);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    for (const auto& counts : diagnostic_counts.operators) {
        EXPECT_EQ(counts.total, 3);
    }
    // One pair assembly executes the two off-diagonal cell kernels.  The
    // second state and accepted refresh are geometry/exact-load cache hits.
    EXPECT_EQ(diagnostic_counts.pressure_representability_pair.total, 2);
    EXPECT_NE(telemetry.find("pressure_representability_cache_hit=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_pair_matrix_state_independent=1"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_cache_rejection_reason=no_prior_sample"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_cache_rejection_reason=none"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_pair_every_assembly=0"),
              std::string::npos);
    EXPECT_NE(telemetry.find("phase='line_search_accepted_refresh'"),
              std::string::npos);
}

TEST(NewtonSolver,
     FreeSurfaceDiagnosticCachesInstalledMixedSymbolicCutVolumePair)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/true);
    constexpr std::string_view pair_op{
        "equations_diagnostic_ns_free_surface_pressure_representability_pair"};
    ASSERT_TRUE(problem.sys->meshAccess().revisionTrackingAvailable());
    ASSERT_NE(problem.sys->cutIntegrationContext(), nullptr);
    ASSERT_TRUE(problem.sys->operatorMatrixStateIndependent(
        std::string(pair_op)));

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = true;
    options.line_search_max_iterations = 1;
    options.accepted_state_sync_invalidates_residual = true;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    // The Newton update changes only pressure.  The installed cut-volume pair,
    // generated quadrature, background mesh, constraints, and surface load
    // remain identical across the two distinct sampled states.
    const auto n_dofs = problem.sys->dofHandler().getNumDofs();
    std::vector<svmp::FE::Real> current(
        static_cast<std::size_t>(n_dofs), svmp::FE::Real{0.0});
    const auto velocity_begin =
        problem.sys->fieldDofOffset(problem.velocity_field);
    const auto velocity_count =
        problem.sys->fieldDofHandler(problem.velocity_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < velocity_count; ++i) {
        current[static_cast<std::size_t>(velocity_begin + i)] = 1.0;
    }
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count =
        problem.sys->fieldDofHandler(problem.pressure_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        current[static_cast<std::size_t>(pressure_begin + i)] = 1.0;
    }
    ts_test::setVectorByDof(problem.history.u(), current);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    for (const auto& counts : diagnostic_counts.operators) {
        // Initial, line-search trial, and accepted refresh are all retained;
        // only the exact-load-guarded mixed-pair/LSQR work is reused.
        EXPECT_EQ(counts.total, 3);
    }
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_pair_matrix_state_independent=1"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_cache_hit=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_cache_rejection_reason=none"),
        std::string::npos);
}

TEST(NewtonSolver,
     FreeSurfaceDiagnosticDoesNotCacheUntrackedBackgroundMeshWithCutContext)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    const std::array<svmp::FE::Real, 4> pressure_pair_diagonal{
        1.0, 1.0, 1.0, 1.0};
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        pressure_pair_diagonal,
        /*track_immutable_mesh_revisions=*/false);
    ASSERT_FALSE(problem.sys->meshAccess().revisionTrackingAvailable());

    // A monotone cut-content revision cannot account for an untracked change
    // to the background coordinates/connectivity on which its quadrature and
    // every fitted/cut operator depend.
    problem.sys->setCutIntegrationContext(
        std::make_shared<svmp::FE::assembly::CutIntegrationContext>());

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = true;
    options.line_search_max_iterations = 1;
    options.accepted_state_sync_invalidates_residual = true;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto n_dofs = problem.sys->dofHandler().getNumDofs();
    std::vector<svmp::FE::Real> current(
        static_cast<std::size_t>(n_dofs), svmp::FE::Real{0.0});
    const auto velocity_begin =
        problem.sys->fieldDofOffset(problem.velocity_field);
    const auto velocity_count =
        problem.sys->fieldDofHandler(problem.velocity_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < velocity_count; ++i) {
        current[static_cast<std::size_t>(velocity_begin + i)] = 1.0;
    }
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count =
        problem.sys->fieldDofHandler(problem.pressure_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        current[static_cast<std::size_t>(pressure_begin + i)] = 1.0;
    }
    ts_test::setVectorByDof(problem.history.u(), current);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    for (const auto& counts : diagnostic_counts.operators) {
        EXPECT_EQ(counts.total, 3);
    }
    EXPECT_EQ(diagnostic_counts.pressure_representability_pair.total, 6);
    EXPECT_EQ(telemetry.find("pressure_representability_cache_hit=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_cache_revision_key_available=0"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_cache_rejection_reason=revision_key_unavailable"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_pair_every_assembly=0"),
              std::string::npos);
}

TEST(NewtonSolver,
     FreeSurfaceDiagnosticEveryAssemblyOverrideForcesRepresentabilityPairAssembly)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    ScopedEnvVar every_assembly(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY",
        "1");
    const std::array<svmp::FE::Real, 4> pressure_pair_diagonal{
        1.0, 1.0, 1.0, 1.0};
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        pressure_pair_diagonal);
    problem.sys->setCutIntegrationContext(
        std::make_shared<svmp::FE::assembly::CutIntegrationContext>());

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = true;
    options.line_search_max_iterations = 1;
    options.accepted_state_sync_invalidates_residual = true;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto n_dofs = problem.sys->dofHandler().getNumDofs();
    std::vector<svmp::FE::Real> current(
        static_cast<std::size_t>(n_dofs), svmp::FE::Real{0.0});
    const auto velocity_begin =
        problem.sys->fieldDofOffset(problem.velocity_field);
    const auto velocity_count =
        problem.sys->fieldDofHandler(problem.velocity_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < velocity_count; ++i) {
        current[static_cast<std::size_t>(velocity_begin + i)] = 1.0;
    }
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count =
        problem.sys->fieldDofHandler(problem.pressure_field).getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        current[static_cast<std::size_t>(pressure_begin + i)] = 1.0;
    }
    ts_test::setVectorByDof(problem.history.u(), current);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    for (const auto& counts : diagnostic_counts.operators) {
        EXPECT_EQ(counts.total, 3);
    }
    // Initial, line-search trial, and accepted refresh each assemble both
    // off-diagonal pair kernels when exhaustive qualification is requested.
    EXPECT_EQ(diagnostic_counts.pressure_representability_pair.total, 6);
    EXPECT_EQ(telemetry.find("pressure_representability_cache_hit=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_cache_rejection_reason=every_assembly_policy"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_pair_every_assembly=1"),
              std::string::npos);
}

TEST(NewtonSolver,
     FreeSurfacePressureRepresentabilityUsesMixedMatrixOnlyLsqrAndRecoversPressure)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    ScopedEnvVar enable_assembly_diagnostics(
        "SVMP_NEWTON_ASSEMBLY_DIAGNOSTICS", "1");
    auto problem =
        makePressureRepresentabilityProblem(diagnostic_counts);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;
    EXPECT_TRUE(report.converged);

    for (const auto& counts : diagnostic_counts.operators) {
        EXPECT_GT(counts.total, 0);
        EXPECT_EQ(counts.vector_only, counts.total);
        EXPECT_EQ(counts.matrix_only, 0);
        EXPECT_EQ(counts.matrix_and_vector, 0);
    }
    EXPECT_GT(
        diagnostic_counts.pressure_representability_pair.total, 0);
    EXPECT_EQ(
        diagnostic_counts.pressure_representability_pair.matrix_only,
        diagnostic_counts.pressure_representability_pair.total);
    EXPECT_EQ(
        diagnostic_counts.pressure_representability_pair.vector_only, 0);
    EXPECT_EQ(
        diagnostic_counts.pressure_representability_pair.matrix_and_vector,
        0);

    ASSERT_NE(workspace.pressure_representability_solution, nullptr);
    ASSERT_NE(workspace.pressure_representability_residual, nullptr);
    ASSERT_NE(
        workspace.pressure_representability_normal_residual, nullptr);
    const auto solution =
        workspace.pressure_representability_solution->localSpan();
    const auto velocity_dof =
        problem.sys->fieldDofOffset(problem.velocity_field);
    const auto pressure_dof =
        problem.sys->fieldDofOffset(problem.pressure_field);
    ASSERT_GE(velocity_dof, 0);
    ASSERT_GE(pressure_dof, 0);
    ASSERT_LT(static_cast<std::size_t>(velocity_dof), solution.size());
    ASSERT_LT(static_cast<std::size_t>(pressure_dof), solution.size());
    EXPECT_NEAR(solution[static_cast<std::size_t>(velocity_dof)], 0.0, 1.0e-14);
    EXPECT_NEAR(solution[static_cast<std::size_t>(pressure_dof)], 2.0, 1.0e-12);
    EXPECT_NEAR(
        workspace.pressure_representability_residual->norm(),
        0.0,
        1.0e-13);
    EXPECT_NEAR(
        workspace.pressure_representability_normal_residual->norm(),
        0.0,
        1.0e-13);

    ASSERT_NE(workspace.pressure_representability_pair_matrix, nullptr);
    const auto& pair =
        *workspace.pressure_representability_pair_matrix;
    EXPECT_NEAR(pair.getEntry(velocity_dof, velocity_dof), 0.0, 1.0e-14);
    EXPECT_NEAR(pair.getEntry(pressure_dof, pressure_dof), 0.0, 1.0e-14);
    EXPECT_NEAR(pair.getEntry(velocity_dof, pressure_dof),
                pair.getEntry(pressure_dof, velocity_dof),
                1.0e-14);
    EXPECT_GT(std::abs(pair.getEntry(velocity_dof, pressure_dof)), 1.0e-14);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_convergence=normal_equation_stationarity"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_distance_gate_applied=0"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_claimed=0"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_active_pressure_dofs=1"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_iteration_cap=4"),
              std::string::npos);
    EXPECT_NEAR(
        report.pressure_representability_residual_norm, 0.0, 1.0e-13);
    EXPECT_TRUE(report.constant_pressure_kkt_available);
    EXPECT_TRUE(
        report.constant_pressure_unit_coefficients_represent_constant);
    EXPECT_TRUE(
        report.constant_pressure_constraints_preserve_constants);
    EXPECT_NEAR(
        report.constant_pressure_kkt_pressure_jump, 2.0, 1.0e-12);
    EXPECT_NEAR(
        report.constant_pressure_kkt_volume_multiplier, -2.0, 1.0e-12);
    EXPECT_NEAR(
        report.constant_pressure_kkt_residual_norm, 0.0, 1.0e-13);
    EXPECT_LE(
        report.constant_pressure_kkt_relative_distance, 1.0e-12);
    EXPECT_LE(
        report.constant_pressure_kkt_relative_orthogonality, 1.0e-12);
    EXPECT_EQ(report.constant_pressure_kkt_reason, "available");
    EXPECT_NE(telemetry.find("constant_pressure_kkt_available=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "constant_pressure_kkt_method=closed_form_one_dimensional_pressure_trace"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("constant_pressure_kkt_force_projection_applied=0"),
        std::string::npos);
    for (const std::string op : {
             "equations_diagnostic_ns_free_surface_pressure_representability_load_virtual_work",
             "equations_diagnostic_ns_free_surface_pressure_representability_pair"}) {
        const auto op_position = telemetry.find("op='" + op + "'");
        ASSERT_NE(op_position, std::string::npos) << op;
        const auto record = telemetry.substr(op_position, 700u);
        EXPECT_NE(
            record.find("suppress_auxiliary_coupling_assembly=1"),
            std::string::npos)
            << op;
    }
}

TEST(NewtonSolver,
     PressureRepresentabilityUsesSurfacePlusGravitationalPotentialLoad)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/false,
        /*install_symbolic_cut_volume_pair=*/false,
        /*production_pressure_target=*/0.0,
        /*entry_pressure_baseline=*/0.0,
        /*use_native_mesh=*/false,
        /*surface_energy_scale=*/1.0,
        /*gravitational_potential_scale=*/0.5);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto telemetry = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(report.converged);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_FALSE(report.pressure_representability_breakdown);
    EXPECT_NEAR(
        report.pressure_representability_residual_norm, 0.0, 1.0e-13);
    ASSERT_NE(workspace.pressure_representability_solution, nullptr);
    const auto pressure_dof =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto solution =
        workspace.pressure_representability_solution->localSpan();
    ASSERT_GE(pressure_dof, 0);
    ASSERT_LT(static_cast<std::size_t>(pressure_dof), solution.size());
    EXPECT_NEAR(
        solution[static_cast<std::size_t>(pressure_dof)], 3.0, 1.0e-12);
    EXPECT_NEAR(
        report.constant_pressure_kkt_pressure_jump, 3.0, 1.0e-12);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_load=prescribed_external_pressure_plus_surface_area_variation_plus_young_wall_energy_plus_gravitational_potential"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("gravitational_potential_virtual_work_norm="),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("physical_potential_virtual_work_norm="),
        std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_load_virtual_work_norm="),
        std::string::npos);
}

TEST(NewtonSolver,
     PressureRepresentabilityIncludesPrescribedExteriorPressureLoad)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/false,
        /*install_symbolic_cut_volume_pair=*/false,
        /*production_pressure_target=*/0.0,
        /*entry_pressure_baseline=*/0.0,
        /*use_native_mesh=*/false,
        /*surface_energy_scale=*/1.0,
        /*gravitational_potential_scale=*/0.5,
        /*prescribed_external_pressure_scale=*/0.75);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_TRUE(report.converged);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_FALSE(report.pressure_representability_breakdown);
    EXPECT_NEAR(
        report.pressure_representability_residual_norm, 0.0, 1.0e-13);
    ASSERT_NE(workspace.pressure_representability_solution, nullptr);
    const auto pressure_dof =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto solution =
        workspace.pressure_representability_solution->localSpan();
    ASSERT_GE(pressure_dof, 0);
    ASSERT_LT(static_cast<std::size_t>(pressure_dof), solution.size());
    EXPECT_NEAR(
        solution[static_cast<std::size_t>(pressure_dof)], 4.5, 1.0e-12);
    EXPECT_NEAR(
        report.constant_pressure_kkt_pressure_jump, 4.5, 1.0e-12);
}

TEST(NewtonSolver,
     ConstantPressureKktReportsNonconstantPressureRemainderWithoutProjectingLoad)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr std::array<svmp::FE::Real, 4> pressure_pair_diagonal{
        1.0, 2.0, 3.0, 4.0};
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        pressure_pair_diagonal);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_LE(report.pressure_representability_relative_distance, 1.0e-10);
    EXPECT_TRUE(report.constant_pressure_kkt_available);
    EXPECT_TRUE(
        report.constant_pressure_unit_coefficients_represent_constant);
    EXPECT_TRUE(
        report.constant_pressure_constraints_preserve_constants);
    EXPECT_NEAR(
        report.constant_pressure_kkt_pressure_jump,
        -1.0 / 3.0,
        1.0e-13);
    EXPECT_NEAR(
        report.constant_pressure_kkt_volume_multiplier,
        1.0 / 3.0,
        1.0e-13);
    EXPECT_NEAR(
        report.constant_pressure_kkt_relative_distance,
        std::sqrt(1.0 / 6.0),
        1.0e-13);
    EXPECT_LE(
        report.constant_pressure_kkt_relative_orthogonality, 1.0e-13);
    EXPECT_GT(report.constant_pressure_kkt_residual_norm, 0.0);
    EXPECT_GT(report.constant_pressure_kkt_relative_distance, 0.4);
    EXPECT_NE(telemetry.find("constant_pressure_kkt_available=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find("constant_pressure_kkt_force_projection_applied=0"),
        std::string::npos);
    RecordProperty(
        "constant_pressure_kkt_nonconstant_remainder_relative_distance",
        ::testing::PrintToString(
            report.constant_pressure_kkt_relative_distance));
    RecordProperty(
        "constant_pressure_kkt_production_load_projected", 0);
}

TEST(NewtonSolver,
     AcceptedStaticConstantPressureKktDistanceGateAcceptsEquilibrium)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "constant_pressure_kkt_distance_gate_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "constant_pressure_kkt_distance_gate_backend_available", 1);
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(diagnostic_counts);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        1.0e-10;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.pressure_representability_diagnostic_sampled);
    EXPECT_TRUE(report.constant_pressure_kkt_available);
    EXPECT_TRUE(report.constant_pressure_kkt_distance_gate_applied);
    EXPECT_TRUE(report.constant_pressure_kkt_distance_gate_passed);
    EXPECT_TRUE(report.pressure_representability_distance_gate_applied);
    EXPECT_TRUE(report.pressure_representability_distance_gate_passed);
    EXPECT_LE(report.pressure_representability_relative_distance, 1.0e-10);
    EXPECT_LE(report.constant_pressure_kkt_relative_distance, 1.0e-10);
    EXPECT_NEAR(
        report.constant_pressure_kkt_max_relative_distance, 1.0e-10, 0.0);
    EXPECT_NE(
        telemetry.find(
            "diagnostic=free_surface_constant_pressure_kkt_distance_gate"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("constant_pressure_kkt_distance_gate_passed=1"),
        std::string::npos);
    EXPECT_NE(telemetry.find("constant_pressure_kkt_claimed=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find("constant_pressure_kkt_force_projection_applied=0"),
        std::string::npos);
    RecordProperty(
        "constant_pressure_kkt_distance_gate_in_range_accept_count", 1);
}

TEST(NewtonSolver,
     AcceptedStaticConstantPressureKktDistanceGateAcceptsRoundoffScaleLoad)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr std::array<svmp::FE::Real, 4> pressure_pair_diagonal{
        1.0, 2.0, 3.0, 4.0};
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        pressure_pair_diagonal,
        /*track_immutable_mesh_revisions=*/false,
        /*install_symbolic_cut_volume_pair=*/false,
        /*production_pressure_target=*/0.0,
        /*entry_pressure_baseline=*/0.0,
        /*use_native_mesh=*/false,
        /*surface_energy_scale=*/
            std::numeric_limits<svmp::FE::Real>::epsilon());

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        1.0e-10;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_TRUE(report.converged);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_TRUE(report.pressure_representability_distance_gate_applied);
    EXPECT_TRUE(report.pressure_representability_distance_gate_passed);
    EXPECT_GT(report.pressure_representability_residual_norm, 0.0);
    EXPECT_LT(report.pressure_representability_residual_norm, 1.0e-14);
    EXPECT_EQ(report.pressure_representability_relative_distance, 0.0);
    EXPECT_TRUE(report.constant_pressure_kkt_available);
    EXPECT_TRUE(report.constant_pressure_kkt_distance_gate_applied);
    EXPECT_TRUE(report.constant_pressure_kkt_distance_gate_passed);
    EXPECT_GT(report.constant_pressure_kkt_residual_norm, 0.0);
    EXPECT_LT(report.constant_pressure_kkt_residual_norm, 1.0e-14);
    EXPECT_EQ(report.constant_pressure_kkt_relative_distance, 0.0);
    EXPECT_EQ(report.constant_pressure_kkt_reason, "available");
}

TEST(NewtonSolver,
     InitialResidualOnlyCertificateSkipsJacobianAndLeavesStateUnchanged)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar ambient_pressure_initializer(
        "SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER",
        "1");
    ScopedEnvVar ambient_pressure_gate(
        "SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE",
        "invalid");
    auto problem = makePressureRepresentabilityProblem(diagnostic_counts);
    const auto entry_state =
        ts_test::getVectorByDof(problem.history.u());

    using SyncPoint = svmp::FE::timestepping::NewtonOptions::
        StateSynchronizationPoint;
    int residual_synchronizations = 0;
    int jacobian_synchronizations = 0;
    int combined_synchronizations = 0;
    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.min_iterations = 0;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initial_residual_only_certificate = true;
    options.read_static_free_surface_environment_options = false;
    options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        1.0e-10;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView&,
            SyncPoint point) {
            residual_synchronizations +=
                point == SyncPoint::ResidualAssembly ? 1 : 0;
            jacobian_synchronizations +=
                point == SyncPoint::JacobianAssembly ? 1 : 0;
            combined_synchronizations +=
                point == SyncPoint::JacobianAndResidualAssembly ? 1 : 0;
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_TRUE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.constant_pressure_kkt_available);
    EXPECT_EQ(jacobian_synchronizations, 0);
    EXPECT_EQ(combined_synchronizations, 0);
    EXPECT_GT(residual_synchronizations, 0);
    EXPECT_EQ(
        ts_test::getVectorByDof(problem.history.u()),
        entry_state);
    RecordProperty(
        "initial_residual_certificate_production_jacobian_assemblies",
        jacobian_synchronizations + combined_synchronizations);
}

TEST(NewtonSolver,
     AcceptedStaticConstantPressureKktDistanceGateRejectsNonconstantRemainder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "constant_pressure_kkt_distance_gate_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "constant_pressure_kkt_distance_gate_backend_available", 1);
    constexpr std::array<svmp::FE::Real, 4> pressure_pair_diagonal{
        1.0, 2.0, 3.0, 4.0};
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        pressure_pair_diagonal);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        0.1;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_LE(report.pressure_representability_relative_distance, 1.0e-10);
    EXPECT_FALSE(report.pressure_representability_distance_gate_applied);
    EXPECT_TRUE(report.constant_pressure_kkt_available);
    EXPECT_TRUE(report.constant_pressure_kkt_distance_gate_applied);
    EXPECT_FALSE(report.constant_pressure_kkt_distance_gate_passed);
    EXPECT_NEAR(
        report.constant_pressure_kkt_relative_distance,
        std::sqrt(1.0 / 6.0),
        1.0e-13);
    EXPECT_NE(
        telemetry.find("reason=relative_distance_exceeds_threshold"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("constant_pressure_kkt_distance_gate_passed=0"),
        std::string::npos);
    EXPECT_NE(telemetry.find("constant_pressure_kkt_claimed=0"),
              std::string::npos);
    RecordProperty(
        "constant_pressure_kkt_distance_gate_nonconstant_reject_count", 1);
    RecordProperty(
        "constant_pressure_kkt_distance_gate_nonconstant_relative_distance",
        ::testing::PrintToString(
            report.constant_pressure_kkt_relative_distance));
}

TEST(NewtonSolver,
     AcceptedStaticConstantPressureKktDistanceGateRejectsRemovedConstantMode)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "constant_pressure_kkt_distance_gate_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "constant_pressure_kkt_distance_gate_backend_available", 1);
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto pressure_dirichlet =
        std::make_shared<FixedPressureDirichletState>();
    pressure_dirichlet->value = 0.0;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        pressure_dirichlet);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        0.5;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.pressure_representability_diagnostic_sampled);
    EXPECT_FALSE(report.constant_pressure_kkt_available);
    EXPECT_FALSE(
        report.constant_pressure_constraints_preserve_constants);
    EXPECT_EQ(
        report.constant_pressure_kkt_reason,
        "pressure_constraints_do_not_preserve_constants");
    EXPECT_TRUE(report.constant_pressure_kkt_distance_gate_applied);
    EXPECT_FALSE(report.constant_pressure_kkt_distance_gate_passed);
    EXPECT_NE(
        telemetry.find("reason=constant_pressure_kkt_unavailable"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("constant_pressure_kkt_distance_gate_passed=0"),
        std::string::npos);
    RecordProperty(
        "constant_pressure_kkt_distance_gate_removed_mode_reject_count", 1);
}

TEST(NewtonSolver,
     AcceptedStaticConstantPressureKktDistanceGateRejectsInvalidThresholds)
{
    svmp::FE::timestepping::NewtonOptions negative_options;
    negative_options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        -1.0;
    EXPECT_THROW(
        svmp::FE::timestepping::NewtonSolver(std::move(negative_options)),
        svmp::FE::InvalidArgumentException);

    svmp::FE::timestepping::NewtonOptions nonfinite_options;
    nonfinite_options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        svmp::FE::timestepping::NewtonSolver(std::move(nonfinite_options)),
        svmp::FE::InvalidArgumentException);
}

TEST(NewtonSolver,
     FreeSurfacePressureRepresentabilityFinitePrecisionLsqrCanExceedPressureDimension)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr std::array<svmp::FE::Real, 4> singular_values{
        1.0,
        1.850641438418369e-5,
        4.214402890007242e-10,
        1.1288280139454186e-10,
    };
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/singular_values);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;
    EXPECT_TRUE(report.converged);

    ASSERT_NE(workspace.pressure_representability_pair_matrix, nullptr);
    ASSERT_NE(workspace.pressure_representability_load, nullptr);
    const auto velocity_begin =
        problem.sys->fieldDofOffset(problem.velocity_field);
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    ASSERT_EQ(
        problem.sys->fieldDofHandler(problem.velocity_field).getNumDofs(),
        4);
    ASSERT_EQ(
        problem.sys->fieldDofHandler(problem.pressure_field).getNumDofs(),
        4);
    const auto load = workspace.pressure_representability_load->localSpan();
    const auto& pair = *workspace.pressure_representability_pair_matrix;
    for (svmp::FE::GlobalIndex i = 0; i < 4; ++i) {
        ASSERT_LT(static_cast<std::size_t>(velocity_begin + i), load.size());
        ASSERT_LT(static_cast<std::size_t>(pressure_begin + i), load.size());
        EXPECT_NEAR(
            load[static_cast<std::size_t>(velocity_begin + i)], 1.0, 1.0e-13);
        EXPECT_NEAR(
            load[static_cast<std::size_t>(pressure_begin + i)], 0.0, 1.0e-14);
        for (svmp::FE::GlobalIndex j = 0; j < 4; ++j) {
            const auto expected = i == j
                                      ? singular_values[static_cast<std::size_t>(i)]
                                      : 0.0;
            EXPECT_NEAR(
                pair.getEntry(velocity_begin + i, pressure_begin + j),
                expected,
                1.0e-15);
            EXPECT_NEAR(
                pair.getEntry(pressure_begin + i, velocity_begin + j),
                expected,
                1.0e-15);
            EXPECT_NEAR(
                pair.getEntry(velocity_begin + i, velocity_begin + j),
                0.0,
                1.0e-15);
            EXPECT_NEAR(
                pair.getEntry(pressure_begin + i, pressure_begin + j),
                0.0,
                1.0e-15);
        }
    }

    const auto telemetry_value = [&](const std::string& name) {
        const auto token = name + "=";
        const auto token_position = telemetry.find(token);
        if (token_position == std::string::npos) {
            ADD_FAILURE() << "Missing telemetry field '" << name << "'";
            return std::numeric_limits<double>::quiet_NaN();
        }
        const auto value_begin = token_position + token.size();
        const auto value_end = telemetry.find_first_of(" \r\n", value_begin);
        return std::stod(
            telemetry.substr(value_begin, value_end - value_begin));
    };

    EXPECT_EQ(
        telemetry_value("pressure_representability_active_pressure_dofs"),
        4.0);
    EXPECT_EQ(
        telemetry_value("pressure_representability_iteration_cap"), 16.0);
    const double iterations =
        telemetry_value("pressure_representability_iterations");
    EXPECT_EQ(iterations, 16.0);
    EXPECT_EQ(
        telemetry_value(
            "pressure_representability_normal_residual_refinements"),
        1.0);
    EXPECT_EQ(
        telemetry_value("pressure_representability_converged"), 1.0);
    EXPECT_EQ(
        telemetry_value("pressure_representability_breakdown"), 0.0);
    const double relative_normal_residual = telemetry_value(
        "pressure_representability_relative_normal_residual");
    EXPECT_TRUE(std::isfinite(relative_normal_residual));
    EXPECT_LE(relative_normal_residual, 1.0e-10);
}

TEST(NewtonSolver,
     PressureRepresentabilityPrimalResidualUsesItsOwnLoadScale)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr std::array<svmp::FE::Real, 4> singular_values{
        1.0e16,
        0.0,
        0.0,
        0.0,
    };
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/singular_values);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_TRUE(report.converged);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_FALSE(report.pressure_representability_breakdown);
    EXPECT_NEAR(
        report.pressure_representability_residual_norm,
        std::sqrt(3.0),
        1.0e-12);
    EXPECT_NEAR(
        report.pressure_representability_relative_distance,
        std::sqrt(3.0) / 2.0,
        1.0e-12);
}

TEST(NewtonSolver,
     FreeSurfacePressureRepresentabilityRemovesArtificialConstrainedPressureRow)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto pressure_dirichlet =
        std::make_shared<FixedPressureDirichletState>();
    pressure_dirichlet->value = 0.0;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        pressure_dirichlet);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;
    EXPECT_TRUE(report.converged);
    ASSERT_NE(workspace.pressure_representability_pair_matrix, nullptr);
    ASSERT_NE(workspace.pressure_representability_load, nullptr);
    ASSERT_NE(workspace.pressure_representability_residual, nullptr);
    const auto velocity_dof =
        problem.sys->fieldDofOffset(problem.velocity_field);
    const auto pressure_dof =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto& pair =
        *workspace.pressure_representability_pair_matrix;

    // Without explicit zeroRows(..., false), constrained assembly leaves an
    // artificial unit pressure row and LSQR would incorrectly cancel f using
    // a constrained coefficient. The physical reduced operator is zero here:
    // p remains fixed at zero and the nonzero surface load is irreducible.
    EXPECT_NEAR(pair.getEntry(pressure_dof, pressure_dof), 0.0, 1.0e-14);
    EXPECT_NEAR(pair.getEntry(velocity_dof, pressure_dof), 0.0, 1.0e-14);
    EXPECT_NEAR(pair.getEntry(pressure_dof, velocity_dof), 0.0, 1.0e-14);
    EXPECT_GT(workspace.pressure_representability_load->norm(), 1.0e-12);
    EXPECT_NEAR(
        workspace.pressure_representability_residual->norm(),
        workspace.pressure_representability_load->norm(),
        1.0e-13);
    EXPECT_NEAR(
        workspace.pressure_representability_normal_residual->norm(),
        0.0,
        1.0e-14);
    EXPECT_NEAR(
        workspace.pressure_representability_solution->norm(),
        0.0,
        1.0e-14);
    // A zero reduced operator makes the load range-orthogonal: LSQR reaches
    // normal-equation stationarity immediately even though the entire load is
    // irreducible.  Telemetry must therefore make explicit that no physical
    // distance gate or representability claim has been applied.
    EXPECT_NE(telemetry.find("pressure_representability_relative_residual=1"),
              std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_iterations=0"),
              std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_converged=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_convergence=normal_equation_stationarity"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_distance_gate_applied=0"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_claimed=0"),
              std::string::npos);
    EXPECT_FALSE(report.constant_pressure_kkt_available);
    EXPECT_TRUE(
        report.constant_pressure_unit_coefficients_represent_constant);
    EXPECT_FALSE(
        report.constant_pressure_constraints_preserve_constants);
    EXPECT_EQ(
        report.constant_pressure_kkt_reason,
        "pressure_constraints_do_not_preserve_constants");
    EXPECT_NE(telemetry.find("constant_pressure_kkt_available=0"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "constant_pressure_kkt_reason=pressure_constraints_do_not_preserve_constants"),
        std::string::npos);
}

TEST(NewtonSolver,
     AcceptedStaticPressureRepresentabilityDistanceGateAcceptsInRangeState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "pressure_representability_distance_gate_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "pressure_representability_distance_gate_backend_available", 1);
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(diagnostic_counts);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.pressure_representability_diagnostic_sampled);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_FALSE(report.pressure_representability_breakdown);
    EXPECT_TRUE(report.pressure_representability_distance_gate_applied);
    EXPECT_TRUE(report.pressure_representability_distance_gate_passed);
    EXPECT_LE(report.pressure_representability_relative_distance, 1.0e-10);
    EXPECT_NE(
        telemetry.find(
            "diagnostic=free_surface_pressure_representability_distance_gate"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_distance_gate_passed=1"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_claimed=1"),
              std::string::npos);
    RecordProperty(
        "pressure_representability_distance_gate_in_range_accept_count", 1);
    RecordProperty(
        "pressure_representability_distance_gate_in_range_relative_distance",
        ::testing::PrintToString(
            report.pressure_representability_relative_distance));
}

TEST(NewtonSolver,
     AcceptedStaticPressureRepresentabilityDistanceGateRejectsOutOfRangeStationaryState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "pressure_representability_distance_gate_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "pressure_representability_distance_gate_backend_available", 1);
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto pressure_dirichlet =
        std::make_shared<FixedPressureDirichletState>();
    pressure_dirichlet->value = 0.0;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        pressure_dirichlet);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        0.5;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.pressure_representability_diagnostic_sampled);
    EXPECT_TRUE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_converged);
    EXPECT_FALSE(report.pressure_representability_breakdown);
    EXPECT_TRUE(report.pressure_representability_distance_gate_applied);
    EXPECT_FALSE(report.pressure_representability_distance_gate_passed);
    EXPECT_NEAR(
        report.pressure_representability_relative_distance, 1.0, 1.0e-13);
    EXPECT_NE(
        telemetry.find("reason=relative_distance_exceeds_threshold"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_distance_gate_passed=0"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_claimed=0"),
              std::string::npos);
    RecordProperty(
        "pressure_representability_distance_gate_out_of_range_reject_count",
        1);
    RecordProperty(
        "pressure_representability_distance_gate_out_of_range_relative_distance",
        ::testing::PrintToString(
            report.pressure_representability_relative_distance));
}

TEST(NewtonSolver,
     AcceptedStaticPressureRepresentabilityDistanceGateRejectsUnavailableDiagnostic)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "pressure_representability_distance_gate_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "pressure_representability_distance_gate_backend_available", 1);
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto pressure_dirichlet =
        std::make_shared<FixedPressureDirichletState>();
    pressure_dirichlet->value = 1.0;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        pressure_dirichlet);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        0.5;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.pressure_representability_diagnostic_sampled);
    EXPECT_FALSE(report.pressure_representability_available);
    EXPECT_TRUE(report.pressure_representability_distance_gate_applied);
    EXPECT_FALSE(report.pressure_representability_distance_gate_passed);
    EXPECT_EQ(report.pressure_representability_reason,
              "nonzero_pressure_constraint_inhomogeneity");
    EXPECT_NE(telemetry.find("reason=diagnostic_unavailable"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_available=0"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_distance_gate_passed=0"),
        std::string::npos);
    RecordProperty(
        "pressure_representability_distance_gate_unavailable_reject_count",
        1);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerRequiresRepresentabilityDistanceGate)
{
    svmp::FE::timestepping::NewtonOptions options;
    options.initialize_static_compatible_free_surface_pressure = true;
    EXPECT_THROW(
        svmp::FE::timestepping::NewtonSolver(std::move(options)),
        svmp::FE::InvalidArgumentException);
    RecordProperty(
        "static_compatible_pressure_initializer_missing_gate_reject_count",
        1);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerAddsToBaselineAndPreservesCommittedState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "static_compatible_pressure_initializer_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "static_compatible_pressure_initializer_backend_available", 1);
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/true,
        /*production_pressure_target=*/4.0,
        /*entry_pressure_baseline=*/2.0);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initialize_static_compatible_free_surface_pressure = true;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);
    problem.history.ensureSecondOrderState(*problem.factory);

    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count = problem.sys
                                    ->fieldDofHandler(problem.pressure_field)
                                    .getNumDofs();
    auto previous = ts_test::getVectorByDof(problem.history.uPrev());
    auto previous2 = ts_test::getVectorByDof(problem.history.uPrev2());
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        previous[static_cast<std::size_t>(pressure_begin + i)] = 11.0;
        previous2[static_cast<std::size_t>(pressure_begin + i)] = -7.0;
    }
    ts_test::setVectorByDof(problem.history.uPrev(), previous);
    ts_test::setVectorByDof(problem.history.uPrev2(), previous2);
    std::vector<svmp::FE::Real> u_dot(previous.size(), 13.0);
    std::vector<svmp::FE::Real> u_ddot(previous.size(), -17.0);
    ts_test::setVectorByDof(problem.history.uDot(), u_dot);
    ts_test::setVectorByDof(problem.history.uDDot(), u_ddot);

    const auto entry_previous =
        ts_test::getVectorByDof(problem.history.uPrev());
    const auto entry_previous2 =
        ts_test::getVectorByDof(problem.history.uPrev2());
    const auto entry_u_dot =
        ts_test::getVectorByDof(problem.history.uDot());
    const auto entry_u_ddot =
        ts_test::getVectorByDof(problem.history.uDDot());

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_TRUE(report.converged);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_TRUE(report.static_compatible_pressure_initializer_requested);
    EXPECT_TRUE(report.static_compatible_pressure_initializer_applied);
    EXPECT_TRUE(report.static_compatible_pressure_initializer_passed);
    EXPECT_EQ(report.static_compatible_pressure_initializer_reason,
              "additive_initial_guess_within_threshold");
    EXPECT_TRUE(workspace.static_compatible_pressure_initialized);
    EXPECT_LE(report.pressure_representability_relative_distance, 1.0e-10);
    EXPECT_NE(
        telemetry.find(
            "diagnostic=free_surface_static_compatible_pressure_initializer"),
        std::string::npos);
    EXPECT_NE(telemetry.find("force_projection_applied=0"),
              std::string::npos);
    EXPECT_NE(telemetry.find("production_capillary_operator_changed=0"),
              std::string::npos);
    EXPECT_NE(telemetry.find("balanced_force_evidence=0"),
              std::string::npos);
    EXPECT_NE(telemetry.find("pressure_update=additive"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_increment=conservative_balance_residual_correction"),
        std::string::npos);
    EXPECT_NE(telemetry.find("existing_pressure_baseline_preserved=1"),
              std::string::npos);
    EXPECT_NE(telemetry.find("committed_history_or_rate_slots_mutated=0"),
              std::string::npos);

    const auto verify_current_state =
        [&](svmp::FE::backends::GenericVector& vector) {
        const auto values = ts_test::getVectorByDof(vector);
        const auto velocity_begin =
            problem.sys->fieldDofOffset(problem.velocity_field);
        const auto velocity_count = problem.sys
                                        ->fieldDofHandler(
                                            problem.velocity_field)
                                        .getNumDofs();
        const auto pressure_begin =
            problem.sys->fieldDofOffset(problem.pressure_field);
        const auto pressure_count = problem.sys
                                        ->fieldDofHandler(
                                            problem.pressure_field)
                                        .getNumDofs();
        for (svmp::FE::GlobalIndex i = 0; i < velocity_count; ++i) {
            EXPECT_NEAR(
                values[static_cast<std::size_t>(velocity_begin + i)],
                1.0,
                1.0e-13);
        }
        for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
            EXPECT_NEAR(
                values[static_cast<std::size_t>(pressure_begin + i)],
                4.0,
                1.0e-12);
        }
    };
    verify_current_state(problem.history.u());
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
              entry_previous);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
              entry_previous2);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uDot()),
              entry_u_dot);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uDDot()),
              entry_u_ddot);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto repeated_report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto repeated_stderr = testing::internal::GetCapturedStderr();
    const auto repeated_stdout = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(repeated_report.converged);
    EXPECT_FALSE(
        repeated_report.static_compatible_pressure_initializer_applied);
    EXPECT_TRUE(repeated_report.static_compatible_pressure_initializer_passed);
    EXPECT_EQ(repeated_report.static_compatible_pressure_initializer_reason,
              "already_initialized");
    EXPECT_NE(
        (repeated_stdout + repeated_stderr).find("reason=already_initialized"),
        std::string::npos);
    verify_current_state(problem.history.u());
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
              entry_previous);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
              entry_previous2);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uDot()),
              entry_u_dot);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uDDot()),
              entry_u_ddot);

    RecordProperty(
        "static_compatible_pressure_initializer_symbolic_assembly_apply_count",
        1);
    RecordProperty(
        "static_compatible_pressure_initializer_relative_distance",
        ::testing::PrintToString(
            report.pressure_representability_relative_distance));
    RecordProperty(
        "static_compatible_pressure_initializer_nonpressure_preserved", 1);
    RecordProperty(
        "static_compatible_pressure_initializer_existing_baseline_preserved",
        1);
    RecordProperty(
        "static_compatible_pressure_initializer_committed_state_preserved", 1);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerRollsBackNonlinearFailureAndCanRetry)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/true,
        /*production_pressure_target=*/7.0,
        /*entry_pressure_baseline=*/2.0);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initialize_static_compatible_free_surface_pressure = true;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    svmp::FE::timestepping::NewtonSolver one_iteration_newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    one_iteration_newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto entry_current =
        ts_test::getVectorByDof(problem.history.u());
    const auto entry_previous =
        ts_test::getVectorByDof(problem.history.uPrev());
    const auto entry_previous2 =
        ts_test::getVectorByDof(problem.history.uPrev2());

    const auto failed_report = one_iteration_newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    EXPECT_FALSE(failed_report.converged);
    EXPECT_EQ(failed_report.iterations, 1);
    EXPECT_TRUE(
        failed_report.static_compatible_pressure_initializer_applied);
    EXPECT_FALSE(workspace.static_compatible_pressure_initialized);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.u()), entry_current);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
              entry_previous);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
              entry_previous2);

    options.max_iterations = 2;
    svmp::FE::timestepping::NewtonSolver retry_newton(options);
    const auto retry_report = retry_newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    EXPECT_TRUE(retry_report.converged);
    EXPECT_EQ(retry_report.iterations, 1);
    EXPECT_TRUE(
        retry_report.static_compatible_pressure_initializer_applied);
    EXPECT_TRUE(workspace.static_compatible_pressure_initialized);

    const auto retry_state =
        ts_test::getVectorByDof(problem.history.u());
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count = problem.sys
                                    ->fieldDofHandler(problem.pressure_field)
                                    .getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        EXPECT_NEAR(
            retry_state[static_cast<std::size_t>(pressure_begin + i)],
            7.0,
            1.0e-12);
    }
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
              entry_previous);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
              entry_previous2);
    RecordProperty(
        "static_compatible_pressure_initializer_nonlinear_failure_rollback",
        1);
    RecordProperty(
        "static_compatible_pressure_initializer_retry_apply_count", 1);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerRollsBackSynchronizationExceptionAndCanRetry)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/true,
        /*production_pressure_target=*/4.0,
        /*entry_pressure_baseline=*/2.0);
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);

    svmp::FE::systems::AuxiliaryStateSpec auxiliary_spec;
    auxiliary_spec.name = "initializer_rollback_auxiliary";
    auxiliary_spec.size = 2;
    const std::array<svmp::FE::Real, 2> auxiliary_initial{
        svmp::FE::Real{3.0}, svmp::FE::Real{-4.0}};
    problem.sys->auxiliaryStateManager().registerBlock(
        auxiliary_spec,
        /*entity_count=*/1,
        auxiliary_initial);
    const auto entry_auxiliary =
        problem.sys->checkpointAuxiliaryState();
    auto& entry_bordered_setup = problem.sys->borderedCoupling();
    entry_bordered_setup.active = false;
    entry_bordered_setup.n_aux = 1;
    entry_bordered_setup.n_field_dofs =
        static_cast<std::size_t>(
            problem.sys->dofHandler().getNumDofs());
    entry_bordered_setup.g = {svmp::FE::Real{5.0}};
    const auto entry_bordered = entry_bordered_setup;

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool throw_on_initialized_state = true;
    bool throw_on_first_restoration = true;
    int injected_failures = 0;
    int restored_effect_count = 0;
    bool restored_callback_saw_entry_state = false;
    double synchronized_pressure_effect =
        std::numeric_limits<double>::quiet_NaN();

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initialize_static_compatible_free_surface_pressure = true;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    options.synchronize_state =
        [&, pressure_begin](
            const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            if (pressure_begin < 0 ||
                static_cast<std::size_t>(pressure_begin) >=
                    state.u.size()) {
                return;
            }
            const double pressure = static_cast<double>(
                state.u[static_cast<std::size_t>(pressure_begin)]);
            if (point == SyncPoint::RestoredNonlinearState) {
                synchronized_pressure_effect = pressure;
                ++restored_effect_count;
                restored_callback_saw_entry_state =
                    problem.sys->checkpointAuxiliaryState() ==
                        entry_auxiliary &&
                    problem.sys->borderedCoupling().active ==
                        entry_bordered.active &&
                    problem.sys->borderedCoupling().n_aux ==
                        entry_bordered.n_aux &&
                    problem.sys->borderedCoupling().g ==
                        entry_bordered.g;
                if (throw_on_first_restoration) {
                    throw_on_first_restoration = false;
                    auto& auxiliary = problem.sys
                                          ->auxiliaryStateManager()
                                          .getBlock(
                                              "initializer_rollback_auxiliary");
                    auxiliary.work()[0] = 90.0;
                    auxiliary.work()[1] = 91.0;
                    auto& bordered =
                        problem.sys->borderedCoupling();
                    bordered.active = true;
                    bordered.g = {-99.0};
                    throw std::runtime_error(
                        "synthetic first restoration callback failure");
                }
                return;
            }
            if (point != SyncPoint::AcceptedNonlinearState) {
                return;
            }

            // Model a reversible generated-state installation outside the
            // algebraic vectors.  The initializer's speculative callback
            // changes this value, and RestoredNonlinearState must rebuild it
            // from the pre-initializer pressure even when that callback threw.
            synchronized_pressure_effect = pressure;
            if (!throw_on_initialized_state ||
                std::abs(pressure - 4.0) > 1.0e-12) {
                return;
            }
            auto& auxiliary = problem.sys
                                  ->auxiliaryStateManager()
                                  .getBlock(
                                      "initializer_rollback_auxiliary");
            auxiliary.work()[0] = 70.0;
            auxiliary.work()[1] = 71.0;
            auto& bordered = problem.sys->borderedCoupling();
            bordered.active = true;
            bordered.g = {-79.0};
            throw_on_initialized_state = false;
            ++injected_failures;
            throw std::runtime_error(
                "synthetic post-pressure-initial-guess synchronization failure");
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);
    const auto entry_current =
        ts_test::getVectorByDof(problem.history.u());
    const auto entry_previous =
        ts_test::getVectorByDof(problem.history.uPrev());
    const auto entry_previous2 =
        ts_test::getVectorByDof(problem.history.uPrev2());

    EXPECT_THROW(
        (void)newton.solveStep(
            *problem.transient,
            *problem.linear,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            workspace),
        std::runtime_error);
    EXPECT_EQ(injected_failures, 1);
    EXPECT_FALSE(workspace.static_compatible_pressure_initialized);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.u()), entry_current);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
              entry_previous);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
              entry_previous2);
    EXPECT_EQ(restored_effect_count, 2);
    EXPECT_TRUE(restored_callback_saw_entry_state);
    EXPECT_NEAR(synchronized_pressure_effect, 2.0, 1.0e-13);
    EXPECT_EQ(problem.sys->checkpointAuxiliaryState(),
              entry_auxiliary);
    EXPECT_EQ(problem.sys->borderedCoupling().active,
              entry_bordered.active);
    EXPECT_EQ(problem.sys->borderedCoupling().n_aux,
              entry_bordered.n_aux);
    EXPECT_EQ(problem.sys->borderedCoupling().g,
              entry_bordered.g);

    const auto retry_report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    EXPECT_TRUE(retry_report.converged);
    EXPECT_TRUE(
        retry_report.static_compatible_pressure_initializer_applied);
    EXPECT_TRUE(workspace.static_compatible_pressure_initialized);
    EXPECT_EQ(injected_failures, 1);
    EXPECT_EQ(restored_effect_count, 2);
    EXPECT_NEAR(synchronized_pressure_effect, 4.0, 1.0e-13);
    RecordProperty(
        "static_compatible_pressure_initializer_sync_exception_rollback",
        1);
    RecordProperty(
        "static_compatible_pressure_initializer_sync_exception_retry", 1);
    RecordProperty(
        "static_compatible_pressure_initializer_reversible_effect_rollback",
        1);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerRetriesAfterConvergedAttemptRejection)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/true,
        /*production_pressure_target=*/4.0,
        /*entry_pressure_baseline=*/2.0);

    svmp::FE::timestepping::TimeLoopOptions loop_options;
    loop_options.t0 = 0.0;
    loop_options.t_end = 0.1;
    loop_options.dt = 0.1;
    loop_options.max_steps = 1;
    loop_options.adjust_last_step = false;
    loop_options.scheme =
        svmp::FE::timestepping::SchemeKind::BackwardEuler;
    loop_options.step_controller =
        std::make_shared<RejectFirstConvergedAttemptController>();
    loop_options.newton.residual_op = "op";
    loop_options.newton.jacobian_op = "op";
    loop_options.newton.max_iterations = 1;
    loop_options.newton.abs_tolerance = 1.0e-13;
    loop_options.newton.rel_tolerance = 0.0;
    loop_options.newton.use_line_search = false;
    loop_options.newton
        .initialize_static_compatible_free_surface_pressure = true;
    loop_options.newton
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;

    std::vector<bool> initializer_applied;
    std::vector<std::string> initializer_reasons;
    int candidate_checks = 0;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done =
        [&](const svmp::FE::timestepping::TimeHistory&,
            const svmp::FE::timestepping::NewtonReport& report) {
            EXPECT_TRUE(report.converged);
            initializer_applied.push_back(
                report.static_compatible_pressure_initializer_applied);
            initializer_reasons.push_back(
                report.static_compatible_pressure_initializer_reason);
        };
    callbacks.on_before_step_accept =
        [&](svmp::FE::timestepping::TimeHistory&,
            const svmp::FE::timestepping::NewtonReport&) {
            ++candidate_checks;
            return candidate_checks > 1;
        };

    svmp::FE::timestepping::TimeLoop loop(loop_options);
    const auto loop_report = loop.run(
        *problem.transient,
        *problem.factory,
        *problem.linear,
        problem.history,
        callbacks);

    ASSERT_TRUE(loop_report.success) << loop_report.message;
    EXPECT_EQ(candidate_checks, 2);
    ASSERT_EQ(initializer_applied.size(), 2u);
    EXPECT_TRUE(initializer_applied[0]);
    EXPECT_TRUE(initializer_applied[1]);
    ASSERT_EQ(initializer_reasons.size(), 2u);
    EXPECT_EQ(initializer_reasons[0],
              "additive_initial_guess_within_threshold");
    EXPECT_EQ(initializer_reasons[1],
              "additive_initial_guess_within_threshold");
    EXPECT_NE(
        initializer_reasons[1],
        "already_initialized");

    const auto accepted_state =
        ts_test::getVectorByDof(problem.history.uPrev());
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count = problem.sys
                                    ->fieldDofHandler(problem.pressure_field)
                                    .getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        EXPECT_NEAR(
            accepted_state[
                static_cast<std::size_t>(pressure_begin + i)],
            4.0,
            1.0e-12);
    }
    RecordProperty(
        "static_compatible_pressure_initializer_attempt_retry_apply_count",
        2);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerClosesGeometryOpenedByRestoration)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#elif !defined(SVMP_FE_WITH_MESH) || !SVMP_FE_WITH_MESH
    GTEST_SKIP()
        << "Geometry transaction regression requires native mesh support";
#else
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/false,
        /*install_symbolic_cut_volume_pair=*/false,
        /*production_pressure_target=*/2.0,
        /*entry_pressure_baseline=*/1.0,
        /*use_native_mesh=*/true);
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    bool throw_on_initialized_state = true;
    int restored_callbacks = 0;
    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 2;
    options.abs_tolerance = 1.0e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initialize_static_compatible_free_surface_pressure = true;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    options.synchronize_state =
        [&, pressure_begin](
            const svmp::FE::systems::SystemStateView& state,
            SyncPoint point) {
            if (point == SyncPoint::RestoredNonlinearState) {
                ++restored_callbacks;
                problem.sys->beginMeshCoordinateTransaction();
                EXPECT_TRUE(
                    problem.sys->meshCoordinateTransactionActive());
                return;
            }
            if (point != SyncPoint::AcceptedNonlinearState ||
                !throw_on_initialized_state ||
                pressure_begin < 0 ||
                static_cast<std::size_t>(pressure_begin) >=
                    state.u.size()) {
                return;
            }
            const double pressure = static_cast<double>(
                state.u[static_cast<std::size_t>(pressure_begin)]);
            if (std::abs(pressure - 2.0) <= 1.0e-12) {
                throw_on_initialized_state = false;
                throw std::runtime_error(
                    "synthetic initialized-state failure before geometry restoration");
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    EXPECT_THROW(
        (void)newton.solveStep(
            *problem.transient,
            *problem.linear,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            workspace),
        std::runtime_error);
    EXPECT_EQ(restored_callbacks, 1);
    EXPECT_FALSE(
        problem.sys->meshCoordinateTransactionActive());
    EXPECT_FALSE(workspace.static_compatible_pressure_initialized);

    const auto retry_report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    EXPECT_TRUE(retry_report.converged);
    EXPECT_TRUE(
        retry_report.static_compatible_pressure_initializer_applied);
    EXPECT_FALSE(
        problem.sys->meshCoordinateTransactionActive());
#endif
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerSurvivesDiscardedVsvoReferenceAndRetry)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/true,
        /*production_pressure_target=*/6.0,
        /*entry_pressure_baseline=*/2.0);

    svmp::FE::timestepping::VSVO_BDF_ControllerOptions controller_options;
    controller_options.abs_tol = 1.0e6;
    controller_options.rel_tol = 0.0;
    controller_options.min_order = 1;
    controller_options.max_order = 1;
    controller_options.initial_order = 1;
    controller_options.max_retries = 1;
    controller_options.safety = 1.0;
    controller_options.min_factor = 1.0;
    controller_options.max_factor = 1.0;
    controller_options.min_dt = 0.1;
    controller_options.max_dt = 0.1;
    controller_options.pi_alpha = 0.0;
    controller_options.pi_beta = 0.0;
    controller_options.increase_order_threshold = 0.0;

    svmp::FE::timestepping::TimeLoopOptions loop_options;
    loop_options.t0 = 0.0;
    loop_options.t_end = 0.1;
    loop_options.dt = 0.1;
    loop_options.max_steps = 1;
    loop_options.adjust_last_step = false;
    loop_options.scheme =
        svmp::FE::timestepping::SchemeKind::VSVO_BDF;
    loop_options.step_controller = std::make_shared<
        svmp::FE::timestepping::VSVO_BDF_Controller>(
        controller_options);
    loop_options.newton.residual_op = "op";
    loop_options.newton.jacobian_op = "op";
    loop_options.newton.max_iterations = 2;
    loop_options.newton.abs_tolerance = 1.0e-13;
    loop_options.newton.rel_tolerance = 0.0;
    loop_options.newton.use_line_search = false;
    loop_options.newton
        .initialize_static_compatible_free_surface_pressure = true;
    loop_options.newton
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;

    std::vector<bool> candidate_initializer_applied;
    std::vector<std::string> candidate_initializer_reasons;
    int candidate_checks = 0;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done =
        [&](const svmp::FE::timestepping::TimeHistory&,
            const svmp::FE::timestepping::NewtonReport& report) {
            EXPECT_TRUE(report.converged);
            candidate_initializer_applied.push_back(
                report.static_compatible_pressure_initializer_applied);
            candidate_initializer_reasons.push_back(
                report.static_compatible_pressure_initializer_reason);
        };
    callbacks.on_before_step_accept =
        [&](svmp::FE::timestepping::TimeHistory&,
            const svmp::FE::timestepping::NewtonReport&) {
            ++candidate_checks;
            return candidate_checks > 1;
        };

    svmp::FE::timestepping::TimeLoop loop(loop_options);
    const auto loop_report = loop.run(
        *problem.transient,
        *problem.factory,
        *problem.linear,
        problem.history,
        callbacks);

    ASSERT_TRUE(loop_report.success) << loop_report.message;
    EXPECT_EQ(candidate_checks, 2);
    ASSERT_EQ(candidate_initializer_applied.size(), 2u);
    EXPECT_TRUE(candidate_initializer_applied[0]);
    EXPECT_TRUE(candidate_initializer_applied[1]);
    ASSERT_EQ(candidate_initializer_reasons.size(), 2u);
    EXPECT_EQ(candidate_initializer_reasons[0],
              "additive_initial_guess_within_threshold");
    EXPECT_EQ(candidate_initializer_reasons[1],
              "additive_initial_guess_within_threshold");

    const auto accepted_state =
        ts_test::getVectorByDof(problem.history.uPrev());
    const auto pressure_begin =
        problem.sys->fieldDofOffset(problem.pressure_field);
    const auto pressure_count = problem.sys
                                    ->fieldDofHandler(problem.pressure_field)
                                    .getNumDofs();
    for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
        EXPECT_NEAR(
            accepted_state[
                static_cast<std::size_t>(pressure_begin + i)],
            6.0,
            1.0e-12);
    }
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerExternalFixedPointRollbackExposesUnconsumedState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/true,
        /*production_pressure_target=*/7.0,
        /*entry_pressure_baseline=*/2.0);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    svmp::FE::timestepping::NewtonWorkspace workspace;
    bool inject_outer_failure = true;
    bool restored_callback_saw_unconsumed_state = false;
    std::vector<bool> outer_entry_initializer_states;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 2;
    options.abs_tolerance = 1.0e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initialize_static_compatible_free_surface_pressure = true;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    options
        .accepted_static_constant_pressure_kkt_max_relative_distance =
        1.0e-10;
    options.external_state_fixed_point.enabled = true;
    options.external_state_fixed_point.max_iterations = 2;
    options.synchronize_state =
        [&](const svmp::FE::systems::SystemStateView&, SyncPoint point) {
            if (point == SyncPoint::OuterFixedPointState) {
                const bool initialized =
                    workspace.static_compatible_pressure_initialized;
                outer_entry_initializer_states.push_back(initialized);
                if (initialized && inject_outer_failure) {
                    inject_outer_failure = false;
                    throw std::runtime_error(
                        "synthetic outer refresh failure after initializer");
                }
                return;
            }
            if (point == SyncPoint::RestoredOuterFixedPointState) {
                restored_callback_saw_unconsumed_state =
                    !workspace.static_compatible_pressure_initialized;
            }
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    EXPECT_THROW(
        (void)newton.solveStep(
            *problem.transient,
            *problem.linear,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            workspace),
        std::runtime_error);
    EXPECT_TRUE(restored_callback_saw_unconsumed_state);
    EXPECT_FALSE(workspace.static_compatible_pressure_initialized);

    const auto retry_report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    EXPECT_TRUE(retry_report.converged);
    EXPECT_TRUE(
        retry_report.pressure_representability_distance_gate_applied);
    EXPECT_TRUE(
        retry_report.pressure_representability_distance_gate_passed);
    EXPECT_TRUE(retry_report.constant_pressure_kkt_distance_gate_applied);
    EXPECT_TRUE(retry_report.constant_pressure_kkt_distance_gate_passed);
    EXPECT_TRUE(workspace.static_compatible_pressure_initialized);
    ASSERT_GE(outer_entry_initializer_states.size(), 4u);
    EXPECT_FALSE(outer_entry_initializer_states[0]);
    EXPECT_TRUE(outer_entry_initializer_states[1]);
    EXPECT_FALSE(outer_entry_initializer_states[2]);
    EXPECT_TRUE(outer_entry_initializer_states[3]);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerRemainsOneShotAcrossWorkspaceAndJacobianReallocation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto pressure_mpc_state =
        std::make_shared<CrossCellPressureMpcState>();
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/true,
        pressure_mpc_state,
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/1.0,
        /*pressure_pair_diagonal=*/std::nullopt,
        /*track_immutable_mesh_revisions=*/true,
        /*install_symbolic_cut_volume_pair=*/false,
        /*production_pressure_target=*/2.0,
        /*entry_pressure_baseline=*/3.0);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-13;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initialize_static_compatible_free_surface_pressure = true;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        1.0e-10;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto initial_report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    ASSERT_TRUE(initial_report.converged);
    ASSERT_TRUE(
        initial_report.static_compatible_pressure_initializer_applied);
    ASSERT_TRUE(workspace.static_compatible_pressure_initialized);

    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    EXPECT_TRUE(workspace.static_compatible_pressure_initialized);
    const auto workspace_reallocated_report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    ASSERT_TRUE(workspace_reallocated_report.converged);
    EXPECT_FALSE(
        workspace_reallocated_report
            .static_compatible_pressure_initializer_applied);
    EXPECT_EQ(
        workspace_reallocated_report
            .static_compatible_pressure_initializer_reason,
        "already_initialized");

    const auto entry_sparsity_revision =
        problem.sys->sparsityPatternRevision();
    pressure_mpc_state->enabled = true;
    problem.sys->rebuildConstraintState();
    ASSERT_GT(
        problem.sys->sparsityPatternRevision(),
        entry_sparsity_revision);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto jacobian_reallocated_report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(jacobian_reallocated_report.converged);
    EXPECT_FALSE(
        jacobian_reallocated_report
            .static_compatible_pressure_initializer_applied);
    EXPECT_EQ(
        jacobian_reallocated_report
            .static_compatible_pressure_initializer_reason,
        "already_initialized");
    EXPECT_TRUE(workspace.static_compatible_pressure_initialized);
    EXPECT_EQ(
        workspace.sparsity_revision,
        problem.sys->sparsityPatternRevision());
    EXPECT_NE(
        (stdout_text + stderr_text)
            .find("diagnostic=jacobian_sparsity_reallocation"),
        std::string::npos);
    RecordProperty(
        "static_compatible_pressure_initializer_workspace_reallocation_reapply_count",
        0);
    RecordProperty(
        "static_compatible_pressure_initializer_jacobian_reallocation_reapply_count",
        0);
}

TEST(NewtonSolver,
     StaticCompatiblePressureInitializerRejectsIncompatibleLoadBeforeMutation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    RecordProperty(
        "static_compatible_pressure_initializer_backend_available", 0);
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    RecordProperty(
        "static_compatible_pressure_initializer_backend_available", 1);
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto pressure_dirichlet =
        std::make_shared<FixedPressureDirichletState>();
    pressure_dirichlet->value = 0.0;
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        pressure_dirichlet);
    const auto entry_current = ts_test::getVectorByDof(problem.history.u());
    const auto entry_previous =
        ts_test::getVectorByDof(problem.history.uPrev());
    const auto entry_previous2 =
        ts_test::getVectorByDof(problem.history.uPrev2());

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    options.initialize_static_compatible_free_surface_pressure = true;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        0.5;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;

    EXPECT_FALSE(report.converged);
    EXPECT_TRUE(report.static_compatible_pressure_initializer_requested);
    EXPECT_FALSE(report.static_compatible_pressure_initializer_applied);
    EXPECT_FALSE(report.static_compatible_pressure_initializer_passed);
    EXPECT_EQ(report.static_compatible_pressure_initializer_reason,
              "relative_distance_exceeds_threshold");
    EXPECT_FALSE(workspace.static_compatible_pressure_initialized);
    EXPECT_NEAR(
        report.pressure_representability_relative_distance, 1.0, 1.0e-13);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.u()), entry_current);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
              entry_previous);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
              entry_previous2);
    EXPECT_NE(
        telemetry.find("reason=relative_distance_exceeds_threshold"),
        std::string::npos);
    EXPECT_NE(telemetry.find("applied=0"), std::string::npos);
    EXPECT_NE(telemetry.find("force_projection_applied=0"),
              std::string::npos);
    RecordProperty(
        "static_compatible_pressure_initializer_out_of_range_reject_count",
        1);
    RecordProperty(
        "static_compatible_pressure_initializer_pre_mutation_preserved", 1);
}

TEST(NewtonSolver,
     FreeSurfacePressureRepresentabilityRejectsNonzeroPressureConstraintInhomogeneity)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto pressure_dirichlet =
        std::make_shared<FixedPressureDirichletState>();
    pressure_dirichlet->value = 1.0;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        pressure_dirichlet);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(report.converged);
    EXPECT_EQ(diagnostic_counts.pressure_representability_pair.total, 0);
    EXPECT_NE(
        (stdout_text + stderr_text).find(
            "pressure_representability_reason=nonzero_pressure_constraint_inhomogeneity"),
        std::string::npos);
    EXPECT_NE(
        (stdout_text + stderr_text).find(
            "pressure_representability_convergence=normal_equation_stationarity"),
        std::string::npos);
    EXPECT_NE(
        (stdout_text + stderr_text).find(
            "pressure_representability_distance_gate_applied=0"),
        std::string::npos);
    EXPECT_NE(
        (stdout_text + stderr_text).find(
            "pressure_representability_claimed=0"),
        std::string::npos);
}

TEST(NewtonSolver,
     FreeSurfacePressureRepresentabilityZeroLoadIsStationaryWithoutIterations)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/false,
        /*pressure_mpc_state=*/{},
        /*pressure_dirichlet_state=*/{},
        /*entry_velocity=*/0.0);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;
    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;
    EXPECT_TRUE(report.converged);
    EXPECT_GT(diagnostic_counts.pressure_representability_pair.total, 0);
    EXPECT_NE(telemetry.find("pressure_representability_iterations=0"),
              std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_converged=1"),
              std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_breakdown=0"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_convergence=normal_equation_stationarity"),
        std::string::npos);
    EXPECT_NE(
        telemetry.find("pressure_representability_distance_gate_applied=0"),
        std::string::npos);
    EXPECT_NE(telemetry.find("pressure_representability_claimed=0"),
              std::string::npos);
    EXPECT_NEAR(
        workspace.pressure_representability_load->norm(), 0.0, 1.0e-14);
    EXPECT_NEAR(
        workspace.pressure_representability_solution->norm(), 0.0, 1.0e-14);
    EXPECT_NEAR(
        workspace.pressure_representability_residual->norm(), 0.0, 1.0e-14);
    EXPECT_NEAR(
        workspace.pressure_representability_normal_residual->norm(),
        0.0,
        1.0e-14);
}

TEST(NewtonSolver,
     FreeSurfacePressureRepresentabilityFailsClosedOnPairVectorLayoutMismatch)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem =
        makePressureRepresentabilityProblem(diagnostic_counts);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    const auto n_dofs = problem.sys->dofHandler().getNumDofs();
    // Replacing the initializer's second-load vector with an equal-backend,
    // wrong-size allocation must be rejected before either pair multiply.
    workspace.pressure_representability_correction_load =
        problem.factory->createVector(n_dofs + 1);
    ASSERT_NE(workspace.pressure_representability_correction_load, nullptr);
    problem.history.repack(*problem.factory);

    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    EXPECT_TRUE(report.converged);

    // The three conservative-balance vectors remain useful, but the mixed
    // matrix is never assembled or multiplied after the layout invariant
    // fails.  Runtime telemetry reports pair_matrix_vector_layout_mismatch.
    for (const auto& counts : diagnostic_counts.operators) {
        EXPECT_GT(counts.total, 0);
        EXPECT_EQ(counts.vector_only, counts.total);
    }
    EXPECT_EQ(diagnostic_counts.pressure_representability_pair.total, 0);
}

TEST(NewtonSolver,
     FreeSurfacePressureRepresentabilityRecreatesPairVectorsAfterReplacementSparsityRefresh)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    FreeSurfaceConservativeBalanceKernelCounts diagnostic_counts;
    auto constraint_state =
        std::make_shared<CrossCellPressureMpcState>();
    ScopedEnvVar enable_balance(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "1");
    auto problem = makePressureRepresentabilityProblem(
        diagnostic_counts,
        /*two_cells=*/true,
        constraint_state);
    problem.factory =
        std::make_unique<ReplacementOnlyMatrixFactory>();
    problem.linear =
        problem.factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(problem.linear, nullptr);
    problem.history.repack(*problem.factory);

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1.0e-14;
    options.rel_tolerance = 0.0;
    options.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(
        *problem.sys, *problem.factory, workspace);
    const auto pair_op = std::string(
        "equations_diagnostic_ns_free_surface_pressure_representability_pair");
    const auto base_pair_nnz =
        problem.sys->sparsity(pair_op).computeStats().nnz;
    const auto entry_revision =
        problem.sys->sparsityPatternRevision();
    const std::array<const void*, 9> old_vectors{
        workspace.pressure_representability_load.get(),
        workspace.pressure_representability_correction_load.get(),
        workspace.pressure_representability_solution.get(),
        workspace.pressure_representability_left_basis.get(),
        workspace.pressure_representability_right_basis.get(),
        workspace.pressure_representability_direction.get(),
        workspace.pressure_representability_work.get(),
        workspace.pressure_representability_residual.get(),
        workspace.pressure_representability_normal_residual.get(),
    };

    constraint_state->enabled = true;
    problem.sys->rebuildConstraintState();
    EXPECT_GT(problem.sys->sparsityPatternRevision(), entry_revision);
    EXPECT_GT(problem.sys->sparsity(pair_op).computeStats().nnz,
              base_pair_nnz);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const auto report = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);
    const auto stderr_text = testing::internal::GetCapturedStderr();
    const auto stdout_text = testing::internal::GetCapturedStdout();
    const auto telemetry = stdout_text + stderr_text;
    EXPECT_TRUE(report.converged);
    EXPECT_EQ(workspace.sparsity_revision,
              problem.sys->sparsityPatternRevision());
    EXPECT_TRUE(
        report.constant_pressure_unit_coefficients_represent_constant);
    EXPECT_TRUE(
        report.constant_pressure_constraints_preserve_constants);
    EXPECT_TRUE(report.constant_pressure_kkt_available);

    const std::array<const void*, 9> refreshed_vectors{
        workspace.pressure_representability_load.get(),
        workspace.pressure_representability_correction_load.get(),
        workspace.pressure_representability_solution.get(),
        workspace.pressure_representability_left_basis.get(),
        workspace.pressure_representability_right_basis.get(),
        workspace.pressure_representability_direction.get(),
        workspace.pressure_representability_work.get(),
        workspace.pressure_representability_residual.get(),
        workspace.pressure_representability_normal_residual.get(),
    };
    for (std::size_t i = 0; i < old_vectors.size(); ++i) {
        EXPECT_NE(refreshed_vectors[i], old_vectors[i]);
    }
    EXPECT_GT(diagnostic_counts.pressure_representability_pair.total, 0);
    EXPECT_EQ(
        diagnostic_counts.pressure_representability_pair.matrix_only,
        diagnostic_counts.pressure_representability_pair.total);
    EXPECT_NEAR(
        workspace.pressure_representability_normal_residual->norm(),
        0.0,
        1.0e-12);
    EXPECT_NE(telemetry.find("pressure_representability_available=1"),
              std::string::npos);
    EXPECT_NE(
        telemetry.find(
            "pressure_representability_convergence=normal_equation_stationarity"),
        std::string::npos);
}

TEST(NewtonSolver, MonolithicAuxiliaryReportsComponentResidualNorms)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-20;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient,
                                      *problem.linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_TRUE(rep.component_residual_convergence);
    EXPECT_TRUE(std::isfinite(rep.field_residual_norm0));
    EXPECT_TRUE(std::isfinite(rep.field_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.auxiliary_residual_norm0));
    EXPECT_TRUE(std::isfinite(rep.auxiliary_residual_norm));
    EXPECT_GT(rep.field_residual_norm0, 0.0);
    EXPECT_GT(rep.auxiliary_residual_norm0, 0.0);

    const double initial_combined =
        std::hypot(rep.field_residual_norm0, rep.auxiliary_residual_norm0);
    const double final_combined =
        std::hypot(rep.field_residual_norm, rep.auxiliary_residual_norm);
    EXPECT_NEAR(rep.residual_norm0,
                initial_combined,
                1e-12 * std::max(1.0, initial_combined));
    EXPECT_NEAR(rep.residual_norm,
                final_combined,
                1e-12 * std::max(1.0, final_combined));
}

TEST(NewtonSolver, MonolithicAuxiliaryConvergenceRequiresEachResidualComponent)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    svmp::FE::timestepping::NewtonOptions probe_opts;
    probe_opts.residual_op = "op";
    probe_opts.jacobian_op = "op";
    probe_opts.max_iterations = 1;
    probe_opts.abs_tolerance = 1e-30;
    probe_opts.rel_tolerance = 0.0;
    probe_opts.step_tolerance = 0.0;
    probe_opts.use_line_search = true;
    probe_opts.line_search_max_iterations = 1;

    auto probe_problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});
    svmp::FE::timestepping::NewtonSolver probe_newton(probe_opts);
    svmp::FE::timestepping::NewtonWorkspace probe_ws;
    probe_newton.allocateWorkspace(*probe_problem.sys, *probe_problem.factory, probe_ws);
    probe_problem.history.repack(*probe_problem.factory);

    const auto probe_rep = probe_newton.solveStep(*probe_problem.transient,
                                                  *probe_problem.linear,
                                                  /*solve_time=*/probe_problem.history.dt(),
                                                  probe_problem.history,
                                                  probe_ws);
    ASSERT_TRUE(probe_rep.component_residual_convergence);
    ASSERT_GT(probe_rep.residual_norm0, 0.0);
    ASSERT_GT(probe_rep.field_residual_norm0, 0.0);

    const double combined_rel = probe_rep.residual_norm / probe_rep.residual_norm0;
    const double field_rel = probe_rep.field_residual_norm / probe_rep.field_residual_norm0;
    if (!(std::isfinite(combined_rel) && std::isfinite(field_rel) &&
          combined_rel > 0.0 && combined_rel < field_rel)) {
        GTEST_SKIP() << "Fixture did not produce a combined-relative gap for this backend/configuration";
    }

    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});
    auto nopt = probe_opts;
    nopt.rel_tolerance = std::sqrt(combined_rel * field_rel);
    nopt.abs_tolerance = 0.0;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient,
                                      *problem.linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_LT(rep.residual_norm / rep.residual_norm0, nopt.rel_tolerance);
    EXPECT_GT(rep.field_residual_norm / rep.field_residual_norm0, nopt.rel_tolerance);
}

TEST(NewtonSolver, StagnationDoesNotOverrideRequestedTolerances)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            return ((u - svmp::FE::forms::FormExpr::constant(1.0)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{0.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 3;
    nopt.abs_tolerance = 1e-12;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.stagnation_tolerance = 0.99;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver damped(*problem.linear, /*scale=*/0.01);
    const auto rep = newton.solveStep(*problem.transient,
                                      damped,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, nopt.max_iterations);
    EXPECT_TRUE(std::isfinite(rep.residual_norm));
    EXPECT_GT(rep.residual_norm, 0.9 * rep.residual_norm0);
    EXPECT_GT(rep.residual_norm, nopt.abs_tolerance);
}

TEST(NewtonSolver, RelativeToleranceCanConvergeAfterFirstUpdate)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 5;
    nopt.abs_tolerance = 1e-30;
    nopt.rel_tolerance = 1e-2;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient,
                                      *problem.linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_TRUE(rep.converged);
    EXPECT_GE(rep.iterations, 1);
    EXPECT_LT(rep.residual_norm / rep.residual_norm0, nopt.rel_tolerance);
    EXPECT_GT(rep.residual_norm, nopt.abs_tolerance);
}

TEST(NewtonSolver, ValidatesFieldResidualCriteria)
{
    using Criterion =
        svmp::FE::timestepping::NewtonOptions::FieldResidualCriterion;

    svmp::FE::timestepping::NewtonOptions options;
    options.field_residual_criteria = {
        Criterion{.field = 0, .abs_tolerance = 0.0, .rel_tolerance = 0.0}};
    EXPECT_THROW(
        (void)svmp::FE::timestepping::NewtonSolver(options),
        svmp::FE::InvalidArgumentException);

    options.field_residual_criteria = {
        Criterion{.field = 0, .abs_tolerance = -1.0, .rel_tolerance = 0.0}};
    EXPECT_THROW(
        (void)svmp::FE::timestepping::NewtonSolver(options),
        svmp::FE::InvalidArgumentException);

    options.field_residual_criteria = {
        Criterion{.field = 0, .abs_tolerance = 1e-8, .rel_tolerance = 0.0},
        Criterion{.field = 0, .abs_tolerance = 0.0, .rel_tolerance = 1e-4}};
    EXPECT_THROW(
        (void)svmp::FE::timestepping::NewtonSolver(options),
        svmp::FE::InvalidArgumentException);
}

TEST(NewtonSolver,
     FieldResidualCriterionPreventsDominantFieldMaskingAndStepBypass)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto run = [](bool require_selected_field) {
        auto problem = makeTwoFieldAffineProblem(
            /*selected_target=*/1.0,
            /*dominant_target=*/1000.0,
            /*dt=*/0.1);

        svmp::FE::timestepping::NewtonOptions options;
        options.residual_op = "op";
        options.jacobian_op = "op";
        options.max_iterations = 6;
        options.abs_tolerance = 0.0;
        options.rel_tolerance = 1e-2;
        // Exercise the historical step-only exit as well: with a configured
        // field gate, a small step may not bypass the selected residual.
        options.step_tolerance = 2000.0;
        options.use_line_search = false;
        if (require_selected_field) {
            options.field_residual_criteria.push_back({
                .field = problem.selected_field,
                .abs_tolerance = 0.0,
                .rel_tolerance = 1e-1});
        }

        svmp::FE::timestepping::NewtonSolver newton(options);
        svmp::FE::timestepping::NewtonWorkspace workspace;
        newton.allocateWorkspace(
            *problem.sys, *problem.factory, workspace);
        problem.history.repack(*problem.factory);

        SelectedDofScalingLinearSolver selectively_damped(
            *problem.linear,
            problem.sys->fieldDofOffset(problem.selected_field),
            /*scale=*/0.5);
        return newton.solveStep(
            *problem.transient,
            selectively_damped,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            workspace);
    };

    const auto monolithic_only = run(/*require_selected_field=*/false);
    ASSERT_TRUE(monolithic_only.converged);
    EXPECT_EQ(monolithic_only.iterations, 1);

    const auto field_gated = run(/*require_selected_field=*/true);
    EXPECT_TRUE(field_gated.converged);
    // The dominant field is exact after the first update, while the selected
    // residual contracts by 1/2 per update and first meets 0.1 at 1/16.
    EXPECT_EQ(field_gated.iterations, 4);
}

TEST(NewtonSolver,
     FieldRelativeReferenceActivatesAfterAcceptedStateRefreshAndRequiresContraction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    TwoFieldProblem problem;
    problem.mesh =
        std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    problem.space = std::make_shared<svmp::FE::spaces::L2Space>(
        svmp::FE::ElementType::Tetra4, /*order=*/0);
    problem.sys =
        std::make_unique<svmp::FE::systems::FESystem>(problem.mesh);
    problem.selected_field = problem.sys->addField(
        svmp::FE::systems::FieldSpec{
            .name = "selected", .space = problem.space, .components = 1});
    problem.dominant_field = problem.sys->addField(
        svmp::FE::systems::FieldSpec{
            .name = "dominant", .space = problem.space, .components = 1});
    problem.sys->addOperator("op");
    problem.sys->addCellKernel(
        "op",
        problem.selected_field,
        problem.selected_field,
        std::make_shared<RefreshedGeometryRootKernel>(
            &refreshed_measure, static_cast<svmp::FE::Real>(1.0)));
    problem.sys->addCellKernel(
        "op",
        problem.dominant_field,
        problem.dominant_field,
        std::make_shared<AffineScalarCellKernel>(
            static_cast<svmp::FE::Real>(100.0)));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    problem.sys->setup({}, inputs);
    problem.integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    problem.transient =
        std::make_unique<svmp::FE::systems::TransientSystem>(
            *problem.sys, problem.integrator);
    problem.factory = ts_test::createTestFactory();
    ASSERT_NE(problem.factory, nullptr);
    problem.linear = problem.factory->createLinearSolver(ts_test::directSolve());
    ASSERT_NE(problem.linear, nullptr);

    const auto selected_dof =
        problem.sys->fieldDofOffset(problem.selected_field);
    const auto dominant_dof =
        problem.sys->fieldDofOffset(problem.dominant_field);
    const auto n_dofs = problem.sys->dofHandler().getNumDofs();
    problem.history = svmp::FE::timestepping::TimeHistory::allocate(
        *problem.factory, n_dofs);
    problem.history.setDt(0.1);
    problem.history.setPrevDt(0.1);
    std::vector<svmp::FE::Real> initial_values(
        static_cast<std::size_t>(n_dofs), svmp::FE::Real{0.0});
    initial_values[static_cast<std::size_t>(selected_dof)] =
        svmp::FE::Real{1.0};
    ts_test::setVectorByDof(problem.history.uPrev(), initial_values);
    ts_test::setVectorByDof(problem.history.uPrev2(), initial_values);
    problem.history.resetCurrentToPrevious();

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct AcceptedStateRecord {
        double selected{0.0};
        double dominant{0.0};
        double measure{0.0};
    };
    std::vector<AcceptedStateRecord> accepted_states;

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 4;
    options.abs_tolerance = 0.0;
    options.rel_tolerance = 2e-2;
    options.step_tolerance = 0.0;
    options.use_line_search = false;
    options.assemble_both_when_possible = true;
    options.field_residual_criteria.push_back({
        .field = problem.selected_field,
        .abs_tolerance = 0.0,
        // A value of one makes the activation sample satisfy the numerical
        // ratio unless the solver explicitly requires a later contraction.
        .rel_tolerance = 1.0});
    options.synchronize_state =
        [&refreshed_measure,
         &accepted_states,
         selected_dof,
         dominant_dof](const svmp::FE::systems::SystemStateView& state,
                       SyncPoint point) {
            if (point != SyncPoint::AcceptedNonlinearState) {
                return;
            }
            ASSERT_GT(state.u.size(), static_cast<std::size_t>(dominant_dof));
            const auto selected =
                static_cast<double>(state.u[static_cast<std::size_t>(selected_dof)]);
            const auto dominant =
                static_cast<double>(state.u[static_cast<std::size_t>(dominant_dof)]);
            refreshed_measure = dominant > 50.0 ? 2.0 : 1.0;
            accepted_states.push_back(
                AcceptedStateRecord{selected, dominant, refreshed_measure});
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    SelectedDofScalingLinearSolver selectively_damped(
        *problem.linear, selected_dof, /*scale=*/0.5);
    const auto report = newton.solveStep(
        *problem.transient,
        selectively_damped,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    ASSERT_TRUE(report.converged);
    // The initial selected residual is exactly 1*1-1 == 0.  Updating the
    // dominant field refreshes the accepted geometry to measure two, creating
    // the first positive selected residual 2*1-1 == 1.  Although rel_tol=1,
    // that activation sample must not terminate Newton.  One damped selected
    // update then contracts it to 2*0.75-1 == 0.5 before convergence.
    EXPECT_EQ(report.iterations, 2);
    ASSERT_GE(accepted_states.size(), 3u);
    EXPECT_DOUBLE_EQ(accepted_states[0].selected, 1.0);
    EXPECT_DOUBLE_EQ(accepted_states[0].dominant, 0.0);
    EXPECT_DOUBLE_EQ(accepted_states[0].measure, 1.0);
    EXPECT_DOUBLE_EQ(accepted_states[1].selected, 1.0);
    EXPECT_NEAR(accepted_states[1].dominant, 100.0, 1e-12);
    EXPECT_DOUBLE_EQ(accepted_states[1].measure, 2.0);
    EXPECT_NEAR(accepted_states[2].selected, 0.75, 1e-12);
    EXPECT_NEAR(accepted_states[2].dominant, 100.0, 1e-12);
    EXPECT_DOUBLE_EQ(accepted_states[2].measure, 2.0);

    const auto final_values = ts_test::getVectorByDof(problem.history.u());
    EXPECT_NEAR(final_values[static_cast<std::size_t>(selected_dof)],
                0.75,
                1e-12);
    EXPECT_NEAR(final_values[static_cast<std::size_t>(dominant_dof)],
                100.0,
                1e-12);
}

TEST(NewtonSolver,
     FieldAbsoluteFloorAcceptsContractedActivatedResidualBeforeRelativeTarget)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    auto problem = makeTwoFieldRefreshedGeometryProblem(
        &refreshed_measure,
        /*selected_initial=*/1.0,
        /*selected_target=*/1.0,
        /*dominant_target=*/100.0,
        /*dt=*/0.1);
    const auto selected_dof =
        problem.sys->fieldDofOffset(problem.selected_field);
    const auto dominant_dof =
        problem.sys->fieldDofOffset(problem.dominant_field);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 4;
    options.abs_tolerance = 0.0;
    options.rel_tolerance = 2e-2;
    options.step_tolerance = 0.0;
    options.use_line_search = false;
    options.assemble_both_when_possible = true;
    options.field_residual_criteria.push_back({
        .field = problem.selected_field,
        .abs_tolerance = 1e-1,
        .rel_tolerance = 1e-6});
    options.synchronize_state =
        [&refreshed_measure,
         dominant_dof](const svmp::FE::systems::SystemStateView& state,
                       SyncPoint point) {
            if (point != SyncPoint::AcceptedNonlinearState) {
                return;
            }
            ASSERT_GT(state.u.size(), static_cast<std::size_t>(dominant_dof));
            const auto dominant = static_cast<double>(
                state.u[static_cast<std::size_t>(dominant_dof)]);
            refreshed_measure = dominant > 50.0 ? 2.0 : 1.0;
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    SelectedDofScalingLinearSolver selectively_damped(
        *problem.linear, selected_dof, /*scale=*/0.5);
    const auto report = newton.solveStep(
        *problem.transient,
        selectively_damped,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    ASSERT_TRUE(report.converged);
    // On the unit tetra the activated selected residual is 1/6, above the
    // 0.1 absolute floor.  The activation sample therefore cannot terminate
    // Newton.  One damped selected update contracts the residual to 1/12,
    // below the absolute floor but still at relative ratio 0.5 >> 1e-6.
    EXPECT_EQ(report.iterations, 2);
    const auto final_values = ts_test::getVectorByDof(problem.history.u());
    EXPECT_NEAR(final_values[static_cast<std::size_t>(selected_dof)],
                0.75,
                1e-12);
    EXPECT_NEAR(final_values[static_cast<std::size_t>(dominant_dof)],
                100.0,
                1e-12);
}

TEST(NewtonSolver,
     FieldAbsoluteFloorMayAcceptTheRelativeReferenceActivationSample)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    double refreshed_measure = 1.0;
    auto problem = makeTwoFieldRefreshedGeometryProblem(
        &refreshed_measure,
        /*selected_initial=*/1.0,
        /*selected_target=*/1.0,
        /*dominant_target=*/100.0,
        /*dt=*/0.1);
    const auto selected_dof =
        problem.sys->fieldDofOffset(problem.selected_field);
    const auto dominant_dof =
        problem.sys->fieldDofOffset(problem.dominant_field);

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 3;
    options.abs_tolerance = 0.0;
    options.rel_tolerance = 2e-2;
    options.step_tolerance = 0.0;
    options.use_line_search = false;
    options.assemble_both_when_possible = true;
    options.field_residual_criteria.push_back({
        .field = problem.selected_field,
        .abs_tolerance = 2e-1,
        .rel_tolerance = 1e-6});
    options.synchronize_state =
        [&refreshed_measure,
         dominant_dof](const svmp::FE::systems::SystemStateView& state,
                       SyncPoint point) {
            if (point != SyncPoint::AcceptedNonlinearState) {
                return;
            }
            ASSERT_GT(state.u.size(), static_cast<std::size_t>(dominant_dof));
            const auto dominant = static_cast<double>(
                state.u[static_cast<std::size_t>(dominant_dof)]);
            refreshed_measure = dominant > 50.0 ? 2.0 : 1.0;
        };

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    SelectedDofScalingLinearSolver selectively_damped(
        *problem.linear, selected_dof, /*scale=*/0.5);
    const auto report = newton.solveStep(
        *problem.transient,
        selectively_damped,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    ASSERT_TRUE(report.converged);
    // The first positive selected residual is 1/6.  It defines the relative
    // reference, so its relative ratio is one and cannot meet 1e-6, but it is
    // already below the independent 0.2 absolute floor.  OR semantics permit
    // convergence without applying a selected-field correction.
    EXPECT_EQ(report.iterations, 1);
    const auto final_values = ts_test::getVectorByDof(problem.history.u());
    EXPECT_NEAR(final_values[static_cast<std::size_t>(selected_dof)],
                1.0,
                1e-12);
    EXPECT_NEAR(final_values[static_cast<std::size_t>(dominant_dof)],
                100.0,
                1e-12);
}

TEST(NewtonSolver, ThrowsWhenLinearSolveFails)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    AlwaysFailLinearSolver failing(*problem.linear);
    EXPECT_THROW((void)newton.solveStep(*problem.transient,
                                        failing,
                                        /*solve_time=*/problem.history.dt(),
                                        problem.history,
                                        ws),
                 svmp::FE::FEException);
}

TEST(NewtonSolver,
     RejectsFiniteInexactResultsWithoutUsableProgressAndPreservesHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using Script = ScriptedFiniteInexactLinearSolver::Result;
    const std::array<std::pair<const char*, Script>, 4> rejected_results{{
        {"zero iterations despite reported reduction",
         Script{0, 1.0, 0.5, 0.5, false}},
        {"nonzero correction without residual reduction",
         Script{1, 1.0, 1.0, 1.0, false}},
        {"reported reduction with relative residual at least one",
         Script{1, 1.0, 0.5, 1.0, false}},
        {"zero correction despite reported reduction",
         Script{1, 1.0, 0.5, 0.5, true}},
    }};

    for (const auto& [description, scripted_result] : rejected_results) {
        SCOPED_TRACE(description);
        auto problem = makeScalarProblem(
            [](const svmp::FE::forms::FormExpr& u,
               const svmp::FE::forms::FormExpr& v) {
                return (u * v).dx();
            },
            /*dt=*/0.1,
            /*u0=*/{1.0});

        svmp::FE::timestepping::NewtonOptions options;
        options.residual_op = "op";
        options.jacobian_op = "op";
        options.max_iterations = 1;
        options.abs_tolerance = 1e-16;
        options.rel_tolerance = 1e-16;
        options.step_tolerance = 0.0;
        options.use_line_search = false;
        options.accept_inexact_linear_solutions = true;

        svmp::FE::timestepping::NewtonSolver newton(options);
        svmp::FE::timestepping::NewtonWorkspace workspace;
        newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
        problem.history.repack(*problem.factory);

        const auto entry_u = ts_test::getVectorByDof(problem.history.u());
        const auto entry_u_prev =
            ts_test::getVectorByDof(problem.history.uPrev());
        const auto entry_u_prev2 =
            ts_test::getVectorByDof(problem.history.uPrev2());

        ScriptedFiniteInexactLinearSolver inexact(
            *problem.linear, scripted_result);
        bool threw = false;
        try {
            (void)newton.solveStep(
                *problem.transient,
                inexact,
                /*solve_time=*/problem.history.dt(),
                problem.history,
                workspace);
        } catch (const svmp::FE::FEException& error) {
            threw = true;
            EXPECT_NE(std::string(error.what()).find(
                          "linear solve did not converge"),
                      std::string::npos)
                << error.what();
        }

        EXPECT_TRUE(threw);
        EXPECT_EQ(inexact.solve_calls, 1);
        EXPECT_EQ(ts_test::getVectorByDof(problem.history.u()), entry_u);
        EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
                  entry_u_prev);
        EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
                  entry_u_prev2);
    }
}

TEST(NewtonSolver,
     AcceptsFiniteInexactPositiveProgressWithNonzeroCorrectionAndPreservesHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u,
           const svmp::FE::forms::FormExpr& v) {
            return (u * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1e-16;
    options.rel_tolerance = 1e-16;
    options.step_tolerance = 0.0;
    options.use_line_search = false;
    options.accept_inexact_linear_solutions = true;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    const auto entry_u = ts_test::getVectorByDof(problem.history.u());
    const auto entry_u_prev =
        ts_test::getVectorByDof(problem.history.uPrev());
    const auto entry_u_prev2 =
        ts_test::getVectorByDof(problem.history.uPrev2());

    ScriptedFiniteInexactLinearSolver inexact(
        *problem.linear,
        ScriptedFiniteInexactLinearSolver::Result{
            1, 1.0, 0.5, 0.5, false});
    const auto report = newton.solveStep(
        *problem.transient,
        inexact,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        workspace);

    EXPECT_EQ(inexact.solve_calls, 1);
    EXPECT_FALSE(report.linear.converged);
    const auto accepted_u = ts_test::getVectorByDof(problem.history.u());
    ASSERT_EQ(accepted_u.size(), entry_u.size());
    ASSERT_EQ(accepted_u.size(), 1u);
    EXPECT_NE(accepted_u, entry_u);
    EXPECT_NEAR(accepted_u.front(), 0.0, 1e-13);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()),
              entry_u_prev);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()),
              entry_u_prev2);
}

TEST(NewtonSolver,
     RejectsNonfiniteLinearReportEvenWhenInexactSolutionsAreAllowed)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u,
           const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.accept_inexact_linear_solutions = true;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    InvalidNumericalLinearSolver invalid(
        *problem.linear,
        InvalidNumericalLinearSolver::Mode::NonfiniteReport);
    try {
        (void)newton.solveStep(*problem.transient,
                               invalid,
                               /*solve_time=*/problem.history.dt(),
                               problem.history,
                               ws);
        FAIL() << "Newton accepted a nonfinite linear report";
    } catch (const svmp::FE::FEException& error) {
        EXPECT_NE(std::string(error.what()).find("numerical breakdown"),
                  std::string::npos)
            << error.what();
    }
    EXPECT_EQ(invalid.solve_calls, 1);
    for (const auto value : problem.history.u().localSpan()) {
        EXPECT_TRUE(std::isfinite(static_cast<double>(value)));
    }
}

TEST(NewtonSolver, RejectsNonfiniteCorrectionDespiteConvergedLinearReport)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u,
           const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.accept_inexact_linear_solutions = true;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    InvalidNumericalLinearSolver invalid(
        *problem.linear,
        InvalidNumericalLinearSolver::Mode::NonfiniteCorrection);
    try {
        (void)newton.solveStep(*problem.transient,
                               invalid,
                               /*solve_time=*/problem.history.dt(),
                               problem.history,
                               ws);
        FAIL() << "Newton accepted a nonfinite correction";
    } catch (const svmp::FE::FEException& error) {
        EXPECT_NE(std::string(error.what()).find("numerical breakdown"),
                  std::string::npos)
            << error.what();
    }
    EXPECT_EQ(invalid.solve_calls, 1);
    for (const auto value : problem.history.u().localSpan()) {
        EXPECT_TRUE(std::isfinite(static_cast<double>(value)));
    }
}

TEST(NewtonSolver,
     RejectsNonfiniteExplicitBorderedRecoveryBeforeMutatingState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(
        /*dt=*/0.1,
        /*u0=*/{0.2, -0.4, 0.1, 0.7});

    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1e-16;
    options.rel_tolerance = 1e-16;
    options.step_tolerance = 0.0;
    options.use_line_search = false;
    options.accept_inexact_linear_solutions = true;

    svmp::FE::timestepping::NewtonSolver newton(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*problem.sys, *problem.factory, workspace);
    problem.history.repack(*problem.factory);

    // Define the entry snapshot after applying the system's automatic gauge
    // relation.  That canonical projection is part of solve initialization,
    // not a consequence of accepting the Newton correction under test.
    problem.sys->updateConstraints(problem.history.dt(), problem.history.dt());
    auto& constraints = problem.sys->constraints();
    constraints.updateGhostsAndDistribute(problem.history.u());
    constraints.distributeMasterBearing(problem.history.uPrev());
    constraints.distributeMasterBearing(problem.history.uPrev2());

    const auto entry_u = ts_test::getVectorByDof(problem.history.u());
    const auto entry_u_prev = ts_test::getVectorByDof(problem.history.uPrev());
    const auto entry_u_prev2 = ts_test::getVectorByDof(problem.history.uPrev2());
    const auto entry_auxiliary = problem.sys->checkpointAuxiliaryState();

    OverflowingBorderedRecoveryLinearSolver overflowing(
        *problem.linear, *problem.sys);
    try {
        (void)newton.solveStep(
            *problem.transient,
            overflowing,
            /*solve_time=*/problem.history.dt(),
            problem.history,
            workspace);
        FAIL() << "Newton accepted a nonfinite explicit bordered recovery";
    } catch (const svmp::FE::FEException& error) {
        EXPECT_NE(std::string(error.what()).find("post-recovery correction"),
                  std::string::npos)
            << error.what();
    }

    EXPECT_GT(overflowing.solve_calls, 1);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.u()), entry_u);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev()), entry_u_prev);
    EXPECT_EQ(ts_test::getVectorByDof(problem.history.uPrev2()), entry_u_prev2);
    EXPECT_EQ(problem.sys->checkpointAuxiliaryState(), entry_auxiliary);
    for (const auto value : workspace.delta->localSpan()) {
        EXPECT_EQ(value, static_cast<svmp::FE::Real>(0.0));
    }
}

TEST(NewtonSolver, PassesEffectiveStageTimeStepToLinearSolver)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto make_problem = [] {
        return makeScalarProblem(
            [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
                return (u * v).dx();
            },
            /*dt=*/0.2,
            /*u0=*/{1.0});
    };

    auto stage_problem = make_problem();
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*stage_problem.sys, *stage_problem.factory, ws);
    stage_problem.history.repack(*stage_problem.factory);

    RecordingEffectiveTimeStepSolver stage_solver(*stage_problem.linear);
    (void)newton.solveStep(*stage_problem.transient,
                           stage_solver,
                           /*solve_time=*/stage_problem.history.time() + 0.05,
                           stage_problem.history,
                           ws);
    ASSERT_FALSE(stage_solver.effective_time_steps.empty());
    EXPECT_NEAR(stage_solver.effective_time_steps.front(), 0.05, 1e-15);

    auto fallback_problem = make_problem();
    svmp::FE::timestepping::NewtonWorkspace ws_fallback;
    newton.allocateWorkspace(*fallback_problem.sys, *fallback_problem.factory, ws_fallback);
    fallback_problem.history.repack(*fallback_problem.factory);

    RecordingEffectiveTimeStepSolver fallback_solver(*fallback_problem.linear);
    (void)newton.solveStep(*fallback_problem.transient,
                           fallback_solver,
                           /*solve_time=*/fallback_problem.history.time(),
                           fallback_problem.history,
                           ws_fallback);
    ASSERT_FALSE(fallback_solver.effective_time_steps.empty());
    EXPECT_NEAR(fallback_solver.effective_time_steps.front(), fallback_problem.history.dt(), 1e-15);
}

TEST(NewtonSolver, PtcRetryAppliesMassLumpedDiagonalShiftAndRestoresJacobian)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;
    constexpr double gamma = 3.5;
    auto problem = makeScalarProblem(
        [&](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            return (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();
        },
        dt,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.pseudo_transient.enabled = true;
    nopt.pseudo_transient.activate_on_linear_failure = true;
    nopt.pseudo_transient.gamma_initial = gamma;
    nopt.pseudo_transient.gamma_growth = 2.0;
    nopt.pseudo_transient.gamma_max = 10.0;
    nopt.pseudo_transient.max_linear_retries = 2;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    FailOnceThenSolveRecordingMatrixSolver linear(*problem.linear);
    EXPECT_NO_THROW((void)newton.solveStep(*problem.transient,
                                           linear,
                                           /*solve_time=*/problem.history.dt(),
                                           problem.history,
                                           ws));

    ASSERT_EQ(linear.solve_calls, 2);
    ASSERT_EQ(linear.observed_diagonals.size(), 2u);
    ASSERT_NE(ws.ptc_mass_lumped, nullptr);
    const auto mass = ws.ptc_mass_lumped->localSpan();
    ASSERT_EQ(mass.size(), 1u);
    const double expected_shift = gamma * std::abs(static_cast<double>(mass[0]));
    EXPECT_GT(expected_shift, 0.0);
    EXPECT_NEAR(linear.observed_diagonals[1] - linear.observed_diagonals[0],
                expected_shift,
                1e-12);
}

// The Jacobian-check path still validates the coupled reduced-update
// contribution, but it now does so through the assembled/system-side operator
// path instead of leaving a non-empty reduced-update set installed on the
// linear solver wrapper after the check. Re-enable with a lower-level probe if
// we need explicit coverage of that operator path.
TEST(NewtonSolver, DISABLED_PreservesCoupledUpdatesAcrossJacobianCheckResidualAssemblies)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    ScopedEnvVar jac_check("SVMP_FE_JACOBIAN_CHECK", "1");
    ScopedEnvVar jac_it("SVMP_FE_JACOBIAN_CHECK_IT", "0");
    ScopedEnvVar jac_step("SVMP_FE_JACOBIAN_CHECK_STEP", "1e-7");

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    RecordingRankOneSolver linear(*problem.linear);
    const auto rep = newton.solveStep(*problem.transient,
                                      linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_TRUE(linear.last_updates.empty());
    EXPECT_TRUE(linear.saw_nonempty_reduced_updates);
    EXPECT_FALSE(linear.saw_nonempty_rank_one_updates);
    EXPECT_TRUE(rep.linear.converged);
}

TEST(NewtonSolver, ExplicitRankOneUsesCoupledSolveOptions)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    ScopedEnvVar force_explicit("SVMP_FORCE_EXPLICIT_RANK_ONE", "1");

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);
    ts_test::setVectorByDof(problem.history.u(), {0.9, -0.1, 0.35, -0.6});

    RecordingSolveOptionsSolver linear(*problem.linear, /*native_rank_one_support=*/true);
    const auto base_options = linear.getOptions();
    (void)newton.solveStep(*problem.transient,
                           linear,
                           /*solve_time=*/problem.history.dt(),
                           problem.history,
                           ws);

    ASSERT_TRUE(linear.saw_solve);
    ASSERT_TRUE(linear.options_seen_in_solve.has_value());
    EXPECT_TRUE(linear.last_updates.empty());
    EXPECT_EQ(linear.options_seen_in_solve->fsils_residual_check_policy,
              svmp::FE::backends::FsilsResidualCheckPolicy::Always);
    EXPECT_EQ(linear.options_seen_in_solve->max_iter, base_options.max_iter);
    EXPECT_EQ(linear.options_seen_in_solve->rel_tol, base_options.rel_tol);
    EXPECT_EQ(linear.options_seen_in_solve->abs_tol, base_options.abs_tol);
}

TEST(NewtonSolver, ExportsMixedAuxiliaryLayoutIntoLinearSolverOptions)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(
        /*dt=*/0.1,
        /*u0=*/{0.2, -0.4, 0.1, 0.7},
        svmp::FE::systems::AuxiliaryBlockRole::Constraint);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    RecordingSolveOptionsSolver linear(*problem.linear);
    (void)newton.solveStep(*problem.transient,
                           linear,
                           /*solve_time=*/problem.history.dt(),
                           problem.history,
                           ws);

    ASSERT_TRUE(linear.options_seen_in_solve.has_value());
    ASSERT_TRUE(linear.options_seen_in_solve->mixed_block_layout.has_value());

    const auto& mixed = *linear.options_seen_in_solve->mixed_block_layout;
    const auto* aux = mixed.findBlock("newton_rank_one_snapshot_inst");
    ASSERT_NE(aux, nullptr);
    EXPECT_EQ(aux->kind, svmp::FE::backends::MixedBlockKind::Auxiliary);
    EXPECT_EQ(aux->role, svmp::FE::backends::BlockRole::ConstraintField);
    EXPECT_EQ(aux->offset, 4);
    EXPECT_EQ(aux->size, 2);
    EXPECT_EQ(linear.options_seen_in_solve->resolveBlockNameForRole(
                  svmp::FE::backends::BlockRole::ConstraintField),
              "newton_rank_one_snapshot_inst");
}

TEST(NewtonSolver, CoupledSolveAcceptsOriginalLinearTarget)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    auto base_opts = problem.linear->getOptions();
    base_opts.rel_tol = static_cast<svmp::FE::Real>(1e-3);
    base_opts.abs_tol = static_cast<svmp::FE::Real>(0.0);
    base_opts.max_iter = 25;
    problem.linear->setOptions(base_opts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);
    ts_test::setVectorByDof(problem.history.u(), {0.9, -0.1, 0.35, -0.6});

    ForceResidualReportLinearSolver linear(*problem.linear,
                                           /*initial_residual_norm=*/1.0,
                                           /*final_residual_norm=*/5e-4,
                                           /*forced_miss_calls=*/1,
                                           /*native_rank_one_support=*/true);
    const auto rep = newton.solveStep(*problem.transient,
                                      linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_TRUE(rep.linear.converged);
    EXPECT_NE(rep.linear.message.find("accepted original coupled target"), std::string::npos);
}
