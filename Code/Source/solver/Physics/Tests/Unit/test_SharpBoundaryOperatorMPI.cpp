/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Auxiliary/AuxiliaryBindings.h"
#include "FE/Auxiliary/AuxiliaryInputRegistry.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/BoundaryReductionService.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TimeIntegrator.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

namespace svmp::Physics::test {
namespace {

namespace ns = formulations::navier_stokes;

constexpr std::array<FE::Real, 8> kWetFractions{{
    FE::Real{1.0e-8},
    FE::Real{1.0e-6},
    FE::Real{1.0e-4},
    FE::Real{1.0e-2},
    FE::Real{0.1},
    FE::Real{0.25},
    FE::Real{0.49},
    FE::Real{1.0},
}};

constexpr std::array<FE::geometry::CutIntegrationSide, 2> kActiveSides{{
    FE::geometry::CutIntegrationSide::Negative,
    FE::geometry::CutIntegrationSide::Positive,
}};

constexpr FE::Real kParentBoundaryMeasure{0.5};
constexpr FE::Real kPartitionTolerance{1.0e-11};

class ScopedEnvVar final {
public:
    ScopedEnvVar(const char* key, std::optional<std::string> value)
        : key_(key)
    {
        if (const char* prior = std::getenv(key_.c_str())) {
            prior_ = std::string(prior);
        }
        set(std::move(value));
    }

    ~ScopedEnvVar() { set(prior_); }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    void set(const std::optional<std::string>& value) const
    {
        if (value.has_value()) {
            ::setenv(key_.c_str(), value->c_str(), 1);
        } else {
            ::unsetenv(key_.c_str());
        }
    }

    std::string key_;
    std::optional<std::string> prior_{};
};

class PartitionedSingleTetraBoundaryMesh final
    : public FE::assembly::IMeshAccess {
public:
    PartitionedSingleTetraBoundaryMesh(int marker,
                                       int rank,
                                       int size,
                                       int owner_rank,
                                       FE::Real length_scale = FE::Real{1.0})
        : marker_(marker)
        , rank_(rank)
        , size_(size)
        , owner_rank_(owner_rank)
    {
        nodes_ = {{
            {{0.0, 0.0, 0.0}},
            {{length_scale, 0.0, 0.0}},
            {{0.0, length_scale, 0.0}},
            {{0.0, 0.0, length_scale}},
        }};
        cell_ = {{0, 1, 2, 3}};
    }

    void enforcePartitionOwnership() { enforce_partition_ = true; }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return ownsCell() ? 1 : 0;
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numOwnedVertices() const override
    {
        return ownsCell() ? 4 : 0;
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 1u; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return enforce_partition_ ? 2u : 1u;
    }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override { return true; }
    [[nodiscard]] FE::GlobalIndex getCellGlobalId(
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 1701;
    }
    [[nodiscard]] FE::GlobalIndex getBoundaryFaceGlobalId(
        FE::GlobalIndex /*face_id*/) const override
    {
        return 2701;
    }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }
    [[nodiscard]] int getCellOwnerRank(
        FE::GlobalIndex /*cell_id*/) const override
    {
        return owner_rank_;
    }
    [[nodiscard]] int getBoundaryFaceOwnerRank(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*parent_cell*/) const override
    {
        return owner_rank_;
    }
    [[nodiscard]] bool isOwnedCell(
        FE::GlobalIndex /*cell_id*/) const override
    {
        return ownsCell();
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra4;
    }
    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }
    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        coordinates.assign(nodes_.begin(), nodes_.end());
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex /*face_id*/) const override
    {
        return marker_;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }
    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        if (ownsCell()) {
            callback(0);
        }
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback)
        const override
    {
        if (marker < 0 || marker == marker_) {
            callback(0, 0);
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex,
                           FE::GlobalIndex,
                           FE::GlobalIndex)> /*callback*/) const override
    {
    }

private:
    [[nodiscard]] bool ownsCell() const
    {
        return !enforce_partition_ || rank_ == owner_rank_;
    }

    int marker_{-1};
    int rank_{0};
    int size_{1};
    int owner_rank_{0};
    bool enforce_partition_{false};
    std::array<std::array<FE::Real, 3>, 4> nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

FE::systems::SetupInputs makePartitionedSingleTetraSetupInputs(int owner_rank)
{
    auto topology = makeSingleTetraTopology();
    topology.cell_owner_ranks = {owner_rank};
    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

constexpr int kChannelInletMarker{241};
constexpr int kChannelOutletMarker{242};
constexpr int kChannelSideWallMarker{243};
constexpr int kChannelOtherWallMarker{244};

enum class ChannelPartition {
    Serial,
    XSlab,
    RoundRobin,
};

class StructuredChannelBoundaryMesh final
    : public FE::assembly::IMeshAccess {
public:
    StructuredChannelBoundaryMesh(int cells_per_unit_length,
                                  int rank,
                                  int size,
                                  ChannelPartition partition)
        : n_(cells_per_unit_length)
        , nx_(2 * cells_per_unit_length)
        , ny_(cells_per_unit_length)
        , nz_(cells_per_unit_length)
        , rank_(rank)
        , size_(size)
        , partition_(partition)
    {
        if (n_ < 1 || rank_ < 0 || size_ < 1 || rank_ >= size_) {
            throw std::invalid_argument(
                "structured sharp-boundary channel dimensions are invalid");
        }

        nodes_.resize(static_cast<std::size_t>(
            (nx_ + 1) * (ny_ + 1) * (nz_ + 1)));
        for (int k = 0; k <= nz_; ++k) {
            for (int j = 0; j <= ny_; ++j) {
                for (int i = 0; i <= nx_; ++i) {
                    nodes_[static_cast<std::size_t>(nodeId(i, j, k))] = {{
                        static_cast<FE::Real>(i) /
                            static_cast<FE::Real>(n_),
                        static_cast<FE::Real>(j) /
                            static_cast<FE::Real>(n_),
                        static_cast<FE::Real>(k) /
                            static_cast<FE::Real>(n_),
                    }};
                }
            }
        }

        cells_.reserve(static_cast<std::size_t>(
            6 * nx_ * ny_ * nz_));
        cell_owner_ranks_.reserve(cells_.capacity());
        for (int k = 0; k < nz_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    const FE::GlobalIndex v000 = nodeId(i, j, k);
                    const FE::GlobalIndex v100 = nodeId(i + 1, j, k);
                    const FE::GlobalIndex v010 = nodeId(i, j + 1, k);
                    const FE::GlobalIndex v110 = nodeId(i + 1, j + 1, k);
                    const FE::GlobalIndex v001 = nodeId(i, j, k + 1);
                    const FE::GlobalIndex v101 = nodeId(i + 1, j, k + 1);
                    const FE::GlobalIndex v011 = nodeId(i, j + 1, k + 1);
                    const FE::GlobalIndex v111 =
                        nodeId(i + 1, j + 1, k + 1);
                    const std::array<std::array<FE::GlobalIndex, 4>, 6>
                        cube_cells{{
                            {{v000, v100, v110, v111}},
                            {{v000, v100, v101, v111}},
                            {{v000, v010, v110, v111}},
                            {{v000, v010, v011, v111}},
                            {{v000, v001, v101, v111}},
                            {{v000, v001, v011, v111}},
                        }};
                    for (std::size_t local_tet = 0;
                         local_tet < cube_cells.size();
                         ++local_tet) {
                        cells_.push_back(cube_cells[local_tet]);
                        cell_owner_ranks_.push_back(
                            cellOwner(i, j, k, local_tet));
                    }
                }
            }
        }

        struct FaceCandidate {
            FE::GlobalIndex parent_cell{0};
            FE::LocalIndex local_face{0};
            int incidence_count{0};
        };
        constexpr std::array<std::array<std::size_t, 3>, 4>
            local_face_corners{{
                {{0u, 2u, 1u}},
                {{0u, 1u, 3u}},
                {{1u, 2u, 3u}},
                {{2u, 0u, 3u}},
            }};
        std::map<std::array<FE::GlobalIndex, 3>, FaceCandidate>
            face_candidates;
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            const auto& vertices =
                cells_[static_cast<std::size_t>(cell)];
            for (std::size_t local_face = 0u;
                 local_face < local_face_corners.size();
                 ++local_face) {
                std::array<FE::GlobalIndex, 3> key{{
                    vertices[local_face_corners[local_face][0]],
                    vertices[local_face_corners[local_face][1]],
                    vertices[local_face_corners[local_face][2]],
                }};
                std::sort(key.begin(), key.end());
                auto [entry, inserted] = face_candidates.emplace(
                    key,
                    FaceCandidate{
                        .parent_cell = cell,
                        .local_face =
                            static_cast<FE::LocalIndex>(local_face),
                        .incidence_count = 1,
                    });
                if (!inserted) {
                    ++entry->second.incidence_count;
                }
            }
        }
        for (const auto& [vertices, candidate] : face_candidates) {
            if (candidate.incidence_count != 1) {
                continue;
            }
            boundary_faces_.push_back(BoundaryFace{
                .parent_cell = candidate.parent_cell,
                .local_face = candidate.local_face,
                .marker = boundaryMarker(vertices),
            });
        }

        vertex_owner_ranks_.assign(nodes_.size(), size_);
        for (std::size_t cell = 0u; cell < cells_.size(); ++cell) {
            const int owner = cell_owner_ranks_[cell];
            for (const auto vertex : cells_[cell]) {
                auto& vertex_owner =
                    vertex_owner_ranks_[static_cast<std::size_t>(vertex)];
                vertex_owner = std::min(vertex_owner, owner);
            }
        }
    }

    void enforcePartitionOwnership() { enforce_partition_ = true; }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(cells_.size());
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        if (!enforce_partition_) {
            return numCells();
        }
        return static_cast<FE::GlobalIndex>(std::count(
            cell_owner_ranks_.begin(), cell_owner_ranks_.end(), rank_));
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(nodes_.size());
    }
    [[nodiscard]] FE::GlobalIndex numOwnedVertices() const override
    {
        if (!enforce_partition_) {
            return numVertices();
        }
        return static_cast<FE::GlobalIndex>(std::count(
            vertex_owner_ranks_.begin(), vertex_owner_ranks_.end(), rank_));
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return static_cast<FE::GlobalIndex>(boundary_faces_.size());
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override
    {
        return 0;
    }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override
    {
        return true;
    }
    [[nodiscard]] std::uint64_t geometryRevision() const override
    {
        return 31u;
    }
    [[nodiscard]] std::uint64_t topologyRevision() const override
    {
        return static_cast<std::uint64_t>(32 + n_);
    }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return static_cast<std::uint64_t>(
            40 + static_cast<int>(partition_));
    }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] FE::GlobalIndex getCellGlobalId(
        FE::GlobalIndex cell_id) const override
    {
        return 17000 + cell_id;
    }
    [[nodiscard]] FE::GlobalIndex getBoundaryFaceGlobalId(
        FE::GlobalIndex face_id) const override
    {
        return 27000 + face_id;
    }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }
    [[nodiscard]] int getCellOwnerRank(
        FE::GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(
            static_cast<std::size_t>(cell_id));
    }
    [[nodiscard]] int getBoundaryFaceOwnerRank(
        FE::GlobalIndex face_id,
        FE::GlobalIndex parent_cell) const override
    {
        const auto& face =
            boundary_faces_.at(static_cast<std::size_t>(face_id));
        if (face.parent_cell != parent_cell) {
            throw std::invalid_argument(
                "structured channel boundary parent-cell mismatch");
        }
        return getCellOwnerRank(parent_cell);
    }
    [[nodiscard]] bool isOwnedCell(
        FE::GlobalIndex cell_id) const override
    {
        return !enforce_partition_ ||
               getCellOwnerRank(cell_id) == rank_;
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra4;
    }
    void getCellNodes(
        FE::GlobalIndex cell_id,
        std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }
    void getCellCoordinates(
        FE::GlobalIndex cell_id,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coordinates.resize(cell.size());
        for (std::size_t i = 0u; i < cell.size(); ++i) {
            coordinates[i] =
                nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face_id,
        FE::GlobalIndex cell_id) const override
    {
        const auto& face =
            boundary_faces_.at(static_cast<std::size_t>(face_id));
        if (face.parent_cell != cell_id) {
            throw std::invalid_argument(
                "structured channel local-face parent mismatch");
        }
        return face.local_face;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex face_id) const override
    {
        return boundary_faces_.at(
            static_cast<std::size_t>(face_id)).marker;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }
    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            if (isOwnedCell(cell)) {
                callback(cell);
            }
        }
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback)
        const override
    {
        for (FE::GlobalIndex face = 0;
             face < numBoundaryFaces();
             ++face) {
            const auto& record =
                boundary_faces_[static_cast<std::size_t>(face)];
            if (marker < 0 || record.marker == marker) {
                callback(face, record.parent_cell);
            }
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex,
                           FE::GlobalIndex,
                           FE::GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] int cellsPerUnitLength() const noexcept { return n_; }
    [[nodiscard]] std::span<const int> cellOwnerRanks() const noexcept
    {
        return cell_owner_ranks_;
    }

    [[nodiscard]] FE::Real boundaryFaceMeasure(
        FE::GlobalIndex face_id) const
    {
        constexpr std::array<std::array<std::size_t, 3>, 4>
            local_face_corners{{
                {{0u, 2u, 1u}},
                {{0u, 1u, 3u}},
                {{1u, 2u, 3u}},
                {{2u, 0u, 3u}},
            }};
        const auto& face =
            boundary_faces_.at(static_cast<std::size_t>(face_id));
        const auto& cell =
            cells_.at(static_cast<std::size_t>(face.parent_cell));
        const auto& corners = local_face_corners.at(
            static_cast<std::size_t>(face.local_face));
        const auto& a =
            nodes_.at(static_cast<std::size_t>(cell[corners[0]]));
        const auto& b =
            nodes_.at(static_cast<std::size_t>(cell[corners[1]]));
        const auto& c =
            nodes_.at(static_cast<std::size_t>(cell[corners[2]]));
        const std::array<FE::Real, 3> ab{{
            b[0] - a[0], b[1] - a[1], b[2] - a[2]}};
        const std::array<FE::Real, 3> ac{{
            c[0] - a[0], c[1] - a[1], c[2] - a[2]}};
        const std::array<FE::Real, 3> cross{{
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        }};
        return FE::Real{0.5} *
               std::sqrt(cross[0] * cross[0] +
                         cross[1] * cross[1] +
                         cross[2] * cross[2]);
    }

    [[nodiscard]] FE::Real boundaryMarkerMeasure(int marker) const
    {
        FE::Real measure{0.0};
        for (FE::GlobalIndex face = 0;
             face < numBoundaryFaces();
             ++face) {
            if (getBoundaryFaceMarker(face) == marker) {
                measure += boundaryFaceMeasure(face);
            }
        }
        return measure;
    }

private:
    struct BoundaryFace {
        FE::GlobalIndex parent_cell{0};
        FE::LocalIndex local_face{0};
        int marker{-1};
    };

    [[nodiscard]] FE::GlobalIndex nodeId(int i, int j, int k) const
    {
        return static_cast<FE::GlobalIndex>(
            i + (nx_ + 1) * (j + (ny_ + 1) * k));
    }

    [[nodiscard]] int cellOwner(
        int i,
        int j,
        int k,
        std::size_t local_tet) const
    {
        if (partition_ == ChannelPartition::Serial || size_ == 1) {
            return 0;
        }
        if (partition_ == ChannelPartition::XSlab) {
            return std::min(size_ - 1, (i * size_) / nx_);
        }
        const std::size_t cube = static_cast<std::size_t>(
            i + nx_ * (j + ny_ * k));
        return static_cast<int>((6u * cube + local_tet) %
                                static_cast<std::size_t>(size_));
    }

    [[nodiscard]] int boundaryMarker(
        const std::array<FE::GlobalIndex, 3>& vertices) const
    {
        const auto all_on = [&](int component, FE::Real value) {
            return std::all_of(
                vertices.begin(),
                vertices.end(),
                [&](FE::GlobalIndex vertex) {
                    return std::abs(
                               nodes_[static_cast<std::size_t>(vertex)]
                                     [static_cast<std::size_t>(component)] -
                               value) <= FE::Real{1.0e-14};
                });
        };
        if (all_on(0, FE::Real{0.0})) {
            return kChannelInletMarker;
        }
        if (all_on(0, FE::Real{2.0})) {
            return kChannelOutletMarker;
        }
        if (all_on(2, FE::Real{0.0})) {
            return kChannelSideWallMarker;
        }
        return kChannelOtherWallMarker;
    }

    int n_{1};
    int nx_{2};
    int ny_{1};
    int nz_{1};
    int rank_{0};
    int size_{1};
    ChannelPartition partition_{ChannelPartition::Serial};
    bool enforce_partition_{false};
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::vector<std::array<FE::GlobalIndex, 4>> cells_{};
    std::vector<int> cell_owner_ranks_{};
    std::vector<int> vertex_owner_ranks_{};
    std::vector<BoundaryFace> boundary_faces_{};
};

FE::systems::SetupInputs makeStructuredChannelSetupInputs(
    const StructuredChannelBoundaryMesh& mesh,
    int rank,
    int size)
{
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = mesh.numCells();
    topology.n_vertices = mesh.numVertices();
    topology.dim = 3;
    topology.cell2vertex_offsets.resize(
        static_cast<std::size_t>(topology.n_cells) + 1u, 0);
    topology.cell2vertex_data.resize(
        static_cast<std::size_t>(4 * topology.n_cells), 0);
    std::vector<FE::GlobalIndex> nodes;
    for (FE::GlobalIndex cell = 0; cell < topology.n_cells; ++cell) {
        topology.cell2vertex_offsets[static_cast<std::size_t>(cell)] =
            static_cast<FE::MeshOffset>(4 * cell);
        mesh.getCellNodes(cell, nodes);
        if (nodes.size() != 4u) {
            throw std::runtime_error(
                "structured channel topology requires Tetra4 cells");
        }
        for (std::size_t local = 0u; local < 4u; ++local) {
            topology.cell2vertex_data[static_cast<std::size_t>(
                4 * cell + static_cast<FE::GlobalIndex>(local))] =
                static_cast<FE::MeshIndex>(nodes[local]);
        }
    }
    topology.cell2vertex_offsets.back() =
        static_cast<FE::MeshOffset>(4 * topology.n_cells);
    topology.vertex_gids.resize(
        static_cast<std::size_t>(topology.n_vertices), 0);
    for (FE::GlobalIndex vertex = 0;
         vertex < topology.n_vertices;
         ++vertex) {
        topology.vertex_gids[static_cast<std::size_t>(vertex)] =
            static_cast<FE::dofs::gid_t>(vertex);
    }
    topology.cell_gids.resize(
        static_cast<std::size_t>(topology.n_cells), 0);
    topology.cell_owner_ranks.assign(
        mesh.cellOwnerRanks().begin(), mesh.cellOwnerRanks().end());
    for (FE::GlobalIndex cell = 0; cell < topology.n_cells; ++cell) {
        topology.cell_gids[static_cast<std::size_t>(cell)] =
            static_cast<FE::dofs::gid_t>(cell);
    }
    for (int neighbor = 0; neighbor < size; ++neighbor) {
        if (neighbor != rank) {
            topology.neighbor_ranks.push_back(neighbor);
        }
    }
    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

std::shared_ptr<FE::spaces::FunctionSpace> makeVelocitySpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, 1, 3);
}

std::shared_ptr<FE::spaces::FunctionSpace> makePressureSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, 1, 1);
}

ns::IncompressibleNavierStokesVMSOptions baseOptions()
{
    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = 1.0;
    options.viscosity = 0.01;
    options.enable_convection = false;
    options.enable_vms = false;
    options.jit_policy.enable = false;
    return options;
}

using FreeSurfaceContactLine =
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

FreeSurfaceContactLine dynamicContactLine(int wall_marker)
{
    return FreeSurfaceContactLine{
        .configuration = FreeSurfaceContactLine::DynamicRenE{
            .wall_boundary_marker = wall_marker,
            .contact_line_marker = -1,
            .equilibrium_contact_angle_radians =
                FE::Real{1.57079632679489661923},
            .wall_normal = {0.0, 0.0, -1.0},
            .mobility = 0.5,
            .slip_length = 0.2,
        },
    };
}

int stableContactMarker(FE::FieldId level_set_field,
                        int interface_marker,
                        int wall_marker)
{
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(
        level_set_field);
    key.domain_id = "free_surface";
    key.interface_marker = interface_marker;
    key.boundary_marker = wall_marker;
    return FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
        key);
}

std::vector<FE::Real> affineScalarTetraCoefficients(
    FE::Real offset,
    const std::array<FE::Real, 3>& gradient)
{
    return {
        offset,
        offset + gradient[0],
        offset + gradient[1],
        offset + gradient[2],
    };
}

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            int component,
                            FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "sharp-boundary field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 ||
        static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error(
            "sharp-boundary field component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        offset + dofs[static_cast<std::size_t>(component)]);
    if (index >= solution.size()) {
        throw std::runtime_error(
            "sharp-boundary field DOF is out of range");
    }
    solution[index] = value;
}

void setPrescribedVertexValue(
    std::vector<FE::Real>& coefficients,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    FE::GlobalIndex vertex,
    FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "structured channel prescribed field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.size() != 1u ||
        static_cast<std::size_t>(dofs.front()) >= coefficients.size()) {
        throw std::runtime_error(
            "structured channel prescribed vertex DOF is invalid");
    }
    coefficients[static_cast<std::size_t>(dofs.front())] = value;
}

struct StructuredChannelGeometry {
    std::shared_ptr<FE::assembly::CutIntegrationContext> context{};
    std::array<FE::Real, 3> active_measures{};
    std::array<FE::Real, 3> parent_measures{};
    std::array<std::size_t, 3> active_rule_counts{};
    std::array<int, 3> active_markers{};
};

StructuredChannelGeometry makeStructuredChannelGeometry(
    const StructuredChannelBoundaryMesh& mesh,
    FE::FieldId level_set_field,
    int interface_marker,
    FE::Real interface_height,
    FE::geometry::CutIntegrationSide active_side)
{
    namespace interfaces = FE::interfaces;
    if (!std::isfinite(interface_height) ||
        interface_height <= FE::Real{0.0} ||
        interface_height >= FE::Real{1.0}) {
        throw std::invalid_argument(
            "structured channel interface height must lie in (0,1)");
    }

    constexpr std::uint64_t source_revision = 71u;
    constexpr std::uint64_t quadrature_policy_key = 73u;
    interfaces::CutInterfaceDomainRequest interface_request;
    interface_request.source =
        interfaces::LevelSetInterfaceSource::fromField(
            level_set_field, 0u, source_revision);
    interface_request.interface_marker = interface_marker;
    interface_request.tolerance = FE::Real{1.0e-12};
    interface_request.quadrature_order = 2;
    interface_request.interface_quadrature_order = 1;
    interface_request.volume_quadrature_order = 2;
    interface_request.mesh_geometry_revision = mesh.geometryRevision();
    interface_request.mesh_topology_revision = mesh.topologyRevision();
    interface_request.ownership_revision = mesh.ownershipRevision();
    interface_request.quadrature_policy_key = quadrature_policy_key;
    interface_request.implicit_geometry_mode = "LinearCorner";
    interface_request.implicit_quadrature_backend = "LinearCorner";
    interface_request.implicit_fallback_status = "None";

    interfaces::LevelSetInterfaceDomain interface_domain(
        interface_request);
    std::uint64_t stable_id = 1001u;
    std::vector<FE::GlobalIndex> cell_nodes;
    for (FE::GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = cell;
        input.element_type = FE::ElementType::Tetra4;
        input.node_coordinates = {
            {{0.0, 0.0, 0.0}},
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}},
        };
        mesh.getCellNodes(cell, cell_nodes);
        input.level_set_values.reserve(cell_nodes.size());
        for (const auto vertex : cell_nodes) {
            input.level_set_values.push_back(
                mesh.getNodeCoordinates(vertex)[1] - interface_height);
        }
        auto cut = interfaces::cutLinearLevelSetCell3D(
            interface_request, input);
        if (!cut.supported) {
            throw std::runtime_error(
                "structured channel cut is unsupported: " +
                cut.diagnostic);
        }
        for (auto& fragment : cut.fragments) {
            fragment.parent_cell_global_id =
                mesh.getCellGlobalId(cell);
            fragment.owner_rank = mesh.getCellOwnerRank(cell);
            fragment.stable_id = stable_id++;
            if (fragment.implicit_quadrature_backend.empty()) {
                fragment.implicit_quadrature_backend = "LinearCorner";
            }
            if (fragment.implicit_fallback_status.empty()) {
                fragment.implicit_fallback_status = "None";
            }
            interface_domain.addFragment(std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            region.parent_cell_global_id = mesh.getCellGlobalId(cell);
            region.owner_rank = mesh.getCellOwnerRank(cell);
            region.stable_id = stable_id++;
            if (region.implicit_quadrature_backend.empty()) {
                region.implicit_quadrature_backend = "LinearCorner";
            }
            if (region.implicit_fallback_status.empty()) {
                region.implicit_fallback_status = "None";
            }
            interface_domain.addVolumeRegion(std::move(region));
        }
    }

    auto context =
        std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(interface_domain, active_side);

    constexpr std::array<int, 3> physical_markers{{
        kChannelInletMarker,
        kChannelOutletMarker,
        kChannelSideWallMarker,
    }};
    StructuredChannelGeometry result;
    result.context = context;
    for (std::size_t role = 0u;
         role < physical_markers.size();
         ++role) {
        const int boundary_marker = physical_markers[role];
        interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey
            contact_key{};
        contact_key.source = interface_request.source;
        contact_key.domain_id = "free_surface";
        contact_key.interface_marker = interface_marker;
        contact_key.boundary_marker = boundary_marker;

        interfaces::GeneratedInterfaceBoundaryIntersectionRequest
            contact_request;
        contact_request.source = interface_request.source;
        contact_request.generated_domain_id = "free_surface";
        contact_request.interface_marker = interface_marker;
        contact_request.boundary_marker = boundary_marker;
        contact_request.intersection_marker =
            interfaces::
                stableGeneratedInterfaceBoundaryIntersectionMarker(
                    contact_key);
        contact_request.tolerance = interface_request.tolerance;
        contact_request.quadrature_order = 2;
        contact_request.frame = interface_request.frame;
        contact_request.mesh_geometry_revision =
            interface_request.mesh_geometry_revision;
        contact_request.mesh_topology_revision =
            interface_request.mesh_topology_revision;
        contact_request.ownership_revision =
            interface_request.ownership_revision;
        contact_request.quadrature_policy_key =
            interface_request.quadrature_policy_key;
        contact_request.source_value_revision = source_revision;
        auto contact_domain =
            interfaces::
                buildGeneratedInterfaceBoundaryIntersectionDomain(
                    std::move(contact_request),
                    interface_domain,
                    mesh);

        interfaces::GeneratedActiveBoundaryMarkerKey active_key{};
        active_key.source = interface_request.source;
        active_key.domain_id = "free_surface";
        active_key.interface_marker = interface_marker;
        active_key.boundary_marker = boundary_marker;
        active_key.side = active_side;
        const int active_marker =
            interfaces::stableGeneratedActiveBoundaryMarker(active_key);

        interfaces::GeneratedActiveBoundaryRequest active_request;
        active_request.source = interface_request.source;
        active_request.generated_domain_id = "free_surface";
        active_request.interface_marker = interface_marker;
        active_request.boundary_marker = boundary_marker;
        active_request.active_boundary_marker = active_marker;
        active_request.side = active_side;
        active_request.tolerance = interface_request.tolerance;
        active_request.quadrature_order = 2;
        active_request.frame = interface_request.frame;
        active_request.mesh_geometry_revision =
            interface_request.mesh_geometry_revision;
        active_request.mesh_topology_revision =
            interface_request.mesh_topology_revision;
        active_request.ownership_revision =
            interface_request.ownership_revision;
        active_request.quadrature_policy_key =
            interface_request.quadrature_policy_key;
        active_request.source_value_revision = source_revision;

        interfaces::GeneratedActiveBoundaryScalarField scalar_field;
        scalar_field.value_at_node =
            [&mesh, interface_height](FE::GlobalIndex vertex) {
                return mesh.getNodeCoordinates(vertex)[1] -
                       interface_height;
            };
        auto active_domain =
            interfaces::buildGeneratedActiveBoundaryDomain(
                std::move(active_request),
                interface_domain,
                contact_domain,
                mesh,
                scalar_field);
        result.parent_measures[role] =
            mesh.boundaryMarkerMeasure(boundary_marker);
        for (const auto& fragment : active_domain.fragments()) {
            if (!(fragment.parent_measure > FE::Real{0.0})) {
                throw std::runtime_error(
                    "structured sharp boundary has nonpositive reference parent measure");
            }
            const FE::Real physical_parent_measure =
                mesh.boundaryFaceMeasure(fragment.parent_face);
            result.active_measures[role] +=
                physical_parent_measure *
                fragment.measure / fragment.parent_measure;
        }
        result.active_rule_counts[role] =
            active_domain.boundaryQuadratureRules().size();
        result.active_markers[role] = active_marker;
        context->addGeneratedActiveBoundaryDomain(active_domain);
    }
    return result;
}

struct StructuredChannelOperatorScales {
    FE::Real inlet_traction{1.0};
    FE::Real outlet_pressure{1.0};
    FE::Real sidewall_nitsche_gamma{12.0};
};

struct StructuredChannelResidualSample {
    std::vector<FE::Real> residual{};
    StructuredChannelGeometry geometry{};
};

struct StructuredChannelOwnedRuleObservation {
    int rule_count{0};
    FE::Real measure{0.0};
};

class StructuredChannelAssemblyHarness {
public:
    StructuredChannelAssemblyHarness(
        int cells_per_unit_length,
        int rank,
        int size,
        ChannelPartition partition,
        StructuredChannelOperatorScales scales,
        bool serial_communicator)
        : rank_(rank)
        , size_(size)
        , scales_(scales)
        , mesh_(std::make_shared<StructuredChannelBoundaryMesh>(
              cells_per_unit_length,
              serial_communicator ? 0 : rank,
              serial_communicator ? 1 : size,
              serial_communicator ? ChannelPartition::Serial
                                  : partition))
        , system_(std::make_unique<FE::systems::FESystem>(mesh_))
    {
        auto velocity_space = makeVelocitySpace(mesh_);
        auto pressure_space = makePressureSpace(mesh_);
        auto options = baseOptions();
        options.traction_neumann.push_back(
            ns::IncompressibleNavierStokesVMSOptions::TractionNeumannBC{
                .boundary_marker = kChannelInletMarker,
                .traction = {
                    FE::Real{1.25} * scales_.inlet_traction,
                    FE::Real{-0.5} * scales_.inlet_traction,
                    FE::Real{0.75} * scales_.inlet_traction,
                },
            });
        options.pressure_outflow.push_back(
            ns::IncompressibleNavierStokesVMSOptions::PressureOutflowBC{
                .boundary_marker = kChannelOutletMarker,
                .pressure = FE::Real{1.2} * scales_.outlet_pressure,
                .backflow_beta = FE::Real{0.0},
            });
        options.velocity_dirichlet_weak.push_back(
            ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                .boundary_marker = kChannelSideWallMarker,
                .value = {0.1, -0.05, 0.2},
            });
        options.nitsche_gamma = scales_.sidewall_nitsche_gamma;
        options.nitsche_symmetric = true;
        options.nitsche_scale_with_p = false;
        options.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker_,
                .level_set_field_name = "phi_structured_channel",
                .generated_interface_domain_id = "free_surface",
                .active_domain =
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method =
                    ns::FreeSurfaceActiveDomainMethod::CutVolume,
                .surface_tension = FE::Real{0.0},
                .small_cut_aggregation = false,
            });

        level_set_ = system_->addField(FE::systems::FieldSpec{
            .name = "phi_structured_channel",
            .space = pressure_space,
            .components = 1,
            .source_kind =
                FE::systems::FieldSourceKind::PrescribedData,
        });
        ns::IncompressibleNavierStokesVMSModule module(
            velocity_space, pressure_space, std::move(options));
        module.registerOn(*system_);
        mesh_->enforcePartitionOwnership();

        velocity_ = system_->findFieldByName("u");
        pressure_ = system_->findFieldByName("p");
        if (velocity_ == FE::INVALID_FIELD_ID ||
            pressure_ == FE::INVALID_FIELD_ID) {
            throw std::runtime_error(
                "structured channel did not register fluid fields");
        }

        auto initial_geometry = makeStructuredChannelGeometry(
            *mesh_,
            level_set_,
            interface_marker_,
            FE::Real{0.37},
            FE::geometry::CutIntegrationSide::Negative);
        system_->setCutIntegrationContext(initial_geometry.context);

        FE::systems::SetupOptions setup_options;
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
        if (serial_communicator) {
            setup_options.dof_options.my_rank = 0;
            setup_options.dof_options.world_size = 1;
            setup_options.dof_options.mpi_comm = MPI_COMM_SELF;
        } else {
            setup_options.dof_options.my_rank = rank_;
            setup_options.dof_options.world_size = size_;
            setup_options.dof_options.mpi_comm = MPI_COMM_WORLD;
        }
#else
        (void)serial_communicator;
#endif
        system_->setup(
            setup_options,
            makeStructuredChannelSetupInputs(
                *mesh_,
                serial_communicator ? 0 : rank_,
                serial_communicator ? 1 : size_));

        solution_.assign(
            static_cast<std::size_t>(
                system_->dofHandler().getNumDofs()),
            FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh_->numVertices();
             ++vertex) {
            setFieldComponentValue(
                solution_, *system_, velocity_, vertex, 0, 0.4);
            setFieldComponentValue(
                solution_, *system_, velocity_, vertex, 1, -0.2);
            setFieldComponentValue(
                solution_, *system_, velocity_, vertex, 2, 0.3);
            setFieldComponentValue(
                solution_, *system_, pressure_, vertex, 0, 0.0);
        }
        previous_solution_ = solution_;
        setLevelSetCoefficients(FE::Real{0.37});
    }

    [[nodiscard]] StructuredChannelResidualSample assemble(
        FE::Real interface_height)
    {
        auto geometry = makeStructuredChannelGeometry(
            *mesh_,
            level_set_,
            interface_marker_,
            interface_height,
            FE::geometry::CutIntegrationSide::Negative);
        setLevelSetCoefficients(interface_height);
        system_->setCutIntegrationContext(geometry.context);

        const auto dof_count = system_->dofHandler().getNumDofs();
        FE::assembly::DenseVectorView residual(dof_count);
        residual.zero();
        FE::systems::SystemStateView state;
        state.dt = FE::Real{1.0};
        state.u = std::span<const FE::Real>(solution_);
        state.u_prev = std::span<const FE::Real>(previous_solution_);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto time_context = integrator.buildContext(1, state);
        state.time_integration = &time_context;
        FE::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_matrix = false;
        request.want_vector = true;
        const auto assembly =
            system_->assemble(request, state, nullptr, &residual);
        if (!assembly.success) {
            throw std::runtime_error(
                "structured channel assembly failed: " +
                assembly.error_message);
        }

        StructuredChannelResidualSample sample;
        sample.geometry = std::move(geometry);
        sample.residual.resize(static_cast<std::size_t>(dof_count));
        for (FE::GlobalIndex row = 0; row < dof_count; ++row) {
            sample.residual[static_cast<std::size_t>(row)] =
                residual[row];
        }
        return sample;
    }

    [[nodiscard]] std::vector<FE::Real> constantVelocityProbe(
        const std::array<FE::Real, 3>& value) const
    {
        std::vector<FE::Real> probe(solution_.size(), FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh_->numVertices();
             ++vertex) {
            for (int component = 0; component < 3; ++component) {
                setFieldComponentValue(
                    probe,
                    *system_,
                    velocity_,
                    vertex,
                    component,
                    value[static_cast<std::size_t>(component)]);
            }
        }
        return probe;
    }

    [[nodiscard]] FE::GlobalIndex cellCount() const noexcept
    {
        return mesh_->numCells();
    }

    [[nodiscard]] StructuredChannelOwnedRuleObservation observeOwnedRules(
        const StructuredChannelGeometry& geometry) const
    {
        StructuredChannelOwnedRuleObservation observation;
        observation.measure = std::accumulate(
            geometry.active_measures.begin(),
            geometry.active_measures.end(),
            FE::Real{0.0});
        for (const int marker : geometry.active_markers) {
            for (const auto* rule :
                 geometry.context->interfaceRulesForMarker(marker)) {
                if (rule != nullptr &&
                    rule->provenance.owner_rank == rank_) {
                    ++observation.rule_count;
                }
            }
        }
        return observation;
    }

private:
    void setLevelSetCoefficients(FE::Real interface_height)
    {
        const auto& handler = system_->fieldDofHandler(level_set_);
        std::vector<FE::Real> coefficients(
            static_cast<std::size_t>(handler.getNumDofs()),
            FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh_->numVertices();
             ++vertex) {
            setPrescribedVertexValue(
                coefficients,
                *system_,
                level_set_,
                vertex,
                mesh_->getNodeCoordinates(vertex)[1] -
                    interface_height);
        }
        system_->setPrescribedFieldCoefficients(
            level_set_, std::move(coefficients));
    }

    static constexpr int interface_marker_{239};
    int rank_{0};
    int size_{1};
    StructuredChannelOperatorScales scales_{};
    std::shared_ptr<StructuredChannelBoundaryMesh> mesh_{};
    std::unique_ptr<FE::systems::FESystem> system_{};
    FE::FieldId level_set_{FE::INVALID_FIELD_ID};
    FE::FieldId velocity_{FE::INVALID_FIELD_ID};
    FE::FieldId pressure_{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution_{};
    std::vector<FE::Real> previous_solution_{};
};

struct StructuredGeneratedCoupledOutflowSample {
    FE::Real flow_rate{0.0};
    std::array<FE::Real, 4> local_gradient_actions{};
    FE::Real local_traction_work{0.0};
    FE::Real local_active_measure{0.0};
    int local_owned_rule_count{0};
    int local_owner_mismatch_count{0};
    int local_outlet_contributor_count{0};
};

class StructuredGeneratedCoupledOutflowHarness {
public:
    StructuredGeneratedCoupledOutflowHarness(
        int rank,
        int size,
        ChannelPartition partition,
        bool serial_communicator)
        : rank_(serial_communicator ? 0 : rank)
        , size_(serial_communicator ? 1 : size)
        , mesh_(std::make_shared<StructuredChannelBoundaryMesh>(
              1,
              rank_,
              size_,
              serial_communicator ? ChannelPartition::Serial
                                  : partition))
        , system_(std::make_unique<FE::systems::FESystem>(mesh_))
    {
        const auto velocity_space = makeVelocitySpace(mesh_);
        const auto level_set_space = makePressureSpace(mesh_);
        velocity_ = system_->addField(FE::systems::FieldSpec{
            .name = "u_generated_coupled_outflow",
            .space = velocity_space,
            .components = 3,
        });
        level_set_ = system_->addField(FE::systems::FieldSpec{
            .name = "phi_generated_coupled_outflow",
            .space = level_set_space,
            .components = 1,
            .source_kind =
                FE::systems::FieldSourceKind::PrescribedData,
        });
        system_->addOperator(operator_tag_);
        mesh_->enforcePartitionOwnership();

        geometry_ = makeStructuredChannelGeometry(
            *mesh_,
            level_set_,
            interface_marker_,
            interface_height_,
            FE::geometry::CutIntegrationSide::Negative);
        active_marker_ = geometry_.active_markers[1];
        system_->setCutIntegrationContext(geometry_.context);
        system_->registerGeneratedEmbeddedInterfaceMarker(active_marker_);

        FE::forms::BoundaryFunctional flow;
        flow.integrand = FE::forms::inner(
            FE::forms::FormExpr::discreteField(
                velocity_, *velocity_space, "u_generated_coupled_outflow"),
            FE::forms::FormExpr::normal());
        flow.boundary_marker = kChannelOutletMarker;
        flow.generated_active_boundary_marker = active_marker_;
        flow.reduction =
            FE::forms::BoundaryFunctional::Reduction::Sum;
        flow_rate_ = system_->boundaryIntegral(
            std::move(flow),
            FE::systems::AuxiliaryInputUpdateSchedule::
                EachNonlinearIteration);

        const auto u = FE::forms::FormExpr::stateField(
            velocity_, *velocity_space, "u_generated_coupled_outflow");
        const auto v = FE::forms::FormExpr::testFunction(
            *velocity_space, "v_generated_coupled_outflow");
        const auto pressure =
            FE::forms::FormExpr::constant(resistance_) *
            flow_rate_.expr();
        const auto flux =
            -(pressure * FE::forms::FormExpr::normal());
        FE::systems::BoundaryConditionManager boundary_conditions;
        boundary_conditions.add(
            std::make_unique<FE::forms::bc::NaturalBC>(
                kChannelOutletMarker,
                flux,
                active_marker_));
        auto residual =
            (FE::forms::FormExpr::constant(0.0) *
             FE::forms::inner(u, v))
                .dx();
        boundary_conditions.applyAll(
            *system_, residual, u, v, velocity_);
        (void)FE::systems::installFormulation(
            *system_, operator_tag_, {velocity_}, residual);

        FE::systems::SetupOptions setup_options;
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
        setup_options.dof_options.my_rank = rank_;
        setup_options.dof_options.world_size = size_;
        setup_options.dof_options.mpi_comm =
            serial_communicator ? MPI_COMM_SELF : MPI_COMM_WORLD;
#else
        (void)serial_communicator;
#endif
        system_->setup(
            setup_options,
            makeStructuredChannelSetupInputs(
                *mesh_, rank_, size_));

        solution_.assign(
            static_cast<std::size_t>(
                system_->dofHandler().getNumDofs()),
            FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh_->numVertices();
             ++vertex) {
            setFieldComponentValue(
                solution_,
                *system_,
                velocity_,
                vertex,
                0,
                velocity_magnitude_);
        }
        setLevelSetCoefficients();
    }

    [[nodiscard]] StructuredGeneratedCoupledOutflowSample sample()
    {
        FE::systems::SystemStateView state;
        state.dt = FE::Real{1.0};
        state.u = std::span<const FE::Real>(solution_);
        system_->prepareAuxiliaryForAssembly(
            state, /*is_nonlinear_iteration=*/true);

        const auto* registry =
            system_->auxiliaryInputRegistryIfPresent();
        if (registry == nullptr ||
            !registry->hasInput(flow_rate_.registryName())) {
            throw std::runtime_error(
                "generated coupled-outflow flow input is unavailable");
        }

        StructuredGeneratedCoupledOutflowSample result;
        result.flow_rate =
            registry->get(flow_rate_.registryName());

        auto* reductions =
            system_->boundaryReductionServiceIfPresent(velocity_);
        if (reductions == nullptr ||
            !reductions->hasFunctional(flow_rate_.registryName())) {
            throw std::runtime_error(
                "generated coupled-outflow reduction service is unavailable");
        }
        const auto gradient =
            reductions->evaluateFunctionalGradient(
                flow_rate_.registryName(),
                velocity_,
                state,
                /*apply_constraints=*/false);
        const auto probes = velocityProbes();
        for (const auto& entry : gradient) {
            if (entry.dof < 0 ||
                static_cast<std::size_t>(entry.dof) >=
                    solution_.size()) {
                throw std::runtime_error(
                    "generated coupled-outflow gradient DOF is out of range");
            }
            const auto index =
                static_cast<std::size_t>(entry.dof);
            for (std::size_t probe_index = 0;
                 probe_index < probes.size();
                 ++probe_index) {
                result.local_gradient_actions[probe_index] +=
                    entry.value * probes[probe_index][index];
            }
        }

        const auto dof_count =
            system_->dofHandler().getNumDofs();
        FE::assembly::DenseVectorView residual(dof_count);
        residual.zero();
        FE::systems::AssemblyRequest request;
        request.op = operator_tag_;
        request.want_matrix = false;
        request.want_vector = true;
        request.is_nonlinear_iteration = true;
        const auto assembly =
            system_->assemble(
                request, state, nullptr, &residual);
        if (!assembly.success) {
            throw std::runtime_error(
                "generated coupled-outflow assembly failed: " +
                assembly.error_message);
        }
        std::vector<FE::Real> residual_values(
            static_cast<std::size_t>(dof_count),
            FE::Real{0.0});
        for (FE::GlobalIndex row = 0;
             row < dof_count;
             ++row) {
            residual_values[static_cast<std::size_t>(row)] =
                residual[row];
        }
        for (std::size_t dof = 0;
             dof < residual_values.size();
             ++dof) {
            result.local_traction_work +=
                residual_values[dof] * probes[0][dof];
        }
        result.local_active_measure =
            geometry_.active_measures[1];
        result.local_outlet_contributor_count =
            result.local_active_measure >
                    FE::Real{1.0e-14}
                ? 1
                : 0;

        for (const auto* rule :
             geometry_.context->interfaceRulesForMarker(
                 active_marker_)) {
            if (rule == nullptr) {
                ++result.local_owner_mismatch_count;
                continue;
            }
            const auto parent_cell =
                static_cast<FE::GlobalIndex>(
                    rule->provenance.parent_entity);
            if (parent_cell < 0 ||
                parent_cell >= mesh_->numCells()) {
                ++result.local_owner_mismatch_count;
                continue;
            }
            const int expected_owner =
                mesh_->getCellOwnerRank(parent_cell);
            if (rule->provenance.owner_rank != expected_owner ||
                expected_owner != rank_ ||
                !mesh_->isOwnedCell(parent_cell)) {
                ++result.local_owner_mismatch_count;
                continue;
            }
            ++result.local_owned_rule_count;
        }
        return result;
    }

    [[nodiscard]] bool hasDualMarkerContract() const
    {
        const auto* definition = flow_rate_.definition();
        const auto* registry =
            system_->auxiliaryInputRegistryIfPresent();
        const auto* reductions =
            system_->boundaryReductionServiceIfPresent(velocity_);
        if (definition == nullptr ||
            registry == nullptr ||
            reductions == nullptr ||
            !registry->hasInput(flow_rate_.registryName()) ||
            !reductions->hasFunctional(flow_rate_.registryName())) {
            return false;
        }
        const auto& functional =
            reductions->functionalDef(flow_rate_.registryName());
        return definition->boundary_marker ==
                   kChannelOutletMarker &&
               definition->generated_active_boundary_marker ==
                   active_marker_ &&
               registry->specOf(flow_rate_.registryName())
                       .boundary_marker ==
                   kChannelOutletMarker &&
               functional.boundary_marker ==
                   kChannelOutletMarker &&
               functional.generated_active_boundary_marker ==
                   active_marker_;
    }

    [[nodiscard]] FE::GlobalIndex cellCount() const noexcept
    {
        return mesh_->numCells();
    }

private:
    [[nodiscard]] std::array<std::vector<FE::Real>, 4>
    velocityProbes() const
    {
        std::array<std::vector<FE::Real>, 4> probes;
        for (auto& probe : probes) {
            probe.assign(solution_.size(), FE::Real{0.0});
        }
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh_->numVertices();
             ++vertex) {
            const auto point =
                mesh_->getNodeCoordinates(vertex);
            setFieldComponentValue(
                probes[0],
                *system_,
                velocity_,
                vertex,
                0,
                FE::Real{1.0});
            setFieldComponentValue(
                probes[1],
                *system_,
                velocity_,
                vertex,
                0,
                point[1]);
            setFieldComponentValue(
                probes[2],
                *system_,
                velocity_,
                vertex,
                0,
                point[2]);
            setFieldComponentValue(
                probes[3],
                *system_,
                velocity_,
                vertex,
                1,
                FE::Real{1.0});
        }
        return probes;
    }

    void setLevelSetCoefficients()
    {
        const auto& handler =
            system_->fieldDofHandler(level_set_);
        std::vector<FE::Real> coefficients(
            static_cast<std::size_t>(
                handler.getNumDofs()),
            FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0;
             vertex < mesh_->numVertices();
             ++vertex) {
            setPrescribedVertexValue(
                coefficients,
                *system_,
                level_set_,
                vertex,
                mesh_->getNodeCoordinates(vertex)[1] -
                    interface_height_);
        }
        system_->setPrescribedFieldCoefficients(
            level_set_, std::move(coefficients));
    }

    static constexpr const char* operator_tag_ =
        "generated_active_coupled_outflow";
    static constexpr int interface_marker_{238};
    static constexpr FE::Real interface_height_{0.37};
    static constexpr FE::Real velocity_magnitude_{0.4};
    static constexpr FE::Real resistance_{2.0};
    int rank_{0};
    int size_{1};
    std::shared_ptr<StructuredChannelBoundaryMesh> mesh_{};
    std::unique_ptr<FE::systems::FESystem> system_{};
    FE::FieldId velocity_{FE::INVALID_FIELD_ID};
    FE::FieldId level_set_{FE::INVALID_FIELD_ID};
    int active_marker_{-1};
    StructuredChannelGeometry geometry_{};
    FE::systems::AuxiliaryInputHandle flow_rate_{};
    std::vector<FE::Real> solution_{};
};

std::shared_ptr<FE::assembly::CutIntegrationContext>
makeFreeSurfaceContext(
    int interface_marker,
    int wall_marker,
    int contact_marker,
    FE::FieldId level_set_field,
    FE::geometry::CutIntegrationSide active_side,
    int owner_rank,
    std::optional<FE::Real> active_boundary_measure)
{
    namespace interfaces = FE::interfaces;

    const std::array<FE::Real, 3> interface_normal =
        active_side == FE::geometry::CutIntegrationSide::Positive
            ? std::array<FE::Real, 3>{{0.0, -1.0, 0.0}}
            : std::array<FE::Real, 3>{{0.0, 1.0, 0.0}};
    constexpr std::array<FE::Real, 3> boundary_normal{{0.0, 0.0, -1.0}};

    interfaces::CutInterfaceDomainRequest interface_request;
    interface_request.source = interfaces::LevelSetInterfaceSource::fromField(
        level_set_field, 0u, 1u);
    interface_request.interface_marker = interface_marker;
    interface_request.quadrature_order = 0;
    interface_request.interface_quadrature_order = 0;
    interface_request.volume_quadrature_order = 0;

    interfaces::LevelSetInterfaceDomain interface_domain(interface_request);
    interfaces::CutInterfaceFragment interface_fragment;
    interface_fragment.interface_marker = interface_marker;
    interface_fragment.parent_cell = 0;
    interface_fragment.parent_cell_global_id = 1701;
    interface_fragment.owner_rank = owner_rank;
    interface_fragment.local_fragment_index = 0;
    interface_fragment.stable_id = 10;
    interface_fragment.kind = interfaces::CutInterfaceFragmentKind::Polygon;
    interface_fragment.measure = FE::Real{0.038};
    interface_fragment.normal = interface_normal;
    interface_fragment.quadrature_points.push_back(
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.20, 0.25, 0.15}},
            .parent_coordinate = {{0.20, 0.25, 0.15}},
            .normal = interface_normal,
            .weight = FE::Real{0.020},
        });
    interface_fragment.quadrature_points.push_back(
        interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.35, 0.18, 0.22}},
            .parent_coordinate = {{0.35, 0.18, 0.22}},
            .normal = interface_normal,
            .weight = FE::Real{0.018},
        });
    interface_domain.addFragment(std::move(interface_fragment));

    for (const auto side : kActiveSides) {
        interfaces::CutInterfaceVolumeRegion region;
        region.interface_marker = interface_marker;
        region.parent_cell = 0;
        region.parent_cell_global_id = 1701;
        region.owner_rank = owner_rank;
        region.local_region_index =
            side == FE::geometry::CutIntegrationSide::Negative ? 0 : 1;
        region.stable_id =
            side == FE::geometry::CutIntegrationSide::Negative ? 11 : 12;
        region.side = side;
        region.measure = FE::Real{1.0} / FE::Real{12.0};
        region.parent_measure = FE::Real{1.0} / FE::Real{6.0};
        region.volume_fraction = region.measure / region.parent_measure;
        region.centroid =
            side == FE::geometry::CutIntegrationSide::Negative
                ? std::array<FE::Real, 3>{{0.24, 0.18, 0.20}}
                : std::array<FE::Real, 3>{{0.44, 0.20, 0.18}};
        region.normal = interface_normal;
        interface_domain.addVolumeRegion(std::move(region));
    }

    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(interface_domain, active_side);

    interfaces::GeneratedInterfaceBoundaryIntersectionRequest contact_request;
    contact_request.source = interfaces::LevelSetInterfaceSource::fromField(
        level_set_field, 0u, 1u);
    contact_request.generated_domain_id = "free_surface";
    contact_request.interface_marker = interface_marker;
    contact_request.boundary_marker = wall_marker;
    contact_request.intersection_marker = contact_marker;
    contact_request.quadrature_order = 1;

    interfaces::GeneratedInterfaceBoundaryIntersectionDomain contact_domain(
        contact_request);
    interfaces::GeneratedInterfaceBoundaryIntersectionFragment contact_fragment;
    contact_fragment.interface_marker = interface_marker;
    contact_fragment.boundary_marker = wall_marker;
    contact_fragment.intersection_marker = contact_marker;
    contact_fragment.parent_cell = 0;
    contact_fragment.parent_face = 0;
    contact_fragment.parent_cell_global_id = 1701;
    contact_fragment.parent_face_global_id = 2701;
    contact_fragment.owner_rank = owner_rank;
    contact_fragment.kind =
        interfaces::GeneratedInterfaceBoundaryIntersectionKind::Segment;
    contact_fragment.measure = FE::Real{0.125};
    contact_fragment.interface_normal = interface_normal;
    contact_fragment.boundary_normal = boundary_normal;
    contact_fragment.tangent = {{1.0, 0.0, 0.0}};
    contact_fragment.quadrature_points.push_back(
        interfaces::GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
            .point = {{0.2, 0.2, 0.0}},
            .parent_coordinate = {{0.2, 0.2, 0.0}},
            .interface_normal = interface_normal,
            .boundary_normal = boundary_normal,
            .tangent = {{1.0, 0.0, 0.0}},
            .weight = contact_fragment.measure,
            .reference_measure_factor = contact_fragment.measure,
            .gradient_norm = FE::Real{1.0},
        });
    contact_domain.addFragment(std::move(contact_fragment));
    context->addGeneratedInterfaceBoundaryIntersectionDomain(contact_domain);

    if (active_boundary_measure.has_value()) {
        interfaces::GeneratedActiveBoundaryRequest active_request;
        active_request.source = interfaces::LevelSetInterfaceSource::fromField(
            level_set_field, 0u, 1u);
        active_request.generated_domain_id = "free_surface";
        active_request.interface_marker = interface_marker;
        active_request.boundary_marker = wall_marker;
        active_request.side = active_side;
        active_request.quadrature_order = 1;

        interfaces::GeneratedActiveBoundaryDomain active_domain(active_request);
        if (*active_boundary_measure > FE::Real{0.0}) {
            interfaces::GeneratedActiveBoundaryFragment active_fragment;
            active_fragment.interface_marker = interface_marker;
            active_fragment.boundary_marker = wall_marker;
            active_fragment.parent_cell = 0;
            active_fragment.parent_face = 0;
            active_fragment.parent_cell_global_id = 1701;
            active_fragment.parent_face_global_id = 2701;
            active_fragment.owner_rank = owner_rank;
            active_fragment.side = active_side;
            active_fragment.represented_implicit_geometry_mode =
                "LinearCorner";
            active_fragment.represented_implicit_quadrature_backend =
                "LinearCorner";
            active_fragment.represented_implicit_fallback_status = "None";
            active_fragment.boundary_normal = boundary_normal;
            active_fragment.measure = *active_boundary_measure;
            active_fragment.parent_measure = kParentBoundaryMeasure;
            active_fragment.achieved_quadrature_order = 1;
            active_fragment.topology_id =
                "partitioned-single-tetra-sharp-wall";
            active_fragment.quadrature_points.push_back(
                FE::geometry::CutQuadraturePoint{
                    .point = {{0.10, 0.30, 0.0}},
                    .normal = boundary_normal,
                    .weight = *active_boundary_measure,
                    .parent_coordinate = {{0.10, 0.30, 0.0}},
                    .reference_measure_factor = *active_boundary_measure,
                });
            active_domain.addFragment(std::move(active_fragment));
        }
        context->addGeneratedActiveBoundaryDomain(active_domain);
    }
    return context;
}

enum class OperatorFamily {
    Traction,
    Robin,
    PressureFlux,
    Outflow,
    SymmetricNitsche,
    UnsymmetricNitsche,
    WallSlip,
};

constexpr std::array<OperatorFamily, 7> kOperatorFamilies{{
    OperatorFamily::Traction,
    OperatorFamily::Robin,
    OperatorFamily::PressureFlux,
    OperatorFamily::Outflow,
    OperatorFamily::SymmetricNitsche,
    OperatorFamily::UnsymmetricNitsche,
    OperatorFamily::WallSlip,
}};

struct AssemblySample {
    std::vector<FE::Real> residual{};
    std::vector<FE::Real> jacobian{};
};

struct ActiveRuleObservation {
    bool valid{false};
    int owner_rank{-1};
    FE::Real measure{0.0};
};

struct GeneratedBoundaryAssemblySample {
    AssemblySample assembly{};
    FE::Real active_measure{0.0};
    FE::Real parent_measure{0.0};
    std::size_t active_rule_count{0u};
};

struct GeneratedBoundaryContext {
    std::shared_ptr<FE::assembly::CutIntegrationContext> context{};
    FE::Real active_measure{0.0};
    FE::Real parent_measure{0.0};
    std::size_t active_rule_count{0u};
};

struct GeneratedBoundaryMetricSample {
    FE::Real active_measure{0.0};
    FE::Real parent_measure{0.0};
    FE::Real cell_diameter{0.0};
    FE::Real cell_measure{0.0};
    FE::Real facet_measure{0.0};
    std::size_t active_rule_count{0u};
};

struct EmptyGeneratedDomainObservation {
    bool active_marker_registered{false};
    bool contact_marker_registered{false};
    std::size_t active_rule_count{0u};
    std::size_t contact_rule_count{0u};
};

FE::Real work(std::span<const FE::Real> residual,
              std::span<const FE::Real> solution);

class SharpBoundaryAssemblyHarness {
public:
    SharpBoundaryAssemblyHarness(OperatorFamily family,
                                 FE::geometry::CutIntegrationSide active_side,
                                 int rank,
                                 int size,
                                 int owner_rank,
                                 bool serial_communicator = false,
                                 FE::Real operator_scale = FE::Real{1.0},
                                 FE::Real length_scale = FE::Real{1.0},
                                 FE::Real nitsche_gamma = FE::Real{12.0})
        : family_(family)
        , active_side_(active_side)
        , owner_rank_(owner_rank)
        , operator_scale_(operator_scale)
        , length_scale_(length_scale)
        , nitsche_gamma_(nitsche_gamma)
        , mesh_(std::make_shared<PartitionedSingleTetraBoundaryMesh>(
              wall_marker_, rank, size, owner_rank, length_scale))
        , system_(std::make_unique<FE::systems::FESystem>(mesh_))
    {
        auto velocity_space = makeVelocitySpace(mesh_);
        auto pressure_space = makePressureSpace(mesh_);
        auto options = baseOptions();

        auto free_surface =
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker_,
                .level_set_field_name = "phi_sharp_partition_work",
                .generated_interface_domain_id = "free_surface",
                .active_domain =
                    active_side == FE::geometry::CutIntegrationSide::Positive
                        ? ns::FreeSurfaceActiveDomain::LevelSetPositive
                        : ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method =
                    ns::FreeSurfaceActiveDomainMethod::CutVolume,
                .surface_tension = FE::Real{0.0},
                .small_cut_aggregation = false,
            };

        switch (family_) {
        case OperatorFamily::Traction:
            options.traction_neumann.push_back(
                ns::IncompressibleNavierStokesVMSOptions::TractionNeumannBC{
                    .boundary_marker = wall_marker_,
                    .traction = {1.25 * operator_scale_,
                                 -0.5 * operator_scale_,
                                 0.75 * operator_scale_},
                });
            break;
        case OperatorFamily::Robin:
            options.traction_robin.push_back(
                ns::IncompressibleNavierStokesVMSOptions::TractionRobinBC{
                    .boundary_marker = wall_marker_,
                    .alpha = 1.7,
                    .rhs = {0.1, -0.2, 0.3},
                });
            break;
        case OperatorFamily::PressureFlux:
            options.enable_vms = true;
            options.velocity_dirichlet_weak.push_back(
                ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                    .boundary_marker = wall_marker_,
                    .value = {0.0, 0.0, 0.0},
                });
            options.nitsche_gamma = FE::Real{12.0};
            options.nitsche_scale_with_p = false;
            break;
        case OperatorFamily::Outflow:
            options.pressure_outflow.push_back(
                ns::IncompressibleNavierStokesVMSOptions::PressureOutflowBC{
                    .boundary_marker = wall_marker_,
                    .pressure = 1.2,
                    .backflow_beta = 0.25,
                });
            break;
        case OperatorFamily::SymmetricNitsche:
        case OperatorFamily::UnsymmetricNitsche:
            options.velocity_dirichlet_weak.push_back(
                ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                    .boundary_marker = wall_marker_,
                    .value = {0.1, -0.05, 0.2},
                });
            options.nitsche_gamma = nitsche_gamma_;
            options.nitsche_symmetric =
                family_ == OperatorFamily::SymmetricNitsche;
            options.nitsche_scale_with_p = false;
            break;
        case OperatorFamily::WallSlip:
            options.velocity_dirichlet.push_back(
                ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                    .boundary_marker = wall_marker_,
                    .value = {0.0, 0.0, 0.0},
                    .active_components = {false, false, true},
                });
            free_surface.surface_tension = 0.8;
            free_surface.surface_tension_form =
                ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction;
            free_surface.curvature = 0.0;
            free_surface.use_level_set_curvature = false;
            free_surface.contact_lines.push_back(
                dynamicContactLine(wall_marker_));
            break;
        }
        options.free_surface.push_back(std::move(free_surface));

        level_set_ = system_->addField(FE::systems::FieldSpec{
            .name = "phi_sharp_partition_work",
            .space = pressure_space,
            .components = 1,
            .source_kind =
                family_ == OperatorFamily::WallSlip
                    ? FE::systems::FieldSourceKind::Unknown
                    : FE::systems::FieldSourceKind::PrescribedData,
        });
        if (family_ == OperatorFamily::WallSlip) {
            system_->addOperator("equations");
            const auto phi_state = FE::forms::StateField(
                level_set_, *pressure_space, "phi_partition_owner");
            const auto eta = FE::forms::TestField(
                level_set_, *pressure_space, "eta_partition_owner");
            (void)FE::systems::installFormulation(
                *system_,
                "equations",
                {level_set_},
                (FE::forms::dt(phi_state) * eta).dx());
            system_->gaugeRegistry().addAnchoring(
                FE::gauge::AnchoringEvidence{
                    .field = level_set_,
                    .component = -1,
                    .region = -1,
                    .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
                    .verdict = FE::gauge::AnchoringVerdict::Anchored,
                    .source =
                        "Transient level-set owner in partition fixture",
                });
        }

        std::optional<ScopedEnvVar> pressure_flux_scale;
        std::optional<ScopedEnvVar> pressure_flux_scale_alias;
        std::optional<ScopedEnvVar> force_vms_enable;
        std::optional<ScopedEnvVar> clear_vms_disable;
        if (family_ == OperatorFamily::PressureFlux) {
            pressure_flux_scale.emplace(
                "SVMP_NS_PSPG_BOUNDARY_PRESSURE_FLUX_SCALE",
                operator_scale_ > FE::Real{0.0}
                    ? std::optional<std::string>(
                          std::to_string(
                              static_cast<double>(operator_scale_)))
                    : std::nullopt);
            pressure_flux_scale_alias.emplace(
                "SVMP_PSPG_BOUNDARY_PRESSURE_FLUX_SCALE",
                std::nullopt);
            force_vms_enable.emplace(
                "SVMP_NS_ENABLE_VMS", std::string("1"));
            clear_vms_disable.emplace(
                "SVMP_NS_DISABLE_VMS", std::nullopt);
        }
        ns::IncompressibleNavierStokesVMSModule module(
            velocity_space, pressure_space, std::move(options));
        module.registerOn(*system_);
        // Registration-time wall validation precedes FESystem setup and thus
        // has only a self communicator for synthetic mesh adapters.  Every
        // rank validates its visible ghost face locally; assembly ownership is
        // enforced immediately afterward and remains enforced for all samples.
        mesh_->enforcePartitionOwnership();
        velocity_ = system_->findFieldByName("u");
        pressure_ = system_->findFieldByName("p");
        if (velocity_ == FE::INVALID_FIELD_ID ||
            pressure_ == FE::INVALID_FIELD_ID) {
            throw std::runtime_error(
                "sharp-boundary partition fixture did not register fluid fields");
        }

        FE::interfaces::GeneratedActiveBoundaryMarkerKey marker_key;
        marker_key.source =
            FE::interfaces::LevelSetInterfaceSource::fromField(level_set_);
        marker_key.domain_id = "free_surface";
        marker_key.interface_marker = interface_marker_;
        marker_key.boundary_marker = wall_marker_;
        marker_key.side = active_side_;
        active_marker_ =
            FE::interfaces::stableGeneratedActiveBoundaryMarker(marker_key);
        contact_marker_ = stableContactMarker(
            level_set_, interface_marker_, wall_marker_);

        system_->addOperator("sharp_entity_measure_probe");
        system_->addOperator("sharp_empty_probe");
        const auto metric_test = FE::forms::TestField(
            velocity_, *velocity_space, "sharp_entity_measure_test");
        const auto metric_values = FE::forms::FormExpr::asVector({
            FE::forms::h(),
            FE::forms::vol(),
            FE::forms::area(),
        });
        (void)FE::systems::installFormulation(
            *system_,
            "sharp_entity_measure_probe",
            {velocity_},
            FE::forms::inner(metric_values, metric_test).dI(active_marker_));

        system_->setCutIntegrationContext(context(FE::Real{0.0}));
        FE::systems::SetupOptions setup_options;
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
        if (serial_communicator) {
            setup_options.dof_options.my_rank = 0;
            setup_options.dof_options.world_size = 1;
            setup_options.dof_options.mpi_comm = MPI_COMM_SELF;
        } else {
            setup_options.dof_options.my_rank = rank;
            setup_options.dof_options.world_size = size;
            setup_options.dof_options.mpi_comm = MPI_COMM_WORLD;
        }
#else
        (void)serial_communicator;
#endif
        system_->setup(
            setup_options,
            makePartitionedSingleTetraSetupInputs(
                serial_communicator ? 0 : owner_rank_));

        const std::array<FE::Real, 3> level_set_gradient =
            active_side_ == FE::geometry::CutIntegrationSide::Positive
                ? std::array<FE::Real, 3>{{0.0, -1.0, 0.0}}
                : std::array<FE::Real, 3>{{0.0, 1.0, 0.0}};
        const auto level_set_values =
            affineScalarTetraCoefficients(-0.2, level_set_gradient);
        if (family_ != OperatorFamily::WallSlip) {
            system_->setPrescribedFieldCoefficients(
                level_set_, level_set_values);
        }

        solution_.assign(
            static_cast<std::size_t>(system_->dofHandler().getNumDofs()),
            FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            setFieldComponentValue(
                solution_, *system_, velocity_, vertex, 0, 0.4);
            setFieldComponentValue(
                solution_, *system_, velocity_, vertex, 1, -0.2);
            setFieldComponentValue(
                solution_, *system_, velocity_, vertex, 2, 0.4);
            setFieldComponentValue(
                solution_,
                *system_,
                pressure_,
                vertex,
                0,
                family_ == OperatorFamily::PressureFlux
                    ? mesh_->getNodeCoordinates(vertex)[2]
                    : FE::Real{0.0});
            if (family_ == OperatorFamily::WallSlip) {
                setFieldComponentValue(
                    solution_,
                    *system_,
                    level_set_,
                    vertex,
                    0,
                    level_set_values[static_cast<std::size_t>(vertex)]);
            }
        }
        previous_solution_ = solution_;
    }

    [[nodiscard]] AssemblySample assemble(
        std::optional<FE::Real> active_measure)
    {
        return assembleContext(context(active_measure));
    }

    [[nodiscard]] GeneratedBoundaryAssemblySample assembleGeneratedFraction(
        FE::Real fraction,
        const char* operation = "equations")
    {
        if (!std::isfinite(fraction) || fraction < FE::Real{0.0} ||
            fraction > FE::Real{1.0}) {
            throw std::invalid_argument(
                "generated sharp-boundary fraction must lie in [0,1]");
        }
        FE::Real intercept{0.0};
        if (fraction <= FE::Real{0.0}) {
            intercept = -FE::Real{0.125} * length_scale_;
        } else if (fraction >= FE::Real{1.0}) {
            auto generated = generatedContext(
                FE::Real{0.5} * length_scale_, true);
            return GeneratedBoundaryAssemblySample{
                .assembly = assembleContext(generated.context, operation),
                .active_measure = generated.active_measure,
                .parent_measure = generated.parent_measure,
                .active_rule_count = generated.active_rule_count,
            };
        } else {
            intercept = length_scale_ *
                        (FE::Real{1.0} -
                         std::sqrt(FE::Real{1.0} - fraction));
        }
        return assembleGeneratedIntercept(intercept, operation);
    }

    [[nodiscard]] EmptyGeneratedDomainObservation
    observeExplicitlyEmptyGeneratedDomains() const
    {
        auto generated = generatedContext(
            -FE::Real{0.125} * length_scale_);
        const auto active_rules =
            generated.context->interfaceRulesForMarker(active_marker_);
        const auto contact_rules =
            generated.context->interfaceRulesForMarker(contact_marker_);
        return EmptyGeneratedDomainObservation{
            .active_marker_registered =
                generated.context->hasGeneratedInterfaceMarker(
                    active_marker_),
            .contact_marker_registered =
                generated.context->hasGeneratedInterfaceMarker(
                    contact_marker_),
            .active_rule_count = active_rules.size(),
            .contact_rule_count = contact_rules.size(),
        };
    }

    [[nodiscard]] GeneratedBoundaryMetricSample
    observeGeneratedMetrics(FE::Real fraction)
    {
        if (!std::isfinite(fraction) || !(fraction > FE::Real{0.0}) ||
            fraction > FE::Real{1.0}) {
            throw std::invalid_argument(
                "generated metric fraction must lie in (0,1]");
        }
        GeneratedBoundaryContext generated;
        if (fraction >= FE::Real{1.0}) {
            generated = generatedContext(
                FE::Real{0.5} * length_scale_, true);
        } else {
            const FE::Real intercept =
                length_scale_ *
                (FE::Real{1.0} -
                 std::sqrt(FE::Real{1.0} - fraction));
            generated = generatedContext(intercept);
        }
        if (!(generated.active_measure > FE::Real{0.0})) {
            throw std::runtime_error(
                "generated metric trace has nonpositive measure");
        }
        const auto metric_sample = assembleContext(
            generated.context, "sharp_entity_measure_probe");
        std::array<FE::Real, 3> values{};
        for (int component = 0; component < 3; ++component) {
            std::array<FE::Real, 3> direction{{0.0, 0.0, 0.0}};
            direction[static_cast<std::size_t>(component)] = FE::Real{1.0};
            const auto probe = constantVelocityProbe(direction);
            values[static_cast<std::size_t>(component)] =
                work(metric_sample.residual, probe) /
                generated.active_measure;
        }
        return GeneratedBoundaryMetricSample{
            .active_measure = generated.active_measure,
            .parent_measure = generated.parent_measure,
            .cell_diameter = values[0],
            .cell_measure = values[1],
            .facet_measure = values[2],
            .active_rule_count = generated.active_rule_count,
        };
    }

    [[nodiscard]] GeneratedBoundaryAssemblySample
    assembleGeneratedVertexOffset(FE::Real relative_offset)
    {
        if (!std::isfinite(relative_offset)) {
            throw std::invalid_argument(
                "generated sharp-boundary vertex offset must be finite");
        }
        return assembleGeneratedIntercept(relative_offset * length_scale_);
    }

    [[nodiscard]] GeneratedBoundaryAssemblySample
    assembleGeneratedPhysicalIntercept(FE::Real intercept)
    {
        auto generated = generatedContext(
            intercept,
            false,
            false);
        return GeneratedBoundaryAssemblySample{
            .assembly = assembleContext(generated.context),
            .active_measure = generated.active_measure,
            .parent_measure = generated.parent_measure,
            .active_rule_count = generated.active_rule_count,
        };
    }

    [[nodiscard]] std::vector<FE::Real> constantVelocityProbe(
        const std::array<FE::Real, 3>& value) const
    {
        std::vector<FE::Real> probe(solution_.size(), FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            for (int component = 0; component < 3; ++component) {
                setFieldComponentValue(
                    probe,
                    *system_,
                    velocity_,
                    vertex,
                    component,
                    value[static_cast<std::size_t>(component)]);
            }
        }
        return probe;
    }

    [[nodiscard]] std::vector<FE::Real> constantPressureProbe(
        FE::Real value) const
    {
        std::vector<FE::Real> probe(solution_.size(), FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            setFieldComponentValue(
                probe, *system_, pressure_, vertex, 0, value);
        }
        return probe;
    }

    [[nodiscard]] FE::Real parentBoundaryMeasure() const noexcept
    {
        return FE::Real{0.5} * length_scale_ * length_scale_;
    }

    [[nodiscard]] ActiveRuleObservation observeActiveRule(
        FE::Real active_measure) const
    {
        const auto local_context = context(active_measure);
        const auto rules =
            local_context->interfaceRulesForMarker(active_marker_);
        if (rules.size() != 1u || rules.front() == nullptr) {
            return {};
        }
        return ActiveRuleObservation{
            .valid = true,
            .owner_rank = rules.front()->provenance.owner_rank,
            .measure = rules.front()->measure,
        };
    }

    [[nodiscard]] std::span<const FE::Real> solution() const
    {
        return solution_;
    }

private:
    [[nodiscard]] AssemblySample assembleContext(
        const std::shared_ptr<FE::assembly::CutIntegrationContext>&
            assembly_context,
        const char* operation = "equations")
    {
        system_->setCutIntegrationContext(assembly_context);

        const auto dof_count = system_->dofHandler().getNumDofs();
        FE::assembly::DenseMatrixView jacobian(dof_count);
        FE::assembly::DenseVectorView residual(dof_count);
        jacobian.zero();
        residual.zero();

        FE::systems::SystemStateView state;
        state.dt = 1.0;
        state.u = std::span<const FE::Real>(solution_);
        state.u_prev = std::span<const FE::Real>(previous_solution_);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto time_context =
            integrator.buildContext(1, state);
        state.time_integration = &time_context;

        FE::systems::AssemblyRequest request;
        request.op = operation;
        request.want_matrix = true;
        request.want_vector = true;
        const auto result =
            system_->assemble(request, state, &jacobian, &residual);
        if (!result.success) {
            throw std::runtime_error(
                "sharp-boundary partition assembly failed: " +
                result.error_message);
        }

        AssemblySample sample;
        sample.residual.resize(static_cast<std::size_t>(dof_count));
        sample.jacobian.resize(
            static_cast<std::size_t>(dof_count * dof_count));
        for (FE::GlobalIndex row = 0; row < dof_count; ++row) {
            sample.residual[static_cast<std::size_t>(row)] = residual[row];
            for (FE::GlobalIndex column = 0; column < dof_count; ++column) {
                sample.jacobian[static_cast<std::size_t>(
                    row * dof_count + column)] =
                    jacobian.getMatrixEntry(row, column);
            }
        }
        return sample;
    }

    [[nodiscard]] GeneratedBoundaryAssemblySample
    assembleGeneratedIntercept(FE::Real intercept,
                               const char* operation = "equations")
    {
        auto generated = generatedContext(intercept);
        return GeneratedBoundaryAssemblySample{
            .assembly = assembleContext(generated.context, operation),
            .active_measure = generated.active_measure,
            .parent_measure = generated.parent_measure,
            .active_rule_count = generated.active_rule_count,
        };
    }

    [[nodiscard]] GeneratedBoundaryContext generatedContext(
        FE::Real intercept,
        bool full_boundary_with_interior_cut = false,
        bool orient_level_set_to_active_side = true) const
    {
        namespace interfaces = FE::interfaces;
        constexpr std::uint64_t source_revision = 17u;
        constexpr std::uint64_t quadrature_policy_key = 23u;

        const auto signed_value = [&](FE::GlobalIndex vertex) {
            const auto coordinates = mesh_->getNodeCoordinates(vertex);
            const FE::Real negative_side_value =
                full_boundary_with_interior_cut
                    ? coordinates[2] - intercept
                    : coordinates[0] - intercept;
            return !orient_level_set_to_active_side ||
                           active_side_ ==
                               FE::geometry::CutIntegrationSide::Negative
                       ? negative_side_value
                       : -negative_side_value;
        };

        interfaces::CutInterfaceDomainRequest interface_request;
        interface_request.source =
            interfaces::LevelSetInterfaceSource::fromField(
                level_set_, 0u, source_revision);
        interface_request.interface_marker = interface_marker_;
        interface_request.tolerance = FE::Real{1.0e-12};
        interface_request.quadrature_order = 2;
        interface_request.interface_quadrature_order = 1;
        interface_request.volume_quadrature_order = 2;
        interface_request.mesh_geometry_revision =
            mesh_->geometryRevision();
        interface_request.mesh_topology_revision =
            mesh_->topologyRevision();
        interface_request.ownership_revision =
            mesh_->ownershipRevision();
        interface_request.quadrature_policy_key = quadrature_policy_key;
        interface_request.implicit_geometry_mode = "LinearCorner";
        interface_request.implicit_quadrature_backend = "LinearCorner";
        interface_request.implicit_fallback_status = "None";

        interfaces::LevelSetCellCutInput input;
        input.parent_cell = 0;
        input.element_type = FE::ElementType::Tetra4;
        input.node_coordinates = {
            {{0.0, 0.0, 0.0}},
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}},
        };
        input.level_set_values.reserve(4u);
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            input.level_set_values.push_back(signed_value(vertex));
        }
        auto cut = interfaces::cutLinearLevelSetCell3D(
            interface_request, input);
        if (!cut.supported) {
            throw std::runtime_error(
                "generated sharp-boundary plane cut is unsupported: " +
                cut.diagnostic);
        }

        interfaces::LevelSetInterfaceDomain interface_domain(
            interface_request);
        std::uint64_t stable_id = 101u;
        for (auto& fragment : cut.fragments) {
            fragment.parent_cell_global_id = 1701;
            fragment.owner_rank = owner_rank_;
            fragment.stable_id = stable_id++;
            if (fragment.implicit_quadrature_backend.empty()) {
                fragment.implicit_quadrature_backend = "LinearCorner";
            }
            if (fragment.implicit_fallback_status.empty()) {
                fragment.implicit_fallback_status = "None";
            }
            interface_domain.addFragment(std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            region.parent_cell_global_id = 1701;
            region.owner_rank = owner_rank_;
            region.stable_id = stable_id++;
            if (region.implicit_quadrature_backend.empty()) {
                region.implicit_quadrature_backend = "LinearCorner";
            }
            if (region.implicit_fallback_status.empty()) {
                region.implicit_fallback_status = "None";
            }
            interface_domain.addVolumeRegion(std::move(region));
        }

        interfaces::GeneratedInterfaceBoundaryIntersectionRequest
            contact_request;
        contact_request.source = interface_request.source;
        contact_request.generated_domain_id = "free_surface";
        contact_request.interface_marker = interface_marker_;
        contact_request.boundary_marker = wall_marker_;
        contact_request.intersection_marker = contact_marker_;
        contact_request.tolerance = interface_request.tolerance;
        contact_request.quadrature_order = 2;
        contact_request.frame = interface_request.frame;
        contact_request.mesh_geometry_revision =
            interface_request.mesh_geometry_revision;
        contact_request.mesh_topology_revision =
            interface_request.mesh_topology_revision;
        contact_request.ownership_revision =
            interface_request.ownership_revision;
        contact_request.quadrature_policy_key =
            interface_request.quadrature_policy_key;
        contact_request.source_value_revision = source_revision;
        auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
                std::move(contact_request), interface_domain, *mesh_);

        interfaces::GeneratedActiveBoundaryRequest active_request;
        active_request.source = interface_request.source;
        active_request.generated_domain_id = "free_surface";
        active_request.interface_marker = interface_marker_;
        active_request.boundary_marker = wall_marker_;
        active_request.active_boundary_marker = active_marker_;
        active_request.side = active_side_;
        active_request.tolerance = interface_request.tolerance;
        active_request.quadrature_order = 2;
        active_request.frame = interface_request.frame;
        active_request.mesh_geometry_revision =
            interface_request.mesh_geometry_revision;
        active_request.mesh_topology_revision =
            interface_request.mesh_topology_revision;
        active_request.ownership_revision =
            interface_request.ownership_revision;
        active_request.quadrature_policy_key =
            interface_request.quadrature_policy_key;
        active_request.source_value_revision = source_revision;
        interfaces::GeneratedActiveBoundaryScalarField scalar_field;
        scalar_field.value_at_node = signed_value;
        auto active_domain = interfaces::buildGeneratedActiveBoundaryDomain(
            std::move(active_request),
            interface_domain,
            contact_domain,
            *mesh_,
            scalar_field);

        const auto summary = active_domain.summary();
        const auto active_rule_count =
            active_domain.boundaryQuadratureRules().size();
        std::optional<interfaces::LevelSetInterfaceDomain> pressure_support;
        if (interface_domain.empty()) {
            // Immediately before a boundary-vertex crossing, the local plane
            // has left this one-cell wall patch even though a global free
            // surface can remain elsewhere.  Retain a separate interior cut
            // for the pressure anchor; the exterior rule above still comes
            // exclusively from the manufactured crossing field.
            interfaces::LevelSetCellCutInput support_input = input;
            support_input.level_set_values.clear();
            support_input.level_set_values.reserve(4u);
            for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
                const FE::Real negative_side_value =
                    mesh_->getNodeCoordinates(vertex)[2] -
                    FE::Real{0.5} * length_scale_;
                support_input.level_set_values.push_back(
                    active_side_ ==
                            FE::geometry::CutIntegrationSide::Negative
                        ? negative_side_value
                        : -negative_side_value);
            }
            auto support_cut = interfaces::cutLinearLevelSetCell3D(
                interface_request, support_input);
            if (!support_cut.supported || support_cut.fragments.empty()) {
                throw std::runtime_error(
                    "generated dry-wall pressure support cut is unavailable");
            }
            pressure_support.emplace(interface_request);
            std::uint64_t support_stable_id = 501u;
            for (auto& fragment : support_cut.fragments) {
                fragment.parent_cell_global_id = 1701;
                fragment.owner_rank = owner_rank_;
                fragment.stable_id = support_stable_id++;
                pressure_support->addFragment(std::move(fragment));
            }
            for (auto& region : support_cut.volume_regions) {
                region.parent_cell_global_id = 1701;
                region.owner_rank = owner_rank_;
                region.stable_id = support_stable_id++;
                pressure_support->addVolumeRegion(std::move(region));
            }
        }
        auto generated_context =
            std::make_shared<FE::assembly::CutIntegrationContext>();
        generated_context->addGeneratedInterfaceDomain(
            pressure_support.has_value() ? *pressure_support
                                         : interface_domain,
            active_side_);
        generated_context->addGeneratedInterfaceBoundaryIntersectionDomain(
            contact_domain);
        generated_context->addGeneratedActiveBoundaryDomain(active_domain);
        return GeneratedBoundaryContext{
            .context = std::move(generated_context),
            .active_measure = summary.measure * length_scale_ * length_scale_,
            .parent_measure = summary.parent_measure *
                              length_scale_ * length_scale_,
            .active_rule_count = active_rule_count,
        };
    }

    [[nodiscard]] std::shared_ptr<FE::assembly::CutIntegrationContext> context(
        std::optional<FE::Real> active_measure) const
    {
        return makeFreeSurfaceContext(
            interface_marker_,
            wall_marker_,
            contact_marker_,
            level_set_,
            active_side_,
            owner_rank_,
            active_measure);
    }

    static constexpr int interface_marker_{231};
    static constexpr int wall_marker_{232};
    OperatorFamily family_;
    FE::geometry::CutIntegrationSide active_side_;
    int owner_rank_{0};
    FE::Real operator_scale_{1.0};
    FE::Real length_scale_{1.0};
    FE::Real nitsche_gamma_{12.0};
    std::shared_ptr<PartitionedSingleTetraBoundaryMesh> mesh_{};
    std::unique_ptr<FE::systems::FESystem> system_{};
    FE::FieldId level_set_{FE::INVALID_FIELD_ID};
    FE::FieldId velocity_{FE::INVALID_FIELD_ID};
    FE::FieldId pressure_{FE::INVALID_FIELD_ID};
    int active_marker_{-1};
    int contact_marker_{-1};
    std::vector<FE::Real> solution_{};
    std::vector<FE::Real> previous_solution_{};
};

std::vector<FE::Real> difference(std::span<const FE::Real> wet,
                                 std::span<const FE::Real> dry)
{
    if (wet.size() != dry.size()) {
        throw std::invalid_argument(
            "sharp-boundary sample sizes do not match");
    }
    std::vector<FE::Real> result(wet.size(), FE::Real{0.0});
    for (std::size_t i = 0; i < wet.size(); ++i) {
        result[i] = wet[i] - dry[i];
    }
    return result;
}

FE::Real maximumAbsoluteDifference(std::span<const FE::Real> first,
                                   std::span<const FE::Real> second)
{
    if (first.size() != second.size()) {
        throw std::invalid_argument(
            "sharp-boundary comparison sizes do not match");
    }
    FE::Real maximum{0.0};
    for (std::size_t i = 0; i < first.size(); ++i) {
        maximum = std::max(maximum, std::abs(first[i] - second[i]));
    }
    return maximum;
}

FE::Real maximumScalingError(std::span<const FE::Real> sample,
                             std::span<const FE::Real> full,
                             FE::Real fraction)
{
    if (sample.size() != full.size()) {
        throw std::invalid_argument(
            "sharp-boundary scaling sizes do not match");
    }
    FE::Real maximum{0.0};
    for (std::size_t i = 0; i < sample.size(); ++i) {
        maximum = std::max(
            maximum, std::abs(sample[i] - fraction * full[i]));
    }
    return maximum;
}

FE::Real vectorNorm(std::span<const FE::Real> values)
{
    FE::Real squared{0.0};
    for (const auto value : values) {
        squared += value * value;
    }
    return std::sqrt(squared);
}

FE::Real work(std::span<const FE::Real> residual,
              std::span<const FE::Real> solution)
{
    if (residual.size() != solution.size()) {
        throw std::invalid_argument(
            "sharp-boundary work vectors do not match");
    }
    FE::Real value{0.0};
    for (std::size_t i = 0; i < residual.size(); ++i) {
        value += residual[i] * solution[i];
    }
    return value;
}

FE::Real quadraticWork(std::span<const FE::Real> matrix,
                       std::span<const FE::Real> probe)
{
    const std::size_t dimension = probe.size();
    if (matrix.size() != dimension * dimension) {
        throw std::invalid_argument(
            "sharp-boundary quadratic work dimensions do not match");
    }
    FE::Real value{0.0};
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column < dimension; ++column) {
            value += probe[row] * matrix[row * dimension + column] *
                     probe[column];
        }
    }
    return value;
}

FE::Real scaledAbsoluteError(FE::Real actual,
                             FE::Real expected,
                             FE::Real scale)
{
    if (!std::isfinite(scale) || !(std::abs(scale) > FE::Real{0.0})) {
        throw std::invalid_argument(
            "sharp-boundary comparison scale must be finite and nonzero");
    }
    return std::abs(actual - expected) / std::abs(scale);
}

void recordRealProperty(const char* name, FE::Real value)
{
    std::ostringstream text;
    text << std::scientific
         << std::setprecision(std::numeric_limits<FE::Real>::max_digits10)
         << value;
    ::testing::Test::RecordProperty(name, text.str());
}

TEST(FreeSurfaceSharpBoundaryOperators,
     WetFractionSweepMatchesAnalyticOperatorWork)
{
    FE::Real maximum_force_error{0.0};
    FE::Real maximum_flux_error{0.0};
    FE::Real maximum_robin_error{0.0};
    FE::Real maximum_penalty_error{0.0};
    for (const auto family : kOperatorFamilies) {
        for (const auto side : kActiveSides) {
            SharpBoundaryAssemblyHarness harness(
                family, side, 0, 1, 0, true);
            const auto dry = harness.assemble(FE::Real{0.0});
            const auto full =
                harness.assemble(kParentBoundaryMeasure);
            const auto full_residual =
                difference(full.residual, dry.residual);
            const auto full_jacobian =
                difference(full.jacobian, dry.jacobian);
            ASSERT_GT(
                std::max(vectorNorm(full_residual),
                         vectorNorm(full_jacobian)),
                FE::Real{1.0e-12});

            for (const FE::Real fraction : kWetFractions) {
                const auto sample = harness.assemble(
                    kParentBoundaryMeasure * fraction);
                const FE::Real residual_error =
                    maximumScalingError(
                        difference(sample.residual, dry.residual),
                        full_residual,
                        fraction);
                const FE::Real jacobian_error =
                    maximumScalingError(
                        difference(sample.jacobian, dry.jacobian),
                        full_jacobian,
                        fraction);
                const FE::Real family_error =
                    std::max(residual_error, jacobian_error);
                switch (family) {
                case OperatorFamily::Traction:
                    maximum_force_error =
                        std::max(maximum_force_error, family_error);
                    break;
                case OperatorFamily::Robin:
                    maximum_robin_error =
                        std::max(maximum_robin_error, family_error);
                    break;
                case OperatorFamily::PressureFlux:
                case OperatorFamily::Outflow:
                    maximum_flux_error =
                        std::max(maximum_flux_error, family_error);
                    break;
                case OperatorFamily::SymmetricNitsche:
                case OperatorFamily::UnsymmetricNitsche:
                case OperatorFamily::WallSlip:
                    maximum_penalty_error =
                        std::max(maximum_penalty_error, family_error);
                    break;
                }
            }
        }
    }

    EXPECT_LE(maximum_force_error, FE::Real{1.0e-10});
    EXPECT_LE(maximum_flux_error, FE::Real{1.0e-10});
    EXPECT_LE(maximum_robin_error, FE::Real{1.0e-10});
    EXPECT_LE(maximum_penalty_error, FE::Real{1.0e-10});
    ::testing::Test::RecordProperty(
        "sharp_operator_fraction_case_count",
        static_cast<int>(kWetFractions.size()));
    ::testing::Test::RecordProperty(
        "sharp_operator_family_count",
        static_cast<int>(kOperatorFamilies.size()));
    ::testing::Test::RecordProperty(
        "sharp_operator_active_side_case_count",
        static_cast<int>(kActiveSides.size()));
    recordRealProperty(
        "sharp_operator_maximum_scaled_force_error",
        maximum_force_error);
    recordRealProperty(
        "sharp_operator_maximum_scaled_flux_error",
        maximum_flux_error);
    recordRealProperty(
        "sharp_operator_maximum_scaled_robin_error",
        maximum_robin_error);
    recordRealProperty(
        "sharp_operator_maximum_scaled_penalty_error",
        maximum_penalty_error);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     PspgBoundaryPressureFluxUsesGeneratedWetWallMeasure)
{
    SharpBoundaryAssemblyHarness enabled(
        OperatorFamily::PressureFlux,
        FE::geometry::CutIntegrationSide::Negative,
        0,
        1,
        0,
        true,
        FE::Real{1.0});
    SharpBoundaryAssemblyHarness disabled(
        OperatorFamily::PressureFlux,
        FE::geometry::CutIntegrationSide::Negative,
        0,
        1,
        0,
        true,
        FE::Real{0.0});
    const auto pressure_probe =
        enabled.constantPressureProbe(FE::Real{1.0});
    const auto enabled_full =
        enabled.assembleGeneratedFraction(FE::Real{1.0});
    const auto disabled_full =
        disabled.assembleGeneratedFraction(FE::Real{1.0});
    const FE::Real full_work = work(
        difference(enabled_full.assembly.residual,
                   disabled_full.assembly.residual),
        pressure_probe);
    ASSERT_GT(std::abs(full_work), FE::Real{1.0e-12});

    FE::Real maximum_work_error{0.0};
    FE::Real maximum_measure_error{0.0};
    for (const FE::Real fraction : kWetFractions) {
        const auto enabled_sample =
            enabled.assembleGeneratedFraction(fraction);
        const auto disabled_sample =
            disabled.assembleGeneratedFraction(fraction);
        const FE::Real sample_work = work(
            difference(enabled_sample.assembly.residual,
                       disabled_sample.assembly.residual),
            pressure_probe);
        maximum_work_error = std::max(
            maximum_work_error,
            scaledAbsoluteError(
                sample_work, fraction * full_work, full_work));
        maximum_measure_error = std::max(
            maximum_measure_error,
            std::abs(
                enabled_sample.active_measure -
                fraction * enabled_sample.parent_measure));
    }

    EXPECT_LE(maximum_work_error, FE::Real{1.0e-10});
    EXPECT_LE(maximum_measure_error, FE::Real{2.0e-12});
    ::testing::Test::RecordProperty(
        "sharp_pspg_pressure_flux_fraction_case_count",
        static_cast<int>(kWetFractions.size()));
    recordRealProperty(
        "sharp_pspg_pressure_flux_maximum_scaled_work_error",
        maximum_work_error);
    recordRealProperty(
        "sharp_pspg_pressure_flux_maximum_wet_measure_error",
        maximum_measure_error);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     ActiveSideReversalUsesComplementarySharpSubset)
{
    SharpBoundaryAssemblyHarness negative(
        OperatorFamily::Traction,
        FE::geometry::CutIntegrationSide::Negative,
        0,
        1,
        0,
        true);
    SharpBoundaryAssemblyHarness positive(
        OperatorFamily::Traction,
        FE::geometry::CutIntegrationSide::Positive,
        0,
        1,
        0,
        true);
    SharpBoundaryAssemblyHarness negative_zero(
        OperatorFamily::Traction,
        FE::geometry::CutIntegrationSide::Negative,
        0,
        1,
        0,
        true,
        FE::Real{0.0});
    SharpBoundaryAssemblyHarness positive_zero(
        OperatorFamily::Traction,
        FE::geometry::CutIntegrationSide::Positive,
        0,
        1,
        0,
        true,
        FE::Real{0.0});
    const auto negative_full =
        negative.assembleGeneratedFraction(FE::Real{1.0});
    const auto negative_zero_full =
        negative_zero.assembleGeneratedFraction(FE::Real{1.0});
    const auto negative_full_delta = difference(
        negative_full.assembly.residual,
        negative_zero_full.assembly.residual);

    FE::Real maximum_complement_error{0.0};
    int marker_mismatch_count = 0;
    constexpr std::array<FE::Real, 4> fractions{{
        FE::Real{0.01},
        FE::Real{0.25},
        FE::Real{0.49},
        FE::Real{0.8},
    }};
    for (const FE::Real fraction : fractions) {
        const FE::Real intercept =
            FE::Real{1.0} -
            std::sqrt(FE::Real{1.0} - fraction);
        const auto negative_sample =
            negative.assembleGeneratedPhysicalIntercept(intercept);
        const auto positive_sample =
            positive.assembleGeneratedPhysicalIntercept(intercept);
        const auto negative_zero_sample =
            negative_zero.assembleGeneratedPhysicalIntercept(intercept);
        const auto positive_zero_sample =
            positive_zero.assembleGeneratedPhysicalIntercept(intercept);
        const auto negative_delta = difference(
            negative_sample.assembly.residual,
            negative_zero_sample.assembly.residual);
        const auto positive_delta = difference(
            positive_sample.assembly.residual,
            positive_zero_sample.assembly.residual);
        std::vector<FE::Real> partition_sum = negative_delta;
        for (std::size_t i = 0u; i < partition_sum.size(); ++i) {
            partition_sum[i] += positive_delta[i];
        }
        maximum_complement_error = std::max(
            maximum_complement_error,
            maximumAbsoluteDifference(
                partition_sum, negative_full_delta));
        EXPECT_NEAR(
            negative_sample.active_measure +
                positive_sample.active_measure,
            kParentBoundaryMeasure,
            FE::Real{2.0e-12});
        if (negative_sample.active_rule_count == 0u ||
            positive_sample.active_rule_count == 0u) {
            ++marker_mismatch_count;
        }
    }

    EXPECT_LE(maximum_complement_error, FE::Real{1.0e-12});
    EXPECT_EQ(marker_mismatch_count, 0);
    ::testing::Test::RecordProperty(
        "sharp_active_side_case_count",
        static_cast<int>(kActiveSides.size()));
    recordRealProperty(
        "sharp_active_side_maximum_complement_error",
        maximum_complement_error);
    ::testing::Test::RecordProperty(
        "sharp_active_side_marker_mismatch_count",
        marker_mismatch_count);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     CompletelyDryBoundaryProducesExactlyZeroWetRows)
{
    FE::Real maximum_repeatability_error{0.0};
    FE::Real maximum_dry_entry_magnitude{0.0};
    int dry_rule_count = 0;
    int explicit_empty_marker_registration_count = 0;
    int whole_face_contribution_count = 0;
    for (const auto family : kOperatorFamilies) {
        SharpBoundaryAssemblyHarness harness(
            family,
            FE::geometry::CutIntegrationSide::Negative,
            0,
            1,
            0,
            true);
        const auto empty_domains =
            harness.observeExplicitlyEmptyGeneratedDomains();
        EXPECT_TRUE(empty_domains.active_marker_registered);
        EXPECT_TRUE(empty_domains.contact_marker_registered);
        EXPECT_EQ(empty_domains.active_rule_count, 0u);
        EXPECT_EQ(empty_domains.contact_rule_count, 0u);
        if (empty_domains.active_marker_registered &&
            empty_domains.contact_marker_registered) {
            ++explicit_empty_marker_registration_count;
        }
        EXPECT_NO_THROW(
            (void)harness.assembleGeneratedFraction(FE::Real{0.0}));
        const auto first =
            harness.assembleGeneratedFraction(
                FE::Real{0.0}, "sharp_entity_measure_probe");
        const auto second =
            harness.assembleGeneratedFraction(
                FE::Real{0.0}, "sharp_entity_measure_probe");
        const auto empty =
            harness.assembleGeneratedFraction(
                FE::Real{0.0}, "sharp_empty_probe");
        dry_rule_count += static_cast<int>(first.active_rule_count);
        maximum_repeatability_error = std::max(
            maximum_repeatability_error,
            std::max(
                maximumAbsoluteDifference(
                    first.assembly.residual,
                    second.assembly.residual),
                maximumAbsoluteDifference(
                    first.assembly.jacobian,
                    second.assembly.jacobian)));
        for (const auto value : difference(
                 first.assembly.residual, empty.assembly.residual)) {
            maximum_dry_entry_magnitude = std::max(
                maximum_dry_entry_magnitude, std::abs(value));
        }
        for (const auto value : difference(
                 first.assembly.jacobian, empty.assembly.jacobian)) {
            maximum_dry_entry_magnitude = std::max(
                maximum_dry_entry_magnitude, std::abs(value));
        }
        if (first.active_measure != FE::Real{0.0} ||
            first.active_rule_count != 0u) {
            ++whole_face_contribution_count;
        }
    }

    EXPECT_EQ(dry_rule_count, 0);
    EXPECT_EQ(maximum_repeatability_error, FE::Real{0.0});
    EXPECT_EQ(maximum_dry_entry_magnitude, FE::Real{0.0});
    EXPECT_EQ(
        explicit_empty_marker_registration_count,
        static_cast<int>(kOperatorFamilies.size()));
    EXPECT_EQ(whole_face_contribution_count, 0);
    ::testing::Test::RecordProperty(
        "sharp_dry_operator_family_count",
        static_cast<int>(kOperatorFamilies.size()));
    ::testing::Test::RecordProperty(
        "sharp_dry_generated_rule_count",
        dry_rule_count);
    recordRealProperty(
        "sharp_dry_repeatability_maximum_error",
        maximum_repeatability_error);
    recordRealProperty(
        "sharp_dry_maximum_entry_magnitude",
        maximum_dry_entry_magnitude);
    ::testing::Test::RecordProperty(
        "sharp_dry_explicit_empty_marker_registration_count",
        explicit_empty_marker_registration_count);
    ::testing::Test::RecordProperty(
        "sharp_dry_whole_face_contribution_count",
        whole_face_contribution_count);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     MissingGeneratedActiveDomainFailsClosed)
{
    int rejection_count = 0;
    for (const auto family : kOperatorFamilies) {
        SharpBoundaryAssemblyHarness harness(
            family,
            FE::geometry::CutIntegrationSide::Negative,
            0,
            1,
            0,
            true);
        try {
            (void)harness.assemble(std::nullopt);
        } catch (const std::exception& error) {
            if (std::string(error.what()).find(
                    "missing InterfaceMesh or generated cut-interface rules") !=
                std::string::npos) {
                ++rejection_count;
                continue;
            }
            throw;
        }
        ADD_FAILURE()
            << "missing generated active domain did not fail for family "
            << static_cast<int>(family);
    }
    EXPECT_EQ(
        rejection_count,
        static_cast<int>(kOperatorFamilies.size()));
    ::testing::Test::RecordProperty(
        "sharp_missing_domain_rejection_count",
        rejection_count);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     HigherOrderSpacesAndImplicitGeometryFailClosed)
{
    const auto expect_rejection =
        [](int velocity_order,
           int pressure_order,
           std::string generated_geometry,
           std::string_view expected_message) {
            auto mesh =
                std::make_shared<PartitionedSingleTetraBoundaryMesh>(
                    232, 0, 1, 0);
            auto velocity_space = FE::spaces::VectorSpace(
                FE::spaces::SpaceType::H1,
                mesh,
                velocity_order,
                3);
            auto pressure_space = FE::spaces::Space(
                FE::spaces::SpaceType::H1,
                mesh,
                pressure_order,
                1);
            auto level_set_space = FE::spaces::Space(
                FE::spaces::SpaceType::H1, mesh, 1, 1);
            FE::systems::FESystem system(mesh);
            (void)system.addField(FE::systems::FieldSpec{
                .name = "phi_high_order_guard",
                .space = level_set_space,
                .components = 1,
                .source_kind =
                    FE::systems::FieldSourceKind::PrescribedData,
            });
            auto options = baseOptions();
            options.traction_neumann.push_back(
                ns::IncompressibleNavierStokesVMSOptions::
                    TractionNeumannBC{
                        .boundary_marker = 232,
                        .traction = {1.0, 0.0, 0.0},
                    });
            options.free_surface.push_back(
                ns::IncompressibleNavierStokesVMSOptions::
                    FreeSurfaceBoundary{
                        .implementation =
                            ns::FreeSurfaceImplementation::
                                UnfittedLevelSet,
                        .interface_marker = 231,
                        .level_set_field_name =
                            "phi_high_order_guard",
                        .generated_interface_domain_id =
                            "free_surface",
                        .generated_interface_geometry =
                            std::move(generated_geometry),
                        .active_domain =
                            ns::FreeSurfaceActiveDomain::
                                LevelSetNegative,
                        .active_domain_method =
                            ns::FreeSurfaceActiveDomainMethod::
                                CutVolume,
                        .surface_tension = FE::Real{0.0},
                        .small_cut_aggregation = false,
                    });
            ns::IncompressibleNavierStokesVMSModule module(
                velocity_space,
                pressure_space,
                std::move(options));
            try {
                module.registerOn(system);
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(
                    std::string(error.what()).find(expected_message),
                    std::string::npos);
                return;
            }
            FAIL() << "sharp high-order envelope did not fail closed";
        };

    expect_rejection(2, 1, "LinearCorner", "order-1 velocity and pressure");
    expect_rejection(1, 2, "LinearCorner", "order-1 velocity and pressure");
    expect_rejection(
        1,
        1,
        "HighOrderImplicit",
        "Generated_interface_geometry=LinearCorner");
    ::testing::Test::RecordProperty(
        "sharp_high_order_envelope_rejection_count", 3);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     StructuredManufacturedChannelTracksDistinctExteriorRoles)
{
    constexpr std::array<FE::Real, 9> interface_heights{{
        FE::Real{0.05},
        FE::Real{0.125},
        FE::Real{0.25},
        FE::Real{0.375},
        FE::Real{0.505},
        FE::Real{0.625},
        FE::Real{0.75},
        FE::Real{0.875},
        FE::Real{0.95},
    }};
    constexpr FE::Real reference_height = FE::Real{0.75};
    constexpr std::array<FE::Real, 3> probe_value{{
        FE::Real{0.7}, FE::Real{-0.4}, FE::Real{0.6}}};

    const StructuredChannelOperatorScales baseline_scales{
        .inlet_traction = FE::Real{0.0},
        .outlet_pressure = FE::Real{0.0},
        .sidewall_nitsche_gamma = FE::Real{6.0},
    };
    auto force_scales = baseline_scales;
    force_scales.inlet_traction = FE::Real{1.0};
    auto flux_scales = baseline_scales;
    flux_scales.outlet_pressure = FE::Real{1.0};
    auto penalty_scales = baseline_scales;
    penalty_scales.sidewall_nitsche_gamma = FE::Real{12.0};

    StructuredChannelAssemblyHarness baseline(
        2, 0, 1, ChannelPartition::Serial, baseline_scales, true);
    StructuredChannelAssemblyHarness force(
        2, 0, 1, ChannelPartition::Serial, force_scales, true);
    StructuredChannelAssemblyHarness flux(
        2, 0, 1, ChannelPartition::Serial, flux_scales, true);
    StructuredChannelAssemblyHarness penalty(
        2, 0, 1, ChannelPartition::Serial, penalty_scales, true);
    const auto probe = baseline.constantVelocityProbe(probe_value);

    const auto baseline_reference = baseline.assemble(reference_height);
    const auto force_reference = force.assemble(reference_height);
    const auto flux_reference = flux.assemble(reference_height);
    const auto penalty_reference = penalty.assemble(reference_height);
    const FE::Real reference_force_work = work(
        difference(force_reference.residual,
                   baseline_reference.residual),
        probe);
    const FE::Real reference_flux_work = work(
        difference(flux_reference.residual,
                   baseline_reference.residual),
        probe);
    const FE::Real reference_penalty_work = work(
        difference(penalty_reference.residual,
                   baseline_reference.residual),
        probe);
    ASSERT_GT(std::abs(reference_force_work), FE::Real{1.0e-12});
    ASSERT_GT(std::abs(reference_flux_work), FE::Real{1.0e-12});
    ASSERT_GT(std::abs(reference_penalty_work), FE::Real{1.0e-12});

    FE::Real maximum_force_error{0.0};
    FE::Real maximum_flux_error{0.0};
    FE::Real maximum_penalty_error{0.0};
    FE::Real maximum_measure_error{0.0};
    for (const FE::Real height : interface_heights) {
        const auto baseline_sample = baseline.assemble(height);
        const auto force_sample = force.assemble(height);
        const auto flux_sample = flux.assemble(height);
        const auto penalty_sample = penalty.assemble(height);

        constexpr std::array<FE::Real, 3> parent_measures{{
            FE::Real{1.0}, FE::Real{1.0}, FE::Real{2.0}}};
        for (std::size_t role = 0u; role < 3u; ++role) {
            const FE::Real expected_measure =
                parent_measures[role] * height;
            maximum_measure_error = std::max(
                maximum_measure_error,
                std::abs(
                    force_sample.geometry.active_measures[role] -
                    expected_measure));
            EXPECT_NEAR(
                force_sample.geometry.parent_measures[role],
                parent_measures[role],
                FE::Real{2.0e-12});
            EXPECT_GT(
                force_sample.geometry.active_rule_counts[role], 0u);
        }
        const FE::Real force_work = work(
            difference(force_sample.residual,
                       baseline_sample.residual),
            probe);
        const FE::Real flux_work = work(
            difference(flux_sample.residual,
                       baseline_sample.residual),
            probe);
        const FE::Real penalty_work = work(
            difference(penalty_sample.residual,
                       baseline_sample.residual),
            probe);
        const FE::Real ratio = height / reference_height;
        maximum_force_error = std::max(
            maximum_force_error,
            scaledAbsoluteError(force_work,
                                ratio * reference_force_work,
                                reference_force_work));
        maximum_flux_error = std::max(
            maximum_flux_error,
            scaledAbsoluteError(flux_work,
                                ratio * reference_flux_work,
                                reference_flux_work));
        maximum_penalty_error = std::max(
            maximum_penalty_error,
            scaledAbsoluteError(penalty_work,
                                ratio * reference_penalty_work,
                                reference_penalty_work));
    }

    EXPECT_LE(maximum_measure_error, FE::Real{2.0e-11});
    EXPECT_LE(maximum_force_error, FE::Real{1.0e-9});
    EXPECT_LE(maximum_flux_error, FE::Real{1.0e-9});
    EXPECT_LE(maximum_penalty_error, FE::Real{1.0e-9});
    ::testing::Test::RecordProperty("sharp_channel_boundary_role_count", 3);
    ::testing::Test::RecordProperty(
        "sharp_channel_structured_cell_count",
        static_cast<int>(baseline.cellCount()));
    ::testing::Test::RecordProperty(
        "sharp_channel_crossing_state_count",
        static_cast<int>(interface_heights.size()));
    recordRealProperty(
        "sharp_channel_maximum_wet_measure_error",
        maximum_measure_error);
    recordRealProperty(
        "sharp_channel_maximum_scaled_force_error",
        maximum_force_error);
    recordRealProperty(
        "sharp_channel_maximum_scaled_flux_error",
        maximum_flux_error);
    recordRealProperty(
        "sharp_channel_maximum_scaled_penalty_work_error",
        maximum_penalty_error);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     StructuredVertexCrossingGlobalWorkJumpConvergesUnderRefinement)
{
    constexpr std::array<int, 4> cells_per_unit_length{{
        2,
        4,
        6,
        8,
    }};
    constexpr FE::Real crossing_offset = FE::Real{0.05};
    constexpr std::array<FE::Real, 3> probe_value{{
        FE::Real{1.25}, FE::Real{-0.5}, FE::Real{0.75}}};
    const StructuredChannelOperatorScales baseline_scales{
        .inlet_traction = FE::Real{0.0},
        .outlet_pressure = FE::Real{0.0},
        .sidewall_nitsche_gamma = FE::Real{6.0},
    };
    auto traction_scales = baseline_scales;
    traction_scales.inlet_traction = FE::Real{1.0};

    std::array<FE::Real, cells_per_unit_length.size()> jumps{};
    int finest_cell_count = 0;
    for (std::size_t level = 0u;
         level < cells_per_unit_length.size();
         ++level) {
        const int resolution = cells_per_unit_length[level];
        StructuredChannelAssemblyHarness baseline(
            resolution,
            0,
            1,
            ChannelPartition::Serial,
            baseline_scales,
            true);
        StructuredChannelAssemblyHarness traction(
            resolution,
            0,
            1,
            ChannelPartition::Serial,
            traction_scales,
            true);
        const auto probe = baseline.constantVelocityProbe(probe_value);
        const FE::Real mesh_size =
            FE::Real{1.0} / static_cast<FE::Real>(resolution);
        const FE::Real left_height =
            FE::Real{0.5} - crossing_offset * mesh_size;
        const FE::Real right_height =
            FE::Real{0.5} + crossing_offset * mesh_size;
        const auto baseline_left = baseline.assemble(left_height);
        const auto traction_left = traction.assemble(left_height);
        const auto baseline_right = baseline.assemble(right_height);
        const auto traction_right = traction.assemble(right_height);
        const auto left_work = difference(
            traction_left.residual, baseline_left.residual);
        const auto right_work = difference(
            traction_right.residual, baseline_right.residual);
        jumps[level] = std::abs(
            work(difference(right_work, left_work), probe));
        EXPECT_TRUE(std::isfinite(jumps[level]));
        EXPECT_GT(jumps[level], FE::Real{0.0});
        EXPECT_NEAR(
            traction_left.geometry.active_measures[0],
            left_height,
            FE::Real{2.0e-11});
        EXPECT_NEAR(
            traction_right.geometry.active_measures[0],
            right_height,
            FE::Real{2.0e-11});
        finest_cell_count = static_cast<int>(traction.cellCount());
    }

    int strict_decrease_count = 0;
    for (std::size_t level = 1u; level < jumps.size(); ++level) {
        if (jumps[level] < jumps[level - 1u]) {
            ++strict_decrease_count;
        }
        EXPECT_LT(jumps[level], jumps[level - 1u]);
    }
    const FE::Real fine_to_coarse_ratio = jumps.back() / jumps.front();
    EXPECT_LT(fine_to_coarse_ratio, FE::Real{1.0});

    ::testing::Test::RecordProperty(
        "sharp_vertex_crossing_refinement_level_count",
        static_cast<int>(cells_per_unit_length.size()));
    ::testing::Test::RecordProperty(
        "sharp_vertex_crossing_strict_decrease_count",
        strict_decrease_count);
    ::testing::Test::RecordProperty(
        "sharp_vertex_crossing_finest_structured_cell_count",
        finest_cell_count);
    recordRealProperty(
        "sharp_vertex_crossing_fine_to_coarse_jump_ratio",
        fine_to_coarse_ratio);
}

TEST(FreeSurfaceSharpBoundaryOperators,
     NitscheTraceScalingProducesFiniteSampledMargins)
{
    constexpr std::array<OperatorFamily, 2> nitsche_policies{{
        OperatorFamily::SymmetricNitsche,
        OperatorFamily::UnsymmetricNitsche,
    }};
    constexpr FE::Real high_gamma = FE::Real{12.0};
    constexpr FE::Real low_gamma = FE::Real{6.0};
    constexpr FE::Real viscosity = FE::Real{0.01};
    constexpr std::array<FE::Real, 3> probe_value{{
        FE::Real{0.3}, FE::Real{-0.15}, FE::Real{0.2}}};
    constexpr FE::Real probe_squared =
        probe_value[0] * probe_value[0] +
        probe_value[1] * probe_value[1] +
        probe_value[2] * probe_value[2];
    const FE::Real expected_diameter = std::sqrt(FE::Real{2.0});
    constexpr FE::Real expected_cell_measure = FE::Real{1.0} / FE::Real{6.0};
    constexpr FE::Real expected_facet_measure = FE::Real{0.5};
    constexpr FE::Real expected_h_normal = FE::Real{2.0} / FE::Real{3.0};
    const FE::Real full_penalty_scale =
        (high_gamma - low_gamma) * viscosity /
        expected_h_normal * expected_facet_measure * probe_squared;

    int finite_positive_count = 0;
    int nonpositive_count = 0;
    int zero_area_rejection_count = 0;
    FE::Real minimum_margin = std::numeric_limits<FE::Real>::infinity();
    FE::Real maximum_consistency_error{0.0};
    FE::Real maximum_diameter_error{0.0};
    FE::Real maximum_cell_measure_error{0.0};
    FE::Real maximum_facet_measure_error{0.0};
    FE::Real maximum_h_normal_error{0.0};

    for (const auto policy : nitsche_policies) {
        std::array<std::unique_ptr<SharpBoundaryAssemblyHarness>, 2>
            high_harnesses;
        std::array<std::unique_ptr<SharpBoundaryAssemblyHarness>, 2>
            low_harnesses;
        for (std::size_t side_index = 0u;
             side_index < kActiveSides.size();
             ++side_index) {
            high_harnesses[side_index] =
                std::make_unique<SharpBoundaryAssemblyHarness>(
                    policy,
                    kActiveSides[side_index],
                    0,
                    1,
                    0,
                    true,
                    FE::Real{1.0},
                    FE::Real{1.0},
                    high_gamma);
            low_harnesses[side_index] =
                std::make_unique<SharpBoundaryAssemblyHarness>(
                    policy,
                    kActiveSides[side_index],
                    0,
                    1,
                    0,
                    true,
                    FE::Real{1.0},
                    FE::Real{1.0},
                    low_gamma);
        }

        for (const FE::Real fraction : kWetFractions) {
            bool both_sides_positive = true;
            for (std::size_t side_index = 0u;
                 side_index < kActiveSides.size();
                 ++side_index) {
                auto& high_harness = *high_harnesses[side_index];
                auto& low_harness = *low_harnesses[side_index];
                const auto high_sample =
                    high_harness.assembleGeneratedFraction(fraction);
                const auto low_sample =
                    low_harness.assembleGeneratedFraction(fraction);
                const auto metric_sample =
                    high_harness.observeGeneratedMetrics(fraction);
                const auto probe =
                    high_harness.constantVelocityProbe(probe_value);
                const FE::Real margin = quadraticWork(
                    difference(high_sample.assembly.jacobian,
                               low_sample.assembly.jacobian),
                    probe);
                const FE::Real expected_margin =
                    (high_gamma - low_gamma) * viscosity /
                    expected_h_normal *
                    (expected_facet_measure * fraction) * probe_squared;

                EXPECT_EQ(high_sample.active_rule_count, 1u);
                EXPECT_EQ(metric_sample.active_rule_count, 1u);
                EXPECT_NEAR(metric_sample.active_measure,
                            expected_facet_measure * fraction,
                            FE::Real{2.0e-12});
                maximum_diameter_error = std::max(
                    maximum_diameter_error,
                    std::abs(metric_sample.cell_diameter -
                             expected_diameter));
                maximum_cell_measure_error = std::max(
                    maximum_cell_measure_error,
                    std::abs(metric_sample.cell_measure -
                             expected_cell_measure));
                maximum_facet_measure_error = std::max(
                    maximum_facet_measure_error,
                    std::abs(metric_sample.facet_measure -
                             expected_facet_measure));
                const FE::Real observed_h_normal =
                    FE::Real{2.0} * metric_sample.cell_measure /
                    metric_sample.facet_measure;
                maximum_h_normal_error = std::max(
                    maximum_h_normal_error,
                    std::abs(observed_h_normal - expected_h_normal));
                maximum_consistency_error = std::max(
                    maximum_consistency_error,
                    scaledAbsoluteError(margin,
                                        expected_margin,
                                        full_penalty_scale));
                minimum_margin = std::min(minimum_margin, margin);
                if (!std::isfinite(margin) ||
                    !(margin > FE::Real{0.0})) {
                    both_sides_positive = false;
                }
            }
            if (both_sides_positive) {
                ++finite_positive_count;
            } else {
                ++nonpositive_count;
            }

            const FE::Real zero_area_margin = FE::Real{0.0};
            EXPECT_FALSE(zero_area_margin > FE::Real{0.0});
            if (!(zero_area_margin > FE::Real{0.0})) {
                ++zero_area_rejection_count;
            }
        }
    }

    EXPECT_EQ(finite_positive_count, 16);
    EXPECT_EQ(nonpositive_count, 0);
    EXPECT_EQ(zero_area_rejection_count, 16);
    EXPECT_GT(minimum_margin, FE::Real{0.0});
    EXPECT_LE(maximum_consistency_error, FE::Real{1.0e-10});
    EXPECT_LE(maximum_diameter_error, FE::Real{1.0e-12});
    EXPECT_LE(maximum_cell_measure_error, FE::Real{1.0e-12});
    EXPECT_LE(maximum_facet_measure_error, FE::Real{1.0e-12});
    EXPECT_LE(maximum_h_normal_error, FE::Real{1.0e-12});

    ::testing::Test::RecordProperty(
        "sharp_nitsche_fraction_case_count",
        static_cast<int>(kWetFractions.size()));
    ::testing::Test::RecordProperty(
        "sharp_nitsche_policy_count",
        static_cast<int>(nitsche_policies.size()));
    ::testing::Test::RecordProperty(
        "sharp_nitsche_finite_positive_sample_count",
        finite_positive_count);
    ::testing::Test::RecordProperty(
        "sharp_nitsche_nonpositive_sample_count",
        nonpositive_count);
    recordRealProperty(
        "sharp_nitsche_maximum_scaled_consistency_error",
        maximum_consistency_error);
    ::testing::Test::RecordProperty(
        "sharp_nitsche_active_side_count",
        static_cast<int>(kActiveSides.size()));
    ::testing::Test::RecordProperty(
        "sharp_nitsche_zero_area_rejection_count",
        zero_area_rejection_count);
    recordRealProperty("sharp_nitsche_minimum_sampled_margin", minimum_margin);
    recordRealProperty(
        "sharp_nitsche_maximum_cell_diameter_error",
        maximum_diameter_error);
    recordRealProperty(
        "sharp_nitsche_maximum_cell_measure_error",
        maximum_cell_measure_error);
    recordRealProperty(
        "sharp_nitsche_maximum_parent_facet_measure_error",
        maximum_facet_measure_error);
    recordRealProperty(
        "sharp_nitsche_maximum_h_normal_error",
        maximum_h_normal_error);
}

#if FE_HAS_MPI || defined(MESH_HAS_MPI)

std::vector<FE::Real> globalSum(std::span<const FE::Real> local)
{
    static_assert(std::is_same_v<FE::Real, double>);
    if (local.size() > static_cast<std::size_t>(
                           std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "sharp-boundary reduction exceeds the MPI count range");
    }
    std::vector<FE::Real> global(local.size(), FE::Real{0.0});
    const int count = static_cast<int>(local.size());
    if (MPI_Allreduce(local.data(),
                      global.data(),
                      count,
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("sharp-boundary vector reduction failed");
    }
    return global;
}

int globalSum(int local)
{
    int global = 0;
    if (MPI_Allreduce(
            &local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) !=
        MPI_SUCCESS) {
        throw std::runtime_error("sharp-boundary count reduction failed");
    }
    return global;
}

FE::Real globalSum(FE::Real local)
{
    FE::Real global{0.0};
    if (MPI_Allreduce(
            &local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) !=
        MPI_SUCCESS) {
        throw std::runtime_error("sharp-boundary scalar reduction failed");
    }
    return global;
}

FE::Real globalMaximum(FE::Real local)
{
    FE::Real global{0.0};
    if (MPI_Allreduce(
            &local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) !=
        MPI_SUCCESS) {
        throw std::runtime_error("sharp-boundary maximum reduction failed");
    }
    return global;
}

bool mpiWorld(int& rank, int& size)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return false;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return true;
}

#endif

TEST(MovingDomainPhysicsMPI,
     GeneratedActiveCoupledOutflowReductionGradientAndTractionArePartitionIndependent)
{
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
    int rank = 0;
    int size = 1;
    if (!mpiWorld(rank, size) || size < 2) {
        GTEST_SKIP() << "requires an MPI launch with at least two ranks";
    }

    constexpr FE::Real interface_height = FE::Real{0.37};
    constexpr FE::Real velocity_magnitude = FE::Real{0.4};
    constexpr FE::Real resistance = FE::Real{2.0};
    constexpr FE::Real expected_flow_rate =
        velocity_magnitude * interface_height;
    constexpr FE::Real expected_traction_work =
        resistance * velocity_magnitude *
        interface_height * interface_height;
    constexpr std::array<FE::Real, 4>
        expected_gradient_actions{{
            interface_height,
            interface_height * interface_height / FE::Real{2.0},
            interface_height / FE::Real{2.0},
            FE::Real{0.0},
        }};

    StructuredGeneratedCoupledOutflowHarness serial(
        rank,
        size,
        ChannelPartition::Serial,
        true);
    StructuredGeneratedCoupledOutflowHarness slab(
        rank,
        size,
        ChannelPartition::XSlab,
        false);
    StructuredGeneratedCoupledOutflowHarness round_robin(
        rank,
        size,
        ChannelPartition::RoundRobin,
        false);

    const bool serial_dual_marker =
        serial.hasDualMarkerContract();
    const bool slab_dual_marker =
        slab.hasDualMarkerContract();
    const bool round_robin_dual_marker =
        round_robin.hasDualMarkerContract();
    EXPECT_TRUE(serial_dual_marker);
    EXPECT_TRUE(slab_dual_marker);
    EXPECT_TRUE(round_robin_dual_marker);
    const int dual_marker_contract_count =
        static_cast<int>(serial_dual_marker) +
        static_cast<int>(slab_dual_marker) +
        static_cast<int>(round_robin_dual_marker);

    const auto serial_sample = serial.sample();
    const auto slab_sample = slab.sample();
    const auto round_robin_sample =
        round_robin.sample();

    std::array<FE::Real, 4> slab_gradient_actions{};
    std::array<FE::Real, 4>
        round_robin_gradient_actions{};
    for (std::size_t probe_index = 0;
         probe_index < expected_gradient_actions.size();
         ++probe_index) {
        slab_gradient_actions[probe_index] = globalSum(
            slab_sample.local_gradient_actions[probe_index]);
        round_robin_gradient_actions[probe_index] = globalSum(
            round_robin_sample.local_gradient_actions[probe_index]);
    }

    const FE::Real slab_traction_work =
        globalSum(slab_sample.local_traction_work);
    const FE::Real round_robin_traction_work =
        globalSum(round_robin_sample.local_traction_work);
    const FE::Real slab_active_measure =
        globalSum(slab_sample.local_active_measure);
    const FE::Real round_robin_active_measure =
        globalSum(round_robin_sample.local_active_measure);
    const int slab_rule_count =
        globalSum(slab_sample.local_owned_rule_count);
    const int round_robin_rule_count =
        globalSum(
            round_robin_sample.local_owned_rule_count);
    const int owner_mismatch_count =
        serial_sample.local_owner_mismatch_count +
        globalSum(
            slab_sample.local_owner_mismatch_count) +
        globalSum(
            round_robin_sample.local_owner_mismatch_count);
    const int slab_outlet_contributor_count =
        globalSum(
            slab_sample.local_outlet_contributor_count);
    const int round_robin_outlet_contributor_count =
        globalSum(
            round_robin_sample
                .local_outlet_contributor_count);
    const int expected_rule_count =
        serial_sample.local_owned_rule_count;
    const int rule_count_mismatch =
        std::abs(slab_rule_count - expected_rule_count) +
        std::abs(
            round_robin_rule_count -
            expected_rule_count);

    FE::Real maximum_gradient_action_error{0.0};
    for (std::size_t probe_index = 0;
         probe_index < expected_gradient_actions.size();
         ++probe_index) {
        EXPECT_NEAR(
            serial_sample.local_gradient_actions[probe_index],
            expected_gradient_actions[probe_index],
            kPartitionTolerance);
        EXPECT_NEAR(
            slab_gradient_actions[probe_index],
            expected_gradient_actions[probe_index],
            kPartitionTolerance);
        EXPECT_NEAR(
            round_robin_gradient_actions[probe_index],
            expected_gradient_actions[probe_index],
            kPartitionTolerance);
        maximum_gradient_action_error = std::max({
            maximum_gradient_action_error,
            std::abs(
                serial_sample.local_gradient_actions[probe_index] -
                expected_gradient_actions[probe_index]),
            std::abs(
                slab_gradient_actions[probe_index] -
                expected_gradient_actions[probe_index]),
            std::abs(
                round_robin_gradient_actions[probe_index] -
                expected_gradient_actions[probe_index]),
        });
    }
    maximum_gradient_action_error =
        globalMaximum(maximum_gradient_action_error);

    EXPECT_NEAR(
        serial_sample.flow_rate,
        expected_flow_rate,
        kPartitionTolerance);
    EXPECT_NEAR(
        slab_sample.flow_rate,
        expected_flow_rate,
        kPartitionTolerance);
    EXPECT_NEAR(
        round_robin_sample.flow_rate,
        expected_flow_rate,
        kPartitionTolerance);
    EXPECT_NEAR(
        serial_sample.local_traction_work,
        expected_traction_work,
        kPartitionTolerance);
    EXPECT_NEAR(
        slab_traction_work,
        expected_traction_work,
        kPartitionTolerance);
    EXPECT_NEAR(
        round_robin_traction_work,
        expected_traction_work,
        kPartitionTolerance);
    EXPECT_NEAR(
        serial_sample.local_active_measure,
        interface_height,
        kPartitionTolerance);
    EXPECT_NEAR(
        slab_active_measure,
        interface_height,
        kPartitionTolerance);
    EXPECT_NEAR(
        round_robin_active_measure,
        interface_height,
        kPartitionTolerance);

    ASSERT_GT(expected_rule_count, 0);
    EXPECT_EQ(slab_rule_count, expected_rule_count);
    EXPECT_EQ(
        round_robin_rule_count,
        expected_rule_count);
    EXPECT_EQ(rule_count_mismatch, 0);
    EXPECT_EQ(owner_mismatch_count, 0);
    EXPECT_EQ(slab_outlet_contributor_count, 1);
    EXPECT_EQ(round_robin_outlet_contributor_count, 2);
    EXPECT_EQ(serial.cellCount(), 12);

    const int whole_face_fallback_count =
        static_cast<int>(
            std::abs(
                serial_sample.flow_rate -
                velocity_magnitude) <=
            kPartitionTolerance) +
        static_cast<int>(
            std::abs(
                slab_sample.flow_rate -
                velocity_magnitude) <=
            kPartitionTolerance) +
        static_cast<int>(
            std::abs(
                round_robin_sample.flow_rate -
                velocity_magnitude) <=
            kPartitionTolerance);
    EXPECT_EQ(whole_face_fallback_count, 0);

    const FE::Real maximum_measure_error =
        globalMaximum(std::max({
            std::abs(
                serial_sample.local_active_measure -
                interface_height),
            std::abs(
                slab_active_measure - interface_height),
            std::abs(
                round_robin_active_measure -
                interface_height),
        }));
    const FE::Real maximum_flow_error =
        globalMaximum(std::max({
            std::abs(
                serial_sample.flow_rate -
                expected_flow_rate),
            std::abs(
                slab_sample.flow_rate -
                expected_flow_rate),
            std::abs(
                round_robin_sample.flow_rate -
                expected_flow_rate),
        }));
    const FE::Real maximum_traction_work_error =
        globalMaximum(std::max({
            std::abs(
                serial_sample.local_traction_work -
                expected_traction_work),
            std::abs(
                slab_traction_work -
                expected_traction_work),
            std::abs(
                round_robin_traction_work -
                expected_traction_work),
        }));
    EXPECT_LE(maximum_measure_error, kPartitionTolerance);
    EXPECT_LE(maximum_flow_error, kPartitionTolerance);
    EXPECT_LE(
        maximum_gradient_action_error,
        kPartitionTolerance);
    EXPECT_LE(
        maximum_traction_work_error,
        kPartitionTolerance);

    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_rank_count",
        size);
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_cell_count",
        static_cast<int>(serial.cellCount()));
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_partition_count",
        2);
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_gradient_probe_count",
        static_cast<int>(
            expected_gradient_actions.size()));
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_dual_marker_contract_count",
        dual_marker_contract_count);
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_rule_count_mismatch",
        rule_count_mismatch);
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_owner_mismatch_count",
        owner_mismatch_count);
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_whole_face_fallback_count",
        whole_face_fallback_count);
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_slab_outlet_contributor_count",
        slab_outlet_contributor_count);
    ::testing::Test::RecordProperty(
        "sharp_coupled_outflow_mpi_round_robin_outlet_contributor_count",
        round_robin_outlet_contributor_count);
    recordRealProperty(
        "sharp_coupled_outflow_mpi_maximum_measure_error",
        maximum_measure_error);
    recordRealProperty(
        "sharp_coupled_outflow_mpi_maximum_flow_error",
        maximum_flow_error);
    recordRealProperty(
        "sharp_coupled_outflow_mpi_maximum_gradient_action_error",
        maximum_gradient_action_error);
    recordRealProperty(
        "sharp_coupled_outflow_mpi_maximum_traction_work_error",
        maximum_traction_work_error);
#else
    GTEST_SKIP() << "requires an MPI-enabled build";
#endif
}

TEST(FreeSurfaceSharpBoundaryOperatorsMPI,
     StructuredChannelWorkIsInvariantUnderActualRepartition)
{
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
    int rank = 0;
    int size = 1;
    if (!mpiWorld(rank, size) || size < 2) {
        GTEST_SKIP() << "requires an MPI launch with at least two ranks";
    }

    constexpr FE::Real interface_height = FE::Real{0.37};
    constexpr std::array<FE::Real, 3> probe_value{{
        FE::Real{0.7}, FE::Real{-0.4}, FE::Real{0.6}}};
    const StructuredChannelOperatorScales baseline_scales{
        .inlet_traction = FE::Real{0.0},
        .outlet_pressure = FE::Real{0.0},
        .sidewall_nitsche_gamma = FE::Real{6.0},
    };
    const StructuredChannelOperatorScales active_scales{
        .inlet_traction = FE::Real{1.0},
        .outlet_pressure = FE::Real{1.0},
        .sidewall_nitsche_gamma = FE::Real{12.0},
    };

    StructuredChannelAssemblyHarness serial_baseline(
        2,
        0,
        1,
        ChannelPartition::Serial,
        baseline_scales,
        true);
    StructuredChannelAssemblyHarness serial_active(
        2,
        0,
        1,
        ChannelPartition::Serial,
        active_scales,
        true);
    StructuredChannelAssemblyHarness slab_baseline(
        2,
        rank,
        size,
        ChannelPartition::XSlab,
        baseline_scales,
        false);
    StructuredChannelAssemblyHarness slab_active(
        2,
        rank,
        size,
        ChannelPartition::XSlab,
        active_scales,
        false);
    StructuredChannelAssemblyHarness round_robin_baseline(
        2,
        rank,
        size,
        ChannelPartition::RoundRobin,
        baseline_scales,
        false);
    StructuredChannelAssemblyHarness round_robin_active(
        2,
        rank,
        size,
        ChannelPartition::RoundRobin,
        active_scales,
        false);

    const auto serial_baseline_sample =
        serial_baseline.assemble(interface_height);
    const auto serial_active_sample =
        serial_active.assemble(interface_height);
    const auto slab_baseline_sample =
        slab_baseline.assemble(interface_height);
    const auto slab_active_sample =
        slab_active.assemble(interface_height);
    const auto round_robin_baseline_sample =
        round_robin_baseline.assemble(interface_height);
    const auto round_robin_active_sample =
        round_robin_active.assemble(interface_height);

    const auto serial_probe =
        serial_active.constantVelocityProbe(probe_value);
    const auto slab_probe =
        slab_active.constantVelocityProbe(probe_value);
    const auto round_robin_probe =
        round_robin_active.constantVelocityProbe(probe_value);
    const FE::Real serial_work = work(
        difference(serial_active_sample.residual,
                   serial_baseline_sample.residual),
        serial_probe);
    const FE::Real slab_work = globalSum(work(
        difference(slab_active_sample.residual,
                   slab_baseline_sample.residual),
        slab_probe));
    const FE::Real round_robin_work = globalSum(work(
        difference(round_robin_active_sample.residual,
                   round_robin_baseline_sample.residual),
        round_robin_probe));
    ASSERT_GT(std::abs(serial_work), FE::Real{1.0e-12});

    const auto slab_owned =
        slab_active.observeOwnedRules(slab_active_sample.geometry);
    const auto round_robin_owned =
        round_robin_active.observeOwnedRules(
            round_robin_active_sample.geometry);
    const int slab_rule_count = globalSum(slab_owned.rule_count);
    const int round_robin_rule_count =
        globalSum(round_robin_owned.rule_count);
    const FE::Real slab_measure = globalSum(slab_owned.measure);
    const FE::Real round_robin_measure =
        globalSum(round_robin_owned.measure);
    const int expected_rule_count = std::accumulate(
        serial_active_sample.geometry.active_rule_counts.begin(),
        serial_active_sample.geometry.active_rule_counts.end(),
        0);
    constexpr FE::Real expected_measure =
        FE::Real{4.0} * interface_height;

    const FE::Real maximum_work_error = std::max(
        std::abs(slab_work - serial_work),
        std::abs(round_robin_work - serial_work));
    const FE::Real maximum_measure_error = std::max(
        std::abs(slab_measure - expected_measure),
        std::abs(round_robin_measure - expected_measure));
    const int rule_count_mismatch =
        std::abs(slab_rule_count - expected_rule_count) +
        std::abs(round_robin_rule_count - expected_rule_count);

    EXPECT_EQ(slab_rule_count, expected_rule_count);
    EXPECT_EQ(round_robin_rule_count, expected_rule_count);
    EXPECT_EQ(rule_count_mismatch, 0);
    EXPECT_LE(maximum_measure_error, kPartitionTolerance);
    EXPECT_LE(maximum_work_error, kPartitionTolerance);
    ::testing::Test::RecordProperty(
        "sharp_structured_mpi_rank_count", size);
    ::testing::Test::RecordProperty(
        "sharp_structured_mpi_cell_count",
        static_cast<int>(serial_active.cellCount()));
    ::testing::Test::RecordProperty(
        "sharp_structured_mpi_partition_count", 2);
    ::testing::Test::RecordProperty(
        "sharp_structured_mpi_boundary_role_count", 3);
    ::testing::Test::RecordProperty(
        "sharp_structured_mpi_rule_count_mismatch",
        rule_count_mismatch);
    recordRealProperty(
        "sharp_structured_mpi_maximum_measure_error",
        maximum_measure_error);
    recordRealProperty(
        "sharp_structured_mpi_maximum_work_error",
        maximum_work_error);
#else
    GTEST_SKIP() << "requires an MPI-enabled build";
#endif
}

TEST(GeneratedActiveBoundaryDomainMPI,
     WetFractionSweepIsOwnershipUniqueAndPartitionIndependent)
{
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
    int rank = 0;
    int size = 1;
    if (!mpiWorld(rank, size) || size < 2) {
        GTEST_SKIP() << "requires an MPI launch with at least two ranks";
    }

    int minimum_owner_multiplicity = std::numeric_limits<int>::max();
    int maximum_owner_multiplicity = 0;
    FE::Real maximum_measure_error{0.0};
    FE::Real maximum_residual_error{0.0};
    FE::Real maximum_work_error{0.0};
    FE::Real maximum_scaling_error{0.0};

    for (std::size_t side_index = 0;
         side_index < kActiveSides.size();
         ++side_index) {
        const int owner_rank = static_cast<int>(side_index) % size;
        SharpBoundaryAssemblyHarness reference_harness(
            OperatorFamily::Traction,
            kActiveSides[side_index],
            rank,
            size,
            rank,
            true);
        SharpBoundaryAssemblyHarness partition_harness(
            OperatorFamily::Traction,
            kActiveSides[side_index],
            rank,
            size,
            owner_rank);

        const auto reference_dry =
            reference_harness.assemble(FE::Real{0.0});
        const auto reference_full =
            reference_harness.assemble(kParentBoundaryMeasure);
        const auto reference_full_delta = difference(
            reference_full.residual, reference_dry.residual);
        const auto partition_dry =
            partition_harness.assemble(FE::Real{0.0});

        for (const auto fraction : kWetFractions) {
            const FE::Real measure =
                kParentBoundaryMeasure * fraction;
            const auto observation =
                partition_harness.observeActiveRule(measure);
            EXPECT_TRUE(observation.valid);
            EXPECT_EQ(observation.owner_rank, owner_rank);

            const int owner_multiplicity = globalSum(
                observation.valid && observation.owner_rank == rank ? 1 : 0);
            minimum_owner_multiplicity =
                std::min(minimum_owner_multiplicity, owner_multiplicity);
            maximum_owner_multiplicity =
                std::max(maximum_owner_multiplicity, owner_multiplicity);
            EXPECT_EQ(owner_multiplicity, 1);

            const FE::Real global_measure = globalSum(
                observation.valid && observation.owner_rank == rank
                    ? observation.measure
                    : FE::Real{0.0});
            maximum_measure_error = std::max(
                maximum_measure_error,
                std::abs(global_measure - measure));

            const auto reference_wet = reference_harness.assemble(measure);
            const auto reference_delta = difference(
                reference_wet.residual, reference_dry.residual);
            const auto partition_wet = partition_harness.assemble(measure);
            const auto partition_delta = difference(
                partition_wet.residual, partition_dry.residual);
            const auto global_delta = globalSum(partition_delta);

            maximum_residual_error = std::max(
                maximum_residual_error,
                maximumAbsoluteDifference(global_delta, reference_delta));
            maximum_scaling_error = std::max(
                maximum_scaling_error,
                maximumScalingError(
                    reference_delta, reference_full_delta, fraction));
            maximum_work_error = std::max(
                maximum_work_error,
                std::abs(work(global_delta, partition_harness.solution()) -
                         work(reference_delta,
                              reference_harness.solution())));
        }
    }

    maximum_measure_error = globalMaximum(maximum_measure_error);
    maximum_residual_error = globalMaximum(maximum_residual_error);
    maximum_work_error = globalMaximum(maximum_work_error);
    maximum_scaling_error = globalMaximum(maximum_scaling_error);

    EXPECT_EQ(minimum_owner_multiplicity, 1);
    EXPECT_EQ(maximum_owner_multiplicity, 1);
    EXPECT_LE(maximum_measure_error, kPartitionTolerance);
    EXPECT_LE(maximum_residual_error, kPartitionTolerance);
    EXPECT_LE(maximum_work_error, kPartitionTolerance);
    EXPECT_LE(maximum_scaling_error, kPartitionTolerance);

    ::testing::Test::RecordProperty("sharp_boundary_mpi_rank_count", size);
    ::testing::Test::RecordProperty(
        "sharp_boundary_mpi_fraction_case_count",
        static_cast<int>(kWetFractions.size()));
    ::testing::Test::RecordProperty(
        "sharp_boundary_mpi_active_side_case_count",
        static_cast<int>(kActiveSides.size()));
    ::testing::Test::RecordProperty(
        "sharp_boundary_mpi_partition_case_count",
        static_cast<int>(kWetFractions.size() * kActiveSides.size()));
    ::testing::Test::RecordProperty(
        "sharp_boundary_mpi_minimum_owner_multiplicity",
        minimum_owner_multiplicity);
    ::testing::Test::RecordProperty(
        "sharp_boundary_mpi_maximum_owner_multiplicity",
        maximum_owner_multiplicity);
    recordRealProperty(
        "sharp_boundary_mpi_maximum_measure_error",
        maximum_measure_error);
    recordRealProperty(
        "sharp_boundary_mpi_maximum_residual_error",
        maximum_residual_error);
    recordRealProperty(
        "sharp_boundary_mpi_maximum_work_error",
        maximum_work_error);
    recordRealProperty(
        "sharp_boundary_mpi_maximum_fraction_scaling_error",
        maximum_scaling_error);
#else
    GTEST_SKIP() << "requires an MPI-enabled build";
#endif
}

TEST(FreeSurfaceSharpBoundaryOperatorsMPI,
     OperatorWorkIsPartitionIndependent)
{
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
    int rank = 0;
    int size = 1;
    if (!mpiWorld(rank, size) || size < 2) {
        GTEST_SKIP() << "requires an MPI launch with at least two ranks";
    }

    int minimum_owner_multiplicity = std::numeric_limits<int>::max();
    int maximum_owner_multiplicity = 0;
    FE::Real maximum_residual_error{0.0};
    FE::Real maximum_jacobian_error{0.0};
    FE::Real maximum_work_error{0.0};
    FE::Real maximum_residual_scaling_error{0.0};
    FE::Real maximum_jacobian_scaling_error{0.0};
    FE::Real minimum_full_residual_norm =
        std::numeric_limits<FE::Real>::infinity();
    int trace_sample_count = 0;
    int positive_trace_sample_count = 0;
    FE::Real minimum_trace_margin =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real maximum_trace_partition_error{0.0};
    FE::Real maximum_trace_consistency_error{0.0};
    constexpr FE::Real trace_high_gamma = FE::Real{12.0};
    constexpr FE::Real trace_low_gamma = FE::Real{6.0};
    constexpr FE::Real trace_viscosity = FE::Real{0.01};
    constexpr FE::Real trace_h_normal = FE::Real{2.0} / FE::Real{3.0};
    constexpr std::array<FE::Real, 3> trace_probe_value{{
        FE::Real{0.3}, FE::Real{-0.15}, FE::Real{0.2}}};
    constexpr FE::Real trace_probe_squared =
        trace_probe_value[0] * trace_probe_value[0] +
        trace_probe_value[1] * trace_probe_value[1] +
        trace_probe_value[2] * trace_probe_value[2];
    constexpr FE::Real full_trace_margin =
        (trace_high_gamma - trace_low_gamma) * trace_viscosity /
        trace_h_normal * kParentBoundaryMeasure * trace_probe_squared;

    for (std::size_t family_index = 0;
         family_index < kOperatorFamilies.size();
         ++family_index) {
        for (std::size_t side_index = 0;
             side_index < kActiveSides.size();
             ++side_index) {
            const int owner_rank = static_cast<int>(
                family_index * kActiveSides.size() + side_index) % size;
            SharpBoundaryAssemblyHarness reference_harness(
                kOperatorFamilies[family_index],
                kActiveSides[side_index],
                rank,
                size,
                rank,
                true);
            SharpBoundaryAssemblyHarness partition_harness(
                kOperatorFamilies[family_index],
                kActiveSides[side_index],
                rank,
                size,
                owner_rank);
            const bool is_nitsche =
                kOperatorFamilies[family_index] ==
                    OperatorFamily::SymmetricNitsche ||
                kOperatorFamilies[family_index] ==
                    OperatorFamily::UnsymmetricNitsche;
            std::unique_ptr<SharpBoundaryAssemblyHarness>
                reference_low_penalty;
            std::unique_ptr<SharpBoundaryAssemblyHarness>
                partition_low_penalty;
            if (is_nitsche) {
                reference_low_penalty =
                    std::make_unique<SharpBoundaryAssemblyHarness>(
                        kOperatorFamilies[family_index],
                        kActiveSides[side_index],
                        rank,
                        size,
                        rank,
                        true,
                        FE::Real{1.0},
                        FE::Real{1.0},
                        trace_low_gamma);
                partition_low_penalty =
                    std::make_unique<SharpBoundaryAssemblyHarness>(
                        kOperatorFamilies[family_index],
                        kActiveSides[side_index],
                        rank,
                        size,
                        owner_rank,
                        false,
                        FE::Real{1.0},
                        FE::Real{1.0},
                        trace_low_gamma);
            }

            const auto reference_dry =
                reference_harness.assemble(FE::Real{0.0});
            const auto reference_full =
                reference_harness.assemble(kParentBoundaryMeasure);
            const auto reference_full_residual = difference(
                reference_full.residual, reference_dry.residual);
            const auto reference_full_jacobian = difference(
                reference_full.jacobian, reference_dry.jacobian);
            minimum_full_residual_norm = std::min(
                minimum_full_residual_norm,
                vectorNorm(reference_full_residual));
            const auto partition_dry =
                partition_harness.assemble(FE::Real{0.0});

            for (const auto fraction : kWetFractions) {
                const FE::Real measure =
                    kParentBoundaryMeasure * fraction;
                const auto observation =
                    partition_harness.observeActiveRule(measure);
                EXPECT_TRUE(observation.valid);
                EXPECT_EQ(observation.owner_rank, owner_rank);

                const int owner_multiplicity = globalSum(
                    observation.valid && observation.owner_rank == rank ? 1 : 0);
                minimum_owner_multiplicity =
                    std::min(minimum_owner_multiplicity, owner_multiplicity);
                maximum_owner_multiplicity =
                    std::max(maximum_owner_multiplicity, owner_multiplicity);
                EXPECT_EQ(owner_multiplicity, 1);

                const auto reference_wet =
                    reference_harness.assemble(measure);
                const auto reference_residual = difference(
                    reference_wet.residual, reference_dry.residual);
                const auto reference_jacobian = difference(
                    reference_wet.jacobian, reference_dry.jacobian);

                const auto partition_wet =
                    partition_harness.assemble(measure);
                const auto local_residual = difference(
                    partition_wet.residual, partition_dry.residual);
                const auto local_jacobian = difference(
                    partition_wet.jacobian, partition_dry.jacobian);
                const auto global_residual = globalSum(local_residual);
                const auto global_jacobian = globalSum(local_jacobian);

                maximum_residual_error = std::max(
                    maximum_residual_error,
                    maximumAbsoluteDifference(
                        global_residual, reference_residual));
                maximum_jacobian_error = std::max(
                    maximum_jacobian_error,
                    maximumAbsoluteDifference(
                        global_jacobian, reference_jacobian));
                maximum_work_error = std::max(
                    maximum_work_error,
                    std::abs(work(global_residual,
                                  partition_harness.solution()) -
                             work(reference_residual,
                                  reference_harness.solution())));
                maximum_residual_scaling_error = std::max(
                    maximum_residual_scaling_error,
                    maximumScalingError(reference_residual,
                                        reference_full_residual,
                                        fraction));
                maximum_jacobian_scaling_error = std::max(
                    maximum_jacobian_scaling_error,
                    maximumScalingError(reference_jacobian,
                                        reference_full_jacobian,
                                        fraction));

                if (is_nitsche) {
                    const auto reference_low =
                        reference_low_penalty->assemble(measure);
                    const auto partition_low =
                        partition_low_penalty->assemble(measure);
                    const auto reference_penalty_matrix = difference(
                        reference_wet.jacobian,
                        reference_low.jacobian);
                    const auto local_penalty_matrix = difference(
                        partition_wet.jacobian,
                        partition_low.jacobian);
                    const auto global_penalty_matrix =
                        globalSum(local_penalty_matrix);
                    const auto trace_probe =
                        reference_harness.constantVelocityProbe(
                            trace_probe_value);
                    const FE::Real reference_margin = quadraticWork(
                        reference_penalty_matrix, trace_probe);
                    const FE::Real partition_margin = quadraticWork(
                        global_penalty_matrix, trace_probe);
                    const FE::Real expected_margin =
                        (trace_high_gamma - trace_low_gamma) *
                        trace_viscosity / trace_h_normal * measure *
                        trace_probe_squared;
                    ++trace_sample_count;
                    if (std::isfinite(reference_margin) &&
                        reference_margin > FE::Real{0.0} &&
                        std::isfinite(partition_margin) &&
                        partition_margin > FE::Real{0.0}) {
                        ++positive_trace_sample_count;
                    }
                    minimum_trace_margin = std::min(
                        minimum_trace_margin,
                        std::min(reference_margin, partition_margin));
                    maximum_trace_partition_error = std::max(
                        maximum_trace_partition_error,
                        scaledAbsoluteError(partition_margin,
                                            reference_margin,
                                            full_trace_margin));
                    maximum_trace_consistency_error = std::max(
                        maximum_trace_consistency_error,
                        scaledAbsoluteError(reference_margin,
                                            expected_margin,
                                            full_trace_margin));
                }
            }
        }
    }

    maximum_residual_error = globalMaximum(maximum_residual_error);
    maximum_jacobian_error = globalMaximum(maximum_jacobian_error);
    maximum_work_error = globalMaximum(maximum_work_error);
    maximum_residual_scaling_error =
        globalMaximum(maximum_residual_scaling_error);
    maximum_jacobian_scaling_error =
        globalMaximum(maximum_jacobian_scaling_error);
    minimum_full_residual_norm = -globalMaximum(-minimum_full_residual_norm);
    minimum_trace_margin = -globalMaximum(-minimum_trace_margin);
    maximum_trace_partition_error =
        globalMaximum(maximum_trace_partition_error);
    maximum_trace_consistency_error =
        globalMaximum(maximum_trace_consistency_error);

    EXPECT_EQ(minimum_owner_multiplicity, 1);
    EXPECT_EQ(maximum_owner_multiplicity, 1);
    EXPECT_GT(minimum_full_residual_norm, FE::Real{1.0e-12});
    EXPECT_LE(maximum_residual_error, kPartitionTolerance);
    EXPECT_LE(maximum_jacobian_error, kPartitionTolerance);
    EXPECT_LE(maximum_work_error, kPartitionTolerance);
    EXPECT_LE(maximum_residual_scaling_error, kPartitionTolerance);
    EXPECT_LE(maximum_jacobian_scaling_error, kPartitionTolerance);
    EXPECT_EQ(trace_sample_count,
              static_cast<int>(2u * kActiveSides.size() *
                               kWetFractions.size()));
    EXPECT_EQ(positive_trace_sample_count, trace_sample_count);
    EXPECT_GT(minimum_trace_margin, FE::Real{0.0});
    EXPECT_LE(maximum_trace_partition_error, kPartitionTolerance);
    EXPECT_LE(maximum_trace_consistency_error, FE::Real{1.0e-10});

    ::testing::Test::RecordProperty("sharp_operator_mpi_rank_count", size);
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_fraction_case_count",
        static_cast<int>(kWetFractions.size()));
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_active_side_case_count",
        static_cast<int>(kActiveSides.size()));
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_family_count",
        static_cast<int>(kOperatorFamilies.size()));
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_partition_case_count",
        static_cast<int>(kWetFractions.size() * kActiveSides.size() *
                         kOperatorFamilies.size()));
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_minimum_owner_multiplicity",
        minimum_owner_multiplicity);
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_maximum_owner_multiplicity",
        maximum_owner_multiplicity);
    recordRealProperty(
        "sharp_operator_mpi_minimum_full_residual_norm",
        minimum_full_residual_norm);
    recordRealProperty(
        "sharp_operator_mpi_maximum_residual_error",
        maximum_residual_error);
    recordRealProperty(
        "sharp_operator_mpi_maximum_jacobian_error",
        maximum_jacobian_error);
    recordRealProperty(
        "sharp_operator_mpi_maximum_work_error",
        maximum_work_error);
    recordRealProperty(
        "sharp_operator_mpi_maximum_residual_scaling_error",
        maximum_residual_scaling_error);
    recordRealProperty(
        "sharp_operator_mpi_maximum_jacobian_scaling_error",
        maximum_jacobian_scaling_error);
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_trace_sample_count",
        trace_sample_count);
    ::testing::Test::RecordProperty(
        "sharp_operator_mpi_positive_trace_sample_count",
        positive_trace_sample_count);
    recordRealProperty(
        "sharp_operator_mpi_minimum_trace_margin",
        minimum_trace_margin);
    recordRealProperty(
        "sharp_operator_mpi_maximum_trace_partition_error",
        maximum_trace_partition_error);
    recordRealProperty(
        "sharp_operator_mpi_maximum_trace_consistency_error",
        maximum_trace_consistency_error);
#else
    GTEST_SKIP() << "requires an MPI-enabled build";
#endif
}

} // namespace
} // namespace svmp::Physics::test
