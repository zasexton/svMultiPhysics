/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ManufacturedSolutionConvergence.cpp
 * @brief End-to-end FE verification: assemble + constrain + solve + convergence rate.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Constraints/AffineConstraints.h"
#include "Core/FEException.h"
#include "Dofs/DofHandler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Geometry/GeometryMapping.h"
#include "Geometry/MappingFactory.h"
#include "Math/Vector.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

class StructuredQuadMeshAccess final : public IMeshAccess {
public:
    explicit StructuredQuadMeshAccess(int n_cells_per_axis)
        : n_(n_cells_per_axis)
    {
        FE_THROW_IF(n_ <= 0, InvalidArgumentException,
                    "StructuredQuadMeshAccess: n must be >= 1");

        const int n_nodes_1d = n_ + 1;
        nodes_.resize(static_cast<std::size_t>(n_nodes_1d * n_nodes_1d));
        for (int j = 0; j < n_nodes_1d; ++j) {
            for (int i = 0; i < n_nodes_1d; ++i) {
                const Real x = static_cast<Real>(i) / static_cast<Real>(n_);
                const Real y = static_cast<Real>(j) / static_cast<Real>(n_);
                nodes_[static_cast<std::size_t>(nodeId(i, j))] = {x, y, 0.0};
            }
        }

        cells_.resize(static_cast<std::size_t>(n_ * n_));
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < n_; ++i) {
                const GlobalIndex c = cellId(i, j);
                cells_[static_cast<std::size_t>(c)] = {
                    nodeId(i, j),
                    nodeId(i + 1, j),
                    nodeId(i + 1, j + 1),
                    nodeId(i, j + 1)};
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Quad4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] int cellsPerAxis() const noexcept { return n_; }

private:
    [[nodiscard]] GlobalIndex nodeId(int i, int j) const
    {
        const int n_nodes_1d = n_ + 1;
        return static_cast<GlobalIndex>(i + n_nodes_1d * j);
    }

    [[nodiscard]] GlobalIndex cellId(int i, int j) const
    {
        return static_cast<GlobalIndex>(i + n_ * j);
    }

    int n_{1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildQuadGridTopology(const StructuredQuadMeshAccess& mesh)
{
    const int n = mesh.cellsPerAxis();
    const GlobalIndex n_cells = static_cast<GlobalIndex>(n * n);
    const GlobalIndex n_vertices = static_cast<GlobalIndex>((n + 1) * (n + 1));

    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_vertices;
    topo.dim = 2;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 4u, 0);

    std::vector<GlobalIndex> cell_nodes;
    cell_nodes.reserve(4);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(4u * c);
        mesh.getCellNodes(c, cell_nodes);
        FE_CHECK_ARG(cell_nodes.size() == 4u, "buildQuadGridTopology: Quad4 cell node count");
        for (std::size_t i = 0; i < 4u; ++i) {
            topo.cell2vertex_data[static_cast<std::size_t>(4u * c + static_cast<GlobalIndex>(i))] =
                static_cast<MeshIndex>(cell_nodes[i]);
        }
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(4u * n_cells);

    topo.vertex_gids.resize(static_cast<std::size_t>(n_vertices), 0);
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(n_cells), 0);
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(n_cells), 0);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = 0;
    }

    return topo;
}

class StructuredQuadBoundaryMeshAccess final : public IMeshAccess {
public:
    explicit StructuredQuadBoundaryMeshAccess(int n_cells_per_axis, int boundary_marker)
        : n_(n_cells_per_axis)
        , boundary_marker_(boundary_marker)
    {
        FE_THROW_IF(n_ <= 0, InvalidArgumentException,
                    "StructuredQuadBoundaryMeshAccess: n must be >= 1");

        const int n_nodes_1d = n_ + 1;
        nodes_.resize(static_cast<std::size_t>(n_nodes_1d * n_nodes_1d));
        for (int j = 0; j < n_nodes_1d; ++j) {
            for (int i = 0; i < n_nodes_1d; ++i) {
                const Real x = static_cast<Real>(i) / static_cast<Real>(n_);
                const Real y = static_cast<Real>(j) / static_cast<Real>(n_);
                nodes_[static_cast<std::size_t>(nodeId(i, j))] = {x, y, 0.0};
            }
        }

        cells_.resize(static_cast<std::size_t>(n_ * n_));
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < n_; ++i) {
                const GlobalIndex c = cellId(i, j);
                cells_[static_cast<std::size_t>(c)] = {
                    nodeId(i, j),
                    nodeId(i + 1, j),
                    nodeId(i + 1, j + 1),
                    nodeId(i, j + 1)};
            }
        }

        // Boundary faces: one edge per boundary cell edge.
        //
        // Quad4 reference edge order: (0-1) bottom, (1-2) right, (2-3) top, (3-0) left.
        boundary_faces_.clear();
        boundary_faces_.reserve(static_cast<std::size_t>(4 * n_));

        // Bottom (y=0): cell (i,0), local edge 0.
        for (int i = 0; i < n_; ++i) {
            boundary_faces_.push_back({cellId(i, 0), LocalIndex{0}, boundary_marker_});
        }
        // Right (x=1): cell (n-1,j), local edge 1.
        for (int j = 0; j < n_; ++j) {
            boundary_faces_.push_back({cellId(n_ - 1, j), LocalIndex{1}, boundary_marker_});
        }
        // Top (y=1): cell (i,n-1), local edge 2.
        for (int i = 0; i < n_; ++i) {
            boundary_faces_.push_back({cellId(i, n_ - 1), LocalIndex{2}, boundary_marker_});
        }
        // Left (x=0): cell (0,j), local edge 3.
        for (int j = 0; j < n_; ++j) {
            boundary_faces_.push_back({cellId(0, j), LocalIndex{3}, boundary_marker_});
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return static_cast<GlobalIndex>(boundary_faces_.size()); }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Quad4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        const auto& f = boundary_faces_.at(static_cast<std::size_t>(face_id));
        FE_THROW_IF(f.cell_id != cell_id, InvalidArgumentException,
                    "StructuredQuadBoundaryMeshAccess::getLocalFaceIndex: face/cell mismatch");
        return f.local_face;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return boundary_faces_.at(static_cast<std::size_t>(face_id)).marker;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        for (GlobalIndex f = 0; f < numBoundaryFaces(); ++f) {
            const auto& face = boundary_faces_.at(static_cast<std::size_t>(f));
            if (marker < 0 || marker == face.marker) {
                callback(f, face.cell_id);
            }
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] int cellsPerAxis() const noexcept { return n_; }
    [[nodiscard]] int boundaryMarker() const noexcept { return boundary_marker_; }

private:
    struct BoundaryFace {
        GlobalIndex cell_id{-1};
        LocalIndex local_face{0};
        int marker{-1};
    };

    [[nodiscard]] GlobalIndex nodeId(int i, int j) const
    {
        const int n_nodes_1d = n_ + 1;
        return static_cast<GlobalIndex>(i + n_nodes_1d * j);
    }

    [[nodiscard]] GlobalIndex cellId(int i, int j) const
    {
        return static_cast<GlobalIndex>(i + n_ * j);
    }

    int n_{1};
    int boundary_marker_{-1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
    std::vector<BoundaryFace> boundary_faces_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildQuadGridTopology(const StructuredQuadBoundaryMeshAccess& mesh)
{
    const int n = mesh.cellsPerAxis();
    const GlobalIndex n_cells = static_cast<GlobalIndex>(n * n);
    const GlobalIndex n_vertices = static_cast<GlobalIndex>((n + 1) * (n + 1));

    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_vertices;
    topo.dim = 2;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 4u, 0);

    std::vector<GlobalIndex> cell_nodes;
    cell_nodes.reserve(4);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(4u * c);
        mesh.getCellNodes(c, cell_nodes);
        FE_CHECK_ARG(cell_nodes.size() == 4u, "buildQuadGridTopology: Quad4 cell node count");
        for (std::size_t i = 0; i < 4u; ++i) {
            topo.cell2vertex_data[static_cast<std::size_t>(4u * c + static_cast<GlobalIndex>(i))] =
                static_cast<MeshIndex>(cell_nodes[i]);
        }
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(4u * n_cells);

    topo.vertex_gids.resize(static_cast<std::size_t>(n_vertices), 0);
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(n_cells), 0);
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(n_cells), 0);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = 0;
    }

    return topo;
}

struct DenseSolveResult {
    bool ok{true};
    std::string message{};
    std::vector<Real> x{};
};

[[nodiscard]] DenseSolveResult solveDenseWithPartialPivoting(DenseMatrixView A, std::vector<Real> b)
{
    const GlobalIndex n = A.numRows();
    if (n != A.numCols()) {
        return {false, "solveDenseWithPartialPivoting: matrix is not square", {}};
    }
    if (static_cast<GlobalIndex>(b.size()) != n) {
        return {false, "solveDenseWithPartialPivoting: rhs size mismatch", {}};
    }

    auto data = A.dataMutable();
    auto idx = [n](GlobalIndex i, GlobalIndex j) {
        return static_cast<std::size_t>(i * n + j);
    };

    // Gaussian elimination with partial pivoting.
    for (GlobalIndex k = 0; k < n; ++k) {
        GlobalIndex pivot = k;
        Real max_abs = std::abs(data[idx(k, k)]);
        for (GlobalIndex i = k + 1; i < n; ++i) {
            const Real v = std::abs(data[idx(i, k)]);
            if (v > max_abs) {
                max_abs = v;
                pivot = i;
            }
        }
        if (max_abs < Real(1e-14)) {
            return {false, "solveDenseWithPartialPivoting: near-singular matrix", {}};
        }
        if (pivot != k) {
            for (GlobalIndex j = 0; j < n; ++j) {
                std::swap(data[idx(k, j)], data[idx(pivot, j)]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot)]);
        }

        const Real akk = data[idx(k, k)];
        for (GlobalIndex i = k + 1; i < n; ++i) {
            const Real factor = data[idx(i, k)] / akk;
            data[idx(i, k)] = 0.0;
            for (GlobalIndex j = k + 1; j < n; ++j) {
                data[idx(i, j)] -= factor * data[idx(k, j)];
            }
            b[static_cast<std::size_t>(i)] -= factor * b[static_cast<std::size_t>(k)];
        }
    }

    // Back-substitution.
    std::vector<Real> x(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex ii = 0; ii < n; ++ii) {
        const GlobalIndex i = n - 1 - ii;
        Real s = b[static_cast<std::size_t>(i)];
        for (GlobalIndex j = i + 1; j < n; ++j) {
            s -= data[idx(i, j)] * x[static_cast<std::size_t>(j)];
        }
        x[static_cast<std::size_t>(i)] = s / data[idx(i, i)];
    }

    return {true, {}, std::move(x)};
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> buildAffineMappingForQuad4(
    const IMeshAccess& mesh,
    GlobalIndex cell_id)
{
    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(cell_id, coords);
    FE_CHECK_ARG(coords.size() == 4u, "buildAffineMappingForQuad4: expected 4 geometry nodes");

    std::vector<math::Vector<Real, 3>> node_coords(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i) {
        node_coords[i] = math::Vector<Real, 3>{coords[i][0], coords[i][1], coords[i][2]};
    }

    geometry::MappingRequest req;
    req.element_type = ElementType::Quad4;
    req.geometry_order = 1;
    req.use_affine = true;
    return geometry::MappingFactory::create(req, node_coords);
}

[[nodiscard]] std::vector<std::array<Real, 3>> computeDofCoordinates(const IMeshAccess& mesh,
                                                                     const spaces::FunctionSpace& space,
                                                                     const dofs::DofMap& dof_map)
{
    const GlobalIndex n_dofs = dof_map.getNumDofs();
    std::vector<std::array<Real, 3>> coords(static_cast<std::size_t>(n_dofs),
                                            {std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN()});
    std::vector<char> has_coord(static_cast<std::size_t>(n_dofs), 0);

    const auto& element = space.getElement(ElementType::Quad4, /*cell_id=*/0);
    const auto& basis = element.basis();
    const auto* lag = dynamic_cast<const basis::LagrangeBasis*>(&basis);
    FE_CHECK_NOT_NULL(lag, "computeDofCoordinates: expected LagrangeBasis for H1Space");
    const auto& ref_nodes = lag->nodes();

    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForQuad4(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeDofCoordinates: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        FE_CHECK_ARG(ref_nodes.size() == cell_dofs.size(), "computeDofCoordinates: basis node count mismatch");

        for (LocalIndex li = 0; li < static_cast<LocalIndex>(cell_dofs.size()); ++li) {
            const GlobalIndex gd = cell_dofs[static_cast<std::size_t>(li)];
            if (gd < 0 || gd >= n_dofs) continue;
            if (has_coord[static_cast<std::size_t>(gd)]) continue;

            const auto& Xi = ref_nodes[static_cast<std::size_t>(li)];
            const math::Vector<Real, 3> xi{Xi[0], Xi[1], Xi[2]};
            const auto x = mapping->map_to_physical(xi);
            coords[static_cast<std::size_t>(gd)] = {x[0], x[1], x[2]};
            has_coord[static_cast<std::size_t>(gd)] = 1;
        }
    }

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        FE_THROW_IF(has_coord[static_cast<std::size_t>(i)] == 0, FEException,
                    "computeDofCoordinates: missing coordinate for DOF");
    }

    return coords;
}

struct DofCoordinatesAndComponents {
    std::vector<std::array<Real, 3>> coords;
    std::vector<int> component;
};

[[nodiscard]] DofCoordinatesAndComponents computeDofCoordinatesProductSpace(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    const dofs::DofMap& dof_map)
{
    FE_CHECK_ARG(space.space_type() == spaces::SpaceType::Product,
                 "computeDofCoordinatesProductSpace: expected ProductSpace");

    const GlobalIndex n_dofs = dof_map.getNumDofs();
    std::vector<std::array<Real, 3>> coords(static_cast<std::size_t>(n_dofs),
                                            {std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN()});
    std::vector<int> component(static_cast<std::size_t>(n_dofs), -1);
    std::vector<char> has_coord(static_cast<std::size_t>(n_dofs), 0);

    const int n_components = space.value_dimension();
    FE_THROW_IF(n_components <= 0, InvalidArgumentException,
                "computeDofCoordinatesProductSpace: value_dimension must be positive");

    const auto& element = space.getElement(ElementType::Quad4, /*cell_id=*/0);
    const auto& basis = element.basis();
    const auto* lag = dynamic_cast<const basis::LagrangeBasis*>(&basis);
    FE_CHECK_NOT_NULL(lag, "computeDofCoordinatesProductSpace: expected LagrangeBasis");
    const auto& ref_nodes = lag->nodes();

    const std::size_t dofs_per_comp = space.dofs_per_element() / static_cast<std::size_t>(n_components);
    FE_CHECK_ARG(dofs_per_comp * static_cast<std::size_t>(n_components) == space.dofs_per_element(),
                 "computeDofCoordinatesProductSpace: dofs_per_element not divisible by components");
    FE_CHECK_ARG(dofs_per_comp == ref_nodes.size(),
                 "computeDofCoordinatesProductSpace: scalar DOFs per component mismatch");

    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForQuad4(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeDofCoordinatesProductSpace: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        FE_CHECK_ARG(cell_dofs.size() == ref_nodes.size() * static_cast<std::size_t>(n_components),
                     "computeDofCoordinatesProductSpace: basis node count mismatch");

        for (LocalIndex li = 0; li < static_cast<LocalIndex>(cell_dofs.size()); ++li) {
            const GlobalIndex gd = cell_dofs[static_cast<std::size_t>(li)];
            if (gd < 0 || gd >= n_dofs) continue;

            const std::size_t comp = static_cast<std::size_t>(li) / dofs_per_comp;
            const std::size_t scalar_li = static_cast<std::size_t>(li) % dofs_per_comp;

            if (!has_coord[static_cast<std::size_t>(gd)]) {
                const auto& Xi = ref_nodes[scalar_li];
                const math::Vector<Real, 3> xi{Xi[0], Xi[1], Xi[2]};
                const auto x = mapping->map_to_physical(xi);
                coords[static_cast<std::size_t>(gd)] = {x[0], x[1], x[2]};
                has_coord[static_cast<std::size_t>(gd)] = 1;
            }

            if (component[static_cast<std::size_t>(gd)] == -1) {
                component[static_cast<std::size_t>(gd)] = static_cast<int>(comp);
            } else {
                FE_THROW_IF(component[static_cast<std::size_t>(gd)] != static_cast<int>(comp), FEException,
                            "computeDofCoordinatesProductSpace: inconsistent component index for DOF");
            }
        }
    }

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        FE_THROW_IF(has_coord[static_cast<std::size_t>(i)] == 0, FEException,
                    "computeDofCoordinatesProductSpace: missing coordinate for DOF");
        FE_THROW_IF(component[static_cast<std::size_t>(i)] < 0, FEException,
                    "computeDofCoordinatesProductSpace: missing component for DOF");
    }

    return {std::move(coords), std::move(component)};
}

[[nodiscard]] bool near(Real a, Real b, Real tol = Real(1e-12))
{
    return std::abs(a - b) <= tol;
}

[[nodiscard]] Real uExactPolyBiharmonicLike(Real x, Real y)
{
    // u = x(1-x) y(1-y) on [0,1]^2 with homogeneous Dirichlet.
    return x * (Real(1.0) - x) * y * (Real(1.0) - y);
}

[[nodiscard]] Real uExactPolyNotInQ2(Real x, Real y)
{
    // u = x^3(1-x) y(1-y) on [0,1]^2 with homogeneous Dirichlet.
    return x * x * x * (Real(1.0) - x) * y * (Real(1.0) - y);
}

[[nodiscard]] forms::FormExpr fRhsPolyBiharmonicLike()
{
    using svmp::FE::forms::FormExpr;
    using svmp::FE::forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);

    // f = -Δu = 2(x - x^2 + y - y^2) = 2(x(1-x) + y(1-y))
    return FormExpr::constant(Real(2.0)) *
           (xx * (FormExpr::constant(Real(1.0)) - xx) + yy * (FormExpr::constant(Real(1.0)) - yy));
}

[[nodiscard]] forms::FormExpr fRhsAdvectionDiffusionPolyBiharmonicLike(Real kappa,
                                                                       Real beta_x,
                                                                       Real beta_y)
{
    // For u = x(1-x) y(1-y):
    //   f = -kappa*Δu + beta·∇u
    //     = kappa*(-Δu) + beta_x*du/dx + beta_y*du/dy.
    using svmp::FE::forms::FormExpr;
    using svmp::FE::forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);

    const auto one = FormExpr::constant(Real(1.0));
    const auto two = FormExpr::constant(Real(2.0));

    const auto g = xx * (one - xx);
    const auto h = yy * (one - yy);
    const auto dudx = (one - two * xx) * h;
    const auto dudy = g * (one - two * yy);

    const auto adv = FormExpr::constant(beta_x) * dudx + FormExpr::constant(beta_y) * dudy;
    return FormExpr::constant(kappa) * fRhsPolyBiharmonicLike() + adv;
}

[[nodiscard]] forms::FormExpr fRhsPolyNotInQ2()
{
    // For u = x^3(1-x) y(1-y):
    //   f = -Δu = -(g''(x) h(y) + g(x) h''(y)) = -g''(x) h(y) + 2 g(x),
    // where g(x) = x^3(1-x), h(y) = y(1-y), g''(x) = 6x - 12x^2, h''(y) = -2.
    using svmp::FE::forms::FormExpr;
    using svmp::FE::forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);
    const auto one = FormExpr::constant(Real(1.0));

    const auto g = xx * xx * xx * (one - xx);
    const auto h = yy * (one - yy);
    const auto gpp = FormExpr::constant(Real(6.0)) * xx - FormExpr::constant(Real(12.0)) * xx * xx;

    return (-gpp * h) + FormExpr::constant(Real(2.0)) * g;
}

[[nodiscard]] forms::FormExpr fRhsAdvectionDiffusionPolyNotInQ2(Real kappa,
                                                                Real beta_x,
                                                                Real beta_y)
{
    // For u = x^3(1-x) y(1-y):
    //   f = -kappa*Δu + beta·∇u
    //     = kappa*(-Δu) + beta_x*du/dx + beta_y*du/dy.
    using svmp::FE::forms::FormExpr;
    using svmp::FE::forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);

    const auto one = FormExpr::constant(Real(1.0));
    const auto two = FormExpr::constant(Real(2.0));
    const auto three = FormExpr::constant(Real(3.0));
    const auto four = FormExpr::constant(Real(4.0));

    const auto g = xx * xx * xx * (one - xx);
    const auto h = yy * (one - yy);

    const auto gprime = three * xx * xx - four * xx * xx * xx;
    const auto hprime = one - two * yy;

    const auto adv = FormExpr::constant(beta_x) * gprime * h + FormExpr::constant(beta_y) * g * hprime;
    return FormExpr::constant(kappa) * fRhsPolyNotInQ2() + adv;
}

[[nodiscard]] Real uExactPeriodicInX_Q2(Real x, Real y)
{
    (void)x;
    // In Q2 space: u(y) = 1 + y(1-y)
    return Real(1.0) + y * (Real(1.0) - y);
}

[[nodiscard]] forms::FormExpr fRhsPeriodicInX_Q2()
{
    // For u(y) = 1 + y(1-y): -Δu = 2.
    return forms::FormExpr::constant(Real(2.0));
}

struct PoissonSolveOutput {
    std::vector<Real> u;
    std::vector<std::array<Real, 3>> dof_coords;
};

[[nodiscard]] PoissonSolveOutput solvePoissonOnQuadGrid(const IMeshAccess& mesh,
                                                        const dofs::MeshTopologyInfo& topo,
                                                        int order,
                                                        const forms::FormExpr& rhs_f,
                                                        constraints::AffineConstraints constraints)
{
    spaces::H1Space space(ElementType::Quad4, order);

    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const GlobalIndex n_dofs = dof_map.getNumDofs();

    auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

    // Constraints must be closed before assembly.
    if (!constraints.isClosed()) {
        constraints.close();
    }

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    // Assemble as a linear system Ku = b (not a residual Ku - b), because the current
    // constraint distribution logic applies inhomogeneous constraints with RHS semantics.
    const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + rhs_f * v).dx();

    auto ir = compiler.compileResidual(residual);
    forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setConstraints(&constraints);

    std::vector<Real> U0(static_cast<std::size_t>(n_dofs), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n_dofs);
    DenseVectorView R(n_dofs);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    std::vector<Real> rhs(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    // For multi-point constraints (periodic/MPC), the ConstraintDistributor condenses slave DOFs
    // by redistributing contributions to masters, leaving all-slave rows/cols zero in the
    // assembled matrix. Solve the reduced system (excluding non-Dirichlet constrained DOFs),
    // then recover constrained DOFs via AffineConstraints::distribute().
    std::vector<GlobalIndex> keep_dofs;
    keep_dofs.reserve(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex d = 0; d < n_dofs; ++d) {
        if (!constraints.isConstrained(d)) {
            keep_dofs.push_back(d);
            continue;
        }
        const auto c = constraints.getConstraint(d);
        FE_THROW_IF(!c.has_value(), FEException, "solvePoissonOnQuadGrid: constrained DOF missing constraint view");
        if (c->isDirichlet()) {
            keep_dofs.push_back(d);
        }
    }
    FE_THROW_IF(keep_dofs.empty(), FEException, "solvePoissonOnQuadGrid: no unconstrained DOFs to solve");

    const GlobalIndex n_keep = static_cast<GlobalIndex>(keep_dofs.size());
    DenseMatrixView A_keep(n_keep);
    std::vector<Real> rhs_keep(static_cast<std::size_t>(n_keep), 0.0);
    for (GlobalIndex ii = 0; ii < n_keep; ++ii) {
        const auto gi = keep_dofs[static_cast<std::size_t>(ii)];
        rhs_keep[static_cast<std::size_t>(ii)] = rhs[static_cast<std::size_t>(gi)];
        for (GlobalIndex jj = 0; jj < n_keep; ++jj) {
            const auto gj = keep_dofs[static_cast<std::size_t>(jj)];
            A_keep.addMatrixEntry(ii, jj, J.getMatrixEntry(gi, gj), AddMode::Insert);
        }
    }

    auto solve = solveDenseWithPartialPivoting(std::move(A_keep), std::move(rhs_keep));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n_keep, FEException,
                "solvePoissonOnQuadGrid: reduced solution size");

    std::vector<Real> u_full(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex ii = 0; ii < n_keep; ++ii) {
        const auto gi = keep_dofs[static_cast<std::size_t>(ii)];
        u_full[static_cast<std::size_t>(gi)] = solve.x[static_cast<std::size_t>(ii)];
    }
    constraints.distribute(u_full.data(), n_dofs);

    return {std::move(u_full), std::move(dof_coords)};
}

[[nodiscard]] PoissonSolveOutput solvePoissonWithNitscheDirichletOnQuadGrid(const IMeshAccess& mesh,
                                                                           int boundary_marker,
                                                                           const dofs::MeshTopologyInfo& topo,
                                                                           int order,
                                                                           const forms::FormExpr& rhs_f,
                                                                           Real gamma)
{
    spaces::H1Space space(ElementType::Quad4, order);

    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const GlobalIndex n_dofs = dof_map.getNumDofs();

    auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto n = forms::FormExpr::normal();

    const auto cell_residual = (forms::inner(forms::grad(u), forms::grad(v)) + rhs_f * v).dx();

    const auto gamma_c = forms::FormExpr::constant(gamma);
    const auto boundary_residual =
        (-forms::inner(forms::grad(u), n) * v - u * forms::inner(forms::grad(v), n) + (gamma_c / forms::h()) * u * v)
            .ds(boundary_marker);

    auto cell_ir = compiler.compileResidual(cell_residual);
    forms::SymbolicNonlinearFormKernel cell_kernel(std::move(cell_ir), forms::NonlinearKernelOutput::Both);
    cell_kernel.resolveInlinableConstitutives();

    auto boundary_ir = compiler.compileResidual(boundary_residual);
    forms::SymbolicNonlinearFormKernel boundary_kernel(std::move(boundary_ir), forms::NonlinearKernelOutput::Both);
    boundary_kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U0(static_cast<std::size_t>(n_dofs), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n_dofs);
    DenseVectorView R(n_dofs);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, cell_kernel, J, R);
    (void)assembler.assembleBoundaryFaces(mesh, boundary_marker, space, boundary_kernel, &J, &R);
    assembler.finalize(&J, &R);

    std::vector<Real> rhs(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    auto solve = solveDenseWithPartialPivoting(std::move(J), std::move(rhs));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n_dofs, FEException,
                "solvePoissonWithNitscheDirichletOnQuadGrid: solution size mismatch");

    return {std::move(solve.x), std::move(dof_coords)};
}

[[nodiscard]] PoissonSolveOutput solveAdvectionDiffusionOnQuadGrid(const IMeshAccess& mesh,
                                                                   const dofs::MeshTopologyInfo& topo,
                                                                   int order,
                                                                   Real kappa,
                                                                   Real beta_x,
                                                                   Real beta_y,
                                                                   forms::FormExpr rhs_f,
                                                                   constraints::AffineConstraints constraints)
{
    spaces::H1Space space(ElementType::Quad4, order);

    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const GlobalIndex n_dofs = dof_map.getNumDofs();

    auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

    // Constraints must be closed before assembly.
    if (!constraints.isClosed()) {
        constraints.close();
    }

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");

    const auto beta = forms::as_vector({forms::FormExpr::constant(beta_x),
                                        forms::FormExpr::constant(beta_y)});

    const auto f = std::move(rhs_f);

    // Assemble as a linear system Ku = b (not a residual Ku - b), because the current
    // constraint distribution logic applies inhomogeneous constraints with RHS semantics.
    const auto residual =
        (forms::FormExpr::constant(kappa) * forms::inner(forms::grad(u), forms::grad(v)) +
         forms::inner(beta, forms::grad(u)) * v +
         f * v)
            .dx();

    auto ir = compiler.compileResidual(residual);
    forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setConstraints(&constraints);

    std::vector<Real> U0(static_cast<std::size_t>(n_dofs), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n_dofs);
    DenseVectorView R(n_dofs);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    std::vector<Real> rhs(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    auto solve = solveDenseWithPartialPivoting(std::move(J), std::move(rhs));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n_dofs, FEException,
                "solveAdvectionDiffusionOnQuadGrid: solution size mismatch");

    constraints.distribute(solve.x.data(), n_dofs);

    return {std::move(solve.x), std::move(dof_coords)};
}

[[nodiscard]] Real computeL2Error(const IMeshAccess& mesh,
                                  const spaces::FunctionSpace& space,
                                  const dofs::DofMap& dof_map,
                                  std::span<const Real> u_h,
                                  const std::function<Real(Real, Real)>& u_exact,
                                  int quad_order)
{
    FE_CHECK_ARG(space.space_type() == spaces::SpaceType::H1, "computeL2Error: expected H1 space");

    auto quad = quadrature::QuadratureFactory::create(ElementType::Quad4, quad_order);
    FE_CHECK_NOT_NULL(quad.get(), "computeL2Error: quadrature");

    const auto& element = space.getElement(ElementType::Quad4, /*cell_id=*/0);
    const auto& basis = element.basis();

    std::vector<Real> phi;

    Real sum = 0.0;
    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForQuad4(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeL2Error: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        for (LocalIndex q = 0; q < static_cast<LocalIndex>(quad->num_points()); ++q) {
            const auto& qp = quad->point(static_cast<std::size_t>(q));
            const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};

            basis.evaluate_values(xi, phi);
            FE_CHECK_ARG(phi.size() == cell_dofs.size(), "computeL2Error: basis value size mismatch");

            Real uh = 0.0;
            for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
                const auto gd = cell_dofs[i];
                uh += u_h[static_cast<std::size_t>(gd)] * phi[i];
            }

            const auto x = mapping->map_to_physical(xi);
            const Real ue = u_exact(x[0], x[1]);
            const Real e = uh - ue;

            const Real detJ = mapping->jacobian_determinant(xi);
            sum += e * e * quad->weight(static_cast<std::size_t>(q)) * std::abs(detJ);
        }
    }

    return std::sqrt(sum);
}

[[nodiscard]] Real computeL2ErrorVector(const IMeshAccess& mesh,
                                        const spaces::FunctionSpace& space,
                                        const dofs::DofMap& dof_map,
                                        std::span<const Real> u_h,
                                        const std::function<std::array<Real, 2>(Real, Real)>& u_exact,
                                        int quad_order)
{
    FE_CHECK_ARG(space.space_type() == spaces::SpaceType::Product, "computeL2ErrorVector: expected ProductSpace");
    FE_CHECK_ARG(space.value_dimension() == 2, "computeL2ErrorVector: expected 2D vector field");

    auto quad = quadrature::QuadratureFactory::create(ElementType::Quad4, quad_order);
    FE_CHECK_NOT_NULL(quad.get(), "computeL2ErrorVector: quadrature");

    std::vector<Real> cell_coeffs;

    Real sum = 0.0;
    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForQuad4(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeL2ErrorVector: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        cell_coeffs.resize(cell_dofs.size());
        for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
            const auto gd = cell_dofs[i];
            cell_coeffs[i] = u_h[static_cast<std::size_t>(gd)];
        }

        for (LocalIndex q = 0; q < static_cast<LocalIndex>(quad->num_points()); ++q) {
            const auto& qp = quad->point(static_cast<std::size_t>(q));
            const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};

            const auto uh = space.evaluate(xi, cell_coeffs);

            const auto x = mapping->map_to_physical(xi);
            const auto ue = u_exact(x[0], x[1]);
            const Real e0 = uh[0] - ue[0];
            const Real e1 = uh[1] - ue[1];

            const Real detJ = mapping->jacobian_determinant(xi);
            const Real e2 = e0 * e0 + e1 * e1;
            sum += e2 * quad->weight(static_cast<std::size_t>(q)) * std::abs(detJ);
        }
    }

    return std::sqrt(sum);
}

void addDirichletOnBoundaryVector(const DofCoordinatesAndComponents& dof_info,
                                  GlobalIndex dof_offset,
                                  const std::function<std::array<Real, 2>(Real, Real)>& value,
                                  constraints::AffineConstraints& c,
                                  Real tol = Real(1e-12))
{
    FE_CHECK_ARG(dof_info.coords.size() == dof_info.component.size(), "addDirichletOnBoundaryVector: size mismatch");

    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_info.coords.size()); ++dof) {
        const auto& x = dof_info.coords[static_cast<std::size_t>(dof)];
        if (!(near(x[0], Real(0.0), tol) || near(x[0], Real(1.0), tol) ||
              near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol))) {
            continue;
        }

        const auto u = value(x[0], x[1]);
        const int comp = dof_info.component[static_cast<std::size_t>(dof)];
        FE_THROW_IF(comp < 0 || comp >= 2, InvalidArgumentException,
                    "addDirichletOnBoundaryVector: component out of range");

        c.addDirichlet(dof_offset + dof, u[static_cast<std::size_t>(comp)]);
    }
}

[[nodiscard]] GlobalIndex findDofAtPoint(const std::vector<std::array<Real, 3>>& dof_coords,
                                        Real x,
                                        Real y,
                                        Real tol = Real(1e-12))
{
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& xd = dof_coords[static_cast<std::size_t>(dof)];
        if (near(xd[0], x, tol) && near(xd[1], y, tol)) {
            return dof;
        }
    }
    FE_THROW(FEException, "findDofAtPoint: DOF not found");
}

[[nodiscard]] constraints::AffineConstraints makeDirichletOnBoundary(
    const std::vector<std::array<Real, 3>>& dof_coords,
    const std::function<Real(Real, Real)>& value,
    Real tol = Real(1e-12))
{
    constraints::AffineConstraints c;
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (near(x[0], Real(0.0), tol) || near(x[0], Real(1.0), tol) ||
            near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol)) {
            c.addDirichlet(dof, value(x[0], x[1]));
        }
    }
    c.close();
    return c;
}

[[nodiscard]] constraints::AffineConstraints makeDirichletOnYAndPeriodicInX(
    const std::vector<std::array<Real, 3>>& dof_coords,
    const std::function<Real(Real, Real)>& dirichlet_value,
    Real tol = Real(1e-12))
{
    constraints::AffineConstraints c;

    // Dirichlet on y=0 and y=1 (inhomogeneous allowed).
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol)) {
            c.addDirichlet(dof, dirichlet_value(x[0], x[1]));
        }
    }

    // Periodic in x: map x=1 DOFs to x=0 DOFs at matching y (exclude y-Dirichlet DOFs).
    struct Candidate {
        Real y{0.0};
        GlobalIndex dof{-1};
    };
    std::vector<Candidate> masters;
    masters.reserve(dof_coords.size());

    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (near(x[0], Real(0.0), tol) &&
            !near(x[1], Real(0.0), tol) && !near(x[1], Real(1.0), tol)) {
            masters.push_back({x[1], dof});
        }
    }

    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (!near(x[0], Real(1.0), tol) || near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol)) {
            continue;
        }

        // Skip if already Dirichlet.
        if (c.isConstrained(dof)) {
            continue;
        }

        GlobalIndex master = -1;
        for (const auto& cand : masters) {
            if (near(cand.y, x[1], tol)) {
                master = cand.dof;
                break;
            }
        }
        FE_THROW_IF(master < 0, FEException,
                    "makeDirichletOnYAndPeriodicInX: missing periodic master DOF");

        c.addLine(dof);
        c.addEntry(dof, master, 1.0);
    }

    c.close();
    return c;
}

[[nodiscard]] std::vector<Real> convergenceRates(const std::vector<Real>& errors)
{
    std::vector<Real> rates;
    if (errors.size() < 2u) return rates;
    rates.reserve(errors.size() - 1u);
    for (std::size_t i = 0; i + 1u < errors.size(); ++i) {
        rates.push_back(std::log(errors[i] / errors[i + 1u]) / std::log(Real(2.0)));
    }
    return rates;
}

struct StokesTaylorHoodMmsErrors {
    Real u_l2{0.0};
    Real p_l2{0.0};
};

[[nodiscard]] StokesTaylorHoodMmsErrors stokesTaylorHoodMmsErrors(int n_cells_per_axis)
{
    StructuredQuadMeshAccess mesh(n_cells_per_axis);
    const auto topo = buildQuadGridTopology(mesh);

    constexpr Real mu = 1.0;
    const Real pi = std::acos(Real(-1.0));

    auto u_scalar = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/2);
    spaces::ProductSpace u_space(std::move(u_scalar), /*components=*/2);
    spaces::H1Space p_space(ElementType::Quad4, /*order=*/1);

    dofs::DofHandler u_dofs;
    u_dofs.distributeDofs(topo, u_space);
    u_dofs.finalize();

    dofs::DofHandler p_dofs;
    p_dofs.distributeDofs(topo, p_space);
    p_dofs.finalize();

    const auto& u_map = u_dofs.getDofMap();
    const auto& p_map = p_dofs.getDofMap();

    const GlobalIndex n_u = u_map.getNumDofs();
    const GlobalIndex n_p = p_map.getNumDofs();
    const GlobalIndex n = n_u + n_p;

    const auto u_dof_info = computeDofCoordinatesProductSpace(mesh, u_space, u_map);
    const auto p_dof_coords = computeDofCoordinates(mesh, p_space, p_map);

    const auto u_exact = [pi](Real x, Real y) -> std::array<Real, 2> {
        const Real u1 = std::sin(pi * x) * std::cos(pi * y);
        const Real u2 = -std::cos(pi * x) * std::sin(pi * y);
        return {u1, u2};
    };

    const auto p_exact = [pi](Real x, Real y) -> Real {
        return std::sin(pi * x) * std::sin(pi * y);
    };

    constraints::AffineConstraints constraints;
    addDirichletOnBoundaryVector(u_dof_info, /*dof_offset=*/0, u_exact, constraints);

    // Pressure nullspace pin.
    const auto p0 = findDofAtPoint(p_dof_coords, Real(0.0), Real(0.0));
    constraints.addDirichlet(n_u + p0, p_exact(Real(0.0), Real(0.0)));
    constraints.close();

    forms::FormCompiler compiler;

    const auto u = forms::TrialFunction(u_space, "u");
    const auto v = forms::TestFunction(u_space, "v");
    const auto p = forms::TrialFunction(p_space, "p");
    const auto q = forms::TestFunction(p_space, "q");

    const auto mu_c = forms::FormExpr::constant(mu);

    const auto f = forms::FormExpr::coefficient("f", [pi](Real x, Real y, Real /*z*/) {
        const Real u1 = std::sin(pi * x) * std::cos(pi * y);
        const Real u2 = -std::cos(pi * x) * std::sin(pi * y);
        const Real dpdx = pi * std::cos(pi * x) * std::sin(pi * y);
        const Real dpdy = pi * std::sin(pi * x) * std::cos(pi * y);
        const Real f1 = Real(2.0) * pi * pi * u1 + dpdx;
        const Real f2 = Real(2.0) * pi * pi * u2 + dpdy;
        return std::array<Real, 3>{f1, f2, 0.0};
    });

    // Assemble as a linear system Ku = b (not residual Ku - b): evaluate at U=0 so R == RHS.
    const auto uu_residual =
        (forms::FormExpr::constant(2.0) * mu_c * forms::inner(forms::sym(forms::grad(u)), forms::sym(forms::grad(v))) +
         forms::inner(f, v))
            .dx();
    const auto up_residual = (-p * forms::div(v)).dx();
    const auto pu_residual = (q * forms::div(u)).dx();

    auto uu_ir = compiler.compileResidual(uu_residual);
    forms::SymbolicNonlinearFormKernel uu_kernel(std::move(uu_ir), forms::NonlinearKernelOutput::Both);
    uu_kernel.resolveInlinableConstitutives();

    auto up_ir = compiler.compileResidual(up_residual);
    forms::SymbolicNonlinearFormKernel up_kernel(std::move(up_ir), forms::NonlinearKernelOutput::Both);
    up_kernel.resolveInlinableConstitutives();

    auto pu_ir = compiler.compileResidual(pu_residual);
    forms::SymbolicNonlinearFormKernel pu_kernel(std::move(pu_ir), forms::NonlinearKernelOutput::Both);
    pu_kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setConstraints(&constraints);

    std::vector<Real> U0(static_cast<std::size_t>(n), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n);
    DenseVectorView R(n);
    J.zero();
    R.zero();

    assembler.setRowDofMap(u_map, /*row_offset=*/0);
    assembler.setColDofMap(u_map, /*col_offset=*/0);
    (void)assembler.assembleBoth(mesh, u_space, u_space, uu_kernel, J, R);

    assembler.setRowDofMap(u_map, /*row_offset=*/0);
    assembler.setColDofMap(p_map, /*col_offset=*/n_u);
    (void)assembler.assembleBoth(mesh, u_space, p_space, up_kernel, J, R);

    assembler.setRowDofMap(p_map, /*row_offset=*/n_u);
    assembler.setColDofMap(u_map, /*col_offset=*/0);
    (void)assembler.assembleBoth(mesh, p_space, u_space, pu_kernel, J, R);

    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    auto solve = solveDenseWithPartialPivoting(std::move(J), std::move(rhs));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n, FEException,
                "stokesTaylorHoodMmsErrors: solution size mismatch");

    constraints.distribute(solve.x.data(), n);

    std::vector<Real> u_sol(static_cast<std::size_t>(n_u), 0.0);
    std::vector<Real> p_sol(static_cast<std::size_t>(n_p), 0.0);
    for (GlobalIndex i = 0; i < n_u; ++i) {
        u_sol[static_cast<std::size_t>(i)] = solve.x[static_cast<std::size_t>(i)];
    }
    for (GlobalIndex i = 0; i < n_p; ++i) {
        p_sol[static_cast<std::size_t>(i)] = solve.x[static_cast<std::size_t>(n_u + i)];
    }

    const Real u_err = computeL2ErrorVector(mesh, u_space, u_map, u_sol, u_exact, /*quad_order=*/12);
    const Real p_err = computeL2Error(mesh, p_space, p_map, p_sol, p_exact, /*quad_order=*/10);

    return {u_err, p_err};
}

[[nodiscard]] Real elasticityMmsError(int n_cells_per_axis, Real nu)
{
    StructuredQuadMeshAccess mesh(n_cells_per_axis);
    const auto topo = buildQuadGridTopology(mesh);

    const Real pi = std::acos(Real(-1.0));
    constexpr Real E = 1.0;
    const Real mu = E / (Real(2.0) * (Real(1.0) + nu));
    const Real lambda = E * nu / ((Real(1.0) + nu) * (Real(1.0) - Real(2.0) * nu));

    auto u_scalar = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    spaces::ProductSpace u_space(std::move(u_scalar), /*components=*/2);

    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, u_space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const GlobalIndex n = dof_map.getNumDofs();

    const auto dof_info = computeDofCoordinatesProductSpace(mesh, u_space, dof_map);

    const auto u_exact = [pi](Real x, Real y) -> std::array<Real, 2> {
        const Real g = std::sin(pi * x) * std::sin(pi * y);
        return {g, Real(0.5) * g};
    };

    constraints::AffineConstraints constraints;
    addDirichletOnBoundaryVector(dof_info, /*dof_offset=*/0, u_exact, constraints);
    constraints.close();

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(u_space, "u");
    const auto v = forms::TestFunction(u_space, "v");

    const auto mu_c = forms::FormExpr::constant(mu);
    const auto lambda_c = forms::FormExpr::constant(lambda);

    const auto f = forms::FormExpr::coefficient("f", [pi, mu, lambda](Real x, Real y, Real /*z*/) {
        const Real g = std::sin(pi * x) * std::sin(pi * y);
        const Real c = std::cos(pi * x) * std::cos(pi * y);

        const Real f1 = pi * pi * ((Real(3.0) * mu + lambda) * g - Real(0.5) * (lambda + mu) * c);
        const Real f2 = pi * pi * ((Real(1.5) * mu + Real(0.5) * lambda) * g - (lambda + mu) * c);
        return std::array<Real, 3>{f1, f2, 0.0};
    });

    const auto residual =
        (forms::FormExpr::constant(2.0) * mu_c * forms::inner(forms::sym(forms::grad(u)), forms::sym(forms::grad(v))) +
         lambda_c * forms::div(u) * forms::div(v) +
         forms::inner(f, v))
            .dx();

    auto ir = compiler.compileResidual(residual);
    forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setConstraints(&constraints);

    std::vector<Real> U0(static_cast<std::size_t>(n), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n);
    DenseVectorView R(n);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, u_space, u_space, kernel, J, R);

    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    auto solve = solveDenseWithPartialPivoting(std::move(J), std::move(rhs));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n, FEException,
                "elasticityMmsError: solution size mismatch");

    constraints.distribute(solve.x.data(), n);

    return computeL2ErrorVector(mesh, u_space, dof_map, solve.x, u_exact, /*quad_order=*/10);
}

} // namespace

TEST(ManufacturedSolutionConvergence, PoissonQuad4_Q1_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredQuadMeshAccess mesh(N);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/1);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundary(
            dof_coords,
            [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); });

        auto out = solvePoissonOnQuadGrid(mesh, topo, /*order=*/1, fRhsPolyBiharmonicLike(), std::move(constraints));

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); },
            /*quad_order=*/10);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(1.8));
    EXPECT_GT(rates[1], Real(1.8));
}

TEST(ManufacturedSolutionConvergence, PoissonQuad4_Q2_L2ConvergesAtOrderThree)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredQuadMeshAccess mesh(N);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/2);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundary(
            dof_coords,
            [](Real x, Real y) { return uExactPolyNotInQ2(x, y); });

        auto out = solvePoissonOnQuadGrid(mesh, topo, /*order=*/2, fRhsPolyNotInQ2(), std::move(constraints));

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyNotInQ2(x, y); },
            /*quad_order=*/10);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(2.6));
    EXPECT_GT(rates[1], Real(2.6));
}

TEST(ManufacturedSolutionConvergence, PoissonQuad4_Q1_NitscheDirichlet_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    constexpr int marker = 2;
    constexpr Real gamma = Real(40.0);

    for (const int N : Ns) {
        StructuredQuadBoundaryMeshAccess mesh(N, marker);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/1);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();

        auto out = solvePoissonWithNitscheDirichletOnQuadGrid(mesh, marker, topo, /*order=*/1,
                                                              fRhsPolyBiharmonicLike(), gamma);

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); },
            /*quad_order=*/12);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(1.8));
    EXPECT_GT(rates[1], Real(1.8));
}

TEST(ManufacturedSolutionConvergence, PoissonQuad4_Q2_NitscheDirichlet_L2ConvergesAtOrderThree)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    constexpr int marker = 2;
    constexpr Real gamma = Real(50.0);

    for (const int N : Ns) {
        StructuredQuadBoundaryMeshAccess mesh(N, marker);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/2);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();

        auto out = solvePoissonWithNitscheDirichletOnQuadGrid(mesh, marker, topo, /*order=*/2,
                                                              fRhsPolyNotInQ2(), gamma);

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyNotInQ2(x, y); },
            /*quad_order=*/14);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(2.6));
    EXPECT_GT(rates[1], Real(2.6));
}

TEST(ManufacturedSolutionConvergence, AdvectionDiffusionQuad4_Q1_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    constexpr Real kappa = Real(0.1);
    constexpr Real beta_x = Real(1.0);
    constexpr Real beta_y = Real(0.5);

    for (const int N : Ns) {
        StructuredQuadMeshAccess mesh(N);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/1);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundary(
            dof_coords,
            [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); });

        const auto f = fRhsAdvectionDiffusionPolyBiharmonicLike(kappa, beta_x, beta_y);
        auto out = solveAdvectionDiffusionOnQuadGrid(mesh, topo, /*order=*/1,
                                                     kappa, beta_x, beta_y,
                                                     f,
                                                     std::move(constraints));

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); },
            /*quad_order=*/12);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(1.8));
    EXPECT_GT(rates[1], Real(1.8));
}

TEST(ManufacturedSolutionConvergence, AdvectionDiffusionQuad4_Q2_L2ConvergesAtOrderThree)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    constexpr Real kappa = Real(0.1);
    constexpr Real beta_x = Real(1.0);
    constexpr Real beta_y = Real(0.5);

    for (const int N : Ns) {
        StructuredQuadMeshAccess mesh(N);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/2);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundary(
            dof_coords,
            [](Real x, Real y) { return uExactPolyNotInQ2(x, y); });

        const auto f = fRhsAdvectionDiffusionPolyNotInQ2(kappa, beta_x, beta_y);
        auto out = solveAdvectionDiffusionOnQuadGrid(mesh, topo, /*order=*/2,
                                                     kappa, beta_x, beta_y,
                                                     f,
                                                     std::move(constraints));

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyNotInQ2(x, y); },
            /*quad_order=*/14);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(2.6));
    EXPECT_GT(rates[1], Real(2.6));
}

TEST(ConstrainedSolve, PeriodicXAndInhomogeneousDirichletY_Q2PolynomialSolvesToRoundoff)
{
    constexpr int N = 4;

    StructuredQuadMeshAccess mesh(N);
    auto topo = buildQuadGridTopology(mesh);

    spaces::H1Space space(ElementType::Quad4, /*order=*/2);
    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

    auto constraints = makeDirichletOnYAndPeriodicInX(
        dof_coords,
        [](Real x, Real y) { return uExactPeriodicInX_Q2(x, y); });

    auto out = solvePoissonOnQuadGrid(mesh, topo, /*order=*/2, fRhsPeriodicInX_Q2(), std::move(constraints));

    // Verify solution matches exact at DOF points (polynomial lies in Q2 space).
    const Real tol = 5e-11;
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(out.u.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        EXPECT_NEAR(out.u[static_cast<std::size_t>(dof)], uExactPeriodicInX_Q2(x[0], x[1]), tol);
    }

    // Verify periodic DOFs: x=1 matches x=0 for interior y.
    for (GlobalIndex dof_s = 0; dof_s < static_cast<GlobalIndex>(dof_coords.size()); ++dof_s) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof_s)];
        if (!near(x[0], Real(1.0)) || near(x[1], Real(0.0)) || near(x[1], Real(1.0))) {
            continue;
        }

        GlobalIndex dof_m = -1;
        for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
            const auto& xm = dof_coords[static_cast<std::size_t>(dof)];
            if (near(xm[0], Real(0.0)) && near(xm[1], x[1])) {
                dof_m = dof;
                break;
            }
        }
        ASSERT_GE(dof_m, 0);
        EXPECT_NEAR(out.u[static_cast<std::size_t>(dof_s)], out.u[static_cast<std::size_t>(dof_m)], tol);
    }
}

TEST(ManufacturedSolutionConvergence, StokesTaylorHoodQuad4_Q2Q1_L2ConvergesAtExpectedRates)
{
    std::vector<Real> u_errors;
    std::vector<Real> p_errors;
    std::vector<int> Ns = {2, 4, 8};
    u_errors.reserve(Ns.size());
    p_errors.reserve(Ns.size());

    for (const int N : Ns) {
        const auto err = stokesTaylorHoodMmsErrors(N);
        u_errors.push_back(err.u_l2);
        p_errors.push_back(err.p_l2);
    }

    ASSERT_EQ(u_errors.size(), Ns.size());
    ASSERT_EQ(p_errors.size(), Ns.size());
    EXPECT_GT(u_errors[0], u_errors[1]);
    EXPECT_GT(u_errors[1], u_errors[2]);
    EXPECT_GT(p_errors[0], p_errors[1]);
    EXPECT_GT(p_errors[1], p_errors[2]);

    const auto u_rates = convergenceRates(u_errors);
    const auto p_rates = convergenceRates(p_errors);
    ASSERT_EQ(u_rates.size(), 2u);
    ASSERT_EQ(p_rates.size(), 2u);
    EXPECT_GT(u_rates[0], Real(2.6));
    EXPECT_GT(u_rates[1], Real(2.6));
    EXPECT_GT(p_rates[0], Real(1.8));
    EXPECT_GT(p_rates[1], Real(1.8));
}

TEST(ManufacturedSolutionConvergence, ElasticityQuad4_Q1_L2Converges_Nu03)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {4, 8, 16};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        errors.push_back(elasticityMmsError(N, /*nu=*/Real(0.3)));
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(1.8));
    EXPECT_GT(rates[1], Real(1.8));
}

TEST(ManufacturedSolutionConvergence, ElasticityQuad4_Q1_L2Converges_Nu0499)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {4, 8, 16};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        errors.push_back(elasticityMmsError(N, /*nu=*/Real(0.499)));
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    // Q1 displacement with large λ (ν→0.5) exhibits volumetric locking; we only assert that
    // the method remains stable and still converges (rates remain positive).
    EXPECT_GT(rates[0], Real(0.5));
    EXPECT_GT(rates[1], Real(0.5));
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
