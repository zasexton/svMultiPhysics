/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"
#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Forms/FormExpr.h"
#include "FE/Forms/FormCompiler.h"
#include "FE/Forms/FormKernels.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Assembly/StandardAssembler.h"
#include "FE/Analysis/FormExprScanner.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Constraints/AffineConstraints.h"
#include "FE/Dofs/DofMap.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "FE/Assembly/MeshAccess.h"
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

namespace svmp {
namespace Physics {
namespace test {
namespace {

using FE::forms::FormExpr;
using FE::forms::FormExprNode;
using FE::forms::FormExprType;
constexpr FE::FieldId kMeshVelocityField = 907;
namespace mm = formulations::mesh_motion;
namespace ls = FE::level_set;
namespace ns = formulations::navier_stokes;

using FreeSurfaceContactLine =
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

template <typename Configuration>
concept HasContactMobility = requires(Configuration value) {
    value.mobility;
};

template <typename Configuration>
concept HasContactSlipLength = requires(Configuration value) {
    value.slip_length;
};

template <typename Configuration>
concept HasContactAnglePenalty = requires(Configuration value) {
    value.contact_angle_penalty;
};

static_assert(!HasContactMobility<FreeSurfaceContactLine::PrescribedAngle>);
static_assert(!HasContactSlipLength<FreeSurfaceContactLine::PrescribedAngle>);
static_assert(!HasContactAnglePenalty<FreeSurfaceContactLine::DynamicRenE>);

FreeSurfaceContactLine pinnedContactLine(int wall_marker = -1,
                                         int contact_marker = -1)
{
    return FreeSurfaceContactLine{
        .configuration = FreeSurfaceContactLine::Pinned{
            .wall_boundary_marker = wall_marker,
            .contact_line_marker = contact_marker,
        },
    };
}

FreeSurfaceContactLine prescribedContactLine(
    int wall_marker,
    FE::Real angle,
    std::array<FE::Real, 3> wall_normal,
    FE::Real penalty = 1.0,
    int contact_marker = -1)
{
    return FreeSurfaceContactLine{
        .configuration = FreeSurfaceContactLine::PrescribedAngle{
            .wall_boundary_marker = wall_marker,
            .contact_line_marker = contact_marker,
            .contact_angle_radians = angle,
            .wall_normal = {wall_normal[0], wall_normal[1], wall_normal[2]},
            .contact_angle_penalty = penalty,
        },
    };
}

FreeSurfaceContactLine dynamicRenEContactLine(
    int wall_marker,
    FE::Real equilibrium_angle,
    std::array<FE::Real, 3> wall_normal,
    FE::Real mobility,
    FE::Real slip_length,
    int contact_marker = -1)
{
    return FreeSurfaceContactLine{
        .configuration = FreeSurfaceContactLine::DynamicRenE{
            .wall_boundary_marker = wall_marker,
            .contact_line_marker = contact_marker,
            .equilibrium_contact_angle_radians = equilibrium_angle,
            .wall_normal = {wall_normal[0], wall_normal[1], wall_normal[2]},
            .mobility = mobility,
            .slip_length = slip_length,
        },
    };
}

FreeSurfaceContactLine::PrescribedAngle& prescribedContactConfiguration(
    FreeSurfaceContactLine& contact_line)
{
    return std::get<FreeSurfaceContactLine::PrescribedAngle>(
        contact_line.configuration);
}

FreeSurfaceContactLine::DynamicRenE& dynamicContactConfiguration(
    FreeSurfaceContactLine& contact_line)
{
    return std::get<FreeSurfaceContactLine::DynamicRenE>(
        contact_line.configuration);
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* key, std::optional<std::string> value)
        : key_(key)
    {
        if (const char* existing = std::getenv(key_); existing != nullptr) {
            original_ = std::string(existing);
        }
        set(std::move(value));
    }

    ~ScopedEnvVar() { set(original_); }

private:
    void set(std::optional<std::string> value)
    {
        if (value.has_value()) {
            setenv(key_, value->c_str(), 1);
        } else {
            unsetenv(key_);
        }
    }

    const char* key_{nullptr};
    std::optional<std::string> original_{};
};

bool containsExprType(const FormExprNode* node, FormExprType target)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == target) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsExprType(child, target)) {
            return true;
        }
    }
    return false;
}

bool containsExprType(const FormExpr& expr, FormExprType target)
{
    return expr.isValid() && containsExprType(expr.node(), target);
}

bool containsGradientOfTrialFunction(const FormExprNode* node)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == FormExprType::Gradient &&
        containsExprType(node, FormExprType::TrialFunction)) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsGradientOfTrialFunction(child)) {
            return true;
        }
    }
    return false;
}

bool containsFieldExprType(const FormExprNode* node,
                           FormExprType target,
                           FE::FieldId field)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == target && node->fieldId().has_value() &&
        *node->fieldId() == field) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsFieldExprType(child, target, field)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsContain(const FE::systems::FESystem& system, FormExprType target)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsExprType(record.residual_expr.get(), target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsExprType(expr.get(), target)) {
                return true;
            }
        }
    }
    return false;
}

bool interfaceIntegralContainsExprType(const FormExprNode* node,
                                       int marker,
                                       FormExprType target)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == FormExprType::InterfaceIntegral &&
        node->interfaceMarker() == marker) {
        for (const auto* child : node->children()) {
            if (containsExprType(child, target)) {
                return true;
            }
        }
        return false;
    }
    for (const auto* child : node->children()) {
        if (interfaceIntegralContainsExprType(child, marker, target)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsInterfaceIntegralContainsExprType(
    const FE::systems::FESystem& system,
    int marker,
    FormExprType target)
{
    for (const auto& record : system.formulationRecords()) {
        if (interfaceIntegralContainsExprType(
                record.residual_expr.get(), marker, target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (interfaceIntegralContainsExprType(
                    expr.get(), marker, target)) {
                return true;
            }
        }
    }
    return false;
}

bool formulationRecordsContainFieldExprType(const FE::systems::FESystem& system,
                                            FormExprType target,
                                            FE::FieldId field)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsFieldExprType(record.residual_expr.get(), target, field)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsFieldExprType(expr.get(), target, field)) {
                return true;
            }
        }
    }
    return false;
}

bool interfaceKernelContainsGradientOfTrialFunction(
    const FE::systems::FESystem& system,
    int interface_marker,
    FE::FieldId trial_field)
{
    const auto& equations = system.operatorDefinition("equations");
    for (const auto& term : equations.interface_faces) {
        if (term.marker != interface_marker ||
            term.trial_field != trial_field ||
            !term.kernel) {
            continue;
        }
        const auto form_kernel =
            std::dynamic_pointer_cast<FE::forms::FormKernel>(term.kernel);
        if (!form_kernel) {
            continue;
        }
        for (const auto& integral : form_kernel->ir().terms()) {
            if (integral.domain != FE::forms::IntegralDomain::InterfaceFace ||
                integral.interface_marker != interface_marker) {
                continue;
            }
            if (containsGradientOfTrialFunction(integral.integrand.node())) {
                return true;
            }
        }
    }
    return false;
}

bool formulationRecordsContainInterfaceMarker(const FE::systems::FESystem& system,
                                              int marker)
{
    for (const auto& record : system.formulationRecords()) {
        if (record.residual_expr) {
            const auto scan = FE::analysis::scanFormExpr(*record.residual_expr);
            if (std::find(scan.interface_markers.begin(),
                          scan.interface_markers.end(),
                          marker) != scan.interface_markers.end()) {
                return true;
            }
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (!expr) {
                continue;
            }
            const auto scan = FE::analysis::scanFormExpr(*expr);
            if (std::find(scan.interface_markers.begin(),
                          scan.interface_markers.end(),
                          marker) != scan.interface_markers.end()) {
                return true;
            }
        }
    }
    return false;
}

bool formulationRecordsContainBoundaryMarker(
    const FE::systems::FESystem& system,
    int marker)
{
    for (const auto& record : system.formulationRecords()) {
        if (record.residual_expr) {
            const auto scan = FE::analysis::scanFormExpr(*record.residual_expr);
            if (std::find(scan.boundary_markers.begin(),
                          scan.boundary_markers.end(),
                          marker) != scan.boundary_markers.end()) {
                return true;
            }
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (!expr) {
                continue;
            }
            const auto scan = FE::analysis::scanFormExpr(*expr);
            if (std::find(scan.boundary_markers.begin(),
                          scan.boundary_markers.end(),
                          marker) != scan.boundary_markers.end()) {
                return true;
            }
        }
    }
    return false;
}

int stableContactLineMarker(FE::FieldId phi_field,
                            int interface_marker,
                            int wall_boundary_marker,
                            std::string domain_id = "free_surface",
                            FE::Real isovalue = 0.0)
{
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi_field);
    key.domain_id = std::move(domain_id);
    key.isovalue = isovalue;
    key.interface_marker = interface_marker;
    key.boundary_marker = wall_boundary_marker;
    return FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
        key);
}

bool containsInteriorFaceMarker(const FormExprNode* node, int marker)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == FormExprType::InteriorFaceIntegral) {
        const auto found = node->interfaceMarker();
        if (found.has_value() && *found == marker) {
            return true;
        }
    }
    for (const auto* child : node->children()) {
        if (containsInteriorFaceMarker(child, marker)) {
            return true;
        }
    }
    return false;
}

bool containsUnmarkedInteriorFaceIntegral(const FormExprNode* node)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == FormExprType::InteriorFaceIntegral) {
        const auto marker = node->interfaceMarker();
        if (!marker.has_value() || *marker < 0) {
            return true;
        }
    }
    for (const auto* child : node->children()) {
        if (containsUnmarkedInteriorFaceIntegral(child)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsContainInteriorFaceMarker(const FE::systems::FESystem& system,
                                                 int marker)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsInteriorFaceMarker(record.residual_expr.get(), marker)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsInteriorFaceMarker(expr.get(), marker)) {
                return true;
            }
        }
    }
    return false;
}

bool formulationRecordsContainUnmarkedInteriorFaceIntegral(
    const FE::systems::FESystem& system)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsUnmarkedInteriorFaceIntegral(record.residual_expr.get())) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsUnmarkedInteriorFaceIntegral(expr.get())) {
                return true;
            }
        }
    }
    return false;
}

std::size_t interiorFaceKernelCountForBlock(const FE::systems::FESystem& system,
                                            FE::FieldId test_field,
                                            FE::FieldId trial_field,
                                            int marker)
{
    if (!system.hasOperator("equations")) {
        return 0u;
    }
    std::size_t count = 0u;
    const auto& equations = system.operatorDefinition("equations");
    for (const auto& term : equations.interior) {
        if (term.test_field == test_field &&
            term.trial_field == trial_field &&
            term.marker == marker) {
            ++count;
        }
    }
    return count;
}

std::shared_ptr<SingleTetraMeshAccess> makeMesh()
{
    return std::make_shared<SingleTetraMeshAccess>();
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
std::shared_ptr<Mesh> makeRegistryQuadMesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return create_mesh(std::move(base));
}

std::shared_ptr<Mesh> makeStructuredQuadMesh(int cells_per_axis,
                                             FE::Real min_coord,
                                             FE::Real max_coord)
{
    auto base = std::make_shared<MeshBase>();

    const int nodes_per_axis = cells_per_axis + 1;
    const FE::Real h = (max_coord - min_coord) /
                       static_cast<FE::Real>(cells_per_axis);

    std::vector<real_t> x_ref;
    x_ref.reserve(static_cast<std::size_t>(nodes_per_axis * nodes_per_axis * 2));
    for (int j = 0; j < nodes_per_axis; ++j) {
        for (int i = 0; i < nodes_per_axis; ++i) {
            x_ref.push_back(static_cast<real_t>(
                min_coord + h * static_cast<FE::Real>(i)));
            x_ref.push_back(static_cast<real_t>(
                min_coord + h * static_cast<FE::Real>(j)));
        }
    }

    std::vector<offset_t> cell2vertex_offsets;
    std::vector<index_t> cell2vertex;
    cell2vertex_offsets.reserve(
        static_cast<std::size_t>(cells_per_axis * cells_per_axis + 1));
    cell2vertex.reserve(
        static_cast<std::size_t>(cells_per_axis * cells_per_axis * 4));
    cell2vertex_offsets.push_back(0);
    for (int j = 0; j < cells_per_axis; ++j) {
        for (int i = 0; i < cells_per_axis; ++i) {
            const index_t v00 = static_cast<index_t>(j * nodes_per_axis + i);
            const index_t v10 = v00 + 1;
            const index_t v01 = static_cast<index_t>((j + 1) * nodes_per_axis + i);
            const index_t v11 = v01 + 1;
            cell2vertex.push_back(v00);
            cell2vertex.push_back(v10);
            cell2vertex.push_back(v11);
            cell2vertex.push_back(v01);
            cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
        }
    }

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    const std::vector<CellShape> cell_shapes(
        static_cast<std::size_t>(cells_per_axis * cells_per_axis),
        shape);
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        cell_shapes);
    base->finalize();

    return create_mesh(std::move(base));
}

std::shared_ptr<Mesh> makeOpenTankQuadMesh(int left_marker,
                                           int right_marker,
                                           int bottom_marker,
                                           int free_surface_marker,
                                           std::string_view free_surface_set,
                                           FE::Real bottom_y = -1.0,
                                           FE::Real middle_y = 0.0,
                                           FE::Real top_y = 1.0)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<real_t> x_ref = {
        -1.0, static_cast<real_t>(bottom_y),
         0.0, static_cast<real_t>(bottom_y),
         1.0, static_cast<real_t>(bottom_y),
        -1.0, static_cast<real_t>(middle_y),
         0.0, static_cast<real_t>(middle_y),
         1.0, static_cast<real_t>(middle_y),
        -1.0, static_cast<real_t>(top_y),
         0.0, static_cast<real_t>(top_y),
         1.0, static_cast<real_t>(top_y),
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4, 8, 12, 16};
    const std::vector<index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4,
        3, 4, 7, 6,
        4, 5, 8, 7,
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(4, shape));
    base->finalize();

    base->register_label("wall_left", static_cast<label_t>(left_marker));
    base->register_label("wall_right", static_cast<label_t>(right_marker));
    base->register_label("wall_bottom", static_cast<label_t>(bottom_marker));
    base->register_label("free_surface", static_cast<label_t>(free_surface_marker));

    const auto coordinate = [&](index_t vertex, int component) {
        return base->X_ref().at(static_cast<std::size_t>(2 * vertex + component));
    };
    const auto all_vertices_match = [&](std::span<const index_t> vertices,
                                        int component,
                                        real_t value) {
        return std::all_of(vertices.begin(), vertices.end(), [&](index_t vertex) {
            return std::abs(coordinate(vertex, component) - value) < real_t(1.0e-14);
        });
    };

    for (index_t face = 0; face < static_cast<index_t>(base->n_faces()); ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() != 2u) {
            continue;
        }
        label_t label = INVALID_LABEL;
        if (all_vertices_match(vertices, /*component=*/1, static_cast<real_t>(top_y))) {
            label = static_cast<label_t>(free_surface_marker);
        } else if (all_vertices_match(vertices, /*component=*/1, static_cast<real_t>(bottom_y))) {
            label = static_cast<label_t>(bottom_marker);
        } else if (all_vertices_match(vertices, /*component=*/0, real_t(-1.0))) {
            label = static_cast<label_t>(left_marker);
        } else if (all_vertices_match(vertices, /*component=*/0, real_t(1.0))) {
            label = static_cast<label_t>(right_marker);
        }

        if (label == INVALID_LABEL) {
            continue;
        }
        base->set_boundary_label(face, label);
        if (label == static_cast<label_t>(free_surface_marker)) {
            base->add_to_set(EntityKind::Face, std::string(free_surface_set), face);
        }
    }

    return create_mesh(std::move(base));
}
#endif

FE::systems::SetupInputs makeSingleTriangleSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 3;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 3};
    topo.cell2vertex_data = {0, 1, 2};
    topo.vertex_gids = {0, 1, 2};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

class SingleTetraBoundaryMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleTetraBoundaryMeshAccess(int marker,
                                           bool expose_all_faces = false)
        : marker_(marker)
        , expose_all_faces_(expose_all_faces)
    {
        reference_nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        current_nodes_ = reference_nodes_;
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numOwnedVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return expose_all_faces_ ? 4 : 1;
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return geometry_revision_; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra4;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/, std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(FE::GlobalIndex node_id) const override
    {
        return current_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(FE::GlobalIndex /*cell_id*/,
                            std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords = current_nodes_;
    }

    [[nodiscard]] bool supportsCoordinateFrame(FE::assembly::CoordinateFrame frame) const override
    {
        return frame == FE::assembly::CoordinateFrame::Active ||
               frame == FE::assembly::CoordinateFrame::Reference ||
               frame == FE::assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(FE::GlobalIndex /*cell_id*/,
                            FE::assembly::CoordinateFrame frame,
                            std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        switch (frame) {
            case FE::assembly::CoordinateFrame::Active:
            case FE::assembly::CoordinateFrame::Current:
                coords = current_nodes_;
                return;
            case FE::assembly::CoordinateFrame::Reference:
                coords = reference_nodes_;
                return;
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex face_id,
                                                   FE::GlobalIndex /*cell_id*/) const override
    {
        return expose_all_faces_ ? static_cast<FE::LocalIndex>(face_id) : 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return marker_;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback) const override
    {
        if (marker < 0 || marker == marker_) {
            const FE::GlobalIndex count = expose_all_faces_ ? 4 : 1;
            for (FE::GlobalIndex face = 0; face < count; ++face) {
                callback(face, 0);
            }
        }
    }

    void forEachInteriorFace(std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] const std::array<FE::Real, 3>& referenceNodeCoordinates(
        FE::GlobalIndex node_id) const
    {
        return reference_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void setCurrentNodeCoordinates(FE::GlobalIndex node_id,
                                   std::array<FE::Real, 3> coords)
    {
        current_nodes_.at(static_cast<std::size_t>(node_id)) = coords;
        ++geometry_revision_;
    }

private:
    int marker_{-1};
    bool expose_all_faces_{false};
    std::uint64_t geometry_revision_{1};
    std::vector<std::array<FE::Real, 3>> reference_nodes_{};
    std::vector<std::array<FE::Real, 3>> current_nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

std::shared_ptr<FE::spaces::FunctionSpace> makeVelocitySpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
}

std::shared_ptr<FE::spaces::FunctionSpace> makePressureSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
}

ns::IncompressibleNavierStokesVMSOptions baseNavierStokesOptions()
{
    ns::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.density = 1.0;
    opts.viscosity = 0.01;
    opts.enable_convection = true;
    opts.enable_vms = false;
    return opts;
}

FormExpr manufacturedScalarField()
{
    using namespace FE::forms;
    const auto x0 = component(currentCoordinate(), 0);
    const auto x1 = component(currentCoordinate(), 1);
    return x0 * x0 + FormExpr::constant(0.5) * x1 + t();
}

FormExpr constantVector3(FE::Real x, FE::Real y, FE::Real z)
{
    return FormExpr::asVector({
        FormExpr::constant(x),
        FormExpr::constant(y),
        FormExpr::constant(z),
    });
}

FormExpr movingBoundaryKinematicResidual(const FormExpr& physical_velocity,
                                         const FormExpr& test_scalar)
{
    using namespace FE::forms;

    return test_scalar * dot(physical_velocity - meshVelocity(), currentNormal()) *
           currentMeasure();
}

FormExpr fsiDisplacementCompatibilityResidual(const FormExpr& structural_displacement,
                                              const FormExpr& test_scalar)
{
    using namespace FE::forms;

    return test_scalar * dot(structural_displacement - meshDisplacement(), currentNormal()) *
           currentMeasure();
}

FormExpr fsiSurfaceTractionPowerResidual(const FormExpr& current_traction,
                                         const FormExpr& interface_velocity_test)
{
    using namespace FE::forms;

    return inner(current_traction, interface_velocity_test) * currentMeasure();
}

FormExpr referenceSurfaceMeasureMismatchProbe()
{
    using namespace FE::forms;

    return currentMeasure() - referenceMeasure() +
           dot(currentNormal() - referenceNormal(),
               currentNormal() - referenceNormal());
}

FE::dofs::DofMap createSingleTetraDenseDofMap(FE::LocalIndex n_dofs)
{
    FE::dofs::DofMap dof_map(1, n_dofs, n_dofs);
    std::vector<FE::GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (FE::LocalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

std::vector<FE::Real> constantScalarTetraCoefficients(FE::Real value)
{
    return std::vector<FE::Real>(4u, value);
}

std::vector<FE::Real> affineZScalarTetraCoefficients(FE::Real offset,
                                                     FE::Real scale)
{
    // Unit tetra nodal z-coordinates are {0, 0, 0, 1}.
    std::vector<FE::Real> coeffs(4u, offset);
    coeffs[3] = offset + scale;
    return coeffs;
}

std::vector<FE::Real> affineScalarTetraCoefficients(
    FE::Real offset,
    const std::array<FE::Real, 3>& gradient)
{
    // Unit tetra vertices are the origin and the three Cartesian unit
    // vectors, so these nodal values reproduce offset + gradient.x exactly.
    return {
        offset,
        offset + gradient[0],
        offset + gradient[1],
        offset + gradient[2],
    };
}

std::vector<FE::Real> constantVectorTetraCoefficients(FE::Real x,
                                                      FE::Real y,
                                                      FE::Real z)
{
    std::vector<FE::Real> coeffs(12u, 0.0);
    for (std::size_t node = 0; node < 4u; ++node) {
        coeffs[node] = x;
        coeffs[4u + node] = y;
        coeffs[8u + node] = z;
    }
    return coeffs;
}

std::vector<FE::Real> affineXVectorTetraCoefficients()
{
    // ProductSpace coefficients are component-major.  The unit tetra nodal
    // coordinates are x={0,1,0,0}, y={0,0,1,0}, z={0,0,0,1}.
    std::vector<FE::Real> coeffs(12u, 0.0);
    coeffs[0] = 0.0;
    coeffs[1] = 1.0;
    coeffs[2] = 0.0;
    coeffs[3] = 0.0;
    return coeffs;
}

std::shared_ptr<FE::assembly::CutIntegrationContext>
makeSingleTetraCutVolumeContext(
    int marker,
    std::vector<FE::geometry::CutQuadraturePoint> points)
{
    auto cut_context = std::make_shared<FE::assembly::CutIntegrationContext>();

    // A CutVolume free surface uses the embedded natural-traction boundary as
    // its absolute-pressure datum.  Keep this hand-built volume fixture
    // physically complete by publishing a positive interface rule for the
    // same marker, just as the production level-set lifecycle does.
    FE::interfaces::CutInterfaceDomainRequest interface_request;
    interface_request.source =
        FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
            "single_tetra_cut_volume", /*layout_revision=*/0u,
            /*value_revision=*/1u);
    interface_request.interface_marker = marker;
    interface_request.quadrature_order = 0;
    interface_request.interface_quadrature_order = 0;
    interface_request.volume_quadrature_order = 0;
    FE::interfaces::LevelSetInterfaceDomain interface_domain(
        interface_request);
    FE::interfaces::CutInterfaceFragment interface_fragment;
    interface_fragment.interface_marker = marker;
    interface_fragment.parent_cell = 0;
    interface_fragment.local_fragment_index = 0;
    interface_fragment.stable_id = 1;
    interface_fragment.kind =
        FE::interfaces::CutInterfaceFragmentKind::Polygon;
    interface_fragment.measure = FE::Real{0.125};
    interface_fragment.normal = {{0.0, 0.0, 1.0}};
    interface_fragment.quadrature_points.push_back(
        FE::interfaces::CutInterfaceQuadraturePoint{
            .point = {{0.25, 0.25, 0.25}},
            .parent_coordinate = {{0.25, 0.25, 0.25}},
            .normal = interface_fragment.normal,
            .weight = interface_fragment.measure,
        });
    interface_domain.addFragment(std::move(interface_fragment));
    cut_context->addGeneratedInterfaceDomain(
        interface_domain, FE::geometry::CutIntegrationSide::Negative);

    FE::geometry::CutQuadratureRule rule;
    rule.kind = FE::geometry::CutQuadratureKind::Volume;
    rule.side = FE::geometry::CutIntegrationSide::Negative;
    rule.parent_measure = FE::Real{1.0} / FE::Real{6.0};
    rule.frame = FE::geometry::CutGeometryFrame::Reference;
    rule.provenance.parent_entity = 0;
    rule.provenance.marker = marker;
    rule.provenance.embedded_geometry_id = "single_tetra_cut_volume";
    rule.provenance.cut_topology_id = "single_tetra_cut_volume";
    rule.provenance.cut_topology_revision = 1u;
    rule.provenance.source_value_revision = 1u;
    rule.points = std::move(points);
    for (const auto& point : rule.points) {
        rule.measure += point.weight;
    }
    rule.volume_fraction = rule.measure / rule.parent_measure;

    FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.cell = 0;
    metadata.parent_entity = 0;
    metadata.side = rule.side;
    metadata.volume_fraction = rule.volume_fraction;
    metadata.provenance_id = rule.provenance.embedded_geometry_id;
    metadata.cut_topology_id = rule.provenance.cut_topology_id;
    metadata.revision_key = rule.provenance.cut_topology_revision;
    metadata.cut_topology_revision = rule.provenance.cut_topology_revision;
    metadata.source_value_revision = 1u;

    cut_context->addGeneratedVolumeRule(marker, std::move(metadata), std::move(rule));
    return cut_context;
}

std::shared_ptr<FE::assembly::CutIntegrationContext>
makeSingleTetraFreeSurfaceCutContext(
    int marker,
    FE::FieldId level_set_field,
    std::array<FE::Real, 3> interface_normal = {0.0, 0.0, 1.0},
    FE::geometry::CutIntegrationSide retained_volume_side =
        FE::geometry::CutIntegrationSide::Negative)
{
    namespace interfaces = FE::interfaces;

    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        level_set_field,
        /*layout_revision=*/0u,
        /*value_revision=*/1u);
    request.interface_marker = marker;
    request.quadrature_order = 0;
    request.interface_quadrature_order = 0;
    request.volume_quadrature_order = 0;

    interfaces::LevelSetInterfaceDomain domain(request);

    interfaces::CutInterfaceFragment fragment;
    fragment.interface_marker = marker;
    fragment.parent_cell = 0;
    fragment.local_fragment_index = 0;
    fragment.stable_id = 10;
    fragment.kind = interfaces::CutInterfaceFragmentKind::Polygon;
    fragment.measure = FE::Real{0.038};
    fragment.normal = interface_normal;
    fragment.quadrature_points.push_back(interfaces::CutInterfaceQuadraturePoint{
        .point = {{0.20, 0.25, 0.15}},
        .parent_coordinate = {{0.20, 0.25, 0.15}},
        .normal = fragment.normal,
        .weight = FE::Real{0.020},
    });
    fragment.quadrature_points.push_back(interfaces::CutInterfaceQuadraturePoint{
        .point = {{0.35, 0.18, 0.22}},
        .parent_coordinate = {{0.35, 0.18, 0.22}},
        .normal = fragment.normal,
        .weight = FE::Real{0.018},
    });
    domain.addFragment(std::move(fragment));

    interfaces::CutInterfaceVolumeRegion negative_region;
    negative_region.interface_marker = marker;
    negative_region.parent_cell = 0;
    negative_region.local_region_index = 0;
    negative_region.stable_id = 11;
    negative_region.side = FE::geometry::CutIntegrationSide::Negative;
    negative_region.measure = FE::Real{1.0} / FE::Real{12.0};
    negative_region.parent_measure = FE::Real{1.0} / FE::Real{6.0};
    negative_region.volume_fraction =
        negative_region.measure / negative_region.parent_measure;
    negative_region.centroid = {{0.24, 0.18, 0.20}};
    negative_region.normal = interface_normal;
    domain.addVolumeRegion(std::move(negative_region));

    interfaces::CutInterfaceVolumeRegion positive_region;
    positive_region.interface_marker = marker;
    positive_region.parent_cell = 0;
    positive_region.local_region_index = 1;
    positive_region.stable_id = 12;
    positive_region.side = FE::geometry::CutIntegrationSide::Positive;
    positive_region.measure = FE::Real{1.0} / FE::Real{12.0};
    positive_region.parent_measure = FE::Real{1.0} / FE::Real{6.0};
    positive_region.volume_fraction =
        positive_region.measure / positive_region.parent_measure;
    positive_region.centroid = {{0.44, 0.20, 0.18}};
    positive_region.normal = interface_normal;
    domain.addVolumeRegion(std::move(positive_region));

    auto cut_context = std::make_shared<FE::assembly::CutIntegrationContext>();
    cut_context->addGeneratedInterfaceDomain(
        domain,
        retained_volume_side);

    return cut_context;
}

std::shared_ptr<FE::assembly::CutIntegrationContext>
makeSingleTetraContactLineCutContext(int interface_marker,
                                     int wall_marker,
                                     int contact_marker,
                                     FE::FieldId level_set_field,
                                     std::array<FE::Real, 3> boundary_normal =
                                         {0.0, 0.0, 1.0},
                                     std::array<FE::Real, 3> interface_normal =
                                         {0.0, 1.0, 0.0},
                                     std::array<FE::Real, 3> contact_point =
                                         {0.10, 0.20, 0.30},
                                     std::array<FE::Real, 3> contact_tangent =
                                         {1.0, 0.0, 0.0},
                                     FE::geometry::CutIntegrationSide
                                         retained_volume_side =
                                             FE::geometry::CutIntegrationSide::Negative,
                                     std::optional<FE::Real>
                                         active_boundary_measure = std::nullopt)
{
    namespace interfaces = FE::interfaces;

    auto cut_context =
        makeSingleTetraFreeSurfaceCutContext(
            interface_marker,
            level_set_field,
            interface_normal,
            retained_volume_side);

    interfaces::GeneratedInterfaceBoundaryIntersectionRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        level_set_field,
        /*layout_revision=*/0u,
        /*value_revision=*/1u);
    request.generated_domain_id = "free_surface";
    request.interface_marker = interface_marker;
    request.boundary_marker = wall_marker;
    request.intersection_marker = contact_marker;
    request.quadrature_order = 1;

    interfaces::GeneratedInterfaceBoundaryIntersectionDomain domain(request);
    interfaces::GeneratedInterfaceBoundaryIntersectionFragment fragment;
    fragment.interface_marker = interface_marker;
    fragment.boundary_marker = wall_marker;
    fragment.intersection_marker = contact_marker;
    fragment.parent_cell = 0;
    fragment.parent_face = 0;
    fragment.kind =
        interfaces::GeneratedInterfaceBoundaryIntersectionKind::Segment;
    fragment.measure = FE::Real{0.125};
    fragment.interface_normal = interface_normal;
    fragment.boundary_normal = boundary_normal;
    fragment.tangent = contact_tangent;
    fragment.quadrature_points.push_back(
        interfaces::GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
            .point = contact_point,
            .parent_coordinate = contact_point,
            .interface_normal = fragment.interface_normal,
            .boundary_normal = fragment.boundary_normal,
            .tangent = fragment.tangent,
            .weight = fragment.measure,
            .reference_measure_factor = fragment.measure,
            .gradient_norm = FE::Real{1.0},
        });
    domain.addFragment(std::move(fragment));
    cut_context->addGeneratedInterfaceBoundaryIntersectionDomain(domain);

    if (active_boundary_measure.has_value()) {
        interfaces::GeneratedActiveBoundaryRequest active_request;
        active_request.source = interfaces::LevelSetInterfaceSource::fromField(
            level_set_field,
            /*layout_revision=*/0u,
            /*value_revision=*/1u);
        active_request.generated_domain_id = "free_surface";
        active_request.interface_marker = interface_marker;
        active_request.boundary_marker = wall_marker;
        active_request.side = retained_volume_side;
        active_request.quadrature_order = 1;

        interfaces::GeneratedActiveBoundaryDomain active_domain(
            active_request);
        if (*active_boundary_measure > FE::Real{0.0}) {
            interfaces::GeneratedActiveBoundaryFragment active_fragment;
            active_fragment.interface_marker = interface_marker;
            active_fragment.boundary_marker = wall_marker;
            active_fragment.parent_cell = 0;
            active_fragment.parent_face = 0;
            active_fragment.side = retained_volume_side;
            active_fragment.represented_implicit_geometry_mode =
                "LinearCorner";
            active_fragment.represented_implicit_quadrature_backend =
                "LinearCorner";
            active_fragment.represented_implicit_fallback_status = "None";
            active_fragment.boundary_normal = boundary_normal;
            active_fragment.measure = *active_boundary_measure;
            active_fragment.parent_measure = FE::Real{0.5};
            active_fragment.achieved_quadrature_order = 1;
            active_fragment.topology_id =
                "single-tetra-sharp-active-wall";
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
        cut_context->addGeneratedActiveBoundaryDomain(active_domain);
    }
    return cut_context;
}

FE::Real residualNorm(FE::systems::FESystem& system,
                      const FE::systems::SystemStateView& state,
                      std::string_view op)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();
    FE::systems::AssemblyRequest req;
    req.op = std::string(op);
    req.want_vector = true;
    const auto result = system.assemble(req, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    FE::Real norm2 = 0.0;
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        norm2 += residual[i] * residual[i];
    }
    return std::sqrt(norm2);
}

std::vector<FE::Real> residualVector(FE::systems::FESystem& system,
                                     const FE::systems::SystemStateView& state,
                                     std::string_view op);

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            int component,
                            FE::Real value);

struct ContactAngleAssemblyProbe {
    FE::GlobalIndex phi_offset{0};
    FE::GlobalIndex phi_dofs{0};
    FE::GlobalIndex vertex_one_phi_dof{0};
    bool phi_has_constraint{false};
    std::vector<FE::Real> phi_jacobian{};
};

struct DynamicContactAngleAssembly {
    FE::GlobalIndex total_dofs{0};
    FE::FieldId velocity_field{FE::INVALID_FIELD_ID};
    FE::FieldId pressure_field{FE::INVALID_FIELD_ID};
    FE::FieldId level_set_field{FE::INVALID_FIELD_ID};
    FE::GlobalIndex velocity_offset{0};
    FE::GlobalIndex velocity_dofs{0};
    FE::GlobalIndex pressure_offset{0};
    FE::GlobalIndex pressure_dofs{0};
    std::array<FE::GlobalIndex, 4> velocity_x_dofs{};
    std::array<FE::GlobalIndex, 4> level_set_dofs{};
    bool has_velocity_level_set_coupling{false};
    std::vector<FE::Real> solution{};
    std::vector<FE::Real> residual{};
    std::vector<FE::Real> conservative_pressure_residual{};
    std::vector<FE::Real> conservative_surface_energy_residual{};
    std::vector<FE::Real> conservative_balance_residual{};
    std::vector<FE::Real> pressure_representability_pair_jacobian{};
    std::vector<FE::Real> jacobian{};
};

DynamicContactAngleAssembly assembleDynamicContactAngleCase(
    FE::Real equilibrium_angle,
    FE::Real dynamic_angle,
    const std::array<FE::Real, 4>& velocity_x,
    bool include_dynamic_contact_angle,
    bool assemble_jacobian = false,
    std::array<FE::Real, 4> level_set_nodal_perturbation =
        std::array<FE::Real, 4>{0.0, 0.0, 0.0, 0.0},
    std::array<FE::Real, 3> generated_boundary_normal =
        std::array<FE::Real, 3>{0.0, 0.0, -1.0},
    FE::Real level_set_scale = FE::Real{1.0},
    FE::Real level_set_shift = FE::Real{0.0},
    int velocity_component = 0,
    ns::FreeSurfaceActiveDomain active_domain =
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
    ns::FreeSurfaceSurfaceTensionForm surface_tension_form =
        ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
    FE::Real liquid_pressure = FE::Real{0.0},
    FE::Real external_pressure = FE::Real{0.0})
{
    constexpr int interface_marker = 167;
    constexpr int wall_marker = 57;
    constexpr FE::Real gamma = 0.8;
    constexpr FE::Real mobility = 0.5;
    constexpr FE::Real slip_length = 0.2;
    constexpr FE::Real contact_x = 0.2;

    const auto mesh =
        std::make_shared<SingleTetraBoundaryMeshAccess>(wall_marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.velocity_dirichlet.push_back(
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = wall_marker,
            .value = {0.0, 0.0, 0.0},
            .active_components = {false, false, true},
        });

    auto free_surface =
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation =
                ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_dynamic_contact",
            .active_domain = active_domain,
            .active_domain_method =
                ns::FreeSurfaceActiveDomainMethod::CutVolume,
            .active_domain_smoothing_width = 0.25,
            .external_pressure = external_pressure,
            .surface_tension = gamma,
            .surface_tension_form = surface_tension_form,
            .curvature = 0.0,
            .use_level_set_curvature = false,
            .small_cut_aggregation = false,
        };
    if (include_dynamic_contact_angle) {
        free_surface.contact_lines.push_back(dynamicRenEContactLine(
            wall_marker,
            equilibrium_angle,
            {0.0, 0.0, -1.0},
            mobility,
            slip_length));
    }
    opts.free_surface.push_back(std::move(free_surface));

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi_dynamic_contact",
        .space = p_space,
        .components = 1,
    });
    system.addOperator("equations");
    const auto phi_state =
        FE::forms::StateField(phi, *p_space, "phi_dynamic_owner");
    const auto eta =
        FE::forms::TestField(phi, *p_space, "eta_dynamic_owner");
    (void)FE::systems::installFormulation(
        system,
        "equations",
        {phi},
        (FE::forms::dt(phi_state) * eta).dx());
    system.gaugeRegistry().addAnchoring(FE::gauge::AnchoringEvidence{
        .field = phi,
        .component = -1,
        .region = -1,
        .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
        .verdict = FE::gauge::AnchoringVerdict::Anchored,
        .source =
            "Transient level-set owner in dynamic-contact assembly fixture",
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    const auto velocity = system.findFieldByName("u");
    const auto pressure = system.findFieldByName("p");
    const int contact_marker =
        stableContactLineMarker(phi, interface_marker, wall_marker);
    std::array<FE::Real, 3> level_set_gradient{
        std::sin(dynamic_angle), 0.0, std::cos(dynamic_angle)};
    const auto retained_volume_side =
        active_domain == ns::FreeSurfaceActiveDomain::LevelSetPositive
            ? FE::geometry::CutIntegrationSide::Positive
            : FE::geometry::CutIntegrationSide::Negative;
    if (active_domain == ns::FreeSurfaceActiveDomain::LevelSetPositive) {
        for (auto& component : level_set_gradient) {
            component = -component;
        }
    }
    const auto wall_gradient_magnitude = std::abs(level_set_gradient[0]);
    const auto wet_limit = wall_gradient_magnitude > FE::Real{1.0e-12}
        ? contact_x - level_set_shift / wall_gradient_magnitude
        : contact_x;
    const auto clipped_wet_limit =
        std::clamp(wet_limit, FE::Real{0.0}, FE::Real{1.0});
    const auto active_boundary_measure =
        clipped_wet_limit -
        FE::Real{0.5} * clipped_wet_limit * clipped_wet_limit;
    system.setCutIntegrationContext(makeSingleTetraContactLineCutContext(
        interface_marker,
        wall_marker,
        contact_marker,
        phi,
        generated_boundary_normal,
        level_set_gradient,
        {contact_x, 0.2, 0.0},
        {0.0, 1.0, 0.0},
        retained_volume_side,
        active_boundary_measure));
    system.setup({}, makeSingleTetraSetupInputs());

    DynamicContactAngleAssembly out;
    out.total_dofs = system.dofHandler().getNumDofs();
    out.velocity_field = velocity;
    out.pressure_field = pressure;
    out.level_set_field = phi;
    out.velocity_offset = system.fieldDofOffset(velocity);
    out.velocity_dofs = system.fieldDofHandler(velocity).getNumDofs();
    out.pressure_offset = system.fieldDofOffset(pressure);
    out.pressure_dofs = system.fieldDofHandler(pressure).getNumDofs();
    out.solution.assign(static_cast<std::size_t>(out.total_dofs), 0.0);

    const std::array<FE::Real, 3> scaled_level_set_gradient{
        level_set_scale * level_set_gradient[0],
        level_set_scale * level_set_gradient[1],
        level_set_scale * level_set_gradient[2]};
    auto phi_values = affineScalarTetraCoefficients(
        -scaled_level_set_gradient[0] * contact_x +
            level_set_scale * level_set_shift,
        scaled_level_set_gradient);
    const auto* velocity_entity_map =
        system.fieldDofHandler(velocity).getEntityDofMap();
    if (velocity_entity_map == nullptr) {
        throw std::runtime_error(
            "dynamic contact-angle test velocity has no entity DOF map");
    }
    const auto velocity_offset = system.fieldDofOffset(velocity);
    const auto* phi_entity_map =
        system.fieldDofHandler(phi).getEntityDofMap();
    if (phi_entity_map == nullptr) {
        throw std::runtime_error(
            "dynamic contact-angle test level set has no entity DOF map");
    }
    const auto phi_offset = system.fieldDofOffset(phi);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        phi_values[static_cast<std::size_t>(vertex)] +=
            level_set_nodal_perturbation[static_cast<std::size_t>(vertex)];
        setFieldComponentValue(
            out.solution,
            system,
            velocity,
            vertex,
            velocity_component,
            velocity_x[static_cast<std::size_t>(vertex)]);
        setFieldComponentValue(
            out.solution,
            system,
            phi,
            vertex,
            0,
            phi_values[static_cast<std::size_t>(vertex)]);
        setFieldComponentValue(
            out.solution,
            system,
            pressure,
            vertex,
            0,
            liquid_pressure);
        const auto vertex_dofs = velocity_entity_map->getVertexDofs(vertex);
        out.velocity_x_dofs[static_cast<std::size_t>(vertex)] =
            velocity_offset +
            vertex_dofs[static_cast<std::size_t>(velocity_component)];
        const auto phi_vertex_dofs = phi_entity_map->getVertexDofs(vertex);
        out.level_set_dofs[static_cast<std::size_t>(vertex)] =
            phi_offset + phi_vertex_dofs.front();
    }

    const std::vector<FE::Real> previous_solution = out.solution;
    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(out.solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    out.residual = residualVector(system, state, "equations");
    if (system.hasOperator(std::string(
            ns::FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_virtual_work))) {
        out.conservative_pressure_residual = residualVector(
            system,
            state,
            ns::FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_virtual_work);
        out.conservative_surface_energy_residual = residualVector(
            system,
            state,
            ns::FreeSurfaceConservativeBalanceDiagnosticOperators::
                surface_energy_virtual_work);
        out.conservative_balance_residual = residualVector(
            system,
            state,
            ns::FreeSurfaceConservativeBalanceDiagnosticOperators::
                conservative_balance);

        const auto pair_op = std::string(
            ns::FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_representability_pair);
        EXPECT_TRUE(system.hasOperator(pair_op));
        FE::assembly::DenseMatrixView pair_matrix(out.total_dofs);
        pair_matrix.zero();
        FE::systems::AssemblyRequest pair_request;
        pair_request.op = pair_op;
        pair_request.want_matrix = true;
        pair_request.want_vector = false;
        pair_request.suppress_constraint_inhomogeneity = true;
        const auto pair_result =
            system.assemble(pair_request, state, &pair_matrix, nullptr);
        EXPECT_TRUE(pair_result.success) << pair_result.error_message;

        std::vector<FE::GlobalIndex> constrained_dofs;
        system.constraints().forEach(
            [&constrained_dofs](
                const FE::constraints::AffineConstraints::ConstraintView&
                    line) {
                if (line.slave_dof >= 0) {
                    constrained_dofs.push_back(line.slave_dof);
                }
            });
        pair_matrix.zeroRows(
            constrained_dofs, /*set_diagonal=*/false);
        out.pressure_representability_pair_jacobian.assign(
            static_cast<std::size_t>(out.total_dofs * out.total_dofs),
            FE::Real{0.0});
        for (FE::GlobalIndex row = 0; row < out.total_dofs; ++row) {
            for (FE::GlobalIndex column = 0; column < out.total_dofs;
                 ++column) {
                out.pressure_representability_pair_jacobian
                    [static_cast<std::size_t>(
                        row * out.total_dofs + column)] =
                    pair_matrix.getMatrixEntry(row, column);
            }
        }
    }

    for (const auto& record : system.formulationRecords()) {
        if (record.operator_tag != "equations") {
            continue;
        }
        out.has_velocity_level_set_coupling =
            out.has_velocity_level_set_coupling ||
            std::find(record.block_couplings.begin(),
                      record.block_couplings.end(),
                      std::pair<FE::FieldId, FE::FieldId>{velocity, phi}) !=
                record.block_couplings.end();
    }

    if (assemble_jacobian) {
        FE::assembly::DenseMatrixView jacobian(out.total_dofs);
        jacobian.zero();
        FE::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_matrix = true;
        const auto result = system.assemble(request, state, &jacobian, nullptr);
        EXPECT_TRUE(result.success) << result.error_message;
        out.jacobian.assign(
            static_cast<std::size_t>(out.total_dofs * out.total_dofs),
            FE::Real{0.0});
        for (FE::GlobalIndex row = 0; row < out.total_dofs; ++row) {
            for (FE::GlobalIndex column = 0; column < out.total_dofs;
                 ++column) {
                out.jacobian[static_cast<std::size_t>(
                    row * out.total_dofs + column)] =
                    jacobian.getMatrixEntry(row, column);
            }
        }
    }
    return out;
}

std::vector<FE::Real> unfittedContactAngleResidualVector(
    ns::FreeSurfaceActiveDomain active_domain,
    FE::Real contact_angle_radians,
    const std::array<FE::Real, 3>& level_set_gradient,
    const std::array<FE::Real, 3>& outward_wall_normal =
        std::array<FE::Real, 3>{0.0, 0.0, 1.0},
    ContactAngleAssemblyProbe* assembly_probe = nullptr,
    bool include_contact_angle = true,
    bool include_transient_owner = true,
    std::optional<std::array<FE::Real, 3>> generated_interface_normal =
        std::nullopt,
    FE::Real surface_tension = FE::Real{0.0},
    ns::FreeSurfaceSurfaceTensionForm surface_tension_form =
        ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction)
{
    constexpr int interface_marker = 66;
    constexpr int wall_marker = 16;
    const auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(wall_marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    auto free_surface =
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi",
            .active_domain = active_domain,
            .surface_tension = surface_tension,
            .surface_tension_form = surface_tension_form,
            .use_level_set_curvature = false,
        };
    if (include_contact_angle) {
        free_surface.contact_lines.push_back(prescribedContactLine(
            wall_marker,
            contact_angle_radians,
            outward_wall_normal,
            3.0));
    }
    opts.free_surface.push_back(std::move(free_surface));

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });
    system.addOperator("equations");
    if (include_transient_owner) {
        const auto phi_state =
            FE::forms::StateField(phi, *p_space, "phi_equations_owner");
        const auto eta =
            FE::forms::TestField(phi, *p_space, "eta_equations_owner");
        (void)FE::systems::installFormulation(
            system,
            "equations",
            {phi},
            (FE::forms::dt(phi_state) * eta).dx());
    } else {
        const auto phi_state =
            FE::forms::StateField(phi, *p_space, "phi_gradient_owner");
        const auto eta =
            FE::forms::TestField(phi, *p_space, "eta_gradient_owner");
        (void)FE::systems::installFormulation(
            system,
            "equations",
            {phi},
            FE::forms::dot(FE::forms::grad(phi_state),
                           FE::forms::grad(eta))
                .dx());
    }
    const int contact_marker =
        stableContactLineMarker(phi, interface_marker, wall_marker);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setCutIntegrationContext(makeSingleTetraContactLineCutContext(
        interface_marker,
        wall_marker,
        contact_marker,
        phi,
        outward_wall_normal,
        generated_interface_normal.value_or(level_set_gradient)));
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto phi_values = affineScalarTetraCoefficients(
        FE::Real{0.0}, level_set_gradient);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(solution,
                               system,
                               phi,
                               vertex,
                               0,
                               phi_values[static_cast<std::size_t>(vertex)]);
    }

    const std::vector<FE::Real> previous_solution = solution;
    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    if (assembly_probe != nullptr) {
        const auto n = system.dofHandler().getNumDofs();
        FE::assembly::DenseMatrixView jacobian(n);
        jacobian.zero();
        FE::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_matrix = true;
        const auto result = system.assemble(request, state, &jacobian, nullptr);
        EXPECT_TRUE(result.success) << result.error_message;

        const auto offset = system.fieldDofOffset(phi);
        const auto n_phi = system.fieldDofHandler(phi).getNumDofs();
        assembly_probe->phi_offset = offset;
        assembly_probe->phi_dofs = n_phi;
        assembly_probe->phi_has_constraint = false;
        for (FE::GlobalIndex row = 0; row < n_phi; ++row) {
            if (system.constraints().isConstrained(offset + row)) {
                assembly_probe->phi_has_constraint = true;
                break;
            }
        }
        const auto* entity_map =
            system.fieldDofHandler(phi).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error(
                "contact-angle assembly probe has no phi entity DOF map");
        }
        const auto vertex_one_dofs = entity_map->getVertexDofs(1);
        if (vertex_one_dofs.empty()) {
            throw std::runtime_error(
                "contact-angle assembly probe has no vertex-one phi DOF");
        }
        assembly_probe->vertex_one_phi_dof = vertex_one_dofs.front();
        assembly_probe->phi_jacobian.assign(
            static_cast<std::size_t>(n_phi * n_phi), FE::Real{0.0});
        for (FE::GlobalIndex row = 0; row < n_phi; ++row) {
            for (FE::GlobalIndex column = 0; column < n_phi; ++column) {
                const auto value = jacobian.getMatrixEntry(
                    offset + row, offset + column);
                assembly_probe->phi_jacobian[static_cast<std::size_t>(
                    row * n_phi + column)] = value;
            }
        }
    }
    return residualVector(system, state, "equations");
}

std::vector<FE::Real> residualVector(FE::systems::FESystem& system,
                                     const FE::systems::SystemStateView& state,
                                     std::string_view op)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();
    FE::systems::AssemblyRequest req;
    req.op = std::string(op);
    req.want_vector = true;
    const auto result = system.assemble(req, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    std::vector<FE::Real> out(static_cast<std::size_t>(n), 0.0);
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = residual[i];
    }
    return out;
}

FE::Real vectorNorm(std::span<const FE::Real> values)
{
    FE::Real norm2 = 0.0;
    for (const auto value : values) {
        norm2 += value * value;
    }
    return std::sqrt(norm2);
}

std::vector<FE::Real> fittedFreeSurfaceResidualVector(FE::Real external_pressure,
                                                      FE::Real surface_tension,
                                                      FE::Real curvature,
                                                      FE::Real liquid_pressure = 0.0,
                                                      bool expose_all_faces = false)
{
    constexpr int marker = 32;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(
        marker, expose_all_faces);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = external_pressure,
        .surface_tension = surface_tension,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
        .curvature = curvature,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        0.0);
    const auto pressure = system.findFieldByName("p");
    if (pressure == FE::INVALID_FIELD_ID) {
        ADD_FAILURE() << "Navier--Stokes test helper did not register pressure";
        return {};
    }
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(
            solution, system, pressure, vertex, 0, liquid_pressure);
    }
    const std::vector<FE::Real> previous_solution = solution;
    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    return residualVector(system, state, "equations");
}

std::vector<FE::Real> unfittedFreeSurfaceResidualVector(FE::Real external_pressure,
                                                        FE::Real surface_tension,
                                                        FE::Real curvature,
                                                        ns::FreeSurfaceActiveDomain active_domain =
                                                            ns::FreeSurfaceActiveDomain::LevelSetNegative,
                                                        ns::FreeSurfaceSurfaceTensionForm surface_tension_form =
                                                            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
                                                        std::array<FE::Real, 3> generated_interface_normal =
                                                            {0.0, 0.0, 1.0},
                                                        std::array<FE::Real, 3> level_set_gradient =
                                                            {0.0, 0.0, 1.0})
{
    constexpr int interface_marker = 146;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.enable_vms = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = active_domain,
        .external_pressure = external_pressure,
        .surface_tension = surface_tension,
        .surface_tension_form = surface_tension_form,
        .curvature = curvature,
        .use_level_set_curvature = false,
    });

    FE::systems::FESystem system(mesh);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setCutIntegrationContext(
        makeSingleTetraFreeSurfaceCutContext(
            interface_marker, phi_field, generated_interface_normal));
    system.setup({}, makeSingleTetraSetupInputs());
    system.setPrescribedFieldCoefficients(
        phi_field,
        affineScalarTetraCoefficients(FE::Real{-0.25}, level_set_gradient));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        0.0);
    const std::vector<FE::Real> previous_solution = solution;
    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    return residualVector(system, state, "equations");
}

std::vector<FE::Real> unfittedFreeSurfaceCurvatureFieldResidualVector(
    FE::Real external_pressure,
    FE::Real surface_tension,
    FE::Real curvature,
    ns::FreeSurfaceActiveDomain active_domain =
        ns::FreeSurfaceActiveDomain::LevelSetNegative)
{
    constexpr int interface_marker = 149;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.enable_vms = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = active_domain,
        .external_pressure = external_pressure,
        .surface_tension = surface_tension,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
        .curvature_field_name = "kappa_projected",
    });

    FE::systems::FESystem system(mesh);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    const auto kappa_field = system.addField(FE::systems::FieldSpec{
        .name = "kappa_projected",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setCutIntegrationContext(
        makeSingleTetraFreeSurfaceCutContext(interface_marker, phi_field));
    system.setup({}, makeSingleTetraSetupInputs());
    system.setPrescribedFieldCoefficients(
        phi_field,
        affineZScalarTetraCoefficients(FE::Real{-0.25}, FE::Real{1.0}));
    system.setPrescribedFieldCoefficients(
        kappa_field,
        constantScalarTetraCoefficients(curvature));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        0.0);
    const std::vector<FE::Real> previous_solution = solution;
    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    return residualVector(system, state, "equations");
}

FE::assembly::DenseVectorView assembleMovingDomainScalarResidual(
    const FE::assembly::IMeshAccess& mesh,
    const FE::spaces::FunctionSpace& scalar_space,
    FE::dofs::DofMap& scalar_dof_map,
    const FE::spaces::FunctionSpace* mesh_velocity_space,
    const FE::dofs::DofMap* mesh_velocity_dof_map,
    const FormExpr& residual_integrand,
    const std::vector<FE::Real>& current_solution,
    std::span<const FE::Real> prescribed_mesh_velocity = {})
{
    using namespace FE::forms;

    FE::forms::FormCompiler compiler;
    const auto form = residual_integrand.dx();
    auto ir = compiler.compileResidual(form);
    FE::forms::NonlinearFormKernel kernel(std::move(ir), FE::forms::ADMode::Forward);

    FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(scalar_dof_map);
    if (mesh_velocity_space != nullptr && mesh_velocity_dof_map != nullptr) {
        const std::array<FE::assembly::FieldSolutionAccess, 1> field_access = {{
            FE::assembly::FieldSolutionAccess{
                .field = kMeshVelocityField,
                .space = mesh_velocity_space,
                .dof_map = mesh_velocity_dof_map,
                .dof_offset = 0,
                .coefficient_source =
                    FE::assembly::FieldSolutionAccess::CoefficientSource::PrescribedData,
                .prescribed_coefficients = prescribed_mesh_velocity,
                .prescribed_revision = 1,
            },
        }};
        assembler.setFieldSolutionAccess(field_access);
        assembler.setMeshMotionFieldAccess(FE::assembly::MeshMotionFieldAccess{
            .mesh_velocity = kMeshVelocityField,
        });
    }
    assembler.setCurrentSolution(current_solution);

    FE::assembly::DenseVectorView residual(static_cast<FE::GlobalIndex>(scalar_dof_map.getNumDofs()));
    residual.zero();
    (void)assembler.assembleVector(mesh, scalar_space, kernel, residual);
    return residual;
}

FE::Real fieldComponentValue(const std::vector<FE::Real>& solution,
                             const FE::systems::FESystem& system,
                             FE::FieldId field,
                             FE::GlobalIndex vertex,
                             int component)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("fieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("fieldComponentValue: component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        dofs[static_cast<std::size_t>(component)] + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("fieldComponentValue: DOF index is out of range");
    }
    return solution[index];
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
        throw std::runtime_error("setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("setFieldComponentValue: component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        dofs[static_cast<std::size_t>(component)] + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

void updateBoundaryMeshCurrentCoordinates(SingleTetraBoundaryMeshAccess& mesh,
                                          const FE::systems::FESystem& system,
                                          FE::FieldId displacement,
                                          const std::vector<FE::Real>& solution)
{
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        auto coords = mesh.referenceNodeCoordinates(vertex);
        for (int component = 0; component < 3; ++component) {
            coords[static_cast<std::size_t>(component)] +=
                fieldComponentValue(solution, system, displacement, vertex, component);
        }
        mesh.setCurrentNodeCoordinates(vertex, coords);
    }
}

std::vector<FE::Real> assembleOperatorResidualWithCurrentMesh(
    FE::systems::FESystem& system,
    SingleTetraBoundaryMeshAccess& mesh,
    FE::FieldId displacement,
    const std::vector<FE::Real>& solution,
    std::string_view op)
{
    updateBoundaryMeshCurrentCoordinates(mesh, system, displacement, solution);

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(solution);

    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();

    FE::systems::AssemblyRequest req;
    req.op = std::string(op);
    req.want_vector = true;
    const auto result = system.assemble(req, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    std::vector<FE::Real> out(static_cast<std::size_t>(n), 0.0);
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = residual.getVectorEntry(i);
    }
    return out;
}

void expectOperatorJacobianMatchesMovingBoundaryFD(
    FE::systems::FESystem& system,
    SingleTetraBoundaryMeshAccess& mesh,
    FE::FieldId displacement,
    const std::vector<FE::Real>& base_solution,
    std::string_view op,
    FE::Real eps,
    FE::Real rtol,
    FE::Real atol)
{
    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(static_cast<FE::GlobalIndex>(base_solution.size()), n);

    updateBoundaryMeshCurrentCoordinates(mesh, system, displacement, base_solution);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(base_solution);

    FE::assembly::DenseMatrixView jacobian(n);
    jacobian.zero();
    {
        FE::systems::AssemblyRequest req;
        req.op = std::string(op);
        req.want_matrix = true;
        const auto result = system.assemble(req, state, &jacobian, nullptr);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    for (FE::GlobalIndex col = 0; col < n; ++col) {
        std::vector<FE::Real> plus = base_solution;
        std::vector<FE::Real> minus = base_solution;
        plus[static_cast<std::size_t>(col)] += eps;
        minus[static_cast<std::size_t>(col)] -= eps;

        const auto r_plus =
            assembleOperatorResidualWithCurrentMesh(system, mesh, displacement, plus, op);
        const auto r_minus =
            assembleOperatorResidualWithCurrentMesh(system, mesh, displacement, minus, op);

        for (FE::GlobalIndex row = 0; row < n; ++row) {
            const FE::Real fd =
                (r_plus[static_cast<std::size_t>(row)] -
                 r_minus[static_cast<std::size_t>(row)]) /
                (FE::Real(2.0) * eps);
            const FE::Real actual = jacobian.getMatrixEntry(row, col);
            const FE::Real tol = atol + rtol * std::max<FE::Real>(1.0, std::abs(fd));
            SCOPED_TRACE(::testing::Message() << "row=" << row << ", col=" << col);
            EXPECT_NEAR(actual, fd, tol);
        }
    }
}

} // namespace

TEST(MovingDomainPhysics, NavierStokesALEDisabledDoesNotConsumeMovingDomainTerminals)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_FALSE(system.hasField(opts.mesh_velocity_field_name));
    EXPECT_FALSE(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity).has_value());
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CurrentMeasure));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CurrentNormal));
}

TEST(MovingDomainPhysics, NavierStokesBodyForceFieldRegistersPrescribedSource)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.body_force_field_name = "ManufacturedSource";
    opts.enable_vms = true;

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto source = system.findFieldByName("ManufacturedSource");
    ASSERT_NE(source, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(source).source_kind,
              FE::systems::FieldSourceKind::PrescribedData);
    EXPECT_TRUE(formulationRecordsContainFieldExprType(
        system,
        FormExprType::StateField,
        source));
}

TEST(MovingDomainPhysics, PrescribedBodyForceFieldSyncsFromMeshPointData)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = makeRegistryQuadMesh();
    const auto handle = MeshFields::attach_field(
        mesh->local_mesh(),
        EntityKind::Vertex,
        "ManufacturedSource",
        FieldScalarType::Float64,
        2);
    auto* values = MeshFields::field_data_as<real_t>(mesh->local_mesh(), handle);
    ASSERT_NE(values, nullptr);
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
        values[2u * vertex] = static_cast<real_t>(1.0 + vertex);
        values[2u * vertex + 1u] = static_cast<real_t>(10.0 + vertex);
    }

    auto u_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/2);
    auto p_space =
        FE::spaces::SpaceFactory::create_h1(FE::ElementType::Quad4, /*order=*/1);
    auto opts = baseNavierStokesOptions();
    opts.body_force_field_name = "ManufacturedSource";

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    ASSERT_NO_THROW(system.setup());

    const auto source = system.findFieldByName("ManufacturedSource");
    ASSERT_NE(source, FE::INVALID_FIELD_ID);
    EXPECT_GT(system.syncPrescribedVertexFieldsFromMeshFields(), 0u);

    const auto coefficients = system.prescribedFieldCoefficients(source);
    const auto* entity_map = system.fieldDofHandler(source).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(mesh->n_vertices());
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_GE(dofs.size(), 2u);
        EXPECT_DOUBLE_EQ(coefficients[static_cast<std::size_t>(dofs[0])],
                         1.0 + static_cast<double>(vertex));
        EXPECT_DOUBLE_EQ(coefficients[static_cast<std::size_t>(dofs[1])],
                         10.0 + static_cast<double>(vertex));
    }
#endif
}

TEST(MovingDomainPhysics, MovingMeshTangentPathDefaultsToSymbolicRequired)
{
    EXPECT_EQ(mm::HarmonicMeshMotionOptions{}.tangent_path,
              FE::forms::GeometryTangentPath::SymbolicRequired);
    EXPECT_EQ(mm::PseudoElasticMeshMotionOptions{}.tangent_path,
              FE::forms::GeometryTangentPath::SymbolicRequired);
    EXPECT_EQ(ns::IncompressibleNavierStokesVMSOptions{}.moving_mesh_tangent_path,
              FE::forms::GeometryTangentPath::SymbolicRequired);
}

TEST(MovingDomainPhysics, UnfittedFreeSurfaceActiveDomainDefaultsToInactiveCutVolume)
{
    const ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary free_surface{};
    EXPECT_EQ(free_surface.active_domain, ns::FreeSurfaceActiveDomain::None);
    EXPECT_EQ(free_surface.active_domain_method,
              ns::FreeSurfaceActiveDomainMethod::CutVolume);
}

TEST(MovingDomainPhysics, NavierStokesOperatorTagOwnsMainResidual)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.operator_tag = "coupled_fluid_system";

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(system.hasOperator("coupled_fluid_system"));
    EXPECT_FALSE(system.hasOperator("equations"));
    ASSERT_FALSE(system.formulationRecords().empty());
    for (const auto& record : system.formulationRecords()) {
        EXPECT_EQ(record.operator_tag, "coupled_fluid_system");
    }
}

TEST(MovingDomainPhysics, MeshMotionRegistryTranslatesHarmonicSmoothingEquation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = makeRegistryQuadMesh();

    EquationModuleInput input{};
    input.equation_type = "mesh_motion";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Model"] = ParameterValue{true, "Harmonic"};
    input.equation_params["Field_name"] = ParameterValue{true, "mesh_displacement"};
    input.equation_params["Operator_tag"] = ParameterValue{true, "equations"};
    input.equation_params["Kappa"] = ParameterValue{true, "2.5"};
    input.equation_params["Moving_mesh_tangent_path"] =
        ParameterValue{true, "symbolic"};

    BoundaryConditionInput wall{};
    wall.name = "wall";
    wall.boundary_marker = 4;
    wall.params["Type"] = ParameterValue{true, "Dirichlet"};
    wall.params["Value"] = ParameterValue{true, "0.0"};
    input.boundary_conditions.push_back(std::move(wall));

    FE::systems::FESystem system(mesh);
    system.addOperator("equations");
    auto module = EquationModuleRegistry::instance().create("mesh_motion", input, system);

    ASSERT_TRUE(module);
    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    ASSERT_NO_THROW(system.setup());
#endif
}

TEST(MovingDomainPhysics, MeshMotionRegistryTranslatesPseudoElasticSmoothingEquation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = makeRegistryQuadMesh();

    EquationModuleInput input{};
    input.equation_type = "pseudo_elastic_mesh_motion";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Field_name"] = ParameterValue{true, "mesh_displacement"};
    input.equation_params["Operator_tag"] = ParameterValue{true, "equations"};
    input.equation_params["Lambda_mesh"] = ParameterValue{true, "3.0"};
    input.equation_params["Mu_mesh"] = ParameterValue{true, "1.5"};

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create(
        "pseudo_elastic_mesh_motion", input, system);

    ASSERT_TRUE(module);
    ASSERT_NE(system.findFieldByName("mesh_displacement"), FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::SymmetricPart));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Trace));
    ASSERT_NO_THROW(system.setup());
#endif
}

TEST(MovingDomainPhysics, NavierStokesALEEnabledRegistersMeshVelocityAndConsumesMeshVelocity)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.mesh_velocity_field_name = "mesh_velocity";

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity), mesh_velocity_id);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).source_kind,
              FE::systems::FieldSourceKind::PrescribedData);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity_id));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 16);
    EXPECT_EQ(system.fieldMap().numFields(), 2u);
    ASSERT_NE(system.blockMap(), nullptr);
    EXPECT_EQ(system.blockMap()->numBlocks(), 2u);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceAddsBoundaryResidual)
{
    constexpr int marker = 31;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 2.0,
        .surface_tension = 0.5,
        .curvature = 1.25,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Normal));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
}

TEST(MovingDomainPhysics, ExternalPressureAndCapillaryTractionsHaveExpectedSign)
{
    const auto external_pressure =
        fittedFreeSurfaceResidualVector(/*external_pressure=*/1.0,
                                        /*surface_tension=*/0.0,
                                        /*curvature=*/0.0);
    const auto surface_tension =
        fittedFreeSurfaceResidualVector(/*external_pressure=*/0.0,
                                        /*surface_tension=*/1.0,
                                        /*curvature=*/1.0);

    ASSERT_EQ(external_pressure.size(), surface_tension.size());
    EXPECT_GT(vectorNorm(external_pressure), 1.0e-14);
    for (std::size_t i = 0; i < external_pressure.size(); ++i) {
        EXPECT_NEAR(external_pressure[i], surface_tension[i], 1.0e-12);
    }
}

TEST(MovingDomainPhysics, StaticYoungLaplacePressureStateHasZeroResidual)
{
    constexpr FE::Real gamma = 0.072;
    constexpr FE::Real sphere_radius = 0.60;
    constexpr FE::Real curvature = 2.0 / sphere_radius;
    constexpr FE::Real external_pressure = 0.031;
    constexpr FE::Real liquid_pressure =
        external_pressure + gamma * curvature;

    // Expose all four tetrahedron faces as the same free surface. This makes
    // the weak constant-pressure volume term and the complete boundary
    // traction an exact discrete divergence-theorem pair, so this evaluates
    // an actual Young--Laplace equilibrium rather than cancellation between
    // two prescribed boundary loads at p=0.
    const auto equilibrium = fittedFreeSurfaceResidualVector(
        external_pressure,
        gamma,
        curvature,
        liquid_pressure,
        /*expose_all_faces=*/true);
    EXPECT_LT(vectorNorm(equilibrium), 1.0e-12);

    const auto wrong_jump = fittedFreeSurfaceResidualVector(
        external_pressure,
        gamma,
        curvature,
        external_pressure - gamma * curvature,
        /*expose_all_faces=*/true);
    EXPECT_GT(vectorNorm(wrong_jump), 1.0e-8);
}

TEST(MovingDomainPhysics, UnfittedExternalPressureAndCapillaryTractionSignsAgree)
{
    constexpr FE::Real gamma = 0.072;
    constexpr FE::Real circle_radius = 0.45;
    constexpr FE::Real curvature = 1.0 / circle_radius;

    const auto external_pressure =
        unfittedFreeSurfaceResidualVector(/*external_pressure=*/gamma * curvature,
                                          /*surface_tension=*/0.0,
                                          /*curvature=*/0.0);
    const auto surface_tension =
        unfittedFreeSurfaceResidualVector(/*external_pressure=*/0.0,
                                          /*surface_tension=*/gamma,
                                          /*curvature=*/curvature);
    ASSERT_EQ(external_pressure.size(), surface_tension.size());
    EXPECT_GT(vectorNorm(external_pressure), 1.0e-14);
    for (std::size_t i = 0; i < external_pressure.size(); ++i) {
        EXPECT_NEAR(external_pressure[i], surface_tension[i], 1.0e-12);
    }
}

TEST(MovingDomainPhysics,
     UnfittedSurfaceTensionSuppliedCurvatureIsLevelSetNormalSigned)
{
    constexpr FE::Real gamma = 0.072;
    constexpr FE::Real circle_radius = 0.45;
    constexpr FE::Real curvature = 1.0 / circle_radius;

    const auto negative_active_capillary =
        unfittedFreeSurfaceResidualVector(
            /*external_pressure=*/0.0,
            /*surface_tension=*/gamma,
            /*curvature=*/curvature,
            ns::FreeSurfaceActiveDomain::LevelSetNegative);
    const auto positive_active_capillary =
        unfittedFreeSurfaceResidualVector(
            /*external_pressure=*/0.0,
            /*surface_tension=*/gamma,
            /*curvature=*/curvature,
            ns::FreeSurfaceActiveDomain::LevelSetPositive);

    ASSERT_EQ(negative_active_capillary.size(),
              positive_active_capillary.size());
    EXPECT_GT(vectorNorm(negative_active_capillary), 1.0e-14);
    for (std::size_t i = 0; i < negative_active_capillary.size(); ++i) {
        EXPECT_NEAR(negative_active_capillary[i],
                    positive_active_capillary[i],
                    1.0e-12);
    }

    const auto negative_active_balanced =
        unfittedFreeSurfaceResidualVector(
            /*external_pressure=*/-gamma * curvature,
            /*surface_tension=*/gamma,
            /*curvature=*/curvature,
            ns::FreeSurfaceActiveDomain::LevelSetNegative);
    const auto positive_active_balanced =
        unfittedFreeSurfaceResidualVector(
            /*external_pressure=*/gamma * curvature,
            /*surface_tension=*/gamma,
            /*curvature=*/curvature,
            ns::FreeSurfaceActiveDomain::LevelSetPositive);

    EXPECT_LT(vectorNorm(negative_active_balanced), 1.0e-12);
    EXPECT_LT(vectorNorm(positive_active_balanced), 1.0e-12);
}

TEST(MovingDomainPhysics,
     UnfittedSurfaceStressUsesGeneratedGeometryAndIsOrientationInvariant)
{
    constexpr FE::Real gamma = 0.072;
    const auto negative_active = unfittedFreeSurfaceResidualVector(
        /*external_pressure=*/0.0,
        gamma,
        /*curvature_is_ignored=*/1.25,
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);
    const auto positive_active = unfittedFreeSurfaceResidualVector(
        /*external_pressure=*/0.0,
        gamma,
        /*curvature_is_ignored=*/-17.0,
        ns::FreeSurfaceActiveDomain::LevelSetPositive,
        ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);

    ASSERT_EQ(negative_active.size(), positive_active.size());
    EXPECT_GT(vectorNorm(negative_active), 1.0e-14);
    for (std::size_t i = 0; i < negative_active.size(); ++i) {
        EXPECT_NEAR(negative_active[i], positive_active[i], 1.0e-12)
            << "P=I-n*n must be invariant under the active-side normal flip";
    }

    const auto pressure_negative = unfittedFreeSurfaceResidualVector(
        /*external_pressure=*/0.031,
        /*surface_tension=*/0.0,
        /*curvature=*/0.0,
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);
    const auto pressure_positive = unfittedFreeSurfaceResidualVector(
        /*external_pressure=*/0.031,
        /*surface_tension=*/0.0,
        /*curvature=*/0.0,
        ns::FreeSurfaceActiveDomain::LevelSetPositive,
        ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);
    ASSERT_EQ(pressure_negative.size(), pressure_positive.size());
    EXPECT_GT(vectorNorm(pressure_negative), 1.0e-14);
    for (std::size_t i = 0; i < pressure_negative.size(); ++i) {
        EXPECT_NEAR(pressure_negative[i], -pressure_positive[i], 1.0e-12)
            << "external pressure must use the directed outward-liquid normal";
    }
}

TEST(MovingDomainPhysics,
     UnfittedPressureOnlySurfaceStressUsesGeneratedInterfaceNormal)
{
    constexpr FE::Real external_pressure = FE::Real{0.031};
    constexpr std::array<FE::Real, 3> x_normal{1.0, 0.0, 0.0};
    constexpr std::array<FE::Real, 3> z_normal{0.0, 0.0, 1.0};

    // Automatic selects SurfaceStress for an unfitted interface.  A literal
    // zero gamma must retain the generated normal used by the pressure work,
    // without leaving a zero-coefficient surface projector in the registered
    // form (and therefore in the JIT kernel).
    constexpr int interface_marker = 215;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation =
                ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_pressure_only_surface_stress",
            .active_domain =
                ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .external_pressure = external_pressure,
            .surface_tension = 0.0,
            .surface_tension_form =
                ns::FreeSurfaceSurfaceTensionForm::Automatic,
        });

    FE::systems::FESystem structural_system(mesh);
    structural_system.addField(FE::systems::FieldSpec{
        .name = "phi_pressure_only_surface_stress",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    ns::IncompressibleNavierStokesVMSModule structural_module(
        u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    structural_module.registerOn(structural_system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsInterfaceIntegralContainsExprType(
        structural_system, interface_marker, FormExprType::Normal));
    EXPECT_FALSE(formulationRecordsInterfaceIntegralContainsExprType(
        structural_system, interface_marker, FormExprType::Identity));
    EXPECT_FALSE(formulationRecordsInterfaceIntegralContainsExprType(
        structural_system, interface_marker, FormExprType::OuterProduct));
    EXPECT_FALSE(formulationRecordsInterfaceIntegralContainsExprType(
        structural_system, interface_marker, FormExprType::Gradient));
    EXPECT_NE(log_output.find("surface_tension_form=SurfaceStress"),
              std::string::npos);
    EXPECT_NE(log_output.find("normal_source=integration_rule_geometry"),
              std::string::npos);
    EXPECT_NE(log_output.find(
                  "surface_energy_form=omitted_literal_zero_gamma"),
              std::string::npos);
    EXPECT_NE(log_output.find(
                  "diagnostic=free_surface_pressure_only_surface_stress"),
              std::string::npos);
    EXPECT_EQ(log_output.find(
                  "diagnostic=free_surface_variational_surface_stress"),
              std::string::npos);

    const auto generated_x_live_z = unfittedFreeSurfaceResidualVector(
        external_pressure,
        /*surface_tension=*/0.0,
        /*curvature=*/0.0,
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        ns::FreeSurfaceSurfaceTensionForm::Automatic,
        x_normal,
        z_normal);
    const auto generated_x_live_x = unfittedFreeSurfaceResidualVector(
        external_pressure,
        /*surface_tension=*/0.0,
        /*curvature=*/0.0,
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        ns::FreeSurfaceSurfaceTensionForm::Automatic,
        x_normal,
        x_normal);
    const auto generated_z_live_z = unfittedFreeSurfaceResidualVector(
        external_pressure,
        /*surface_tension=*/0.0,
        /*curvature=*/0.0,
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        ns::FreeSurfaceSurfaceTensionForm::Automatic,
        z_normal,
        z_normal);

    ASSERT_EQ(generated_x_live_z.size(), generated_x_live_x.size());
    ASSERT_EQ(generated_x_live_z.size(), generated_z_live_z.size());
    FE::Real changed_geometry_norm2 = FE::Real{0.0};
    for (std::size_t i = 0; i < generated_x_live_z.size(); ++i) {
        EXPECT_NEAR(generated_x_live_z[i], generated_x_live_x[i], 1.0e-12)
            << "pressure traction must be independent of the separately "
               "evaluated Q1 level-set gradient";
        const auto difference =
            generated_x_live_z[i] - generated_z_live_z[i];
        changed_geometry_norm2 += difference * difference;
    }
    EXPECT_GT(std::sqrt(changed_geometry_norm2), 1.0e-12)
        << "pressure traction must follow the generated interface normal";
}

TEST(MovingDomainPhysics,
     UnfittedProjectedCurvatureRespondsToOpposingBoundaryLoad)
{
    constexpr FE::Real gamma = 0.0728;
    constexpr FE::Real droplet_radius = 0.52;
    constexpr FE::Real curvature = FE::Real{2.0} / droplet_radius;
    constexpr FE::Real laplace_pressure = gamma * curvature;

    const auto pressure_only =
        unfittedFreeSurfaceCurvatureFieldResidualVector(-laplace_pressure,
                                                        /*surface_tension=*/0.0,
                                                        /*curvature=*/0.0);
    const auto balanced =
        unfittedFreeSurfaceCurvatureFieldResidualVector(-laplace_pressure,
                                                        gamma,
                                                        curvature);
    const auto pressure_perturbed =
        unfittedFreeSurfaceCurvatureFieldResidualVector(
            -laplace_pressure * FE::Real{1.05},
            gamma,
            curvature);

    EXPECT_GT(vectorNorm(pressure_only), 1.0e-14);
    EXPECT_LT(vectorNorm(balanced), 1.0e-12);
    EXPECT_GT(vectorNorm(pressure_perturbed), vectorNorm(balanced) + 1.0e-8);
}

TEST(MovingDomainPhysics,
     UnfittedSurfaceTensionCurvatureFieldIsLevelSetNormalSigned)
{
    constexpr FE::Real gamma = 0.0728;
    constexpr FE::Real droplet_radius = 0.52;
    constexpr FE::Real curvature = FE::Real{2.0} / droplet_radius;

    const auto negative_active_capillary =
        unfittedFreeSurfaceCurvatureFieldResidualVector(
            /*external_pressure=*/0.0,
            /*surface_tension=*/gamma,
            /*curvature=*/curvature,
            ns::FreeSurfaceActiveDomain::LevelSetNegative);
    const auto positive_active_capillary =
        unfittedFreeSurfaceCurvatureFieldResidualVector(
            /*external_pressure=*/0.0,
            /*surface_tension=*/gamma,
            /*curvature=*/curvature,
            ns::FreeSurfaceActiveDomain::LevelSetPositive);

    ASSERT_EQ(negative_active_capillary.size(),
              positive_active_capillary.size());
    EXPECT_GT(vectorNorm(negative_active_capillary), 1.0e-14);
    for (std::size_t i = 0; i < negative_active_capillary.size(); ++i) {
        EXPECT_NEAR(negative_active_capillary[i],
                    positive_active_capillary[i],
                    1.0e-12);
    }

    const auto positive_active_balanced =
        unfittedFreeSurfaceCurvatureFieldResidualVector(
            /*external_pressure=*/gamma * curvature,
            /*surface_tension=*/gamma,
            /*curvature=*/curvature,
            ns::FreeSurfaceActiveDomain::LevelSetPositive);
    EXPECT_LT(vectorNorm(positive_active_balanced), 1.0e-12);
}

TEST(MovingDomainPhysics, StaticFlatWaterSurfaceWithGravityRemainsAtRest)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr int left_marker = 101;
    constexpr int right_marker = 102;
    constexpr int bottom_marker = 103;
    constexpr int free_surface_marker = 104;
    constexpr FE::GlobalIndex top_middle_vertex = 7;
    constexpr FE::Real density = 2.0;
    constexpr FE::Real gravity_y = -9.81;
    constexpr FE::Real surface_y = 1.0;
    constexpr FE::Real atmospheric_pressure = 0.0;

    auto mesh = makeOpenTankQuadMesh(left_marker,
                                     right_marker,
                                     bottom_marker,
                                     free_surface_marker,
                                     "free_surface");
    auto scalar_space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4,
        /*order=*/1);
    auto u_space = std::make_shared<FE::spaces::ProductSpace>(
        scalar_space,
        /*components=*/2);
    auto p_space = scalar_space;

    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.density = density;
    opts.viscosity = 1.0e-3;
    opts.body_force = {0.0, gravity_y, 0.0};
    opts.velocity_dirichlet = {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = left_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = right_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = bottom_marker,
        },
    };
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = free_surface_marker,
        .external_pressure = atmospheric_pressure,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    ASSERT_NO_THROW(system.setup());

    const auto u = system.findFieldByName(opts.velocity_field_name);
    const auto p = system.findFieldByName(opts.pressure_field_name);
    ASSERT_NE(u, FE::INVALID_FIELD_ID);
    ASSERT_NE(p, FE::INVALID_FIELD_ID);

    const auto* velocity_entity_map = system.fieldDofHandler(u).getEntityDofMap();
    ASSERT_NE(velocity_entity_map, nullptr);
    const auto top_middle_velocity_dofs =
        velocity_entity_map->getVertexDofs(top_middle_vertex);
    ASSERT_EQ(top_middle_velocity_dofs.size(), 2u);
    const auto velocity_offset = system.fieldDofOffset(u);
    for (const auto dof : top_middle_velocity_dofs) {
        EXPECT_FALSE(system.constraints().isConstrained(velocity_offset + dof));
    }

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        0.0);
    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(mesh->n_vertices());
         ++vertex) {
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        const auto pressure =
            atmospheric_pressure +
            density * gravity_y * (x[1] - surface_y);
        setFieldComponentValue(solution, system, p, vertex, 0, pressure);
    }
    const auto previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    const auto residual = residualVector(system, state, "equations");
    EXPECT_LT(vectorNorm(residual), 1.0e-10);
#endif
}

TEST(MovingDomainPhysics, FittedAndUnfittedFlatStaticFreeSurfaceAgree)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    ScopedEnvVar enable_shape_tangents(
        "SVMP_ENABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
        std::string("1"));

    constexpr int left_marker = 111;
    constexpr int right_marker = 112;
    constexpr int bottom_marker = 113;
    constexpr int free_surface_marker = 114;
    constexpr int interface_marker = 115;
    constexpr FE::GlobalIndex top_middle_vertex = 7;
    constexpr FE::Real density = 2.0;
    constexpr FE::Real gravity_y = -9.81;
    constexpr FE::Real bottom_y = -1.0;
    constexpr FE::Real interface_y = -0.5;
    constexpr FE::Real fitted_middle_y = -0.75;
    constexpr FE::Real external_pressure = 1.25;
    constexpr FE::Real expected_surface_length = 2.0;
    constexpr FE::Real expected_reference_surface_length = 4.0;
    constexpr FE::Real reference_to_physical_surface_scale = 0.5;

    auto scalar_space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4,
        /*order=*/1);
    auto u_space = std::make_shared<FE::spaces::ProductSpace>(
        scalar_space,
        /*components=*/2);

    auto fitted_mesh = makeOpenTankQuadMesh(left_marker,
                                            right_marker,
                                            bottom_marker,
                                            free_surface_marker,
                                            "free_surface",
                                            bottom_y,
                                            fitted_middle_y,
                                            interface_y);
    auto fitted_opts = baseNavierStokesOptions();
    fitted_opts.enable_convection = false;
    fitted_opts.density = density;
    fitted_opts.viscosity = 1.0e-3;
    fitted_opts.body_force = {0.0, gravity_y, 0.0};
    fitted_opts.velocity_dirichlet = {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = left_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = right_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = bottom_marker,
        },
    };
    fitted_opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::FittedALE,
            .boundary_marker = free_surface_marker,
            .external_pressure = external_pressure,
        });

    FE::systems::FESystem fitted_system(fitted_mesh);
    ns::IncompressibleNavierStokesVMSModule fitted_module(
        u_space,
        scalar_space,
        fitted_opts);
    fitted_module.registerOn(fitted_system);
    ASSERT_NO_THROW(fitted_system.setup());

    const auto fitted_u =
        fitted_system.findFieldByName(fitted_opts.velocity_field_name);
    const auto fitted_p =
        fitted_system.findFieldByName(fitted_opts.pressure_field_name);
    ASSERT_NE(fitted_u, FE::INVALID_FIELD_ID);
    ASSERT_NE(fitted_p, FE::INVALID_FIELD_ID);
    const auto* fitted_velocity_entity_map =
        fitted_system.fieldDofHandler(fitted_u).getEntityDofMap();
    ASSERT_NE(fitted_velocity_entity_map, nullptr);
    const auto top_middle_velocity_dofs =
        fitted_velocity_entity_map->getVertexDofs(top_middle_vertex);
    ASSERT_EQ(top_middle_velocity_dofs.size(), 2u);
    const auto fitted_velocity_offset = fitted_system.fieldDofOffset(fitted_u);
    for (const auto dof : top_middle_velocity_dofs) {
        EXPECT_FALSE(
            fitted_system.constraints().isConstrained(fitted_velocity_offset + dof));
    }

    std::vector<FE::Real> fitted_solution(
        static_cast<std::size_t>(fitted_system.dofHandler().getNumDofs()),
        0.0);
    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(fitted_mesh->n_vertices());
         ++vertex) {
        const auto x = fitted_system.meshAccess().getNodeCoordinates(vertex);
        const auto pressure =
            external_pressure +
            density * gravity_y * (x[1] - interface_y);
        setFieldComponentValue(
            fitted_solution,
            fitted_system,
            fitted_p,
            vertex,
            0,
            pressure);
    }
    const auto fitted_previous_solution = fitted_solution;
    FE::systems::SystemStateView fitted_state;
    fitted_state.dt = 1.0;
    fitted_state.u = std::span<const FE::Real>(fitted_solution);
    fitted_state.u_prev = std::span<const FE::Real>(fitted_previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto fitted_time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, fitted_state);
    fitted_state.time_integration = &fitted_time_context;
    const auto fitted_residual =
        residualVector(fitted_system, fitted_state, "equations");
    EXPECT_LT(vectorNorm(fitted_residual), 1.0e-10);

    auto background_mesh = makeOpenTankQuadMesh(left_marker,
                                                right_marker,
                                                bottom_marker,
                                                free_surface_marker,
                                                "outer_free_surface");
    FE::systems::FESystem unfitted_system(background_mesh);
    const auto phi = unfitted_system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(unfitted_system.setup());

    std::vector<FE::Real> unfitted_solution(
        static_cast<std::size_t>(unfitted_system.dofHandler().getNumDofs()),
        0.0);
    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(background_mesh->n_vertices());
         ++vertex) {
        const auto x = unfitted_system.meshAccess().getNodeCoordinates(vertex);
        setFieldComponentValue(
            unfitted_solution,
            unfitted_system,
            phi,
            vertex,
            0,
            x[1] - interface_y);
    }

    ls::LevelSetGeneratedInterfaceOptions interface_options{};
    interface_options.level_set_field_name = "phi";
    interface_options.domain_id = "flat_static_surface";
    interface_options.requested_interface_marker = interface_marker;
    interface_options.tolerance = 1.0e-12;

    ls::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated =
        lifecycle.build(unfitted_system, interface_options, unfitted_solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;
    EXPECT_EQ(generated.interface_marker, interface_marker);
    EXPECT_EQ(generated.summary.active_fragment_count, 2u);
    EXPECT_NEAR(generated.summary.measure,
                expected_reference_surface_length,
                1.0e-12);
    EXPECT_NEAR(reference_to_physical_surface_scale * generated.summary.measure,
                expected_surface_length,
                1.0e-12);
    EXPECT_NEAR(external_pressure * reference_to_physical_surface_scale *
                    generated.summary.measure,
                external_pressure * expected_surface_length,
                1.0e-12);

    for (const auto& fragment : generated.domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        EXPECT_NEAR(fragment.measure, 2.0, 1.0e-12);
        EXPECT_NEAR(reference_to_physical_surface_scale * fragment.measure,
                    1.0,
                    1.0e-12);
        EXPECT_NEAR(fragment.normal[0], 0.0, 1.0e-12);
        EXPECT_NEAR(fragment.normal[1], 1.0, 1.0e-12);
        EXPECT_NEAR(fragment.normal[2], 0.0, 1.0e-12);
    }

    auto unfitted_opts = baseNavierStokesOptions();
    unfitted_opts.enable_convection = false;
    unfitted_opts.density = density;
    unfitted_opts.viscosity = 1.0e-3;
    unfitted_opts.body_force = {0.0, gravity_y, 0.0};
    unfitted_opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi",
            .generated_interface_domain_id = "flat_static_surface",
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .external_pressure = external_pressure,
            .surface_tension_form =
                ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
        });
    ns::IncompressibleNavierStokesVMSModule unfitted_module(
        u_space,
        scalar_space,
        unfitted_opts);
    unfitted_module.registerOn(unfitted_system);
    const auto unfitted_u =
        unfitted_system.findFieldByName(unfitted_opts.velocity_field_name);
    const auto unfitted_p =
        unfitted_system.findFieldByName(unfitted_opts.pressure_field_name);
    ASSERT_NE(unfitted_u, FE::INVALID_FIELD_ID);
    ASSERT_NE(unfitted_p, FE::INVALID_FIELD_ID);
    EXPECT_TRUE(formulationRecordsContain(unfitted_system,
                                          FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(unfitted_system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(unfitted_system,
                                                        interface_marker));
    const auto& unfitted_operator =
        unfitted_system.operatorDefinition("equations");
    const auto has_phi_shape_tangent =
        std::any_of(unfitted_operator.interface_faces.begin(),
                    unfitted_operator.interface_faces.end(),
                    [&](const auto& term) {
                        return term.marker == interface_marker &&
                               term.trial_field == phi &&
                               (term.test_field == unfitted_u ||
                                term.test_field == unfitted_p);
                    });
    EXPECT_TRUE(has_phi_shape_tangent);
#endif
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceALEUsesCurrentBoundaryGeometry)
{
    constexpr int marker = 34;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 2.0,
        .surface_tension = 0.5,
        .curvature = 1.25,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
}

TEST(MovingDomainPhysics, NavierStokesRotatingFrameCoriolisAddsVelocityCoupling)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.rotating_frame_coriolis_enabled = true;
    opts.rotating_frame_angular_velocity = {0.0, 0.0, 2.5};

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CrossProduct));
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
}

TEST(MovingDomainPhysics, CurrentFaceGeometryMeanCurvatureTracksCurvedHexFace)
{
    auto basis = std::make_shared<FE::basis::LagrangeBasis>(FE::ElementType::Hex27, 2);
    std::vector<FE::math::Vector<FE::Real, 3>> nodes;
    nodes.reserve(basis->nodes().size());

    constexpr FE::Real radius = 2.0;
    for (const auto& xi : basis->nodes()) {
        const FE::Real x = xi[0];
        const FE::Real y = xi[1];
        const FE::Real zeta = xi[2];
        const FE::Real top_offset =
            ((FE::Real(1) + zeta) * (x * x + y * y)) / (FE::Real(4) * radius);
        nodes.push_back({x, y, zeta - top_offset});
    }

    FE::geometry::IsoparametricMapping mapping(basis, nodes);
    const auto quad = FE::quadrature::QuadratureFactory::create(FE::ElementType::Quad4, 2);
    const auto face = FE::geometry::evaluateFaceFrame(mapping,
                                                      FE::ElementType::Hex27,
                                                      /*local_face_id=*/1,
                                                      FE::ElementType::Quad4,
                                                      *quad);

    ASSERT_EQ(face.mean_curvatures.size(), quad->num_points());
    constexpr FE::Real gamma = 0.072;
    constexpr FE::Real external_pressure = 0.031;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto point = quad->points()[q];
        const FE::Real r2 = point[0] * point[0] + point[1] * point[1];
        const FE::Real grad2 = r2 / (radius * radius);
        const FE::Real expected =
            (FE::Real(2) + grad2) /
            (radius * std::pow(FE::Real(1) + grad2, FE::Real(1.5)));
        EXPECT_NEAR(face.mean_curvatures[q], expected, 2e-12);

        // With the fitted free-surface convention used by the
        // Navier--Stokes formulation, the liquid pressure that cancels the
        // current-geometry capillary traction is p_ext + gamma*kappa.
        const FE::Real liquid_pressure = external_pressure + gamma * expected;
        EXPECT_NEAR(liquid_pressure - external_pressure,
                    gamma * face.mean_curvatures[q],
                    2e-13);
        const FE::Real wrong_sign_pressure = external_pressure - gamma * expected;
        EXPECT_GT(std::abs((wrong_sign_pressure - external_pressure) -
                           gamma * face.mean_curvatures[q]),
                  1e-3);
    }
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceCanUseCurrentGeometryCurvature)
{
    constexpr int marker = 35;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .surface_tension = 0.5,
        .use_current_geometry_curvature = true,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeanCurvature));
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfacePenaltyKinematicsAddsBoundaryResidual)
{
    constexpr int marker = 33;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
        .kinematic_penalty = 12.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    system.setPrescribedFieldCoefficients(
        mesh_velocity_id,
        constantVectorTetraCoefficients(0.25, 0.5, 0.75));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    EXPECT_GT(residualNorm(system, state, "equations"), 0.0);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfacePenaltyKinematicsRequiresALE)
{
    constexpr int marker = 34;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = false;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
        .kinematic_penalty = 12.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfacePenaltyKinematicsRequiresPenalty)
{
    constexpr int marker = 35;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceNitscheKinematicsAddsBoundaryResidual)
{
    constexpr int marker = 36;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Nitsche,
        .kinematic_nitsche_gamma = 16.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::FacetArea));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    system.setPrescribedFieldCoefficients(
        mesh_velocity_id,
        constantVectorTetraCoefficients(0.25, 0.5, 0.75));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    EXPECT_GT(residualNorm(system, state, "equations"), 0.0);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceNitscheKinematicsRejectsNonPositiveGamma)
{
    constexpr int marker = 38;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Nitsche,
        .kinematic_nitsche_gamma = 0.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics,
     FittedFreeSurfaceNitschePoliciesAreBoundaryLocalAndOrderInvariant)
{
    constexpr int low_marker = 381;
    constexpr int high_marker = 382;
    using Boundary =
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;

    struct AssemblySnapshot {
        std::vector<FE::Real> residual;
        std::vector<FE::Real> jacobian;
    };

    struct GenericNitschePolicy {
        FE::Real gamma;
        bool symmetric;
        bool scale_with_p;
    };

    const auto assemble = [](std::vector<Boundary> boundaries,
                             bool add_generic_weak_velocity,
                             int active_marker,
                             GenericNitschePolicy generic_policy =
                                 {43.0, false, false}) {
        auto mesh =
            std::make_shared<SingleTetraBoundaryMeshAccess>(active_marker);
        auto u_space = makeVelocitySpace(mesh);
        auto p_space = makePressureSpace(mesh);
        auto opts = baseNavierStokesOptions();
        opts.enable_ale = true;
        opts.enable_convection = false;
        opts.mesh_velocity_field_name = "mesh_velocity";
        opts.free_surface = std::move(boundaries);

        // Deliberately distinct from either free-surface policy. These values
        // belong only to generic weak velocity conditions.
        opts.nitsche_gamma = generic_policy.gamma;
        opts.nitsche_symmetric = generic_policy.symmetric;
        opts.nitsche_scale_with_p = generic_policy.scale_with_p;
        if (add_generic_weak_velocity) {
            opts.velocity_dirichlet_weak.push_back(
                ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                    .boundary_marker = active_marker,
                    .value = {0.25, -0.5, 0.75},
                });
        }

        FE::systems::FESystem system(mesh);
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        module.registerOn(system);
        system.setup({}, makeSingleTetraSetupInputs());

        const auto mesh_velocity = system.findFieldByName("mesh_velocity");
        EXPECT_NE(mesh_velocity, FE::INVALID_FIELD_ID);
        system.setPrescribedFieldCoefficients(
            mesh_velocity,
            constantVectorTetraCoefficients(0.25, 0.5, 0.75));

        const auto dofs = system.dofHandler().getNumDofs();
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(dofs), FE::Real{0.0});
        const std::vector<FE::Real> previous_solution = solution;
        FE::systems::SystemStateView state;
        state.dt = 1.0;
        state.u = std::span<const FE::Real>(solution);
        state.u_prev = std::span<const FE::Real>(previous_solution);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto time_context = integrator.buildContext(
            /*max_time_derivative_order=*/1, state);
        state.time_integration = &time_context;

        FE::assembly::DenseMatrixView matrix(dofs);
        FE::assembly::DenseVectorView residual(dofs);
        matrix.zero();
        residual.zero();
        FE::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_matrix = true;
        request.want_vector = true;
        const auto result =
            system.assemble(request, state, &matrix, &residual);
        EXPECT_TRUE(result.success) << result.error_message;

        AssemblySnapshot snapshot;
        snapshot.residual.resize(static_cast<std::size_t>(dofs));
        snapshot.jacobian.resize(
            static_cast<std::size_t>(dofs * dofs));
        for (FE::GlobalIndex row = 0; row < dofs; ++row) {
            snapshot.residual[static_cast<std::size_t>(row)] = residual[row];
            for (FE::GlobalIndex column = 0; column < dofs; ++column) {
                snapshot.jacobian[static_cast<std::size_t>(
                    row * dofs + column)] =
                    matrix.getMatrixEntry(row, column);
            }
        }
        return snapshot;
    };

    const Boundary low{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = low_marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Nitsche,
        .kinematic_nitsche_gamma = 7.0,
        .kinematic_nitsche_symmetric = true,
        .kinematic_nitsche_scale_with_p = false,
    };
    const Boundary high{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = high_marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Nitsche,
        .kinematic_nitsche_gamma = 29.0,
        .kinematic_nitsche_symmetric = false,
        .kinematic_nitsche_scale_with_p = true,
    };

    const auto expect_same = [](const AssemblySnapshot& lhs,
                                const AssemblySnapshot& rhs) {
        ASSERT_EQ(lhs.residual.size(), rhs.residual.size());
        ASSERT_EQ(lhs.jacobian.size(), rhs.jacobian.size());
        for (std::size_t i = 0; i < lhs.residual.size(); ++i) {
            EXPECT_NEAR(lhs.residual[i], rhs.residual[i], 1.0e-13);
        }
        for (std::size_t i = 0; i < lhs.jacobian.size(); ++i) {
            EXPECT_NEAR(lhs.jacobian[i], rhs.jacobian[i], 1.0e-13);
        }
    };

    const auto forward_on_low =
        assemble({low, high}, false, low_marker);
    const auto reverse_on_low =
        assemble({high, low}, false, low_marker);
    const auto forward_on_high =
        assemble({low, high}, false, high_marker);
    const auto reverse_on_high =
        assemble({high, low}, false, high_marker);
    expect_same(forward_on_low, reverse_on_low);
    expect_same(forward_on_high, reverse_on_high);

    const auto low_only = assemble({low}, false, low_marker);
    const auto high_only = assemble({high}, false, high_marker);
    expect_same(forward_on_low, low_only);
    expect_same(forward_on_high, high_only);
    EXPECT_NE(low_only.residual, high_only.residual);
    EXPECT_NE(low_only.jacobian, high_only.jacobian);

    const auto low_with_other_generic_policy = assemble(
        {low}, false, low_marker, {61.0, true, true});
    expect_same(low_only, low_with_other_generic_policy);

    constexpr int generic_marker = 383;
    const auto generic_with_low_policy =
        assemble({low}, true, generic_marker);
    const auto generic_with_high_policy =
        assemble({high}, true, generic_marker);
    expect_same(generic_with_low_policy, generic_with_high_policy);
}

TEST(MovingDomainPhysics, FittedFreeSurfaceKinematicPolicyOptionsAreExplicit)
{
    using FreeSurfaceBoundary = ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;

    const FreeSurfaceBoundary bc{};

    EXPECT_EQ(bc.normal_kinematic_policy,
              ns::FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity);
    EXPECT_EQ(bc.tangential_mesh_policy,
              ns::FreeSurfaceTangentialMeshPolicy::SmoothingOnly);
    EXPECT_EQ(bc.kinematic_enforcement,
              ns::FreeSurfaceKinematicEnforcement::None);
    EXPECT_DOUBLE_EQ(bc.kinematic_nitsche_gamma, 10.0);
    EXPECT_TRUE(bc.kinematic_nitsche_symmetric);
    EXPECT_TRUE(bc.kinematic_nitsche_scale_with_p);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(bc.prescribed_tangential_mesh_velocity[0]), 0.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(bc.prescribed_tangential_mesh_velocity[1]), 0.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(bc.prescribed_tangential_mesh_velocity[2]), 0.0);
}

TEST(MovingDomainPhysics, FittedFreeSurfaceUnwiredTangentialPoliciesFailClosed)
{
    constexpr int marker = 39;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    for (const auto policy : {ns::FreeSurfaceTangentialMeshPolicy::Free,
                              ns::FreeSurfaceTangentialMeshPolicy::Prescribed}) {
        auto opts = baseNavierStokesOptions();
        opts.enable_ale = true;
        opts.enable_convection = false;
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation = ns::FreeSurfaceImplementation::FittedALE,
                .boundary_marker = marker,
                .tangential_mesh_policy = policy,
            });

        FE::systems::FESystem system(mesh);
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    }
}

TEST(MovingDomainPhysics, FreeSurfaceContactLineOptionsAreExplicit)
{
    using FreeSurfaceBoundary = ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;

    const FreeSurfaceContactLine contact_line{};
    EXPECT_TRUE(std::holds_alternative<FreeSurfaceContactLine::None>(
        contact_line.configuration));

    FreeSurfaceBoundary free_surface{};
    EXPECT_TRUE(free_surface.contact_lines.empty());

    free_surface.contact_lines.push_back(prescribedContactLine(
        7,
        0.78539816339744830962,
        {0.0, 1.0, 0.0},
        12.0,
        8));

    ASSERT_EQ(free_surface.contact_lines.size(), 1u);
    const auto& prescribed =
        std::get<FreeSurfaceContactLine::PrescribedAngle>(
            free_surface.contact_lines.front().configuration);
    EXPECT_EQ(prescribed.wall_boundary_marker, 7);
    EXPECT_EQ(prescribed.contact_line_marker, 8);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(prescribed.contact_angle_radians),
                     0.78539816339744830962);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(prescribed.contact_angle_penalty),
                     12.0);

    const auto dynamic = dynamicRenEContactLine(
        9, 1.1, {1.0, 0.0, 0.0}, 0.25, 0.01, 10);
    const auto& ren_e = std::get<FreeSurfaceContactLine::DynamicRenE>(
        dynamic.configuration);
    EXPECT_EQ(ren_e.wall_boundary_marker, 9);
    EXPECT_EQ(ren_e.contact_line_marker, 10);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(ren_e.mobility), 0.25);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(ren_e.slip_length), 0.01);
}

TEST(MovingDomainPhysics, FittedPinnedContactLineConstrainsMeshDisplacement)
{
    constexpr int marker = 40;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto displacement = system.addField(FE::systems::FieldSpec{
        .name = "mesh_displacement",
        .space = u_space,
        .components = 3,
    });

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            pinnedContactLine(/*wall_marker=*/-1, marker),
        },
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto offset = system.fieldDofOffset(displacement);
    const auto n_displacement_dofs = system.fieldDofHandler(displacement).getNumDofs();
    std::size_t constrained_displacement_dofs = 0;
    for (FE::GlobalIndex local_dof = 0; local_dof < n_displacement_dofs; ++local_dof) {
        const auto global_dof = offset + local_dof;
        if (!system.constraints().isConstrained(global_dof)) {
            continue;
        }
        ++constrained_displacement_dofs;
        EXPECT_NEAR(system.constraints().getInhomogeneity(global_dof), 0.0, 1.0e-15);
    }
    EXPECT_GT(constrained_displacement_dofs, 0u);
}

TEST(MovingDomainPhysics, FittedPinnedContactLineRequiresALE)
{
    constexpr int marker = 41;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = false;
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            pinnedContactLine(/*wall_marker=*/-1, marker),
        },
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics,
     FittedPinnedContactLineRejectsPrescribedALEBeforeSystemMutation)
{
    constexpr int marker = 141;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::PrescribedData;
    opts.auto_register_mesh_velocity_field = true;
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::FittedALE,
            .boundary_marker = marker,
            .contact_lines = {
                pinnedContactLine(/*wall_marker=*/-1, marker),
            },
        });

    const auto velocity_name = opts.velocity_field_name;
    const auto pressure_name = opts.pressure_field_name;
    const auto mesh_velocity_name = opts.mesh_velocity_field_name;
    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));

    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    EXPECT_EQ(system.findFieldByName(velocity_name), FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName(pressure_name), FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName(mesh_velocity_name), FE::INVALID_FIELD_ID);
    EXPECT_FALSE(system.hasOperator("equations"));
    EXPECT_TRUE(system.formulationRecords().empty());
}

TEST(MovingDomainPhysics, FittedPrescribedContactAngleFailsClosedWithoutContactLineIntegration)
{
    constexpr int marker = 42;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);
    (void)system.addField(FE::systems::FieldSpec{
        .name = "mesh_displacement",
        .space = u_space,
        .components = 3,
    });

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            prescribedContactLine(marker, 1.0, {1.0, 0.0, 0.0}, 5.0),
        },
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    EXPECT_FALSE(system.hasOperator("mesh_motion"));
}

TEST(MovingDomainPhysics, FittedPrescribedContactAngleFailsClosedWithoutALEToo)
{
    constexpr int marker = 43;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = false;
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            prescribedContactLine(marker, 1.0, {1.0, 0.0, 0.0}),
        },
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceReservesBoundaryMarker)
{
    constexpr int marker = 32;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.velocity_dirichlet.push_back(ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
        .boundary_marker = marker,
        .value = {0.0, 0.0, 0.0},
    });
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 1.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceUsesLevelSetInterfaceGeometry)
{
    constexpr int interface_marker = 41;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .surface_tension = 0.25,
        .curvature = 2.0,
        .use_level_set_curvature = false,
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Identity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::OuterProduct));
    EXPECT_TRUE(formulationRecordsInterfaceIntegralContainsExprType(
        system, interface_marker, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsInterfaceIntegralContainsExprType(
        system, interface_marker, FormExprType::Identity));
    EXPECT_TRUE(formulationRecordsInterfaceIntegralContainsExprType(
        system, interface_marker, FormExprType::OuterProduct));
    EXPECT_NE(log_output.find(
                  "diagnostic=free_surface_variational_surface_stress"),
              std::string::npos);
    EXPECT_EQ(log_output.find("diagnostic=unfitted_level_set_raw_curvature"),
              std::string::npos);
}

TEST(MovingDomainPhysics,
     NavierStokesSurfaceStressRejectsUnsupportedGeometryTangentPolicies)
{
    constexpr int interface_marker = 214;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    const auto register_with = [&](std::string tangent_policy,
                                   bool enable_experimental_tangents) {
        ScopedEnvVar enable_shape_tangents(
            "SVMP_ENABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
            enable_experimental_tangents
                ? std::optional<std::string>("1")
                : std::nullopt);
        ScopedEnvVar disable_shape_tangents(
            "SVMP_DISABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
            enable_experimental_tangents
                ? std::nullopt
                : std::optional<std::string>("1"));
        auto opts = baseNavierStokesOptions();
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi_surface_stress_tangent",
                .geometry_tangent_policy = std::move(tangent_policy),
                .active_domain =
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .surface_tension = 0.072,
            });
        FE::systems::FESystem system(mesh);
        system.addField(FE::systems::FieldSpec{
            .name = "phi_surface_stress_tangent",
            .space = p_space,
            .components = 1,
            .source_kind = FE::systems::FieldSourceKind::PrescribedData,
        });
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        module.registerOn(system);
    };

    EXPECT_NO_THROW(register_with("RefreshedFrozenQuadrature", false));
    EXPECT_THROW(register_with("DifferentiatedQuadrature", false),
                 std::invalid_argument);
    EXPECT_THROW(register_with("RefreshedFrozenQuadrature", true),
                 std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesFittedSurfaceStressFailsClosed)
{
    constexpr int marker = 215;
    const auto mesh =
        std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::FittedALE,
            .boundary_marker = marker,
            .surface_tension = 0.072,
            .surface_tension_form =
                ns::FreeSurfaceSurfaceTensionForm::SurfaceStress,
        });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceRejectsNitscheKinematics)
{
    constexpr int interface_marker = 42;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Nitsche,
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceRejectsPenaltyKinematics)
{
    constexpr int interface_marker = 142;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi",
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .kinematic_enforcement =
                ns::FreeSurfaceKinematicEnforcement::Penalty,
            .kinematic_penalty = 10.0,
        });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedNaturalAndWeakBoundaryOperatorsUseSharpActiveTrace)
{
    constexpr int physical_marker = 214;
    constexpr int interface_marker = 215;
    const auto mesh =
        std::make_shared<SingleTetraBoundaryMeshAccess>(physical_marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    const auto verify = [&](const auto& configure_boundary) {
        auto opts = baseNavierStokesOptions();
        opts.enable_convection = false;
        opts.jit_policy.enable = false;
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi_sharp_boundary",
                .generated_interface_domain_id = "sharp_boundary_routing",
                .active_domain =
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method =
                    ns::FreeSurfaceActiveDomainMethod::CutVolume,
            });
        configure_boundary(opts);

        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi_sharp_boundary",
            .space = p_space,
            .components = 1,
            .source_kind = FE::systems::FieldSourceKind::PrescribedData,
        });
        FE::interfaces::GeneratedActiveBoundaryMarkerKey key;
        key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
        key.domain_id = "sharp_boundary_routing";
        key.interface_marker = interface_marker;
        key.boundary_marker = physical_marker;
        key.side = FE::geometry::CutIntegrationSide::Negative;
        const int sharp_marker =
            FE::interfaces::stableGeneratedActiveBoundaryMarker(key);

        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        module.registerOn(system);
        EXPECT_TRUE(formulationRecordsContainInterfaceMarker(
            system, sharp_marker));
        EXPECT_FALSE(formulationRecordsContainBoundaryMarker(
            system, physical_marker));
    };

    verify([](auto& opts) {
        opts.traction_neumann.push_back(
            ns::IncompressibleNavierStokesVMSOptions::TractionNeumannBC{
                .boundary_marker = physical_marker,
                .traction = {1.0, 2.0, 3.0},
            });
    });
    verify([](auto& opts) {
        opts.traction_robin.push_back(
            ns::IncompressibleNavierStokesVMSOptions::TractionRobinBC{
                .boundary_marker = physical_marker,
                .alpha = 2.0,
                .rhs = {1.0, 0.0, 0.0},
            });
    });
    verify([](auto& opts) {
        opts.pressure_outflow.push_back(
            ns::IncompressibleNavierStokesVMSOptions::PressureOutflowBC{
                .boundary_marker = physical_marker,
                .pressure = 3.0,
                .backflow_beta = 0.25,
            });
    });
    verify([](auto& opts) {
        opts.velocity_dirichlet_weak.push_back(
            ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                .boundary_marker = physical_marker,
                .value = {0.0, 0.0, 0.0},
            });
        opts.nitsche_gamma = 12.0;
    });
    verify([](auto& opts) {
        opts.velocity_dirichlet_weak.push_back(
            ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                .boundary_marker = physical_marker,
                .value = {0.0, 0.0, 0.0},
            });
        opts.nitsche_gamma = 12.0;
        opts.nitsche_symmetric = false;
    });
}

TEST(MovingDomainPhysics, NavierStokesUnfittedSurfaceTensionRejectsRawLevelSetCurvature)
{
    constexpr int interface_marker = 43;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .surface_tension = 0.0728,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedLevelSetShapeTangentsDisabledByDefault)
{
    ScopedEnvVar enable_shape_tangents(
        "SVMP_ENABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
        std::nullopt);
    ScopedEnvVar disable_shape_tangents(
        "SVMP_DISABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
        std::nullopt);

    constexpr int interface_marker = 143;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.jit_policy.enable = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 7.5,
        .surface_tension = 0.0,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
    });

    FE::systems::FESystem system(mesh);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto& equations = system.operatorDefinition("equations");
    const auto has_phi_shape_tangent =
        std::any_of(equations.interface_faces.begin(),
                    equations.interface_faces.end(),
                    [&](const auto& term) {
                        return term.marker == interface_marker &&
                               term.trial_field == phi_field;
                    });

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, interface_marker));
    EXPECT_FALSE(has_phi_shape_tangent);
    EXPECT_EQ(log_output.find(
                  "diagnostic=unfitted_free_surface_interface_measure_shape_tangent"),
              std::string::npos);
    EXPECT_EQ(log_output.find(
                  "diagnostic=unfitted_free_surface_interface_point_location_shape_tangent"),
              std::string::npos);
    EXPECT_NE(log_output.find("cut_volume_phi_shape_tangent=disabled_by_default"),
              std::string::npos);
    EXPECT_EQ(log_output.find("experimental_path=unfitted_level_set_shape_tangent"),
              std::string::npos);
    EXPECT_EQ(log_output.find("qualification=Experimental"), std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedSurfaceTensionUsesSuppliedCurvatureField)
{
    ScopedEnvVar enable_shape_tangents(
        "SVMP_ENABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
        std::string("1"));

    constexpr int interface_marker = 44;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.jit_policy.enable = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .surface_tension = 0.0728,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
        .curvature_field_name = "kappa_projected",
    });

    FE::systems::FESystem system(mesh);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });
    const auto kappa_field = system.addField(FE::systems::FieldSpec{
        .name = "kappa_projected",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContainFieldExprType(
        system, FormExprType::DiscreteField, kappa_field));
    const auto& equations = system.operatorDefinition("equations");
    const auto has_phi_shape_tangent =
        std::any_of(equations.interface_faces.begin(),
                    equations.interface_faces.end(),
                    [&](const auto& term) {
                        return term.marker == interface_marker &&
                               term.trial_field == phi_field;
                    });
    EXPECT_TRUE(has_phi_shape_tangent);
    EXPECT_TRUE(interfaceKernelContainsGradientOfTrialFunction(
        system, interface_marker, phi_field));
    EXPECT_NE(log_output.find(
                  "diagnostic=unfitted_free_surface_interface_point_location_shape_tangent"),
              std::string::npos);
    EXPECT_NE(log_output.find("experimental_path=unfitted_level_set_shape_tangent"),
              std::string::npos);
    EXPECT_NE(log_output.find("qualification=Experimental"), std::string::npos);
    EXPECT_EQ(log_output.find("diagnostic=unfitted_level_set_raw_curvature"),
              std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedSurfaceTensionCouplesUnknownCurvatureField)
{
    constexpr int interface_marker = 144;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .surface_tension = 0.0728,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
        .curvature_field_name = "kappa_unknown",
    });

    FE::systems::FESystem system(mesh);
    (void)system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });
    const auto kappa_field = system.addField(FE::systems::FieldSpec{
        .name = "kappa_unknown",
        .space = p_space,
        .components = 1,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto u = system.findFieldByName(opts.velocity_field_name);
    const auto p = system.findFieldByName(opts.pressure_field_name);
    ASSERT_NE(u, FE::INVALID_FIELD_ID);
    ASSERT_NE(p, FE::INVALID_FIELD_ID);
    EXPECT_TRUE(formulationRecordsContainFieldExprType(
        system, FormExprType::StateField, kappa_field));

    const auto& equations = system.operatorDefinition("equations");
    const auto has_kappa_tangent =
        std::any_of(equations.interface_faces.begin(),
                    equations.interface_faces.end(),
                    [&](const auto& term) {
                        return term.marker == interface_marker &&
                               term.trial_field == kappa_field &&
                               (term.test_field == u || term.test_field == p);
                    });
    EXPECT_TRUE(has_kappa_tangent);
    EXPECT_NE(log_output.find(
                  "diagnostic=unfitted_free_surface_curvature_trial_field"),
              std::string::npos);
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedSurfaceTensionUnknownCurvatureJacobianMatchesFiniteDifference)
{
    constexpr int interface_marker = 145;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.enable_vms = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .surface_tension = 0.0728,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
        .curvature_field_name = "kappa_unknown",
    });

    FE::systems::FESystem system(mesh);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    const auto kappa_field = system.addField(FE::systems::FieldSpec{
        .name = "kappa_unknown",
        .space = p_space,
        .components = 1,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setCutIntegrationContext(
        makeSingleTetraFreeSurfaceCutContext(interface_marker, phi_field));
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    system.setPrescribedFieldCoefficients(
        phi_field,
        affineZScalarTetraCoefficients(FE::Real{-0.25}, FE::Real{1.0}));

    const auto velocity = system.findFieldByName(opts.velocity_field_name);
    const auto pressure = system.findFieldByName(opts.pressure_field_name);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    ASSERT_NE(pressure, FE::INVALID_FIELD_ID);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, velocity, vertex, 0,
                               FE::Real{0.04} + FE::Real{0.02} * x[0]);
        setFieldComponentValue(solution, system, velocity, vertex, 1,
                               FE::Real{-0.03} + FE::Real{0.01} * x[1]);
        setFieldComponentValue(solution, system, velocity, vertex, 2,
                               FE::Real{0.02} - FE::Real{0.015} * x[2]);
        setFieldComponentValue(solution, system, pressure, vertex, 0,
                               FE::Real{0.10} + FE::Real{0.03} * x[0] -
                                   FE::Real{0.02} * x[1]);
        setFieldComponentValue(solution, system, kappa_field, vertex, 0,
                               FE::Real{1.20} + FE::Real{0.10} * x[0] +
                                   FE::Real{0.07} * x[2]);
    }
    previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(previous_solution, system, velocity, vertex, 0,
                               FE::Real{0.01});
        setFieldComponentValue(previous_solution, system, velocity, vertex, 1,
                               FE::Real{-0.02});
        setFieldComponentValue(previous_solution, system, velocity, vertex, 2,
                               FE::Real{0.015});
    }

    FE::systems::SystemStateView state;
    state.dt = 0.2;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        "equations",
        /*eps=*/1.0e-6,
        /*rtol=*/2.0e-5,
        /*atol=*/2.0e-8);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceUsesGeneratedInterfaceMarker)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, expected_marker));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceAddsCutCellStabilization)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
        },
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContainInteriorFaceMarker(system, expected_marker));
    EXPECT_FALSE(formulationRecordsContainUnmarkedInteriorFaceIntegral(system));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Jump));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Average));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::ParameterRef));
    EXPECT_NE(log_output.find("cut-cell stabilization"), std::string::npos);
    EXPECT_NE(log_output.find("interface_side=Minus"), std::string::npos);
    EXPECT_NE(log_output.find("active_domain_side=Negative"), std::string::npos);
    EXPECT_NE(log_output.find("use_cut_metadata_scale=false"), std::string::npos);
    EXPECT_NE(log_output.find("velocity_polynomial_order=1"), std::string::npos);
    EXPECT_NE(log_output.find("pressure_polynomial_order=1"), std::string::npos);
    EXPECT_NE(log_output.find("derivative_orders=1"), std::string::npos);
    EXPECT_NE(log_output.find("velocity_ghost_penalty_mode=retired_replaced_by_aggregation"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_scaling=0.01*h^3/(mu+rho*h^2/dt)"),
              std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceUsesCutMetadataScale)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
            .use_cut_metadata_scale = true,
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::ParameterRef));
    EXPECT_NE(log_output.find("cut-cell stabilization"), std::string::npos);
    EXPECT_NE(log_output.find("use_cut_metadata_scale=true"), std::string::npos);
    EXPECT_NE(log_output.find("cut_metadata_scale_cap=unbounded"), std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceCapsCutMetadataScale)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
            .use_cut_metadata_scale = true,
            .cut_metadata_scale_cap = 3.0,
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::ParameterRef));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Minimum));
    EXPECT_NE(log_output.find("use_cut_metadata_scale=true"), std::string::npos);
    EXPECT_NE(log_output.find("cut_metadata_scale_cap=3"), std::string::npos);
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedHighOrderCutCellStabilizationAddsSecondDerivativeTerms)
{
    const auto mesh = makeMesh();
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/3);
    auto p_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
        },
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Jump));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Hessian));
    const auto p_id = system.findFieldByName("p");
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    EXPECT_GT(interiorFaceKernelCountForBlock(system, p_id, p_id, expected_marker), 0u);
    EXPECT_NE(log_output.find("velocity_polynomial_order=2"), std::string::npos);
    EXPECT_NE(log_output.find("pressure_polynomial_order=2"), std::string::npos);
    EXPECT_NE(log_output.find("derivative_orders=1,2"), std::string::npos);
    EXPECT_NE(log_output.find("pressure_derivative_orders=1,2"), std::string::npos);
    EXPECT_NE(log_output.find(
                  "pressure_scaling=0.01*h^3/(mu+rho*h^2/dt),0.01*h^5/(mu+rho*h^2/dt)"),
              std::string::npos);
    EXPECT_EQ(log_output.find("first-gradient ghost penalties only"), std::string::npos);
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedHighOrderPressurePolicyDisablesRefreshedFrozenPressureTerms)
{
    const auto mesh = makeMesh();
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/3);
    auto p_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .generated_interface_geometry = "HighOrderImplicit",
        .geometry_tangent_policy = "RefreshedFrozenQuadrature",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
            .pressure_policy =
                ns::FreeSurfacePressureStabilizationPolicy::
                    DisabledForRefreshedFrozenHighOrder,
        },
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto u_id = system.findFieldByName("u");
    const auto p_id = system.findFieldByName("p");
    ASSERT_NE(u_id, FE::INVALID_FIELD_ID);
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    // The velocity ghost penalty is retired (small-cut aggregation replaces
    // it), so a policy-disabled pressure term leaves no cut-facet terms.
    EXPECT_EQ(interiorFaceKernelCountForBlock(system, u_id, u_id, expected_marker), 0u);
    EXPECT_EQ(interiorFaceKernelCountForBlock(system, p_id, p_id, expected_marker), 0u);
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::Hessian));
    EXPECT_NE(log_output.find(
                  "pressure_stabilization_policy=DisabledForRefreshedFrozenHighOrder"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_stabilization=disabled"), std::string::npos);
    EXPECT_NE(log_output.find(
                  "pressure_disabled_reason=refreshed_frozen_high_order_policy"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_derivative_orders=disabled"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_scaling=disabled"), std::string::npos);
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedIncrementalPressurePolicyInstallsRuntimeDtForm)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
            .pressure_policy =
                ns::FreeSurfacePressureStabilizationPolicy::Incremental,
        },
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto p_id = system.findFieldByName("p");
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    EXPECT_GT(interiorFaceKernelCountForBlock(system, p_id, p_id, expected_marker), 0u);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::EffectiveTimeStep));
    EXPECT_NE(log_output.find("pressure_stabilization_policy=Incremental"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_stabilization=enabled"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_stabilization_form=incremental"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_derivative_orders=1"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_scaling=0.01*h^3/(mu+rho*h^2/dt)"),
              std::string::npos);
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedVelocityGhostPenaltyRetired)
{
    const auto mesh = makeMesh();
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/3);
    auto p_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .generated_interface_geometry = "HighOrderImplicit",
        .geometry_tangent_policy = "RefreshedFrozenQuadrature",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
            .pressure_policy =
                ns::FreeSurfacePressureStabilizationPolicy::
                    DisabledForRefreshedFrozenHighOrder,
            .use_cut_metadata_scale = false,
        },
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto u_id = system.findFieldByName("u");
    const auto p_id = system.findFieldByName("p");
    ASSERT_NE(u_id, FE::INVALID_FIELD_ID);
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    // No velocity cut-facet terms remain: the ghost penalty is retired in
    // favor of small-cut aggregation, independent of any derivative policy.
    EXPECT_EQ(interiorFaceKernelCountForBlock(system, u_id, u_id, expected_marker), 0u);
    EXPECT_EQ(interiorFaceKernelCountForBlock(system, p_id, p_id, expected_marker), 0u);
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::Hessian));
    EXPECT_NE(log_output.find("velocity_ghost_penalty_mode=retired_replaced_by_aggregation"),
              std::string::npos);
    EXPECT_NE(log_output.find("pressure_derivative_orders=disabled"),
              std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedZeroTractionFreeSurfaceAvoidsInterfaceIntegral)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 0.0,
        .surface_tension = 0.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .pressure_gradient_penalty = 0.25,
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainZeroTractionFreeSurfaceAvoidsInterfaceIntegral)
{
    constexpr int interface_marker = 46;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 0.0,
        .surface_tension = 0.0,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainUnknownLevelSetAddsMatrixOnlyShapeTangent)
{
    ScopedEnvVar enable_shape_tangents(
        "SVMP_ENABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
        std::string("1"));

    constexpr int interface_marker = 146;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 0.0,
        .surface_tension = 0.0,
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto u = system.findFieldByName(opts.velocity_field_name);
    const auto p = system.findFieldByName(opts.pressure_field_name);
    ASSERT_NE(u, FE::INVALID_FIELD_ID);
    ASSERT_NE(p, FE::INVALID_FIELD_ID);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));

    const auto& equations = system.operatorDefinition("equations");
    const auto has_phi_shape_tangent =
        std::any_of(equations.interface_faces.begin(),
                    equations.interface_faces.end(),
                    [&](const auto& term) {
                        return term.marker == interface_marker &&
                               term.trial_field == phi &&
                               (term.test_field == u || term.test_field == p);
                    });
    EXPECT_TRUE(has_phi_shape_tangent);
    EXPECT_NE(log_output.find("cut_volume_phi_shape_tangent=matrix_only_hadamard"),
              std::string::npos);
    EXPECT_NE(log_output.find("experimental_path=unfitted_level_set_shape_tangent"),
              std::string::npos);
    EXPECT_NE(log_output.find("qualification=Experimental"), std::string::npos);
    EXPECT_EQ(log_output.find(
                  "no_explicit_level_set_dependence_under_frozen_cut_geometry"),
              std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainExternalPressureInstallsInterfaceTraction)
{
    constexpr int interface_marker = 48;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 12.0,
        .surface_tension = 0.0,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, interface_marker));
}

TEST(MovingDomainPhysics,
     NavierStokesActiveDomainExternalPressureTractionJacobianMatchesFiniteDifference)
{
    constexpr int interface_marker = 54;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.viscosity = 0.037;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 12.0,
        .surface_tension = 0.0,
    });

    FE::systems::FESystem system(mesh);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setCutIntegrationContext(
        makeSingleTetraFreeSurfaceCutContext(interface_marker, phi_field));
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    system.setPrescribedFieldCoefficients(
        phi_field,
        affineZScalarTetraCoefficients(FE::Real{-0.25}, FE::Real{1.0}));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution(solution.size(), 0.0);

    FE::systems::SystemStateView state;
    state.dt = 0.25;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    EXPECT_GT(residualNorm(system, state, "equations"), 1.0e-8);
    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        "equations",
        /*eps=*/1.0e-6,
        /*rtol=*/2.0e-5,
        /*atol=*/2.0e-8);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedInterfaceMeasureAddsLevelSetShapeTangent)
{
    ScopedEnvVar enable_shape_tangents(
        "SVMP_ENABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS",
        std::string("1"));

    constexpr int interface_marker = 148;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 7.5,
        .surface_tension = 0.0,
        .surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(module.registerOn(system));
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto u = system.findFieldByName(opts.velocity_field_name);
    const auto p = system.findFieldByName(opts.pressure_field_name);
    ASSERT_NE(u, FE::INVALID_FIELD_ID);
    ASSERT_NE(p, FE::INVALID_FIELD_ID);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, interface_marker));

    const auto& equations = system.operatorDefinition("equations");
    const auto has_phi_shape_tangent =
        std::any_of(equations.interface_faces.begin(),
                    equations.interface_faces.end(),
                    [&](const auto& term) {
                        return term.marker == interface_marker &&
                               term.trial_field == phi &&
                               (term.test_field == u || term.test_field == p);
                    });
    EXPECT_TRUE(has_phi_shape_tangent);
    EXPECT_NE(log_output.find(
                  "diagnostic=unfitted_free_surface_interface_measure_shape_tangent"),
              std::string::npos);
    EXPECT_NE(log_output.find(
                  "diagnostic=unfitted_free_surface_interface_point_location_shape_tangent"),
              std::string::npos);
    EXPECT_NE(log_output.find("experimental_path=unfitted_level_set_shape_tangent"),
              std::string::npos);
    EXPECT_NE(log_output.find("qualification=Experimental"), std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainSurfaceTensionInstallsInterfaceTraction)
{
    constexpr int interface_marker = 49;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 0.0,
        .surface_tension = 0.0728,
        .curvature = 3.0,
        .use_level_set_curvature = false,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, interface_marker));
    EXPECT_EQ(log_output.find("diagnostic=unfitted_level_set_raw_curvature"),
              std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesInactiveActiveDomainKeepsFullCellVolumeKernels)
{
    constexpr int interface_marker = 47;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .allow_full_domain_unfitted_free_surface = true,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_FALSE(equations.cells.empty());
    EXPECT_TRUE(equations.cut_volumes.empty());
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainInstallsCutVolumeKernels)
{
    constexpr int interface_marker = 48;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_TRUE(equations.cells.empty());
    ASSERT_FALSE(equations.cut_volumes.empty());
    for (const auto& term : equations.cut_volumes) {
        EXPECT_EQ(term.marker, interface_marker);
        EXPECT_EQ(term.side, FE::geometry::CutIntegrationSide::Negative);
    }
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CellIntegral));
}

TEST(MovingDomainPhysics, NavierStokesPspgContinuityFullCellSupportAddsCellKernel)
{
    ScopedEnvVar full_cell_support(
        "SVMP_NS_PSPG_CONTINUITY_FULL_CELL_SUPPORT",
        std::string("1"));

    constexpr int interface_marker = 52;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    module.registerOn(system);
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_FALSE(equations.cells.empty());
    ASSERT_FALSE(equations.cut_volumes.empty());
    for (const auto& term : equations.cut_volumes) {
        EXPECT_EQ(term.marker, interface_marker);
        EXPECT_EQ(term.side, FE::geometry::CutIntegrationSide::Negative);
    }
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
    EXPECT_NE(log_output.find(
                  "diagnostic=navier_stokes_pspg_continuity_volume_support"),
              std::string::npos);
    EXPECT_NE(log_output.find("full_cell_vms_pspg_plus_active_galerkin"),
              std::string::npos);
    EXPECT_NE(log_output.find("qualification=DiagnosticOnly"),
              std::string::npos);
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainCutVolumeSamplesNonconstantVelocity)
{
    constexpr int interface_marker = 51;
    constexpr FE::Real measure = FE::Real{1.0} / FE::Real{12.0};
    auto constant_only_context = makeSingleTetraCutVolumeContext(
        interface_marker,
        {FE::geometry::CutQuadraturePoint{
            .point = {{0.25, 0.25, 0.25}},
            .weight = measure,
        }});
    auto split_context = makeSingleTetraCutVolumeContext(
        interface_marker,
        {
            FE::geometry::CutQuadraturePoint{
                .point = {{0.10, 0.20, 0.25}},
                .weight = measure / FE::Real{2.0},
            },
            FE::geometry::CutQuadraturePoint{
                .point = {{0.40, 0.30, 0.25}},
                .weight = measure / FE::Real{2.0},
            },
        });

    const auto assemble_with_context =
        [&](const std::shared_ptr<FE::assembly::CutIntegrationContext>& cut_context) {
            const auto mesh = makeMesh();
            auto u_space = makeVelocitySpace(mesh);
            auto p_space = makePressureSpace(mesh);
            auto opts = baseNavierStokesOptions();
            opts.enable_convection = false;
            opts.enable_vms = false;
            opts.viscosity = 1.0e-12;

            opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi",
                .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            });

            FE::systems::FESystem system(mesh);
            system.addField(FE::systems::FieldSpec{
                .name = "phi",
                .space = p_space,
                .components = 1,
                .source_kind = FE::systems::FieldSourceKind::PrescribedData,
            });

            ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
            module.registerOn(system);
            system.setCutIntegrationContext(cut_context);
            system.setup({}, makeSingleTetraSetupInputs());

            const FE::FieldId u_id = system.findFieldByName(opts.velocity_field_name);
            if (u_id == FE::INVALID_FIELD_ID) {
                ADD_FAILURE() << "velocity field was not registered";
                return std::vector<FE::Real>{};
            }

            std::vector<FE::Real> solution(
                static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
            for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
                const auto x = mesh->getNodeCoordinates(vertex);
                setFieldComponentValue(solution, system, u_id, vertex, 0, x[0]);
            }
            const std::vector<FE::Real> previous_solution(
                static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);

            FE::systems::SystemStateView state;
            state.dt = 1.0;
            state.u = std::span<const FE::Real>(solution);
            state.u_prev = std::span<const FE::Real>(previous_solution);
            const FE::systems::BackwardDifferenceIntegrator integrator;
            const auto time_context =
                integrator.buildContext(/*max_time_derivative_order=*/1, state);
            state.time_integration = &time_context;

            return residualVector(system, state, "equations");
        };

    const auto constant_only_residual = assemble_with_context(constant_only_context);
    const auto split_residual = assemble_with_context(split_context);

    std::vector<FE::Real> residual_delta(constant_only_residual.size(), 0.0);
    for (std::size_t i = 0; i < residual_delta.size(); ++i) {
        residual_delta[i] = split_residual[i] - constant_only_residual[i];
    }

    EXPECT_GT(vectorNorm(residual_delta), 1.0e-5);
}

TEST(MovingDomainPhysics,
     NavierStokesActiveDomainCutVolumeResidualJacobianMatchesFiniteDifference)
{
    constexpr int interface_marker = 53;
    constexpr FE::Real measure = FE::Real{1.0} / FE::Real{12.0};
    auto cut_context = makeSingleTetraCutVolumeContext(
        interface_marker,
        {
            FE::geometry::CutQuadraturePoint{
                .point = {{0.12, 0.22, 0.18}},
                .weight = measure / FE::Real{2.0},
            },
            FE::geometry::CutQuadraturePoint{
                .point = {{0.36, 0.19, 0.28}},
                .weight = measure / FE::Real{2.0},
            },
        });

    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.viscosity = 0.037;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 0.0,
        .surface_tension = 0.0,
    });

    FE::systems::FESystem system(mesh);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setCutIntegrationContext(std::move(cut_context));
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    system.setPrescribedFieldCoefficients(
        phi_field,
        constantScalarTetraCoefficients(FE::Real{-0.10}));

    const auto velocity = system.findFieldByName(opts.velocity_field_name);
    const auto pressure = system.findFieldByName(opts.pressure_field_name);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    ASSERT_NE(pressure, FE::INVALID_FIELD_ID);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, velocity, vertex, 0,
                               FE::Real{0.05} + FE::Real{0.03} * x[0]);
        setFieldComponentValue(solution, system, velocity, vertex, 1,
                               FE::Real{-0.02} + FE::Real{0.04} * x[1]);
        setFieldComponentValue(solution, system, velocity, vertex, 2,
                               FE::Real{0.01} - FE::Real{0.02} * x[2]);
        setFieldComponentValue(solution, system, pressure, vertex, 0,
                               FE::Real{0.14} + FE::Real{0.05} * x[0] -
                                   FE::Real{0.03} * x[2]);
        setFieldComponentValue(previous_solution, system, velocity, vertex, 0,
                               FE::Real{0.01});
        setFieldComponentValue(previous_solution, system, velocity, vertex, 1,
                               FE::Real{-0.015});
        setFieldComponentValue(previous_solution, system, velocity, vertex, 2,
                               FE::Real{0.02});
    }

    FE::systems::SystemStateView state;
    state.dt = 0.25;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        "equations",
        /*eps=*/1.0e-6,
        /*rtol=*/2.0e-5,
        /*atol=*/2.0e-8);
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainResidualFollowsNewtonLevelSetSignChange)
{
    constexpr int interface_marker = 52;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.viscosity = 1.0e-12;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
    });

    FE::systems::FESystem system(mesh);
    const auto phi_id = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    const FE::FieldId u_id = system.findFieldByName(opts.velocity_field_name);
    ASSERT_NE(u_id, FE::INVALID_FIELD_ID);

    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
    cut_options.level_set_field_name = "phi";
    cut_options.domain_id = "free_surface";
    cut_options.requested_interface_marker = interface_marker;
    cut_options.quadrature_order = 0;
    cut_options.interface_quadrature_order = 0;
    cut_options.volume_quadrature_order = 0;

    struct ResidualSample {
        std::vector<FE::Real> residual;
        FE::Real negative_volume{0.0};
        std::uint64_t value_revision{0};
    };

    const auto residual_for_phi = [&](const std::array<FE::Real, 4>& phi_values) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            const auto x = mesh->getNodeCoordinates(vertex);
            setFieldComponentValue(solution, system, u_id, vertex, 0, x[0]);
            setFieldComponentValue(solution, system, phi_id, vertex, 0,
                                   phi_values[static_cast<std::size_t>(vertex)]);
        }

        const auto result = lifecycle.build(system, cut_options, solution);
        EXPECT_TRUE(result.success) << result.diagnostic;
        auto cut_context = std::make_shared<FE::assembly::CutIntegrationContext>();
        cut_context->addGeneratedInterfaceDomain(result.domain);
        system.setCutIntegrationContext(std::move(cut_context));

        const std::vector<FE::Real> previous_solution(solution.size(), 0.0);
        FE::systems::SystemStateView state;
        state.dt = 1.0;
        state.u = std::span<const FE::Real>(solution);
        state.u_prev = std::span<const FE::Real>(previous_solution);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto time_context =
            integrator.buildContext(/*max_time_derivative_order=*/1, state);
        state.time_integration = &time_context;

        return ResidualSample{
            .residual = residualVector(system, state, "equations"),
            .negative_volume = result.summary.negative_volume_measure,
            .value_revision = result.value_revision,
        };
    };

    const auto initial = residual_for_phi({FE::Real{-0.1},
                                           FE::Real{1.0},
                                           FE::Real{1.0},
                                           FE::Real{1.0}});
    const auto updated = residual_for_phi({FE::Real{-1.0},
                                           FE::Real{-1.0},
                                           FE::Real{-1.0},
                                           FE::Real{0.1}});

    ASSERT_EQ(initial.residual.size(), updated.residual.size());
    std::vector<FE::Real> residual_delta(initial.residual.size(), 0.0);
    for (std::size_t i = 0; i < residual_delta.size(); ++i) {
        residual_delta[i] = updated.residual[i] - initial.residual[i];
    }

    EXPECT_GT(updated.value_revision, initial.value_revision);
    EXPECT_GT(updated.negative_volume, initial.negative_volume);
    EXPECT_GT(vectorNorm(residual_delta), 1.0e-5);
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainPositiveUsesPositiveCutVolumeSide)
{
    constexpr int interface_marker = 49;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetPositive,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_TRUE(equations.cells.empty());
    ASSERT_FALSE(equations.cut_volumes.empty());
    for (const auto& term : equations.cut_volumes) {
        EXPECT_EQ(term.marker, interface_marker);
        EXPECT_EQ(term.side, FE::geometry::CutIntegrationSide::Positive);
    }
}

TEST(MovingDomainPhysics, NavierStokesSmoothedIndicatorActiveDomainUsesWeightedCellKernels)
{
    constexpr int interface_marker = 50;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;
    opts.input_configuration_schema_version = 1;
    opts.explicit_legacy_configuration = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .active_domain_method = ns::FreeSurfaceActiveDomainMethod::SmoothedIndicator,
        .active_domain_smoothing_width = 0.125,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_FALSE(equations.cells.empty());
    EXPECT_TRUE(equations.cut_volumes.empty());
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::SmoothHeaviside));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
}

TEST(MovingDomainPhysics, NavierStokesRejectsCutCellStabilizationOnFittedSurface)
{
    constexpr int marker = 45;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .cut_cell_stabilization = {
            .enabled = true,
        },
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesRejectsActiveDomainOnFittedSurface)
{
    constexpr int marker = 46;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceRejectsUnknownLevelSet)
{
    constexpr int interface_marker = 42;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "missing_phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 1.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceRejectsALEUntilTransportUsesRelativeVelocity)
{
    constexpr int interface_marker = 43;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_ale",
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi_ale",
        .space = p_space,
        .components = 1,
    });
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    try {
        module.registerOn(system);
        FAIL() << "Expected unfitted level-set plus ALE to fail closed";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("relative velocity u-w"),
                  std::string::npos)
            << error.what();
    }
}

TEST(MovingDomainPhysics, NavierStokesUnfittedContactAngleUsesOwningEquationsOperator)
{
    constexpr int interface_marker = 44;
    constexpr int wall_marker = 12;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .contact_lines = {
            prescribedContactLine(
                wall_marker,
                1.0471975511965977462,
                {1.0, 0.0, 0.0},
                4.0),
        },
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });
    system.addOperator("equations");
    const auto phi_state =
        FE::forms::StateField(phi, *p_space, "phi_equations_owner");
    const auto eta =
        FE::forms::TestField(phi, *p_space, "eta_equations_owner");
    (void)FE::systems::installFormulation(
        system,
        "equations",
        {phi},
        (phi_state * eta).dx());
    const int contact_marker =
        stableContactLineMarker(phi, interface_marker, wall_marker);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_FALSE(system.hasOperator("level_set"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, contact_marker));
    EXPECT_FALSE(formulationRecordsContainInterfaceMarker(system,
                                                         interface_marker));
    bool found_contact_record = false;
    for (const auto& record : system.formulationRecords()) {
        if (!record.residual_expr) {
            continue;
        }
        const auto scan = FE::analysis::scanFormExpr(*record.residual_expr);
        if (std::find(scan.interface_markers.begin(),
                      scan.interface_markers.end(),
                      contact_marker) == scan.interface_markers.end()) {
            continue;
        }
        found_contact_record = true;
        EXPECT_EQ(record.operator_tag, "equations");
    }
    EXPECT_TRUE(found_contact_record);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedContactAnglePropagatesExistingFieldOwner)
{
    constexpr int interface_marker = 54;
    constexpr int wall_marker = 22;
    constexpr std::string_view owner_tag = "coupled_level_set_system";
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi",
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .contact_lines = {
                prescribedContactLine(
                    wall_marker,
                    1.0471975511965977462,
                    {1.0, 0.0, 0.0},
                    4.0),
            },
        });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });
    system.addOperator(std::string(owner_tag));
    const auto phi_state =
        FE::forms::StateField(phi, *p_space, "phi_owner_state");
    const auto eta = FE::forms::TestField(phi, *p_space, "eta_owner");
    (void)FE::systems::installFormulation(
        system,
        std::string(owner_tag),
        {phi},
        (phi_state * eta).dx());

    const int contact_marker =
        stableContactLineMarker(phi, interface_marker, wall_marker);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    bool found_contact_record = false;
    for (const auto& record : system.formulationRecords()) {
        if (!record.residual_expr) {
            continue;
        }
        const auto scan = FE::analysis::scanFormExpr(*record.residual_expr);
        if (std::find(scan.interface_markers.begin(),
                      scan.interface_markers.end(),
                      contact_marker) == scan.interface_markers.end()) {
            continue;
        }
        found_contact_record = true;
        EXPECT_EQ(record.operator_tag, owner_tag);
    }
    EXPECT_TRUE(found_contact_record);
    EXPECT_FALSE(system.hasOperator("level_set"));
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedContactAngleRequiresOwnerBeforeSystemMutation)
{
    constexpr int interface_marker = 154;
    constexpr int wall_marker = 24;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    const auto velocity_name = opts.velocity_field_name;
    const auto pressure_name = opts.pressure_field_name;
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_without_owner",
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .contact_lines = {
                prescribedContactLine(
                    wall_marker,
                    1.0471975511965977462,
                    {1.0, 0.0, 0.0},
                    4.0),
            },
        });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi_without_owner",
        .space = p_space,
        .components = 1,
    });
    ASSERT_EQ(system.findFieldByName(velocity_name), FE::INVALID_FIELD_ID);
    ASSERT_EQ(system.findFieldByName(pressure_name), FE::INVALID_FIELD_ID);

    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    try {
        module.registerOn(system);
        FAIL() << "Expected missing level-set ownership to fail preflight";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find(
                      "no installed owner formulation"),
                  std::string::npos)
            << error.what();
    }

    EXPECT_EQ(system.findFieldByName(velocity_name), FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName(pressure_name), FE::INVALID_FIELD_ID);
    EXPECT_FALSE(system.hasOperator("equations"));
}

TEST(MovingDomainPhysics,
     NavierStokesRejectsIncompatiblePressureBeforeVelocityMutation)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    const auto velocity_name = opts.velocity_field_name;
    const auto pressure_name = opts.pressure_field_name;

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = pressure_name,
        .space = u_space,
        .components = 3,
    });

    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    EXPECT_EQ(system.findFieldByName(velocity_name), FE::INVALID_FIELD_ID);
    EXPECT_FALSE(system.hasOperator("equations"));
}

TEST(MovingDomainPhysics,
     NavierStokesRejectsMissingALEDataBeforeFluidMutation)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.auto_register_mesh_velocity_field = false;
    const auto velocity_name = opts.velocity_field_name;
    const auto pressure_name = opts.pressure_field_name;

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    EXPECT_EQ(system.findFieldByName(velocity_name), FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName(pressure_name), FE::INVALID_FIELD_ID);
    EXPECT_FALSE(system.hasOperator("equations"));
}

TEST(MovingDomainPhysics,
     NavierStokesRejectsPendingFieldNameCollisionWithoutMutation)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.pressure_field_name = opts.velocity_field_name;
    const auto shared_name = opts.velocity_field_name;

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    EXPECT_EQ(system.findFieldByName(shared_name), FE::INVALID_FIELD_ID);
    EXPECT_FALSE(system.hasOperator("equations"));
}

TEST(MovingDomainPhysics,
     NavierStokesRejectsBoundaryConflictBeforeFluidMutation)
{
    constexpr int marker = 25;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.velocity_dirichlet.push_back(
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = marker,
            .value = {0.0, 0.0, 0.0},
        });
    opts.traction_neumann.push_back(
        ns::IncompressibleNavierStokesVMSOptions::TractionNeumannBC{
            .boundary_marker = marker,
            .traction = {0.0, 0.0, 0.0},
        });
    const auto velocity_name = opts.velocity_field_name;
    const auto pressure_name = opts.pressure_field_name;

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    EXPECT_EQ(system.findFieldByName(velocity_name), FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName(pressure_name), FE::INVALID_FIELD_ID);
    EXPECT_FALSE(system.hasOperator("equations"));
}

TEST(MovingDomainPhysics,
     NavierStokesConfigurationSchemaRejectsBeforeSystemMutation)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    const auto expect_rejected = [&](int version, bool explicit_legacy) {
        auto opts = baseNavierStokesOptions();
        opts.input_configuration_schema_version = version;
        opts.explicit_legacy_configuration = explicit_legacy;
        const auto velocity_name = opts.velocity_field_name;
        const auto pressure_name = opts.pressure_field_name;

        FE::systems::FESystem system(mesh);
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
        EXPECT_EQ(system.findFieldByName(velocity_name), FE::INVALID_FIELD_ID);
        EXPECT_EQ(system.findFieldByName(pressure_name), FE::INVALID_FIELD_ID);
        EXPECT_FALSE(system.hasOperator("equations"));
        EXPECT_FALSE(module.effectiveConfigurationArtifact().has_value());
    };

    expect_rejected(/*version=*/1, /*explicit_legacy=*/false);
    expect_rejected(
        ns::IncompressibleNavierStokesVMSOptions::
            current_configuration_schema_version,
        /*explicit_legacy=*/true);
    expect_rejected(/*version=*/99, /*explicit_legacy=*/false);
}

TEST(MovingDomainPhysics,
     NavierStokesLegacySchemaIsExplicitAndLosesCurrentCapabilityLabel)
{
    constexpr int interface_marker = 171;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.input_configuration_schema_version = 1;
    opts.explicit_legacy_configuration = true;
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation =
                ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_legacy_schema",
            .active_domain =
                ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .active_domain_method =
                ns::FreeSurfaceActiveDomainMethod::SmoothedIndicator,
            .active_domain_smoothing_width = 0.125,
        });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi_legacy_schema",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    ASSERT_NO_THROW(module.registerOn(system));

    const auto artifact = module.effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_EQ(artifact->component,
              "incompressible_navier_stokes_free_surface");
    EXPECT_NE(artifact->json.find("\"migration_mode\":\"explicit_legacy\""),
              std::string::npos);
    EXPECT_NE(artifact->json.find("\"capability_label\":\"legacy_diagnostic\""),
              std::string::npos);
    EXPECT_EQ(artifact->json.find("one_phase_liquid_sharp_interface"),
              std::string::npos);
    EXPECT_NE(artifact->json.find("\"active_domain_method\":\"SmoothedIndicator\""),
              std::string::npos);
}

TEST(MovingDomainPhysics,
     NavierStokesEffectiveConfigurationSnapshotExpandsBoundaryDefaults)
{
    constexpr int marker = 172;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.nitsche_gamma = 23.0;
    opts.nitsche_symmetric = false;
    opts.nitsche_scale_with_p = false;
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::FittedALE,
            .boundary_marker = marker,
            .external_pressure = 2.25,
            .surface_tension = 0.5,
            .curvature = 3.0,
            .kinematic_enforcement =
                ns::FreeSurfaceKinematicEnforcement::Nitsche,
            .kinematic_nitsche_gamma = 17.0,
            .kinematic_nitsche_symmetric = false,
            .kinematic_nitsche_scale_with_p = false,
            .contact_lines = {
                ns::IncompressibleNavierStokesVMSOptions::
                    FreeSurfaceContactLine{},
            },
        });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    ASSERT_NO_THROW(module.registerOn(system));

    const auto artifact = module.effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    constexpr std::string_view expected =
        R"json({"artifact_schema_version":1,"component":"incompressible_navier_stokes_free_surface","configuration_schema":{"input_version":2,"effective_version":2,"migration_mode":"current"},"capability_label":"one_phase_liquid_sharp_interface","units":{"system":"consistent_solver_units","angle":"radian","length":"solver_length","pressure":"solver_pressure","surface_tension":"force_per_length"},"fields":{"velocity":"u","pressure":"p","operator":"equations","dimension":3},"ale":{"enabled":true,"mesh_velocity_source":"PrescribedData","mesh_velocity_field":"mesh_velocity","mesh_displacement_field":"mesh_displacement","geometry_tangent_path":"SymbolicRequired"},"generic_velocity_nitsche":{"gamma":23,"symmetric":false,"scale_with_polynomial_order":false},"stabilization":{"vms_enabled":false,"ct_m":1,"ct_c":36,"epsilon":9.9999999999999998e-13},"maintenance_policy":{"owner_component":"level_set_transport","coupling":"one_way_velocity_to_extension_to_level_set"},"extension_guards":{"physical_momentum_dry_extension_allowed":false,"auxiliary_extension_owner":"level_set_transport","external_owner_required":true},"free_surfaces":[{"implementation":"FittedALE","boundary_marker":172,"interface_marker":-1,"level_set_field":"level_set","generated_interface_domain":"free_surface","generated_interface_geometry":"LinearCorner","geometry_tangent_policy":"RefreshedFrozenQuadrature","level_set_isovalue":0,"active_domain":"None","active_phase_sign":"full_domain","active_domain_method":"CutVolume","active_domain_smoothing_width":0,"smoothing_width_unit":"length","allow_full_domain_unfitted_free_surface":false,"external_pressure":2.25,"surface_tension":0.5,"surface_tension_form_requested":"Automatic","surface_tension_form_effective":"CurvatureTraction","curvature_policy":"supplied_scalar","curvature_tangent_policy":"supplied_scalar_frozen","kinematic":{"normal_policy":"MatchFluidNormalVelocity","tangential_mesh_policy":"SmoothingOnly","prescribed_tangential_mesh_velocity":[0,0,0],"enforcement":"Nitsche","penalty":0,"nitsche":{"gamma":17,"symmetric":false,"scale_with_polynomial_order":false}},"stabilization":{"enabled":false,"small_cut_aggregation":true,"pressure_policy":"Enabled","pressure_gradient_penalty":1,"use_cut_metadata_scale":false,"cut_metadata_scale_cap":null},"pruning":{"decision_owner":"authoritative_geometry_snapshot","fallback_to_whole_face":false},"legacy_dry_velocity_diffusion":{"enabled":false,"diffusivity":1,"production_allowed":false},"contact_lines":[{"model":"None"}]}]})json";
    EXPECT_EQ(artifact->json, expected);
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedContactAngleUsesThroughLiquidConventionOnAllWallOrientations)
{
    constexpr FE::Real pi = 3.14159265358979323846;
    for (const FE::Real angle : {pi / 3.0, 2.0 * pi / 3.0}) {
        for (int axis = 0; axis < 3; ++axis) {
            for (const FE::Real orientation : {-1.0, 1.0}) {
                std::array<FE::Real, 3> wall{0.0, 0.0, 0.0};
                wall[static_cast<std::size_t>(axis)] = orientation;
                std::array<FE::Real, 3> active_outward_normal{0.0, 0.0, 0.0};
                active_outward_normal[static_cast<std::size_t>(axis)] =
                    -std::cos(angle) * orientation;
                active_outward_normal[static_cast<std::size_t>((axis + 1) % 3)] =
                    std::sin(angle);

                const auto negative_active = unfittedContactAngleResidualVector(
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                    angle,
                    active_outward_normal,
                    wall);
                EXPECT_NEAR(vectorNorm(negative_active), 0.0, 1.0e-11)
                    << "angle=" << angle << " axis=" << axis
                    << " orientation=" << orientation
                    << " active=negative";

                for (auto& component : active_outward_normal) {
                    component = -component;
                }
                const auto positive_active = unfittedContactAngleResidualVector(
                    ns::FreeSurfaceActiveDomain::LevelSetPositive,
                    angle,
                    active_outward_normal,
                    wall);
                EXPECT_NEAR(vectorNorm(positive_active), 0.0, 1.0e-11)
                    << "angle=" << angle << " axis=" << axis
                    << " orientation=" << orientation
                    << " active=positive";
            }
        }
    }
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleResidualChangesSignWithLevelSetNormal)
{
    constexpr FE::Real half_pi = 1.57079632679489661923;
    constexpr FE::Real inv_sqrt_two = 0.70710678118654752440;
    const std::array<FE::Real, 3> gradient{
        inv_sqrt_two, 0.0, inv_sqrt_two};
    const std::array<FE::Real, 3> wall{0.0, 0.0, 1.0};
    ContactAngleAssemblyProbe contact_probe;
    const auto positive_gap = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        gradient,
        wall,
        &contact_probe);
    ContactAngleAssemblyProbe owner_probe;
    (void)unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        gradient,
        wall,
        &owner_probe,
        /*include_contact_angle=*/false);
    const auto negative_gap = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        std::array<FE::Real, 3>{-inv_sqrt_two, 0.0, -inv_sqrt_two});

    ASSERT_EQ(positive_gap.size(), negative_gap.size());
    ASSERT_EQ(contact_probe.phi_jacobian.size(),
              owner_probe.phi_jacobian.size());
    EXPECT_GT(vectorNorm(positive_gap), 1.0e-12);
    EXPECT_GT(vectorNorm(negative_gap), 1.0e-12);
    FE::Real contact_jacobian_norm2 = 0.0;
    for (std::size_t i = 0; i < contact_probe.phi_jacobian.size(); ++i) {
        const auto contact_entry =
            contact_probe.phi_jacobian[i] - owner_probe.phi_jacobian[i];
        contact_jacobian_norm2 += contact_entry * contact_entry;
    }
    EXPECT_GT(std::sqrt(contact_jacobian_norm2), 1.0e-12);
    for (std::size_t i = 0; i < positive_gap.size(); ++i) {
        EXPECT_NEAR(positive_gap[i], -negative_gap[i], 1.0e-12);
    }
}

TEST(MovingDomainPhysics,
     NavierStokesSurfaceStressPrescribedAngleUsesGeneratedContactNormal)
{
    constexpr FE::Real theta = FE::Real{1.0471975511965977462};
    constexpr FE::Real gamma = FE::Real{0.8};
    constexpr std::array<FE::Real, 3> wall_normal{0.0, 0.0, 1.0};
    constexpr std::array<FE::Real, 3> target_outward_normal{
        FE::Real{0.86602540378443864676}, FE::Real{0.0}, FE::Real{-0.5}};
    constexpr std::array<FE::Real, 3> wrong_outward_normal{
        FE::Real{1.0}, FE::Real{0.0}, FE::Real{0.0}};

    const auto signed_raw_normal = [](std::array<FE::Real, 3> outward,
                                      ns::FreeSurfaceActiveDomain side) {
        if (side == ns::FreeSurfaceActiveDomain::LevelSetPositive) {
            for (auto& component : outward) {
                component = -component;
            }
        }
        return outward;
    };
    const auto phi_row_norm = [](const std::vector<FE::Real>& residual,
                                 const ContactAngleAssemblyProbe& probe) {
        FE::Real norm2 = FE::Real{0.0};
        for (FE::GlobalIndex row = 0; row < probe.phi_dofs; ++row) {
            const auto value = residual[static_cast<std::size_t>(
                probe.phi_offset + row)];
            norm2 += value * value;
        }
        return std::sqrt(norm2);
    };

    for (const auto active_domain : {
             ns::FreeSurfaceActiveDomain::LevelSetNegative,
             ns::FreeSurfaceActiveDomain::LevelSetPositive}) {
        ContactAngleAssemblyProbe generated_target_probe;
        const auto generated_target = unfittedContactAngleResidualVector(
            active_domain,
            theta,
            signed_raw_normal(wrong_outward_normal, active_domain),
            wall_normal,
            &generated_target_probe,
            /*include_contact_angle=*/true,
            /*include_transient_owner=*/true,
            signed_raw_normal(target_outward_normal, active_domain),
            gamma,
            ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);
        EXPECT_NEAR(phi_row_norm(generated_target, generated_target_probe),
                    FE::Real{0.0},
                    2.0e-11)
            << "the generated contact normal satisfies the target even though "
               "the separately evaluated Q1 gradient does not";

        ContactAngleAssemblyProbe zero_gamma_probe;
        const auto zero_gamma = unfittedContactAngleResidualVector(
            active_domain,
            theta,
            signed_raw_normal(wrong_outward_normal, active_domain),
            wall_normal,
            &zero_gamma_probe,
            /*include_contact_angle=*/true,
            /*include_transient_owner=*/true,
            signed_raw_normal(target_outward_normal, active_domain),
            /*surface_tension=*/0.0,
            ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);
        EXPECT_NEAR(phi_row_norm(zero_gamma, zero_gamma_probe),
                    FE::Real{0.0},
                    2.0e-11)
            << "explicit SurfaceStress keeps one generated contact geometry "
               "even when gamma is zero";

        ContactAngleAssemblyProbe generated_wrong_probe;
        const auto generated_wrong = unfittedContactAngleResidualVector(
            active_domain,
            theta,
            signed_raw_normal(target_outward_normal, active_domain),
            wall_normal,
            &generated_wrong_probe,
            /*include_contact_angle=*/true,
            /*include_transient_owner=*/true,
            signed_raw_normal(wrong_outward_normal, active_domain),
            gamma,
            ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);
        EXPECT_GT(phi_row_norm(generated_wrong, generated_wrong_probe),
                  FE::Real{1.0e-10})
            << "the generated contact normal must control the SurfaceStress "
               "angle gap even when the Q1 gradient satisfies the target";
    }
}

TEST(MovingDomainPhysics,
     NavierStokesUnfittedContactAngleJacobianMatchesFiniteDifferenceOnEquations)
{
    constexpr FE::Real half_pi = 1.57079632679489661923;
    constexpr FE::Real inv_sqrt_two = 0.70710678118654752440;
    constexpr FE::Real epsilon = 1.0e-6;
    const std::array<FE::Real, 3> wall{0.0, 0.0, 1.0};
    const std::array<FE::Real, 3> gradient{
        inv_sqrt_two, 0.0, inv_sqrt_two};

    ContactAngleAssemblyProbe contact_probe;
    (void)unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        gradient,
        wall,
        &contact_probe);
    ContactAngleAssemblyProbe owner_probe;
    (void)unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        gradient,
        wall,
        &owner_probe,
        /*include_contact_angle=*/false);

    auto plus_gradient = gradient;
    auto minus_gradient = gradient;
    plus_gradient[0] += epsilon;
    minus_gradient[0] -= epsilon;
    const auto plus_residual = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        plus_gradient,
        wall);
    const auto plus_owner_residual = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        plus_gradient,
        wall,
        nullptr,
        /*include_contact_angle=*/false);
    const auto minus_residual = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        minus_gradient,
        wall);
    const auto minus_owner_residual = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        minus_gradient,
        wall,
        nullptr,
        /*include_contact_angle=*/false);

    ASSERT_EQ(contact_probe.phi_dofs, owner_probe.phi_dofs);
    ASSERT_EQ(contact_probe.phi_offset, owner_probe.phi_offset);
    ASSERT_EQ(contact_probe.vertex_one_phi_dof,
              owner_probe.vertex_one_phi_dof);
    ASSERT_EQ(contact_probe.phi_jacobian.size(),
              owner_probe.phi_jacobian.size());
    ASSERT_EQ(plus_residual.size(), minus_residual.size());
    ASSERT_EQ(plus_residual.size(), plus_owner_residual.size());
    ASSERT_EQ(minus_residual.size(), minus_owner_residual.size());
    ASSERT_LE(static_cast<std::size_t>(
                  contact_probe.phi_offset + contact_probe.phi_dofs),
              plus_residual.size());

    FE::Real analytic_norm2 = 0.0;
    FE::Real error_norm2 = 0.0;
    for (FE::GlobalIndex row = 0; row < contact_probe.phi_dofs; ++row) {
        const auto index = static_cast<std::size_t>(
            row * contact_probe.phi_dofs +
            contact_probe.vertex_one_phi_dof);
        const auto analytic = contact_probe.phi_jacobian[index] -
                              owner_probe.phi_jacobian[index];
        const auto global_row = static_cast<std::size_t>(
            contact_probe.phi_offset + row);
        const auto finite_difference =
            ((plus_residual[global_row] - plus_owner_residual[global_row]) -
             (minus_residual[global_row] - minus_owner_residual[global_row])) /
            (FE::Real{2.0} * epsilon);
        const auto error = analytic - finite_difference;
        EXPECT_NEAR(analytic, finite_difference, 1.0e-7)
            << "phi row=" << row
            << " analytic=" << analytic
            << " finite_difference=" << finite_difference;
        analytic_norm2 += analytic * analytic;
        error_norm2 += error * error;
    }
    const auto analytic_norm = std::sqrt(analytic_norm2);
    EXPECT_GT(analytic_norm, 1.0e-10);
    EXPECT_LT(std::sqrt(error_norm2),
              1.0e-5 * std::max(FE::Real{1.0}, analytic_norm));
}

TEST(MovingDomainPhysics,
     NavierStokesContactAngleGaugeEvidenceRequiresTransientPhiOwner)
{
    constexpr FE::Real half_pi = 1.57079632679489661923;
    constexpr FE::Real inv_sqrt_two = 0.70710678118654752440;
    const std::array<FE::Real, 3> gradient{
        inv_sqrt_two, 0.0, inv_sqrt_two};
    const std::array<FE::Real, 3> wall{0.0, 0.0, 1.0};

    ContactAngleAssemblyProbe transient_probe;
    (void)unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        gradient,
        wall,
        &transient_probe);
    EXPECT_FALSE(transient_probe.phi_has_constraint);

    ContactAngleAssemblyProbe gradient_only_probe;
    (void)unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        gradient,
        wall,
        &gradient_only_probe,
        /*include_contact_angle=*/true,
        /*include_transient_owner=*/false);
    EXPECT_TRUE(gradient_only_probe.phi_has_constraint);
}

TEST(MovingDomainPhysics,
     NavierStokesCurvatureTractionDynamicContactAngleEquilibriumHasZeroAddedResidual)
{
    constexpr FE::Real theta = 1.0471975511965977462;
    constexpr std::array<FE::Real, 4> zero_velocity{0.0, 0.0, 0.0, 0.0};
    const auto dynamic = assembleDynamicContactAngleCase(
        theta, theta, zero_velocity, /*include_dynamic_contact_angle=*/true);
    const auto baseline = assembleDynamicContactAngleCase(
        theta, theta, zero_velocity, /*include_dynamic_contact_angle=*/false);

    ASSERT_EQ(dynamic.residual.size(), baseline.residual.size());
    std::vector<FE::Real> added_residual(dynamic.residual.size(), 0.0);
    for (std::size_t i = 0; i < added_residual.size(); ++i) {
        added_residual[i] = dynamic.residual[i] - baseline.residual[i];
    }
    EXPECT_NEAR(vectorNorm(added_residual), 0.0, 2.0e-11);
    EXPECT_TRUE(dynamic.has_velocity_level_set_coupling);
}

TEST(MovingDomainPhysics,
     NavierStokesSurfaceStressContactWallEnergyMatchesCombinedVirtualWorkForBothActiveSides)
{
    ScopedEnvVar conservative_balance_diagnostics(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        std::string("1"));
    constexpr FE::Real pi =
        FE::Real{3.141592653589793238462643383279502884};
    constexpr FE::Real gamma = FE::Real{0.8};
    constexpr FE::Real surface_measure = FE::Real{0.038};
    constexpr FE::Real contact_measure = FE::Real{0.125};
    constexpr FE::Real contact_x = FE::Real{0.2};
    constexpr std::array<FE::Real, 4> zero_velocity{
        FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}};
    // v_h=(1+x,0,0) on the affine unit tetra.  For
    // n=(sin(theta),0,cos(theta)), m=(1,0,0), so
    //   P:grad(v_h)=cos(theta)^2,
    //   v_h(x_contact).m=1+contact_x.
    constexpr std::array<FE::Real, 4> virtual_velocity_x{
        FE::Real{1.0}, FE::Real{2.0}, FE::Real{1.0}, FE::Real{1.0}};

    const auto virtual_work = [&](const DynamicContactAngleAssembly& result,
                                  std::span<const FE::Real> residual) {
        FE::Real work = FE::Real{0.0};
        for (std::size_t vertex = 0; vertex < virtual_velocity_x.size();
             ++vertex) {
            work += virtual_velocity_x[vertex] *
                    residual[static_cast<std::size_t>(
                        result.velocity_x_dofs[vertex])];
        }
        return work;
    };

    std::array<FE::Real, 3> negative_combined_work{};
    std::size_t angle_index = 0u;
    for (const FE::Real theta : {pi / FE::Real{3.0},
                                 pi / FE::Real{2.0},
                                 FE::Real{2.0} * pi / FE::Real{3.0}}) {
        const auto cosine = std::cos(theta);
        const auto expected_surface_work =
            gamma * surface_measure * cosine * cosine;
        const auto expected_wall_work =
            -gamma * cosine * contact_measure *
            (FE::Real{1.0} + contact_x);
        const auto expected_combined_work =
            expected_surface_work + expected_wall_work;

        for (const auto active_domain : {
                 ns::FreeSurfaceActiveDomain::LevelSetNegative,
                 ns::FreeSurfaceActiveDomain::LevelSetPositive}) {
            const auto surface_only = assembleDynamicContactAngleCase(
                theta,
                theta,
                zero_velocity,
                /*include_dynamic_contact_angle=*/false,
                /*assemble_jacobian=*/false,
                std::array<FE::Real, 4>{0.0, 0.0, 0.0, 0.0},
                std::array<FE::Real, 3>{0.0, 0.0, -1.0},
                /*level_set_scale=*/1.0,
                /*level_set_shift=*/0.0,
                /*velocity_component=*/0,
                active_domain,
                ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);
            const auto combined = assembleDynamicContactAngleCase(
                theta,
                theta,
                zero_velocity,
                /*include_dynamic_contact_angle=*/true,
                /*assemble_jacobian=*/false,
                std::array<FE::Real, 4>{0.0, 0.0, 0.0, 0.0},
                std::array<FE::Real, 3>{0.0, 0.0, -1.0},
                /*level_set_scale=*/1.0,
                /*level_set_shift=*/0.0,
                /*velocity_component=*/0,
                active_domain,
                ns::FreeSurfaceSurfaceTensionForm::SurfaceStress);

            const auto assembled_surface_work =
                virtual_work(surface_only, surface_only.residual);
            const auto assembled_combined_work =
                virtual_work(combined, combined.residual);
            const auto assembled_wall_work =
                assembled_combined_work - assembled_surface_work;
            ASSERT_FALSE(
                combined.conservative_surface_energy_residual.empty());
            const auto diagnostic_surface_energy_work = virtual_work(
                combined,
                combined.conservative_surface_energy_residual);
            EXPECT_NEAR(assembled_surface_work,
                        expected_surface_work,
                        2.0e-11)
                << "theta=" << theta
                << " active_domain=" << static_cast<int>(active_domain);
            EXPECT_NEAR(assembled_wall_work, expected_wall_work, 2.0e-11)
                << "theta=" << theta
                << " active_domain=" << static_cast<int>(active_domain);
            EXPECT_NEAR(assembled_combined_work,
                        expected_combined_work,
                        2.0e-11)
                << "theta=" << theta
                << " active_domain=" << static_cast<int>(active_domain);
            EXPECT_NEAR(diagnostic_surface_energy_work,
                        expected_combined_work,
                        2.0e-11)
                << "the conservative surface-energy operator must contain "
                   "the generated-interface area variation and Young wall "
                   "energy exactly once";

            if (active_domain ==
                ns::FreeSurfaceActiveDomain::LevelSetNegative) {
                negative_combined_work[angle_index] =
                    assembled_combined_work;
            } else {
                EXPECT_NEAR(assembled_combined_work,
                            negative_combined_work[angle_index],
                            2.0e-11)
                    << "SurfaceStress and the directed footprint normal must "
                       "be invariant when the active level-set side is "
                       "swapped consistently";
            }
        }
        ++angle_index;
    }
}

TEST(MovingDomainPhysics,
     NavierStokesSurfaceStressConservativeBalanceDiagnosticSplitsTermsAndExcludesDissipation)
{
    constexpr FE::Real theta =
        FE::Real{1.04719755119659774615421446109316763};
    constexpr FE::Real external_pressure = FE::Real{0.031};
    constexpr std::array<FE::Real, 4> zero_velocity{
        FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}};
    constexpr std::array<FE::Real, 4> uniform_tangential_velocity{
        FE::Real{0.25}, FE::Real{0.25}, FE::Real{0.25}, FE::Real{0.25}};

    const auto assemble = [&](FE::Real equilibrium_angle,
                              const std::array<FE::Real, 4>& velocity,
                              FE::Real liquid_pressure,
                              ns::FreeSurfaceSurfaceTensionForm form =
                                  ns::FreeSurfaceSurfaceTensionForm::
                                      SurfaceStress) {
        return assembleDynamicContactAngleCase(
            equilibrium_angle,
            theta,
            velocity,
            /*include_dynamic_contact_angle=*/true,
            /*assemble_jacobian=*/false,
            std::array<FE::Real, 4>{0.0, 0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.0, -1.0},
            /*level_set_scale=*/1.0,
            /*level_set_shift=*/0.0,
            /*velocity_component=*/0,
            ns::FreeSurfaceActiveDomain::LevelSetNegative,
            form,
            liquid_pressure,
            external_pressure);
    };

    {
        ScopedEnvVar diagnostics_disabled(
            "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
            std::string("0"));
        const auto disabled = assemble(theta, zero_velocity, FE::Real{0.37});
        EXPECT_TRUE(disabled.conservative_pressure_residual.empty());
        EXPECT_TRUE(disabled.conservative_surface_energy_residual.empty());
        EXPECT_TRUE(disabled.conservative_balance_residual.empty());
    }

    ScopedEnvVar diagnostics_enabled(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        std::string("1"));
    const auto stationary =
        assemble(theta, zero_velocity, FE::Real{0.37});
    const auto moving =
        assemble(theta, uniform_tangential_velocity, FE::Real{0.37});
    const auto changed_pressure =
        assemble(theta, zero_velocity, FE::Real{0.53});
    const auto changed_angle =
        assemble(theta + FE::Real{0.12}, zero_velocity, FE::Real{0.37});
    const auto legacy = assemble(
        theta,
        zero_velocity,
        FE::Real{0.37},
        ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction);

    ASSERT_FALSE(stationary.conservative_pressure_residual.empty());
    ASSERT_EQ(stationary.conservative_pressure_residual.size(),
              stationary.conservative_surface_energy_residual.size());
    ASSERT_EQ(stationary.conservative_pressure_residual.size(),
              stationary.conservative_balance_residual.size());
    EXPECT_GT(vectorNorm(stationary.conservative_pressure_residual),
              FE::Real{1.0e-12});
    EXPECT_GT(vectorNorm(stationary.conservative_surface_energy_residual),
              FE::Real{1.0e-12});

    for (std::size_t dof = 0;
         dof < stationary.conservative_balance_residual.size(); ++dof) {
        EXPECT_NEAR(
            stationary.conservative_balance_residual[dof],
            stationary.conservative_pressure_residual[dof] +
                stationary.conservative_surface_energy_residual[dof],
            FE::Real{2.0e-12})
            << "dof=" << dof;
        EXPECT_NEAR(moving.conservative_pressure_residual[dof],
                    stationary.conservative_pressure_residual[dof],
                    FE::Real{2.0e-12});
        EXPECT_NEAR(moving.conservative_surface_energy_residual[dof],
                    stationary.conservative_surface_energy_residual[dof],
                    FE::Real{2.0e-12});
        EXPECT_NEAR(changed_pressure.conservative_surface_energy_residual[dof],
                    stationary.conservative_surface_energy_residual[dof],
                    FE::Real{2.0e-12});
        EXPECT_NEAR(changed_angle.conservative_pressure_residual[dof],
                    stationary.conservative_pressure_residual[dof],
                    FE::Real{2.0e-12});
    }

    std::vector<FE::Real> production_velocity_change(
        stationary.residual.size(), FE::Real{0.0});
    std::vector<FE::Real> pressure_term_change(
        stationary.residual.size(), FE::Real{0.0});
    std::vector<FE::Real> surface_term_change(
        stationary.residual.size(), FE::Real{0.0});
    for (std::size_t dof = 0; dof < stationary.residual.size(); ++dof) {
        production_velocity_change[dof] =
            moving.residual[dof] - stationary.residual[dof];
        pressure_term_change[dof] =
            changed_pressure.conservative_pressure_residual[dof] -
            stationary.conservative_pressure_residual[dof];
        surface_term_change[dof] =
            changed_angle.conservative_surface_energy_residual[dof] -
            stationary.conservative_surface_energy_residual[dof];
    }
    EXPECT_GT(vectorNorm(production_velocity_change), FE::Real{1.0e-8})
        << "the nonzero line-friction/Navier-slip control did not change the "
           "production residual";
    EXPECT_GT(vectorNorm(pressure_term_change), FE::Real{1.0e-8});
    EXPECT_GT(vectorNorm(surface_term_change), FE::Real{1.0e-8});

    EXPECT_TRUE(legacy.conservative_pressure_residual.empty());
    EXPECT_TRUE(legacy.conservative_surface_energy_residual.empty());
    EXPECT_TRUE(legacy.conservative_balance_residual.empty());
}

TEST(MovingDomainPhysics,
     NavierStokesSurfaceStressPressureRepresentabilityPairIsTransposeWithZeroDiagonalBlocks)
{
    ScopedEnvVar diagnostics_enabled(
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        std::string("1"));
    constexpr FE::Real theta =
        FE::Real{1.04719755119659774615421446109316763};
    constexpr std::array<FE::Real, 4> zero_velocity{
        FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}};
    const auto assembled = assembleDynamicContactAngleCase(
        theta,
        theta,
        zero_velocity,
        /*include_dynamic_contact_angle=*/true,
        /*assemble_jacobian=*/false,
        std::array<FE::Real, 4>{0.0, 0.0, 0.0, 0.0},
        std::array<FE::Real, 3>{0.0, 0.0, -1.0},
        /*level_set_scale=*/1.0,
        /*level_set_shift=*/0.0,
        /*velocity_component=*/0,
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        ns::FreeSurfaceSurfaceTensionForm::SurfaceStress,
        /*liquid_pressure=*/FE::Real{0.37},
        /*external_pressure=*/FE::Real{0.0});

    ASSERT_FALSE(
        assembled.pressure_representability_pair_jacobian.empty());
    ASSERT_FALSE(assembled.conservative_pressure_residual.empty());
    ASSERT_EQ(assembled.conservative_pressure_residual.size(),
              assembled.solution.size());
    ASSERT_NE(assembled.velocity_field, FE::INVALID_FIELD_ID);
    ASSERT_NE(assembled.pressure_field, FE::INVALID_FIELD_ID);
    const auto velocity_first = assembled.velocity_offset;
    const auto velocity_count = assembled.velocity_dofs;
    const auto pressure_first = assembled.pressure_offset;
    const auto pressure_count = assembled.pressure_dofs;
    ASSERT_GT(velocity_count, 0);
    ASSERT_GT(pressure_count, 0);
    ASSERT_LE(pressure_first + pressure_count, assembled.total_dofs);

    const auto entry = [&](FE::GlobalIndex row, FE::GlobalIndex column) {
        return assembled.pressure_representability_pair_jacobian
            [static_cast<std::size_t>(
                row * assembled.total_dofs + column)];
    };
    const auto in_range = [](FE::GlobalIndex dof,
                             FE::GlobalIndex begin,
                             FE::GlobalIndex count) {
        return dof >= begin && dof < begin + count;
    };

    FE::Real cross_norm2 = FE::Real{0.0};
    for (FE::GlobalIndex row = 0; row < assembled.total_dofs; ++row) {
        for (FE::GlobalIndex column = 0; column < assembled.total_dofs;
             ++column) {
            const bool velocity_pressure =
                in_range(row, velocity_first, velocity_count) &&
                in_range(column, pressure_first, pressure_count);
            const bool pressure_velocity =
                in_range(row, pressure_first, pressure_count) &&
                in_range(column, velocity_first, velocity_count);
            if (velocity_pressure) {
                EXPECT_NEAR(entry(row, column),
                            entry(column, row),
                            FE::Real{2.0e-12})
                    << "row=" << row << " column=" << column;
                cross_norm2 += entry(row, column) * entry(row, column);
            } else if (!pressure_velocity) {
                EXPECT_NEAR(entry(row, column), FE::Real{0.0},
                            FE::Real{2.0e-12})
                    << "the representability pair must contain no u-u, p-p, "
                       "or level-set blocks; row="
                    << row << " column=" << column;
            }
        }
    }
    EXPECT_GT(std::sqrt(cross_norm2), FE::Real{1.0e-10});

    // The upper-right block must be the pressure virtual-work operator with
    // the production sign, not merely some symmetric transpose pair.  With
    // zero external pressure, applying G to the assembled pressure
    // coefficients must reproduce the independently assembled pressure-only
    // residual on every velocity test row.
    FE::Real projected_pressure_norm2 = FE::Real{0.0};
    for (FE::GlobalIndex velocity_row = velocity_first;
         velocity_row < velocity_first + velocity_count;
         ++velocity_row) {
        FE::Real projected = FE::Real{0.0};
        for (FE::GlobalIndex pressure_column = pressure_first;
             pressure_column < pressure_first + pressure_count;
             ++pressure_column) {
            projected +=
                entry(velocity_row, pressure_column) *
                assembled.solution[static_cast<std::size_t>(pressure_column)];
        }
        const auto expected = assembled.conservative_pressure_residual
            [static_cast<std::size_t>(velocity_row)];
        EXPECT_NEAR(projected, expected, FE::Real{2.0e-12})
            << "velocity_row=" << velocity_row;
        projected_pressure_norm2 += projected * projected;
    }
    EXPECT_GT(std::sqrt(projected_pressure_norm2), FE::Real{1.0e-10});
}

TEST(MovingDomainPhysics,
     NavierStokesCurvatureTractionDynamicContactAngleReportsLiveQ1OperatorGeometry)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr int left_marker = 171;
    constexpr int right_marker = 172;
    constexpr int bottom_marker = 173;
    constexpr int outer_marker = 174;
    constexpr int interface_marker = 175;
    constexpr FE::Real half_pi =
        FE::Real{1.57079632679489661923132169163975144};

    auto mesh = makeOpenTankQuadMesh(
        left_marker,
        right_marker,
        bottom_marker,
        outer_marker,
        "outer_boundary",
        /*bottom_y=*/0.0,
        /*middle_y=*/1.0,
        /*top_y=*/2.0);
    const auto phi_handle = MeshFields::attach_field(
        mesh->local_mesh(),
        EntityKind::Vertex,
        "phi_dynamic_operator_diagnostic",
        FieldScalarType::Float64,
        1);
    auto* phi_values = MeshFields::field_data_as<real_t>(
        mesh->local_mesh(), phi_handle);
    ASSERT_NE(phi_values, nullptr);
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
        const auto x = mesh->local_mesh().X_ref().at(2u * vertex);
        const auto y = mesh->local_mesh().X_ref().at(2u * vertex + 1u);
        // The exact Q1 field has physical gradient (1, 1/2) and its wall
        // root in parent cell 0 is x=-0.6 (xi=-0.2, eta=-1).
        phi_values[vertex] = x + FE::Real{0.6} + FE::Real{0.5} * y;
    }

    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4, /*order=*/1);
    auto velocity_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4, /*order=*/1, /*components=*/2);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.velocity_dirichlet.push_back(
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = bottom_marker,
            .value = {0.0, 0.0, 0.0},
            .active_components = {false, true, false},
        });
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation =
                ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name =
                "phi_dynamic_operator_diagnostic",
            .active_domain =
                ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .active_domain_method =
                ns::FreeSurfaceActiveDomainMethod::CutVolume,
            .surface_tension = 0.8,
            .surface_tension_form =
                ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction,
            .curvature = 0.0,
            .use_level_set_curvature = false,
            .contact_lines = {
                dynamicRenEContactLine(
                    bottom_marker,
                    half_pi,
                    {0.0, -1.0, 0.0},
                    0.5,
                    0.2)},
            .small_cut_aggregation = false,
        });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi_dynamic_operator_diagnostic",
        .space = scalar_space,
        .components = 1,
    });
    system.addOperator("equations");
    const auto phi_state = FE::forms::StateField(
        phi, *scalar_space, "phi_dynamic_operator_diagnostic_owner");
    const auto eta = FE::forms::TestField(
        phi, *scalar_space, "eta_dynamic_operator_diagnostic_owner");
    (void)FE::systems::installFormulation(
        system,
        "equations",
        {phi},
        (FE::forms::dt(phi_state) * eta).dx());

    ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, scalar_space, opts);
    module.registerOn(system);
    ASSERT_NO_THROW(system.setup());

    const int contact_marker =
        stableContactLineMarker(phi, interface_marker, bottom_marker);
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionRequest request;
    request.source = FE::interfaces::LevelSetInterfaceSource::fromField(
        phi, /*layout_revision=*/0u, /*value_revision=*/1u);
    request.generated_domain_id = "free_surface";
    request.interface_marker = interface_marker;
    request.boundary_marker = bottom_marker;
    request.intersection_marker = contact_marker;
    request.quadrature_order = 1;
    request.frame = FE::geometry::CutGeometryFrame::Reference;
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionDomain domain(
        request);
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionFragment fragment;
    fragment.interface_marker = interface_marker;
    fragment.boundary_marker = bottom_marker;
    fragment.intersection_marker = contact_marker;
    fragment.parent_cell = 0;
    fragment.parent_face = 0;
    fragment.kind = FE::interfaces::
        GeneratedInterfaceBoundaryIntersectionKind::Point;
    fragment.measure = 1.0;
    fragment.interface_normal = {{2.0 / std::sqrt(5.0),
                                  1.0 / std::sqrt(5.0),
                                  0.0}};
    fragment.boundary_normal = {{0.0, -1.0, 0.0}};
    fragment.quadrature_points.push_back(
        FE::interfaces::
            GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
                .point = {{-0.2, -1.0, 0.0}},
                .parent_coordinate = {{-0.2, -1.0, 0.0}},
                .interface_normal = fragment.interface_normal,
                .boundary_normal = fragment.boundary_normal,
                .weight = 1.0,
                .reference_measure_factor = 1.0,
                .gradient_norm = std::sqrt(1.25),
            });
    domain.addFragment(std::move(fragment));
    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    FE::interfaces::CutInterfaceDomainRequest interface_request;
    interface_request.source =
        FE::interfaces::LevelSetInterfaceSource::fromField(
            phi, /*layout_revision=*/0u, /*value_revision=*/1u);
    interface_request.interface_marker = interface_marker;
    interface_request.quadrature_order = 1;
    interface_request.interface_quadrature_order = 1;
    interface_request.volume_quadrature_order = 1;
    FE::interfaces::LevelSetInterfaceDomain interface_domain(
        interface_request);
    FE::interfaces::CutInterfaceFragment interface_fragment;
    interface_fragment.interface_marker = interface_marker;
    interface_fragment.parent_cell = 0;
    interface_fragment.local_fragment_index = 0;
    interface_fragment.stable_id = 1;
    interface_fragment.kind =
        FE::interfaces::CutInterfaceFragmentKind::Segment;
    interface_fragment.measure = 1.0;
    interface_fragment.normal = {{2.0 / std::sqrt(5.0),
                                  1.0 / std::sqrt(5.0),
                                  0.0}};
    interface_fragment.quadrature_points.push_back(
        FE::interfaces::CutInterfaceQuadraturePoint{
            .point = {{-0.2, -0.5, 0.0}},
            .parent_coordinate = {{-0.2, -0.5, 0.0}},
            .normal = interface_fragment.normal,
            .weight = 1.0,
        });
    interface_domain.addFragment(std::move(interface_fragment));
    context->addGeneratedInterfaceDomain(
        interface_domain, FE::geometry::CutIntegrationSide::Negative);
    context->addGeneratedInterfaceBoundaryIntersectionDomain(domain);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(system.setCutIntegrationContext(std::move(context)));
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();
    ASSERT_NE(log_output.find(
                  "diagnostic=dynamic_contact_operator_angle status=available"),
              std::string::npos)
        << log_output;
    EXPECT_NE(log_output.find("normal_source=unitNormalFromLevelSet_Q1"),
              std::string::npos)
        << log_output;
    EXPECT_NE(log_output.find("evaluation_location=generated_contact_root"),
              std::string::npos)
        << log_output;
    EXPECT_NE(log_output.find("samples=1"), std::string::npos) << log_output;
    EXPECT_NE(log_output.find("reference_rules=1"), std::string::npos)
        << log_output;

    const auto diagnostic_value = [&](std::string_view key) {
        const auto position = log_output.find(std::string(key));
        EXPECT_NE(position, std::string::npos) << log_output;
        if (position == std::string::npos) {
            return FE::Real{0.0};
        }
        const auto begin = position + key.size();
        return static_cast<FE::Real>(std::stod(log_output.substr(begin)));
    };
    const auto expected_cos = FE::Real{1.0} / std::sqrt(FE::Real{5.0});
    EXPECT_NEAR(diagnostic_value("mean_dynamic_cos="), expected_cos, 1.0e-6);
    EXPECT_NEAR(diagnostic_value("mean_young_gap="), -expected_cos, 1.0e-6);
    EXPECT_NEAR(diagnostic_value("min_wall_tangential_normal_norm="),
                FE::Real{2.0} / std::sqrt(FE::Real{5.0}),
                1.0e-6);
    EXPECT_NE(log_output.find("transversality_satisfied=true"),
              std::string::npos)
        << log_output;
#endif
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleRejectsMismatchedGeneratedWallNormal)
{
    constexpr FE::Real theta = 1.0471975511965977462;
    constexpr std::array<FE::Real, 4> zero_velocity{0.0, 0.0, 0.0, 0.0};
    EXPECT_THROW(
        assembleDynamicContactAngleCase(
            theta,
            theta,
            zero_velocity,
            /*include_dynamic_contact_angle=*/true,
            /*assemble_jacobian=*/false,
            std::array<FE::Real, 4>{0.0, 0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.0, 1.0}),
        std::invalid_argument);
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleValidatesWholeWallWithoutCurrentContact)
{
    constexpr int interface_marker = 169;
    constexpr int wall_marker = 59;

    const auto register_case = [&](bool compound_marker) {
        const auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(
            wall_marker, compound_marker);
        auto u_space = makeVelocitySpace(mesh);
        auto p_space = makePressureSpace(mesh);
        auto opts = baseNavierStokesOptions();
        opts.enable_convection = false;
        opts.velocity_dirichlet.push_back(
            ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                .boundary_marker = wall_marker,
                .value = {0.0, 0.0, 0.0},
                .active_components = {false, false, true},
            });
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi_dynamic_whole_wall",
                .active_domain =
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method =
                    ns::FreeSurfaceActiveDomainMethod::CutVolume,
                .surface_tension = 0.8,
                .curvature = 0.0,
                .use_level_set_curvature = false,
                .contact_lines = {
                    dynamicRenEContactLine(
                        wall_marker,
                        1.1,
                        {0.0, 0.0, -1.0},
                        0.5,
                        0.2)},
                .small_cut_aggregation = false,
            });

        FE::systems::FESystem system(mesh);
        system.addField(FE::systems::FieldSpec{
            .name = "phi_dynamic_whole_wall",
            .space = p_space,
            .components = 1,
        });
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        module.registerOn(system);
    };

    // No generated contact rule exists yet.  The complete wall marker still
    // has enough mesh geometry to validate and must be accepted.
    EXPECT_NO_THROW(register_case(/*compound_marker=*/false));

    // Faces 0..3 have different outward normals.  Giving all of them the
    // same marker must fail even though no current contact rule samples the
    // three incompatible faces.
    try {
        register_case(/*compound_marker=*/true);
        FAIL() << "Expected a mixed-normal wall marker to fail closed";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("requires every face"), std::string::npos)
            << message;
        EXPECT_NE(message.find("global_invalid_faces"), std::string::npos)
            << message;
    }
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleRejectsNontransverseGeneratedContact)
{
    constexpr FE::Real theta = 1.0471975511965977462;
    constexpr std::array<FE::Real, 4> zero_velocity{0.0, 0.0, 0.0, 0.0};
    try {
        (void)assembleDynamicContactAngleCase(
            theta,
            /*dynamic_angle=*/0.0,
            zero_velocity,
            /*include_dynamic_contact_angle=*/true);
        FAIL() << "Expected a wall-parallel interface normal to fail closed";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("not transverse to its wall"),
                  std::string::npos)
            << message;
        EXPECT_NE(message.find("transverse_projection"), std::string::npos)
            << message;
    }
}

TEST(MovingDomainPhysics,
     NavierStokesCurvatureTractionDynamicContactAngleAdvancingAndRecedingForcesHaveOppositeSigns)
{
    constexpr FE::Real pi = 3.14159265358979323846;
    constexpr FE::Real equilibrium_angle = pi / 2.0;
    constexpr std::array<FE::Real, 4> zero_velocity{0.0, 0.0, 0.0, 0.0};

    const auto resultant = [&](FE::Real dynamic_angle) {
        const auto dynamic = assembleDynamicContactAngleCase(
            equilibrium_angle,
            dynamic_angle,
            zero_velocity,
            /*include_dynamic_contact_angle=*/true);
        const auto baseline = assembleDynamicContactAngleCase(
            equilibrium_angle,
            dynamic_angle,
            zero_velocity,
            /*include_dynamic_contact_angle=*/false);
        FE::Real x_resultant = 0.0;
        for (const auto dof : dynamic.velocity_x_dofs) {
            x_resultant += dynamic.residual[static_cast<std::size_t>(dof)] -
                           baseline.residual[static_cast<std::size_t>(dof)];
        }
        return x_resultant;
    };

    const auto advancing_residual = resultant(2.0 * pi / 3.0);
    const auto receding_residual = resultant(pi / 3.0);
    constexpr FE::Real gamma = 0.8;
    constexpr FE::Real mobility = 0.5;
    const auto advancing_velocity = gamma * mobility *
        (std::cos(equilibrium_angle) - std::cos(2.0 * pi / 3.0));
    const auto receding_velocity = gamma * mobility *
        (std::cos(equilibrium_angle) - std::cos(pi / 3.0));
    // m points out of the wetted footprint: the Ren--E law predicts
    // advancing V_CL>0 for theta_d>theta_e and receding V_CL<0 for the
    // opposite ordering.  The residual at V_CL=0 has the driving-force sign
    // opposite to that velocity, as Newton balance requires.
    EXPECT_GT(advancing_velocity, 0.0);
    EXPECT_LT(receding_velocity, 0.0);
    EXPECT_LT(advancing_residual, -1.0e-6);
    EXPECT_GT(receding_residual, 1.0e-6);
    EXPECT_NEAR(advancing_residual, -receding_residual, 1.0e-11);
    constexpr FE::Real contact_measure = 0.125;
    const auto expected_advancing_residual =
        -gamma * (std::cos(equilibrium_angle) -
                  std::cos(2.0 * pi / 3.0)) *
        contact_measure;
    const auto expected_receding_residual =
        -gamma * (std::cos(equilibrium_angle) -
                  std::cos(pi / 3.0)) *
        contact_measure;
    EXPECT_NEAR(advancing_residual, expected_advancing_residual, 2.0e-11);
    EXPECT_NEAR(receding_residual, expected_receding_residual, 2.0e-11);
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleWallAndLineDissipationAreNonnegative)
{
    constexpr FE::Real half_pi = 1.57079632679489661923;
    constexpr FE::Real speed = 0.4;
    constexpr FE::Real mobility = 0.5;
    constexpr FE::Real contact_measure = 0.125;
    constexpr std::array<FE::Real, 4> velocity{
        speed, speed, speed, speed};
    const auto dynamic = assembleDynamicContactAngleCase(
        half_pi,
        half_pi,
        velocity,
        /*include_dynamic_contact_angle=*/true);
    const auto baseline = assembleDynamicContactAngleCase(
        half_pi,
        half_pi,
        velocity,
        /*include_dynamic_contact_angle=*/false);

    ASSERT_EQ(dynamic.residual.size(), baseline.residual.size());
    FE::Real total_dissipation = 0.0;
    for (std::size_t i = 0; i < dynamic.residual.size(); ++i) {
        total_dissipation +=
            dynamic.solution[i] *
            (dynamic.residual[i] - baseline.residual[i]);
    }
    const FE::Real line_dissipation =
        speed * speed * contact_measure / mobility;
    const FE::Real wall_dissipation =
        total_dissipation - line_dissipation;
    EXPECT_GT(line_dissipation, 0.0);
    EXPECT_GE(wall_dissipation, -2.0e-11);
    EXPECT_GT(total_dissipation, line_dissipation);
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleVelocityJacobianMatchesFiniteDifference)
{
    constexpr FE::Real theta = 1.0471975511965977462;
    constexpr FE::Real epsilon = 1.0e-7;
    const std::array<FE::Real, 4> velocity{0.20, 0.28, 0.17, 0.31};
    const auto dynamic = assembleDynamicContactAngleCase(
        theta,
        theta,
        velocity,
        /*include_dynamic_contact_angle=*/true,
        /*assemble_jacobian=*/true);
    const auto baseline = assembleDynamicContactAngleCase(
        theta,
        theta,
        velocity,
        /*include_dynamic_contact_angle=*/false,
        /*assemble_jacobian=*/true);
    ASSERT_TRUE(dynamic.has_velocity_level_set_coupling);
    ASSERT_EQ(dynamic.total_dofs, baseline.total_dofs);
    ASSERT_EQ(dynamic.jacobian.size(), baseline.jacobian.size());

    auto plus_velocity = velocity;
    auto minus_velocity = velocity;
    plus_velocity[1] += epsilon;
    minus_velocity[1] -= epsilon;
    const auto plus_dynamic = assembleDynamicContactAngleCase(
        theta, theta, plus_velocity, true);
    const auto plus_baseline = assembleDynamicContactAngleCase(
        theta, theta, plus_velocity, false);
    const auto minus_dynamic = assembleDynamicContactAngleCase(
        theta, theta, minus_velocity, true);
    const auto minus_baseline = assembleDynamicContactAngleCase(
        theta, theta, minus_velocity, false);

    const auto column = dynamic.velocity_x_dofs[1];
    FE::Real analytic_norm2 = 0.0;
    FE::Real error_norm2 = 0.0;
    for (FE::GlobalIndex row = 0; row < dynamic.total_dofs; ++row) {
        const auto matrix_index = static_cast<std::size_t>(
            row * dynamic.total_dofs + column);
        const auto analytic = dynamic.jacobian[matrix_index] -
                              baseline.jacobian[matrix_index];
        const auto row_index = static_cast<std::size_t>(row);
        const auto finite_difference =
            ((plus_dynamic.residual[row_index] -
              plus_baseline.residual[row_index]) -
             (minus_dynamic.residual[row_index] -
              minus_baseline.residual[row_index])) /
            (FE::Real{2.0} * epsilon);
        const auto error = analytic - finite_difference;
        EXPECT_NEAR(analytic, finite_difference, 2.0e-7)
            << "row=" << row;
        analytic_norm2 += analytic * analytic;
        error_norm2 += error * error;
    }
    const auto analytic_norm = std::sqrt(analytic_norm2);
    EXPECT_GT(analytic_norm, 1.0e-8);
    EXPECT_LT(std::sqrt(error_norm2),
              2.0e-5 * std::max(FE::Real{1.0}, analytic_norm));
}

TEST(MovingDomainPhysics,
     NavierStokesCurvatureTractionDynamicContactAngleLevelSetJacobianMatchesFiniteDifference)
{
    constexpr FE::Real equilibrium_angle = 1.15;
    constexpr FE::Real dynamic_angle = 0.92;
    constexpr FE::Real epsilon = 1.0e-7;
    const std::array<FE::Real, 4> velocity{0.18, 0.27, 0.11, 0.34};
    const auto dynamic = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        /*include_dynamic_contact_angle=*/true,
        /*assemble_jacobian=*/true);
    const auto baseline = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        /*include_dynamic_contact_angle=*/false,
        /*assemble_jacobian=*/true);
    ASSERT_TRUE(dynamic.has_velocity_level_set_coupling);
    ASSERT_EQ(dynamic.total_dofs, baseline.total_dofs);

    std::array<FE::Real, 4> plus_phi{0.0, epsilon, 0.0, 0.0};
    std::array<FE::Real, 4> minus_phi{0.0, -epsilon, 0.0, 0.0};
    const auto plus_dynamic = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        true,
        false,
        plus_phi);
    const auto plus_baseline = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        false,
        false,
        plus_phi);
    const auto minus_dynamic = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        true,
        false,
        minus_phi);
    const auto minus_baseline = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        false,
        false,
        minus_phi);

    const auto column = dynamic.level_set_dofs[1];
    FE::Real analytic_norm2 = 0.0;
    FE::Real error_norm2 = 0.0;
    for (FE::GlobalIndex row = 0; row < dynamic.total_dofs; ++row) {
        const auto matrix_index = static_cast<std::size_t>(
            row * dynamic.total_dofs + column);
        const auto analytic = dynamic.jacobian[matrix_index] -
                              baseline.jacobian[matrix_index];
        const auto row_index = static_cast<std::size_t>(row);
        const auto finite_difference =
            ((plus_dynamic.residual[row_index] -
              plus_baseline.residual[row_index]) -
             (minus_dynamic.residual[row_index] -
              minus_baseline.residual[row_index])) /
            (FE::Real{2.0} * epsilon);
        const auto error = analytic - finite_difference;
        EXPECT_NEAR(analytic, finite_difference, 5.0e-7)
            << "row=" << row
            << " velocity_offset=" << dynamic.velocity_offset
            << " level_set_offset=" << dynamic.level_set_dofs.front();
        analytic_norm2 += analytic * analytic;
        error_norm2 += error * error;
    }
    const auto analytic_norm = std::sqrt(analytic_norm2);
    EXPECT_GT(analytic_norm, 1.0e-8);
    EXPECT_LT(std::sqrt(error_norm2),
              5.0e-5 * std::max(FE::Real{1.0}, analytic_norm));
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleWettedWallIsInvariantToPhiRescaling)
{
    constexpr FE::Real equilibrium_angle = 1.18;
    constexpr FE::Real dynamic_angle = 0.87;
    constexpr FE::Real scale = 7.0;
    const std::array<FE::Real, 4> velocity{0.19, 0.31, 0.14, 0.27};
    constexpr std::array<FE::Real, 4> no_perturbation{0.0, 0.0, 0.0, 0.0};
    constexpr std::array<FE::Real, 3> outward_wall_normal{0.0, 0.0, -1.0};

    const auto unscaled_dynamic = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        /*include_dynamic_contact_angle=*/true,
        /*assemble_jacobian=*/true,
        no_perturbation,
        outward_wall_normal,
        /*level_set_scale=*/1.0);
    const auto unscaled_baseline = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        /*include_dynamic_contact_angle=*/false,
        /*assemble_jacobian=*/true,
        no_perturbation,
        outward_wall_normal,
        /*level_set_scale=*/1.0);
    const auto scaled_dynamic = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        /*include_dynamic_contact_angle=*/true,
        /*assemble_jacobian=*/true,
        no_perturbation,
        outward_wall_normal,
        scale);
    const auto scaled_baseline = assembleDynamicContactAngleCase(
        equilibrium_angle,
        dynamic_angle,
        velocity,
        /*include_dynamic_contact_angle=*/false,
        /*assemble_jacobian=*/true,
        no_perturbation,
        outward_wall_normal,
        scale);

    ASSERT_EQ(unscaled_dynamic.total_dofs, scaled_dynamic.total_dofs);
    FE::Real added_residual_norm2 = 0.0;
    for (FE::GlobalIndex row = 0; row < unscaled_dynamic.total_dofs; ++row) {
        const auto index = static_cast<std::size_t>(row);
        const auto unscaled_added =
            unscaled_dynamic.residual[index] -
            unscaled_baseline.residual[index];
        const auto scaled_added =
            scaled_dynamic.residual[index] - scaled_baseline.residual[index];
        EXPECT_NEAR(unscaled_added, scaled_added, 2.0e-11)
            << "residual row=" << row;
        added_residual_norm2 += unscaled_added * unscaled_added;
    }
    EXPECT_GT(std::sqrt(added_residual_norm2), 1.0e-8);

    // R(c phi)=R(phi) implies the level-set Jacobian transforms covariantly:
    // c*dR/d(c phi)=dR/dphi.  Check all four phi columns, not just a sampled
    // directional derivative.
    FE::Real phi_jacobian_norm2 = 0.0;
    for (const auto unscaled_column : unscaled_dynamic.level_set_dofs) {
        const auto local_column = static_cast<std::size_t>(
            std::find(unscaled_dynamic.level_set_dofs.begin(),
                      unscaled_dynamic.level_set_dofs.end(),
                      unscaled_column) -
            unscaled_dynamic.level_set_dofs.begin());
        const auto scaled_column =
            scaled_dynamic.level_set_dofs[local_column];
        for (FE::GlobalIndex row = 0; row < unscaled_dynamic.total_dofs;
             ++row) {
            const auto unscaled_index = static_cast<std::size_t>(
                row * unscaled_dynamic.total_dofs + unscaled_column);
            const auto scaled_index = static_cast<std::size_t>(
                row * scaled_dynamic.total_dofs + scaled_column);
            const auto unscaled_added =
                unscaled_dynamic.jacobian[unscaled_index] -
                unscaled_baseline.jacobian[unscaled_index];
            const auto scaled_added =
                scaled_dynamic.jacobian[scaled_index] -
                scaled_baseline.jacobian[scaled_index];
            EXPECT_NEAR(unscaled_added, scale * scaled_added, 2.0e-9)
                << "row=" << row << " phi_column=" << local_column
                << " velocity_offset=" << unscaled_dynamic.velocity_offset
                << " level_set_offset="
                << unscaled_dynamic.level_set_dofs.front();
            phi_jacobian_norm2 += unscaled_added * unscaled_added;
        }
    }
    EXPECT_GT(std::sqrt(phi_jacobian_norm2), 1.0e-8);
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleCompactIndicatorHasZeroDryWallResidual)
{
    constexpr FE::Real angle = 0.93;
    constexpr std::array<FE::Real, 4> velocity{0.4, 0.4, 0.4, 0.4};
    constexpr std::array<FE::Real, 4> no_perturbation{0.0, 0.0, 0.0, 0.0};
    constexpr std::array<FE::Real, 3> outward_wall_normal{0.0, 0.0, -1.0};
    constexpr FE::Real fully_dry_shift = 2.0;

    const auto dynamic = assembleDynamicContactAngleCase(
        angle,
        angle,
        velocity,
        /*include_dynamic_contact_angle=*/true,
        /*assemble_jacobian=*/true,
        no_perturbation,
        outward_wall_normal,
        /*level_set_scale=*/1.0,
        fully_dry_shift,
        /*velocity_component=*/1);
    const auto baseline = assembleDynamicContactAngleCase(
        angle,
        angle,
        velocity,
        /*include_dynamic_contact_angle=*/false,
        /*assemble_jacobian=*/true,
        no_perturbation,
        outward_wall_normal,
        /*level_set_scale=*/1.0,
        fully_dry_shift,
        /*velocity_component=*/1);

    ASSERT_EQ(dynamic.residual.size(), baseline.residual.size());
    FE::Real added_residual_norm2 = 0.0;
    for (std::size_t row = 0; row < dynamic.residual.size(); ++row) {
        const auto added = dynamic.residual[row] - baseline.residual[row];
        added_residual_norm2 += added * added;
    }
    EXPECT_NEAR(std::sqrt(added_residual_norm2), 0.0, 2.0e-12);

    ASSERT_EQ(dynamic.jacobian.size(), baseline.jacobian.size());
    FE::Real added_jacobian_norm2 = 0.0;
    // The synthetic contact rule remains present because dInterfaceBoundary
    // assembly requires a generated marker.  Velocity is wall-tangential and
    // orthogonal to the line footprint, so its selected y-y block isolates
    // the wetted-wall slip tangent from the x-directed line law.
    for (const auto row : dynamic.velocity_x_dofs) {
        for (const auto column : dynamic.velocity_x_dofs) {
            const auto entry = static_cast<std::size_t>(
                row * dynamic.total_dofs + column);
            const auto added =
                dynamic.jacobian[entry] - baseline.jacobian[entry];
            added_jacobian_norm2 += added * added;
        }
    }
    EXPECT_NEAR(std::sqrt(added_jacobian_norm2), 0.0, 2.0e-12);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleResidualFollowsActiveDomainSign)
{
    constexpr FE::Real half_pi = 1.57079632679489661923;
    constexpr FE::Real inv_sqrt_two = 0.70710678118654752440;
    const std::array<FE::Real, 3> transverse_gradient{
        inv_sqrt_two, 0.0, inv_sqrt_two};
    const auto negative_active = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetNegative,
        half_pi,
        transverse_gradient);
    const auto positive_active = unfittedContactAngleResidualVector(
        ns::FreeSurfaceActiveDomain::LevelSetPositive,
        half_pi,
        transverse_gradient);

    ASSERT_EQ(negative_active.size(), positive_active.size());
    EXPECT_GT(vectorNorm(negative_active), 1.0e-12);
    EXPECT_GT(vectorNorm(positive_active), 1.0e-12);
    for (std::size_t i = 0; i < negative_active.size(); ++i) {
        EXPECT_NEAR(negative_active[i], -positive_active[i], 1.0e-12);
    }
}

TEST(MovingDomainPhysics, NavierStokesFreeSurfaceRejectsInvalidScalarParameters)
{
    constexpr int marker = 145;
    const auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    const auto expect_rejected = [&](FE::Real external_pressure,
                                     FE::Real surface_tension,
                                     FE::Real kinematic_penalty,
                                     bool enable_penalty) {
        auto opts = baseNavierStokesOptions();
        opts.enable_ale = enable_penalty;
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation = ns::FreeSurfaceImplementation::FittedALE,
                .boundary_marker = marker,
                .external_pressure = external_pressure,
                .surface_tension = surface_tension,
                .kinematic_enforcement = enable_penalty
                    ? ns::FreeSurfaceKinematicEnforcement::Penalty
                    : ns::FreeSurfaceKinematicEnforcement::None,
                .kinematic_penalty = kinematic_penalty,
            });
        FE::systems::FESystem system(mesh);
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    };

    expect_rejected(std::numeric_limits<FE::Real>::infinity(), 0.0, 0.0, false);
    expect_rejected(0.0, -0.01, 0.0, false);
    expect_rejected(0.0, 0.0, 0.0, true);
    expect_rejected(0.0, 0.0, -1.0, true);
}

TEST(MovingDomainPhysics, NavierStokesFreeSurfaceRejectsVariableSurfaceTensionWithoutMarangoniTraction)
{
    constexpr int marker = 146;
    const auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::FittedALE,
            .boundary_marker = marker,
            .surface_tension = FE::forms::ScalarCoefficient{
                [](FE::Real x, FE::Real, FE::Real) {
                    return FE::Real{0.07} + FE::Real{0.01} * x;
                }},
        });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesContactAngleRejectsInvalidParameters)
{
    constexpr int interface_marker = 155;
    constexpr int wall_marker = 25;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    using ContactLine =
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;
    const auto expect_rejected = [&](ContactLine contact_line) {
        auto opts = baseNavierStokesOptions();
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi",
                .active_domain =
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .contact_lines = {std::move(contact_line)},
            });
        FE::systems::FESystem system(mesh);
        system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = p_space,
            .components = 1,
        });
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    };

    auto invalid_angle = prescribedContactLine(
        wall_marker, 3.2, {1.0, 0.0, 0.0});
    expect_rejected(invalid_angle);

    auto zero_angle = invalid_angle;
    prescribedContactConfiguration(zero_angle).contact_angle_radians = 0.0;
    expect_rejected(zero_angle);

    auto complete_wetting_angle = invalid_angle;
    prescribedContactConfiguration(complete_wetting_angle)
        .contact_angle_radians =
        FE::Real{3.14159265358979323846};
    expect_rejected(complete_wetting_angle);

    auto zero_penalty = invalid_angle;
    prescribedContactConfiguration(zero_penalty).contact_angle_radians = 1.0;
    prescribedContactConfiguration(zero_penalty).contact_angle_penalty = 0.0;
    expect_rejected(zero_penalty);

    auto variable_penalty = invalid_angle;
    prescribedContactConfiguration(variable_penalty).contact_angle_radians =
        1.0;
    prescribedContactConfiguration(variable_penalty).contact_angle_penalty =
        FE::forms::ScalarCoefficient{
        [](FE::Real x, FE::Real, FE::Real) {
            return FE::Real{1.0} + x;
        }};
    expect_rejected(variable_penalty);

    auto nonfinite_normal = invalid_angle;
    prescribedContactConfiguration(nonfinite_normal).contact_angle_radians =
        1.0;
    prescribedContactConfiguration(nonfinite_normal).wall_normal[0] =
        std::numeric_limits<FE::Real>::quiet_NaN();
    expect_rejected(nonfinite_normal);
}

TEST(MovingDomainPhysics, NavierStokesPrescribedContactAngleRejectsOutOfPlaneNormal)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr int interface_marker = 156;
    constexpr int wall_marker = 26;
    const auto mesh = makeRegistryQuadMesh();
    auto u_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4,
        /*order=*/1);
    auto opts = baseNavierStokesOptions();
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_2d_contact",
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .contact_lines = {
                prescribedContactLine(
                    wall_marker, 1.0, {1.0, 0.0, 1.0})},
        });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi_2d_contact",
        .space = p_space,
        .components = 1,
    });
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
#endif
}

TEST(MovingDomainPhysics,
     NavierStokesPrescribedContactAngleValidatesMappedPhysicalWallNormal)
{
    constexpr int interface_marker = 157;
    constexpr int wall_marker = 27;
    constexpr FE::Real inv_sqrt_five =
        FE::Real{0.44721359549995793928};

    const auto install_with_normal = [&](std::array<FE::Real, 3> wall_normal,
                                         FE::Real coordinate_scale = FE::Real{1.0}) {
        const auto mesh =
            std::make_shared<SingleTetraBoundaryMeshAccess>(wall_marker);
        // J has first column scale*(1,0,1/2). Therefore the reference covector
        // (0,0,1) carried by the generated rule maps with J^{-T} to
        // (-1/2,0,1), normalized below.
        mesh->setCurrentNodeCoordinates(1, {coordinate_scale, 0.0,
                                            FE::Real{0.5} * coordinate_scale});
        mesh->setCurrentNodeCoordinates(2, {0.0, coordinate_scale, 0.0});
        mesh->setCurrentNodeCoordinates(3, {0.0, 0.0, coordinate_scale});
        auto u_space = makeVelocitySpace(mesh);
        auto p_space = makePressureSpace(mesh);
        auto opts = baseNavierStokesOptions();
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi_mapped_wall_normal",
                .active_domain =
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .contact_lines = {
                    prescribedContactLine(
                        wall_marker, 1.0, wall_normal, 1.0)},
            });

        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi_mapped_wall_normal",
            .space = p_space,
            .components = 1,
        });
        system.addOperator("level_set_owner");
        const auto phi_state = FE::forms::StateField(
            phi, *p_space, "phi_mapped_wall_normal_owner");
        const auto eta = FE::forms::TestField(
            phi, *p_space, "eta_mapped_wall_normal_owner");
        (void)FE::systems::installFormulation(
            system,
            "level_set_owner",
            {phi},
            (phi_state * eta).dx());
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        module.registerOn(system);
        const int contact_marker =
            stableContactLineMarker(phi, interface_marker, wall_marker);
        system.setCutIntegrationContext(makeSingleTetraContactLineCutContext(
            interface_marker,
            wall_marker,
            contact_marker,
            phi,
            {0.0, 0.0, 1.0}));
    };

    EXPECT_NO_THROW(install_with_normal(
        {-inv_sqrt_five, 0.0, 2.0 * inv_sqrt_five}));
    EXPECT_NO_THROW(install_with_normal(
        {-inv_sqrt_five, 0.0, 2.0 * inv_sqrt_five}, FE::Real{1e-16}));
    EXPECT_THROW(install_with_normal({0.0, 0.0, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(install_with_normal(
                     {inv_sqrt_five, 0.0, -2.0 * inv_sqrt_five}),
                 std::invalid_argument);
}

TEST(MovingDomainPhysics,
     NavierStokesDynamicContactAngleRejectsIncompatibleConfigurations)
{
    constexpr int interface_marker = 168;
    constexpr int wall_marker = 58;
    const auto mesh =
        std::make_shared<SingleTetraBoundaryMeshAccess>(wall_marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    const auto valid_options = [&]() {
        auto opts = baseNavierStokesOptions();
        opts.enable_convection = false;
        opts.velocity_dirichlet.push_back(
            ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                .boundary_marker = wall_marker,
                .value = {0.0, 0.0, 0.0},
                .active_components = {false, false, true},
            });
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary fs;
        fs.implementation =
            ns::FreeSurfaceImplementation::UnfittedLevelSet;
        fs.interface_marker = interface_marker;
        fs.level_set_field_name = "phi_dynamic_validation";
        fs.active_domain =
            ns::FreeSurfaceActiveDomain::LevelSetNegative;
        fs.active_domain_method =
            ns::FreeSurfaceActiveDomainMethod::CutVolume;
        fs.surface_tension = 0.8;
        fs.curvature = 0.0;
        fs.use_level_set_curvature = false;
        fs.small_cut_aggregation = false;
        fs.contact_lines.push_back(dynamicRenEContactLine(
            wall_marker, 1.1, {0.0, 0.0, -1.0}, 0.5, 0.2));
        opts.free_surface.push_back(std::move(fs));
        return opts;
    };
    const auto register_options = [&](auto opts) {
        FE::systems::FESystem system(mesh);
        system.addField(FE::systems::FieldSpec{
            .name = "phi_dynamic_validation",
            .space = p_space,
            .components = 1,
        });
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, std::move(opts));
        module.registerOn(system);
    };
    ASSERT_NO_THROW(register_options(valid_options()));

    constexpr FE::Real pi = 3.14159265358979323846;
    const auto expect_transversality_angle_rejected =
        [&](FE::Real angle) {
            auto opts = valid_options();
            dynamicContactConfiguration(
                opts.free_surface.front().contact_lines.front())
                .equilibrium_contact_angle_radians = angle;
            try {
                register_options(std::move(opts));
                FAIL() << "Expected near-endpoint equilibrium angle to fail "
                          "the transverse-contact contract";
            } catch (const std::invalid_argument& error) {
                const std::string message = error.what();
                EXPECT_NE(message.find("sin(theta_e)"), std::string::npos)
                    << message;
                EXPECT_NE(message.find("minimum_transverse_sine"),
                          std::string::npos)
                    << message;
            }
        };
    expect_transversality_angle_rejected(FE::Real{0.5e-6});
    expect_transversality_angle_rejected(pi - FE::Real{0.5e-6});

    for (const auto supported_angle :
         {FE::Real{2.0e-6}, pi - FE::Real{2.0e-6}}) {
        auto opts = valid_options();
        dynamicContactConfiguration(
            opts.free_surface.front().contact_lines.front())
            .equilibrium_contact_angle_radians = supported_angle;
        EXPECT_NO_THROW(register_options(std::move(opts)))
            << "supported_angle=" << supported_angle;
    }

    const auto expect_rejected = [&](auto mutate) {
        auto opts = valid_options();
        mutate(opts);
        EXPECT_THROW(register_options(std::move(opts)), std::invalid_argument);
    };

    expect_rejected([](auto& opts) {
        opts.free_surface.front().surface_tension = 0.0;
    });
    expect_rejected([](auto& opts) {
        dynamicContactConfiguration(
            opts.free_surface.front().contact_lines.front()).mobility = 0.0;
    });
    expect_rejected([](auto& opts) {
        dynamicContactConfiguration(
            opts.free_surface.front().contact_lines.front()).slip_length = 0.0;
    });
    expect_rejected([](auto& opts) {
        dynamicContactConfiguration(
            opts.free_surface.front().contact_lines.front())
            .equilibrium_contact_angle_radians = 0.0;
    });
    expect_rejected([&](auto& opts) {
        dynamicContactConfiguration(
            opts.free_surface.front().contact_lines.front())
            .equilibrium_contact_angle_radians = pi;
    });
    expect_rejected([](auto& opts) {
        opts.free_surface.front().generated_interface_geometry =
            "HighOrderImplicit";
    });
    expect_rejected([](auto& opts) {
        opts.free_surface.front().active_domain_method =
            ns::FreeSurfaceActiveDomainMethod::SmoothedIndicator;
    });
    expect_rejected([](auto& opts) {
        opts.velocity_dirichlet.clear();
    });
    expect_rejected([](auto& opts) {
        opts.velocity_dirichlet.front().active_components =
            {true, true, true};
    });
    expect_rejected([](auto& opts) {
        opts.velocity_dirichlet_weak.push_back(
            opts.velocity_dirichlet.front());
    });
    expect_rejected([](auto& opts) {
        constexpr FE::Real inv_sqrt_two = 0.70710678118654752440;
        dynamicContactConfiguration(
            opts.free_surface.front().contact_lines.front()).wall_normal =
                {inv_sqrt_two, 0.0, inv_sqrt_two};
    });
    expect_rejected([](auto& opts) {
        opts.free_surface.front().contact_lines.push_back(
            opts.free_surface.front().contact_lines.front());
    });
    expect_rejected([](auto& opts) {
        const auto wall_marker = dynamicContactConfiguration(
            opts.free_surface.front().contact_lines.front())
                                     .wall_boundary_marker;
        opts.free_surface.front().contact_lines.push_back(
            prescribedContactLine(
                wall_marker, 1.0, {0.0, 0.0, -1.0}));
    });
    expect_rejected([](auto& opts) {
        auto& fs = opts.free_surface.front();
        fs.implementation = ns::FreeSurfaceImplementation::FittedALE;
        fs.boundary_marker = 58;
        fs.active_domain = ns::FreeSurfaceActiveDomain::None;
    });
}

TEST(MovingDomainPhysics,
     NavierStokesRejectsDuplicatePrescribedWallOwnershipWithUniqueMarkers)
{
    constexpr int interface_marker = 216;
    constexpr int wall_marker = 59;
    const auto mesh =
        std::make_shared<SingleTetraBoundaryMeshAccess>(wall_marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation =
                ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_duplicate_prescribed_wall",
            .active_domain =
                ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .contact_lines = {
                prescribedContactLine(
                    wall_marker, 1.0, {0.0, 0.0, -1.0}, 1.0, 2216),
                prescribedContactLine(
                    wall_marker, 1.1, {0.0, 0.0, -1.0}, 1.0, 2217),
            },
        });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi_duplicate_prescribed_wall",
        .space = p_space,
        .components = 1,
    });
    ns::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, std::move(opts));
    try {
        module.registerOn(system);
        FAIL() << "duplicate prescribed wall ownership must fail closed";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("exactly one contact-line model"),
                  std::string::npos)
            << message;
    }
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleRequiresWallMarker)
{
    constexpr int interface_marker = 46;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .contact_lines = {
            prescribedContactLine(
                /*wall_marker=*/-1,
                1.0471975511965977462,
                {1.0, 0.0, 0.0}),
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleHonorsExplicitUniqueMarker)
{
    constexpr int interface_marker = 47;
    constexpr int wall_marker = 13;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .contact_lines = {
            prescribedContactLine(
                wall_marker,
                1.0471975511965977462,
                {1.0, 0.0, 0.0},
                1.0),
        },
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });
    prescribedContactConfiguration(
        opts.free_surface.front().contact_lines.front())
        .contact_line_marker =
        stableContactLineMarker(phi, interface_marker, wall_marker);
    system.addOperator("level_set_owner");
    const auto phi_state =
        FE::forms::StateField(phi, *p_space, "phi_explicit_marker_owner");
    const auto eta =
        FE::forms::TestField(phi, *p_space, "eta_explicit_marker_owner");
    (void)FE::systems::installFormulation(
        system,
        "level_set_owner",
        {phi},
        (phi_state * eta).dx());

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_NO_THROW(module.registerOn(system));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedContactMarkersRejectDefinitionTimeCollision)
{
    constexpr int interface_marker = 147;
    constexpr int explicit_contact_marker = 2099;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_collision",
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .contact_lines = {
                prescribedContactLine(
                    31,
                    1.0,
                    {1.0, 0.0, 0.0},
                    1.0,
                    explicit_contact_marker),
                prescribedContactLine(
                    32,
                    1.0,
                    {0.0, 1.0, 0.0},
                    1.0,
                    explicit_contact_marker),
            },
        });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi_collision",
        .space = p_space,
        .components = 1,
    });
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, FreeSurfaceContactAlternativesExcludeCrossModelState)
{
    EXPECT_FALSE(
        (HasContactMobility<FreeSurfaceContactLine::PrescribedAngle>));
    EXPECT_FALSE(
        (HasContactSlipLength<FreeSurfaceContactLine::PrescribedAngle>));
    EXPECT_FALSE(
        (HasContactAnglePenalty<FreeSurfaceContactLine::DynamicRenE>));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleRequiresLevelSetUnknown)
{
    constexpr int interface_marker = 45;
    constexpr int wall_marker = 15;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .contact_lines = {
            prescribedContactLine(
                wall_marker,
                1.0471975511965977462,
                {1.0, 0.0, 0.0}),
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleRequiresLinearP1Geometry)
{
    constexpr int interface_marker = 49;
    constexpr int wall_marker = 16;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p1_space = makePressureSpace(mesh);

    const auto make_options = [&]() {
        auto opts = baseNavierStokesOptions();
        opts.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
                .implementation =
                    ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi_contact_geometry",
                .active_domain =
                    ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .contact_lines = {
                    prescribedContactLine(
                        wall_marker, 1.0, {1.0, 0.0, 0.0})},
            });
        return opts;
    };

    const auto expect_rejected = [&](auto phi_space, auto mutate) {
        auto opts = make_options();
        mutate(opts.free_surface.front());
        FE::systems::FESystem system(mesh);
        system.addField(FE::systems::FieldSpec{
            .name = "phi_contact_geometry",
            .space = std::move(phi_space),
            .components = 1,
        });
        ns::IncompressibleNavierStokesVMSModule module(
            u_space, p1_space, std::move(opts));
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    };

    expect_rejected(p1_space, [](auto& free_surface) {
        free_surface.generated_interface_geometry = "HighOrderImplicit";
    });
    auto p2_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);
    expect_rejected(std::move(p2_space), [](auto&) {});
    auto discontinuous_p1_space = FE::spaces::Space(
        FE::spaces::SpaceType::L2, mesh, /*order=*/1, /*components=*/1);
    expect_rejected(std::move(discontinuous_p1_space), [](auto&) {});
}

TEST(MovingDomainPhysics, NavierStokesCoupledALEDerivesMeshVelocityFromDisplacement)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";

    FE::systems::FESystem system(mesh);
    const auto displacement =
        system.addField(FE::systems::FieldSpec{.name = "mesh_displacement",
                                               .space = u_space,
                                               .components = 3});

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(displacement).source_kind,
              FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).source_kind,
              FE::systems::FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).derived.source_field, displacement);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).derived.role,
              FE::systems::DerivedFieldRole::TimeDerivative);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity_id));
    EXPECT_TRUE(system.geometricNonlinearityPolicy().enabled);
    EXPECT_TRUE(system.geometricNonlinearityPolicy().update_current_coordinates_on_trial);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity),
              mesh_velocity_id);

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
    EXPECT_EQ(system.fieldMap().numFields(), 3u);
    ASSERT_NE(system.blockMap(), nullptr);
    EXPECT_EQ(system.blockMap()->numBlocks(), 3u);

    bool has_fluid_mesh_coupling = false;
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (trial_field == displacement &&
                (test_field == system.findFieldByName(opts.velocity_field_name) ||
                 test_field == system.findFieldByName(opts.pressure_field_name))) {
                has_fluid_mesh_coupling = true;
            }
        }
    }
    EXPECT_TRUE(has_fluid_mesh_coupling);
}

TEST(MovingDomainPhysics, NavierStokesCoupledALEAcceptsADReferenceTangentPathOverride)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";
    opts.moving_mesh_tangent_path = FE::forms::GeometryTangentPath::ADReference;

    FE::systems::FESystem system(mesh);
    const auto displacement =
        system.addField(FE::systems::FieldSpec{.name = "mesh_displacement",
                                               .space = u_space,
                                               .components = 3});

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_NE(system.findFieldByName("mesh_velocity"), FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
}

TEST(MovingDomainPhysics, MixedFluidMeshBoundaryGeometryResidualMatchesFiniteDifference)
{
    using namespace FE::forms;

    constexpr int marker = 37;
    constexpr std::string_view op = "free_surface_boundary";
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto velocity_space = makeVelocitySpace(mesh);
    auto displacement_space = makeVelocitySpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto velocity = system.addField(FE::systems::FieldSpec{
        .name = "fluid_velocity",
        .space = velocity_space,
        .components = 3,
    });
    const auto displacement = system.addField(FE::systems::FieldSpec{
        .name = "mesh_displacement",
        .space = displacement_space,
        .components = 3,
    });
    system.bindMeshMotionField(FE::systems::MeshMotionFieldRole::Displacement,
                               displacement);
    auto geometry_policy = system.geometricNonlinearityPolicy();
    geometry_policy.enabled = true;
    geometry_policy.update_current_coordinates_on_trial = true;
    system.setGeometricNonlinearityPolicy(geometry_policy);
    system.addOperator(std::string(op));

    const auto u = FormExpr::stateField(velocity, *velocity_space, "u");
    const auto v = FormExpr::testFunction(velocity, *velocity_space, "v");
    const auto normal = currentNormal();
    const auto residual =
        (dot(u, normal) * dot(v, normal) * currentMeasure()).ds(marker);

    FE::systems::FormInstallOptions install;
    install.compiler_options.geometry_sensitivity.mode =
        GeometrySensitivityMode::MeshMotionUnknowns;
    install.compiler_options.geometry_sensitivity.mesh_motion_field = displacement;
    install.compiler_options.geometry_tangent_path = GeometryTangentPath::SymbolicRequired;
    install.extra_trial_fields.push_back(displacement);
    const auto kernels =
        FE::systems::installFormulation(system, std::string(op), {velocity}, residual, install);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));
    ASSERT_EQ(kernels.jacobian_blocks.size(), 1u);
    ASSERT_EQ(kernels.jacobian_blocks.front().size(), 2u);
    EXPECT_NE(kernels.jacobian_blocks.front()[0], nullptr);
    EXPECT_NE(kernels.jacobian_blocks.front()[1], nullptr);

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    ASSERT_EQ(system.dofHandler().getNumDofs(), 24);

    bool has_fluid_mesh_block = false;
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (test_field == velocity && trial_field == displacement) {
                has_fluid_mesh_block = true;
            }
        }
    }
    EXPECT_TRUE(has_fluid_mesh_block);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(solution, system, velocity, vertex, 0,
                               FE::Real(0.35) + FE::Real(0.03) * x);
        setFieldComponentValue(solution, system, velocity, vertex, 1,
                               FE::Real(-0.20) + FE::Real(0.02) * x);
        setFieldComponentValue(solution, system, velocity, vertex, 2,
                               FE::Real(0.45) - FE::Real(0.015) * x);

        setFieldComponentValue(solution, system, displacement, vertex, 0,
                               FE::Real(0.04) + FE::Real(0.006) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 1,
                               FE::Real(-0.025) + FE::Real(0.004) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 2,
                               FE::Real(0.03) - FE::Real(0.005) * x);
    }

    expectOperatorJacobianMatchesMovingBoundaryFD(
        system, *mesh, displacement, solution, op,
        /*eps=*/1.0e-7, /*rtol=*/2.0e-5, /*atol=*/2.0e-7);
}

TEST(MovingDomainPhysics, NavierStokesWeakVelocityNitschePenaltyUsesTraceHeight)
{
    constexpr int marker = 21;
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Tetra4,
        /*order=*/2,
        /*components=*/3);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Tetra4,
        /*order=*/1);

    auto opts = baseNavierStokesOptions();
    opts.velocity_dirichlet_weak.push_back(ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
        .boundary_marker = marker,
        .value = {0.0, 0.0, 0.0},
    });
    opts.nitsche_gamma = 8.0;
    opts.nitsche_scale_with_p = true;

    const auto u = FormExpr::trialFunction(*u_space, "u");
    const auto p = FormExpr::trialFunction(*p_space, "p");
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    const auto mu = FormExpr::constant(0.04);
    auto momentum_form = FormExpr::constant(0.0).dx();
    auto continuity_form = FormExpr::constant(0.0).dx();

    ns::Factories::applyVelocityNitscheBCs(
        momentum_form,
        continuity_form,
        opts,
        /*dim=*/3,
        u,
        p,
        v,
        q,
        mu);

    EXPECT_TRUE(containsExprType(momentum_form, FormExprType::CellVolume));
    EXPECT_TRUE(containsExprType(momentum_form, FormExprType::FacetArea));
    EXPECT_FALSE(containsExprType(momentum_form, FormExprType::CellDiameter));
}

TEST(MovingDomainPhysics, NavierStokesVMS2DUsesPhysicalMetricShape)
{
    auto mesh = std::make_shared<FE::forms::test::SingleTriangleMeshAccess>();
    auto u_space = FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto p_space = FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;
    opts.enable_convection = false;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));
    ASSERT_EQ(system.dofHandler().getNumDofs(), 9);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRegistersDisplacementUnknownOnly)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.field_name = "mesh_displacement";
    opts.operator_tag = "mesh_motion";
    opts.kappa = 2.0;

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);

    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_FALSE(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity).has_value());
    EXPECT_FALSE(system.hasField("mesh_velocity"));
    EXPECT_EQ(system.fieldRecord(displacement).source_kind,
              FE::systems::FieldSourceKind::Unknown);
    EXPECT_TRUE(system.fieldParticipatesInUnknownVector(displacement));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 12);
    EXPECT_EQ(system.fieldMap().numFields(), 1u);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionWithSpatialKappaMatchesFiniteDifference)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    opts.kappa = FE::forms::ScalarCoefficient{
        [](FE::Real x, FE::Real y, FE::Real z) {
            return 1.0 + x + 0.25 * y + 0.125 * z;
        }};

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 12);

    std::vector<FE::Real> u(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = static_cast<FE::Real>(0.01 * (static_cast<int>(i) - 5));
    }

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    expectOperatorJacobianMatchesCentralFD(
        system, state, "mesh_motion", /*eps=*/1e-6, /*rtol=*/1e-6, /*atol=*/1e-10);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionNaturalBoundaryLoadAssembles)
{
    constexpr int marker = 7;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    mm::HarmonicMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {1.0, 0.0, 0.0};
    opts.natural.push_back(natural);

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRobinBoundarySpringAssembles)
{
    constexpr int marker = 9;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 4.0;
    robin.target = {0.0, 1.0, 0.0};
    opts.robin.push_back(robin);

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionNormalConstraintAcceptsVelocityTarget)
{
    constexpr int marker = 10;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    mm::NormalConstraintBC normal;
    normal.boundary_marker = marker;
    normal.quantity = mm::NormalConstraintQuantity::Velocity;
    normal.target = 2.0;
    normal.velocity_time_scale = 0.25;
    normal.penalty = 6.0;
    opts.normal_constraint.push_back(normal);

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Normal));
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionTangentialPoliciesSelectBoundaryTerms)
{
    constexpr int marker = 11;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    const auto registers_boundary_integral =
        [&](mm::TangentialMeshPolicy policy) {
            mm::HarmonicMeshMotionOptions opts;
            opts.operator_tag = "mesh_motion";
            mm::TangentialPolicyBC tangent;
            tangent.boundary_marker = marker;
            tangent.policy = policy;
            tangent.quantity = mm::TangentialConstraintQuantity::Velocity;
            tangent.target = {1.0, 0.5, 0.0};
            tangent.velocity_time_scale = 0.25;
            tangent.penalty = 8.0;
            opts.tangential_policy.push_back(tangent);

            FE::systems::FESystem system(mesh);
            mm::HarmonicMeshMotionModule module(d_space, opts);
            module.registerOn(system);
            return formulationRecordsContain(system, FormExprType::BoundaryIntegral);
        };

    EXPECT_FALSE(registers_boundary_integral(mm::TangentialMeshPolicy::Free));
    EXPECT_FALSE(registers_boundary_integral(mm::TangentialMeshPolicy::SmoothingOnly));
    EXPECT_TRUE(registers_boundary_integral(mm::TangentialMeshPolicy::Prescribed));
}

TEST(MovingDomainPhysics, HarmonicMeshMotionWeakBoundaryTermsOnSameMarkerAreAdditive)
{
    constexpr int marker = 12;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {1.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 2.0;
    robin.target = {0.0, 1.0, 0.0};

    const auto assemble = [&](const mm::HarmonicMeshMotionOptions& opts) {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, opts);
        module.registerOn(system);
        system.setup({}, makeSingleTetraSetupInputs());

        std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        FE::systems::SystemStateView state;
        state.u = std::span<const FE::Real>(u);
        return residualVector(system, state, "mesh_motion");
    };

    mm::HarmonicMeshMotionOptions combined_opts;
    combined_opts.operator_tag = "mesh_motion";
    combined_opts.natural.push_back(natural);
    combined_opts.robin.push_back(robin);

    mm::HarmonicMeshMotionOptions equivalent_opts;
    equivalent_opts.operator_tag = "mesh_motion";
    auto equivalent_robin = robin;
    // NaturalBC adds to RobinBC's boundary RHS, so alpha * target_x increases by 1.
    equivalent_robin.target = {0.5, 1.0, 0.0};
    equivalent_opts.robin.push_back(equivalent_robin);

    const auto combined_residual = assemble(combined_opts);
    const auto equivalent_residual = assemble(equivalent_opts);

    ASSERT_EQ(combined_residual.size(), equivalent_residual.size());
    for (std::size_t i = 0; i < combined_residual.size(); ++i) {
        EXPECT_NEAR(combined_residual[i],
                    equivalent_residual[i],
                    1.0e-12);
    }
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRobinTargetMatchesEquivalentNaturalLoad)
{
    constexpr int marker = 13;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 4.0;
    robin.target = {0.0, 1.5, -0.5};

    mm::HarmonicMeshMotionOptions robin_opts;
    robin_opts.operator_tag = "mesh_motion";
    robin_opts.robin.push_back(robin);

    mm::HarmonicMeshMotionOptions::RobinBC homogeneous_robin = robin;
    homogeneous_robin.target = {0.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions::NaturalBC equivalent_load;
    equivalent_load.boundary_marker = marker;
    equivalent_load.value = {0.0, 6.0, -2.0};

    mm::HarmonicMeshMotionOptions split_opts;
    split_opts.operator_tag = "mesh_motion";
    split_opts.robin.push_back(homogeneous_robin);
    split_opts.natural.push_back(equivalent_load);

    const auto assemble = [&](const mm::HarmonicMeshMotionOptions& opts) {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, opts);
        module.registerOn(system);
        system.setup({}, makeSingleTetraSetupInputs());

        const auto n = system.dofHandler().getNumDofs();
        std::vector<FE::Real> u(static_cast<std::size_t>(n), 0.0);
        for (std::size_t i = 0; i < u.size(); ++i) {
            u[i] = static_cast<FE::Real>(0.01 * (static_cast<int>(i) + 1));
        }

        FE::systems::SystemStateView state;
        state.u = std::span<const FE::Real>(u);
        return residualVector(system, state, "mesh_motion");
    };

    const auto robin_residual = assemble(robin_opts);
    const auto split_residual = assemble(split_opts);

    ASSERT_EQ(robin_residual.size(), split_residual.size());
    for (std::size_t i = 0; i < robin_residual.size(); ++i) {
        EXPECT_NEAR(robin_residual[i], split_residual[i], 1.0e-12);
    }
}

TEST(MovingDomainPhysics, HarmonicMeshMotionDirichletConflictsWithWeakBoundaryTerms)
{
    constexpr int marker = 14;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions::DirichletBC dirichlet;
    dirichlet.boundary_marker = marker;
    dirichlet.value = {0.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {1.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions natural_conflict;
    natural_conflict.operator_tag = "mesh_motion";
    natural_conflict.dirichlet.push_back(dirichlet);
    natural_conflict.natural.push_back(natural);
    {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, natural_conflict);
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    }

    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 3.0;
    robin.target = {0.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions robin_conflict;
    robin_conflict.operator_tag = "mesh_motion";
    robin_conflict.dirichlet.push_back(dirichlet);
    robin_conflict.robin.push_back(robin);
    {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, robin_conflict);
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    }
}

TEST(MovingDomainPhysics, MeshMotionDirichletComponentCoefficientNamesUseComponentStyle)
{
    constexpr int marker = 15;
    const std::array<mm::HarmonicMeshMotionOptions::ScalarValue, 3> values = {
        mm::HarmonicMeshMotionOptions::ScalarValue{FE::forms::ScalarCoefficient(
            [](FE::Real, FE::Real, FE::Real) { return 1.0; })},
        mm::HarmonicMeshMotionOptions::ScalarValue{FE::forms::ScalarCoefficient(
            [](FE::Real, FE::Real, FE::Real) { return 2.0; })},
        mm::HarmonicMeshMotionOptions::ScalarValue{FE::forms::ScalarCoefficient(
            [](FE::Real, FE::Real, FE::Real) { return 3.0; })},
    };

    auto components = FE::forms::bc::toVectorExpr(
        values,
        /*dim=*/3,
        "mesh_displacement",
        marker,
        FE::forms::bc::ComponentValueNameStyle::Component);
    FE::forms::bc::EssentialBC bc(marker, std::move(components), "d_mesh");
    const auto strong = bc.getStrongConstraints(/*field_id=*/123);

    ASSERT_EQ(strong.size(), 3u);
    EXPECT_EQ(strong[0].value.toString(), "mesh_displacement_15_c0");
    EXPECT_EQ(strong[1].value.toString(), "mesh_displacement_15_c1");
    EXPECT_EQ(strong[2].value.toString(), "mesh_displacement_15_c2");
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRejectsInvalidLiteralParameters)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    const auto expect_invalid = [&](const mm::HarmonicMeshMotionOptions& opts,
                                    std::string_view expected_message) {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, opts);
        try {
            module.registerOn(system);
            FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument& ex) {
            EXPECT_NE(std::string(ex.what()).find(expected_message), std::string::npos)
                << ex.what();
        }
    };

    mm::HarmonicMeshMotionOptions zero_kappa;
    zero_kappa.kappa = 0.0;
    expect_invalid(zero_kappa, "kappa must be positive");

    mm::HarmonicMeshMotionOptions negative_kappa;
    negative_kappa.kappa = -1.0;
    expect_invalid(negative_kappa, "kappa must be positive");

    mm::HarmonicMeshMotionOptions zero_stiffness;
    zero_stiffness.stiffness = 0.0;
    expect_invalid(zero_stiffness, "stiffness must be positive");

    mm::HarmonicMeshMotionOptions negative_stiffness;
    negative_stiffness.stiffness = -1.0;
    expect_invalid(negative_stiffness, "stiffness must be positive");

    mm::HarmonicMeshMotionOptions conflicting_literals;
    conflicting_literals.kappa = 2.0;
    conflicting_literals.stiffness = 3.0;
    expect_invalid(conflicting_literals,
                   "both kappa and deprecated stiffness were set");
}

TEST(MovingDomainPhysics, PseudoElasticMeshMotionMatchesFiniteDifference)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::PseudoElasticMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    opts.lambda_mesh = FE::forms::ScalarCoefficient{
        [](FE::Real x, FE::Real, FE::Real) { return 1.5 + 0.25 * x; }};
    opts.mu_mesh = FE::forms::ScalarCoefficient{
        [](FE::Real, FE::Real y, FE::Real z) { return 0.75 + 0.125 * y + 0.0625 * z; }};

    FE::systems::FESystem system(mesh);
    mm::PseudoElasticMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::SymmetricPart));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Trace));
    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 12);

    std::vector<FE::Real> u(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = static_cast<FE::Real>(0.005 * (static_cast<int>(i) - 4));
    }

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    expectOperatorJacobianMatchesCentralFD(
        system, state, "mesh_motion", /*eps=*/1e-6, /*rtol=*/1e-6, /*atol=*/1e-10);
}

TEST(MovingDomainPhysics, PseudoElasticMeshMotionWeakBoundaryTermsOnSameMarkerAreAdditive)
{
    constexpr int marker = 15;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::PseudoElasticMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {0.0, 2.0, 0.0};

    mm::PseudoElasticMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 4.0;
    robin.target = {1.0, 0.0, 0.0};

    const auto assemble = [&](const mm::PseudoElasticMeshMotionOptions& opts) {
        FE::systems::FESystem system(mesh);
        mm::PseudoElasticMeshMotionModule module(d_space, opts);
        module.registerOn(system);
        system.setup({}, makeSingleTetraSetupInputs());

        std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        FE::systems::SystemStateView state;
        state.u = std::span<const FE::Real>(u);
        return residualVector(system, state, "mesh_motion");
    };

    mm::PseudoElasticMeshMotionOptions combined_opts;
    combined_opts.operator_tag = "mesh_motion";
    combined_opts.natural.push_back(natural);
    combined_opts.robin.push_back(robin);

    mm::PseudoElasticMeshMotionOptions equivalent_opts;
    equivalent_opts.operator_tag = "mesh_motion";
    auto equivalent_robin = robin;
    equivalent_robin.target = {1.0, 0.5, 0.0};
    equivalent_opts.robin.push_back(equivalent_robin);

    const auto combined_residual = assemble(combined_opts);
    const auto equivalent_residual = assemble(equivalent_opts);

    ASSERT_EQ(combined_residual.size(), equivalent_residual.size());
    for (std::size_t i = 0; i < combined_residual.size(); ++i) {
        EXPECT_NEAR(combined_residual[i],
                    equivalent_residual[i],
                    1.0e-12);
    }
}

TEST(MovingDomainPhysics, MeshMotionModulesInstallEquivalentBoundaryConditionDescriptors)
{
    auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions harmonic_opts;
    harmonic_opts.operator_tag = "mesh_motion";
    harmonic_opts.natural.push_back(mm::HarmonicMeshMotionOptions::NaturalBC{
        .boundary_marker = 21,
        .value = {1.0, -2.0, 0.5},
    });
    harmonic_opts.robin.push_back(mm::HarmonicMeshMotionOptions::RobinBC{
        .boundary_marker = 22,
        .alpha = 3.0,
        .target = {0.25, -0.5, 1.0},
    });
    harmonic_opts.dirichlet.push_back(mm::HarmonicMeshMotionOptions::DirichletBC{
        .boundary_marker = 23,
        .value = {0.0, 0.1, -0.2},
    });

    mm::PseudoElasticMeshMotionOptions pseudo_opts;
    pseudo_opts.operator_tag = "mesh_motion";
    pseudo_opts.natural.push_back(mm::PseudoElasticMeshMotionOptions::NaturalBC{
        .boundary_marker = 21,
        .value = {1.0, -2.0, 0.5},
    });
    pseudo_opts.robin.push_back(mm::PseudoElasticMeshMotionOptions::RobinBC{
        .boundary_marker = 22,
        .alpha = 3.0,
        .target = {0.25, -0.5, 1.0},
    });
    pseudo_opts.dirichlet.push_back(mm::PseudoElasticMeshMotionOptions::DirichletBC{
        .boundary_marker = 23,
        .value = {0.0, 0.1, -0.2},
    });

    FE::systems::FESystem harmonic_system(mesh);
    mm::HarmonicMeshMotionModule harmonic_module(d_space, harmonic_opts);
    harmonic_module.registerOn(harmonic_system);

    FE::systems::FESystem pseudo_system(mesh);
    mm::PseudoElasticMeshMotionModule pseudo_module(d_space, pseudo_opts);
    pseudo_module.registerOn(pseudo_system);

    const auto& harmonic_desc = harmonic_system.boundaryConditionDescriptors();
    const auto& pseudo_desc = pseudo_system.boundaryConditionDescriptors();
    ASSERT_EQ(harmonic_desc.size(), pseudo_desc.size());
    ASSERT_EQ(harmonic_desc.size(), 5u);

    for (std::size_t i = 0; i < harmonic_desc.size(); ++i) {
        EXPECT_EQ(harmonic_desc[i].boundary_marker, pseudo_desc[i].boundary_marker);
        EXPECT_EQ(harmonic_desc[i].component, pseudo_desc[i].component);
        EXPECT_EQ(harmonic_desc[i].trace_kind, pseudo_desc[i].trace_kind);
        EXPECT_EQ(harmonic_desc[i].enforcement_kind, pseudo_desc[i].enforcement_kind);
        EXPECT_EQ(harmonic_desc[i].source, pseudo_desc[i].source);
    }
}

TEST(MovingDomainPhysics, PseudoElasticMeshMotionRejectsInvalidLiteralParameters)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    const auto expect_invalid = [&](const mm::PseudoElasticMeshMotionOptions& opts,
                                    std::string_view expected_message) {
        FE::systems::FESystem system(mesh);
        mm::PseudoElasticMeshMotionModule module(d_space, opts);
        try {
            module.registerOn(system);
            FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument& ex) {
            EXPECT_NE(std::string(ex.what()).find(expected_message), std::string::npos)
                << ex.what();
        }
    };

    mm::PseudoElasticMeshMotionOptions zero_lambda;
    zero_lambda.lambda_mesh = 0.0;
    expect_invalid(zero_lambda, "lambda_mesh must be positive");

    mm::PseudoElasticMeshMotionOptions negative_lambda;
    negative_lambda.lambda_mesh = -1.0;
    expect_invalid(negative_lambda, "lambda_mesh must be positive");

    mm::PseudoElasticMeshMotionOptions zero_mu;
    zero_mu.mu_mesh = 0.0;
    expect_invalid(zero_mu, "mu_mesh must be positive");

    mm::PseudoElasticMeshMotionOptions negative_mu;
    negative_mu.mu_mesh = -1.0;
    expect_invalid(negative_mu, "mu_mesh must be positive");
}

TEST(MovingDomainPhysics, CoupledALEAndHarmonicMeshMotionShareDisplacementUnknown)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);

    mm::HarmonicMeshMotionOptions mesh_opts;
    mesh_opts.operator_tag = "mesh_motion";
    mm::HarmonicMeshMotionModule mesh_module(u_space, mesh_opts);
    mesh_module.registerOn(system);
    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);

    auto ns_opts = baseNavierStokesOptions();
    ns_opts.enable_ale = true;
    ns_opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    ns_opts.mesh_displacement_field_name = "mesh_displacement";
    ns_opts.mesh_velocity_field_name = "mesh_velocity";

    ns::IncompressibleNavierStokesVMSModule ns_module(u_space, p_space, ns_opts);
    ns_module.registerOn(system);

    const auto mesh_velocity = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).source_kind,
              FE::systems::FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).derived.source_field, displacement);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
    EXPECT_EQ(system.fieldMap().numFields(), 3u);

    bool has_mesh_rows = false;
    bool has_fluid_mesh_columns = false;
    const auto u = system.findFieldByName(ns_opts.velocity_field_name);
    const auto p = system.findFieldByName(ns_opts.pressure_field_name);
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (test_field == displacement && trial_field == displacement) {
                has_mesh_rows = true;
            }
            if (trial_field == displacement && (test_field == u || test_field == p)) {
                has_fluid_mesh_columns = true;
            }
        }
    }
    EXPECT_TRUE(has_mesh_rows);
    EXPECT_TRUE(has_fluid_mesh_columns);
}

TEST(MovingDomainPhysics, CoupledFittedFreeSurfaceALEAndHarmonicMeshMotionSetup)
{
    constexpr int marker = 39;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);

    mm::HarmonicMeshMotionOptions mesh_opts;
    mesh_opts.field_name = "mesh_displacement";
    mesh_opts.operator_tag = "mesh_motion";
    mesh_opts.kappa = 1.5;

    mm::NormalConstraintBC normal;
    normal.boundary_marker = marker;
    normal.quantity = mm::NormalConstraintQuantity::Velocity;
    normal.target = 0.15;
    normal.penalty = 6.0;
    normal.velocity_time_scale = 1.0;
    mesh_opts.normal_constraint.push_back(normal);

    mm::TangentialPolicyBC tangent;
    tangent.boundary_marker = marker;
    tangent.policy = mm::TangentialMeshPolicy::Prescribed;
    tangent.quantity = mm::TangentialConstraintQuantity::Velocity;
    tangent.target = {0.05, -0.02, 0.01};
    tangent.penalty = 5.0;
    tangent.velocity_time_scale = 1.0;
    mesh_opts.tangential_policy.push_back(tangent);

    mm::HarmonicMeshMotionModule mesh_module(u_space, mesh_opts);
    mesh_module.registerOn(system);
    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);

    auto ns_opts = baseNavierStokesOptions();
    ns_opts.enable_ale = true;
    ns_opts.enable_convection = false;
    ns_opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    ns_opts.mesh_displacement_field_name = "mesh_displacement";
    ns_opts.mesh_velocity_field_name = "mesh_velocity";

    ns_opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 1.25,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
        .kinematic_penalty = 9.0,
    });

    ns::IncompressibleNavierStokesVMSModule ns_module(u_space, p_space, ns_opts);
    ns_module.registerOn(system);

    const auto mesh_velocity = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity),
              mesh_velocity);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).source_kind,
              FE::systems::FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).derived.source_field, displacement);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
    EXPECT_EQ(system.fieldMap().numFields(), 3u);
    ASSERT_NE(system.blockMap(), nullptr);
    EXPECT_EQ(system.blockMap()->numBlocks(), 3u);

    const auto u = system.findFieldByName(ns_opts.velocity_field_name);
    const auto p = system.findFieldByName(ns_opts.pressure_field_name);
    bool has_mesh_rows = false;
    bool has_fluid_mesh_columns = false;
    bool has_mesh_fluid_columns = false;
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (test_field == displacement && trial_field == displacement) {
                has_mesh_rows = true;
            }
            if (trial_field == displacement && (test_field == u || test_field == p)) {
                has_fluid_mesh_columns = true;
            }
            if (test_field == displacement && trial_field == u) {
                has_mesh_fluid_columns = true;
            }
        }
    }
    EXPECT_TRUE(has_mesh_rows);
    EXPECT_TRUE(has_fluid_mesh_columns);
    EXPECT_TRUE(has_mesh_fluid_columns);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(solution, system, u, vertex, 0,
                               FE::Real(0.30) + FE::Real(0.01) * x);
        setFieldComponentValue(solution, system, u, vertex, 1,
                               FE::Real(-0.10) + FE::Real(0.015) * x);
        setFieldComponentValue(solution, system, u, vertex, 2,
                               FE::Real(0.20) - FE::Real(0.005) * x);
        setFieldComponentValue(solution, system, p, vertex, 0,
                               FE::Real(0.05) + FE::Real(0.02) * x);

        setFieldComponentValue(solution, system, displacement, vertex, 0,
                               FE::Real(0.02) + FE::Real(0.004) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 1,
                               FE::Real(-0.015) + FE::Real(0.003) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 2,
                               FE::Real(0.01) - FE::Real(0.002) * x);
    }
    updateBoundaryMeshCurrentCoordinates(*mesh, system, displacement, solution);

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
    EXPECT_GT(residualNorm(system, state, "equations"), 0.0);
}

TEST(MovingDomainPhysics, ALEAdvectionDiffusionManufacturedResidualUsesPhysicalMinusMeshVelocity)
{
    using namespace FE::forms;

    const auto phi = manufacturedScalarField();
    const auto psi = FormExpr::constant(1.0);
    const auto rho = FormExpr::constant(2.0);
    const auto physical_advection = constantVector3(1.0, -0.25, 0.5);
    const auto w_mesh = meshVelocity();
    const auto relative_advection = physical_advection - w_mesh;

    const auto residual =
        rho * div(w_mesh) * phi * psi +
        rho * dot(relative_advection, grad(phi)) * psi +
        FormExpr::constant(0.01) * dot(grad(phi), grad(psi));

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshVelocity));
}

TEST(MovingDomainPhysics, ExplicitPhysicalMinusMeshVelocityAssemblesCorrectly)
{
    using namespace FE::forms;

    SingleTetraMeshAccess mesh;
    FE::spaces::H1Space scalar_space(FE::ElementType::Tetra4, 1);
    auto scalar_dof_map = createSingleTetraDenseDofMap(4);

    auto base_space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
    FE::spaces::ProductSpace vector_space(base_space, 3);
    auto vector_dof_map = createSingleTetraDenseDofMap(12);

    const auto u = FormExpr::trialFunction(scalar_space, "temperature");
    const auto v = FormExpr::testFunction(scalar_space, "test");
    const auto rho = FormExpr::constant(2.0);

    const auto ale_relative = constantVector3(1.0, -0.25, 0.5) - meshVelocity();
    const auto static_equivalent = constantVector3(0.75, -0.125, 0.0);

    const auto ale_integrand = rho * dot(ale_relative, grad(u)) * v;
    const auto static_integrand = rho * dot(static_equivalent, grad(u)) * v;

    const std::vector<FE::Real> ale_solution = {0.0, 1.0, 1.0, 1.0};
    const auto mesh_velocity = constantVectorTetraCoefficients(0.25, -0.125, 0.5);

    const auto ale_residual = assembleMovingDomainScalarResidual(mesh,
                                                                 scalar_space,
                                                                 scalar_dof_map,
                                                                 &vector_space,
                                                                 &vector_dof_map,
                                                                 ale_integrand,
                                                                 ale_solution,
                                                                 mesh_velocity);

    const std::vector<FE::Real> static_solution = {0.0, 1.0, 1.0, 1.0};
    const auto static_residual = assembleMovingDomainScalarResidual(mesh,
                                                                    scalar_space,
                                                                    scalar_dof_map,
                                                                    nullptr,
                                                                    nullptr,
                                                                    static_integrand,
                                                                    static_solution);

    for (FE::GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(ale_residual.getVectorEntry(i),
                    static_residual.getVectorEntry(i),
                    1.0e-12);
    }
}

TEST(MovingDomainPhysics, MovingControlVolumeDivergenceTermAssemblesKnownValue)
{
    using namespace FE::forms;

    SingleTetraMeshAccess mesh;
    FE::spaces::H1Space scalar_space(FE::ElementType::Tetra4, 1);
    auto scalar_dof_map = createSingleTetraDenseDofMap(4);

    auto base_space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
    FE::spaces::ProductSpace vector_space(base_space, 3);
    auto vector_dof_map = createSingleTetraDenseDofMap(12);

    const auto u = FormExpr::trialFunction(scalar_space, "temperature");
    const auto v = FormExpr::testFunction(scalar_space, "test");
    const auto integrand = FormExpr::constant(2.0) * div(meshVelocity()) * u * v;

    const std::vector<FE::Real> solution = constantScalarTetraCoefficients(3.0);
    const auto mesh_velocity = affineXVectorTetraCoefficients();

    const auto residual = assembleMovingDomainScalarResidual(mesh,
                                                             scalar_space,
                                                             scalar_dof_map,
                                                             &vector_space,
                                                             &vector_dof_map,
                                                             integrand,
                                                             solution,
                                                             mesh_velocity);

    // div(w)=1 for w=(x,0,0), u=3, rho=2, and int_T phi_i dx = volume/4 = 1/24.
    const FE::Real expected = 2.0 * 1.0 * 3.0 * (1.0 / 24.0);
    for (FE::GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(residual.getVectorEntry(i), expected, 1.0e-12);
    }
}

TEST(MovingDomainPhysics, ALEIncompressibleNavierStokesManufacturedResidualUsesMovingDomainExpressions)
{
    using namespace FE::forms;

    const auto x0 = component(currentCoordinate(), 0);
    const auto x1 = component(currentCoordinate(), 1);
    const auto x2 = component(currentCoordinate(), 2);
    const auto u = FormExpr::asVector({
        x0 + t(),
        x1 * x1,
        x2 - FormExpr::constant(0.25) * t(),
    });
    const auto p = x0 - x1 + FormExpr::constant(0.5) * x2;
    const auto v = constantVector3(0.5, -1.0, 0.25);
    const auto q = FormExpr::constant(2.0);
    const auto rho = FormExpr::constant(1.25);
    const auto mu = FormExpr::constant(0.02);
    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto w_mesh = meshVelocity();
    const auto relative_advection = u - w_mesh;

    const auto momentum =
        rho * inner(dt(u) + grad(u) * relative_advection, v) +
        rho * div(w_mesh) * inner(u, v) +
        FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v))) -
        p * div(v);
    const auto continuity = q * div(u);
    const auto residual = momentum + continuity - inner(div(stress), v);

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshVelocity));
}

TEST(MovingDomainPhysics, MovingBoundaryFlowSmokeUsesGenericBoundaryTerminals)
{
    const auto test_scalar = FormExpr::constant(1.0);
    const auto boundary_velocity = constantVector3(0.0, 0.0, 1.0);

    const auto residual = movingBoundaryKinematicResidual(boundary_velocity, test_scalar);

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshVelocity));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentNormal));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentMeasure));
}

TEST(MovingDomainPhysics, FSIInterfaceKinematicsAndTractionsUseGenericGeometryTerminals)
{
    const auto test_scalar = FormExpr::constant(1.0);
    const auto structural_displacement = constantVector3(0.1, -0.2, 0.3);
    const auto traction = constantVector3(2.0, 3.0, 4.0);
    const auto velocity_test = constantVector3(0.25, 0.5, 0.75);

    const auto residual =
        fsiDisplacementCompatibilityResidual(structural_displacement, test_scalar) +
        fsiSurfaceTractionPowerResidual(traction, velocity_test) +
        referenceSurfaceMeasureMismatchProbe();

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshDisplacement));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentNormal));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentMeasure));
    EXPECT_TRUE(containsExprType(residual, FormExprType::ReferenceNormal));
    EXPECT_TRUE(containsExprType(residual, FormExprType::ReferenceMeasure));
}

} // namespace test
} // namespace Physics
} // namespace svmp
