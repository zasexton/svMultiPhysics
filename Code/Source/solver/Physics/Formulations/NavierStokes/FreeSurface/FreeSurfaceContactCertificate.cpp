/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Constraints/ContactPointVelocityConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/SymbolicDifferentiation.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/SystemCollectives.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

[[nodiscard]] FE::Real dot3(const std::array<FE::Real, 3>& a,
                            const std::array<FE::Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] FE::Real norm3(const std::array<FE::Real, 3>& a) noexcept
{
    return std::sqrt(dot3(a, a));
}

[[nodiscard]] std::array<FE::Real, 3> normalizedOrZero(
    std::array<FE::Real, 3> v) noexcept
{
    const auto n = norm3(v);
    if (!(n > FE::Real{0.0}) || !std::isfinite(n)) {
        return {FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}};
    }
    for (auto& component : v) {
        component /= n;
    }
    return v;
}

[[nodiscard]] std::array<FE::Real, 3> subtract3(
    const std::array<FE::Real, 3>& a,
    const std::array<FE::Real, 3>& b) noexcept
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

[[nodiscard]] std::array<FE::Real, 3> cross3(
    const std::array<FE::Real, 3>& a,
    const std::array<FE::Real, 3>& b) noexcept
{
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

struct PlanarWallMarkerNormalEvidence {
    std::uint64_t face_count{0u};
    std::uint64_t invalid_face_count{0u};
    FE::Real minimum_alignment{FE::Real{1.0}};
    std::string first_failure{};
};

[[nodiscard]] PlanarWallMarkerNormalEvidence planarWallMarkerNormalEvidence(
    const FE::assembly::IMeshAccess& mesh,
    int wall_marker,
    const std::array<FE::Real, 3>& configured)
{
    constexpr FE::Real alignment_tolerance = FE::Real{1.0e-8};
    PlanarWallMarkerNormalEvidence evidence;
    const int dim = mesh.dimension();

    const auto record_failure = [&](FE::GlobalIndex face,
                                    FE::GlobalIndex cell,
                                    std::string_view reason,
                                    FE::Real alignment =
                                        -std::numeric_limits<FE::Real>::infinity()) {
        ++evidence.invalid_face_count;
        evidence.minimum_alignment =
            std::min(evidence.minimum_alignment, alignment);
        if (evidence.first_failure.empty()) {
            std::ostringstream message;
            message << "face=" << face << " parent_cell=" << cell
                    << " reason=" << reason;
            if (std::isfinite(alignment)) {
                message << " alignment=" << alignment;
            }
            evidence.first_failure = message.str();
        }
    };

    mesh.forEachBoundaryFace(
        wall_marker,
        [&](FE::GlobalIndex face, FE::GlobalIndex cell) {
            if (!mesh.isOwnedCell(cell)) {
                return;
            }
            ++evidence.face_count;

            if (mesh.getCellGeometryOrder(cell) != 1) {
                record_failure(
                    face,
                    cell,
                    "higher_order_wall_geometry_is_not_certified_planar");
                return;
            }

            const auto cell_type = mesh.getCellType(cell);
            const auto reference =
                FE::elements::ReferenceElement::create(cell_type);
            const auto local_face = mesh.getLocalFaceIndex(face, cell);
            if (local_face < 0 ||
                static_cast<std::size_t>(local_face) >=
                    reference.num_faces()) {
                record_failure(face, cell, "invalid_local_face_index");
                return;
            }
            const auto& face_nodes =
                reference.face_nodes(static_cast<std::size_t>(local_face));
            if ((dim == 2 && face_nodes.size() < 2u) ||
                (dim == 3 && face_nodes.size() < 3u)) {
                record_failure(face, cell, "invalid_face_topology");
                return;
            }

            std::vector<std::array<FE::Real, 3>> coordinates;
            if (mesh.supportsCoordinateFrame(
                    FE::assembly::CoordinateFrame::Current)) {
                mesh.getCellCoordinates(
                    cell,
                    FE::assembly::CoordinateFrame::Current,
                    coordinates);
            } else {
                // DynamicContactAngle rejects ALE, so the adapter's active
                // coordinates are the current physical coordinates here.
                mesh.getCellCoordinates(cell, coordinates);
            }
            if (coordinates.empty()) {
                record_failure(face, cell, "missing_current_coordinates");
                return;
            }
            for (const auto node : face_nodes) {
                if (node < 0 || static_cast<std::size_t>(node) >=
                                    coordinates.size()) {
                    record_failure(face, cell, "face_node_out_of_range");
                    return;
                }
            }

            std::array<FE::Real, 3> cell_centroid{0.0, 0.0, 0.0};
            for (const auto& coordinate : coordinates) {
                for (int component = 0; component < dim; ++component) {
                    cell_centroid[static_cast<std::size_t>(component)] +=
                        coordinate[static_cast<std::size_t>(component)];
                }
            }
            for (int component = 0; component < dim; ++component) {
                cell_centroid[static_cast<std::size_t>(component)] /=
                    static_cast<FE::Real>(coordinates.size());
            }

            std::array<FE::Real, 3> face_centroid{0.0, 0.0, 0.0};
            for (const auto node : face_nodes) {
                const auto& coordinate =
                    coordinates[static_cast<std::size_t>(node)];
                for (int component = 0; component < dim; ++component) {
                    face_centroid[static_cast<std::size_t>(component)] +=
                        coordinate[static_cast<std::size_t>(component)];
                }
            }
            for (int component = 0; component < dim; ++component) {
                face_centroid[static_cast<std::size_t>(component)] /=
                    static_cast<FE::Real>(face_nodes.size());
            }

            const auto& origin = coordinates[static_cast<std::size_t>(
                face_nodes.front())];
            const auto first_edge = subtract3(
                coordinates[static_cast<std::size_t>(face_nodes[1])],
                origin);
            std::array<FE::Real, 3> physical_normal{};
            if (dim == 2) {
                physical_normal =
                    {first_edge[1], -first_edge[0], FE::Real{0.0}};
            } else if (dim == 3) {
                const auto second_edge = subtract3(
                    coordinates[static_cast<std::size_t>(face_nodes[2])],
                    origin);
                physical_normal = cross3(first_edge, second_edge);
            } else {
                record_failure(face, cell, "unsupported_spatial_dimension");
                return;
            }
            physical_normal = normalizedOrZero(physical_normal);
            if (!(norm3(physical_normal) > FE::Real{0.0})) {
                record_failure(face, cell, "degenerate_face_normal");
                return;
            }

            const auto outward_hint = subtract3(face_centroid, cell_centroid);
            const auto orientation = dot3(physical_normal, outward_hint);
            if (!std::isfinite(orientation) ||
                orientation == FE::Real{0.0}) {
                record_failure(face, cell, "indeterminate_outward_orientation");
                return;
            }
            if (orientation < FE::Real{0.0}) {
                for (auto& component : physical_normal) {
                    component = -component;
                }
            }

            FE::Real face_scale = FE::Real{0.0};
            FE::Real maximum_plane_offset = FE::Real{0.0};
            for (const auto node : face_nodes) {
                const auto offset = subtract3(
                    coordinates[static_cast<std::size_t>(node)], origin);
                face_scale = std::max(face_scale, norm3(offset));
                maximum_plane_offset = std::max(
                    maximum_plane_offset,
                    std::abs(dot3(offset, physical_normal)));
            }
            if (!(face_scale > FE::Real{0.0}) ||
                !std::isfinite(maximum_plane_offset) ||
                maximum_plane_offset > alignment_tolerance * face_scale) {
                record_failure(face, cell, "nonplanar_wall_face");
                return;
            }

            const auto alignment = dot3(configured, physical_normal);
            evidence.minimum_alignment =
                std::min(evidence.minimum_alignment, alignment);
            if (!std::isfinite(alignment) ||
                FE::Real{1.0} - alignment > alignment_tolerance) {
                record_failure(
                    face, cell, "configured_normal_mismatch", alignment);
            }
        });
    return evidence;
}

[[nodiscard]] std::array<FE::Real, 3> physicalContactCovector(
    const FE::assembly::IMeshAccess& mesh,
    const FE::geometry::CutQuadratureRule& rule,
    const FE::geometry::CutQuadraturePoint& point,
    std::array<FE::Real, 3> normal,
    std::string_view role)
{
    if (rule.frame == FE::geometry::CutGeometryFrame::Current) {
        if (!(norm3(normal) > FE::Real{0.0})) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: generated current-frame contact rule has an invalid physical " +
                std::string(role) + " covector");
        }
        return normal;
    }

    const auto cell = static_cast<FE::GlobalIndex>(
        rule.provenance.parent_entity);
    if (cell < 0 || cell >= mesh.numCells()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated reference-frame contact rule has an invalid parent cell");
    }

    std::vector<std::array<FE::Real, 3>> coordinates;
    mesh.getCellCoordinates(cell, coordinates);
    std::vector<FE::math::Vector<FE::Real, 3>> nodes;
    nodes.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        nodes.push_back(FE::math::Vector<FE::Real, 3>{
            coordinate[0], coordinate[1], coordinate[2]});
    }

    FE::geometry::MappingRequest mapping_request;
    mapping_request.element_type = mesh.getCellType(cell);
    mapping_request.geometry_order = mesh.getCellGeometryOrder(cell);
    mapping_request.use_affine = mapping_request.geometry_order <= 1;
    const auto mapping =
        FE::geometry::MappingFactory::create(mapping_request, nodes);
    const FE::math::Vector<FE::Real, 3> xi{
        point.point[0], point.point[1], point.point[2]};
    const auto inverse = FE::geometry::scaleConditionedJacobianInverse(
        mapping->jacobian(xi));
    const std::array<FE::Real, 3> mapped{{
        inverse(0, 0) * normal[0] + inverse(1, 0) * normal[1] +
            inverse(2, 0) * normal[2],
        inverse(0, 1) * normal[0] + inverse(1, 1) * normal[1] +
            inverse(2, 1) * normal[2],
        inverse(0, 2) * normal[0] + inverse(1, 2) * normal[1] +
            inverse(2, 2) * normal[2]}};
    if (!(norm3(mapped) > FE::Real{0.0})) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated reference-frame contact rule maps to an invalid physical " +
            std::string(role) + " covector");
    }
    return mapped;
}

[[nodiscard]] std::array<FE::Real, 3> physicalContactNormal(
    const FE::assembly::IMeshAccess& mesh,
    const FE::geometry::CutQuadratureRule& rule,
    const FE::geometry::CutQuadraturePoint& point,
    std::array<FE::Real, 3> normal,
    std::string_view role)
{
    normal = normalizedOrZero(physicalContactCovector(
        mesh, rule, point, normal, role));
    if (!(norm3(normal) > FE::Real{0.0})) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated contact rule has an invalid physical " +
            std::string(role) + " normal");
    }
    return normal;
}

[[nodiscard]] std::array<FE::Real, 3> physicalContactBoundaryNormal(
    const FE::assembly::IMeshAccess& mesh,
    const FE::geometry::CutQuadratureRule& rule,
    const FE::geometry::CutQuadraturePoint& point)
{
    return physicalContactNormal(
        mesh, rule, point, point.boundary_normal, "boundary");
}

[[nodiscard]] std::array<FE::Real, 3> physicalContactInterfaceNormal(
    const FE::assembly::IMeshAccess& mesh,
    const FE::geometry::CutQuadratureRule& rule,
    const FE::geometry::CutQuadraturePoint& point)
{
    return physicalContactNormal(
        mesh, rule, point, point.normal, "interface");
}

void validateUnfittedContactWallNormal(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext* context,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker)
{
    if (context == nullptr) {
        return;
    }
    context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        system.meshAccess());
    const auto rules = context->interfaceRulesForMarker(contact_marker);
    if (rules.empty()) {
        // An interface need not currently intersect every configured wall.
        // Validation becomes actionable as soon as that contact rule exists.
        return;
    }

    constexpr FE::Real alignment_tolerance = FE::Real{1.0e-8};
    // Normalizing the wall-tangential projection defines the contact-line
    // footprint direction.  Reject a nearly parallel interface/wall normal
    // before the regularized symbolic normalization can hide its singularity.
    const auto configured = normalizedWallNormal(contact_line);
    const auto& mesh = system.meshAccess();
    std::size_t samples = 0u;
    FE::Real minimum_alignment = FE::Real{1.0};
    FE::Real minimum_transverse_projection = FE::Real{1.0};
    for (const auto* rule : rules) {
        if (rule == nullptr || rule->points.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: generated contact marker contains a null or empty quadrature rule while validating wall_normal");
        }
        if (rule->geometric_dimension != mesh.dimension() - 2) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: generated contact marker is not a codimension-two rule while validating wall_normal");
        }
        for (const auto& point : rule->points) {
            const auto physical =
                physicalContactBoundaryNormal(mesh, *rule, point);
            const auto interface_normal =
                physicalContactInterfaceNormal(mesh, *rule, point);
            const auto alignment = dot3(configured, physical);
            const auto normal_dot = dot3(interface_normal, physical);
            const std::array<FE::Real, 3> transverse{{
                interface_normal[0] - normal_dot * physical[0],
                interface_normal[1] - normal_dot * physical[1],
                interface_normal[2] - normal_dot * physical[2]}};
            const auto transverse_projection = norm3(transverse);
            minimum_alignment = std::min(minimum_alignment, alignment);
            minimum_transverse_projection = std::min(
                minimum_transverse_projection, transverse_projection);
            ++samples;
            if (!std::isfinite(alignment) ||
                FE::Real{1.0} - alignment > alignment_tolerance) {
                std::ostringstream message;
                message
                    << "IncompressibleNavierStokesVMSModule: configured "
                       "contact-line wall_normal does not match the physical "
                       "outward normal carried by its generated boundary rule"
                    << " wall_boundary_marker="
                    << contactLineWallBoundaryMarker(contact_line)
                    << " contact_line_marker=" << contact_marker
                    << " parent_cell="
                    << rule->provenance.parent_entity
                    << " frame="
                    << (rule->frame ==
                                FE::geometry::CutGeometryFrame::Reference
                            ? "reference_mapped_to_physical"
                            : "current_physical")
                    << " alignment=" << alignment
                    << " required_alignment>="
                    << FE::Real{1.0} - alignment_tolerance
                    << " configured_normal=(" << configured[0] << ','
                    << configured[1] << ',' << configured[2] << ')'
                    << " physical_boundary_normal=(" << physical[0] << ','
                    << physical[1] << ',' << physical[2] << ')';
                throw std::invalid_argument(message.str());
            }
            if (!std::isfinite(transverse_projection) ||
                transverse_projection < kMinimumContactTransverseSine) {
                std::ostringstream message;
                message
                    << "IncompressibleNavierStokesVMSModule: generated "
                       "contact rule is not transverse to its wall; the "
                       "contact-line footprint direction is singular"
                    << " wall_boundary_marker="
                    << contactLineWallBoundaryMarker(contact_line)
                    << " contact_line_marker=" << contact_marker
                    << " parent_cell="
                    << rule->provenance.parent_entity
                    << " transverse_projection=" << transverse_projection
                    << " required_transverse_projection>="
                    << kMinimumContactTransverseSine
                    << " interface_normal=(" << interface_normal[0] << ','
                    << interface_normal[1] << ',' << interface_normal[2]
                    << ") physical_boundary_normal=(" << physical[0] << ','
                    << physical[1] << ',' << physical[2] << ')';
                throw std::invalid_argument(message.str());
            }
        }
    }

    // Detailed per-refresh line, printed (at INFO) only when
    // FE_LOG_LEVEL=DEBUG.
    if (FE::Logger::instance().get_level() > FE::LogLevel::DEBUG) {
        return;
    }
    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: validated generated contact-line wall normal") +
        " wall_boundary_marker=" +
        std::to_string(contactLineWallBoundaryMarker(contact_line)) +
        " contact_line_marker=" + std::to_string(contact_marker) +
        " samples=" + std::to_string(samples) +
        " minimum_alignment=" + std::to_string(minimum_alignment) +
        " minimum_transverse_projection=" +
        std::to_string(minimum_transverse_projection) +
        " alignment_tolerance=1e-8" +
        " minimum_transverse_sine=1e-6" +
        " diagnostic=unfitted_contact_wall_normal_validation");
}

[[nodiscard]] bool validateContactWallNormalCallbackPreflight(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext* context,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker,
    bool local_followup_available,
    std::string_view phase)
{
    std::exception_ptr local_exception;
    try {
        validateUnfittedContactWallNormal(
            system, context, contact_line, contact_marker);
    } catch (...) {
        local_exception = std::current_exception();
    }
    return FE::systems::agreeOnLocalPhase(
        system,
        local_exception,
        local_followup_available,
        phase,
        "IncompressibleNavierStokesVMSModule");
}

[[nodiscard]] int axisAlignedWallNormalAxis(
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const auto normal = normalizedWallNormal(contact_line);
    const auto law = contactWallLawName(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    int axis = -1;
    for (int component = 0; component < 3; ++component) {
        const auto magnitude =
            std::abs(normal[static_cast<std::size_t>(component)]);
        if (component >= dim) {
            if (magnitude > tolerance) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " wall_normal must lie in the velocity space dimension");
            }
            continue;
        }
        if (std::abs(magnitude - FE::Real{1.0}) <= tolerance) {
            if (axis >= 0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " wall_normal must have exactly one axis-aligned component");
            }
            axis = component;
        } else if (magnitude > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " + law +
                " currently requires an axis-aligned wall_normal because general linear-combination normal constraints are unavailable");
        }
    }
    if (axis < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires an axis-aligned wall_normal");
    }
    return axis;
}

} // namespace

void validateDynamicContactPlanarWallMarker(
    const FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line)
{
    constexpr FE::Real alignment_tolerance = FE::Real{1.0e-8};
    const auto configured = normalizedWallNormal(contact_line);
    auto evidence = planarWallMarkerNormalEvidence(
        system.meshAccess(),
        contactLineWallBoundaryMarker(contact_line),
        configured);

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        MPI_Finalized(&mpi_finalized);
    }
    const auto communicator = system.activeMpiCommunicator();
    if (mpi_initialized != 0 && mpi_finalized == 0 &&
        communicator != MPI_COMM_NULL) {
        int communicator_size = 1;
        MPI_Comm_size(communicator, &communicator_size);
        if (communicator_size > 1) {
            const std::array<unsigned long long, 2> local_counts{
                static_cast<unsigned long long>(evidence.face_count),
                static_cast<unsigned long long>(evidence.invalid_face_count)};
            std::array<unsigned long long, 2> global_counts{};
            MPI_Allreduce(local_counts.data(),
                          global_counts.data(),
                          static_cast<int>(local_counts.size()),
                          MPI_UNSIGNED_LONG_LONG,
                          MPI_SUM,
                          communicator);
            FE::Real global_minimum_alignment = FE::Real{1.0};
            MPI_Allreduce(&evidence.minimum_alignment,
                          &global_minimum_alignment,
                          1,
                          MPI_DOUBLE,
                          MPI_MIN,
                          communicator);
            evidence.face_count =
                static_cast<std::uint64_t>(global_counts[0]);
            evidence.invalid_face_count =
                static_cast<std::uint64_t>(global_counts[1]);
            evidence.minimum_alignment = global_minimum_alignment;
        }
    }
#endif

    if (evidence.face_count == 0u || evidence.invalid_face_count != 0u) {
        std::ostringstream message;
        message
            << "IncompressibleNavierStokesVMSModule: "
            << contactWallLawName(contact_line)
            << " requires every face on Contact_line_wall_marker to be a "
               "first-order planar face with the configured physical "
               "outward wall_normal"
            << " wall_boundary_marker="
            << contactLineWallBoundaryMarker(contact_line)
            << " global_owned_faces=" << evidence.face_count
            << " global_invalid_faces=" << evidence.invalid_face_count
            << " minimum_alignment=" << evidence.minimum_alignment
            << " required_alignment>="
            << FE::Real{1.0} - alignment_tolerance
            << " configured_normal=(" << configured[0] << ','
            << configured[1] << ',' << configured[2] << ')';
        if (!evidence.first_failure.empty()) {
            message << " local_first_failure='" << evidence.first_failure
                    << '\'';
        }
        throw std::invalid_argument(message.str());
    }

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: validated complete DynamicContactAngle planar wall marker") +
        " wall_boundary_marker=" +
        std::to_string(contactLineWallBoundaryMarker(contact_line)) +
        " global_owned_faces=" + std::to_string(evidence.face_count) +
        " minimum_alignment=" +
        std::to_string(evidence.minimum_alignment) +
        " alignment_tolerance=1e-8" +
        " geometry_contract=first_order_planar_current_physical" +
        " diagnostic=dynamic_contact_complete_wall_marker_validation");
}

void validateFreeSurfaceContactGeometryPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system)
{
    const auto* context = system.cutIntegrationContext();
    if (context == nullptr) {
        return;
    }
    for (const auto& boundary : free_surfaces) {
        if (!isUnfittedLevelSet(boundary)) {
            continue;
        }
        for (const auto& contact_line : boundary.contact_lines) {
            const auto kind = contactLineKind(contact_line);
            if (kind != ContactLineKind::PrescribedAngle &&
                kind != ContactLineKind::DynamicRenE) {
                continue;
            }
            static_cast<void>(
                validateContactWallNormalCallbackPreflight(
                    system,
                    context,
                    contact_line,
                    generatedUnfittedContactLineMarker(
                        contact_line, boundary, system),
                    true,
                    "free_surface_contact_geometry_preflight"));
        }
    }
}

void validateDynamicContactWallEssentialBC(
    const IncompressibleNavierStokesVMSOptions& options,
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const int normal_axis = axisAlignedWallNormalAxis(contact_line, dim);
    const auto law = contactWallLawName(contact_line);
    bool normal_is_constrained = false;
    bool saw_wall_bc = false;
    for (const auto& dirichlet : options.velocity_dirichlet) {
        if (dirichlet.boundary_marker != contactLineWallBoundaryMarker(contact_line)) {
            continue;
        }
        saw_wall_bc = true;
        for (int component = 0; component < dim; ++component) {
            if (!dirichlet.active_components[
                    static_cast<std::size_t>(component)]) {
                continue;
            }
            if (!FE::forms::bc::isZeroConstantScalarValue(
                    dirichlet.value[static_cast<std::size_t>(component)])) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " requires a stationary zero-valued normal wall essential condition");
            }
            if (component == normal_axis) {
                normal_is_constrained = true;
            } else {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " rejects tangential/full no-slip Dirichlet constraints on its Navier-slip wall");
            }
        }
    }
    for (const auto& dirichlet : options.velocity_dirichlet_weak) {
        if (dirichlet.boundary_marker == contactLineWallBoundaryMarker(contact_line)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " + law +
                " requires a strong zero normal-only wall essential condition; weak velocity Dirichlet data on the wall are unsupported");
        }
    }
    if (!saw_wall_bc || !normal_is_constrained) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + law +
            " requires an axis-aligned zero normal-only velocity Dirichlet condition on Contact_line_wall_marker");
    }
}

void registerFreeSurfacePrescribedAngleGeometry(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc)
{
    for (const auto& contact_line : bc.contact_lines) {
        if (contactLineKind(contact_line) != ContactLineKind::PrescribedAngle) {
            continue;
        }
        if (isUnfittedLevelSet(bc)) {
            const auto phi_id = system.findFieldByName(bc.level_set_field_name);
            if (phi_id == FE::INVALID_FIELD_ID) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle references unknown level-set field '" +
                    bc.level_set_field_name + "'");
            }
            const auto& rec = system.fieldRecord(phi_id);
            if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: level-set field '" +
                    bc.level_set_field_name + "' must be scalar");
            }
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require the level-set field to be an unknown");
            }

            const int contact_marker =
                generatedUnfittedContactLineMarker(
                    contact_line,
                    bc,
                    system);
            system.registerGeneratedEmbeddedInterfaceMarker(contact_marker);
            system.addCutIntegrationContextUpdateCallback(
                FE::systems::CutIntegrationContextUpdateCallback{
                    .name = "navier_stokes_prescribed_contact_geometry:" +
                            std::to_string(contact_marker),
                    .callback =
                        [&system, contact_line, contact_marker](
                            const FE::assembly::CutIntegrationContext*
                                cut_context) {
                            static_cast<void>(
                                validateContactWallNormalCallbackPreflight(
                                    system,
                                    cut_context,
                                    contact_line,
                                    contact_marker,
                                    cut_context != nullptr,
                                    "prescribed_contact_geometry"));
                        }});
            const auto* current_cut_context =
                system.cutIntegrationContext();
            static_cast<void>(validateContactWallNormalCallbackPreflight(
                system,
                current_cut_context,
                contact_line,
                contact_marker,
                true,
                "prescribed_contact_geometry_initial"));
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: registered prescribed-angle contact geometry without a level-set residual") +
                " interface_marker=" +
                std::to_string(bc.interface_marker) +
                " contact_marker=" + std::to_string(contact_marker) +
                " level_set_geometry_owner=none_angle_preserving_wall_maintenance"
                " momentum_owner=" +
                std::string(usesKinematicAreaGradientTraction(bc)
                                ? "total_energy_gradient_traction"
                                : "young_wall_energy") +
                " literal_codimension_two_level_set_residual=retired"
                " diagnostic=navier_stokes_prescribed_contact_geometry");
            continue;
        }

        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: fitted prescribed contact-angle validation did not fail closed");
    }
}

void registerDynamicContactWallNormalCheck(
    FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker)
{
    system.addCutIntegrationContextUpdateCallback(
        FE::systems::CutIntegrationContextUpdateCallback{
            .name = "navier_stokes_dynamic_contact_wall_normal:" +
                    std::to_string(contact_marker),
            .callback =
                [&system, contact_line, contact_marker](
                    const FE::assembly::CutIntegrationContext*
                        cut_context) {
                    static_cast<void>(
                        validateContactWallNormalCallbackPreflight(
                            system,
                            cut_context,
                            contact_line,
                            contact_marker,
                            cut_context != nullptr &&
                                system.isSetup(),
                            "dynamic_contact_operator_angle"));
                }});
    const auto* current_cut_context =
        system.cutIntegrationContext();
    static_cast<void>(validateContactWallNormalCallbackPreflight(
        system,
        current_cut_context,
        contact_line,
        contact_marker,
        current_cut_context != nullptr && system.isSetup(),
        "dynamic_contact_operator_angle_initial"));
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
