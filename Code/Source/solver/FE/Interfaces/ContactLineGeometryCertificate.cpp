/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/ContactLineGeometryCertificate.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Core/Logger.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Forms/CutCellForms.h"
#include "Geometry/CutQuadrature.h"
#include "Geometry/FrameGeometry.h"
#include "Geometry/MappingFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"

#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace interfaces {

namespace {

[[nodiscard]] Real dot3(const std::array<Real, 3>& a,
                            const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm3(const std::array<Real, 3>& a) noexcept
{
    return std::sqrt(dot3(a, a));
}

[[nodiscard]] std::array<Real, 3> normalizedOrZero(
    std::array<Real, 3> v) noexcept
{
    const auto n = norm3(v);
    if (!(n > Real{0.0}) || !std::isfinite(n)) {
        return {Real{0.0}, Real{0.0}, Real{0.0}};
    }
    for (auto& component : v) {
        component /= n;
    }
    return v;
}

[[nodiscard]] std::array<Real, 3> subtract3(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b) noexcept
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

[[nodiscard]] std::array<Real, 3> cross3(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b) noexcept
{
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

[[nodiscard]] PlanarWallEvidence localPlanarWallEvidence(
    const assembly::IMeshAccess& mesh,
    int wall_marker,
    const std::array<Real, 3>& configured)
{
    constexpr Real alignment_tolerance = Real{1.0e-8};
    PlanarWallEvidence evidence;
    const int dim = mesh.dimension();

    const auto record_failure = [&](GlobalIndex face,
                                    GlobalIndex cell,
                                    std::string_view reason,
                                    Real alignment =
                                        -std::numeric_limits<Real>::infinity()) {
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
        [&](GlobalIndex face, GlobalIndex cell) {
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
                elements::ReferenceElement::create(cell_type);
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

            std::vector<std::array<Real, 3>> coordinates;
            if (mesh.supportsCoordinateFrame(
                    assembly::CoordinateFrame::Current)) {
                mesh.getCellCoordinates(
                    cell,
                    assembly::CoordinateFrame::Current,
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

            std::array<Real, 3> cell_centroid{0.0, 0.0, 0.0};
            for (const auto& coordinate : coordinates) {
                for (int component = 0; component < dim; ++component) {
                    cell_centroid[static_cast<std::size_t>(component)] +=
                        coordinate[static_cast<std::size_t>(component)];
                }
            }
            for (int component = 0; component < dim; ++component) {
                cell_centroid[static_cast<std::size_t>(component)] /=
                    static_cast<Real>(coordinates.size());
            }

            std::array<Real, 3> face_centroid{0.0, 0.0, 0.0};
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
                    static_cast<Real>(face_nodes.size());
            }

            const auto& origin = coordinates[static_cast<std::size_t>(
                face_nodes.front())];
            const auto first_edge = subtract3(
                coordinates[static_cast<std::size_t>(face_nodes[1])],
                origin);
            std::array<Real, 3> physical_normal{};
            if (dim == 2) {
                physical_normal =
                    {first_edge[1], -first_edge[0], Real{0.0}};
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
            if (!(norm3(physical_normal) > Real{0.0})) {
                record_failure(face, cell, "degenerate_face_normal");
                return;
            }

            const auto outward_hint = subtract3(face_centroid, cell_centroid);
            const auto orientation = dot3(physical_normal, outward_hint);
            if (!std::isfinite(orientation) ||
                orientation == Real{0.0}) {
                record_failure(face, cell, "indeterminate_outward_orientation");
                return;
            }
            if (orientation < Real{0.0}) {
                for (auto& component : physical_normal) {
                    component = -component;
                }
            }

            Real face_scale = Real{0.0};
            Real maximum_plane_offset = Real{0.0};
            for (const auto node : face_nodes) {
                const auto offset = subtract3(
                    coordinates[static_cast<std::size_t>(node)], origin);
                face_scale = std::max(face_scale, norm3(offset));
                maximum_plane_offset = std::max(
                    maximum_plane_offset,
                    std::abs(dot3(offset, physical_normal)));
            }
            if (!(face_scale > Real{0.0}) ||
                !std::isfinite(maximum_plane_offset) ||
                maximum_plane_offset > alignment_tolerance * face_scale) {
                record_failure(face, cell, "nonplanar_wall_face");
                return;
            }

            const auto alignment = dot3(configured, physical_normal);
            evidence.minimum_alignment =
                std::min(evidence.minimum_alignment, alignment);
            if (!std::isfinite(alignment) ||
                Real{1.0} - alignment > alignment_tolerance) {
                record_failure(
                    face, cell, "configured_normal_mismatch", alignment);
            }
        });
    return evidence;
}

[[nodiscard]] std::array<Real, 3> physicalContactCovector(
    const assembly::IMeshAccess& mesh,
    const geometry::CutQuadratureRule& rule,
    const geometry::CutQuadraturePoint& point,
    std::array<Real, 3> normal,
    std::string_view role,
    const std::string& owner)
{
    if (rule.frame == geometry::CutGeometryFrame::Current) {
        if (!(norm3(normal) > Real{0.0})) {
            throw std::invalid_argument(
                owner + ": generated current-frame contact rule has an invalid physical " +
                std::string(role) + " covector");
        }
        return normal;
    }

    const auto cell = static_cast<GlobalIndex>(
        rule.provenance.parent_entity);
    if (cell < 0 || cell >= mesh.numCells()) {
        throw std::invalid_argument(
            owner + ": generated reference-frame contact rule has an invalid parent cell");
    }

    std::vector<std::array<Real, 3>> coordinates;
    mesh.getCellCoordinates(cell, coordinates);
    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        nodes.push_back(math::Vector<Real, 3>{
            coordinate[0], coordinate[1], coordinate[2]});
    }

    geometry::MappingRequest mapping_request;
    mapping_request.element_type = mesh.getCellType(cell);
    mapping_request.geometry_order = mesh.getCellGeometryOrder(cell);
    mapping_request.use_affine = mapping_request.geometry_order <= 1;
    const auto mapping =
        geometry::MappingFactory::create(mapping_request, nodes);
    const math::Vector<Real, 3> xi{
        point.point[0], point.point[1], point.point[2]};
    const auto inverse = geometry::scaleConditionedJacobianInverse(
        mapping->jacobian(xi));
    const std::array<Real, 3> mapped{{
        inverse(0, 0) * normal[0] + inverse(1, 0) * normal[1] +
            inverse(2, 0) * normal[2],
        inverse(0, 1) * normal[0] + inverse(1, 1) * normal[1] +
            inverse(2, 1) * normal[2],
        inverse(0, 2) * normal[0] + inverse(1, 2) * normal[1] +
            inverse(2, 2) * normal[2]}};
    if (!(norm3(mapped) > Real{0.0})) {
        throw std::invalid_argument(
            owner + ": generated reference-frame contact rule maps to an invalid physical " +
            std::string(role) + " covector");
    }
    return mapped;
}

[[nodiscard]] std::array<Real, 3> physicalContactNormal(
    const assembly::IMeshAccess& mesh,
    const geometry::CutQuadratureRule& rule,
    const geometry::CutQuadraturePoint& point,
    std::array<Real, 3> normal,
    std::string_view role,
    const std::string& owner)
{
    normal = normalizedOrZero(physicalContactCovector(
        mesh, rule, point, normal, role, owner));
    if (!(norm3(normal) > Real{0.0})) {
        throw std::invalid_argument(
            owner + ": generated contact rule has an invalid physical " +
            std::string(role) + " normal");
    }
    return normal;
}

[[nodiscard]] std::array<Real, 3> physicalContactBoundaryNormal(
    const assembly::IMeshAccess& mesh,
    const geometry::CutQuadratureRule& rule,
    const geometry::CutQuadraturePoint& point,
    const std::string& owner)
{
    return physicalContactNormal(
        mesh, rule, point, point.boundary_normal, "boundary", owner);
}

[[nodiscard]] std::array<Real, 3> physicalContactInterfaceNormal(
    const assembly::IMeshAccess& mesh,
    const geometry::CutQuadratureRule& rule,
    const geometry::CutQuadraturePoint& point,
    const std::string& owner)
{
    return physicalContactNormal(
        mesh, rule, point, point.normal, "interface", owner);
}

} // namespace

PlanarWallEvidence planarWallEvidence(
    const systems::FESystem& system,
    int wall_marker,
    const std::array<Real, 3>& configured_normal)
{
    auto evidence = localPlanarWallEvidence(
        system.meshAccess(), wall_marker, configured_normal);

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
            Real global_minimum_alignment = Real{1.0};
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
    return evidence;
}

void certifyContactLineRules(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext* context,
    const ContactLineGeometryCheck& check)
{
    const int contact_marker = check.contact_marker;
    const auto& owner = check.owner;
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

    constexpr Real alignment_tolerance = kContactLineAlignmentTolerance;
    // Normalizing the wall-tangential projection defines the contact-line
    // footprint direction.  Reject a nearly parallel interface/wall normal
    // before the regularized symbolic normalization can hide its singularity.
    const auto& configured = check.wall_normal;
    const auto& mesh = system.meshAccess();
    std::size_t samples = 0u;
    Real minimum_alignment = Real{1.0};
    Real minimum_transverse_projection = Real{1.0};
    for (const auto* rule : rules) {
        if (rule == nullptr || rule->points.empty()) {
            throw std::invalid_argument(
                owner + ": generated contact marker contains a null or empty quadrature rule while validating wall_normal");
        }
        if (rule->geometric_dimension != mesh.dimension() - 2) {
            throw std::invalid_argument(
                owner + ": generated contact marker is not a codimension-two rule while validating wall_normal");
        }
        for (const auto& point : rule->points) {
            const auto physical =
                physicalContactBoundaryNormal(mesh, *rule, point, owner);
            const auto interface_normal =
                physicalContactInterfaceNormal(mesh, *rule, point, owner);
            const auto alignment = dot3(configured, physical);
            const auto normal_dot = dot3(interface_normal, physical);
            const std::array<Real, 3> transverse{{
                interface_normal[0] - normal_dot * physical[0],
                interface_normal[1] - normal_dot * physical[1],
                interface_normal[2] - normal_dot * physical[2]}};
            const auto transverse_projection = norm3(transverse);
            minimum_alignment = std::min(minimum_alignment, alignment);
            minimum_transverse_projection = std::min(
                minimum_transverse_projection, transverse_projection);
            ++samples;
            if (!std::isfinite(alignment) ||
                Real{1.0} - alignment > alignment_tolerance) {
                std::ostringstream message;
                message
                    << owner << ": configured "
                       "contact-line wall_normal does not match the physical "
                       "outward normal carried by its generated boundary rule"
                    << " wall_boundary_marker="
                    << check.wall_marker
                    << " contact_line_marker=" << contact_marker
                    << " parent_cell="
                    << rule->provenance.parent_entity
                    << " frame="
                    << (rule->frame ==
                                geometry::CutGeometryFrame::Reference
                            ? "reference_mapped_to_physical"
                            : "current_physical")
                    << " alignment=" << alignment
                    << " required_alignment>="
                    << Real{1.0} - alignment_tolerance
                    << " configured_normal=(" << configured[0] << ','
                    << configured[1] << ',' << configured[2] << ')'
                    << " physical_boundary_normal=(" << physical[0] << ','
                    << physical[1] << ',' << physical[2] << ')';
                throw std::invalid_argument(message.str());
            }
            if (!std::isfinite(transverse_projection) ||
                transverse_projection < kContactLineMinimumTransverseSine) {
                std::ostringstream message;
                message
                    << owner << ": generated "
                       "contact rule is not transverse to its wall; the "
                       "contact-line footprint direction is singular"
                    << " wall_boundary_marker="
                    << check.wall_marker
                    << " contact_line_marker=" << contact_marker
                    << " parent_cell="
                    << rule->provenance.parent_entity
                    << " transverse_projection=" << transverse_projection
                    << " required_transverse_projection>="
                    << kContactLineMinimumTransverseSine
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
    if (Logger::instance().get_level() > LogLevel::DEBUG) {
        return;
    }
    FE_LOG_INFO(
        owner + ": validated generated contact-line wall normal" +
        " wall_boundary_marker=" +
        std::to_string(check.wall_marker) +
        " contact_line_marker=" + std::to_string(contact_marker) +
        " samples=" + std::to_string(samples) +
        " minimum_alignment=" + std::to_string(minimum_alignment) +
        " minimum_transverse_projection=" +
        std::to_string(minimum_transverse_projection) +
        " alignment_tolerance=1e-8" +
        " minimum_transverse_sine=1e-6" +
        " diagnostic=unfitted_contact_wall_normal_validation");
}

bool certifyContactLineRulesCollectively(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext* context,
    const ContactLineGeometryCheck& check,
    bool local_followup_available,
    std::string_view phase)
{
    std::exception_ptr local_exception;
    try {
        certifyContactLineRules(system, context, check);
    } catch (...) {
        local_exception = std::current_exception();
    }
    return systems::agreeOnLocalPhase(
        system,
        local_exception,
        local_followup_available,
        phase,
        check.owner);
}

void registerContactLineCertificate(
    systems::FESystem& system,
    const ContactLineGeometryCheck& check,
    std::string callback_name,
    std::string phase,
    std::string_view initial_phase,
    ContactLineCertificateRefresh refresh)
{
    const bool requires_setup =
        refresh == ContactLineCertificateRefresh::SetUpSystemContext;
    system.addCutIntegrationContextUpdateCallback(
        systems::CutIntegrationContextUpdateCallback{
            .name = std::move(callback_name),
            .callback =
                [&system, check, phase = std::move(phase), requires_setup](
                    const assembly::CutIntegrationContext* cut_context) {
                    static_cast<void>(certifyContactLineRulesCollectively(
                        system,
                        cut_context,
                        check,
                        cut_context != nullptr &&
                            (!requires_setup || system.isSetup()),
                        phase));
                }});
    const auto* current_cut_context = system.cutIntegrationContext();
    static_cast<void>(certifyContactLineRulesCollectively(
        system,
        current_cut_context,
        check,
        requires_setup
            ? current_cut_context != nullptr && system.isSetup()
            : true,
        initial_phase));
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
