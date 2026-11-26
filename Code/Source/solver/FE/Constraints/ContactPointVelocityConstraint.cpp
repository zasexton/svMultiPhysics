/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/ContactPointVelocityConstraint.h"

#include "Constraints/AffineConstraints.h"
#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Core/LogRecord.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Systems/FESystem.h"
#include "Core/EnvironmentNames.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#if FE_HAS_MPI
#  include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

struct WallCrossing {
    GlobalIndex cell;
    GlobalIndex wet;
    GlobalIndex dry;
    Real fraction;
    // A strict interior root needs no confirmation from another wall edge.
    GlobalIndex vertex_dof{INVALID_GLOBAL_INDEX};
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
Real cellCosTheta(const assembly::IMeshAccess& mesh, const std::vector<GlobalIndex>& cell_nodes,
                  const std::function<Real(GlobalIndex)>& phi_at, const std::array<Real, 3>& wall_normal)
{
    // P1 gradient in an affine triangle containing an actual contour segment.
    std::array<std::array<Real, 2>, 3> x{};
    std::array<Real, 3> f{};
    for (std::size_t k = 0; k < 3u; ++k) {
        const auto c = mesh.getNodeCoordinates(cell_nodes[k]);
        x[k] = {static_cast<Real>(c[0]), static_cast<Real>(c[1])};
        f[k] = phi_at(cell_nodes[k]);
    }
    const Real a11 = x[1][0] - x[0][0];
    const Real a12 = x[1][1] - x[0][1];
    const Real a21 = x[2][0] - x[0][0];
    const Real a22 = x[2][1] - x[0][1];
    const Real det = a11 * a22 - a12 * a21;
    if (!(std::abs(det) > Real{0.0})) {
        throw std::invalid_argument("degenerate wall cell");
    }
    const Real r1 = f[1] - f[0];
    const Real r2 = f[2] - f[0];
    const Real gx = (r1 * a22 - r2 * a12) / det;
    const Real gy = (a11 * r2 - a21 * r1) / det;
    const Real gnorm = std::hypot(gx, gy);
    if (!(gnorm > Real{0.0}) || !std::isfinite(gnorm)) {
        throw std::invalid_argument("invalid level-set gradient in a contact cell");
    }
    return -(gx * wall_normal[0] + gy * wall_normal[1]) / gnorm;
}

// A contour crossing the interior of a triangle has one trace. A contour
// along an interior edge has two traces and separates phases only when the
// third vertices have opposite signs. Fold those two traces into one ray.
std::map<GlobalIndex, Real> uniqueVertexAngles(const std::vector<GlobalIndex>& rays)
{
    struct EdgeSides {
        bool dry{};
        std::optional<Real> wet_angle;
    };
    std::map<std::pair<GlobalIndex, GlobalIndex>, EdgeSides> edges;
    std::map<GlobalIndex, Real> angles;
    const auto add = [&](GlobalIndex vertex, Real angle) {
        if (!angles.emplace(vertex, angle).second) {
            throw std::runtime_error("ContactPointVelocityConstraint: exact vertex has no unique interface segment");
        }
    };
    for (std::size_t i = 0; i < rays.size(); i += 4u) {
        const auto vertex = rays[i];
        const auto edge_endpoint = rays[i + 1u];
        const Real angle = std::bit_cast<double>(rays[i + 3u]);
        if (edge_endpoint == INVALID_GLOBAL_INDEX) {
            add(vertex, angle);
        } else {
            auto& sides = edges[{vertex, edge_endpoint}];
            if (rays[i + 2u] < 0) {
                sides.wet_angle = angle;
            } else {
                sides.dry = true;
            }
        }
    }
    for (const auto& [edge, sides] : edges) {
        if (sides.wet_angle && sides.dry) {
            add(edge.first, *sides.wet_angle);
        }
    }
    return angles;
}
#endif

struct LocalLine {
    GlobalIndex slave{INVALID_GLOBAL_INDEX};
    std::vector<std::pair<GlobalIndex, double>> entries{};
    double inhomogeneity{0.0};
    ContactPointVelocitySample sample{};
};

// Gather 64-bit words so both global indices and floating-point data round-trip exactly.
void appendLine(std::vector<GlobalIndex>& out, const LocalLine& line)
{
    const auto append_real = [&](double value) { out.push_back(std::bit_cast<GlobalIndex>(value)); };
    out.push_back(line.slave);
    out.push_back(static_cast<GlobalIndex>(line.entries.size()));
    for (const auto& [master, weight] : line.entries) {
        out.push_back(master);
        append_real(weight);
    }
    append_real(line.inhomogeneity);
    append_real(line.sample.point[0]);
    append_real(line.sample.point[1]);
    append_real(line.sample.fraction);
    append_real(line.sample.cos_theta);
    append_real(line.sample.target);
}

[[nodiscard]] std::vector<LocalLine> readLines(const std::vector<GlobalIndex>& in)
{
    std::vector<LocalLine> lines;
    std::size_t i = 0u;
    const auto take = [&]() {
        if (i >= in.size()) {
            throw std::logic_error("ContactPointVelocityConstraint: truncated line buffer");
        }
        return in[i++];
    };
    const auto take_real = [&]() { return std::bit_cast<double>(take()); };
    while (i < in.size()) {
        LocalLine line;
        line.slave = take();
        const auto n = static_cast<std::size_t>(take());
        for (std::size_t k = 0; k < n; ++k) {
            const auto master = take();
            const double weight = take_real();
            line.entries.emplace_back(master, weight);
        }
        line.inhomogeneity = take_real();
        line.sample.point[0] = take_real();
        line.sample.point[1] = take_real();
        line.sample.fraction = take_real();
        line.sample.cos_theta = take_real();
        line.sample.target = take_real();
        line.sample.slave_dof = line.slave;
        lines.push_back(std::move(line));
    }
    return lines;
}

[[nodiscard]] std::vector<GlobalIndex> gatherAll(const systems::FESystem& system,
                                                 const std::vector<GlobalIndex>& local)
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    const auto comm = system.dofHandler().mpiComm();
    if (initialized != 0 && comm != MPI_COMM_NULL) {
        int size = 1;
        MPI_Comm_size(comm, &size);
        if (size > 1) {
            const auto max_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
            const int count = local.size() <= max_count ? static_cast<int>(local.size()) : -1;
            std::vector<int> counts(static_cast<std::size_t>(size), 0);
            MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
            std::vector<int> displacements(static_cast<std::size_t>(size), 0);
            std::size_t total = 0u;
            for (int r = 0; r < size; ++r) {
                const auto rank_count = counts[static_cast<std::size_t>(r)];
                if (rank_count < 0 || static_cast<std::size_t>(rank_count) > max_count - total) {
                    throw std::overflow_error("ContactPointVelocityConstraint: buffer too large");
                }
                displacements[static_cast<std::size_t>(r)] = static_cast<int>(total);
                total += static_cast<std::size_t>(rank_count);
            }
            if (total == 0u) {
                return {};
            }
            std::vector<GlobalIndex> global(total);
            MPI_Allgatherv(local.data(), count, MPI_INT64_T, global.data(), counts.data(),
                           displacements.data(), MPI_INT64_T, comm);
            return global;
        }
    }
#else
    (void)system;
#endif
    return local;
}

[[nodiscard]] bool anyRank(const systems::FESystem& system, bool local)
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    const auto comm = system.dofHandler().mpiComm();
    if (initialized != 0 && comm != MPI_COMM_NULL) {
        int value = local ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MAX, comm);
        return value != 0;
    }
#else
    (void)system;
#endif
    return local;
}

} // namespace

ContactPointVelocityConstraint::ContactPointVelocityConstraint(FieldId velocity_field,
                                                               ContactPointVelocityLaw law)
    : velocity_field_(velocity_field), law_(std::move(law))
{
    if (law_.wall_boundary_markers.empty()) {
        throw std::invalid_argument(
            "ContactPointVelocityConstraint: needs at least one wall boundary marker");
    }
    if (law_.level_set_field_name.empty() || !law_.contact_speed) {
        throw std::invalid_argument(
            "ContactPointVelocityConstraint: needs a level-set field name and a contact-speed law");
    }
    const auto& n = law_.wall_normal;
    const int normal_axis = std::abs(n[0]) > std::abs(n[1]) ? 0 : 1;
    tangential_axis_ = 1 - normal_axis;
    constexpr Real tolerance = 1.0e-12;
    if (!(std::abs(std::abs(n[normal_axis]) - 1.0) <= tolerance &&
          std::abs(n[tangential_axis_]) <= tolerance && std::abs(n[2]) <= tolerance)) {
        throw std::invalid_argument(
            "ContactPointVelocityConstraint: the wall normal must be +-x or +-y (2D, axis-aligned walls)");
    }
}

void ContactPointVelocityConstraint::apply(const systems::FESystem& system,
                                           AffineConstraints& constraints)
{
    samples_.clear();
    // Mirror the aggregation lifecycle: the initial setup pass precedes the
    // generated geometry and installs nothing.
    if (system.cutIntegrationContext() == nullptr && !system.isSetup()) {
        return;
    }
    std::vector<GlobalIndex> local_buffer;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Coordinate local failures before either gather: the other half of a
    // zero vertex's wall neighborhood can belong to a different rank.
    const auto collectively = [&](const auto& operation) {
        std::string failure;
        try {
            operation();
        } catch (const std::exception& error) {
            failure = error.what();
        }
        if (anyRank(system, !failure.empty())) {
            throw std::runtime_error("ContactPointVelocityConstraint '" + law_.name + "': " +
                                     (failure.empty() ? "another rank failed" : failure));
        }
    };
    std::vector<WallCrossing> crossings;
    std::vector<GlobalIndex> local_neighbors;
    std::function<Real(GlobalIndex)> phi_at;
    const dofs::EntityDofMap* entity_map = nullptr;
    GlobalIndex offset = 0;
    const auto tangential_dof = [&](GlobalIndex vertex) {
        const auto vdofs = entity_map->getVertexDofs(vertex);
        if (vdofs.size() < 2u) {
            throw std::runtime_error("velocity needs two components per vertex");
        }
        return offset + vdofs[static_cast<std::size_t>(tangential_axis_)];
    };
    collectively([&] {
        const auto& mesh = system.meshAccess();
        if (mesh.dimension() != 2) {
            throw std::invalid_argument("supports only 2D meshes");
        }
        // The level set oriented by the liquid side: negative in the liquid.
        const Real orientation = law_.liquid_side == LevelSetConstraintSide::Positive ? Real{-1.0} : Real{1.0};
        if (law_.level_set_vertex_value) {
            phi_at = [&, orientation](GlobalIndex vertex) {
                return orientation * (law_.level_set_vertex_value(vertex) - law_.isovalue);
            };
        } else {
            const auto* native_mesh = system.mesh();
            if (native_mesh == nullptr) {
                throw std::runtime_error("requires native mesh support or a level-set value source");
            }
            const auto& local_mesh = native_mesh->local_mesh();
            if (!svmp::MeshFields::has_field(local_mesh, svmp::EntityKind::Vertex,
                                             law_.level_set_field_name)) {
                throw std::runtime_error("missing vertex level-set field '" +
                                         law_.level_set_field_name + "'");
            }
            const auto handle = svmp::MeshFields::get_field_handle(
                local_mesh, svmp::EntityKind::Vertex, law_.level_set_field_name);
            const auto phi_components = svmp::MeshFields::field_components(local_mesh, handle);
            const auto phi_count = svmp::MeshFields::field_entity_count(local_mesh, handle);
            const auto* phi = svmp::MeshFields::field_data_as<Real>(local_mesh, handle);
            if (phi == nullptr || phi_components < 1u) {
                throw std::runtime_error("level-set field has no scalar data");
            }
            phi_at = [this, phi, phi_components, phi_count, orientation](GlobalIndex vertex) {
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= phi_count) {
                    throw std::runtime_error("vertex outside the level-set field");
                }
                return orientation * (phi[static_cast<std::size_t>(vertex) * phi_components] - law_.isovalue);
            };
        }
        entity_map = system.fieldDofHandler(velocity_field_).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error("velocity field has no EntityDofMap");
        }
        offset = system.fieldDofOffset(velocity_field_);
        std::vector<GlobalIndex> cell_nodes;
        std::set<GlobalIndex> selected_faces;
        for (const auto marker : law_.wall_boundary_markers) {
            mesh.forEachBoundaryFace(marker, [&](GlobalIndex face, GlobalIndex cell) {
                if (!mesh.isOwnedCell(cell)) {
                    return;
                }
                const auto local_face = mesh.getLocalFaceIndex(face, cell);
                if (local_face < 0) {
                    return;
                }
                if (mesh.getCellType(cell) != ElementType::Triangle3) {
                    throw std::invalid_argument("supports only Triangle3 cells");
                }
                const auto& ref = elements::ReferenceElement::shared(ElementType::Triangle3);
                const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face));
                mesh.getCellNodes(cell, cell_nodes);
                if (face_nodes.size() != 2u || cell_nodes.size() < 3u) {
                    throw std::invalid_argument("found an incomplete wall face");
                }
                const auto va = cell_nodes[static_cast<std::size_t>(face_nodes[0])];
                const auto vb = cell_nodes[static_cast<std::size_t>(face_nodes[1])];
                const Real fa = phi_at(va);
                const Real fb = phi_at(vb);
                const bool a_wet = fa < fb;
                const Real fw = a_wet ? fa : fb;
                const Real fd = a_wet ? fb : fa;
                if (!(fw <= 0.0 && fd >= 0.0 && fw < fd)) {
                    return;  // Same sign or an entire edge at the isovalue.
                }
                if (law_.wall_boundary_markers.size() > 1u && !selected_faces.insert(face).second) {
                    return;  // Overlapping marker selections describe the same edge.
                }
                const auto wet = a_wet ? va : vb;
                const auto dry = a_wet ? vb : va;
                GlobalIndex vertex_dof = INVALID_GLOBAL_INDEX;
                if (fw == 0.0 || fd == 0.0) {
                    const auto vertex = fw == 0.0 ? wet : dry;
                    const auto neighbor = fw == 0.0 ? dry : wet;
                    vertex_dof = tangential_dof(vertex);
                    const auto x = mesh.getNodeCoordinates(vertex)[tangential_axis_];
                    const auto xn = mesh.getNodeCoordinates(neighbor)[tangential_axis_];
                    if (xn == x) {
                        throw std::invalid_argument("wall edge has no extent along the tangential axis");
                    }
                    local_neighbors.insert(local_neighbors.end(), {
                        vertex_dof, xn > x ? 1 : 0, fw == 0.0 ? 1 : -1});
                    if (xn > x) {
                        return;  // The lower-coordinate edge owns the velocity row.
                    }
                }
                const Real s = fw == 0.0 ? 0.0 : fw / (fw - fd);
                crossings.push_back({cell, wet, dry, s, vertex_dof});
            });
        }
    });

    const auto neighbors = gatherAll(system, local_neighbors);
    std::map<GlobalIndex, std::array<int, 2>> wall_sides;
    for (std::size_t i = 0; i < neighbors.size(); i += 3u) {
        auto& side = wall_sides[neighbors[i]][static_cast<std::size_t>(neighbors[i + 1u])];
        side = side == 0 ? static_cast<int>(neighbors[i + 2u]) : 2;
    }
    std::erase_if(wall_sides, [](const auto& entry) {
        const auto& sides = entry.second;
        return sides[0] * sides[1] != -1;  // Need one wet and one dry neighbor.
    });
    std::map<GlobalIndex, Real> vertex_angles;
    if (!wall_sides.empty()) {
        std::vector<GlobalIndex> local_rays;
        collectively([&] {
            const auto& mesh = system.meshAccess();
            std::map<GlobalIndex, GlobalIndex> local_roots;
            for (const auto& [global_dof, sides] : wall_sides) {
                const auto vertex = entity_map->getDofEntity(global_dof - offset);
                if (vertex && vertex->kind == dofs::EntityKind::Vertex) {
                    local_roots.emplace(vertex->id, global_dof);
                }
            }
            if (local_roots.empty()) {
                return;
            }
            std::vector<GlobalIndex> nodes;
            mesh.forEachOwnedCell([&](GlobalIndex cell) {
                mesh.getCellNodes(cell, nodes);
                for (std::size_t j = 0; j < nodes.size(); ++j) {
                    const auto root = local_roots.find(nodes[j]);
                    if (root == local_roots.end()) {
                        continue;
                    }
                    if (mesh.getCellType(cell) != ElementType::Triangle3 || nodes.size() != 3u) {
                        throw std::invalid_argument("requires Triangle3 cells around a contact vertex");
                    }
                    const auto a = nodes[(j + 1u) % 3u];
                    const auto b = nodes[(j + 2u) % 3u];
                    const Real fa = phi_at(a);
                    const Real fb = phi_at(b);
                    if ((fa > 0 && fb > 0) || (fa < 0 && fb < 0)) {
                        continue;  // This triangle only touches the contour at the vertex.
                    }
                    if ((fa == 0 && fb == 0) || phi_at(nodes[j]) != 0 ||
                        !std::isfinite(fa) || !std::isfinite(fb)) {
                        throw std::runtime_error("contact vertex has no unique interface segment");
                    }
                    GlobalIndex edge_endpoint = INVALID_GLOBAL_INDEX;
                    int side = 0;
                    if (fa == 0 || fb == 0) {
                        edge_endpoint = tangential_dof(fa == 0 ? a : b);
                        side = (fa < 0 || fb < 0) ? -1 : 1;
                    }
                    const double angle = cellCosTheta(mesh, nodes, phi_at, law_.wall_normal);
                    local_rays.insert(local_rays.end(), {root->second, edge_endpoint, side,
                                                         std::bit_cast<GlobalIndex>(angle)});
                }
            });
        });
        vertex_angles = uniqueVertexAngles(gatherAll(system, local_rays));
        if (vertex_angles.size() != wall_sides.size()) {
            throw std::runtime_error("ContactPointVelocityConstraint: exact vertex has no unique interface segment");
        }
    }
    collectively([&] {
        const auto& mesh = system.meshAccess();
        std::optional<AffineConstraints> closed;
        const auto expand = [&](GlobalIndex vertex, double factor,
                                std::map<GlobalIndex, double>& coefficients,
                                double& constant) {
            if (factor == 0.0) {
                return;
            }
            const auto dof = tangential_dof(vertex);
            if (!constraints.isConstrained(dof)) {
                coefficients[dof] += factor;
                return;
            }
            if (!closed.has_value()) {
                closed.emplace(constraints);
                if (!closed->isClosed()) {
                    closed->close();
                }
            }
            const auto view = closed->getConstraint(dof);
            if (!view.has_value()) {
                throw std::logic_error("constrained DOF without a closed line");
            }
            for (const auto& entry : view->entries) {
                coefficients[entry.master_dof] += factor * entry.weight;
            }
            constant += factor * view->inhomogeneity;
        };

        std::vector<GlobalIndex> cell_nodes;
        for (const auto& [cell, wet, dry, s, vertex_dof] : crossings) {
            if (vertex_dof != INVALID_GLOBAL_INDEX && !vertex_angles.contains(vertex_dof)) {
                continue;
            }
            const auto xw = mesh.getNodeCoordinates(wet);
            const auto xd = mesh.getNodeCoordinates(dry);
            mesh.getCellNodes(cell, cell_nodes);
            const Real cos_theta = vertex_dof == INVALID_GLOBAL_INDEX
                ? cellCosTheta(mesh, cell_nodes, phi_at, law_.wall_normal)
                : vertex_angles.at(vertex_dof);
            const Real target = law_.contact_speed(cos_theta);
            if (!std::isfinite(target)) {
                throw std::runtime_error("non-finite contact speed");
            }
            // Outward footprint direction t_f = sign(x_d - x_w) along the
            // tangential axis.
            const Real dx = static_cast<Real>(xd[static_cast<std::size_t>(tangential_axis_)] -
                                              xw[static_cast<std::size_t>(tangential_axis_)]);
            const double sign = dx > 0.0 ? 1.0 : -1.0;
            // sign * ((1 - s) u_w + s u_d) = target, expanded.
            std::map<GlobalIndex, double> coefficients;
            double constant = 0.0;
            expand(wet, sign * (1.0 - s), coefficients, constant);
            expand(dry, sign * s, coefficients, constant);
            GlobalIndex slave = INVALID_GLOBAL_INDEX;
            double pivot = 0.0;
            for (const auto& [dof, coefficient] : coefficients) {
                if (std::abs(coefficient) > std::abs(pivot)) {
                    slave = dof;
                    pivot = coefficient;
                }
            }
            if (slave == INVALID_GLOBAL_INDEX || !(std::abs(pivot) > 0.0)) {
                throw std::runtime_error("contact-point functional has no free DOF");
            }
            LocalLine line;
            line.slave = slave;
            for (const auto& [dof, coefficient] : coefficients) {
                if (dof != slave && coefficient != 0.0) {
                    line.entries.emplace_back(dof, -coefficient / pivot);
                }
            }
            line.inhomogeneity = (static_cast<double>(target) - constant) / pivot;
            line.sample.point = {std::lerp(xw[0], xd[0], s),
                                 std::lerp(xw[1], xd[1], s), Real{0.0}};
            line.sample.fraction = s;
            line.sample.cos_theta = cos_theta;
            line.sample.target = target;
            appendLine(local_buffer, line);
        }
    });
#else
    throw std::runtime_error("ContactPointVelocityConstraint '" + law_.name +
                             "': requires native mesh support");
#endif

    auto lines = readLines(gatherAll(system, local_buffer));
    std::sort(lines.begin(), lines.end(),
              [](const LocalLine& a, const LocalLine& b) { return a.slave < b.slave; });
    for (std::size_t k = 1; k < lines.size(); ++k) {
        if (lines[k].slave == lines[k - 1u].slave) {
            throw std::runtime_error(
                "ContactPointVelocityConstraint '" + law_.name +
                "': two contact points select the same DOF " +
                std::to_string(lines[k].slave) + "; the contact points are closer than one cell");
        }
    }
    const auto& partition = system.dofHandler().getPartition();
    std::string install_failure;
    for (const auto& line : lines) {
        samples_.push_back(line.sample);
        if (!partition.isRelevant(line.slave)) {
            continue;
        }
        for (const auto& [master, weight] : line.entries) {
            (void)weight;
            if (!partition.isRelevant(master)) {
                install_failure = "master DOF " + std::to_string(master) +
                                  " of contact-point slave " + std::to_string(line.slave) +
                                  " is not locally relevant; increase <Ghost_layers>";
            }
        }
        if (constraints.isConstrained(line.slave)) {
            install_failure = "contact-point slave DOF " + std::to_string(line.slave) +
                              " is already constrained";
        }
        if (!install_failure.empty()) {
            break;
        }
        constraints.addLine(line.slave);
        for (const auto& [master, weight] : line.entries) {
            constraints.addEntry(line.slave, master, weight);
        }
        constraints.setInhomogeneity(line.slave, line.inhomogeneity);
        // The fraction and the contact speed move with the level set, so
        // stored history keeps the relation of its own time.
        constraints.setInstantaneous(line.slave);
    }
    if (anyRank(system, !install_failure.empty())) {
        throw InsufficientGhostHaloError(
            "ContactPointVelocityConstraint '" + law_.name +
            "': diagnostic=contact_point_line_not_installable " +
            (install_failure.empty() ? std::string("on another rank") : install_failure));
    }
    const char* trace = environment::value("SVMP_CONTACT_POINT_VELOCITY_TRACE");
    if (trace != nullptr && trace[0] != '\0' && trace[0] != '0') {
        // One child line per sample point.
        LogRecord line("ContactPointVelocityConstraint", "contact-point velocity");
        line.diagnostic("contact_point_velocity")
            .quoted("constraint", law_.name)
            .value("points", samples_.size());
        std::size_t point = 0u;
        for (const auto& sample : samples_) {
            line.child("contact_point_velocity_point", {})
                .value("point", point++)
                .value("x", sample.point[0])
                .value("y", sample.point[1])
                .value("s", sample.fraction)
                .value("cos_theta", sample.cos_theta)
                .value("target", sample.target)
                .value("slave", sample.slave_dof);
        }
        logRecord(line);
    }
}

bool ContactPointVelocityConstraint::updateValues(const systems::FESystem& /*system*/,
                                                  AffineConstraints& /*constraints*/,
                                                  double /*time*/,
                                                  double /*dt*/)
{
    return false;
}

ConstraintDependencyDeclaration ContactPointVelocityConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = ISystemConstraint::dependencyDeclaration();
    out.structural.fe_constraint_layout = true;
    out.structural.mesh_field_layout = true;
    out.structural.mesh_field_values = true;
    out.structural.active_configuration = true;
    out.mesh_field_values_scoped = true;
    out.mesh_field_value_names = {law_.level_set_field_name};
    return out;
}

systems::SetupStorageRequirements
ContactPointVelocityConstraint::storageRequirements() const noexcept
{
    systems::SetupStorageRequirements req;
    req.entity_dof_map = true;
    req.vertex_topology = true;
    req.cell_topology = true;
    req.boundary_face_topology = true;
    return req;
}

std::string ContactPointVelocityConstraint::manifestDescription() const
{
    std::ostringstream oss;
    oss << std::hexfloat << "field=" << velocity_field_ << " name='" << law_.name << "' walls=";
    for (std::size_t i = 0; i < law_.wall_boundary_markers.size(); ++i) {
        oss << (i == 0u ? "" : ",") << law_.wall_boundary_markers[i];
    }
    oss << " wall_normal=" << law_.wall_normal[0] << "," << law_.wall_normal[1] << ","
        << law_.wall_normal[2]
        << " level_set='" << law_.level_set_field_name << "'"
        << " isovalue=" << law_.isovalue
        << " liquid_side=" << (law_.liquid_side == LevelSetConstraintSide::Positive ? "positive" : "negative")
        << " vertex_value_source=" << (law_.level_set_vertex_value ? "callback" : "mesh_field");
    // The contact-speed law is a closure; sample it so that the manifest
    // compares its values exactly.
    oss << " speed(cos)=";
    if (law_.contact_speed) {
        constexpr std::array<Real, 5> samples{{-1.0, -0.5, 0.0, 0.5, 1.0}};
        for (std::size_t i = 0; i < samples.size(); ++i) {
            oss << (i == 0u ? "" : ",") << law_.contact_speed(samples[i]);
        }
    } else {
        oss << "none";
    }
    return oss.str();
}

} // namespace constraints
} // namespace FE
} // namespace svmp
