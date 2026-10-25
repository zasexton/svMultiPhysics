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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#if FE_HAS_MPI
#  include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

struct LocalLine {
    GlobalIndex slave{INVALID_GLOBAL_INDEX};
    std::vector<std::pair<GlobalIndex, double>> entries{};
    double inhomogeneity{0.0};
    ContactPointVelocitySample sample{};
};

// Flatten / restore lines for the all-gather (DOF numbers are exact in a
// double up to 2^53).
void appendLine(std::vector<double>& out, const LocalLine& line)
{
    out.push_back(static_cast<double>(line.slave));
    out.push_back(static_cast<double>(line.entries.size()));
    for (const auto& [master, weight] : line.entries) {
        out.push_back(static_cast<double>(master));
        out.push_back(weight);
    }
    out.push_back(line.inhomogeneity);
    out.push_back(line.sample.point[0]);
    out.push_back(line.sample.point[1]);
    out.push_back(line.sample.fraction);
    out.push_back(line.sample.cos_theta);
    out.push_back(line.sample.target);
}

[[nodiscard]] std::vector<LocalLine> readLines(const std::vector<double>& in)
{
    std::vector<LocalLine> lines;
    std::size_t i = 0u;
    const auto take = [&]() {
        if (i >= in.size()) {
            throw std::logic_error("ContactPointVelocityConstraint: truncated line buffer");
        }
        return in[i++];
    };
    while (i < in.size()) {
        LocalLine line;
        line.slave = static_cast<GlobalIndex>(take());
        const auto n = static_cast<std::size_t>(take());
        for (std::size_t k = 0; k < n; ++k) {
            const auto master = static_cast<GlobalIndex>(take());
            const double weight = take();
            line.entries.emplace_back(master, weight);
        }
        line.inhomogeneity = take();
        line.sample.point[0] = take();
        line.sample.point[1] = take();
        line.sample.fraction = take();
        line.sample.cos_theta = take();
        line.sample.target = take();
        line.sample.slave_dof = line.slave;
        lines.push_back(std::move(line));
    }
    return lines;
}

[[nodiscard]] std::vector<double> gatherAll(const systems::FESystem& system,
                                            const std::vector<double>& local)
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    const auto comm = system.dofHandler().mpiComm();
    if (initialized != 0 && comm != MPI_COMM_NULL) {
        int size = 1;
        MPI_Comm_size(comm, &size);
        if (size > 1) {
            if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error("ContactPointVelocityConstraint: buffer too large");
            }
            int count = static_cast<int>(local.size());
            std::vector<int> counts(static_cast<std::size_t>(size), 0);
            MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
            std::vector<int> displacements(static_cast<std::size_t>(size), 0);
            std::size_t total = 0u;
            for (int r = 0; r < size; ++r) {
                displacements[static_cast<std::size_t>(r)] = static_cast<int>(total);
                total += static_cast<std::size_t>(counts[static_cast<std::size_t>(r)]);
            }
            std::vector<double> global(total, 0.0);
            MPI_Allgatherv(local.data(), count, MPI_DOUBLE, global.data(), counts.data(),
                           displacements.data(), MPI_DOUBLE, comm);
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
    int normal_axis = -1;
    for (int k = 0; k < 3; ++k) {
        const Real c = law_.wall_normal[static_cast<std::size_t>(k)];
        if (std::abs(std::abs(c) - Real{1.0}) <= Real{1.0e-12}) {
            normal_axis = k;
        } else if (std::abs(c) > Real{1.0e-12}) {
            normal_axis = -2;
            break;
        }
    }
    if (normal_axis != 0 && normal_axis != 1) {
        throw std::invalid_argument(
            "ContactPointVelocityConstraint: the wall normal must be +-x or +-y (2D, axis-aligned walls)");
    }
    tangential_axis_ = 1 - normal_axis;
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
    std::vector<double> local_buffer;
    std::string local_failure;
    try {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        const auto& mesh = system.meshAccess();
        if (mesh.dimension() != 2) {
            throw std::invalid_argument("supports only 2D meshes");
        }
        std::function<Real(GlobalIndex)> phi_at;
        if (law_.level_set_vertex_value) {
            phi_at = [&](GlobalIndex vertex) {
                return law_.level_set_vertex_value(vertex) - law_.isovalue;
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
            phi_at = [this, phi, phi_components, phi_count](GlobalIndex vertex) {
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= phi_count) {
                    throw std::runtime_error("vertex outside the level-set field");
                }
                return phi[static_cast<std::size_t>(vertex) * phi_components] - law_.isovalue;
            };
        }
        const auto& velocity_dofs = system.fieldDofHandler(velocity_field_);
        const auto* entity_map = velocity_dofs.getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error("velocity field has no EntityDofMap");
        }
        const auto offset = system.fieldDofOffset(velocity_field_);
        const auto tangential_dof = [&](GlobalIndex vertex) {
            const auto vdofs = entity_map->getVertexDofs(vertex);
            if (vdofs.size() < 2u) {
                throw std::runtime_error("velocity needs two components per vertex");
            }
            return offset + vdofs[static_cast<std::size_t>(tangential_axis_)];
        };

        std::optional<AffineConstraints> closed;
        const auto expand = [&](GlobalIndex dof, double factor,
                                std::map<GlobalIndex, double>& coefficients,
                                double& constant) {
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
                if (!((fa < 0.0 && fb > 0.0) || (fa > 0.0 && fb < 0.0))) {
                    return;
                }
                const auto wet = fa < 0.0 ? va : vb;
                const auto dry = fa < 0.0 ? vb : va;
                const Real fw = fa < 0.0 ? fa : fb;
                const Real fd = fa < 0.0 ? fb : fa;
                const Real s = fw / (fw - fd);
                const auto xw = mesh.getNodeCoordinates(wet);
                const auto xd = mesh.getNodeCoordinates(dry);
                // P1 gradient of the level set in the wall cell.
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
                if (!(gnorm > Real{0.0})) {
                    throw std::invalid_argument("zero level-set gradient in a contact cell");
                }
                const Real cos_theta =
                    -(gx * law_.wall_normal[0] + gy * law_.wall_normal[1]) / gnorm;
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
                expand(tangential_dof(wet), sign * (1.0 - s), coefficients, constant);
                expand(tangential_dof(dry), sign * s, coefficients, constant);
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
                line.sample.point = {static_cast<Real>(xw[0] + s * (xd[0] - xw[0])),
                                     static_cast<Real>(xw[1] + s * (xd[1] - xw[1])), Real{0.0}};
                line.sample.fraction = s;
                line.sample.cos_theta = cos_theta;
                line.sample.target = target;
                appendLine(local_buffer, line);
            });
        }
#else
        throw std::runtime_error("requires native mesh support");
#endif
    } catch (const std::exception& error) {
        local_failure = error.what();
    }
    if (anyRank(system, !local_failure.empty())) {
        throw std::runtime_error(
            "ContactPointVelocityConstraint '" + law_.name + "': " +
            (local_failure.empty() ? std::string("another rank failed") : local_failure));
    }

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
    }
    if (anyRank(system, !install_failure.empty())) {
        throw InsufficientGhostHaloError(
            "ContactPointVelocityConstraint '" + law_.name +
            "': diagnostic=contact_point_line_not_installable " +
            (install_failure.empty() ? std::string("on another rank") : install_failure));
    }
    const char* trace = std::getenv("SVMP_CONTACT_POINT_VELOCITY_TRACE");
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
