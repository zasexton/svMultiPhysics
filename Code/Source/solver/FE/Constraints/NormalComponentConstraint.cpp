/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/NormalComponentConstraint.h"

#include "Assembly/CutIntegrationContext.h"
#include "Elements/ElementTransform.h"
#include "Elements/LagrangeElement.h"
#include "Elements/ReferenceElement.h"
#include "Forms/PointEvaluator.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/LinearMapping.h"
#include "Geometry/FrameGeometry.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/ProductSpace.h"
#include "Spaces/FaceRestriction.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <utility>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace constraints {

using systems::FESystem;
using systems::InvalidStateException;

namespace {

constexpr Real kPi = 3.14159265358979323846;

/// What one boundary facet says about one support point: its unit outward
/// normal there, its measure, and the point as the facet's cell maps it.
struct FacetContribution {
    std::int64_t facet{-1};                    ///< global facet id
    GlobalIndex local_cell{-1};                ///< used before exchanging facet data
    std::array<GlobalIndex, 3> dofs{-1, -1, -1};  ///< the point's DOF of each component
    std::array<Real, 3> normal{0.0, 0.0, 0.0};
    Real moment_scale{};                       ///< integral of the absolute trace basis
    std::array<Real, 3> flux_moment{};       ///< integral of the scalar trace basis times n dS
    Real measure{0.0};
    std::array<Real, 3> x{0.0, 0.0, 0.0};
};

[[nodiscard]] Real dot3(const std::array<Real, 3>& a, const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// The number of corners of face @p face of a cell of @p type (its nodes
/// list the corners first).
[[nodiscard]] std::size_t faceCornerCount(ElementType type, std::size_t face)
{
    ElementType linear = ElementType::Unknown;
    switch (to_mesh_family(type)) {
    case svmp::CellFamily::Triangle: linear = ElementType::Triangle3; break;
    case svmp::CellFamily::Quad: linear = ElementType::Quad4; break;
    case svmp::CellFamily::Tetra: linear = ElementType::Tetra4; break;
    case svmp::CellFamily::Hex: linear = ElementType::Hex8; break;
    case svmp::CellFamily::Wedge: linear = ElementType::Wedge6; break;
    case svmp::CellFamily::Pyramid: linear = ElementType::Pyramid5; break;
    default:
        throw InvalidStateException("NormalComponentConstraint: unsupported cell type of a boundary facet");
    }
    return elements::ReferenceElement::shared(linear).face_nodes(face).size();
}

/// The measure of the linear facet through @p corners (a segment, a
/// triangle, or a quadrilateral by its diagonals).
[[nodiscard]] Real facetMeasure(const std::vector<std::array<Real, 3>>& corners)
{
    const auto sub = [](const std::array<Real, 3>& a, const std::array<Real, 3>& b) {
        return std::array<Real, 3>{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    const auto cross = [](const std::array<Real, 3>& a, const std::array<Real, 3>& b) {
        return std::array<Real, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    };
    switch (corners.size()) {
    case 2: {
        const auto d = sub(corners[1], corners[0]);
        return std::hypot(d[0], d[1], d[2]);
    }
    case 3: {
        const auto c = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
        return 0.5 * std::hypot(c[0], c[1], c[2]);
    }
    case 4: {
        const auto c = cross(sub(corners[2], corners[0]), sub(corners[3], corners[1]));
        return 0.5 * std::hypot(c[0], c[1], c[2]);
    }
    default:
        throw InvalidStateException("NormalComponentConstraint: unsupported facet with " +
                                    std::to_string(corners.size()) + " corners");
    }
}

using CutRulesByFace = std::unordered_map<GlobalIndex, std::vector<const geometry::CutQuadratureRule*>>;

template<class Visit>
bool visitActiveBoundaryRules(const FESystem& system, int boundary_marker, int active_boundary_marker, Visit visit)
{
    if (active_boundary_marker < 0) {
        return false;
    }
    const auto* context = system.cutIntegrationContext();
    if (context == nullptr || !context->hasGeneratedActiveBoundaryMarker(active_boundary_marker)) {
        return false;
    }
    context->assertAllLevelSetGeometrySnapshotsCurrent(system.meshAccess());
    const auto* provenance = context->findGeneratedActiveBoundaryProvenance(active_boundary_marker);
    FE_THROW_IF(provenance == nullptr || provenance->physicalBoundaryMarker() != boundary_marker,
                InvalidArgumentException,
                "activeBoundaryFaces: generated active-boundary marker " + std::to_string(active_boundary_marker) +
                    " is not a subset of boundary marker " + std::to_string(boundary_marker));
    const auto& rules = context->interfaceRules();
    for (const auto index : context->generatedInterfaceRuleIndexSpanForMarker(active_boundary_marker)) {
        if (index >= rules.size()) {
            continue;
        }
        const auto& rule = rules[index];
        if (std::isfinite(rule.measure) && rule.measure > Real{0.0} &&
            rule.provenance.parent_boundary_entity >= 0) {
            visit(static_cast<GlobalIndex>(rule.provenance.parent_boundary_entity), rule);
        }
    }
    return true;
}

struct NormalMoment {
    std::array<Real, 3> vector{};
    Real scale{};
};

// Coefficient of each nodal velocity in the weak normal flux. For a
// generated subset the retained points and weights select its physical area.
std::vector<NormalMoment> normalFluxMoments(
    const geometry::GeometryMapping& mapping, ElementType cell_type,
    int local_face, ElementType face_type, int geometry_order, const elements::Element& element,
    std::span<const int> local_dofs, const std::vector<const geometry::CutQuadratureRule*>* active)
{
    std::vector<NormalMoment> moments(local_dofs.size());
    std::vector<Real> values;
    const auto accumulate = [&](const math::Vector<Real, 3>& xi, const auto& normal, Real weight) {
        element.basis().evaluate_values(xi, values);
        for (std::size_t a = 0; a < local_dofs.size(); ++a) {
            const Real coefficient = weight * values.at(local_dofs[a]);
            moments[a].scale += std::abs(coefficient);
            for (std::size_t d = 0; d < 3; ++d) moments[a].vector[d] += coefficient * normal[d];
        }
    };
    if (active) {
        // FacetFrame uses the two parent edges as 3D chart tangents. Cut
        // weights already include their reference metric; retain just the
        // physical-to-reference area ratio. Its 2D edge tangent has unit length.
        Real reference_metric{1};
        if (element.dimension() == 3) {
            const auto corners = elements::ElementTransform::facet_vertices(cell_type, local_face).second;
            const math::Vector<Real, 3> t0 = corners[1] - corners[0], t1 = corners[2] - corners[0];
            reference_metric = t0.cross(t1).norm();
        }
        for (const auto* rule : *active) {
            for (const auto& point : rule->points) {
                const bool reference = rule->frame == geometry::CutGeometryFrame::Reference;
                const auto& position = reference ? point.point : point.parent_coordinate;
                const math::Vector<Real, 3> xi{position[0], position[1], position[2]};
                const auto frame = elements::ElementTransform::compute_facet_frame(mapping, xi, local_face, cell_type);
                accumulate(xi, frame.normal, point.weight * (reference ? frame.jacobian_det / reference_metric : Real{1}));
            }
        }
    } else {
        // A tensor-product trace of degree r times its oriented area vector
        // has degree at most r + (d-1)*p - 1 in each facet coordinate.
        const int degree = element.polynomial_order() + (element.dimension() - 1) * geometry_order - 1;
        const auto quad = quadrature::QuadratureFactory::create_pinned(face_type, std::max(1, degree));
        geometry::FaceGeometryData frame;
        geometry::evaluateFaceFrameInto(frame, mapping, cell_type, local_face, face_type, *quad, {}, false, false);
        for (std::size_t q = 0; q < quad->num_points(); ++q) {
            const auto& p = frame.cell_reference_points[q];
            accumulate({p[0], p[1], p[2]}, frame.normals[q], quad->weight(q) * frame.surface_measures[q]);
        }
    }
    return moments;
}

/// The facet contributions this rank sees: the boundary facets of the
/// boundaries on its owned cells (the faces of a generated active
/// subset), at every support point of the field (of @p components
/// components) there.
std::vector<FacetContribution> collectContributions(const FESystem& system,
                                                    FieldId field,
                                                    std::optional<std::span<const forms::ExteriorBoundaryMeasure>> boundaries,
                                                    int components, bool normal_flux)
{
    const auto& rec = system.fieldRecord(field);
    const auto& mesh = system.meshAccess();
    const auto& dof_map = system.fieldDofHandler(field).getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field);

    const auto* product = dynamic_cast<const spaces::ProductSpace*>(rec.space.get());
    const auto& scalar_space = product ? *product->base_space() : *rec.space;
    std::map<std::pair<ElementType, int>, spaces::FaceRestriction> restrictions;
    std::map<std::pair<ElementType, int>, elements::LagrangeElement> mixed_elements;
    std::vector<std::array<Real, 3>> cell_coords;
    std::vector<math::Vector<Real, 3>> geom_nodes;
    std::vector<std::array<Real, 3>> corners;
    std::vector<FacetContribution> out;

    const auto collect_marker = [&](int marker, int active_marker) {
        CutRulesByFace active_rules;
        const bool restricted = visitActiveBoundaryRules(system, marker, active_marker,
            [&](GlobalIndex face, const auto& rule) { active_rules[face].push_back(&rule); });
        mesh.forEachBoundaryFace(marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
            if (restricted && !active_rules.contains(face_id)) {
                return;
            }
            const ElementType cell_type = mesh.getCellType(cell_id);
            const auto* element = &scalar_space.getElement(cell_type, cell_id);
            if (to_mesh_family(element->element_type()) != to_mesh_family(cell_type)) {
                const auto key = std::pair{cell_type, element->polynomial_order()};
                element = &mixed_elements.try_emplace(key, cell_type, key.second).first->second;
            }
            const auto key = std::pair{element->element_type(), element->polynomial_order()};
            auto it = restrictions.find(key);
            if (it == restrictions.end()) {
                it = element == &scalar_space.element()
                         ? restrictions.try_emplace(key, scalar_space).first
                         : restrictions.try_emplace(key, key.first, key.second, scalar_space.continuity()).first;
            }
            const auto& restriction = it->second;
            const auto face_local = static_cast<int>(mesh.getLocalFaceIndex(face_id, cell_id));

            const auto cell_dofs = dof_map.getCellDofs(cell_id);
            const auto per_component = cell_dofs.size() / static_cast<std::size_t>(components);
            FE_THROW_IF(per_component == 0 || per_component * static_cast<std::size_t>(components) != cell_dofs.size(),
                        InvalidStateException,
                        "NormalComponentConstraint: cell DOF list size is not divisible by field components");

            mesh.getCellCoordinates(cell_id, cell_coords);
            geom_nodes.resize(cell_coords.size());
            for (std::size_t i = 0; i < cell_coords.size(); ++i) {
                geom_nodes[i] = math::Vector<Real, 3>{cell_coords[i][0], cell_coords[i][1], cell_coords[i][2]};
            }
            geometry::MappingRequest request;
            request.element_type = cell_type;
            request.geometry_order = mesh.getCellGeometryOrder(cell_id);
            request.use_affine = request.geometry_order <= 1;
            auto mapping = geometry::MappingFactory::create(request, geom_nodes);
            FE_CHECK_NOT_NULL(mapping.get(), "NormalComponentConstraint: mapping");

            const auto& reference = elements::ReferenceElement::shared(cell_type);
            const auto& face_nodes = reference.face_nodes(static_cast<std::size_t>(face_local));
            const auto n_corners = faceCornerCount(cell_type, static_cast<std::size_t>(face_local));
            corners.clear();
            for (std::size_t k = 0; k < n_corners; ++k) {
                corners.push_back(cell_coords.at(static_cast<std::size_t>(face_nodes[k])));
            }
            const Real measure = facetMeasure(corners);
            const auto facet = static_cast<std::int64_t>(mesh.getBoundaryFaceGlobalId(face_id));

            const auto& dof_nodes = restriction.dof_nodes();
            const auto local_dofs = restriction.face_dofs(face_local);
            const ElementType face_type = n_corners == 2 ? ElementType::Line2
                                        : n_corners == 3 ? ElementType::Triangle3 : ElementType::Quad4;
            const bool linear_simplex = !restricted && element->polynomial_order() == 1 &&
                dynamic_cast<const geometry::LinearMapping*>(mapping.get()) &&
                dynamic_cast<const elements::LagrangeElement*>(element);
            const auto moments = normal_flux && !linear_simplex
                ? normalFluxMoments(*mapping, cell_type, face_local, face_type, request.geometry_order,
                                    *element, local_dofs, restricted ? &active_rules.at(face_id) : nullptr)
                : std::vector<NormalMoment>{};
            for (std::size_t a = 0; a < local_dofs.size(); ++a) {
                const int ldof = local_dofs[a];
                const auto local = static_cast<std::size_t>(ldof);
                FE_THROW_IF(ldof < 0 || local >= per_component || local >= dof_nodes.size(), InvalidStateException,
                            "NormalComponentConstraint: local DOF index out of range");
                FacetContribution c;
                c.facet = facet;
                c.local_cell = cell_id;
                c.measure = measure;

                for (int d = 0; d < components; ++d) {
                    c.dofs[static_cast<std::size_t>(d)] =
                        cell_dofs[static_cast<std::size_t>(d) * per_component + local] + offset;
                }
                const auto frame = elements::ElementTransform::compute_facet_frame(*mapping, dof_nodes[local],
                                                                                   face_local, cell_type);
                c.normal = {frame.normal[0], frame.normal[1], frame.normal[2]};
                if (normal_flux) {
                    if (linear_simplex) {
                        // On an affine simplex, each P1 trace has the same integral.
                        c.moment_scale = measure / static_cast<Real>(local_dofs.size());
                        for (std::size_t d = 0; d < 3; ++d) c.flux_moment[d] = c.moment_scale * c.normal[d];
                    } else {
                        c.flux_moment = moments[a].vector;
                        c.moment_scale = moments[a].scale;
                    }
                }
                const auto x = mapping->map_to_physical(dof_nodes[local]);
                c.x = {x[0], x[1], x[2]};
                out.push_back(c);
            }
        });
    };
    if (boundaries) {
        for (const auto& boundary : *boundaries)
            collect_marker(boundary.physicalBoundaryMarker(), boundary.generatedActiveBoundaryMarker());
    } else {
        collect_marker(-1, -1); // The whole exterior, including unlabelled facets.
    }
    return out;
}

/// Gives every DOF owner the contributions of its points that other ranks
/// saw, and keeps only the contributions of owned points.  Collective.
std::vector<FacetContribution> gatherAtOwners(const FESystem& system, std::vector<FacetContribution> local)
{
    const auto& owned = system.dofHandler().getPartition().locallyOwned();
#if FE_HAS_MPI
    static_assert(std::is_same_v<Real, double>, "NormalComponentConstraint sends Real as MPI_DOUBLE");
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    const MPI_Comm comm = (initialized != 0 && finalized == 0) ? system.dofHandler().mpiComm() : MPI_COMM_NULL;
    int size = 1;
    int rank = 0;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
    }
    if (size > 1) {
        FE_THROW_IF(!system.meshAccess().globalEntityIdsAvailable(), InvalidStateException,
                    "NormalComponentConstraint: a distributed system needs the mesh's global facet ids for a "
                    "rank-independent summation order");
        const auto& dof_map = system.dofHandler().getDofMap();
        constexpr std::size_t kInts = 4;   // facet, three DOFs
        constexpr std::size_t kReals = 11;  // nodal normal, facet measure, point, flux moment and scale
        std::vector<std::vector<std::int64_t>> send_ints(static_cast<std::size_t>(size));
        std::vector<std::vector<double>> send_reals(static_cast<std::size_t>(size));
        std::vector<FacetContribution> kept;
        for (const auto& c : local) {
            if (owned.contains(c.dofs[0])) {
                kept.push_back(c);
                continue;
            }
            const int owner = dof_map.getDofOwner(c.dofs[0]);
            FE_THROW_IF(owner < 0 || owner >= size || owner == rank, InvalidStateException,
                        "NormalComponentConstraint: boundary DOF " + std::to_string(c.dofs[0]) +
                            " has no valid remote owner");
            auto& ints = send_ints[static_cast<std::size_t>(owner)];
            ints.insert(ints.end(), {c.facet, c.dofs[0], c.dofs[1], c.dofs[2]});
            auto& reals = send_reals[static_cast<std::size_t>(owner)];
            reals.insert(reals.end(), {c.normal[0], c.normal[1], c.normal[2], c.measure, c.x[0], c.x[1], c.x[2],
                                      c.flux_moment[0], c.flux_moment[1], c.flux_moment[2], c.moment_scale});
        }

        std::vector<int> send_counts(static_cast<std::size_t>(size));
        for (int r = 0; r < size; ++r) {
            send_counts[static_cast<std::size_t>(r)] =
                static_cast<int>(send_ints[static_cast<std::size_t>(r)].size() / kInts);
        }
        std::vector<int> recv_counts(static_cast<std::size_t>(size));
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

        const auto exchange = [&](auto& send, std::size_t stride, MPI_Datatype type, auto value_tag) {
            using Value = decltype(value_tag);
            std::vector<int> scounts(static_cast<std::size_t>(size));
            std::vector<int> rcounts(static_cast<std::size_t>(size));
            std::vector<int> sdispls(static_cast<std::size_t>(size), 0);
            std::vector<int> rdispls(static_cast<std::size_t>(size), 0);
            std::vector<Value> flat;
            for (int r = 0; r < size; ++r) {
                const auto i = static_cast<std::size_t>(r);
                scounts[i] = send_counts[i] * static_cast<int>(stride);
                rcounts[i] = recv_counts[i] * static_cast<int>(stride);
                if (r > 0) {
                    sdispls[i] = sdispls[i - 1] + scounts[i - 1];
                    rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
                }
                flat.insert(flat.end(), send[i].begin(), send[i].end());
            }
            std::vector<Value> received(static_cast<std::size_t>(rdispls.back() + rcounts.back()));
            MPI_Alltoallv(flat.data(), scounts.data(), sdispls.data(), type, received.data(), rcounts.data(),
                          rdispls.data(), type, comm);
            return received;
        };
        const auto ints = exchange(send_ints, kInts, MPI_INT64_T, std::int64_t{});
        const auto reals = exchange(send_reals, kReals, MPI_DOUBLE, double{});
        const std::size_t n = ints.size() / kInts;
        for (std::size_t k = 0; k < n; ++k) {
            FacetContribution c;
            c.facet = ints[kInts * k];
            c.dofs = {static_cast<GlobalIndex>(ints[kInts * k + 1]), static_cast<GlobalIndex>(ints[kInts * k + 2]),
                      static_cast<GlobalIndex>(ints[kInts * k + 3])};
            c.normal = {reals[kReals * k], reals[kReals * k + 1], reals[kReals * k + 2]};
            c.measure = reals[kReals * k + 3];
            c.x = {reals[kReals * k + 4], reals[kReals * k + 5], reals[kReals * k + 6]};
            c.flux_moment = {reals[kReals * k + 7], reals[kReals * k + 8], reals[kReals * k + 9]};
            c.moment_scale = reals[kReals * k + 10];
            FE_THROW_IF(!owned.contains(c.dofs[0]), InvalidStateException,
                        "NormalComponentConstraint: received a point this rank does not own");
            kept.push_back(c);
        }
        return kept;
    }
#endif
    std::erase_if(local, [&](const FacetContribution& c) { return !owned.contains(c.dofs[0]); });
    return local;
}

/// Solves the k x k system A x = b (rows of A in @p a), Gaussian elimination
/// with partial pivoting in a fixed order.
std::vector<Real> solveSmall(std::vector<std::vector<Real>> a, std::vector<Real> b)
{
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t row = col + 1; row < n; ++row) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) {
                pivot = row;
            }
        }
        FE_THROW_IF(std::abs(a[pivot][col]) <= 1e-14, InvalidStateException,
                    "NormalComponentConstraint: the normal directions at a point are linearly dependent");
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const Real factor = a[row][col] / a[col][col];
            for (std::size_t k = col; k < n; ++k) {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    std::vector<Real> x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        Real sum = b[i];
        for (std::size_t k = i + 1; k < n; ++k) {
            sum -= a[i][k] * x[k];
        }
        x[i] = sum / a[i][i];
    }
    return x;
}

/// The normal directions of one point from its contributions, sorted by
/// facet id: a facet normal joins the first direction whose first normal is
/// within the feature angle; each direction is the normalized
/// measure-weighted sum of its normals.
std::vector<std::array<Real, 3>> pointDirections(std::span<const FacetContribution> contributions,
                                                 Real cos_feature_angle, bool normal_flux)
{
    std::vector<std::array<Real, 3>> seeds;
    std::vector<std::array<Real, 3>> sums;
    std::vector<std::array<Real, 3>> moments;
    std::vector<Real> scales;
    for (const auto& c : contributions) {
        std::size_t g = 0;
        while (g < seeds.size() && dot3(seeds[g], c.normal) < cos_feature_angle) {
            ++g;
        }
        if (g == seeds.size()) {
            seeds.push_back(c.normal);
            sums.push_back({});
            moments.push_back({});
            scales.push_back(0);
        }
        scales[g] += c.moment_scale;
        for (int d = 0; d < 3; ++d) {
            moments[g][d] += c.flux_moment[d];
            sums[g][static_cast<std::size_t>(d)] += c.measure * c.normal[static_cast<std::size_t>(d)];
        }
    }
    for (std::size_t g = 0; g < sums.size(); ++g) {
        auto& sum = sums[g];
        const auto& m = moments[g];
        // A quadratic triangle, for example, has zero integral of each
        // vertex shape function on an affine face. Its pointwise outward
        // normal remains a valid constraint on that flux-free coefficient.
        const Real length = std::hypot(sum[0], sum[1], sum[2]);
        FE_THROW_IF(!(length > 0.0), InvalidStateException, "NormalComponentConstraint: a zero normal direction");
        for (auto& c : sum) c /= length;
        const Real moment_length = std::hypot(m[0], m[1], m[2]);
        constexpr Real roundoff = 32 * std::numeric_limits<Real>::epsilon();
        if (!normal_flux || moment_length <= roundoff * scales[g]) continue;
        std::array<Real, 3> direction{m[0] / moment_length, m[1] / moment_length, m[2] / moment_length};
        if (dot3(sum, direction) < 0) {
            for (auto& c : direction) c = -c;
        }
        // Preserve an exact planar direction when integration perturbs it
        // only by roundoff, e.g. for unit normal data on a Cartesian box.
        if (std::hypot(direction[0] - sum[0], direction[1] - sum[1], direction[2] - sum[2]) > roundoff) sum = direction;
    }
    return sums;
}

/// The contributions of each owned point, sorted by (first DOF, facet id).
std::vector<FacetContribution> ownedPointContributions(const FESystem& system,
                                                       FieldId field,
                                                       std::span<const forms::ExteriorBoundaryMeasure> boundaries,
                                                       int components, bool normal_flux)
{
    auto contributions = gatherAtOwners(system, collectContributions(system, field, boundaries, components, normal_flux));
    std::sort(contributions.begin(), contributions.end(), [](const FacetContribution& a, const FacetContribution& b) {
        return a.dofs[0] != b.dofs[0] ? a.dofs[0] < b.dofs[0] : a.facet < b.facet;
    });
    return contributions;
}

/// Calls @p visit with the contributions of each point, in turn.
template <class Visit>
void forEachPoint(const std::vector<FacetContribution>& contributions, Visit&& visit)
{
    for (std::size_t first = 0; first < contributions.size();) {
        std::size_t last = first + 1;
        while (last < contributions.size() && contributions[last].dofs[0] == contributions[first].dofs[0]) {
            ++last;
        }
        visit(std::span<const FacetContribution>(contributions.data() + first, last - first));
        first = last;
    }
}

/// The lines of one point from its contributions, sorted by facet id.
std::vector<NormalComponentConstraint::Line> pointLines(std::span<const FacetContribution> contributions,
                                                        int dim,
                                                        Real cos_feature_angle)
{
    const auto directions = pointDirections(contributions, cos_feature_angle, true);
    const std::size_t k = directions.size();
    std::vector<std::vector<Real>> normals(k, std::vector<Real>(static_cast<std::size_t>(dim)));
    for (std::size_t g = 0; g < k; ++g) {
        for (int d = 0; d < dim; ++d) {
            normals[g][static_cast<std::size_t>(d)] = directions[g][static_cast<std::size_t>(d)];
        }
    }

    const auto& first = contributions.front();
    std::vector<NormalComponentConstraint::Line> lines;
    const auto D = static_cast<std::size_t>(dim);
    if (k >= D) {
        // Every component: N u = g 1, or N^T N u = g N^T 1 when k > dim.
        auto a = normals;
        std::vector<Real> b(D, 1.0);
        if (k > D) {
            a.assign(D, std::vector<Real>(D, 0.0));
            b.assign(D, 0.0);
            for (std::size_t i = 0; i < D; ++i) {
                for (std::size_t g = 0; g < k; ++g) {
                    for (std::size_t j = 0; j < D; ++j) {
                        a[i][j] += normals[g][i] * normals[g][j];
                    }
                    b[i] += normals[g][i];
                }
            }
        }
        const auto factors = solveSmall(std::move(a), std::move(b));
        for (std::size_t c = 0; c < D; ++c) {
            NormalComponentConstraint::Line line;
            line.slave = first.dofs[c];
            line.value_factor = factors[c];
            line.x = first.x;
            lines.push_back(std::move(line));
        }
        return lines;
    }

    // k < dim: the k components with the largest |det N_S| are slaves.
    std::vector<std::size_t> best;
    Real best_det = -1.0;
    std::vector<std::size_t> pick(k);
    std::iota(pick.begin(), pick.end(), std::size_t{0});
    while (true) {
        Real det = 0.0;
        if (k == 1) {
            det = normals[0][pick[0]];
        } else {
            det = normals[0][pick[0]] * normals[1][pick[1]] - normals[0][pick[1]] * normals[1][pick[0]];
        }
        if (std::abs(det) > best_det) {
            best_det = std::abs(det);
            best = pick;
        }
        // Next combination in lexicographic order.
        std::size_t i = k;
        while (i-- > 0 && pick[i] == D - k + i) {
        }
        if (i == std::numeric_limits<std::size_t>::max()) {
            break;
        }
        ++pick[i];
        for (std::size_t j = i + 1; j < k; ++j) {
            pick[j] = pick[j - 1] + 1;
        }
    }
    std::vector<std::size_t> rest;
    for (std::size_t c = 0; c < D; ++c) {
        if (std::find(best.begin(), best.end(), c) == best.end()) {
            rest.push_back(c);
        }
    }

    // u_S = N_S^{-1} (g 1 - N_R u_R): one solve per right-hand side.
    std::vector<std::vector<Real>> a(k, std::vector<Real>(k));
    for (std::size_t g = 0; g < k; ++g) {
        for (std::size_t s = 0; s < k; ++s) {
            a[g][s] = normals[g][best[s]];
        }
    }
    const auto factors = solveSmall(a, std::vector<Real>(k, 1.0));
    std::vector<std::vector<Real>> weights(rest.size());
    for (std::size_t r = 0; r < rest.size(); ++r) {
        std::vector<Real> rhs(k);
        for (std::size_t g = 0; g < k; ++g) {
            rhs[g] = -normals[g][rest[r]];
        }
        weights[r] = solveSmall(a, std::move(rhs));
    }
    for (std::size_t s = 0; s < k; ++s) {
        NormalComponentConstraint::Line line;
        line.slave = first.dofs[best[s]];
        for (std::size_t r = 0; r < rest.size(); ++r) {
            const Real w = weights[r][s];
            if (w != 0.0) {
                line.masters.push_back(first.dofs[rest[r]]);
                line.weights.push_back(w);
            }
        }
        line.value_factor = factors[s];
        line.x = first.x;
        lines.push_back(std::move(line));
    }
    return lines;
}

/// The whole faces of each of @p markers.
std::vector<forms::ExteriorBoundaryMeasure> fullBoundaries(std::span<const int> markers)
{
    std::vector<forms::ExteriorBoundaryMeasure> out;
    out.reserve(markers.size());
    for (const int marker : markers) {
        if (marker < 0) {
            throw std::invalid_argument("NormalComponentConstraint: the boundary markers must be nonempty and >= 0");
        }
        out.push_back(forms::ExteriorBoundaryMeasure::fullPhysical(marker));
    }
    return out;
}

} // namespace

std::vector<int> regionsWithFreeNormalFlux(
    const FESystem& system, FieldId field,
    const AffineConstraints& test_constraints, std::span<const int> cell_regions)
{
#if FE_HAS_MPI
    const systems::CollectiveCommunicator comm(system.dofHandler().mpiComm());
#else
    const systems::CollectiveCommunicator comm;
#endif
    // For a constant scalar multiplier the divergence row is the boundary
    // functional l. Its coefficient on an independent test DOF is (P^T l)_i.
    struct Coefficient { Real value{}, scale{}; };
    std::map<std::pair<int, GlobalIndex>, Coefficient> coefficients;
    std::vector<std::uint64_t> words;
    std::exception_ptr failure;
    try {
        const auto dim = system.meshAccess().dimension();
        for (const auto& c : collectContributions(system, field, std::nullopt, dim, true)) {
            if (!system.meshAccess().isOwnedCell(c.local_cell)) continue;
            const int region = cell_regions.empty() ? -1 : cell_regions[c.local_cell];
            for (int d = 0; d < dim; ++d) {
                auto add = [&](GlobalIndex master, Real weight) {
                    auto& coefficient = coefficients[{region, master}];
                    coefficient.value += weight * c.flux_moment[d];
                    coefficient.scale += std::abs(weight) * c.moment_scale;
                };
                if (const auto line = test_constraints.getConstraint(c.dofs[d])) {
                    for (const auto& e : line->entries) add(e.master_dof, e.weight);
                } else {
                    add(c.dofs[d], Real{1});
                }
            }
        }
        for (const auto& [key, w] : coefficients) {
            words.insert(words.end(), {static_cast<std::uint64_t>(key.first + 1),
                static_cast<std::uint64_t>(key.second), std::bit_cast<std::uint64_t>(w.value),
                std::bit_cast<std::uint64_t>(w.scale)});
        }
    } catch (...) {
        failure = std::current_exception();
    }
    systems::requireCollectivePhasePreparation(failure, comm, "boundary normal test flux");
    const auto all = systems::collectiveGatherWords(words, comm, "boundary normal test flux");
    coefficients.clear();
    for (std::size_t i = 0; i < all.size(); i += 4) {
        auto& w = coefficients[{static_cast<int>(all[i]) - 1, static_cast<GlobalIndex>(all[i + 1])}];
        w.value += std::bit_cast<Real>(all[i + 2]);
        w.scale += std::bit_cast<Real>(all[i + 3]);
    }
    std::vector<int> regions;
    for (const auto& [key, w] : coefficients) {
        if (std::abs(w.value) > 64 * std::numeric_limits<Real>::epsilon() * w.scale &&
            (regions.empty() || regions.back() != key.first)) regions.push_back(key.first);
    }
    return regions;
}

std::optional<std::unordered_set<GlobalIndex>>
activeBoundaryFaces(const FESystem& system, int boundary_marker, int active_boundary_marker)
{
    std::unordered_set<GlobalIndex> faces;
    if (!visitActiveBoundaryRules(system, boundary_marker, active_boundary_marker,
            [&](GlobalIndex face, const auto&) { faces.insert(face); })) return std::nullopt;
    return faces;
}

std::unordered_map<GlobalIndex, std::optional<std::array<Real, 3>>>
boundaryPointNormals(const FESystem& system,
                     FieldId field,
                     std::span<const int> boundary_markers,
                     Real feature_angle_degrees)
{
    const auto& rec = system.fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "boundaryPointNormals: field space");
    FE_THROW_IF(rec.space->continuity() == Continuity::H_curl || rec.space->continuity() == Continuity::H_div ||
                    rec.space->element().basis().is_vector_valued(),
                InvalidArgumentException, "boundaryPointNormals: field '" + rec.name + "' must be a Lagrange field");
    FE_THROW_IF(!(feature_angle_degrees > 0.0 && feature_angle_degrees < 90.0), InvalidArgumentException,
                "boundaryPointNormals: the feature angle must lie in (0, 90) degrees");
    const int components = std::max(1, rec.space->value_dimension());
    FE_THROW_IF(components > 3, InvalidArgumentException,
                "boundaryPointNormals: field '" + rec.name + "' has more than three components");
    const Real cos_feature_angle = std::cos(feature_angle_degrees * kPi / 180.0);

    std::unordered_map<GlobalIndex, std::optional<std::array<Real, 3>>> normals;
    forEachPoint(ownedPointContributions(system, field, fullBoundaries(boundary_markers), components, false),
                 [&](std::span<const FacetContribution> point) {
                     const auto directions = pointDirections(point, cos_feature_angle, false);
                     std::optional<std::array<Real, 3>> normal;
                     if (directions.size() == 1u) {
                         normal = directions.front();
                     }
                     for (int d = 0; d < components; ++d) {
                         normals.emplace(point.front().dofs[static_cast<std::size_t>(d)], normal);
                     }
                 });
    return normals;
}

NormalComponentConstraint::NormalComponentConstraint(FieldId field,
                                                     std::vector<int> boundary_markers,
                                                     forms::FormExpr value,
                                                     NormalComponentOptions options)
    : NormalComponentConstraint(field, fullBoundaries(boundary_markers), std::move(value), options)
{
}

NormalComponentConstraint::NormalComponentConstraint(FieldId field,
                                                     std::vector<forms::ExteriorBoundaryMeasure> boundaries,
                                                     forms::FormExpr value,
                                                     NormalComponentOptions options)
    : field_(field)
    , boundaries_(std::move(boundaries))
    , value_(std::move(value))
    , options_(options)
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("NormalComponentConstraint: invalid FieldId");
    }
    if (boundaries_.empty()) {
        throw std::invalid_argument("NormalComponentConstraint: the boundary markers must be nonempty and >= 0");
    }
    std::sort(boundaries_.begin(), boundaries_.end(), [](const auto& a, const auto& b) {
        return a.physicalBoundaryMarker() < b.physicalBoundaryMarker();
    });
    boundaries_.erase(std::unique(boundaries_.begin(), boundaries_.end()), boundaries_.end());
    for (std::size_t i = 1; i < boundaries_.size(); ++i) {
        if (boundaries_[i].physicalBoundaryMarker() == boundaries_[i - 1].physicalBoundaryMarker()) {
            throw std::invalid_argument("NormalComponentConstraint: boundary marker " +
                                        std::to_string(boundaries_[i].physicalBoundaryMarker()) +
                                        " is given with two different subsets");
        }
    }
    if (!value_.isValid() || value_.hasTest() || value_.hasTrial()) {
        throw std::invalid_argument("NormalComponentConstraint: the value must be a valid expression without "
                                    "test or trial functions");
    }
    if (!(options_.feature_angle_degrees > 0.0 && options_.feature_angle_degrees < 90.0)) {
        throw std::invalid_argument("NormalComponentConstraint: the feature angle must lie in (0, 90) degrees");
    }
    is_time_dependent_ = forms::isTimeDependent(value_);
    normal_frame_can_change_ = std::any_of(boundaries_.begin(), boundaries_.end(),
        [](const auto& boundary) { return boundary.isGeneratedActiveSubset(); });
}

void NormalComponentConstraint::apply(const FESystem& system, AffineConstraints& constraints)
{
    const auto& rec = system.fieldRecord(field_);
    FE_CHECK_NOT_NULL(rec.space.get(), "NormalComponentConstraint: field space");
    FE_THROW_IF(rec.space->continuity() == Continuity::H_curl || rec.space->continuity() == Continuity::H_div ||
                    rec.space->element().basis().is_vector_valued(),
                InvalidArgumentException,
                "NormalComponentConstraint: field '" + rec.name +
                    "' must be a vector Lagrange field; for H(div) use HDivNormalConstraint");
    const int dim = system.meshAccess().dimension();
    FE_THROW_IF(dim < 2 || dim > 3 || rec.space->value_dimension() != dim, InvalidArgumentException,
                "NormalComponentConstraint: field '" + rec.name + "' must have one component per space dimension");

    const auto contributions = ownedPointContributions(system, field_, boundaries_, dim, true);
    const auto revision = system.meshAccess().geometryRevision();
    if (!initial_geometry_revision_) {
        initial_geometry_revision_ = revision;
    }
    normal_frame_can_change_ |= revision != *initial_geometry_revision_;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    normal_frame_can_change_ |= system.coordinateConfiguration() != svmp::Configuration::Reference;
#endif
    const Real cos_feature_angle = std::cos(options_.feature_angle_degrees * kPi / 180.0);
    forms::PointEvalContext pctx;
    lines_.clear();
    forEachPoint(contributions, [&](std::span<const FacetContribution> point) {
        bool held = false;
        for (int d = 0; d < dim; ++d) {
            const auto dof = point.front().dofs[static_cast<std::size_t>(d)];
            FE_THROW_IF(!system.dofHandler().getPartition().locallyOwned().contains(dof), InvalidStateException,
                        "NormalComponentConstraint: the components of a support point have different owners");
            held = held || constraints.isConstrained(dof);
        }
        if (held) {
            return;
        }
        for (auto& line : pointLines(point, dim, cos_feature_angle)) {
            pctx.x = line.x;
            const Real g = forms::evaluateScalarAt(value_, pctx);
            constraints.addLine(line.slave);
            for (std::size_t m = 0; m < line.masters.size(); ++m) {
                constraints.addEntry(line.slave, line.masters[m], line.weights[m]);
            }
            constraints.setInhomogeneity(line.slave, g * line.value_factor);
            if ((is_time_dependent_ && line.value_factor != 0.0) ||
                (normal_frame_can_change_ && !line.masters.empty())) {
                // Accepted states and rates belong to the normal at their own time.
                constraints.setInstantaneous(line.slave);
            }
            lines_.push_back(std::move(line));
        }
    });
}

bool NormalComponentConstraint::updateValues(const FESystem& /*system*/,
                                             AffineConstraints& constraints,
                                             double time,
                                             double dt)
{
    forms::PointEvalContext pctx;
    pctx.time = static_cast<Real>(time);
    pctx.dt = static_cast<Real>(dt);
    for (const auto& line : lines_) {
        pctx.x = line.x;
        constraints.updateInhomogeneity(line.slave, forms::evaluateScalarAt(value_, pctx) * line.value_factor);
    }
    return true;
}

ConstraintDependencyDeclaration NormalComponentConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = ISystemConstraint::dependencyDeclaration();
    // The normals, and so the slaves, follow the geometry.
    merge_into(out.structural, ConstraintDependencyMask::meshGeometry());
    if (std::any_of(boundaries_.begin(), boundaries_.end(),
                    [](const auto& boundary) { return boundary.isGeneratedActiveSubset(); })) {
        // The active faces come from the installed cut-integration context; a
        // new context rebuilds the constraint state.
        out.structural.fe_constraint_layout = true;
        out.structural.mesh_field_layout = true;
        out.structural.active_configuration = true;
    }
    return out;
}

systems::SetupStorageRequirements NormalComponentConstraint::storageRequirements() const noexcept
{
    systems::SetupStorageRequirements req;
    req.entity_dof_map = true;
    req.boundary_face_topology = true;
    return req;
}

std::string NormalComponentConstraint::manifestDescription() const
{
    return manifestDescription("active");
}

std::string NormalComponentConstraint::manifestDescription(std::string_view active_subset_label) const
{
    std::ostringstream oss;
    oss << "field=" << field_ << " markers=";
    for (std::size_t i = 0; i < boundaries_.size(); ++i) {
        oss << (i > 0 ? "," : "") << boundaries_[i].physicalBoundaryMarker();
        if (boundaries_[i].isGeneratedActiveSubset()) {
            oss << "(" << active_subset_label << " " << boundaries_[i].generatedActiveBoundaryMarker() << ")";
        }
    }
    oss << " feature_angle=" << options_.feature_angle_degrees
        << " time_dependent=" << (is_time_dependent_ ? 1 : 0) << " value=" << value_.toString();
    return oss.str();
}

} // namespace constraints
} // namespace FE
} // namespace svmp
