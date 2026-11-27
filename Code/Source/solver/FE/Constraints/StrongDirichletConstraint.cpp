/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/StrongDirichletConstraint.h"

#include "Constraints/BoundaryDofOwnerCompletion.h"
#include "Constraints/NormalComponentConstraint.h"

#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"
#include "Systems/TimeSeries.h"

#include "Core/Logger.h"

#include "Forms/PointEvaluator.h"

#include "Geometry/MappingFactory.h"
#include "Constraints/BoundaryFieldTrace.h"
#include "Spaces/AdaptiveSpace.h"
#include "Elements/ElementFactory.h"
#include "Basis/BernsteinBasis.h"
#include "Math/DenseLinearAlgebra.h"
#include "Systems/SystemCollectives.h"
#include "Core/EnvironmentNames.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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

using TraceKey = std::tuple<ElementType, BasisType, int, int>;

struct BernsteinTrace {
    TraceKey key;
    std::vector<GlobalIndex> dofs;
};

struct BoundaryDofsWithCoords {
    std::vector<GlobalIndex> dofs;
    std::vector<std::array<Real, 3>> coords;
    std::map<TraceKey, math::DenseLUSolver> interpolation;
    std::vector<BernsteinTrace> bernstein_faces;
    std::vector<int> sample_components;  ///< per sampled field
    std::vector<Real> samples;           ///< the sampled field values, DOF by DOF
};

/// Report a failed rank-local interpolation before another rank starts exchanging point data.
template<class Operation>
void prepareTrace(const FESystem& system, bool distributed_interpolation, std::string_view phase, Operation operation)
{
    if (!distributed_interpolation) {
        operation();
        return;
    }
    std::exception_ptr failure;
    try { operation(); } catch (...) { failure = std::current_exception(); }
#if FE_HAS_MPI
    const systems::CollectiveCommunicator comm(system.dofHandler().mpiComm());
#else
    const systems::CollectiveCommunicator comm;
#endif
    systems::requireCollectivePhasePreparation(failure, comm, phase);
}

/// A prescribed field a strong value reads, sampled at the constrained nodes.
struct SampledField {
    std::string name;
    const dofs::DofMap* dof_map{nullptr};
    std::span<const Real> coefficients{};
    int components{1};
    const spaces::FunctionSpace* space{nullptr};
};

/// The parameter names and field ids of @p node, each once in tree order,
/// and whether it reads the boundary normal.
void gatherStrongValueInputs(const forms::FormExprNode& node,
                             std::vector<std::string>& parameter_names,
                             std::vector<FieldId>& fields,
                             bool& reads_normal)
{
    switch (node.type()) {
    case forms::FormExprType::Normal:
        reads_normal = true;
        return;
    case forms::FormExprType::ParameterSymbol: {
        const std::string name(node.symbolName().value_or(std::string_view{}));
        if (name.empty()) {
            throw std::invalid_argument("StrongDirichletConstraint: a parameter of the value has no name");
        }
        if (std::find(parameter_names.begin(), parameter_names.end(), name) == parameter_names.end()) {
            parameter_names.push_back(name);
        }
        return;
    }
    case forms::FormExprType::DiscreteField: {
        const auto field = node.fieldId().value_or(INVALID_FIELD_ID);
        if (std::find(fields.begin(), fields.end(), field) == fields.end()) {
            fields.push_back(field);
        }
        return;
    }
    case forms::FormExprType::StateField:
        throw std::invalid_argument(
            "StrongDirichletConstraint: a strong value must not depend on an unknown (StateField); "
            "read a prescribed field instead");
    default:
        break;
    }
    for (const auto& child : node.childrenShared()) {
        if (child) {
            gatherStrongValueInputs(*child, parameter_names, fields, reads_normal);
        }
    }
}

[[nodiscard]] bool strongDirichletTraceEnabled() noexcept
{
    const char* env = environment::value("SVMP_TRACE_STRONG_DIRICHLET");
    if (env != nullptr && env[0] != '\0') {
        return true;
    }
    env = environment::value("SVMP_OOP_SOLVER_TRACE");
    return env != nullptr && env[0] != '\0';
}

[[nodiscard]] int strongDirichletTraceRank() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
#endif
    return 0;
}

void traceStrongDirichletEvent(const std::string& phase,
                               int boundary_marker,
                               int component,
                               std::size_t total_dofs,
                               std::size_t owned_dofs,
                               std::size_t constrained_dofs,
                               std::size_t missing_constraints,
                               double time,
                               double dt,
                               double min_value,
                               double max_value,
                               const std::string& sample_summary)
{
    if (!strongDirichletTraceEnabled()) {
        return;
    }

    std::ostringstream oss;
    oss << "StrongDirichletConstraint[" << phase << "]"
        << " rank=" << strongDirichletTraceRank()
        << " marker=" << boundary_marker
        << " component=" << component
        << " total_dofs=" << total_dofs
        << " owned_dofs=" << owned_dofs
        << " constrained_dofs=" << constrained_dofs
        << " missing_constraints=" << missing_constraints
        << " time=" << time
        << " dt=" << dt;
    if (total_dofs > 0) {
        oss << " value_range=[" << min_value << "," << max_value << "]";
    }
    if (!sample_summary.empty()) {
        oss << " samples=" << sample_summary;
    }
    FE_LOG_INFO(oss.str());
}

/// The single boundary normal at a constrained node; refuses a node where
/// the marker's facets meet beyond the feature angle (an edge or corner).
std::array<Real, 3> boundaryNormalAt(
    const std::unordered_map<GlobalIndex, std::optional<std::array<Real, 3>>>& normals,
    GlobalIndex dof,
    const std::array<Real, 3>& x,
    int boundary_marker)
{
    const auto it = normals.find(dof);
    FE_THROW_IF(it == normals.end(), InvalidStateException,
                "StrongDirichletConstraint: no boundary normal at DOF " + std::to_string(dof));
    FE_THROW_IF(!it->second.has_value(), InvalidArgumentException,
                "StrongDirichletConstraint: a strong value reads the normal of marker " +
                    std::to_string(boundary_marker) + ", which has no single direction at (" + std::to_string(x[0]) +
                    ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) +
                    "), where its facets meet at an edge or corner");
    return *it->second;
}

/// The values of the sampled fields at one node, from @p samples at @p first.
void fillFieldValues(std::span<const FieldId> fields,
                     std::span<const int> components,
                     std::span<const Real> samples,
                     std::size_t first,
                     std::vector<FieldValueEntry>& out)
{
    std::size_t k = first;
    for (std::size_t f = 0; f < fields.size(); ++f) {
        auto& entry = out[f];
        entry.field = fields[f];
        entry.n_components = components[f];
        FE_THROW_IF(entry.n_components > MAX_FIELD_VALUE_COMPONENTS, InvalidArgumentException,
                    "StrongDirichletConstraint: a sampled field has too many components");
        for (int c = 0; c < entry.n_components; ++c) {
            entry.components[c] = samples[k++];
        }
    }
}

void completeOwnedDofsWithCoords(const FESystem& system,
                                 BoundaryDofsWithCoords& out,
                                 const std::string& context)
{
    const auto n_samples = std::accumulate(out.sample_components.begin(), out.sample_components.end(), std::size_t{});
    const std::size_t stride = 3u + n_samples;
    std::vector<Real> payload;
    payload.reserve(out.coords.size() * stride);
    for (std::size_t i = 0; i < out.coords.size(); ++i) {
        payload.insert(payload.end(), out.coords[i].begin(), out.coords[i].end());
        payload.insert(payload.end(),
                       out.samples.begin() + static_cast<std::ptrdiff_t>(i * n_samples),
                       out.samples.begin() + static_cast<std::ptrdiff_t>((i + 1u) * n_samples));
    }
    if (completeOwnedBoundaryDofs(system, out.dofs, payload, stride, context) == 0u) {
        return;
    }
    out.coords.resize(out.dofs.size());
    out.samples.resize(out.dofs.size() * n_samples);
    for (std::size_t i = 0; i < out.dofs.size(); ++i) {
        out.coords[i] = {payload[stride * i], payload[stride * i + 1u], payload[stride * i + 2u]};
        std::copy_n(payload.begin() + static_cast<std::ptrdiff_t>(stride * i + 3u),
                    n_samples,
                    out.samples.begin() + static_cast<std::ptrdiff_t>(n_samples * i));
    }
}

[[nodiscard]] BoundaryDofsWithCoords boundaryDofsWithCoordsByMarker(
    const FESystem& system,
    FieldId field,
    int boundary_marker,
    int component,
    std::span<const FieldId> sampled_field_ids = {},
    bool require_coefficients = true,
    const std::unordered_set<GlobalIndex>* active_faces = nullptr,
    bool interpolate_bernstein = false)
{
    BoundaryDofsWithCoords out;

    const auto& rec = system.fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "StrongDirichletConstraint: field space");
    FE_THROW_IF(rec.space->continuity() == Continuity::H_curl || rec.space->continuity() == Continuity::H_div ||
                    rec.space->element().basis().is_vector_valued(),
                InvalidArgumentException,
                "StrongDirichletConstraint does not support H(curl)/H(div) vector-basis spaces; use the dedicated H(curl)/H(div) constraint types instead");

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field);

    const int n_components = std::max(1, rec.space->value_dimension());
    if (component < -1) {
        throw std::invalid_argument("StrongDirichletConstraint: component must be >= -1");
    }
    if (component >= n_components) {
        throw std::invalid_argument("StrongDirichletConstraint: component index out of range");
    }

    // A sampled field shares the constrained field's scalar element, so its
    // coefficient at a constrained node is the one at the same local index.
    std::vector<SampledField> sampled;
    std::size_t n_samples = 0;
    for (const auto id : sampled_field_ids) {
        const auto& sampled_rec = system.fieldRecord(id);
        FE_THROW_IF(sampled_rec.source_kind != systems::FieldSourceKind::PrescribedData, InvalidArgumentException,
                    "StrongDirichletConstraint: the value reads field '" + sampled_rec.name +
                        "', which is not prescribed data; a strong value must not depend on an unknown");
        FE_CHECK_NOT_NULL(sampled_rec.space.get(), "StrongDirichletConstraint: sampled field space");
        FE_THROW_IF(sampled_rec.space->polynomial_order() != rec.space->polynomial_order() ||
                        sampled_rec.space->continuity() != rec.space->continuity() ||
                        sampled_rec.space->element().basis().is_vector_valued(),
                    InvalidArgumentException,
                    "StrongDirichletConstraint: field '" + sampled_rec.name +
                        "' in a strong value must share the scalar element of field '" + rec.name + "'");
        SampledField f;
        f.name = sampled_rec.name;
        f.space = &detail::scalarBoundarySpace(*sampled_rec.space);
        f.dof_map = &system.fieldDofHandler(id).getDofMap();
        f.coefficients = system.prescribedFieldCoefficients(id);
        f.components = std::max(1, sampled_rec.space->value_dimension());
        // At setup the coefficients may not be set yet: the value reads zeros
        // until the first update, which needs them.
        FE_THROW_IF(!f.coefficients.empty() &&
                        f.coefficients.size() != static_cast<std::size_t>(system.fieldDofHandler(id).getNumDofs()),
                    InvalidStateException,
                    "StrongDirichletConstraint: prescribed field '" + f.name + "' has the wrong number of coefficients");
        FE_THROW_IF(f.coefficients.empty() && require_coefficients, InvalidStateException,
                    "StrongDirichletConstraint: prescribed field '" + f.name +
                        "' in a strong value has no coefficients; set them before the constraints are updated");
        out.sample_components.push_back(f.components);
        n_samples += static_cast<std::size_t>(f.components);
        sampled.push_back(std::move(f));
    }

    std::unordered_map<GlobalIndex, std::array<Real, 3>> coord_by_dof;
    std::unordered_map<GlobalIndex, std::vector<Real>> samples_by_dof;

    std::vector<std::array<Real, 3>> cell_coords;
    std::vector<math::Vector<Real, 3>> geom_nodes;

    detail::BoundaryFieldTraces traces;
    std::map<std::pair<const spaces::FunctionSpace*, ElementType>, std::shared_ptr<elements::Element>> mixed_sources;
    const auto& scalar = detail::scalarBoundarySpace(*rec.space);
    const auto collect_face = [&](GlobalIndex face_id, GlobalIndex cell_id) {
        if (active_faces != nullptr && !active_faces->contains(face_id)) {
            return;
        }
        const ElementType cell_type = mesh.getCellType(cell_id);

        const auto& space = scalar.element_space(cell_id);
        const auto& basis = space.element().basis();
        const auto& restriction = traces.get(space, cell_type);
        const auto face_local = static_cast<int>(mesh.getLocalFaceIndex(face_id, cell_id));
        const auto local_face_dofs = restriction.face_dofs(face_local);

        const auto cell_dofs = dof_map.getCellDofs(cell_id);
        FE_THROW_IF(cell_dofs.empty(), InvalidStateException,
                    "StrongDirichletConstraint: empty cell DOF list");

        const auto dofs_per_component = cell_dofs.size() / static_cast<std::size_t>(n_components);
        FE_THROW_IF(dofs_per_component == 0 || dofs_per_component * static_cast<std::size_t>(n_components) != cell_dofs.size(),
                    InvalidStateException,
                    "StrongDirichletConstraint: cell DOF list size is not divisible by field components");

        cell_coords.clear();
        mesh.getCellCoordinates(cell_id, cell_coords);
        FE_THROW_IF(cell_coords.empty(), InvalidStateException,
                    "StrongDirichletConstraint: empty cell coordinate list");

        geom_nodes.resize(cell_coords.size());
        for (std::size_t i = 0; i < cell_coords.size(); ++i) {
            geom_nodes[i][0] = cell_coords[i][0];
            geom_nodes[i][1] = cell_coords[i][1];
            geom_nodes[i][2] = cell_coords[i][2];
        }

        geometry::MappingRequest req;
        req.element_type = cell_type;
        req.geometry_order = mesh.getCellGeometryOrder(cell_id);
        req.use_affine = req.geometry_order <= 1;
        auto mapping = geometry::MappingFactory::create(req, geom_nodes);
        FE_CHECK_NOT_NULL(mapping.get(), "StrongDirichletConstraint: mapping");

        const auto& dof_nodes = restriction.dof_nodes();
        if (local_face_dofs.empty()) return;
        for (const int ldof : local_face_dofs) {
            FE_THROW_IF(ldof < 0, InvalidArgumentException,
                        "StrongDirichletConstraint: negative local DOF index");
            const auto ldof_u = static_cast<std::size_t>(ldof);
            FE_THROW_IF(ldof_u >= dofs_per_component, InvalidArgumentException,
                        "StrongDirichletConstraint: local DOF index out of range for cell DOF list");
            FE_THROW_IF(ldof_u >= dof_nodes.size(), InvalidArgumentException,
                        "StrongDirichletConstraint: local DOF index out of range for dof_nodes");
        }
        if (interpolate_bernstein) {
            const TraceKey key{cell_type, basis.basis_type(), space.polynomial_order(), face_local};
            if (basis.basis_type() == BasisType::Bernstein && space.polynomial_order() > 1 &&
                !out.interpolation.contains(key)) {
                const auto n = local_face_dofs.size();
                std::vector<Real> matrix(n * n), values;
                const basis::BernsteinBasis cell_basis(cell_type, space.polynomial_order());
                for (std::size_t i = 0; i < n; ++i) {
                    cell_basis.evaluate_values(dof_nodes.at(local_face_dofs[i]), values);
                    for (std::size_t j = 0; j < n; ++j)
                        matrix[i * n + j] = values.at(local_face_dofs[j]);
                }
                out.interpolation.emplace(key, math::factor_dense_matrix(std::move(matrix), n, "Bernstein boundary trace"));
            }
            for (int c = component < 0 ? 0 : component; c < (component < 0 ? n_components : component + 1); ++c) {
                BernsteinTrace trace{key, {}};
                for (int local : local_face_dofs)
                    trace.dofs.push_back(offset + cell_dofs[c * dofs_per_component + static_cast<std::size_t>(local)]);
                out.bernstein_faces.push_back(std::move(trace));
            }
        }

        for (int ldof : local_face_dofs) {
            const auto ldof_u = static_cast<std::size_t>(ldof);

            const auto x = mapping->map_to_physical(dof_nodes[ldof_u]);

            const int comp_begin = (component < 0) ? 0 : component;
            const int comp_end = (component < 0) ? n_components : (component + 1);
            for (int comp = comp_begin; comp < comp_end; ++comp) {
                const auto local = static_cast<std::size_t>(comp) * dofs_per_component + ldof_u;
                FE_THROW_IF(local >= cell_dofs.size(), InvalidStateException,
                            "StrongDirichletConstraint: component-local DOF index out of range");

                const GlobalIndex dof = cell_dofs[local] + offset;
                if (coord_by_dof.find(dof) != coord_by_dof.end()) {
                    continue;
                }
                coord_by_dof.emplace(dof, std::array<Real, 3>{x[0], x[1], x[2]});
                if (sampled.empty()) {
                    continue;
                }
                auto& values = samples_by_dof[dof];
                values.reserve(n_samples);
                for (const auto& f : sampled) {
                    const auto sampled_dofs = f.dof_map->getCellDofs(cell_id);
                    const auto sampled_per_component =
                        sampled_dofs.size() / static_cast<std::size_t>(f.components);
                    FE_THROW_IF(sampled_per_component != dofs_per_component, InvalidStateException,
                                "StrongDirichletConstraint: field '" + f.name +
                                    "' has another number of DOFs per component on a cell of the constrained field");
                    // A coefficient at a Bernstein control point is generally different from
                    // the value of the prescribed field there. Sample its actual scalar basis.
                    const auto& source = f.space->element_space(cell_id);
                    const auto& source_basis = source.element().basis();
                    std::vector<Real> weights;
                    if (!f.coefficients.empty() && (interpolate_bernstein || source_basis.basis_type() == BasisType::Bernstein
                                                     || source_basis.basis_type() != basis.basis_type())) {
                        if (to_mesh_family(source.element_type()) == to_mesh_family(cell_type)) {
                            source_basis.evaluate_values(dof_nodes[ldof_u], weights);
                        } else {
                            auto& element = mixed_sources[{&source, cell_type}];
                            if (!element) {
                                elements::ElementRequest request;
                                request.element_type = cell_type;
                                request.basis_type = source_basis.basis_type();
                                request.order = source.polynomial_order();
                                element = elements::ElementFactory::create(request);
                            }
                            element->basis().evaluate_values(dof_nodes[ldof_u], weights);
                        }
                        FE_THROW_IF(weights.size() != sampled_per_component, InvalidStateException,
                                    "StrongDirichletConstraint: prescribed basis differs from its cell coefficient layout");
                    }
                    for (int c = 0; c < f.components; ++c) {
                        const auto begin = static_cast<std::size_t>(c) * sampled_per_component;
                        Real value = 0;
                        if (!f.coefficients.empty()) {
                            if (weights.empty()) value = f.coefficients[sampled_dofs[begin + ldof_u]];
                            else for (std::size_t j = 0; j < weights.size(); ++j)
                                value += weights[j] * f.coefficients[sampled_dofs[begin + j]];
                        }
                        values.push_back(value);
                    }
                }
            }
        }
    };
    prepareTrace(system, interpolate_bernstein, "boundary trace interpolation geometry", [&] {
        mesh.forEachBoundaryFace(boundary_marker, collect_face);
    });

    out.dofs.reserve(coord_by_dof.size());
    for (const auto& [dof, _] : coord_by_dof) {
        out.dofs.push_back(dof);
    }
    std::sort(out.dofs.begin(), out.dofs.end());
    out.dofs.erase(std::unique(out.dofs.begin(), out.dofs.end()), out.dofs.end());

    out.coords.reserve(out.dofs.size());
    out.samples.reserve(out.dofs.size() * n_samples);
    for (const auto dof : out.dofs) {
        auto it = coord_by_dof.find(dof);
        FE_THROW_IF(it == coord_by_dof.end(), InvalidStateException,
                    "StrongDirichletConstraint: internal DOF->coord map missing key");
        out.coords.push_back(it->second);
        if (!sampled.empty()) {
            const auto& values = samples_by_dof.at(dof);
            out.samples.insert(out.samples.end(), values.begin(), values.end());
        }
    }

    // Owned DOFs on faces that only other ranks own (as cells) come from
    // those ranks; without this the constrained set depends on the partition.
    completeOwnedDofsWithCoords(
        system, out,
        "constraint=StrongDirichlet field='" + rec.name + "' marker=" + std::to_string(boundary_marker) +
            " component=" + std::to_string(component));

    return out;
}

} // namespace

StrongDirichletConstraint::StrongDirichletConstraint(FieldId field,
                                                     int boundary_marker,
                                                     forms::FormExpr value,
                                                     int component,
                                                     std::vector<Yield> yields,
                                                     int active_boundary_marker)
    : yields_(std::move(yields)),
      field_(field),
      boundary_marker_(boundary_marker),
      component_(component),
      active_boundary_marker_(active_boundary_marker),
      value_(std::move(value))
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("StrongDirichletConstraint: invalid FieldId");
    }
    if (component_ < -1) {
        throw std::invalid_argument("StrongDirichletConstraint: component must be >= -1");
    }
    if (!value_.isValid()) {
        throw std::invalid_argument("StrongDirichletConstraint: invalid value expression");
    }
    gatherStrongValueInputs(*value_.node(), parameter_names_, sampled_fields_, reads_normal_);
    point_value_ = value_;
    if (!parameter_names_.empty()) {
        point_value_ = value_.transformNodes([&](const forms::FormExprNode& node) -> std::optional<forms::FormExpr> {
            if (node.type() != forms::FormExprType::ParameterSymbol) {
                return std::nullopt;
            }
            const auto name = node.symbolName().value_or(std::string_view{});
            const auto slot = std::find(parameter_names_.begin(), parameter_names_.end(), name) -
                              parameter_names_.begin();
            return forms::FormExpr::parameterRef(static_cast<std::uint32_t>(slot));
        });
    }
    is_time_dependent_ = forms::isTimeDependent(value_) || !parameter_names_.empty() || !sampled_fields_.empty();
}

std::unordered_map<GlobalIndex, std::optional<std::array<Real, 3>>>
StrongDirichletConstraint::pointNormals(const FESystem& system) const
{
    if (!reads_normal_) {
        return {};
    }
    return boundaryPointNormals(system, field_, std::vector<int>{boundary_marker_});
}

std::vector<Real> StrongDirichletConstraint::parameterValues(const FESystem& system, double time) const
{
    std::vector<Real> values;
    values.reserve(parameter_names_.size());
    for (const auto& name : parameter_names_) {
        const auto* f = systems::findTimeFunction(system, name);
        FE_THROW_IF(f == nullptr, InvalidArgumentException,
                    "StrongDirichletConstraint: parameter '" + name +
                        "' of a strong value is not a time function of the system (systems::timeFunction)");
        values.push_back((*f)(static_cast<Real>(time)));
    }
    return values;
}

void StrongDirichletConstraint::apply(const FESystem& system, AffineConstraints& constraints)
{
    assignValues(system, constraints, 0, 0, false);
}

bool StrongDirichletConstraint::updateValues(const FESystem& system, AffineConstraints& constraints,
                                            double time, double dt)
{
    assignValues(system, constraints, time, dt, true);
    return true;
}

void StrongDirichletConstraint::assignValues(const FESystem& system, AffineConstraints& constraints,
                                            double time, double dt, bool updating)
{
    if (value_.hasTest() || value_.hasTrial())
        throw std::invalid_argument("StrongDirichletConstraint: value expression must not contain test/trial functions");
    const auto active_faces = activeBoundaryFaces(system, boundary_marker_, active_boundary_marker_);
    const auto& space = detail::scalarBoundarySpace(*system.fieldRecord(field_).space);
    const auto* adaptive = dynamic_cast<const spaces::AdaptiveSpace*>(&space);
    const bool has_bernstein = adaptive ? adaptive->has_basis_type(BasisType::Bernstein)
                                        : space.element().basis().basis_type() == BasisType::Bernstein;
    const bool bernstein = has_bernstein && !point_value_.node()->constantValue().has_value();
    auto extracted = boundaryDofsWithCoordsByMarker(system, field_, boundary_marker_, component_, sampled_fields_,
                                                    updating, active_faces ? &*active_faces : nullptr, bernstein);
    if (!bernstein) dropYieldedPoints(system, extracted.dofs, extracted.coords, extracted.samples);
    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    const auto parameters = parameterValues(system, time);
    const auto normals = pointNormals(system);
    forms::PointEvalContext pctx;
    pctx.time = time;
    pctx.dt = dt;
    pctx.jit_constants = parameters;
    std::vector<FieldValueEntry> field_values(sampled_fields_.size());
    const auto n_samples = std::accumulate(extracted.sample_components.begin(), extracted.sample_components.end(), std::size_t{});
    std::vector<Real> values(extracted.dofs.size());
    const auto evaluate = [&] {
        for (std::size_t i = 0; i < extracted.dofs.size(); ++i) {
            const auto dof = extracted.dofs[i];
            if (!owned.contains(dof)) continue;
            pctx.x = extracted.coords[i];
            if (!field_values.empty()) {
                fillFieldValues(sampled_fields_, extracted.sample_components, extracted.samples, i * n_samples, field_values);
                pctx.field_values = field_values;
            }
            if (reads_normal_) pctx.normal = boundaryNormalAt(normals, dof, pctx.x, boundary_marker_);
            values[i] = forms::evaluateScalarAt(point_value_, pctx);
        }
    };
    prepareTrace(system, bernstein, "Bernstein boundary point values", evaluate);
    if (bernstein) {
        // A face can use samples owned by another rank, including a yielded
        // endpoint needed to interpolate the retained interior controls.
        copyOwnedBoundaryPointValues(system, extracted.dofs, values);
        std::map<GlobalIndex, Real> coefficients;
        prepareTrace(system, true, "Bernstein boundary coefficients", [&] {
            for (const auto& face : extracted.bernstein_faces) {
                std::vector<Real> local;
                local.reserve(face.dofs.size());
                for (const auto dof : face.dofs) {
                    const auto i = std::lower_bound(extracted.dofs.begin(), extracted.dofs.end(), dof) - extracted.dofs.begin();
                    local.push_back(values.at(i));
                }
                const auto factor = extracted.interpolation.find(face.key);
                if (factor != extracted.interpolation.end() &&
                    !std::all_of(local.begin() + 1, local.end(), [&](Real value) { return value == local.front(); })) {
                    factor->second.solve_in_place(local);
                }
                for (std::size_t i = 0; i < face.dofs.size(); ++i) coefficients.try_emplace(face.dofs[i], local[i]);
            }
        });
        std::vector<GlobalIndex> dofs;
        values.clear();
        for (const auto& [dof, value] : coefficients) {
            dofs.push_back(dof);
            values.push_back(value);
        }
        completeOwnedBoundaryDofs(system, dofs, values, 1u, {});
        FE_THROW_IF(dofs != extracted.dofs, InvalidStateException, "StrongDirichletConstraint: incomplete Bernstein trace");
        dropYieldedPoints(system, extracted.dofs, extracted.coords, values);
    }
    FE_THROW_IF(updating && dofs_ != extracted.dofs, InvalidStateException,
                "StrongDirichletConstraint::updateValues: geometry-dependent update changed constrained DOFs; rebuild constraints instead");
    dofs_ = std::move(extracted.dofs);

    std::size_t count = 0, missing = 0;
    Real min_value = std::numeric_limits<Real>::infinity(), max_value = -min_value;
    std::ostringstream sample;
    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        const auto dof = dofs_[i];
        if (!owned.contains(dof)) continue;
        const Real v = values[i];
        const bool constrained = !updating || constraints.isConstrained(dof);
        if (updating) {
            if (!constrained) ++missing;
            constraints.updateInhomogeneity(dof, v);
        } else {
            constraints.addDirichlet(dof, v);
        }
        ++count;
        min_value = std::min(min_value, v);
        max_value = std::max(max_value, v);
        if (strongDirichletTraceEnabled() && count <= (updating ? 6 : 4)) {
            if (count > 1) sample << "; ";
            const auto& x = extracted.coords[i];
            sample << "{dof=" << dof;
            if (updating) sample << ",owned=1,constrained=" << (constrained ? 1 : 0);
            sample << ",x=(" << x[0] << "," << x[1] << "," << x[2] << "),v=" << v << "}";
        }
    }
    traceStrongDirichletEvent(updating ? "update" : "apply", boundary_marker_, component_, dofs_.size(), count,
                              count - missing, missing, time, dt, count ? min_value : 0, count ? max_value : 0, sample.str());
}

ConstraintDependencyDeclaration StrongDirichletConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = ISystemConstraint::dependencyDeclaration();
    merge_into(out.value, ConstraintDependencyMask::meshGeometry());
    if (is_time_dependent_) {
        out.value.time = true;
    }
    if (active_boundary_marker_ >= 0) {
        // The active faces come from the installed cut-integration context; a
        // new context rebuilds the constraint state (as for the constraints of
        // the active side), so the constrained set follows the cut.
        out.structural.fe_constraint_layout = true;
        out.structural.mesh_field_layout = true;
        out.structural.active_configuration = true;
    }
    return out;
}

systems::SetupStorageRequirements StrongDirichletConstraint::storageRequirements() const noexcept
{
    systems::SetupStorageRequirements req;
    req.entity_dof_map = true;
    req.boundary_face_topology = true;
    return req;
}

void StrongDirichletConstraint::dropYieldedPoints(const FESystem& system,
                                                  std::vector<GlobalIndex>& dofs,
                                                  std::vector<std::array<Real, 3>>& coords,
                                                  std::vector<Real>& samples) const
{
    if (yields_.empty()) {
        return;
    }
    std::unordered_set<GlobalIndex> held;
    for (const auto& yield : yields_) {
        const auto active_faces = activeBoundaryFaces(system, yield.boundary_marker, yield.active_boundary_marker);
        for (const auto dof : boundaryDofsWithCoordsByMarker(system, field_, yield.boundary_marker, yield.component,
                                                             {}, /*require_coefficients=*/false,
                                                             active_faces ? &*active_faces : nullptr)
                                  .dofs) {
            held.insert(dof);
        }
    }
    const std::size_t n_samples = dofs.empty() ? 0 : samples.size() / dofs.size();
    std::size_t kept = 0;
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        if (held.contains(dofs[i])) {
            continue;
        }
        dofs[kept] = dofs[i];
        coords[kept] = coords[i];
        std::copy_n(samples.begin() + static_cast<std::ptrdiff_t>(i * n_samples), n_samples,
                    samples.begin() + static_cast<std::ptrdiff_t>(kept * n_samples));
        ++kept;
    }
    dofs.resize(kept);
    coords.resize(kept);
    samples.resize(kept * n_samples);
}

std::string StrongDirichletConstraint::manifestDescription() const
{
    return manifestDescription("active");
}

std::string StrongDirichletConstraint::manifestDescription(std::string_view active_subset_label) const
{
    std::ostringstream oss;
    oss << "field=" << field_ << " marker=" << boundary_marker_;
    if (active_boundary_marker_ >= 0) {
        oss << " " << active_subset_label << "=" << active_boundary_marker_;
    }
    oss << " component=" << component_
        << " time_dependent=" << (is_time_dependent_ ? 1 : 0)
        << " value=" << (value_.isValid() ? value_.toString() : std::string("<none>"));
    if (!yields_.empty()) {
        oss << " yields_to=";
        for (std::size_t i = 0; i < yields_.size(); ++i) {
            oss << (i == 0u ? "" : ",") << yields_[i].boundary_marker << ":" << yields_[i].component;
        }
    }
    return oss.str();
}

} // namespace constraints
} // namespace FE
} // namespace svmp
