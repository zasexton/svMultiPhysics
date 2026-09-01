#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVelocityExtensionConstraint.h"

#include "Basis/NodeOrderingConventions.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Forms/Vocabulary.h"
#include "Interfaces/MaterialInterfaceTransportVelocity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

struct CollectiveContext {
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
    bool active{false};
};

[[nodiscard]] CollectiveContext collectiveContext(
    const dofs::DofHandler& dof_handler)
{
    CollectiveContext context;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0 &&
        dof_handler.mpiComm() != MPI_COMM_NULL) {
        int size = 1;
        MPI_Comm_size(dof_handler.mpiComm(), &size);
        context.communicator = dof_handler.mpiComm();
        context.active = size > 1;
    }
#else
    (void)dof_handler;
#endif
    return context;
}

#if FE_HAS_MPI
[[nodiscard]] MPI_Datatype mpiRealType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    }
    return MPI_DOUBLE;
}
#endif

[[nodiscard]] Real allReduceRealMin(const CollectiveContext& context,
                                    Real local)
{
#if FE_HAS_MPI
    if (context.active) {
        Real global = local;
        MPI_Allreduce(&local, &global, 1, mpiRealType(), MPI_MIN,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] Real allReduceRealMax(const CollectiveContext& context,
                                    Real local)
{
#if FE_HAS_MPI
    if (context.active) {
        Real global = local;
        MPI_Allreduce(&local, &global, 1, mpiRealType(), MPI_MAX,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] std::uint64_t allReduceUnsignedSum(
    const CollectiveContext& context,
    std::uint64_t local)
{
#if FE_HAS_MPI
    if (context.active) {
        std::uint64_t global = local;
#ifdef MPI_UINT64_T
        const MPI_Datatype datatype = MPI_UINT64_T;
#else
        const MPI_Datatype datatype = MPI_UNSIGNED_LONG_LONG;
#endif
        MPI_Allreduce(&local, &global, 1, datatype, MPI_SUM,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] int allReduceIntMin(
    const CollectiveContext& context,
    int local)
{
#if FE_HAS_MPI
    if (context.active) {
        int global = local;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

void allReduceRealVector(const CollectiveContext& context,
                         std::vector<Real>& values,
                         bool take_minimum)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        constexpr std::size_t max_chunk =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        std::vector<Real> reduced(values.size(), Real{0.0});
        for (std::size_t offset = 0; offset < values.size();) {
            const auto count = std::min(max_chunk, values.size() - offset);
            MPI_Allreduce(values.data() + static_cast<std::ptrdiff_t>(offset),
                          reduced.data() + static_cast<std::ptrdiff_t>(offset),
                          static_cast<int>(count),
                          mpiRealType(),
                          take_minimum ? MPI_MIN : MPI_MAX,
                          context.communicator);
            offset += count;
        }
        values.swap(reduced);
    }
#else
    (void)context;
    (void)values;
    (void)take_minimum;
#endif
}

[[nodiscard]] Real vectorNorm(const std::array<Real, 3>& value,
                              int dimension) noexcept
{
    Real squared = 0.0;
    for (int d = 0; d < dimension; ++d) {
        squared += value[static_cast<std::size_t>(d)] *
                   value[static_cast<std::size_t>(d)];
    }
    return std::sqrt(squared);
}

[[nodiscard]] Real distance(const std::array<Real, 3>& a,
                            const std::array<Real, 3>& b,
                            int dimension) noexcept
{
    Real squared = 0.0;
    for (int d = 0; d < dimension; ++d) {
        const Real delta = a[static_cast<std::size_t>(d)] -
                           b[static_cast<std::size_t>(d)];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

[[nodiscard]] std::array<Real, 3> normalizedBoundaryNormal(
    const std::vector<std::array<Real, 3>>& coordinates,
    std::span<const LocalIndex> face_nodes,
    int dimension)
{
    if ((dimension == 2 && face_nodes.size() < 2u) ||
        (dimension == 3 && face_nodes.size() < 3u)) {
        throw std::invalid_argument(
            "evaluateLevelSetTransportSafety: boundary face has insufficient geometry nodes");
    }
    std::array<Real, 3> cell_center{0.0, 0.0, 0.0};
    for (const auto& point : coordinates) {
        for (int d = 0; d < dimension; ++d) {
            cell_center[static_cast<std::size_t>(d)] +=
                point[static_cast<std::size_t>(d)];
        }
    }
    if (coordinates.empty()) {
        throw std::invalid_argument(
            "evaluateLevelSetTransportSafety: boundary parent has no coordinates");
    }
    for (int d = 0; d < dimension; ++d) {
        cell_center[static_cast<std::size_t>(d)] /=
            static_cast<Real>(coordinates.size());
    }

    std::array<Real, 3> face_center{0.0, 0.0, 0.0};
    for (const auto local : face_nodes) {
        const auto index = static_cast<std::size_t>(local);
        if (index >= coordinates.size()) {
            throw std::invalid_argument(
                "evaluateLevelSetTransportSafety: boundary face node is outside parent coordinates");
        }
        for (int d = 0; d < dimension; ++d) {
            face_center[static_cast<std::size_t>(d)] +=
                coordinates[index][static_cast<std::size_t>(d)];
        }
    }
    for (int d = 0; d < dimension; ++d) {
        face_center[static_cast<std::size_t>(d)] /=
            static_cast<Real>(face_nodes.size());
    }

    std::array<Real, 3> normal{0.0, 0.0, 0.0};
    const auto& a = coordinates[static_cast<std::size_t>(face_nodes[0])];
    const auto& b = coordinates[static_cast<std::size_t>(face_nodes[1])];
    if (dimension == 2) {
        normal = {b[1] - a[1], a[0] - b[0], 0.0};
    } else {
        const auto& c = coordinates[static_cast<std::size_t>(face_nodes[2])];
        const std::array<Real, 3> ab{
            b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const std::array<Real, 3> ac{
            c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        normal = {ab[1] * ac[2] - ab[2] * ac[1],
                  ab[2] * ac[0] - ab[0] * ac[2],
                  ab[0] * ac[1] - ab[1] * ac[0]};
    }
    const Real magnitude = vectorNorm(normal, dimension);
    if (!(magnitude > Real{0.0}) || !std::isfinite(magnitude)) {
        throw std::invalid_argument(
            "evaluateLevelSetTransportSafety: degenerate boundary normal");
    }
    for (int d = 0; d < dimension; ++d) {
        normal[static_cast<std::size_t>(d)] /= magnitude;
    }
    Real outward_dot = 0.0;
    for (int d = 0; d < dimension; ++d) {
        outward_dot += normal[static_cast<std::size_t>(d)] *
                       (face_center[static_cast<std::size_t>(d)] -
                        cell_center[static_cast<std::size_t>(d)]);
    }
    if (outward_dot < Real{0.0}) {
        for (int d = 0; d < dimension; ++d) {
            normal[static_cast<std::size_t>(d)] *= Real{-1.0};
        }
    }
    return normal;
}

[[nodiscard]] spaces::FunctionSpace::Value referenceCentroid(
    ElementType type,
    std::span<const LocalIndex> local_nodes = {})
{
    spaces::FunctionSpace::Value point{};
    const auto count = basis::ReferenceNodeLayout::num_nodes(type);
    if (count == 0u) {
        throw std::invalid_argument(
            "evaluateLevelSetTransportSafety: element has no reference nodes");
    }
    if (local_nodes.empty()) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto node = basis::ReferenceNodeLayout::get_node_coords(type, i);
            point[0] += node[0];
            point[1] += node[1];
            point[2] += node[2];
        }
        const Real inverse = Real{1.0} / static_cast<Real>(count);
        point[0] *= inverse;
        point[1] *= inverse;
        point[2] *= inverse;
        return point;
    }
    for (const auto local : local_nodes) {
        const auto index = static_cast<std::size_t>(local);
        if (index >= count) {
            throw std::invalid_argument(
                "evaluateLevelSetTransportSafety: face reference node is outside the parent element");
        }
        const auto node = basis::ReferenceNodeLayout::get_node_coords(type, index);
        point[0] += node[0];
        point[1] += node[1];
        point[2] += node[2];
    }
    const Real inverse = Real{1.0} / static_cast<Real>(local_nodes.size());
    point[0] *= inverse;
    point[1] *= inverse;
    point[2] *= inverse;
    return point;
}

class VelocitySampler {
public:
    VelocitySampler(const systems::FESystem& system,
                    const LevelSetVelocityOptions& options,
                    const systems::SystemStateView& state)
        : system_(system), options_(options), state_(state)
    {
        dimension_ = system_.meshAccess().dimension();
        if (dimension_ < 1 || dimension_ > 3) {
            throw std::invalid_argument(
                "evaluateLevelSetTransportSafety: mesh dimension must be in [1, 3]");
        }
        if (options_.source ==
            LevelSetVelocitySource::MaterialInterfacePhasePair) {
            throw std::invalid_argument(
                "evaluateLevelSetTransportSafety: material-interface phase-pair velocity is reserved for conservative transport and has no legacy bound-preserving safety route");
        }
        if (options_.source == LevelSetVelocitySource::ConstantVector) {
            return;
        }
        field_ = system_.findFieldByName(options_.field_name);
        if (field_ == INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "evaluateLevelSetTransportSafety: missing velocity field '" +
                options_.field_name + "'");
        }
        const auto& record = system_.fieldRecord(field_);
        if (!record.space || record.components < dimension_) {
            throw std::invalid_argument(
                "evaluateLevelSetTransportSafety: velocity field has insufficient components");
        }
        space_ = record.space.get();
        prescribed_ = record.source_kind == systems::FieldSourceKind::PrescribedData;
        if (prescribed_) {
            prescribed_coefficients_ = system_.prescribedFieldCoefficients(field_);
            if (prescribed_coefficients_.empty()) {
                throw std::invalid_argument(
                    "evaluateLevelSetTransportSafety: prescribed velocity field has no synchronized coefficients");
            }
        } else {
            offset_ = system_.fieldDofOffset(field_);
            if (offset_ < 0) {
                throw std::invalid_argument(
                    "evaluateLevelSetTransportSafety: velocity field has an invalid system offset");
            }
        }
    }

    [[nodiscard]] std::array<Real, 3> sample(
        GlobalIndex cell,
        const spaces::FunctionSpace::Value& point) const
    {
        if (options_.source == LevelSetVelocitySource::ConstantVector) {
            return options_.constant_value;
        }
        const auto cell_dofs = system_.fieldDofHandler(field_).getCellDofs(cell);
        std::vector<Real> coefficients;
        coefficients.reserve(cell_dofs.size());
        for (const auto dof : cell_dofs) {
            if (dof < 0) {
                throw std::invalid_argument(
                    "evaluateLevelSetTransportSafety: velocity cell has a negative DOF");
            }
            const auto index = static_cast<std::size_t>(
                prescribed_ ? dof : dof + offset_);
            const auto source = prescribed_ ? prescribed_coefficients_ : state_.u;
            if (index >= source.size()) {
                throw std::invalid_argument(
                    "evaluateLevelSetTransportSafety: velocity coefficient source is too small");
            }
            coefficients.push_back(source[index]);
        }
        const auto value = space_->evaluate(point, coefficients);
        return {value[0], value[1], value[2]};
    }

private:
    const systems::FESystem& system_;
    const LevelSetVelocityOptions& options_;
    const systems::SystemStateView& state_;
    int dimension_{0};
    FieldId field_{INVALID_FIELD_ID};
    const spaces::FunctionSpace* space_{nullptr};
    bool prescribed_{false};
    GlobalIndex offset_{0};
    std::span<const Real> prescribed_coefficients_{};
};

[[nodiscard]] FieldId resolveNamedField(
    const systems::FESystem& system,
    const std::string& field_name,
    const char* context)
{
    const auto field = system.findFieldByName(field_name);
    if (field == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            std::string("installLevelSetTransport: missing ") +
            context + " field '" + field_name + "'");
    }
    return field;
}

[[nodiscard]] const interfaces::
    MaterialInterfaceTransportVelocityDeclaration&
requireMaterialInterfaceVelocityDeclaration(
    const systems::FESystem& system,
    FieldId level_set_field,
    int interface_marker)
{
    const auto declarations =
        system.materialInterfaceTransportVelocityDeclarations();
    const interfaces::MaterialInterfaceTransportVelocityDeclaration*
        found = nullptr;
    for (const auto& declaration : declarations) {
        if (declaration.level_set_field != level_set_field ||
            declaration.interface_marker != interface_marker) {
            continue;
        }
        if (found != nullptr) {
            throw std::invalid_argument(
                "installLevelSetTransport: material-interface phase-pair velocity has more than one matching owner");
        }
        found = &declaration;
    }
    if (found == nullptr) {
        throw std::invalid_argument(
            "installLevelSetTransport: material-interface phase-pair velocity has no matching declaration");
    }
    return *found;
}

[[nodiscard]] systems::FieldSourceKind sourceKind(LevelSetFieldSource source) noexcept
{
    switch (source) {
    case LevelSetFieldSource::Unknown:
        return systems::FieldSourceKind::Unknown;
    case LevelSetFieldSource::PrescribedData:
        // The source describes initialization; transport still owns the field.
        return systems::FieldSourceKind::Unknown;
    }
    return systems::FieldSourceKind::Unknown;
}

[[nodiscard]] systems::FieldSourceKind sourceKind(LevelSetVelocitySource source) noexcept
{
    switch (source) {
    case LevelSetVelocitySource::CoupledField:
        return systems::FieldSourceKind::Unknown;
    case LevelSetVelocitySource::PrescribedData:
    case LevelSetVelocitySource::ConstantVector:
        return systems::FieldSourceKind::PrescribedData;
    case LevelSetVelocitySource::MaterialInterfacePhasePair:
        return systems::FieldSourceKind::Unknown;
    }
    return systems::FieldSourceKind::Unknown;
}

struct PendingTransportField {
    FieldId existing_id{INVALID_FIELD_ID};
    std::shared_ptr<const spaces::FunctionSpace> space{};
    int components{0};
    systems::FieldSourceKind source_kind{systems::FieldSourceKind::Unknown};

    [[nodiscard]] bool needsRegistration() const noexcept
    {
        return existing_id == INVALID_FIELD_ID;
    }
};

[[nodiscard]] PendingTransportField preflightLevelSetField(
    const systems::FESystem& system,
    const LevelSetFieldOptions& options,
    const std::shared_ptr<const spaces::FunctionSpace>& requested_space)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != INVALID_FIELD_ID) {
        const auto& record = system.fieldRecord(existing);
        if (record.components != 1 || !record.space ||
            record.space->value_dimension() != 1) {
            throw std::invalid_argument(
                "installLevelSetTransport: level-set field '" +
                options.field_name + "' must be scalar");
        }
        if (!system.fieldParticipatesInUnknownVector(existing)) {
            throw std::invalid_argument(
                "installLevelSetTransport: level-set field must be an unknown for transport residual assembly");
        }
        return PendingTransportField{
            .existing_id = existing,
            .space = record.space,
            .components = record.components,
            .source_kind = record.source_kind,
        };
    }

    if (!options.auto_register_field) {
        (void)resolveNamedField(system, options.field_name, "level-set");
    }
    if (!requested_space) {
        throw std::invalid_argument(
            "installLevelSetTransport: auto-registering the level-set field requires a function space");
    }
    if (requested_space->value_dimension() != 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field space must be scalar");
    }
    return PendingTransportField{
        .space = requested_space,
        .components = 1,
        .source_kind = sourceKind(options.source),
    };
}

[[nodiscard]] PendingTransportField preflightVelocityField(
    const systems::FESystem& system,
    const LevelSetVelocityOptions& options,
    int expected_dimension)
{
    const auto existing = system.findFieldByName(options.field_name);
    PendingTransportField pending;
    if (existing != INVALID_FIELD_ID) {
        const auto& record = system.fieldRecord(existing);
        pending = PendingTransportField{
            .existing_id = existing,
            .space = record.space,
            .components = record.components,
            .source_kind = record.source_kind,
        };
    } else {
        if (!options.auto_register_field) {
            (void)resolveNamedField(system, options.field_name, "velocity");
        }
        if (!options.space) {
            throw std::invalid_argument(
                "installLevelSetTransport: auto-registering the velocity field requires a function space");
        }
        pending.space = options.space;
        pending.components = options.space->value_dimension();
        pending.source_kind = sourceKind(options.source);
    }

    if (!pending.space || pending.components != expected_dimension ||
        pending.space->value_dimension() != expected_dimension) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity field '" +
            options.field_name + "' must have exactly " +
            std::to_string(expected_dimension) + " component(s)");
    }
    if (pending.space->topological_dimension() != expected_dimension) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity and level-set dimensions differ");
    }
    if (existing != INVALID_FIELD_ID) {
        const bool is_unknown =
            system.fieldParticipatesInUnknownVector(existing);
        if (options.source == LevelSetVelocitySource::CoupledField &&
            !is_unknown) {
            throw std::invalid_argument(
                "installLevelSetTransport: coupled velocity source must be an unknown field");
        }
        if (options.source == LevelSetVelocitySource::PrescribedData &&
            is_unknown) {
            throw std::invalid_argument(
                "installLevelSetTransport: prescribed velocity source must not be an unknown field");
        }
    }
    return pending;
}

[[nodiscard]] FieldId ensureLevelSetField(
    systems::FESystem& system,
    const LevelSetFieldOptions& options,
    std::shared_ptr<const spaces::FunctionSpace> space)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != INVALID_FIELD_ID) {
        return existing;
    }
    if (!options.auto_register_field) {
        return resolveNamedField(system, options.field_name, "level-set");
    }
    if (!space) {
        throw std::invalid_argument(
            "installLevelSetTransport: auto-registering the level-set field requires a function space");
    }
    return system.addField(systems::FieldSpec{
        .name = options.field_name,
        .space = std::move(space),
        .components = 1,
        .source_kind = sourceKind(options.source),
    });
}

[[nodiscard]] FieldId ensureVelocityField(
    systems::FESystem& system,
    const LevelSetVelocityOptions& options)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != INVALID_FIELD_ID) {
        return existing;
    }
    if (!options.auto_register_field) {
        return resolveNamedField(system, options.field_name, "velocity");
    }
    if (!options.space) {
        throw std::invalid_argument(
            "installLevelSetTransport: auto-registering the velocity field requires a function space");
    }
    return system.addField(systems::FieldSpec{
        .name = options.field_name,
        .space = options.space,
        .components = options.space->value_dimension(),
        .source_kind = sourceKind(options.source),
    });
}

void validateVelocityField(const systems::FESystem& system,
                           FieldId field,
                           const std::string& field_name)
{
    const auto& rec = system.fieldRecord(field);
    if (!rec.space || rec.space->value_dimension() < 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity field '" +
            field_name + "' must have a vector function space");
    }
}

void validateBoundaryOptions(const LevelSetBoundaryOptions& boundaries)
{
    std::unordered_set<int> markers;
    markers.reserve(boundaries.inflow.size() + boundaries.outflow.size());

    for (const auto& bc : boundaries.inflow) {
        const int marker = forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "installLevelSetTransport: inflow boundary");
        if (!markers.insert(marker).second) {
            throw std::invalid_argument(
                "installLevelSetTransport: duplicate level-set boundary marker");
        }
        if (!(bc.penalty_scale > 0.0)) {
            throw std::invalid_argument(
                "installLevelSetTransport: inflow boundary penalty_scale must be positive");
        }
    }

    for (const auto& bc : boundaries.outflow) {
        const int marker = forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "installLevelSetTransport: outflow boundary");
        if (!markers.insert(marker).second) {
            throw std::invalid_argument(
                "installLevelSetTransport: duplicate level-set boundary marker");
        }
    }
}

[[nodiscard]] bool isFixedP1ScalarH1(
    const PendingTransportField& field) noexcept
{
    return field.components == 1 && field.space &&
           field.space->space_type() == spaces::SpaceType::H1 &&
           field.space->field_type() == FieldType::Scalar &&
           field.space->continuity() == Continuity::C0 &&
           field.space->value_dimension() == 1 &&
           !field.space->is_variable_order() &&
           field.space->polynomial_order() == 1;
}

void validateConservativePhaseOptions(
    const LevelSetTransportOptions& options)
{
    const auto& phase = options.conservative_phase;
    if (!phase.enabled) {
        return;
    }
    if (phase.liquid_indicator.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase field name must be non-empty");
    }
    if (phase.liquid_indicator.source != LevelSetFieldSource::Unknown) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase field must be an unknown state");
    }
    if (!std::isfinite(phase.invariant_tolerance) ||
        phase.invariant_tolerance < Real{0.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase invariant tolerance must be finite and nonnegative");
    }
    if (!std::isfinite(phase.component_activity_tolerance) ||
        !(phase.component_activity_tolerance > Real{0.0}) ||
        phase.component_activity_tolerance > Real{1.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase component activity tolerance must be in (0, 1]");
    }
    if (!std::isfinite(phase.maximum_courant) ||
        !(phase.maximum_courant > Real{0.0}) ||
        phase.maximum_courant > Real{1.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase maximum Courant number must be in (0, 1]");
    }
    if (phase.flux_artifact_cadence_steps <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase flux artifact cadence must be positive");
    }
    if (!std::isfinite(phase.momentum_relative_tolerance) ||
        !(phase.momentum_relative_tolerance > Real{0.0})) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase momentum relative tolerance must be positive and finite");
    }
    try {
        static_cast<void>(makeAxisAlignedBoxPhaseRegions(
            phase.fixed_flux_regions,
            std::span<const std::array<Real, 3>>{}));
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            std::string(
                "installLevelSetTransport: invalid conservative phase fixed flux region: ") +
            error.what());
    }
    if (!std::isfinite(phase.impermeable_normal_velocity_tolerance) ||
        phase.impermeable_normal_velocity_tolerance < Real{0.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase wall-normal velocity tolerance must be finite and nonnegative");
    }
    if (phase
            .pointwise_impermeable_velocity_tolerance_explicitly_requested) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase pointwise velocity-normal wall enforcement is unsupported; the available closed-domain contract checks only discrete q-flux at invariant_tolerance and can be blind where q=0");
    }
    if (!std::isfinite(phase.geometry_measure_tolerance) ||
        !(phase.geometry_measure_tolerance > Real{0.0})) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase geometry measure tolerance must be positive and finite");
    }
    if (phase.geometry_correction_max_iterations <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase geometry correction requires positive max iterations");
    }
    if (!std::isfinite(phase.maximum_geometry_displacement_fraction) ||
        !(phase.maximum_geometry_displacement_fraction > Real{0.0}) ||
        phase.maximum_geometry_displacement_fraction > Real{1.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase geometry displacement fraction must be in (0, 1]");
    }
    if (!options.boundaries.inflow.empty() ||
        !options.boundaries.outflow.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase transport currently requires a closed boundary; conservative phase boundary flux data are not configured");
    }
    if (options.bound_preserving.enabled) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase transport is incompatible with the legacy nonconservative level-set limiter");
    }
    if (options.volume_correction.enabled) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase transport owns geometry reconciliation and is incompatible with legacy target-volume correction");
    }
}

void validateBoundPreservingOptions(
    const LevelSetBoundPreservingOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (!(options.bound_tolerance >= Real{0.0}) ||
        !std::isfinite(options.bound_tolerance)) {
        throw std::invalid_argument(
            "installLevelSetTransport: bound-preserving limiter bound_tolerance must be finite and nonnegative");
    }
    if (!(options.sign_tolerance >= Real{0.0}) ||
        !std::isfinite(options.sign_tolerance)) {
        throw std::invalid_argument(
            "installLevelSetTransport: bound-preserving limiter sign_tolerance must be finite and nonnegative");
    }
    if (options.enforce_courant_limit &&
        (!(options.maximum_courant > Real{0.0}) ||
         !std::isfinite(options.maximum_courant))) {
        throw std::invalid_argument(
            "installLevelSetTransport: bound-preserving limiter maximum_courant must be positive and finite");
    }
    if (!(options.courant_tolerance >= Real{0.0}) ||
        !std::isfinite(options.courant_tolerance)) {
        throw std::invalid_argument(
            "installLevelSetTransport: bound-preserving limiter courant_tolerance must be finite and nonnegative");
    }
    if (options.enforce_impermeable_boundaries &&
        (!(options.impermeable_normal_velocity_tolerance >= Real{0.0}) ||
         !std::isfinite(options.impermeable_normal_velocity_tolerance))) {
        throw std::invalid_argument(
            "installLevelSetTransport: impermeable normal-velocity tolerance must be finite and nonnegative");
    }
}

void validateReinitializationOptions(const LevelSetReinitializationOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (options.cadence_steps <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization cadence_steps must be positive");
    }
    if (options.method != LevelSetReinitializationMethod::Projection) {
        throw std::invalid_argument(
            "installLevelSetTransport: runtime reinitialization currently supports Projection only");
    }
    if (options.max_iterations <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization max_iterations must be positive");
    }
    if (!(options.pseudo_time_step_scale > 0.0) ||
        !(options.pseudo_time_step_scale <= 1.0) ||
        !std::isfinite(options.pseudo_time_step_scale)) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization pseudo_time_step_scale must be in (0, 1]");
    }
    if (!(options.interface_band_width > 0.0) ||
        !std::isfinite(options.interface_band_width)) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization interface_band_width must be positive");
    }
    if (!(options.signed_distance_tolerance > 0.0) ||
        !std::isfinite(options.signed_distance_tolerance)) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization signed_distance_tolerance must be positive");
    }
    if (options.preserve_band_width > 0.0) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization preserve_band_width is no longer supported; "
            "the projection method now preserves the zero set while repairing its neighborhood");
    }
    if (!(options.max_zero_set_displacement >= 0.0) ||
        !std::isfinite(options.max_zero_set_displacement)) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization max_zero_set_displacement must be finite and nonnegative");
    }
}

void validateVolumeCorrectionOptions(const LevelSetVolumeCorrectionOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (options.cadence_steps <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction cadence_steps must be positive");
    }
    if (!(options.volume_tolerance > 0.0) ||
        !std::isfinite(options.volume_tolerance)) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction volume_tolerance must be positive");
    }
    if (options.max_iterations <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction max_iterations must be positive");
    }
    if (!(options.minimum_relative_volume_error >= 0.0) ||
        !std::isfinite(options.minimum_relative_volume_error)) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction minimum_relative_volume_error must be finite and nonnegative");
    }
    if (!(options.maximum_interface_displacement_fraction > 0.0) ||
        !(options.maximum_interface_displacement_fraction <= 1.0) ||
        !std::isfinite(options.maximum_interface_displacement_fraction)) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction maximum_interface_displacement_fraction must be finite and in (0, 1]");
    }
    if (!(options.maximum_cumulative_interface_displacement_fraction > 0.0) ||
        !std::isfinite(
            options.maximum_cumulative_interface_displacement_fraction)) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction maximum_cumulative_interface_displacement_fraction must be finite and positive");
    }
    if (!options.use_initial_negative_volume_as_target &&
        options.target_negative_volume < 0.0) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction target_negative_volume must be nonnegative");
    }
}

void validateInterfaceKinematicOptions(const LevelSetInterfaceKinematicOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (options.interface_marker < 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: interface kinematic marker must be nonnegative when enabled");
    }
    if (!(options.weight_scale > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: interface kinematic weight_scale must be positive when enabled");
    }
}

} // namespace

LevelSetTransportSafetyResult evaluateLevelSetTransportSafety(
    const systems::FESystem& system,
    const LevelSetVelocityOptions& velocity,
    const LevelSetBoundaryOptions& boundaries,
    const LevelSetBoundPreservingOptions& options,
    const systems::SystemStateView& state,
    Real dt)
{
    LevelSetTransportSafetyResult result;
    if (!(dt > Real{0.0}) || !std::isfinite(dt)) {
        result.diagnostic = "level-set transport safety requires a positive finite time step";
        return result;
    }
    try {
        validateBoundPreservingOptions(options);
        const int dimension = system.meshAccess().dimension();
        if (dimension < 2 || dimension > 3) {
            result.diagnostic =
                "level-set transport safety currently supports two- and three-dimensional meshes";
            return result;
        }
        const FieldId collective_field =
            system.findFieldByName(velocity.source == LevelSetVelocitySource::ConstantVector
                                       ? std::string{}
                                       : velocity.field_name);
        const auto& collective_handler =
            collective_field != INVALID_FIELD_ID
                ? system.fieldDofHandler(collective_field)
                : system.dofHandler();
        const auto collective = collectiveContext(collective_handler);
        VelocitySampler sampler(system, velocity, state);

        Real local_minimum_length = std::numeric_limits<Real>::infinity();
        Real local_maximum_speed = Real{0.0};
        Real local_maximum_courant = Real{0.0};
        std::uint64_t local_cells = 0u;
        std::vector<std::array<Real, 3>> coordinates;
        system.meshAccess().forEachOwnedCell([&](GlobalIndex cell) {
            coordinates.clear();
            system.meshAccess().getCellCoordinates(cell, coordinates);
            if (coordinates.size() < 2u) {
                throw std::invalid_argument(
                    "level-set transport safety found a cell with insufficient coordinates");
            }
            Real cell_length = std::numeric_limits<Real>::infinity();
            for (std::size_t i = 0; i < coordinates.size(); ++i) {
                for (std::size_t j = i + 1u; j < coordinates.size(); ++j) {
                    const Real candidate = distance(coordinates[i], coordinates[j], dimension);
                    if (candidate > Real{0.0} && std::isfinite(candidate)) {
                        cell_length = std::min(cell_length, candidate);
                    }
                }
            }
            if (!std::isfinite(cell_length) || !(cell_length > Real{0.0})) {
                throw std::invalid_argument(
                    "level-set transport safety found a degenerate cell length");
            }

            const auto type = system.meshAccess().getCellType(cell);
            const auto reference_node_count =
                basis::ReferenceNodeLayout::num_nodes(type);
            Real cell_speed = vectorNorm(
                sampler.sample(cell, referenceCentroid(type)), dimension);
            for (std::size_t local = 0; local < reference_node_count; ++local) {
                spaces::FunctionSpace::Value point{};
                const auto node =
                    basis::ReferenceNodeLayout::get_node_coords(type, local);
                point[0] = node[0];
                point[1] = node[1];
                point[2] = node[2];
                cell_speed = std::max(
                    cell_speed,
                    vectorNorm(sampler.sample(cell, point), dimension));
            }
            if (!std::isfinite(cell_speed)) {
                throw std::invalid_argument(
                    "level-set transport safety evaluated a non-finite velocity");
            }
            local_minimum_length = std::min(local_minimum_length, cell_length);
            local_maximum_speed = std::max(local_maximum_speed, cell_speed);
            local_maximum_courant = std::max(
                local_maximum_courant, dt * cell_speed / cell_length);
            ++local_cells;
        });

        result.cells_checked = static_cast<std::size_t>(
            allReduceUnsignedSum(collective, local_cells));
        result.minimum_cell_length =
            allReduceRealMin(collective, local_minimum_length);
        result.maximum_speed =
            allReduceRealMax(collective, local_maximum_speed);
        result.maximum_courant =
            allReduceRealMax(collective, local_maximum_courant);
        if (result.cells_checked == 0u ||
            !std::isfinite(result.minimum_cell_length) ||
            !(result.minimum_cell_length > Real{0.0})) {
            result.diagnostic =
                "level-set transport safety found no nondegenerate owned cells";
            return result;
        }
        result.courant_satisfied =
            !options.enforce_courant_limit ||
            result.maximum_courant <=
                options.maximum_courant + options.courant_tolerance;

        std::set<int> open_markers;
        for (const auto& boundary : boundaries.inflow) {
            open_markers.insert(boundary.boundary_marker);
        }
        for (const auto& boundary : boundaries.outflow) {
            open_markers.insert(boundary.boundary_marker);
        }

        Real local_maximum_normal_velocity = Real{0.0};
        Real local_maximum_normal_ratio = Real{0.0};
        int local_worst_marker = -1;
        std::uint64_t local_boundary_faces = 0u;
        if (options.enforce_impermeable_boundaries) {
            system.meshAccess().forEachBoundaryFace(
                /*marker=*/-1,
                [&](GlobalIndex face, GlobalIndex cell) {
                    const int marker =
                        system.meshAccess().getBoundaryFaceMarker(face);
                    if (open_markers.find(marker) != open_markers.end()) {
                        return;
                    }
                    const auto type = system.meshAccess().getCellType(cell);
                    const auto reference = elements::ReferenceElement::create(type);
                    const auto local_face =
                        system.meshAccess().getLocalFaceIndex(face, cell);
                    if (local_face < 0 ||
                        static_cast<std::size_t>(local_face) >= reference.num_faces()) {
                        throw std::invalid_argument(
                            "level-set transport safety found an invalid local boundary face");
                    }
                    const auto& face_nodes =
                        reference.face_nodes(static_cast<std::size_t>(local_face));
                    coordinates.clear();
                    system.meshAccess().getCellCoordinates(cell, coordinates);
                    const auto normal = normalizedBoundaryNormal(
                        coordinates,
                        std::span<const LocalIndex>(face_nodes.data(), face_nodes.size()),
                        dimension);
                    std::vector<spaces::FunctionSpace::Value> sample_points;
                    sample_points.reserve(face_nodes.size() + 1u);
                    sample_points.push_back(referenceCentroid(
                        type,
                        std::span<const LocalIndex>(face_nodes.data(), face_nodes.size())));
                    const auto reference_node_count =
                        basis::ReferenceNodeLayout::num_nodes(type);
                    for (const auto local_node : face_nodes) {
                        if (local_node < 0 ||
                            static_cast<std::size_t>(local_node) >=
                                reference_node_count) {
                            throw std::invalid_argument(
                                "level-set transport safety found an invalid boundary-face node");
                        }
                        spaces::FunctionSpace::Value point{};
                        const auto node = basis::ReferenceNodeLayout::get_node_coords(
                            type, static_cast<std::size_t>(local_node));
                        point[0] = node[0];
                        point[1] = node[1];
                        point[2] = node[2];
                        sample_points.push_back(point);
                    }
                    for (const auto& point : sample_points) {
                        const auto value = sampler.sample(cell, point);
                        Real normal_velocity = Real{0.0};
                        for (int d = 0; d < dimension; ++d) {
                            normal_velocity +=
                                value[static_cast<std::size_t>(d)] *
                                normal[static_cast<std::size_t>(d)];
                        }
                        const Real magnitude = std::abs(normal_velocity);
                        const Real ratio = magnitude /
                            std::max(Real{1.0}, vectorNorm(value, dimension));
                        const Real tie_tolerance =
                            Real{32.0} * std::numeric_limits<Real>::epsilon() *
                            std::max({Real{1.0}, ratio,
                                      local_maximum_normal_ratio});
                        if (ratio > local_maximum_normal_ratio + tie_tolerance ||
                            (std::abs(ratio - local_maximum_normal_ratio) <=
                                 tie_tolerance &&
                             (local_worst_marker < 0 ||
                              marker < local_worst_marker))) {
                            local_maximum_normal_ratio = ratio;
                            local_worst_marker = marker;
                        }
                        local_maximum_normal_velocity =
                            std::max(local_maximum_normal_velocity, magnitude);
                    }
                    ++local_boundary_faces;
                });
        }
        result.impermeable_boundary_faces_checked =
            static_cast<std::size_t>(
                allReduceUnsignedSum(collective, local_boundary_faces));
        result.maximum_boundary_normal_velocity = allReduceRealMax(
            collective, local_maximum_normal_velocity);
        result.maximum_boundary_normal_velocity_ratio = allReduceRealMax(
            collective, local_maximum_normal_ratio);
        const Real marker_tie_tolerance =
            Real{32.0} * std::numeric_limits<Real>::epsilon() *
            std::max({Real{1.0}, local_maximum_normal_ratio,
                      result.maximum_boundary_normal_velocity_ratio});
        const int local_global_tie_marker =
            local_worst_marker >= 0 &&
                    std::abs(local_maximum_normal_ratio -
                             result.maximum_boundary_normal_velocity_ratio) <=
                        marker_tie_tolerance
                ? local_worst_marker
                : std::numeric_limits<int>::max();
        const int global_worst_marker =
            allReduceIntMin(collective, local_global_tie_marker);
        result.worst_boundary_marker =
            global_worst_marker == std::numeric_limits<int>::max()
                ? -1
                : global_worst_marker;
        result.impermeable_boundaries_satisfied =
            !options.enforce_impermeable_boundaries ||
            result.maximum_boundary_normal_velocity_ratio <=
                options.impermeable_normal_velocity_tolerance;

        result.success = result.courant_satisfied &&
                         result.impermeable_boundaries_satisfied;
        if (!result.courant_satisfied) {
            result.diagnostic =
                "level-set transport cell Courant number exceeds the one-ring limiter contract";
        } else if (!result.impermeable_boundaries_satisfied) {
            result.diagnostic =
                "level-set transport found nonzero advecting normal velocity on an undeclared wall";
        }
    } catch (const std::exception& error) {
        result.success = false;
        result.diagnostic = error.what();
    }
    return result;
}

LevelSetBoundPreservingResult applyLevelSetBoundPreservingLimiter(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetBoundaryOptions& boundaries,
    const LevelSetBoundPreservingOptions& options,
    std::span<const Real> previous_solution,
    std::span<const Real> candidate_solution,
    Real observed_courant,
    std::vector<Real>& limited_solution)
{
    LevelSetBoundPreservingResult result;
    result.observed_courant = observed_courant;
    limited_solution.assign(candidate_solution.begin(), candidate_solution.end());
    try {
        validateBoundPreservingOptions(options);
        if (previous_solution.size() != candidate_solution.size()) {
            result.diagnostic =
                "level-set bound-preserving limiter requires equal previous and candidate system spans";
            return result;
        }
        if (!options.enabled) {
            result.success = true;
            result.bounds_satisfied = true;
            result.sign_preservation_satisfied = true;
            result.diagnostic = "disabled";
            return result;
        }
        if (!std::isfinite(observed_courant) || observed_courant < Real{0.0}) {
            result.diagnostic =
                "level-set bound-preserving limiter received a non-finite or negative Courant number";
            return result;
        }
        if (options.enforce_courant_limit &&
            observed_courant >
                options.maximum_courant + options.courant_tolerance) {
            result.diagnostic =
                "level-set bound-preserving limiter rejected a step outside its one-ring Courant contract";
            return result;
        }
        if (level_set_field == INVALID_FIELD_ID) {
            result.diagnostic =
                "level-set bound-preserving limiter received an invalid field";
            return result;
        }
        const auto& record = system.fieldRecord(level_set_field);
        if (record.components != 1 || !record.space ||
            record.space->value_dimension() != 1 ||
            record.space->space_type() != spaces::SpaceType::H1) {
            result.diagnostic =
                "level-set bound-preserving limiter requires a scalar H1 field";
            return result;
        }
        if (record.space->is_variable_order() ||
            record.space->polynomial_order() != 1) {
            result.diagnostic =
                "level-set bound-preserving limiter currently certifies P1 H1 fields only";
            return result;
        }
        if (!system.fieldParticipatesInUnknownVector(level_set_field)) {
            result.diagnostic =
                "level-set bound-preserving limiter requires a transported unknown field";
            return result;
        }

        const auto& field_dofs = system.fieldDofHandler(level_set_field);
        const auto collective = collectiveContext(field_dofs);
        const auto field_count =
            static_cast<std::size_t>(field_dofs.getNumDofs());
        const auto field_offset = system.fieldDofOffset(level_set_field);
        if (field_offset < 0 ||
            static_cast<std::size_t>(field_offset) + field_count >
                previous_solution.size()) {
            result.diagnostic =
                "level-set bound-preserving limiter received incompatible system solution spans";
            return result;
        }
        result.field_dofs = field_count;
        const auto previous = previous_solution.subspan(
            static_cast<std::size_t>(field_offset), field_count);
        const auto candidate = candidate_solution.subspan(
            static_cast<std::size_t>(field_offset), field_count);

        for (std::size_t i = 0; i < field_count; ++i) {
            if (system.constraints().isConstrained(
                    field_offset + static_cast<GlobalIndex>(i))) {
                result.diagnostic =
                    "level-set bound-preserving limiter does not yet certify constrained level-set DOFs";
                return result;
            }
        }

        std::vector<Real> lower(
            field_count, std::numeric_limits<Real>::infinity());
        std::vector<Real> upper(
            field_count, -std::numeric_limits<Real>::infinity());
        system.meshAccess().forEachCell([&](GlobalIndex cell) {
            if (record.space->polynomial_order(cell) != 1) {
                throw std::invalid_argument(
                    "level-set bound-preserving limiter found a non-P1 cell");
            }
            const auto cell_dofs = field_dofs.getCellDofs(cell);
            if (cell_dofs.empty()) {
                throw std::invalid_argument(
                    "level-set bound-preserving limiter found an empty cell DOF patch");
            }
            Real cell_minimum = std::numeric_limits<Real>::infinity();
            Real cell_maximum = -std::numeric_limits<Real>::infinity();
            for (const auto dof : cell_dofs) {
                if (dof < 0 || static_cast<std::size_t>(dof) >= field_count) {
                    throw std::invalid_argument(
                        "level-set bound-preserving limiter found a cell DOF outside the field span");
                }
                const Real value = previous[static_cast<std::size_t>(dof)];
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "level-set bound-preserving limiter found a non-finite previous coefficient");
                }
                cell_minimum = std::min(cell_minimum, value);
                cell_maximum = std::max(cell_maximum, value);
            }
            for (const auto dof : cell_dofs) {
                const auto index = static_cast<std::size_t>(dof);
                lower[index] = std::min(lower[index], cell_minimum);
                upper[index] = std::max(upper[index], cell_maximum);
            }
        });

        // A prescribed inflow can introduce values outside the previous
        // interior patch.  Literal data are included in the face-DOF bounds;
        // time/space callbacks fail closed until they can be sampled at the
        // accepted time and high-order face nodes.
        for (const auto& boundary : boundaries.inflow) {
            const auto* literal = std::get_if<Real>(&boundary.value);
            if (literal == nullptr || !std::isfinite(*literal)) {
                result.diagnostic =
                    "level-set bound-preserving limiter currently requires finite literal inflow data";
                return result;
            }
            system.meshAccess().forEachBoundaryFace(
                boundary.boundary_marker,
                [&](GlobalIndex face, GlobalIndex cell) {
                    const auto type = system.meshAccess().getCellType(cell);
                    const auto reference = elements::ReferenceElement::create(type);
                    const auto local_face =
                        system.meshAccess().getLocalFaceIndex(face, cell);
                    if (local_face < 0 ||
                        static_cast<std::size_t>(local_face) >= reference.num_faces()) {
                        throw std::invalid_argument(
                            "level-set bound-preserving limiter found an invalid inflow face");
                    }
                    const auto cell_dofs = field_dofs.getCellDofs(cell);
                    const auto& face_nodes = reference.face_nodes(
                        static_cast<std::size_t>(local_face));
                    for (const auto local : face_nodes) {
                        const auto local_index = static_cast<std::size_t>(local);
                        if (local_index >= cell_dofs.size()) {
                            throw std::invalid_argument(
                                "level-set bound-preserving limiter cannot map an inflow face to P1 field DOFs");
                        }
                        const auto dof = cell_dofs[local_index];
                        if (dof < 0 || static_cast<std::size_t>(dof) >= field_count) {
                            throw std::invalid_argument(
                                "level-set bound-preserving limiter found an invalid inflow DOF");
                        }
                        const auto index = static_cast<std::size_t>(dof);
                        lower[index] = std::min(lower[index], *literal);
                        upper[index] = std::max(upper[index], *literal);
                    }
                });
        }

        allReduceRealVector(collective, lower, /*take_minimum=*/true);
        allReduceRealVector(collective, upper, /*take_minimum=*/false);
        for (std::size_t i = 0; i < field_count; ++i) {
            if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) ||
                lower[i] > upper[i]) {
                result.diagnostic =
                    "level-set bound-preserving limiter could not build a complete globally synchronized patch";
                return result;
            }
        }

        Real local_previous_minimum = std::numeric_limits<Real>::infinity();
        Real local_previous_maximum = -std::numeric_limits<Real>::infinity();
        Real local_candidate_minimum = std::numeric_limits<Real>::infinity();
        Real local_candidate_maximum = -std::numeric_limits<Real>::infinity();
        Real local_limited_minimum = std::numeric_limits<Real>::infinity();
        Real local_limited_maximum = -std::numeric_limits<Real>::infinity();
        Real local_maximum_unrelaxed_violation = Real{0.0};
        Real local_maximum_violation = Real{0.0};
        Real local_maximum_correction = Real{0.0};
        std::uint64_t local_limited_count = 0u;
        std::uint64_t local_positive_flips = 0u;
        std::uint64_t local_negative_flips = 0u;
        bool local_bounds_satisfied = true;
        bool local_sign_satisfied = true;
        const auto& dof_map = field_dofs.getDofMap();
        for (std::size_t i = 0; i < field_count; ++i) {
            const Real old_value = previous[i];
            const Real raw_value = candidate[i];
            if (!std::isfinite(raw_value)) {
                result.diagnostic =
                    "level-set bound-preserving limiter found a non-finite candidate coefficient";
                return result;
            }
            const Real scale = std::max(
                {Real{1.0}, std::abs(lower[i]), std::abs(upper[i])});
            const Real tolerance = options.bound_tolerance * scale;
            const Real relaxed_lower = lower[i] - tolerance;
            const Real relaxed_upper = upper[i] + tolerance;
            const Real limited = std::clamp(raw_value,
                                            relaxed_lower,
                                            relaxed_upper);
            limited_solution[static_cast<std::size_t>(field_offset) + i] = limited;

            const Real violation = std::max(
                {Real{0.0}, relaxed_lower - raw_value,
                 raw_value - relaxed_upper});
            const Real unrelaxed_violation = std::max(
                {Real{0.0}, lower[i] - raw_value,
                 raw_value - upper[i]});
            const Real correction = std::abs(limited - raw_value);
            const bool owned = dof_map.isOwnedDof(
                static_cast<GlobalIndex>(i));
            if (owned) {
                local_previous_minimum = std::min(local_previous_minimum, old_value);
                local_previous_maximum = std::max(local_previous_maximum, old_value);
                local_candidate_minimum = std::min(local_candidate_minimum, raw_value);
                local_candidate_maximum = std::max(local_candidate_maximum, raw_value);
                local_limited_minimum = std::min(local_limited_minimum, limited);
                local_limited_maximum = std::max(local_limited_maximum, limited);
                local_maximum_violation =
                    std::max(local_maximum_violation, violation);
                local_maximum_unrelaxed_violation = std::max(
                    local_maximum_unrelaxed_violation,
                    unrelaxed_violation);
                local_maximum_correction =
                    std::max(local_maximum_correction, correction);
                if (correction > Real{0.0}) {
                    ++local_limited_count;
                }
                if (lower[i] > options.sign_tolerance &&
                    raw_value < -options.sign_tolerance) {
                    ++local_positive_flips;
                }
                if (upper[i] < -options.sign_tolerance &&
                    raw_value > options.sign_tolerance) {
                    ++local_negative_flips;
                }
            }
            if (limited < relaxed_lower || limited > relaxed_upper) {
                local_bounds_satisfied = false;
            }
            if ((lower[i] > options.sign_tolerance &&
                 limited < -options.sign_tolerance) ||
                (upper[i] < -options.sign_tolerance &&
                 limited > options.sign_tolerance)) {
                local_sign_satisfied = false;
            }
        }

        result.previous_minimum =
            allReduceRealMin(collective, local_previous_minimum);
        result.previous_maximum =
            allReduceRealMax(collective, local_previous_maximum);
        result.candidate_minimum =
            allReduceRealMin(collective, local_candidate_minimum);
        result.candidate_maximum =
            allReduceRealMax(collective, local_candidate_maximum);
        result.limited_minimum =
            allReduceRealMin(collective, local_limited_minimum);
        result.limited_maximum =
            allReduceRealMax(collective, local_limited_maximum);
        result.maximum_bound_violation =
            allReduceRealMax(collective, local_maximum_violation);
        result.maximum_unrelaxed_bound_violation =
            allReduceRealMax(collective,
                             local_maximum_unrelaxed_violation);
        result.maximum_correction =
            allReduceRealMax(collective, local_maximum_correction);
        result.limited_dofs = static_cast<std::size_t>(
            allReduceUnsignedSum(collective, local_limited_count));
        result.positive_patch_sign_flips_prevented =
            static_cast<std::size_t>(
                allReduceUnsignedSum(collective, local_positive_flips));
        result.negative_patch_sign_flips_prevented =
            static_cast<std::size_t>(
                allReduceUnsignedSum(collective, local_negative_flips));
        result.bounds_satisfied =
            allReduceRealMin(collective,
                             local_bounds_satisfied ? Real{1.0} : Real{0.0}) >
            Real{0.5};
        result.sign_preservation_satisfied =
            allReduceRealMin(collective,
                             local_sign_satisfied ? Real{1.0} : Real{0.0}) >
            Real{0.5};
        result.applied = result.limited_dofs > 0u;
        result.success = result.bounds_satisfied &&
                         result.sign_preservation_satisfied;
        if (!result.success) {
            result.diagnostic =
                "level-set bound-preserving limiter failed its post-projection invariant check";
        } else if (!result.applied) {
            result.diagnostic = "candidate already satisfies previous one-ring bounds";
        }
    } catch (const std::exception& error) {
        result.success = false;
        result.diagnostic = error.what();
    }
    return result;
}

bool shouldReinitializeLevelSet(
    const LevelSetReinitializationOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled &&
           options.cadence_steps > 0 &&
           completed_step_index > 0 &&
           completed_step_index % options.cadence_steps == 0;
}

LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    LevelSetTransportForm transport_form,
    const LevelSetReinitializationOptions& reinitialization,
    const LevelSetVolumeCorrectionOptions& volume_correction) noexcept
{
    if (volume_correction.enabled) {
        return LevelSetConservationDiagnostic::VolumeCorrectedAdvectionNotLocallyConservative;
    }
    if (reinitialization.enabled) {
        return LevelSetConservationDiagnostic::ReinitializedAdvectionNotConservative;
    }
    if (transport_form == LevelSetTransportForm::ConservativeDivergence) {
        return LevelSetConservationDiagnostic::ConservativeDivergenceAdvectionNotLocallyConservative;
    }
    return LevelSetConservationDiagnostic::PlainAdvectionNotConservative;
}

LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    const LevelSetTransportOptions& options) noexcept
{
    if (options.conservative_phase.enabled) {
        return LevelSetConservationDiagnostic::
            ConservativePhaseIndicatorLocallyConservative;
    }
    return levelSetConservationDiagnostic(
        options.transport_form,
        options.reinitialization,
        options.volume_correction);
}

const char* levelSetConservationDiagnosticName(
    LevelSetConservationDiagnostic diagnostic) noexcept
{
    switch (diagnostic) {
    case LevelSetConservationDiagnostic::PlainAdvectionNotConservative:
        return "plain_level_set_advection_not_conservative";
    case LevelSetConservationDiagnostic::ConservativeDivergenceAdvectionNotLocallyConservative:
        return "conservative_divergence_level_set_advection_not_locally_conservative";
    case LevelSetConservationDiagnostic::ReinitializedAdvectionNotConservative:
        return "reinitialized_level_set_advection_not_conservative";
    case LevelSetConservationDiagnostic::VolumeCorrectedAdvectionNotLocallyConservative:
        return "volume_corrected_level_set_advection_not_locally_conservative";
    case LevelSetConservationDiagnostic::ConservativePhaseIndicatorLocallyConservative:
        return "conservative_p1_phase_indicator_locally_conservative";
    }
    return "unknown_level_set_conservation";
}

bool shouldApplyLevelSetVolumeCorrection(
    const LevelSetVolumeCorrectionOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled &&
           options.cadence_steps > 0 &&
           completed_step_index > 0 &&
           completed_step_index % options.cadence_steps == 0;
}

systems::CoupledResidualKernels installLevelSetTransport(
    systems::FESystem& system,
    std::shared_ptr<const spaces::FunctionSpace> level_set_space,
    const LevelSetTransportOptions& options,
    const systems::FormInstallOptions& install_options)
{
    const bool material_interface_velocity =
        options.velocity.source ==
        LevelSetVelocitySource::MaterialInterfacePhasePair;
    if (options.level_set.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field name must be non-empty");
    }
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity &&
        options.velocity.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity field name must be non-empty");
    }
    if (level_set_space && level_set_space->value_dimension() != 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field space must be scalar");
    }
    if (options.supg.enabled &&
        (!(options.supg.tau_scale > 0.0) ||
         !std::isfinite(options.supg.tau_scale))) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG tau_scale must be positive");
    }
    if (options.supg.enabled &&
        (!(options.supg.velocity_epsilon > 0.0) ||
         !std::isfinite(options.supg.velocity_epsilon))) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG velocity_epsilon must be positive");
    }
    if (options.supg.enabled &&
        (!(options.supg.transient_scale > 0.0) ||
         !std::isfinite(options.supg.transient_scale))) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG transient_scale must be positive");
    }
    if (options.supg.discontinuity_capturing_enabled &&
        !options.supg.enabled) {
        throw std::invalid_argument(
            "installLevelSetTransport: discontinuity capturing requires SUPG to be enabled");
    }
    if (options.supg.discontinuity_capturing_enabled &&
        (!(options.supg.discontinuity_capturing_scale > 0.0) ||
         !std::isfinite(options.supg.discontinuity_capturing_scale))) {
        throw std::invalid_argument(
            "installLevelSetTransport: discontinuity_capturing_scale must be positive when enabled");
    }
    if (options.supg.discontinuity_capturing_enabled &&
        (!(options.supg.gradient_epsilon > 0.0) ||
         !std::isfinite(options.supg.gradient_epsilon))) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG gradient_epsilon must be positive when discontinuity capturing is enabled");
    }
    if (options.supg.discontinuity_capturing_enabled &&
        (!(options.supg.discontinuity_capturing_residual_epsilon > 0.0) ||
         !std::isfinite(
             options.supg.discontinuity_capturing_residual_epsilon))) {
        throw std::invalid_argument(
            "installLevelSetTransport: discontinuity_capturing_residual_epsilon must be positive when enabled");
    }
    if (options.supg.discontinuity_capturing_enabled &&
        (!(options.supg.discontinuity_capturing_max_courant > 0.0) ||
         !std::isfinite(options.supg.discontinuity_capturing_max_courant))) {
        throw std::invalid_argument(
            "installLevelSetTransport: discontinuity_capturing_max_courant must be positive when enabled");
    }
    validateReinitializationOptions(options.reinitialization);
    validateVolumeCorrectionOptions(options.volume_correction);
    validateInterfaceKinematicOptions(options.interface_kinematic);
    validateBoundaryOptions(options.boundaries);
    validateBoundPreservingOptions(options.bound_preserving);
    validateConservativePhaseOptions(options);
    if (options.operator_tag.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: operator_tag must be non-empty");
    }

    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity &&
        options.level_set.field_name == options.velocity.field_name) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set and velocity fields must be distinct");
    }
    if (options.conservative_phase.enabled &&
        options.conservative_phase.liquid_indicator.field_name ==
            options.level_set.field_name) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set and conservative phase fields must be distinct");
    }
    if (options.conservative_phase.enabled &&
        options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity &&
        options.conservative_phase.liquid_indicator.field_name ==
            options.velocity.field_name) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase and velocity fields must be distinct");
    }

    // Complete the definition preflight before adding either field.  In
    // particular, a missing/incompatible velocity or extension source must not
    // leave behind an auto-registered level-set field.
    const auto pending_phi = preflightLevelSetField(
        system, options.level_set, level_set_space);
    const int dimension = pending_phi.space
        ? pending_phi.space->topological_dimension()
        : 0;
    if (dimension < 1 || dimension > 3) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set space dimension must be in [1, 3]");
    }

    if (material_interface_velocity &&
        (pending_phi.existing_id == INVALID_FIELD_ID ||
         options.velocity.material_interface_marker < 0 ||
         options.velocity.auto_register_field || options.velocity.space ||
         !options.velocity.algebraic_extension_source_field_name.empty() ||
         options.transport_form != LevelSetTransportForm::Advective ||
         options.bound_preserving.enabled ||
         !options.conservative_phase.enabled ||
         !options.conservative_phase.reconcile_geometry ||
         !options.interface_kinematic.enabled ||
         options.interface_kinematic.interface_marker !=
             options.velocity.material_interface_marker ||
         !options.boundaries.inflow.empty() ||
         !options.boundaries.outflow.empty())) {
        throw std::invalid_argument(
            "installLevelSetTransport: material-interface phase-pair velocity requires an existing level-set field, one matching interface marker, advective-form conservative phase transport with geometry reconciliation, interface kinematic enforcement, no legacy bound projection, no separate velocity registration or extension, and a closed physical boundary");
    }

    std::optional<PendingTransportField> pending_phase;
    if (options.conservative_phase.enabled) {
        pending_phase = preflightLevelSetField(
            system,
            options.conservative_phase.liquid_indicator,
            pending_phi.space);
        if (!isFixedP1ScalarH1(pending_phi) ||
            !isFixedP1ScalarH1(*pending_phase) ||
            pending_phase->space->topological_dimension() != dimension) {
            throw std::invalid_argument(
                "installLevelSetTransport: conservative phase transport requires fixed scalar P1 H1 level-set and phase fields on the same mesh dimension");
        }
    }

    std::optional<PendingTransportField> pending_velocity;
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity) {
        pending_velocity = preflightVelocityField(
            system, options.velocity, dimension);
    }

    const interfaces::MaterialInterfaceTransportVelocityDeclaration*
        material_velocity_declaration = nullptr;
    if (material_interface_velocity) {
        material_velocity_declaration =
            &requireMaterialInterfaceVelocityDeclaration(
                system,
                pending_phi.existing_id,
                options.velocity.material_interface_marker);
        if (material_velocity_declaration->dimension != dimension) {
            throw std::invalid_argument(
                "installLevelSetTransport: material-interface phase-pair velocity dimension differs from the level-set space");
        }
    }

    FieldId algebraic_extension_source_id = INVALID_FIELD_ID;
    if (!options.velocity.algebraic_extension_source_field_name.empty()) {
        if (options.velocity.source != LevelSetVelocitySource::CoupledField) {
            throw std::invalid_argument(
                "installLevelSetTransport: a state-dependent algebraic velocity extension must be a CoupledField unknown");
        }
        if (options.velocity.algebraic_extension_source_field_name ==
            options.velocity.field_name) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension and physical velocity fields must be distinct");
        }
        if (options.conservative_phase.enabled &&
            options.velocity.algebraic_extension_source_field_name ==
                options.conservative_phase.liquid_indicator.field_name) {
            throw std::invalid_argument(
                "installLevelSetTransport: conservative phase and physical velocity fields must be distinct");
        }
        algebraic_extension_source_id = system.findFieldByName(
            options.velocity.algebraic_extension_source_field_name);
        if (algebraic_extension_source_id == INVALID_FIELD_ID ||
            !system.hasField(
                options.velocity.algebraic_extension_source_field_name)) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension references unknown physical velocity field '" +
                options.velocity.algebraic_extension_source_field_name + "'");
        }
        validateVelocityField(
            system,
            algebraic_extension_source_id,
            options.velocity.algebraic_extension_source_field_name);
        if (!system.fieldParticipatesInUnknownVector(
                algebraic_extension_source_id)) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension physical velocity must be an unknown field");
        }
        const auto& source_record =
            system.fieldRecord(algebraic_extension_source_id);
        if (!source_record.space || source_record.components != dimension ||
            source_record.space->value_dimension() != dimension ||
            source_record.space->topological_dimension() != dimension) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension and physical velocity dimensions differ");
        }
        if (pending_velocity.has_value() &&
            pending_velocity->existing_id != INVALID_FIELD_ID &&
            findLevelSetVelocityExtensionConstraintKernel(
                system,
                options.operator_tag,
                pending_velocity->existing_id)) {
            throw std::invalid_argument(
                "installLevelSetTransport: duplicate algebraic velocity extension constraint for field '" +
                options.velocity.field_name + "'");
        }
    }

    const auto phi_id = ensureLevelSetField(
        system, options.level_set, std::move(level_set_space));

    FieldId phase_id = INVALID_FIELD_ID;
    if (options.conservative_phase.enabled) {
        phase_id = ensureLevelSetField(
            system,
            options.conservative_phase.liquid_indicator,
            pending_phase->space);
    }

    FieldId velocity_id = INVALID_FIELD_ID;
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity) {
        velocity_id = ensureVelocityField(system, options.velocity);
    }

    const auto& phi_rec = system.fieldRecord(phi_id);

    using namespace forms;
    const auto phi = StateField(phi_id, *phi_rec.space, options.level_set.field_name);
    const auto eta = TestField(phi_id, *phi_rec.space, "eta");
    FormExpr velocity;
    FormExpr negative_material_velocity;
    FormExpr positive_material_velocity;
    FormExpr material_interface_trace_velocity;
    if (material_interface_velocity) {
        const auto& negative_record = system.fieldRecord(
            material_velocity_declaration->negative_velocity_field);
        const auto& positive_record = system.fieldRecord(
            material_velocity_declaration->positive_velocity_field);
        negative_material_velocity = StateField(
            material_velocity_declaration->negative_velocity_field,
            *negative_record.space,
            negative_record.name);
        positive_material_velocity = StateField(
            material_velocity_declaration->positive_velocity_field,
            *positive_record.space,
            positive_record.name);
        material_interface_trace_velocity =
            FormExpr::constant(
                material_velocity_declaration->negative_trace_weight) *
                negative_material_velocity +
            FormExpr::constant(
                material_velocity_declaration->positive_trace_weight) *
                positive_material_velocity;
    } else if (options.velocity.source ==
               LevelSetVelocitySource::ConstantVector) {
        std::vector<FormExpr> components;
        components.reserve(static_cast<std::size_t>(dimension));
        for (int d = 0; d < dimension; ++d) {
            components.push_back(
                FormExpr::constant(options.velocity.constant_value[static_cast<std::size_t>(d)]));
        }
        velocity = FormExpr::asVector(std::move(components));
    } else {
        const auto& velocity_rec = system.fieldRecord(velocity_id);
        velocity = options.velocity.source == LevelSetVelocitySource::CoupledField
                       ? StateField(velocity_id, *velocity_rec.space, options.velocity.field_name)
                       : FormExpr::discreteField(
                             velocity_id,
                             *velocity_rec.space,
                             options.velocity.field_name);
    }

    const auto time_residual = dt(phi);
    const auto spatialResidual = [&](const FormExpr& advecting_velocity) {
        return options.transport_form ==
                       LevelSetTransportForm::ConservativeDivergence
                   ? div(phi * advecting_velocity)
                   : dot(advecting_velocity, grad(phi));
    };
    auto residual = (time_residual * eta).dx();
    FormExpr interface_spatial_residual;
    if (material_interface_velocity) {
        const auto negative_spatial_residual =
            spatialResidual(negative_material_velocity);
        const auto positive_spatial_residual =
            spatialResidual(positive_material_velocity);
        residual = residual +
                   (negative_spatial_residual * eta)
                       .dCutVolume(
                           material_velocity_declaration->interface_marker,
                           CutVolumeSide::Negative) +
                   (positive_spatial_residual * eta)
                       .dCutVolume(
                           material_velocity_declaration->interface_marker,
                           CutVolumeSide::Positive);
        interface_spatial_residual =
            spatialResidual(material_interface_trace_velocity);
    } else {
        interface_spatial_residual = spatialResidual(velocity);
        residual = residual + (interface_spatial_residual * eta).dx();
    }

    const auto appendSupg = [&](const FormExpr& advecting_velocity,
                                const FormExpr& spatial_residual,
                                std::optional<CutVolumeSide> side) {
        if (!options.supg.enabled) {
            return;
        }
        // Directional/transient SUPG scale:
        //   tau = c_tau / sqrt((c_t/dt_eff)^2 + u . G u + eps),
        //   G   = J^{-T} J^{-1}.
        // Unlike h/|u| this remains finite as u -> 0, responds to anisotropic
        // cells in the streamline direction, and contracts with the actual
        // transient integration scale.
        const auto eps = FormExpr::constant(options.supg.velocity_epsilon);
        const auto dt_eff = deltat_eff();
        const auto metric = transpose(Jinv()) * Jinv();
        const auto transient_rate =
            FormExpr::constant(options.supg.transient_scale) / dt_eff;
        const auto directional_rate_squared =
            inner(advecting_velocity, metric * advecting_velocity);
        const auto tau = FormExpr::constant(options.supg.tau_scale) /
                         sqrt(transient_rate * transient_rate +
                              directional_rate_squared + eps);
        const auto streamline_test =
            tau * dot(advecting_velocity, grad(eta));
        const auto integrate = [&](const FormExpr& integrand) {
            return side.has_value()
                       ? integrand.dCutVolume(
                             material_velocity_declaration->interface_marker,
                             *side)
                       : integrand.dx();
        };
        residual = residual + integrate(streamline_test * time_residual) +
                   integrate(streamline_test * spatial_residual);

        if (options.supg.discontinuity_capturing_enabled) {
            // Residual-based diffusion supplies the cross-stream control that
            // streamline-only SUPG lacks at a gate/interface cliff.  Cap it by
            // h|u| + h^2/dt so transient nonlinear residuals cannot create
            // unbounded artificial diffusion while low-speed wall/interface
            // regions retain a finite transient cap.
            const auto strong_residual = time_residual + spatial_residual;
            const auto grad_norm = sqrt(
                inner(grad(phi), grad(phi)) +
                FormExpr::constant(options.supg.gradient_epsilon));
            const auto speed = sqrt(
                inner(advecting_velocity, advecting_velocity) + eps);
            const auto raw_diffusivity =
                FormExpr::constant(options.supg.discontinuity_capturing_scale) *
                h() *
                (smoothAbs(
                     strong_residual,
                     FormExpr::constant(
                         options.supg
                             .discontinuity_capturing_residual_epsilon)) -
                 FormExpr::constant(
                     options.supg
                         .discontinuity_capturing_residual_epsilon)) /
                grad_norm;
            const auto diffusivity_cap =
                FormExpr::constant(
                    options.supg.discontinuity_capturing_max_courant) *
                (h() * speed + h() * h() / dt_eff);
            const auto diffusivity = min(raw_diffusivity, diffusivity_cap);
            residual = residual + integrate(
                diffusivity * inner(grad(phi), grad(eta)));
        }
    };
    if (material_interface_velocity) {
        appendSupg(
            negative_material_velocity,
            spatialResidual(negative_material_velocity),
            CutVolumeSide::Negative);
        appendSupg(
            positive_material_velocity,
            spatialResidual(positive_material_velocity),
            CutVolumeSide::Positive);
    } else {
        appendSupg(velocity, interface_spatial_residual, std::nullopt);
    }
    if (options.interface_kinematic.enabled) {
        residual = residual +
                   (FormExpr::constant(options.interface_kinematic.weight_scale) *
                    h() * time_residual * eta)
                       .dI(options.interface_kinematic.interface_marker) +
                   (FormExpr::constant(options.interface_kinematic.weight_scale) *
                    h() * interface_spatial_residual * eta)
                       .dI(options.interface_kinematic.interface_marker);
    }

    if (phase_id != INVALID_FIELD_ID) {
        const auto& phase_record = system.fieldRecord(phase_id);
        const auto phase = StateField(
            phase_id,
            *phase_record.space,
            options.conservative_phase.liquid_indicator.field_name);
        const auto phase_test = TestField(
            phase_id, *phase_record.space, "phase_test");
        // The conservative stage is an explicit accepted-candidate operation.
        // This algebraic equation keeps q at its previous accepted endpoint
        // during Newton without classifying q as a time-derivative field.
        residual = residual +
                   ((phase - FormExpr::previousSolution(1)) * phase_test).dx();
    }

    for (const auto& bc : options.boundaries.inflow) {
        const int marker = forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "installLevelSetTransport: inflow boundary");
        const auto normal_velocity = dot(velocity, FormExpr::normal());
        const auto inflow_speed =
            FormExpr::constant(0.5) * (abs(normal_velocity) - normal_velocity);
        const auto target = forms::bc::toScalarExpr(
            bc.value,
            forms::bc::markerValueName("level_set_inflow", marker));
        const auto penalty = FormExpr::constant(bc.penalty_scale) * inflow_speed;
        const auto boundary_measure =
            forms::ExteriorBoundaryMeasure::fullPhysical(marker);
        residual =
            residual +
            (penalty * (phi - target) * eta)
                .dExteriorBoundary(boundary_measure);
    }

    if (!system.hasOperator(options.operator_tag)) {
        system.addOperator(options.operator_tag);
    }

    auto install = install_options;
    install.compiler_options.use_symbolic_tangent = true;
    if (options.velocity.source == LevelSetVelocitySource::CoupledField) {
        install.extra_trial_fields.push_back(velocity_id);
    } else if (material_interface_velocity) {
        install.extra_trial_fields.push_back(
            material_velocity_declaration->negative_velocity_field);
        install.extra_trial_fields.push_back(
            material_velocity_declaration->positive_velocity_field);
    }
    std::vector<FieldId> residual_fields{phi_id};
    if (phase_id != INVALID_FIELD_ID) {
        residual_fields.push_back(phase_id);
    }
    auto kernels = systems::installFormulation(
        system,
        options.operator_tag,
        residual_fields,
        residual,
        install);
    if (algebraic_extension_source_id != INVALID_FIELD_ID) {
        system.addGlobalKernel(
            options.operator_tag,
            std::make_shared<LevelSetVelocityExtensionConstraintKernel>(
                LevelSetVelocityExtensionConstraintConfig{
                    .extension_field = velocity_id,
                    .source_velocity_field =
                        algebraic_extension_source_id,
                    .components =
                        system.fieldRecord(velocity_id).components,
                    .operator_tag = options.operator_tag,
                }));
    }
    return kernels;
}

} // namespace svmp::FE::level_set
