/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/SymbolicDifferentiation.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
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

IncompressibleNavierStokesVMSModule::IncompressibleNavierStokesVMSModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
    IncompressibleNavierStokesVMSOptions options)
    : velocity_space_(std::move(velocity_space))
    , pressure_space_(std::move(pressure_space))
    , options_(std::move(options))
{
}

namespace {

using FreeSurfaceBoundary = IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;
using FreeSurfaceContactLine =
    IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

[[nodiscard]] bool coordinateCutContextCallbackLocalPhase(
    const FE::systems::FESystem& system,
    const std::exception_ptr& local_exception,
    bool local_available,
    std::string_view phase)
{
    bool any_failed = local_exception != nullptr;
    bool all_available = local_available && !any_failed;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        MPI_Finalized(&mpi_finalized);
    }
    if (mpi_initialized != 0 && mpi_finalized == 0) {
        const auto communicator = system.activeMpiCommunicator();
        if (communicator != MPI_COMM_NULL) {
            int communicator_size = 1;
            MPI_Comm_size(communicator, &communicator_size);
            if (communicator_size > 1) {
                // 0 = failed, 1 = unavailable, 2 = available. MPI_MIN makes
                // every rank take the same route after this local phase.
                const int local_state =
                    any_failed ? 0 : (local_available ? 2 : 1);
                int global_state = 0;
                MPI_Allreduce(
                    &local_state,
                    &global_state,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    communicator);
                any_failed = global_state == 0;
                all_available = global_state == 2;
            }
        }
    }
#else
    (void)system;
#endif
    if (any_failed) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: another communicator "
            "rank rejected cut-context callback phase '" +
            std::string(phase) + "'");
    }
    return all_available;
}

struct EmbeddedFreeSurfaceMeasureEvidence {
    FE::Real stored_measure{0.0};
    FE::Real quadrature_weight_measure{0.0};
    std::uint64_t rule_count{0u};
    std::uint64_t quadrature_point_count{0u};
    bool invalid_measure{false};
};

[[nodiscard]] EmbeddedFreeSurfaceMeasureEvidence
embeddedFreeSurfaceMeasureEvidence(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext& context,
    int interface_marker)
{
    EmbeddedFreeSurfaceMeasureEvidence local;
    std::exception_ptr local_preflight_exception;
    try {
        context.assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
        for (const auto* rule :
             context.interfaceRulesForMarker(
                 interface_marker)) {
            if (rule == nullptr) {
                local.invalid_measure = true;
                continue;
            }
            ++local.rule_count;
            if (!std::isfinite(rule->measure)) {
                local.invalid_measure = true;
            } else {
                local.stored_measure += rule->measure;
            }
            for (const auto& point : rule->points) {
                ++local.quadrature_point_count;
                if (!std::isfinite(point.weight)) {
                    local.invalid_measure = true;
                } else {
                    local.quadrature_weight_measure +=
                        point.weight;
                }
            }
        }
    } catch (...) {
        local_preflight_exception =
            std::current_exception();
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        MPI_Finalized(&mpi_finalized);
    }
    MPI_Comm communicator = MPI_COMM_NULL;
    if (mpi_initialized != 0 &&
        mpi_finalized == 0) {
        communicator =
            system.activeMpiCommunicator();
    }
    if (communicator != MPI_COMM_NULL) {
        int communicator_size = 1;
        MPI_Comm_size(communicator, &communicator_size);
        if (communicator_size > 1) {
            const int local_ok =
                local_preflight_exception == nullptr
                    ? 1
                    : 0;
            int all_ok = 0;
            MPI_Allreduce(
                &local_ok,
                &all_ok,
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            if (all_ok == 0) {
                if (local_preflight_exception != nullptr) {
                    std::rethrow_exception(
                        local_preflight_exception);
                }
                throw std::runtime_error(
                    "IncompressibleNavierStokesVMSModule: "
                    "another communicator rank rejected "
                    "embedded free-surface measure preflight");
            }
            const std::array<double, 2> local_measures{
                static_cast<double>(local.stored_measure),
                static_cast<double>(local.quadrature_weight_measure)};
            std::array<double, 2> global_measures{};
            MPI_Allreduce(local_measures.data(), global_measures.data(),
                          static_cast<int>(local_measures.size()), MPI_DOUBLE,
                          MPI_SUM, communicator);

            const std::array<unsigned long long, 2> local_counts{
                static_cast<unsigned long long>(local.rule_count),
                static_cast<unsigned long long>(local.quadrature_point_count)};
            std::array<unsigned long long, 2> global_counts{};
            MPI_Allreduce(local_counts.data(), global_counts.data(),
                          static_cast<int>(local_counts.size()),
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

            const int local_invalid = local.invalid_measure ? 1 : 0;
            int global_invalid = 0;
            MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT,
                          MPI_MAX, communicator);

            local.stored_measure =
                static_cast<FE::Real>(global_measures[0]);
            local.quadrature_weight_measure =
                static_cast<FE::Real>(global_measures[1]);
            local.rule_count =
                static_cast<std::uint64_t>(global_counts[0]);
            local.quadrature_point_count =
                static_cast<std::uint64_t>(global_counts[1]);
            local.invalid_measure = global_invalid != 0;
        }
    }
#else
    (void)system;
#endif
    if (local_preflight_exception != nullptr) {
        std::rethrow_exception(
            local_preflight_exception);
    }
    return local;
}

void requirePositiveEmbeddedFreeSurfaceMeasure(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext* context,
    int interface_marker,
    std::string_view level_set_field_name,
    std::string_view domain_id,
    std::string_view validation_stage)
{
    EmbeddedFreeSurfaceMeasureEvidence evidence;
    if (context != nullptr) {
        evidence = embeddedFreeSurfaceMeasureEvidence(
            system, *context, interface_marker);
    }
    if (context != nullptr && !evidence.invalid_measure &&
        evidence.rule_count > 0u && evidence.quadrature_point_count > 0u &&
        evidence.stored_measure > FE::Real{0.0} &&
        evidence.quadrature_weight_measure > FE::Real{0.0}) {
        return;
    }

    std::ostringstream message;
    message
        << "IncompressibleNavierStokesVMSModule: CutVolume embedded "
           "free-surface pressure anchor requires a positive generated "
           "interface measure before a pressure solve"
        << " validation_stage=" << validation_stage
        << " marker=" << interface_marker
        << " level_set_field='" << level_set_field_name << "'"
        << " domain_id='" << domain_id << "'"
        << " cut_context=" << (context != nullptr ? "present" : "missing")
        << " global_interface_rules=" << evidence.rule_count
        << " global_interface_quadrature_points="
        << evidence.quadrature_point_count
        << " global_stored_measure=" << evidence.stored_measure
        << " global_quadrature_weight_measure="
        << evidence.quadrature_weight_measure
        << " invalid_measure=" << (evidence.invalid_measure ? "true" : "false")
        << " gauge_policy=reject_zero_interface_no_dynamic_gauge_insertion";
    throw std::runtime_error(message.str());
}

class CutVolumePressureAnchorMeasureGuard final
    : public FE::systems::GlobalKernel {
public:
    CutVolumePressureAnchorMeasureGuard(int interface_marker,
                                        std::string level_set_field_name,
                                        std::string domain_id)
        : interface_marker_(interface_marker)
        , level_set_field_name_(std::move(level_set_field_name))
        , domain_id_(std::move(domain_id))
    {
    }

    [[nodiscard]] std::string name() const override
    {
        return "CutVolumePressureAnchorMeasureGuard:" +
               std::to_string(interface_marker_);
    }

    void addSparsityCouplings(
        const FE::systems::FESystem& system,
        FE::sparsity::SparsityPattern&) const override
    {
        // Application setup intentionally precedes its first generated-cut
        // refresh.  A missing context is therefore deferred until refresh or
        // assembly, while a preloaded context must already satisfy the anchor.
        if (system.cutIntegrationContext() != nullptr) {
            validate(system, system.cutIntegrationContext(), "setup");
        }
    }

    [[nodiscard]] FE::assembly::AssemblyResult assemble(
        const FE::systems::FESystem& system,
        const FE::systems::AssemblyRequest&,
        const FE::systems::SystemStateView&,
        FE::assembly::GlobalSystemView*,
        FE::assembly::GlobalSystemView*) override
    {
        validate(system, system.cutIntegrationContext(), "assembly");
        return {};
    }

    void validateContextUpdate(
        const FE::systems::FESystem& system,
        const FE::assembly::CutIntegrationContext* context) const
    {
        validate(system, context, "cut_context_update");
    }

private:
    void validate(const FE::systems::FESystem& system,
                  const FE::assembly::CutIntegrationContext* context,
                  std::string_view stage) const
    {
        requirePositiveEmbeddedFreeSurfaceMeasure(
            system, context, interface_marker_, level_set_field_name_,
            domain_id_, stage);
    }

    int interface_marker_{-1};
    std::string level_set_field_name_{};
    std::string domain_id_{};
};

struct DiagnosticBooleanOverride {
    bool value{false};
    std::string name{};
    std::string raw{};
};

struct DiagnosticScalarOverride {
    FE::Real value{0.0};
    std::string name{};
    std::string raw{};
};

enum class PspgPressureGradientForm {
    Absolute,
    Incremental,
};

struct DiagnosticPspgPressureGradientFormOverride {
    PspgPressureGradientForm value{PspgPressureGradientForm::Absolute};
    std::string name{};
    std::string raw{};
};

const char* pspgPressureGradientFormName(PspgPressureGradientForm form)
{
    switch (form) {
    case PspgPressureGradientForm::Absolute:
        return "absolute";
    case PspgPressureGradientForm::Incremental:
        return "incremental";
    }
    return "unknown";
}

std::optional<DiagnosticBooleanOverride> readDiagnosticBooleanOverride(
    const char* name)
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return std::nullopt;
    }

    std::string normalized(env);
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });

    const bool value =
        normalized != "0" && normalized != "false" &&
        normalized != "no" && normalized != "off";
    return DiagnosticBooleanOverride{value, name, env};
}

std::optional<DiagnosticBooleanOverride> navierStokesVmsDiagnosticOverride()
{
    if (auto enable = readDiagnosticBooleanOverride("SVMP_NS_ENABLE_VMS")) {
        return enable;
    }
    if (auto disable = readDiagnosticBooleanOverride("SVMP_NS_DISABLE_VMS")) {
        if (disable->value) {
            disable->value = false;
            return disable;
        }
    }
    return std::nullopt;
}

bool pressureRowContributionDiagnosticEnabled()
{
    if (auto value =
            readDiagnosticBooleanOverride(
                "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC")) {
        return value->value;
    }
    if (auto value =
            readDiagnosticBooleanOverride(
                "SVMP_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC")) {
        return value->value;
    }
    return false;
}

bool freeSurfaceConservativeBalanceDiagnosticEnabled()
{
    if (auto value = readDiagnosticBooleanOverride(
            "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC")) {
        return value->value;
    }
    if (auto value = readDiagnosticBooleanOverride(
            "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC")) {
        return value->value;
    }
    // Retain the equilibrium spelling as an explicit compatibility alias for
    // qualification scripts written while the diagnostic was being designed.
    if (auto value = readDiagnosticBooleanOverride(
            "SVMP_NS_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC")) {
        return value->value;
    }
    if (auto value = readDiagnosticBooleanOverride(
            "SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC")) {
        return value->value;
    }
    return false;
}

bool symmetricNitscheEnergyDiagnosticEnabled()
{
    if (auto value = readDiagnosticBooleanOverride(
            "SVMP_NS_SYMMETRIC_NITSCHE_ENERGY_DIAGNOSTIC")) {
        return value->value;
    }
    return false;
}

bool navierStokesPspgContinuityFullCellSupportDiagnosticEnabled()
{
    if (auto value = readDiagnosticBooleanOverride(
            "SVMP_NS_PSPG_CONTINUITY_FULL_CELL_SUPPORT")) {
        return value->value;
    }
    if (auto value = readDiagnosticBooleanOverride(
            "SVMP_PSPG_CONTINUITY_FULL_CELL_SUPPORT")) {
        return value->value;
    }
    return false;
}

std::optional<DiagnosticScalarOverride> readPositiveDiagnosticScalarOverride(
    const char* name)
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return std::nullopt;
    }
    try {
        const FE::Real parsed = static_cast<FE::Real>(std::stod(env));
        if (parsed > FE::Real{0.0} && std::isfinite(parsed)) {
            return DiagnosticScalarOverride{parsed, name, env};
        }
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

std::optional<DiagnosticScalarOverride> readNonnegativeDiagnosticScalarOverride(
    const char* name)
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return std::nullopt;
    }
    try {
        const FE::Real parsed = static_cast<FE::Real>(std::stod(env));
        if (parsed >= FE::Real{0.0} && std::isfinite(parsed)) {
            return DiagnosticScalarOverride{parsed, name, env};
        }
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

std::optional<DiagnosticScalarOverride> freeSurfacePressureReferenceProbe()
{
    if (auto value = readPositiveDiagnosticScalarOverride(
            "SVMP_NS_FREE_SURFACE_PRESSURE_REFERENCE_PROBE_PENALTY")) {
        return value;
    }
    return readPositiveDiagnosticScalarOverride(
        "SVMP_FREE_SURFACE_PRESSURE_REFERENCE_PROBE_PENALTY");
}

std::optional<DiagnosticScalarOverride> freeSurfaceTangentialPressureGradientProbe()
{
    if (auto value = readPositiveDiagnosticScalarOverride(
            "SVMP_NS_FREE_SURFACE_TANGENTIAL_PRESSURE_GRADIENT_SCALE")) {
        return value;
    }
    return readPositiveDiagnosticScalarOverride(
        "SVMP_FREE_SURFACE_TANGENTIAL_PRESSURE_GRADIENT_SCALE");
}

std::optional<DiagnosticScalarOverride> navierStokesPspgPressureGradientScale()
{
    if (auto value = readNonnegativeDiagnosticScalarOverride(
            "SVMP_NS_PSPG_PRESSURE_GRADIENT_SCALE")) {
        return value;
    }
    return readNonnegativeDiagnosticScalarOverride(
        "SVMP_PSPG_PRESSURE_GRADIENT_SCALE");
}

std::optional<DiagnosticScalarOverride> navierStokesPspgNonpressureResidualScale()
{
    if (auto value = readNonnegativeDiagnosticScalarOverride(
            "SVMP_NS_PSPG_NONPRESSURE_RESIDUAL_SCALE")) {
        return value;
    }
    return readNonnegativeDiagnosticScalarOverride(
        "SVMP_PSPG_NONPRESSURE_RESIDUAL_SCALE");
}

std::optional<DiagnosticScalarOverride>
navierStokesPspgPressureGradientCutVolumeScaleCap()
{
    if (auto value = readPositiveDiagnosticScalarOverride(
            "SVMP_NS_PSPG_PRESSURE_GRADIENT_CUT_VOLUME_SCALE_CAP")) {
        return value;
    }
    return readPositiveDiagnosticScalarOverride(
        "SVMP_PSPG_PRESSURE_GRADIENT_CUT_VOLUME_SCALE_CAP");
}

std::optional<DiagnosticScalarOverride>
navierStokesPspgBoundaryPressureGradientScale()
{
    if (auto value = readPositiveDiagnosticScalarOverride(
            "SVMP_NS_PSPG_BOUNDARY_PRESSURE_GRADIENT_SCALE")) {
        return value;
    }
    return readPositiveDiagnosticScalarOverride(
        "SVMP_PSPG_BOUNDARY_PRESSURE_GRADIENT_SCALE");
}

std::optional<DiagnosticScalarOverride>
navierStokesPspgBoundaryPressureFluxScale()
{
    if (auto value = readPositiveDiagnosticScalarOverride(
            "SVMP_NS_PSPG_BOUNDARY_PRESSURE_FLUX_SCALE")) {
        return value;
    }
    return readPositiveDiagnosticScalarOverride(
        "SVMP_PSPG_BOUNDARY_PRESSURE_FLUX_SCALE");
}

std::optional<DiagnosticScalarOverride>
navierStokesPspgBoundaryTangentialPressureGradientScale()
{
    if (auto value = readPositiveDiagnosticScalarOverride(
            "SVMP_NS_PSPG_BOUNDARY_TANGENTIAL_PRESSURE_GRADIENT_SCALE")) {
        return value;
    }
    return readPositiveDiagnosticScalarOverride(
        "SVMP_PSPG_BOUNDARY_TANGENTIAL_PRESSURE_GRADIENT_SCALE");
}

std::optional<DiagnosticScalarOverride>
navierStokesPspgBoundaryTangentialMomentumResidualScale()
{
    if (auto value = readPositiveDiagnosticScalarOverride(
            "SVMP_NS_PSPG_BOUNDARY_TANGENTIAL_MOMENTUM_RESIDUAL_SCALE")) {
        return value;
    }
    return readPositiveDiagnosticScalarOverride(
        "SVMP_PSPG_BOUNDARY_TANGENTIAL_MOMENTUM_RESIDUAL_SCALE");
}

std::optional<DiagnosticPspgPressureGradientFormOverride>
readPspgPressureGradientFormOverride(const char* name)
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return std::nullopt;
    }

    std::string normalized(env);
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });

    if (normalized == "absolute" || normalized == "state" ||
        normalized == "current") {
        return DiagnosticPspgPressureGradientFormOverride{
            PspgPressureGradientForm::Absolute,
            name,
            env};
    }
    if (normalized == "incremental" || normalized == "increment" ||
        normalized == "delta") {
        return DiagnosticPspgPressureGradientFormOverride{
            PspgPressureGradientForm::Incremental,
            name,
            env};
    }
    return std::nullopt;
}

std::optional<DiagnosticPspgPressureGradientFormOverride>
navierStokesPspgPressureGradientForm()
{
    if (auto value = readPspgPressureGradientFormOverride(
            "SVMP_NS_PSPG_PRESSURE_GRADIENT_FORM")) {
        return value;
    }
    return readPspgPressureGradientFormOverride(
        "SVMP_PSPG_PRESSURE_GRADIENT_FORM");
}

bool constrainInactiveActiveDomainVelocity()
{
    const char* env = std::getenv("SVMP_CONSTRAIN_INACTIVE_ACTIVE_DOMAIN_VELOCITY");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::string_view(env) != "0" &&
           std::string_view(env) != "false" &&
           std::string_view(env) != "FALSE";
}

bool constrainInactiveActiveDomainVelocity(
    const FreeSurfaceBoundary& active_domain_boundary)
{
    return constrainInactiveActiveDomainVelocity() &&
           !active_domain_boundary.velocity_extension.enabled;
}

bool unfittedLevelSetShapeTangentsDisabled()
{
    const char* enable_env =
        std::getenv("SVMP_ENABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS");
    if (enable_env != nullptr && enable_env[0] != '\0' &&
        std::string_view(enable_env) != "0" &&
        std::string_view(enable_env) != "false" &&
        std::string_view(enable_env) != "FALSE") {
        return false;
    }
    const char* disable_env =
        std::getenv("SVMP_DISABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS");
    if (disable_env != nullptr && disable_env[0] != '\0') {
        return std::string_view(disable_env) != "0" &&
               std::string_view(disable_env) != "false" &&
               std::string_view(disable_env) != "FALSE";
    }
    return true;
}

[[nodiscard]] bool spacesCompatible(const FE::spaces::FunctionSpace& lhs,
                                    const FE::spaces::FunctionSpace& rhs) noexcept
{
    return lhs.space_type() == rhs.space_type() &&
           lhs.field_type() == rhs.field_type() &&
           lhs.value_dimension() == rhs.value_dimension() &&
           lhs.topological_dimension() == rhs.topological_dimension() &&
           lhs.polynomial_order() == rhs.polynomial_order() &&
           lhs.element_type() == rhs.element_type();
}

[[nodiscard]] bool
isSupportedGeneratedBoundaryTraceSpace(
    const FE::spaces::FunctionSpace& space,
    int dimension) noexcept
{
    const auto element_type = space.element_type();
    return space.space_type() ==
               FE::spaces::SpaceType::Product &&
           space.continuity() == FE::Continuity::C0 &&
           space.polynomial_order() == 1 &&
           space.value_dimension() == dimension &&
           !space.element().basis().is_vector_valued() &&
           space.element().basis().size() ==
               static_cast<std::size_t>(dimension + 1) &&
           ((dimension == 2 &&
             element_type == FE::ElementType::Triangle3) ||
            (dimension == 3 &&
             element_type == FE::ElementType::Tetra4));
}

void validateCompatibleField(const FE::systems::FESystem& system,
                             const FE::systems::FieldSpec& spec,
                             FE::systems::FieldSourceKind expected_source,
                             bool allow_missing,
                             const char* context)
{
    if (spec.name.empty()) {
        throw std::invalid_argument(
            std::string(context) + ": field name must be non-empty");
    }
    if (!spec.space) {
        throw std::invalid_argument(
            std::string(context) + ": field '" + spec.name +
            "' requires a function space");
    }
    const auto existing = system.findFieldByName(spec.name);
    if (existing == FE::INVALID_FIELD_ID) {
        if (!allow_missing) {
            throw std::invalid_argument(
                std::string(context) + ": prescribed field '" + spec.name +
                "' was requested but is not registered");
        }
        return;
    }

    const auto& rec = system.fieldRecord(existing);
    if (rec.source_kind != expected_source) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            (expected_source == FE::systems::FieldSourceKind::Unknown
                 ? "' must be an unknown field"
                 : "' must be a prescribed data field"));
    }
    if (rec.components != spec.components) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' has component count " + std::to_string(rec.components) +
            ", expected " + std::to_string(spec.components));
    }
    if (!rec.space || !spacesCompatible(*rec.space, *spec.space)) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' uses an incompatible function space");
    }
}

[[nodiscard]] FE::FieldId ensureCompatibleUnknownField(
    FE::systems::FESystem& system,
    FE::systems::FieldSpec spec,
    const char* context)
{
    validateCompatibleField(
        system,
        spec,
        FE::systems::FieldSourceKind::Unknown,
        /*allow_missing=*/true,
        context);
    const auto existing = system.findFieldByName(spec.name);
    if (existing == FE::INVALID_FIELD_ID) {
        return system.addField(std::move(spec));
    }
    return existing;
}

[[nodiscard]] FE::FieldId ensureCompatiblePrescribedField(
    FE::systems::FESystem& system,
    FE::systems::FieldSpec spec,
    bool auto_register,
    const char* context)
{
    validateCompatibleField(
        system,
        spec,
        FE::systems::FieldSourceKind::PrescribedData,
        /*allow_missing=*/auto_register,
        context);
    const auto existing = system.findFieldByName(spec.name);
    if (existing == FE::INVALID_FIELD_ID) {
        return system.addField(std::move(spec));
    }
    return existing;
}

void requireDistinctFieldNames(
    std::initializer_list<std::pair<std::string_view, std::string_view>> fields)
{
    std::unordered_map<std::string_view, std::string_view> roles;
    for (const auto& [role, name] : fields) {
        if (name.empty()) {
            continue;
        }
        const auto [existing, inserted] = roles.emplace(name, role);
        if (!inserted) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: field name '" +
                std::string(name) + "' is assigned to both " +
                std::string(existing->second) + " and " + std::string(role));
        }
    }
}

constexpr FE::Real kPressureGaugeLevelSetMargin = 1.0e-8;

struct ActivePressureDomain {
    const FreeSurfaceBoundary* boundary{nullptr};
    FreeSurfaceActiveDomain active_domain{FreeSurfaceActiveDomain::None};
};

struct LevelSetVertexFieldView {
    const FE::Real* values{nullptr};
    std::size_t components{0};
    std::size_t entity_count{0};
};

struct VertexScalarFieldView {
    const FE::Real* values{nullptr};
    std::size_t components{0};
    std::size_t entity_count{0};
};

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] std::optional<ActivePressureDomain>
activePressureDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    std::optional<ActivePressureDomain> active_domain;
    for (const auto& bc : free_surfaces) {
        if (bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        if (bc.implementation != FreeSurfaceImplementation::UnfittedLevelSet) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure operations are only valid for unfitted level-set free surfaces");
        }
        if (bc.level_set_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure operations require a non-empty level_set_field_name");
        }
        if (active_domain.has_value()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: at most one active-domain free surface may restrict pressure operations");
        }
        active_domain = ActivePressureDomain{&bc, bc.active_domain};
    }
    return active_domain;
}

[[nodiscard]] LevelSetVertexFieldView activePressureLevelSetField(
    const FE::systems::FESystem& system,
    const ActivePressureDomain& active_domain,
    FE::GlobalIndex n_vertices,
    std::string_view context)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires a native mesh vertex level-set field");
    }

    const auto& local_mesh = native_mesh->local_mesh();
    const auto& bc = *active_domain.boundary;
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex,
                               bc.level_set_field_name)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " could not find vertex level-set field '" +
            bc.level_set_field_name + "'");
    }

    const auto handle = MeshFields::get_field_handle(
        local_mesh, EntityKind::Vertex, bc.level_set_field_name);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires a Float64 vertex level-set field");
    }

    const auto components = MeshFields::field_components(local_mesh, handle);
    if (components < 1u) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires at least one level-set component");
    }

    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " level-set field has fewer entries than pressure vertices");
    }

    const auto* values = MeshFields::field_data_as<FE::Real>(local_mesh, handle);
    if (values == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " found an empty level-set field");
    }

    return LevelSetVertexFieldView{values, components, entity_count};
#else
    (void)system;
    (void)active_domain;
    (void)n_vertices;
    (void)context;
    throw std::runtime_error(
        "IncompressibleNavierStokesVMSModule: active-domain pressure operations require native mesh support");
#endif
}

[[nodiscard]] VertexScalarFieldView pressureInitializationField(
    const FE::systems::FESystem& system,
    std::string_view field_name,
    FE::GlobalIndex n_vertices)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires a native mesh");
    }

    const auto& local_mesh = native_mesh->local_mesh();
    const std::string field_name_string(field_name);
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex, field_name_string)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization could not find vertex field '" +
            field_name_string + "'");
    }

    const auto handle = MeshFields::get_field_handle(local_mesh, EntityKind::Vertex, field_name_string);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires a Float64 vertex field");
    }

    const auto components = MeshFields::field_components(local_mesh, handle);
    if (components < 1u) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires at least one component");
    }

    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization has fewer entries than pressure vertices");
    }

    const auto* values = MeshFields::field_data_as<FE::Real>(local_mesh, handle);
    if (values == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization found an empty vertex field");
    }

    return VertexScalarFieldView{values, components, entity_count};
#else
    (void)system;
    (void)field_name;
    (void)n_vertices;
    throw std::runtime_error(
        "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires native mesh support");
#endif
}

std::size_t initializeStateFieldFromMeshVertexField(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0,
    FE::FieldId field_id,
    std::string_view context)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (field_id == FE::INVALID_FIELD_ID) {
        return 0u;
    }

    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        return 0u;
    }

    const auto& rec = system.fieldRecord(field_id);
    const auto& local_mesh = native_mesh->local_mesh();
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex, rec.name)) {
        return 0u;
    }

    const auto handle =
        MeshFields::get_field_handle(local_mesh, EntityKind::Vertex, rec.name);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires a Float64 vertex mesh field '" + rec.name + "'");
    }

    const auto components = static_cast<std::size_t>(std::max(1, rec.components));
    const auto mesh_components = MeshFields::field_components(local_mesh, handle);
    if (mesh_components < components) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " mesh field '" + rec.name + "' has fewer components than the FE field");
    }

    const auto n_vertices = static_cast<FE::GlobalIndex>(native_mesh->n_vertices());
    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " mesh field '" + rec.name + "' has fewer entries than mesh vertices");
    }

    const auto* mesh_values = MeshFields::field_data_as<FE::Real>(local_mesh, handle);
    if (mesh_values == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " found an empty mesh field '" + rec.name + "'");
    }

    const auto& field_dofs = system.fieldDofHandler(field_id);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr || entity_map->numVertices() < n_vertices) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires FE vertex DOF metadata for field '" + rec.name + "'");
    }

    const auto field_offset = system.fieldDofOffset(field_id);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    std::vector<FE::Real> coefficients(n_field_dofs, FE::Real{0.0});
    std::vector<std::uint8_t> assigned(n_field_dofs, 0u);
    const auto projection =
        system.projectMeshVertexValuesToFieldCoefficients(
            field_id,
            std::span<const FE::Real>(
                reinterpret_cast<const FE::Real*>(mesh_values),
                entity_count * mesh_components),
            mesh_components,
            std::span<FE::Real>(coefficients.data(), coefficients.size()),
            std::span<std::uint8_t>(assigned.data(), assigned.size()),
            std::string("IncompressibleNavierStokesVMSModule: ") +
                std::string(context));
    if (projection.unassigned_dofs != 0u) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " could not safely project " +
            std::to_string(projection.unassigned_dofs) +
            " field coefficient(s) from mesh vertices for field '" +
            rec.name + "'");
    }

    std::vector<FE::GlobalIndex> dofs;
    std::vector<FE::Real> values;
    dofs.reserve(n_field_dofs);
    values.reserve(n_field_dofs);

    for (std::size_t dof = 0; dof < coefficients.size(); ++dof) {
        if (assigned[dof] == 0u) {
            continue;
        }
        dofs.push_back(field_offset + static_cast<FE::GlobalIndex>(dof));
        values.push_back(coefficients[dof]);
    }

    if (dofs.empty()) {
        return 0u;
    }

    auto view = u0.createAssemblyView();
    if (!view) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " could not create a vector view");
    }
    view->beginAssemblyPhase();
    view->setVectorEntries(dofs, values);
    view->endAssemblyPhase();
    view->finalizeAssembly();
    return dofs.size();
#else
    (void)system;
    (void)u0;
    (void)field_id;
    (void)context;
    return 0u;
#endif
}

[[nodiscard]] bool pressureVertexOnActiveSide(
    FE::Real phi,
    FE::Real isovalue,
    FreeSurfaceActiveDomain active_domain) noexcept
{
    switch (active_domain) {
    case FreeSurfaceActiveDomain::None:
        return true;
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return phi <= isovalue;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return phi >= isovalue;
    }
    return true;
}

[[nodiscard]] FE::geometry::CutIntegrationSide activeDomainIntegrationSide(
    FreeSurfaceActiveDomain active_domain) noexcept
{
    switch (active_domain) {
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return FE::geometry::CutIntegrationSide::Negative;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return FE::geometry::CutIntegrationSide::Positive;
    case FreeSurfaceActiveDomain::None:
        break;
    }
    return FE::geometry::CutIntegrationSide::Negative;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] bool pressureCellHasActiveMeasureBySign(
    const LevelSetVertexFieldView& level_set_values,
    FreeSurfaceActiveDomain active_domain,
    FE::Real isovalue,
    const index_t* vertices,
    std::size_t vertex_count)
{
    if (vertices == nullptr || vertex_count == 0u) {
        return false;
    }
    for (std::size_t i = 0; i < vertex_count; ++i) {
        const auto vertex = static_cast<FE::GlobalIndex>(vertices[i]);
        if (vertex < 0 ||
            static_cast<std::size_t>(vertex) >= level_set_values.entity_count) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain "
                << "pressure initialization references vertex " << vertex
                << " outside the level-set field";
            throw std::runtime_error(oss.str());
        }
        const auto phi = level_set_values.values[
            static_cast<std::size_t>(vertex) * level_set_values.components];
        const bool has_positive_active_measure =
            active_domain == FreeSurfaceActiveDomain::LevelSetNegative
                ? phi < isovalue
                : phi > isovalue;
        if (has_positive_active_measure) {
            return true;
        }
    }
    return false;
}
#endif

[[nodiscard]] std::vector<unsigned char> activePressureSupportVertices(
    const FE::systems::FESystem& system,
    const ActivePressureDomain& active_pressure_domain,
    const LevelSetVertexFieldView& level_set_values,
    FE::GlobalIndex n_vertices)
{
    std::vector<unsigned char> active_support(
        static_cast<std::size_t>(n_vertices), static_cast<unsigned char>(0));

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
            const auto phi = level_set_values.values[
                static_cast<std::size_t>(vertex) * level_set_values.components];
            active_support[static_cast<std::size_t>(vertex)] =
                pressureVertexOnActiveSide(
                    phi,
                    active_pressure_domain.boundary->level_set_isovalue,
                    active_pressure_domain.active_domain)
                    ? static_cast<unsigned char>(1)
                    : static_cast<unsigned char>(0);
        }
        return active_support;
    }

    const auto& mesh = native_mesh->local_mesh();
    const auto mark_cell_active = [&](FE::GlobalIndex cell) {
        if (cell < 0 || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain "
                << "pressure initialization references cell " << cell
                << " outside the mesh";
            throw std::runtime_error(oss.str());
        }
        const auto [vertices, count] =
            mesh.cell_vertices_span(static_cast<index_t>(cell));
        if (vertices == nullptr || count == 0u) {
            return;
        }
        for (std::size_t i = 0; i < count; ++i) {
            const auto vertex = static_cast<FE::GlobalIndex>(vertices[i]);
            if (vertex >= 0 && vertex < n_vertices) {
                active_support[static_cast<std::size_t>(vertex)] =
                    static_cast<unsigned char>(1);
            }
        }
    };

    const auto* cut_context = system.cutIntegrationContext();
    if (cut_context != nullptr) {
        cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
    }
    const auto& bc = *active_pressure_domain.boundary;
    if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::CutVolume &&
        bc.interface_marker >= 0 &&
        cut_context != nullptr &&
        cut_context->hasGeneratedVolumeMarker(bc.interface_marker)) {
        const auto rule_indices =
            cut_context->generatedVolumeRuleIndexSpanForMarkerAndSide(
                bc.interface_marker,
                activeDomainIntegrationSide(active_pressure_domain.active_domain));
        const auto& metadata = cut_context->metadata();
        for (const auto index : rule_indices) {
            if (index >= metadata.size()) {
                continue;
            }
            const auto& rule_metadata = metadata[index];
            const auto cell = rule_metadata.parent_entity >= 0
                                  ? rule_metadata.parent_entity
                                  : rule_metadata.cell;
            mark_cell_active(static_cast<FE::GlobalIndex>(cell));
        }
    } else {
        for (FE::GlobalIndex cell = 0;
             cell < static_cast<FE::GlobalIndex>(mesh.n_cells());
             ++cell) {
            const auto [vertices, count] =
                mesh.cell_vertices_span(static_cast<index_t>(cell));
            if (pressureCellHasActiveMeasureBySign(
                    level_set_values,
                    active_pressure_domain.active_domain,
                    bc.level_set_isovalue,
                    vertices,
                    count)) {
                mark_cell_active(cell);
            }
        }
    }
#else
    for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto phi = level_set_values.values[
            static_cast<std::size_t>(vertex) * level_set_values.components];
        active_support[static_cast<std::size_t>(vertex)] =
            pressureVertexOnActiveSide(
                phi,
                active_pressure_domain.boundary->level_set_isovalue,
                active_pressure_domain.active_domain)
                ? static_cast<unsigned char>(1)
                : static_cast<unsigned char>(0);
    }
#endif

    return active_support;
}

[[nodiscard]] FE::Real hydrostaticPressureAt(
    const std::array<FE::Real, 3>& x,
    const IncompressibleNavierStokesVMSOptions& options,
    const IncompressibleNavierStokesVMSOptions::
        HydrostaticPressureInitialization& init) noexcept
{
    FE::Real pressure = init.reference_pressure;
    for (std::size_t d = 0; d < options.body_force.size(); ++d) {
        pressure += options.density * options.body_force[d] *
                    (x[d] - init.reference_point[d]);
    }
    return pressure;
}

[[nodiscard]] std::optional<FE::GlobalIndex> pressureConstraintLocalVertex(
    const FE::systems::FESystem& system,
    IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType id_type,
    FE::GlobalIndex node_id)
{
    if (node_id < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: pressure constraint references a negative node id");
    }

    switch (id_type) {
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::LocalVertexId:
        return node_id;
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::GlobalVertexGid:
        break;
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: active-domain pressure constraint validation requires a native mesh for global vertex ids");
    }
    const auto local_vertex =
        native_mesh->local_mesh().global_to_local_vertex(static_cast<gid_t>(node_id));
    if (local_vertex == INVALID_INDEX) {
        return std::nullopt;
    }
    return static_cast<FE::GlobalIndex>(local_vertex);
#else
    (void)system;
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: active-domain pressure constraint validation requires native mesh support for global vertex ids");
#endif
}

void validateActiveDomainPressureConstraints(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    if (options.node_pressure_constraints.values.empty()) {
        return;
    }

    std::unordered_map<FE::GlobalIndex, FE::Real> pressure_by_node;
    const auto n_vertices = system.meshAccess().numVertices();
    for (const auto& constraint : options.node_pressure_constraints.values) {
        if (constraint.node_id < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint references a negative node id");
        }
        if (!std::isfinite(constraint.pressure)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint value must be finite");
        }
        const auto [existing, inserted] = pressure_by_node.emplace(
            constraint.node_id, constraint.pressure);
        if (!inserted && existing->second != constraint.pressure) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: duplicate pressure constraint node has conflicting values");
        }
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options.node_pressure_constraints.id_type,
            constraint.node_id);
        if (local_vertex.has_value() &&
            (*local_vertex < 0 || *local_vertex >= n_vertices)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint references vertex " +
                std::to_string(constraint.node_id) +
                " outside the local pressure mesh");
        }
    }

    const auto active_pressure_domain = activePressureDomainFor(free_surfaces);
    if (!active_pressure_domain.has_value()) {
        return;
    }

    const auto level_set_values = activePressureLevelSetField(
        system,
        *active_pressure_domain,
        n_vertices,
        "active-domain pressure constraint validation");
    const auto& bc = *active_pressure_domain->boundary;
    std::size_t checked_local_constraints = 0u;
    std::size_t skipped_nonlocal_constraints = 0u;
    FE::Real constraint_pressure_min = std::numeric_limits<FE::Real>::infinity();
    FE::Real constraint_pressure_max = -std::numeric_limits<FE::Real>::infinity();
    FE::Real min_signed_gap = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_signed_gap = -std::numeric_limits<FE::Real>::infinity();
    for (const auto& constraint : options.node_pressure_constraints.values) {
        constraint_pressure_min = std::min(constraint_pressure_min, constraint.pressure);
        constraint_pressure_max = std::max(constraint_pressure_max, constraint.pressure);
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options.node_pressure_constraints.id_type,
            constraint.node_id);
        if (!local_vertex.has_value()) {
            ++skipped_nonlocal_constraints;
            continue;
        }
        if (*local_vertex < 0 || *local_vertex >= n_vertices) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure constraint references vertex " +
                std::to_string(constraint.node_id) +
                " outside the local pressure mesh");
        }

        const auto phi = level_set_values.values[
            static_cast<std::size_t>(*local_vertex) * level_set_values.components];
        const auto signed_gap = phi - bc.level_set_isovalue;
        ++checked_local_constraints;
        min_signed_gap = std::min(min_signed_gap, signed_gap);
        max_signed_gap = std::max(max_signed_gap, signed_gap);
        if (!pressureVertexOnActiveSide(phi,
                                        bc.level_set_isovalue,
                                        active_pressure_domain->active_domain)) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain pressure "
                << "constraint node_id=" << constraint.node_id
                << " local_vertex=" << *local_vertex
                << " is on the dry side for Active_domain="
                << pressureActiveDomainName(active_pressure_domain->active_domain)
                << " phi=" << phi
                << " isovalue=" << bc.level_set_isovalue;
            throw std::invalid_argument(oss.str());
        }
        if (std::abs(signed_gap) < kPressureGaugeLevelSetMargin) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain pressure "
                << "constraint node_id=" << constraint.node_id
                << " local_vertex=" << *local_vertex
                << " is too close to the level-set interface: |phi-isovalue|="
                << std::abs(signed_gap)
                << " margin=" << kPressureGaugeLevelSetMargin;
            throw std::invalid_argument(oss.str());
        }
    }
    if (!std::isfinite(min_signed_gap)) {
        min_signed_gap = FE::Real{0.0};
    }
    if (!std::isfinite(max_signed_gap)) {
        max_signed_gap = FE::Real{0.0};
    }
    if (!std::isfinite(constraint_pressure_min)) {
        constraint_pressure_min = FE::Real{0.0};
    }
    if (!std::isfinite(constraint_pressure_max)) {
        constraint_pressure_max = FE::Real{0.0};
    }
    std::ostringstream oss;
    oss << "IncompressibleNavierStokesVMSModule: pressure gauge diagnostic"
        << " diagnostic=pressure_gauge_check"
        << " constraints=" << options.node_pressure_constraints.values.size()
        << " checked_local_constraints=" << checked_local_constraints
        << " skipped_nonlocal_constraints=" << skipped_nonlocal_constraints
        << " constraint_pressure_min=" << constraint_pressure_min
        << " constraint_pressure_max=" << constraint_pressure_max
        << " Active_domain="
        << pressureActiveDomainName(active_pressure_domain->active_domain)
        << " isovalue=" << bc.level_set_isovalue
        << " min_signed_gap=" << min_signed_gap
        << " max_signed_gap=" << max_signed_gap
        << " margin=" << kPressureGaugeLevelSetMargin;
    FE_LOG_INFO(oss.str());
}

} // namespace

void IncompressibleNavierStokesVMSModule::applyInitialConditions(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0) const
{
    const auto& init = options_.hydrostatic_pressure_initialization;

    std::size_t mesh_field_initialization_dofs = 0u;
    mesh_field_initialization_dofs += initializeStateFieldFromMeshVertexField(
        system,
        u0,
        system.findFieldByName(options_.velocity_field_name),
        "mesh-field velocity initialization");

    if (!init.enabled) {
        mesh_field_initialization_dofs += initializeStateFieldFromMeshVertexField(
            system,
            u0,
            system.findFieldByName(options_.pressure_field_name),
            "mesh-field pressure initialization");
        if (mesh_field_initialization_dofs > 0u) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: mesh-field "
                   "initialization diagnostic=mesh_field_initialization"
                << " initialized_dofs=" << mesh_field_initialization_dofs
                << " velocity_field='" << options_.velocity_field_name << "'"
                << " pressure_field='" << options_.pressure_field_name << "'";
            FE_LOG_INFO(oss.str());
        }
        return;
    }

    if (mesh_field_initialization_dofs > 0u) {
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: mesh-field "
               "initialization diagnostic=mesh_field_initialization"
            << " initialized_dofs=" << mesh_field_initialization_dofs
            << " velocity_field='" << options_.velocity_field_name << "'";
        FE_LOG_INFO(oss.str());
    }

    const auto p_id = system.findFieldByName(options_.pressure_field_name);
    if (p_id == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization could not find pressure field '" +
            options_.pressure_field_name + "'");
    }

    const auto& pressure_dofs = system.fieldDofHandler(p_id);
    const auto* entity_map = pressure_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization requires vertex DOF metadata");
    }

    const auto& mesh = system.meshAccess();
    const auto pressure_offset = system.fieldDofOffset(p_id);
    const auto n_vertices = mesh.numVertices();
    const auto active_pressure_domain =
        activePressureDomainFor(options_.free_surface);
    if (active_pressure_domain.has_value() &&
        active_pressure_domain->boundary->active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume &&
        init.field_name.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: CutVolume active-domain "
            "hydrostatic pressure initialization requires a level-set-aware "
            "mesh vertex field selected by Hydrostatic_pressure_field_name");
    }
    std::optional<LevelSetVertexFieldView> level_set_values;
    if (active_pressure_domain.has_value()) {
        level_set_values =
            activePressureLevelSetField(system, *active_pressure_domain,
                                        n_vertices,
                                        "active-domain hydrostatic pressure initialization");
    }
    std::optional<std::vector<unsigned char>> active_pressure_support;
    if (active_pressure_domain.has_value() && level_set_values.has_value()) {
        active_pressure_support =
            activePressureSupportVertices(
                system, *active_pressure_domain, *level_set_values, n_vertices);
    }
    std::optional<VertexScalarFieldView> pressure_initialization_field;
    if (!init.field_name.empty()) {
        pressure_initialization_field =
            pressureInitializationField(system, init.field_name, n_vertices);
        for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
            const auto value = pressure_initialization_field->values[
                static_cast<std::size_t>(vertex) *
                pressure_initialization_field->components];
            if (!std::isfinite(value)) {
                std::ostringstream oss;
                oss << "IncompressibleNavierStokesVMSModule: hydrostatic "
                       "pressure mesh vertex field '"
                    << init.field_name
                    << "' contains a non-finite value at vertex " << vertex;
                throw std::runtime_error(oss.str());
            }
        }
    }

    std::vector<FE::GlobalIndex> dofs;
    std::vector<FE::Real> values;
    std::size_t active_wet_vertices = 0u;
    std::size_t active_dry_vertices = 0u;
    std::size_t active_support_pressure_vertices = 0u;
    std::size_t dry_sign_active_support_pressure_vertices = 0u;
    FE::Real initialized_pressure_min =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real initialized_pressure_max =
        -std::numeric_limits<FE::Real>::infinity();
    FE::Real wet_pressure_min = std::numeric_limits<FE::Real>::infinity();
    FE::Real wet_pressure_max = -std::numeric_limits<FE::Real>::infinity();
    std::size_t checked_gauge_constraints = 0u;
    std::size_t skipped_gauge_constraints = 0u;
    FE::Real gauge_pressure_min = std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_pressure_max = -std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_initialized_pressure_min =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_initialized_pressure_max =
        -std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_pressure_max_abs_error = FE::Real{0.0};

    for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        if (vertex_dofs.empty()) {
            continue;
        }

        const auto x = mesh.getNodeCoordinates(vertex);
        bool initialize_hydrostatic = true;
        if (active_pressure_domain.has_value()) {
            const auto vertex_offset =
                static_cast<std::size_t>(vertex) * level_set_values->components;
            const auto phi = level_set_values->values[vertex_offset];
            const bool active_side = pressureVertexOnActiveSide(
                phi,
                active_pressure_domain->boundary->level_set_isovalue,
                active_pressure_domain->active_domain);
            initialize_hydrostatic =
                active_pressure_support.has_value()
                    ? ((*active_pressure_support)[static_cast<std::size_t>(vertex)] !=
                       static_cast<unsigned char>(0))
                    : active_side;
            if (active_side) {
                ++active_wet_vertices;
            } else {
                ++active_dry_vertices;
            }
            if (initialize_hydrostatic) {
                ++active_support_pressure_vertices;
                if (!active_side) {
                    ++dry_sign_active_support_pressure_vertices;
                }
            }
        }
        const FE::Real pressure = initialize_hydrostatic
            ? (pressure_initialization_field.has_value()
                   ? pressure_initialization_field->values[
                         static_cast<std::size_t>(vertex) *
                         pressure_initialization_field->components]
                   : hydrostaticPressureAt(x, options_, init))
            : init.reference_pressure;
        if (!std::isfinite(pressure)) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: hydrostatic pressure "
                   "initialization produced a non-finite value at vertex "
                << vertex;
            if (!init.field_name.empty()) {
                oss << " from mesh vertex field '" << init.field_name << "'";
            }
            throw std::runtime_error(oss.str());
        }
        initialized_pressure_min = std::min(initialized_pressure_min, pressure);
        initialized_pressure_max = std::max(initialized_pressure_max, pressure);
        if (initialize_hydrostatic) {
            wet_pressure_min = std::min(wet_pressure_min, pressure);
            wet_pressure_max = std::max(wet_pressure_max, pressure);
        }

        for (const auto local_dof : vertex_dofs) {
            dofs.push_back(pressure_offset + local_dof);
            values.push_back(pressure);
        }
    }

    for (const auto& constraint : options_.node_pressure_constraints.values) {
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options_.node_pressure_constraints.id_type,
            constraint.node_id);
        if (!local_vertex.has_value()) {
            ++skipped_gauge_constraints;
            continue;
        }
        if (*local_vertex < 0 || *local_vertex >= n_vertices) {
            ++skipped_gauge_constraints;
            continue;
        }

        const auto vertex = *local_vertex;
        const auto x = mesh.getNodeCoordinates(vertex);
        bool initialize_hydrostatic = true;
        if (active_pressure_domain.has_value()) {
            const auto vertex_offset =
                static_cast<std::size_t>(vertex) * level_set_values->components;
            const auto phi = level_set_values->values[vertex_offset];
            const bool active_side = pressureVertexOnActiveSide(
                phi,
                active_pressure_domain->boundary->level_set_isovalue,
                active_pressure_domain->active_domain);
            initialize_hydrostatic =
                active_pressure_support.has_value()
                    ? ((*active_pressure_support)[static_cast<std::size_t>(vertex)] !=
                       static_cast<unsigned char>(0))
                    : active_side;
        }
        const FE::Real initialized_pressure = initialize_hydrostatic
            ? (pressure_initialization_field.has_value()
                   ? pressure_initialization_field->values[
                         static_cast<std::size_t>(vertex) *
                         pressure_initialization_field->components]
                   : hydrostaticPressureAt(x, options_, init))
            : init.reference_pressure;

        ++checked_gauge_constraints;
        gauge_pressure_min = std::min(gauge_pressure_min, constraint.pressure);
        gauge_pressure_max = std::max(gauge_pressure_max, constraint.pressure);
        gauge_initialized_pressure_min =
            std::min(gauge_initialized_pressure_min, initialized_pressure);
        gauge_initialized_pressure_max =
            std::max(gauge_initialized_pressure_max, initialized_pressure);
        gauge_pressure_max_abs_error =
            std::max(gauge_pressure_max_abs_error,
                     std::abs(initialized_pressure - constraint.pressure));
    }

    if (dofs.empty()) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization found no pressure vertex DOFs");
    }

    auto view = u0.createAssemblyView();
    if (!view) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization could not create a vector view");
    }
    view->beginAssemblyPhase();
    view->setVectorEntries(dofs, values);
    view->endAssemblyPhase();
    view->finalizeAssembly();

    if (active_pressure_domain.has_value()) {
        if (!std::isfinite(initialized_pressure_min)) {
            initialized_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(initialized_pressure_max)) {
            initialized_pressure_max = FE::Real{0.0};
        }
        if (!std::isfinite(wet_pressure_min)) {
            wet_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(wet_pressure_max)) {
            wet_pressure_max = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_pressure_min)) {
            gauge_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_pressure_max)) {
            gauge_pressure_max = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_initialized_pressure_min)) {
            gauge_initialized_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_initialized_pressure_max)) {
            gauge_initialized_pressure_max = FE::Real{0.0};
        }
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: hydrostatic pressure "
            << "initialization diagnostic=hydrostatic_initialization Active_domain="
            << pressureActiveDomainName(active_pressure_domain->active_domain)
            << " wet_pressure_vertices=" << active_wet_vertices
            << " dry_pressure_vertices=" << active_dry_vertices
            << " active_support_pressure_vertices="
            << active_support_pressure_vertices
            << " dry_sign_active_support_pressure_vertices="
            << dry_sign_active_support_pressure_vertices
            << " reference_pressure=" << init.reference_pressure
            << " initialized_pressure_min=" << initialized_pressure_min
            << " initialized_pressure_max=" << initialized_pressure_max
            << " wet_pressure_min=" << wet_pressure_min
            << " wet_pressure_max=" << wet_pressure_max
            << " gauge_constraints="
            << options_.node_pressure_constraints.values.size()
            << " checked_gauge_constraints=" << checked_gauge_constraints
            << " skipped_gauge_constraints=" << skipped_gauge_constraints
            << " gauge_pressure_min=" << gauge_pressure_min
            << " gauge_pressure_max=" << gauge_pressure_max
            << " gauge_initialized_pressure_min="
            << gauge_initialized_pressure_min
            << " gauge_initialized_pressure_max="
            << gauge_initialized_pressure_max
            << " gauge_pressure_max_abs_error="
            << gauge_pressure_max_abs_error;
        if (!init.field_name.empty()) {
            oss << " pressure_field='" << init.field_name << "'";
        }
        FE_LOG_INFO(oss.str());
    }
}

namespace {

[[nodiscard]] bool isUnfittedLevelSet(const FreeSurfaceBoundary& bc) noexcept
{
    return bc.implementation == FreeSurfaceImplementation::UnfittedLevelSet;
}

[[nodiscard]] bool usesSurfaceStress(
    const FreeSurfaceBoundary& bc) noexcept
{
    switch (bc.surface_tension_form) {
    case FreeSurfaceSurfaceTensionForm::Automatic:
        // The generated-interface rule supplies both dI and its geometric
        // normal, so the unfitted default can be the variation of one
        // discrete polygonal surface energy.  Preserve the historical fitted
        // supplied-curvature contract unless SurfaceStress is requested.
        return isUnfittedLevelSet(bc);
    case FreeSurfaceSurfaceTensionForm::CurvatureTraction:
        return false;
    case FreeSurfaceSurfaceTensionForm::SurfaceStress:
        return true;
    }
    return false;
}

[[nodiscard]] const char* surfaceTensionFormName(
    const FreeSurfaceBoundary& bc) noexcept
{
    return usesSurfaceStress(bc) ? "SurfaceStress" : "CurvatureTraction";
}

[[nodiscard]] bool surfaceStressActive(
    const FreeSurfaceBoundary& bc) noexcept
{
    return usesSurfaceStress(bc) &&
           !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension);
}

/**
 * Normal from the same geometric rule that supplies dI/dInterfaceBoundary.
 * Generated interface normals point from phi<0 to phi>0.  Flip only where a
 * directed outward-liquid normal is required; the surface projector itself
 * is orientation independent.
 */
[[nodiscard]] FE::forms::FormExpr generatedInterfaceOutwardNormal(
    const FreeSurfaceBoundary& bc)
{
    auto n = FE::forms::FormExpr::normal();
    if (isUnfittedLevelSet(bc) &&
        bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        n = -n;
    }
    return n;
}

[[nodiscard]] int freeSurfaceMarker(const FreeSurfaceBoundary& bc)
{
    return isUnfittedLevelSet(bc) ? bc.interface_marker : bc.boundary_marker;
}

[[nodiscard]] bool useFittedCurrentGeometry(const FreeSurfaceBoundary& bc,
                                            bool ale_enabled) noexcept
{
    return ale_enabled && !isUnfittedLevelSet(bc);
}

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc)
{
    const char* kind = isUnfittedLevelSet(bc) ? "_i" : "_b";
    return std::string(prefix) + kind + std::to_string(freeSurfaceMarker(bc));
}

enum class ContactLineKind : std::uint8_t {
    None,
    Pinned,
    PrescribedAngle,
    DynamicRenE
};

[[nodiscard]] ContactLineKind contactLineKind(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    if (std::holds_alternative<FreeSurfaceContactLine::Pinned>(
            contact_line.configuration)) {
        return ContactLineKind::Pinned;
    }
    if (std::holds_alternative<FreeSurfaceContactLine::PrescribedAngle>(
            contact_line.configuration)) {
        return ContactLineKind::PrescribedAngle;
    }
    if (std::holds_alternative<FreeSurfaceContactLine::DynamicRenE>(
            contact_line.configuration)) {
        return ContactLineKind::DynamicRenE;
    }
    return ContactLineKind::None;
}

[[nodiscard]] int contactLineWallBoundaryMarker(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept -> int {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<Configuration,
                                         FreeSurfaceContactLine::None>) {
                return -1;
            } else {
                return configuration.wall_boundary_marker;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] int contactLineMarker(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept -> int {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<Configuration,
                                         FreeSurfaceContactLine::None>) {
                return -1;
            } else {
                return configuration.contact_line_marker;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] const std::array<IncompressibleNavierStokesVMSOptions::ScalarValue, 3>&
contactLineWallNormal(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const std::array<
                IncompressibleNavierStokesVMSOptions::ScalarValue, 3>& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle> ||
                          std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.wall_normal;
            } else {
                throw std::logic_error(
                    "contactLineWallNormal requires an angle-bearing contact configuration");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineAngleRadians(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const IncompressibleNavierStokesVMSOptions::ScalarValue& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle>) {
                return configuration.contact_angle_radians;
            } else if constexpr (std::is_same_v<
                                     Configuration,
                                     FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.equilibrium_contact_angle_radians;
            } else {
                throw std::logic_error(
                    "contactLineAngleRadians requires an angle-bearing contact configuration");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineMobility(const FreeSurfaceContactLine& contact_line)
{
    const auto* configuration =
        std::get_if<FreeSurfaceContactLine::DynamicRenE>(
            &contact_line.configuration);
    if (configuration == nullptr) {
        throw std::logic_error("contactLineMobility requires DynamicRenE");
    }
    return configuration->mobility;
}

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineSlipLength(const FreeSurfaceContactLine& contact_line)
{
    const auto* configuration =
        std::get_if<FreeSurfaceContactLine::DynamicRenE>(
            &contact_line.configuration);
    if (configuration == nullptr) {
        throw std::logic_error("contactLineSlipLength requires DynamicRenE");
    }
    return configuration->slip_length;
}

[[nodiscard]] int contactLineConstraintMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    return contactLineMarker(contact_line) >= 0
               ? contactLineMarker(contact_line)
               : contactLineWallBoundaryMarker(contact_line);
}

[[nodiscard]] FE::FieldId resolveLevelSetFieldId(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    const auto phi_id = system.findFieldByName(bc.level_set_field_name);
    if (phi_id == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unfitted free surface references unknown level-set field '" +
            bc.level_set_field_name + "'");
    }

    const auto& rec = system.fieldRecord(phi_id);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: level-set field '" +
            bc.level_set_field_name + "' must be scalar");
    }
    return phi_id;
}

[[nodiscard]] int generatedInterfaceMarkerFor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || bc.interface_marker >= 0) {
        return bc.interface_marker;
    }
    if (bc.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted free surface requires a non-empty generated_interface_domain_id");
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi_id);
    key.domain_id = bc.generated_interface_domain_id;
    key.isovalue = bc.level_set_isovalue;
    key.requested_marker = bc.interface_marker;
    return FE::interfaces::stableGeneratedInterfaceMarker(key);
}

[[nodiscard]] FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey
generatedUnfittedContactLineMarkerKey(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated contact marker key requires an unfitted free surface");
    }
    if (bc.interface_marker < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires resolved interface_marker >= 0");
    }
    if (bc.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires a non-empty generated_interface_domain_id");
    }
    if (contactLineWallBoundaryMarker(contact_line) < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle requires Contact_line_wall_marker or Contact_line_wall_face");
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi_id);
    key.domain_id = bc.generated_interface_domain_id;
    key.isovalue = bc.level_set_isovalue;
    key.interface_marker = bc.interface_marker;
    key.boundary_marker = contactLineWallBoundaryMarker(contact_line);
    key.requested_marker = contactLineMarker(contact_line);
    return key;
}

[[nodiscard]] int generatedUnfittedContactLineMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return contactLineConstraintMarker(contact_line);
    }
    return FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
        generatedUnfittedContactLineMarkerKey(contact_line, bc, system));
}

[[nodiscard]] FE::interfaces::GeneratedActiveBoundaryMarkerKey
generatedActiveBoundaryMarkerKey(
    const FreeSurfaceBoundary& bc,
    int physical_boundary_marker,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || bc.interface_marker < 0 ||
        physical_boundary_marker < 0 ||
        bc.active_domain == FreeSurfaceActiveDomain::None) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: sharp active-boundary marker requires a resolved unfitted active domain and physical boundary marker");
    }
    FE::interfaces::GeneratedActiveBoundaryMarkerKey key;
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(
        resolveLevelSetFieldId(bc, system));
    key.domain_id = bc.generated_interface_domain_id;
    key.isovalue = bc.level_set_isovalue;
    key.interface_marker = bc.interface_marker;
    key.boundary_marker = physical_boundary_marker;
    key.side = activeDomainIntegrationSide(bc.active_domain);
    return key;
}

[[nodiscard]] int generatedActiveBoundaryMarker(
    const FreeSurfaceBoundary& bc,
    int physical_boundary_marker,
    const FE::systems::FESystem& system)
{
    return FE::interfaces::stableGeneratedActiveBoundaryMarker(
        generatedActiveBoundaryMarkerKey(
            bc, physical_boundary_marker, system));
}

void validateGeneratedFreeSurfaceMarkerUniqueness(
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system)
{
    std::unordered_map<int, std::string> owners;
    const auto register_marker = [&](int marker,
                                     std::string key,
                                     std::string_view kind) {
        const auto [it, inserted] = owners.emplace(marker, std::move(key));
        if (!inserted) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: duplicate/colliding generated " +
                std::string(kind) + " marker " + std::to_string(marker) +
                " between keys '" + it->second +
                "' and another configured free-surface key; assign explicit unique markers");
        }
    };

    for (const auto& bc : boundaries) {
        if (!isUnfittedLevelSet(bc)) {
            continue;
        }
        const auto phi_id = resolveLevelSetFieldId(bc, system);
        FE::interfaces::GeneratedInterfaceMarkerKey interface_key{};
        interface_key.source =
            FE::interfaces::LevelSetInterfaceSource::fromField(phi_id);
        interface_key.domain_id = bc.generated_interface_domain_id;
        interface_key.isovalue = bc.level_set_isovalue;
        interface_key.requested_marker = bc.interface_marker;
        register_marker(
            bc.interface_marker,
            "interface:" + interface_key.stableKey(),
            "interface/contact");

        for (const auto& contact_line : bc.contact_lines) {
            if (contactLineKind(contact_line) !=
                    ContactLineKind::PrescribedAngle &&
                contactLineKind(contact_line) !=
                    ContactLineKind::DynamicRenE) {
                continue;
            }
            const auto key = generatedUnfittedContactLineMarkerKey(
                contact_line, bc, system);
            const auto marker =
                FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
                    key);
            register_marker(
                marker,
                "contact:" + key.stableKey(),
                "interface/contact");
        }
    }
}

[[nodiscard]] std::optional<int> sharpActiveBoundaryMarkerFor(
    int physical_boundary_marker,
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system)
{
    std::optional<int> marker;
    std::string owner_key;
    for (const auto& bc : boundaries) {
        if (!isUnfittedLevelSet(bc) ||
            bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        const auto key = generatedActiveBoundaryMarkerKey(
            bc, physical_boundary_marker, system);
        const int candidate =
            FE::interfaces::stableGeneratedActiveBoundaryMarker(key);
        if (!marker.has_value()) {
            marker = candidate;
            owner_key = key.stableKey();
            continue;
        }
        if (*marker != candidate || owner_key != key.stableKey()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: physical boundary marker " +
                std::to_string(physical_boundary_marker) +
                " is covered by multiple ambiguous unfitted active domains; select one explicit active-domain owner");
        }
    }
    return marker;
}

[[nodiscard]] FreeSurfaceBoundary withResolvedInterfaceMarker(
    FreeSurfaceBoundary bc,
    const FE::systems::FESystem& system)
{
    if (isUnfittedLevelSet(bc) && bc.interface_marker < 0) {
        bc.interface_marker = generatedInterfaceMarkerFor(bc, system);
    }
    return bc;
}

[[nodiscard]] FE::Real constantScalarValueOrThrow(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    const auto* real = std::get_if<FE::Real>(&value);
    if (real == nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " currently requires a literal scalar value");
    }
    return *real;
}

[[nodiscard]] bool hasWallAwareContactLaw(
    const FreeSurfaceBoundary& boundary)
{
    return std::any_of(
        boundary.contact_lines.begin(),
        boundary.contact_lines.end(),
        [](const auto& contact_line) {
            const auto kind = contactLineKind(contact_line);
            return kind == ContactLineKind::PrescribedAngle ||
                   kind == ContactLineKind::DynamicRenE;
        });
}

[[nodiscard]] bool shouldDeclareFreeSurfaceDiscreteFunctional(
    const FreeSurfaceBoundary& boundary)
{
    return isUnfittedLevelSet(boundary) &&
           boundary.active_domain != FreeSurfaceActiveDomain::None &&
           (usesSurfaceStress(boundary) ||
            hasWallAwareContactLaw(boundary));
}

void validateFreeSurfaceDiscreteFunctionalPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system)
{
    const bool has_pending_declaration = std::any_of(
        free_surfaces.begin(),
        free_surfaces.end(),
        shouldDeclareFreeSurfaceDiscreteFunctional);
    if (!has_pending_declaration) {
        return;
    }
    if (!system.freeSurfaceDiscreteFunctionalHistory().empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free-surface discrete-functional declarations cannot be installed after accepted history has begun");
    }

    const auto existing =
        system.freeSurfaceDiscreteFunctionalDeclarations();
    for (const auto& boundary : free_surfaces) {
        if (!shouldDeclareFreeSurfaceDiscreteFunctional(boundary)) {
            continue;
        }
        const auto conflict = std::find_if(
            existing.begin(),
            existing.end(),
            [&](const auto& declaration) {
                return declaration.interface_marker ==
                       boundary.interface_marker;
            });
        if (conflict != existing.end()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: free-surface interface marker " +
                std::to_string(boundary.interface_marker) +
                " already has discrete-functional owner '" +
                conflict->owner_component + "'");
        }
    }
}

void declareFreeSurfaceDiscreteFunctionals(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    FE::systems::FESystem& system,
    FE::FieldId velocity_field,
    FE::Real density,
    const std::array<FE::Real, 3>& gravitational_acceleration,
    FE::Real dynamic_viscosity,
    bool has_constant_dynamic_viscosity)
{
    for (const auto& bc : free_surfaces) {
        if (!shouldDeclareFreeSurfaceDiscreteFunctional(bc)) {
            continue;
        }
        FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
        parameters.liquid_side =
            bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive
                ? FE::geometry::CutIntegrationSide::Positive
                : FE::geometry::CutIntegrationSide::Negative;
        parameters.surface_tension = constantScalarValueOrThrow(
            bc.surface_tension,
            "free-surface discrete-functional surface tension");
        // The pressure unknown is spatially varying and is not the scalar
        // volume multiplier in the capillary functional.  Preserve V_h as a
        // separately reported measure until a scalar multiplier is declared.
        parameters.volume_multiplier = FE::Real{0.0};
        for (const auto& contact_line : bc.contact_lines) {
            const auto kind = contactLineKind(contact_line);
            if (kind != ContactLineKind::PrescribedAngle &&
                kind != ContactLineKind::DynamicRenE) {
                continue;
            }
            parameters.young_wall_coefficients.push_back(
                FE::interfaces::FreeSurfaceYoungWallCoefficient{
                    .boundary_marker =
                        contactLineWallBoundaryMarker(contact_line),
                    .equilibrium_contact_angle_radians =
                        constantScalarValueOrThrow(
                            contactLineAngleRadians(contact_line),
                            "free-surface discrete-functional equilibrium "
                            "contact angle"),
                });
            if (kind == ContactLineKind::DynamicRenE) {
                parameters.dynamic_contact_coefficients.push_back(
                    FE::interfaces::FreeSurfaceDynamicContactCoefficient{
                        .boundary_marker =
                            contactLineWallBoundaryMarker(contact_line),
                        .equilibrium_contact_angle_radians =
                            constantScalarValueOrThrow(
                                contactLineAngleRadians(contact_line),
                                "free-surface discrete-functional dynamic "
                                "contact equilibrium angle"),
                        .mobility = constantScalarValueOrThrow(
                            contactLineMobility(contact_line),
                            "free-surface discrete-functional dynamic "
                            "contact mobility"),
                        .slip_length = constantScalarValueOrThrow(
                            contactLineSlipLength(contact_line),
                            "free-surface discrete-functional Navier slip "
                            "length"),
                        .dynamic_viscosity = dynamic_viscosity,
                    });
            }
        }
        std::optional<
            FE::interfaces::FreeSurfaceActiveVolumeDissipationParameters>
            active_volume_dissipation_parameters;
        if (has_constant_dynamic_viscosity) {
            active_volume_dissipation_parameters =
                FE::interfaces::
                    FreeSurfaceActiveVolumeDissipationParameters{
                        .liquid_side = parameters.liquid_side,
                        .dynamic_viscosity = dynamic_viscosity,
                    };
        }
        std::optional<
            FE::interfaces::FreeSurfaceExternalPressurePowerParameters>
            external_pressure_power_parameters;
        if (const auto* external_pressure =
                std::get_if<FE::Real>(&bc.external_pressure);
            external_pressure != nullptr) {
            external_pressure_power_parameters =
                FE::interfaces::
                    FreeSurfaceExternalPressurePowerParameters{
                        .liquid_side = parameters.liquid_side,
                        .external_pressure = *external_pressure,
                    };
        }
        system.declareFreeSurfaceDiscreteFunctional(
            FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
                .interface_marker = bc.interface_marker,
                .level_set_field = resolveLevelSetFieldId(bc, system),
                .velocity_field = velocity_field,
                .geometry_domain_id = bc.generated_interface_domain_id,
                .parameters = std::move(parameters),
                .active_volume_energy_parameters =
                    FE::interfaces::
                        FreeSurfaceActiveVolumeEnergyParameters{
                            .liquid_side =
                                bc.active_domain ==
                                        FreeSurfaceActiveDomain::
                                            LevelSetPositive
                                    ? FE::geometry::CutIntegrationSide::
                                          Positive
                                    : FE::geometry::CutIntegrationSide::
                                          Negative,
                            .density = density,
                            .gravitational_acceleration =
                                gravitational_acceleration,
                            .gravitational_reference_point =
                                {{FE::Real{0.0},
                                  FE::Real{0.0},
                                  FE::Real{0.0}}},
                        },
                .active_volume_dissipation_parameters =
                    active_volume_dissipation_parameters,
                .external_pressure_power_parameters =
                    external_pressure_power_parameters,
                .endpoint_functional_power_enabled =
                    usesSurfaceStress(bc),
                .capillary_balance_method =
                    usesSurfaceStress(bc)
                        ? FE::systems::FreeSurfaceCapillaryBalanceMethod::
                              DiscreteEnergyVolumeStationarity
                        : FE::systems::FreeSurfaceCapillaryBalanceMethod::
                              Unselected,
                .capillary_balance_qualification =
                    usesSurfaceStress(bc)
                        ? FE::systems::
                              FreeSurfaceCapillaryBalanceQualification::
                                  PrerequisiteOnly
                        : FE::systems::
                              FreeSurfaceCapillaryBalanceQualification::
                                  Unselected,
                .owner_component =
                    "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary",
            });
    }
}

[[nodiscard]] std::array<FE::Real, 3> normalizedWallNormal(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    std::array<FE::Real, 3> normal{
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[0], "contact-line wall_normal"),
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[1], "contact-line wall_normal"),
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[2], "contact-line wall_normal")};
    if (!std::all_of(normal.begin(), normal.end(), [](FE::Real component) {
            return std::isfinite(component);
        })) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed contact angle requires finite wall_normal components");
    }
    const auto norm = std::sqrt(normal[0] * normal[0] +
                                normal[1] * normal[1] +
                                normal[2] * normal[2]);
    if (!(norm > FE::Real{0.0}) || !std::isfinite(norm)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed contact angle requires a finite, nonzero wall_normal");
    }
    for (auto& component : normal) {
        component /= norm;
    }
    return normal;
}

void validateWallNormalDimension(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim,
    std::string_view context)
{
    const auto normal = normalizedWallNormal(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    for (int component = dim; component < 3; ++component) {
        if (std::abs(normal[static_cast<std::size_t>(component)]) > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) +
                " wall_normal must lie in the velocity space dimension");
        }
    }
}

[[nodiscard]] std::optional<FE::Real> constantScalarValue(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value)
{
    const auto* real = std::get_if<FE::Real>(&value);
    if (real == nullptr) {
        return std::nullopt;
    }
    return *real;
}

void validateFiniteConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && !std::isfinite(*literal)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be finite");
    }
}

void validateNonnegativeConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && *literal < FE::Real{0.0}) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be nonnegative");
    }
}

void validatePositiveConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && !(*literal > FE::Real{0.0})) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be positive");
    }
}

[[nodiscard]] bool formulationOwnsFieldRows(
    const FE::analysis::FormulationRecord& record,
    FE::FieldId field)
{
    for (const auto& [test_field, trial_field] : record.block_couplings) {
        (void)trial_field;
        if (test_field == field) {
            return true;
        }
    }
    // Retain compatibility with formulation metadata produced before
    // block-coupling discovery: a single active field is its row owner.
    return record.block_couplings.empty() &&
           record.active_fields.size() == 1u &&
           record.active_fields.front() == field;
}

[[nodiscard]] std::string owningOperatorTagForField(
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::string_view context)
{
    std::optional<std::string> owner;
    for (const auto& record : system.formulationRecords()) {
        if (!formulationOwnsFieldRows(record, field)) {
            continue;
        }
        if (record.operator_tag.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) + " has an empty owning operator tag");
        }
        if (owner.has_value() && *owner != record.operator_tag) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) + " has ambiguous owning operators '" +
                *owner + "' and '" + record.operator_tag + "'");
        }
        owner = record.operator_tag;
    }

    if (owner.has_value()) {
        return *owner;
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: " + std::string(context) +
        " has no installed owner formulation; install the level-set equation before the free-surface contact condition");
}

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

constexpr FE::Real kMinimumContactTransverseSine = FE::Real{1.0e-6};

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
            << "IncompressibleNavierStokesVMSModule: DynamicContactAngle "
               "requires every face on Contact_line_wall_marker to be a "
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
    return coordinateCutContextCallbackLocalPhase(
        system,
        local_exception,
        local_followup_available,
        phase);
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

[[nodiscard]] FE::Real clampUnit(FE::Real value) noexcept
{
    return std::max(FE::Real{-1.0}, std::min(FE::Real{1.0}, value));
}

[[nodiscard]] std::array<FE::Real, 3> activeInterfaceNormalForContactAngle(
    const FreeSurfaceBoundary& bc,
    std::array<FE::Real, 3> normal) noexcept
{
    normal = normalizedOrZero(normal);
    if (bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        for (auto& component : normal) {
            component = -component;
        }
    }
    return normal;
}

void logDynamicContactOperatorAngle(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext* context,
    const FreeSurfaceBoundary& bc,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker)
{
    // Candidate presence is communicator-certified before update callbacks.
    // Before setup there is no distributed diagnostic to reduce.
    if (context == nullptr || !system.isSetup()) {
        return;
    }
    const auto log_unavailable = [&](std::string_view reason) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: DynamicContactAngle operator-consistent contact geometry") +
            " diagnostic=dynamic_contact_operator_angle" +
            " status=unavailable" +
            " reason=" + std::string(reason) +
            " interface_marker=" + std::to_string(bc.interface_marker) +
            " contact_line_marker=" + std::to_string(contact_marker) +
            " wall_boundary_marker=" +
            std::to_string(contactLineWallBoundaryMarker(contact_line)) +
            " normal_source=" +
            std::string(usesSurfaceStress(bc)
                            ? "generated_interface_rule_geometry"
                            : "unitNormalFromLevelSet_Q1") +
            " evaluation_location=generated_contact_root");
    };

    std::exception_ptr local_preflight_exception;
    std::string_view local_unavailable_reason;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const svmp::Mesh* native_mesh = nullptr;
    FE::FieldId phi_id = FE::INVALID_FIELD_ID;
    std::vector<FE::Real> coefficients;
    std::vector<std::uint8_t> assigned;
#endif
    try {
        const auto prepare_local_state = [&]() -> std::string_view {
            if (!context->hasGeneratedInterfaceMarker(contact_marker)) {
                return "generated_contact_marker_unavailable";
            }
            context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
                system.meshAccess());
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
            return "native_mesh_support_disabled";
#else
            native_mesh = system.mesh();
            if (native_mesh == nullptr) {
                return "native_mesh_unavailable";
            }
            phi_id = system.findFieldByName(bc.level_set_field_name);
            if (phi_id == FE::INVALID_FIELD_ID) {
                return "level_set_field_unavailable";
            }
            const auto& phi_record = system.fieldRecord(phi_id);
            if (phi_record.components != 1 || !phi_record.space) {
                return "level_set_space_not_scalar";
            }

            const auto& local_mesh = native_mesh->local_mesh();
            if (!MeshFields::has_field(
                    local_mesh,
                    EntityKind::Vertex,
                    bc.level_set_field_name)) {
                return "synchronized_vertex_field_unavailable";
            }
            const auto handle = MeshFields::get_field_handle(
                local_mesh, EntityKind::Vertex, bc.level_set_field_name);
            if (MeshFields::field_type(local_mesh, handle) !=
                    FieldScalarType::Float64 ||
                MeshFields::field_components(local_mesh, handle) < 1u) {
                return "synchronized_vertex_field_not_scalar_float64";
            }
            const auto entity_count =
                MeshFields::field_entity_count(local_mesh, handle);
            const auto mesh_components =
                MeshFields::field_components(local_mesh, handle);
            const auto* mesh_values =
                MeshFields::field_data_as<FE::Real>(local_mesh, handle);
            if (mesh_values == nullptr ||
                entity_count < native_mesh->n_vertices()) {
                return "synchronized_vertex_field_incomplete";
            }

            const auto& phi_dofs = system.fieldDofHandler(phi_id);
            const auto n_phi_dofs =
                static_cast<std::size_t>(phi_dofs.getNumDofs());
            coefficients.assign(n_phi_dofs, FE::Real{0.0});
            assigned.assign(n_phi_dofs, 0u);
            try {
                const auto projection =
                    system.projectMeshVertexValuesToFieldCoefficients(
                        phi_id,
                        std::span<const FE::Real>(
                            mesh_values,
                            entity_count * mesh_components),
                        mesh_components,
                        std::span<FE::Real>(
                            coefficients.data(), coefficients.size()),
                        std::span<std::uint8_t>(
                            assigned.data(), assigned.size()),
                        "DynamicContactAngle operator-consistent contact diagnostic");
                if (projection.unassigned_dofs != 0u) {
                    return "synchronized_vertex_projection_incomplete";
                }
            } catch (const std::exception&) {
                return "synchronized_vertex_projection_failed";
            }
            return {};
#endif
        };
        local_unavailable_reason = prepare_local_state();
    } catch (...) {
        local_preflight_exception = std::current_exception();
    }

    const bool globally_available =
        coordinateCutContextCallbackLocalPhase(
            system,
            local_preflight_exception,
            local_unavailable_reason.empty(),
            "operator_angle_preflight");
    if (!globally_available) {
        const bool silent_local_state =
            local_unavailable_reason ==
                "generated_contact_marker_unavailable";
        if (!local_unavailable_reason.empty() &&
            !silent_local_state) {
            log_unavailable(local_unavailable_reason);
        } else if (local_unavailable_reason.empty()) {
            log_unavailable("peer_rank_preflight_unavailable");
        }
        return;
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    FE::Real target_angle = 0.0;
    FE::Real target_cos = 0.0;
    std::uint64_t samples = 0u;
    std::uint64_t reference_rules = 0u;
    std::uint64_t current_rules = 0u;
    FE::Real weight_sum = 0.0;
    FE::Real weighted_cos_sum = 0.0;
    FE::Real weighted_gap_sum = 0.0;
    FE::Real weighted_angle_sum = 0.0;
    FE::Real weighted_tangent_norm_sum = 0.0;
    FE::Real min_cos = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_cos = -std::numeric_limits<FE::Real>::infinity();
    FE::Real min_gap = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_gap = -std::numeric_limits<FE::Real>::infinity();
    FE::Real min_tangent_norm = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_tangent_norm = -std::numeric_limits<FE::Real>::infinity();
    FE::Real min_gradient_norm = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_gradient_norm = -std::numeric_limits<FE::Real>::infinity();

    const auto accumulate_local_state = [&]() {
        const auto& phi_record = system.fieldRecord(phi_id);
        const auto& phi_dofs = system.fieldDofHandler(phi_id);
        FE::level_set::LevelSetCellEvaluator evaluator(
            *phi_record.space,
            phi_dofs,
            std::span<const FE::Real>(
                coefficients.data(), coefficients.size()));
        const auto wall_n = normalizedWallNormal(contact_line);
        target_angle = constantScalarValueOrThrow(
            contactLineAngleRadians(contact_line),
            "DynamicContactAngle equilibrium contact_angle_radians");
        target_cos = std::cos(target_angle);

        const auto& mesh_access = system.meshAccess();
        for (const auto* rule :
             context->interfaceRulesForMarker(contact_marker)) {
            if (rule == nullptr) {
                continue;
            }
            const auto cell = static_cast<FE::GlobalIndex>(
                rule->provenance.parent_entity);
            // Generated contexts may retain ghost-parent rules on more than
            // one rank. Count and integrate only the owning rank so that the
            // MPI reduction describes the physical contact set exactly once.
            if (cell < 0 || cell >= mesh_access.numCells() ||
                !mesh_access.isOwnedCell(cell)) {
                continue;
            }
            if (rule->frame ==
                FE::geometry::CutGeometryFrame::Reference) {
                ++reference_rules;
            } else {
                ++current_rules;
                // DynamicContactAngle currently admits only LinearCorner
                // contact geometry, whose production generated-contact
                // contract is a parent-reference rule. A current-frame point
                // does not carry enough inverse-location data to reproduce
                // the form evaluator.
                continue;
            }
            for (const auto& point : rule->points) {
                const auto weight = std::abs(point.weight);
                if (!(weight > FE::Real{0.0}) ||
                    !std::isfinite(weight)) {
                    continue;
                }
                std::array<FE::Real, 3> physical_gradient;
                try {
                    if (usesSurfaceStress(bc)) {
                        // StandardAssembler maps FormExpr::normal() as a
                        // reference covector. Reproduce that exact generated
                        // geometry here so the diagnostic observes the same
                        // normal as the surface-stress/contact residual.
                        physical_gradient = physicalContactCovector(
                            mesh_access,
                            *rule,
                            point,
                            point.normal,
                            "generated interface-rule normal");
                    } else {
                        // The legacy curvature-traction contact law evaluates
                        // unitNormalFromLevelSet(phi) at the contact root.
                        const auto evaluation =
                            evaluator.evaluate(cell, point.point);
                        physical_gradient = physicalContactCovector(
                            mesh_access,
                            *rule,
                            point,
                            evaluation.reference_gradient,
                            "operator level-set gradient");
                    }
                } catch (const std::exception&) {
                    continue;
                }
                const auto gradient_norm = norm3(physical_gradient);
                // This is the literal safeNormalize denominator used by
                // unitNormalFromLevelSet(phi), including its default
                // eps=1e-12.
                const auto safe_gradient_norm = std::sqrt(
                    gradient_norm * gradient_norm + FE::Real{1.0e-24});
                if (!(safe_gradient_norm > FE::Real{0.0}) ||
                    !std::isfinite(safe_gradient_norm)) {
                    continue;
                }
                std::array<FE::Real, 3> active_normal{{
                    physical_gradient[0] / safe_gradient_norm,
                    physical_gradient[1] / safe_gradient_norm,
                    physical_gradient[2] / safe_gradient_norm}};
                if (bc.active_domain ==
                    FreeSurfaceActiveDomain::LevelSetPositive) {
                    for (auto& component : active_normal) {
                        component = -component;
                    }
                }
                const auto normal_dot_wall =
                    dot3(active_normal, wall_n);
                const auto dynamic_cos = -normal_dot_wall;
                const auto young_gap = target_cos - dynamic_cos;
                const std::array<FE::Real, 3> tangent{{
                    active_normal[0] -
                        normal_dot_wall * wall_n[0],
                    active_normal[1] -
                        normal_dot_wall * wall_n[1],
                    active_normal[2] -
                        normal_dot_wall * wall_n[2]}};
                const auto tangent_norm = norm3(tangent);
                const auto angle =
                    std::acos(clampUnit(dynamic_cos));
                if (!std::isfinite(dynamic_cos) ||
                    !std::isfinite(young_gap) ||
                    !std::isfinite(tangent_norm) ||
                    !std::isfinite(angle)) {
                    continue;
                }

                ++samples;
                weight_sum += weight;
                weighted_cos_sum += weight * dynamic_cos;
                weighted_gap_sum += weight * young_gap;
                weighted_angle_sum += weight * angle;
                weighted_tangent_norm_sum += weight * tangent_norm;
                min_cos = std::min(min_cos, dynamic_cos);
                max_cos = std::max(max_cos, dynamic_cos);
                min_gap = std::min(min_gap, young_gap);
                max_gap = std::max(max_gap, young_gap);
                min_tangent_norm =
                    std::min(min_tangent_norm, tangent_norm);
                max_tangent_norm =
                    std::max(max_tangent_norm, tangent_norm);
                min_gradient_norm =
                    std::min(min_gradient_norm, gradient_norm);
                max_gradient_norm =
                    std::max(max_gradient_norm, gradient_norm);
            }
        }
    };

    std::exception_ptr local_accumulation_exception;
    try {
        accumulate_local_state();
    } catch (...) {
        local_accumulation_exception = std::current_exception();
    }
    static_cast<void>(coordinateCutContextCallbackLocalPhase(
        system,
        local_accumulation_exception,
        true,
        "operator_angle_local_accumulation"));

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        MPI_Finalized(&mpi_finalized);
    }
    const auto communicator = system.dofHandler().mpiComm();
    if (mpi_initialized != 0 && mpi_finalized == 0 &&
        communicator != MPI_COMM_NULL) {
        int communicator_size = 1;
        MPI_Comm_size(communicator, &communicator_size);
        if (communicator_size > 1) {
            const std::array<unsigned long long, 3> local_counts{{
                static_cast<unsigned long long>(samples),
                static_cast<unsigned long long>(reference_rules),
                static_cast<unsigned long long>(current_rules)}};
            std::array<unsigned long long, 3> global_counts{};
            MPI_Allreduce(local_counts.data(), global_counts.data(),
                          static_cast<int>(local_counts.size()),
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
            samples = static_cast<std::uint64_t>(global_counts[0]);
            reference_rules = static_cast<std::uint64_t>(global_counts[1]);
            current_rules = static_cast<std::uint64_t>(global_counts[2]);

            const std::array<double, 5> local_sums{{
                static_cast<double>(weight_sum),
                static_cast<double>(weighted_cos_sum),
                static_cast<double>(weighted_gap_sum),
                static_cast<double>(weighted_angle_sum),
                static_cast<double>(weighted_tangent_norm_sum)}};
            std::array<double, 5> global_sums{};
            MPI_Allreduce(local_sums.data(), global_sums.data(),
                          static_cast<int>(local_sums.size()), MPI_DOUBLE,
                          MPI_SUM, communicator);
            weight_sum = static_cast<FE::Real>(global_sums[0]);
            weighted_cos_sum = static_cast<FE::Real>(global_sums[1]);
            weighted_gap_sum = static_cast<FE::Real>(global_sums[2]);
            weighted_angle_sum = static_cast<FE::Real>(global_sums[3]);
            weighted_tangent_norm_sum =
                static_cast<FE::Real>(global_sums[4]);

            const std::array<double, 4> local_minima{{
                static_cast<double>(min_cos),
                static_cast<double>(min_gap),
                static_cast<double>(min_tangent_norm),
                static_cast<double>(min_gradient_norm)}};
            std::array<double, 4> global_minima{};
            MPI_Allreduce(local_minima.data(), global_minima.data(),
                          static_cast<int>(local_minima.size()), MPI_DOUBLE,
                          MPI_MIN, communicator);
            min_cos = static_cast<FE::Real>(global_minima[0]);
            min_gap = static_cast<FE::Real>(global_minima[1]);
            min_tangent_norm = static_cast<FE::Real>(global_minima[2]);
            min_gradient_norm = static_cast<FE::Real>(global_minima[3]);

            const std::array<double, 4> local_maxima{{
                static_cast<double>(max_cos),
                static_cast<double>(max_gap),
                static_cast<double>(max_tangent_norm),
                static_cast<double>(max_gradient_norm)}};
            std::array<double, 4> global_maxima{};
            MPI_Allreduce(local_maxima.data(), global_maxima.data(),
                          static_cast<int>(local_maxima.size()), MPI_DOUBLE,
                          MPI_MAX, communicator);
            max_cos = static_cast<FE::Real>(global_maxima[0]);
            max_gap = static_cast<FE::Real>(global_maxima[1]);
            max_tangent_norm = static_cast<FE::Real>(global_maxima[2]);
            max_gradient_norm = static_cast<FE::Real>(global_maxima[3]);
        }
    }
#endif

    if (samples == 0u || !(weight_sum > FE::Real{0.0}) ||
        current_rules != 0u) {
        log_unavailable(current_rules != 0u
                            ? "current_frame_rule_not_operator_reproducible"
                            : "no_valid_operator_normal_samples");
        return;
    }

    constexpr FE::Real radians_to_degrees =
        FE::Real{180.0} /
        FE::Real{3.141592653589793238462643383279502884};
    const auto mean_cos = weighted_cos_sum / weight_sum;
    const auto mean_gap = weighted_gap_sum / weight_sum;
    const auto mean_angle = weighted_angle_sum / weight_sum;
    const auto mean_tangent_norm =
        weighted_tangent_norm_sum / weight_sum;
    std::ostringstream message;
    message
        << "IncompressibleNavierStokesVMSModule: DynamicContactAngle "
           "operator-consistent contact geometry"
        << " diagnostic=dynamic_contact_operator_angle"
        << " status=available"
        << " interface_marker=" << bc.interface_marker
        << " contact_line_marker=" << contact_marker
        << " wall_boundary_marker=" << contactLineWallBoundaryMarker(contact_line)
        << " normal_source="
        << (usesSurfaceStress(bc)
                ? "generated_interface_rule_geometry"
                : "unitNormalFromLevelSet_Q1")
        << " state_source=synchronized_native_vertex_field"
        << " evaluation_location=generated_contact_root"
        << " coordinate_source=cut_rule_point"
        << " angle_convention=through_liquid"
        << " young_gap_convention=cos_theta_e_minus_cos_theta_d"
        << " samples=" << samples
        << " reference_rules=" << reference_rules
        << " current_rules=" << current_rules
        << " reference_weight_sum=" << static_cast<double>(weight_sum)
        << " target_angle_degrees="
        << static_cast<double>(target_angle * radians_to_degrees)
        << " target_cos=" << static_cast<double>(target_cos)
        << " mean_dynamic_cos=" << static_cast<double>(mean_cos)
        << " min_dynamic_cos=" << static_cast<double>(min_cos)
        << " max_dynamic_cos=" << static_cast<double>(max_cos)
        << " mean_young_gap=" << static_cast<double>(mean_gap)
        << " min_young_gap=" << static_cast<double>(min_gap)
        << " max_young_gap=" << static_cast<double>(max_gap)
        << " mean_dynamic_angle_degrees="
        << static_cast<double>(mean_angle * radians_to_degrees)
        << " mean_wall_tangential_normal_norm="
        << static_cast<double>(mean_tangent_norm)
        << " min_wall_tangential_normal_norm="
        << static_cast<double>(min_tangent_norm)
        << " max_wall_tangential_normal_norm="
        << static_cast<double>(max_tangent_norm)
        << " min_level_set_gradient_norm="
        << static_cast<double>(min_gradient_norm)
        << " max_level_set_gradient_norm="
        << static_cast<double>(max_gradient_norm)
        << " transversality_threshold="
        << static_cast<double>(kMinimumContactTransverseSine)
        << " transversality_satisfied="
        << (min_tangent_norm >= kMinimumContactTransverseSine
                ? "true"
                : "false");
    FE_LOG_INFO(message.str());
#endif
}

[[nodiscard]] FE::forms::FormExpr wallNormalExpression(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim)
{
    const auto wall_normal = normalizedWallNormal(contact_line);
    std::vector<FE::forms::FormExpr> wall_components;
    wall_components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        wall_components.push_back(FE::forms::FormExpr::constant(
            wall_normal[static_cast<std::size_t>(d)]));
    }
    return FE::forms::FormExpr::asVector(std::move(wall_components));
}

[[nodiscard]] int axisAlignedWallNormalAxis(
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const auto normal = normalizedWallNormal(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    int axis = -1;
    for (int component = 0; component < 3; ++component) {
        const auto magnitude =
            std::abs(normal[static_cast<std::size_t>(component)]);
        if (component >= dim) {
            if (magnitude > tolerance) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle wall_normal must lie in the velocity space dimension");
            }
            continue;
        }
        if (std::abs(magnitude - FE::Real{1.0}) <= tolerance) {
            if (axis >= 0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle wall_normal must have exactly one axis-aligned component");
            }
            axis = component;
        } else if (magnitude > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires an axis-aligned wall_normal because general linear-combination normal constraints are unavailable");
        }
    }
    if (axis < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires an axis-aligned wall_normal");
    }
    return axis;
}

void validateDynamicContactWallEssentialBC(
    const IncompressibleNavierStokesVMSOptions& options,
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const int normal_axis = axisAlignedWallNormalAxis(contact_line, dim);
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
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires a stationary zero-valued normal wall essential condition");
            }
            if (component == normal_axis) {
                normal_is_constrained = true;
            } else {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle rejects tangential/full no-slip Dirichlet constraints on its Navier-slip wall");
            }
        }
    }
    for (const auto& dirichlet : options.velocity_dirichlet_weak) {
        if (dirichlet.boundary_marker == contactLineWallBoundaryMarker(contact_line)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires a strong zero normal-only wall essential condition; weak velocity Dirichlet data on the wall are unsupported");
        }
    }
    if (!saw_wall_bc || !normal_is_constrained) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires an axis-aligned zero normal-only velocity Dirichlet condition on Contact_line_wall_marker");
    }
}

[[nodiscard]] std::string normalizedFreeSurfaceOptionToken(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    value.erase(std::remove_if(value.begin(), value.end(),
                               [](unsigned char c) {
                                   return c == '_' || c == '-' || std::isspace(c);
                               }),
                value.end());
    return value;
}

[[nodiscard]] bool isHighOrderImplicitGeneratedInterface(
    const FreeSurfaceBoundary& bc)
{
    const auto token =
        normalizedFreeSurfaceOptionToken(bc.generated_interface_geometry);
    return token == "highorderimplicit" || token == "implicit" ||
           token == "saye" || token == "sayeimplicit";
}

[[nodiscard]] bool isRefreshedFrozenGeometryTangent(
    const FreeSurfaceBoundary& bc)
{
    const auto token =
        normalizedFreeSurfaceOptionToken(bc.geometry_tangent_policy);
    return token.empty() || token == "refreshedfrozenquadrature" ||
           token == "refreshedfrozen" || token == "frozenquadrature";
}

[[nodiscard]] const char* pressureStabilizationPolicyName(
    FreeSurfacePressureStabilizationPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfacePressureStabilizationPolicy::Enabled:
        return "Enabled";
    case FreeSurfacePressureStabilizationPolicy::Incremental:
        return "Incremental";
    case FreeSurfacePressureStabilizationPolicy::Disabled:
        return "Disabled";
    case FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder:
        return "DisabledForRefreshedFrozenHighOrder";
    }
    return "Unknown";
}

[[nodiscard]] const char* pressureStabilizationDisabledReason(
    const FreeSurfaceBoundary& bc)
{
    switch (bc.cut_cell_stabilization.pressure_policy) {
    case FreeSurfacePressureStabilizationPolicy::Enabled:
    case FreeSurfacePressureStabilizationPolicy::Incremental:
        return "none";
    case FreeSurfacePressureStabilizationPolicy::Disabled:
        return "explicit_policy_disabled";
    case FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder:
        break;
    }
    return isHighOrderImplicitGeneratedInterface(bc) &&
                   isRefreshedFrozenGeometryTangent(bc)
               ? "refreshed_frozen_high_order_policy"
               : "none";
}

[[nodiscard]] bool pressureStabilizationDisabledByPolicy(
    const FreeSurfaceBoundary& bc)
{
    return std::string_view(pressureStabilizationDisabledReason(bc)) != "none";
}

[[nodiscard]] bool pressureStabilizationActive(const FreeSurfaceBoundary& bc)
{
    return !pressureStabilizationDisabledByPolicy(bc) &&
           !FE::forms::bc::isZeroConstantScalarValue(
               bc.cut_cell_stabilization.pressure_gradient_penalty);
}

[[nodiscard]] const char* unfittedShapeTangentPolicyName()
{
    return unfittedLevelSetShapeTangentsDisabled()
               ? "disabled_by_default"
               : "enabled_experimental";
}

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "inactive_zero_surface_tension";
    }
    if (usesSurfaceStress(bc)) {
        return "variational_surface_stress_generated_geometry";
    }
    if (isUnfittedLevelSet(bc)) {
        if (!bc.curvature_field_name.empty()) {
            return "curvature_field_level_set_normal_signed";
        }
        if (bc.use_level_set_curvature) {
            return "raw_level_set_curvature_guarded";
        }
        return "supplied_scalar_level_set_normal_signed";
    }
    if (bc.use_current_geometry_curvature) {
        return "current_geometry_curvature";
    }
    if (!bc.curvature_field_name.empty()) {
        return "curvature_field";
    }
    return "supplied_scalar";
}

[[nodiscard]] std::string freeSurfaceCurvatureTangentPolicyName(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "not_applicable";
    }
    if (usesSurfaceStress(bc)) {
        return isUnfittedLevelSet(bc)
                   ? "generated_geometry_refreshed_frozen"
                   : "fitted_geometry_surface_variation";
    }
    if (!bc.curvature_field_name.empty()) {
        const auto kappa_id = system.findFieldByName(bc.curvature_field_name);
        if (kappa_id != FE::INVALID_FIELD_ID &&
            system.fieldParticipatesInUnknownVector(kappa_id)) {
            return "curvature_field_trial_coupled_geometry_curvature_frozen";
        }
        return "curvature_field_picard_frozen";
    }
    if (isUnfittedLevelSet(bc)) {
        return "picard_frozen_or_guarded";
    }
    return bc.use_current_geometry_curvature
               ? "geometry_curvature_from_active_geometry_path"
               : "supplied_scalar_frozen";
}

void validateFreeSurfaceBoundary(const FreeSurfaceBoundary& bc,
                                 const IncompressibleNavierStokesVMSOptions& options,
                                 bool ale_enabled,
                                 const FE::systems::FESystem& system,
                                 int dim)
{
    if (bc.implementation != FreeSurfaceImplementation::FittedALE &&
        bc.implementation !=
            FreeSurfaceImplementation::UnfittedLevelSet) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unsupported "
            "free-surface implementation");
    }
    const bool supported_boundary_policies =
        (bc.active_domain == FreeSurfaceActiveDomain::None ||
         bc.active_domain ==
             FreeSurfaceActiveDomain::LevelSetNegative ||
         bc.active_domain ==
             FreeSurfaceActiveDomain::LevelSetPositive) &&
        (bc.active_domain_method ==
             FreeSurfaceActiveDomainMethod::CutVolume ||
         bc.active_domain_method ==
             FreeSurfaceActiveDomainMethod::SmoothedIndicator) &&
        (bc.kinematic_enforcement ==
             FreeSurfaceKinematicEnforcement::None ||
         bc.kinematic_enforcement ==
             FreeSurfaceKinematicEnforcement::Penalty ||
         bc.kinematic_enforcement ==
             FreeSurfaceKinematicEnforcement::Nitsche) &&
        (bc.normal_kinematic_policy ==
             FreeSurfaceNormalKinematicPolicy::None ||
         bc.normal_kinematic_policy ==
             FreeSurfaceNormalKinematicPolicy::
                 MatchFluidNormalVelocity) &&
        (bc.tangential_mesh_policy ==
             FreeSurfaceTangentialMeshPolicy::Free ||
         bc.tangential_mesh_policy ==
             FreeSurfaceTangentialMeshPolicy::SmoothingOnly ||
         bc.tangential_mesh_policy ==
             FreeSurfaceTangentialMeshPolicy::Prescribed) &&
        (bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::Enabled ||
         bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::Incremental ||
         bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::Disabled ||
         bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::
                 DisabledForRefreshedFrozenHighOrder) &&
        (bc.surface_tension_form ==
             FreeSurfaceSurfaceTensionForm::Automatic ||
         bc.surface_tension_form ==
             FreeSurfaceSurfaceTensionForm::CurvatureTraction ||
         bc.surface_tension_form ==
             FreeSurfaceSurfaceTensionForm::SurfaceStress);
    if (!supported_boundary_policies) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unsupported "
            "free-surface boundary policy");
    }
    if (!std::isfinite(bc.level_set_isovalue)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free-surface level_set_isovalue must be finite");
    }
    if (!std::isfinite(bc.active_domain_smoothing_width) ||
        bc.active_domain_smoothing_width < FE::Real{0.0}) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free-surface active_domain_smoothing_width must be finite and nonnegative");
    }
    validateFiniteConstantScalar(
        bc.external_pressure, "free-surface external_pressure");
    // CurvatureTraction omits tangential Marangoni traction.  SurfaceStress
    // contains the constant-gamma surface-energy variation, but its contact
    // wall energy and current input contract also assume one literal gamma.
    // Keep variable surface chemistry/Marangoni data fail-closed until gamma
    // and the wall-energy law are coupled consistently throughout.
    (void)constantScalarValueOrThrow(
        bc.surface_tension,
        "free-surface surface_tension (variable surface tension/Marangoni traction is unsupported)");
    validateNonnegativeConstantScalar(
        bc.surface_tension, "free-surface surface_tension");
    validateFiniteConstantScalar(bc.curvature, "free-surface curvature");
    validateNonnegativeConstantScalar(
        bc.cut_cell_stabilization.pressure_gradient_penalty,
        "cut-cell pressure-gradient penalty");
    const auto& aggregation_guards = bc.small_cut_aggregation_guards;
    if (aggregation_guards.maximum_root_path_length == 0u ||
        !(aggregation_guards.maximum_reference_extrapolation_distance >=
          FE::Real{0.0}) ||
        !std::isfinite(
            aggregation_guards.maximum_reference_extrapolation_distance) ||
        !(aggregation_guards.maximum_absolute_coefficient >= FE::Real{1.0}) ||
        !std::isfinite(aggregation_guards.maximum_absolute_coefficient) ||
        !(aggregation_guards.maximum_row_l1_norm >= FE::Real{1.0}) ||
        !std::isfinite(aggregation_guards.maximum_row_l1_norm) ||
        aggregation_guards.maximum_row_l1_norm <
            aggregation_guards.maximum_absolute_coefficient) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: small-cut aggregation "
            "guards require a positive root path, finite nonnegative "
            "reference extrapolation distance, finite coefficient and row "
            "L1 limits at least one, and a row L1 limit no smaller than the "
            "coefficient limit");
    }
    validateNonnegativeConstantScalar(
        bc.velocity_extension.diffusivity,
        "velocity-extension diffusivity");
    for (const auto& value : bc.prescribed_tangential_mesh_velocity) {
        validateFiniteConstantScalar(
            value, "prescribed tangential mesh velocity");
    }
    validatePositiveConstantScalar(
        bc.tangential_mesh_penalty, "tangential mesh penalty");
    if (bc.tangential_mesh_policy !=
        FreeSurfaceTangentialMeshPolicy::Prescribed) {
        for (const auto& value : bc.prescribed_tangential_mesh_velocity) {
            if (!FE::forms::bc::isZeroConstantScalarValue(value)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: Free and "
                    "SmoothingOnly tangential mesh policies cannot carry a "
                    "prescribed tangential velocity");
            }
        }
    }

    if (isUnfittedLevelSet(bc)) {
        if (bc.tangential_mesh_policy !=
            FreeSurfaceTangentialMeshPolicy::SmoothingOnly) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: tangential mesh "
                "policies apply only to fitted-ALE free surfaces");
        }
        if (ale_enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set free surfaces cannot currently be combined with ALE; the level-set transport must use the relative velocity u-w before this combination is supported");
        }
        if (bc.interface_marker < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted free surface requires interface_marker >= 0");
        }
        if (bc.level_set_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted free surface requires a non-empty level_set_field_name");
        }
        if (bc.active_domain == FreeSurfaceActiveDomain::None &&
            !bc.allow_full_domain_unfitted_free_surface) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: UnfittedLevelSet free surfaces require Active_domain=LevelSetNegative or LevelSetPositive; set Allow_full_domain_unfitted_free_surface=true only for deliberate full-domain diagnostic runs");
        }
        if (bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set free surfaces use Eulerian level-set transport for kinematics; mesh-kinematic Penalty/Nitsche enforcement would constrain u.n and freeze the interface and is therefore unsupported");
        }
        const bool has_surface_stress_residual =
            usesSurfaceStress(bc) &&
            (!FE::forms::bc::isZeroConstantScalarValue(
                 bc.external_pressure) ||
             !FE::forms::bc::isZeroConstantScalarValue(
                 bc.surface_tension));
        if (has_surface_stress_residual &&
            !isRefreshedFrozenGeometryTangent(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: SurfaceStress currently requires Geometry_tangent_policy=RefreshedFrozenQuadrature; DifferentiatedQuadrature does not yet contain the complete projector-normal, point-location, and measure derivative");
        }
        if (has_surface_stress_residual &&
            !unfittedLevelSetShapeTangentsDisabled()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: SurfaceStress currently requires refreshed-frozen generated geometry; the experimental unfitted level-set shape-tangent switch does not yet contain the complete projector-normal and point-location derivative and is therefore rejected");
        }
        if (!usesSurfaceStress(bc) &&
            !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension) &&
            bc.use_level_set_curvature &&
            bc.curvature_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set surface tension with raw level-set curvature is not validated; set Use_level_set_curvature=false and provide Curvature or a projected curvature field");
        }
        const auto& cut = bc.cut_cell_stabilization;
        if (cut.enabled) {
            (void)constantScalarValueOrThrow(
                cut.pressure_gradient_penalty,
                "enabled cut-cell pressure-gradient penalty");
            if (cut.cut_metadata_scale_cap.has_value() &&
                (!std::isfinite(*cut.cut_metadata_scale_cap) ||
                 *cut.cut_metadata_scale_cap < FE::Real{1.0})) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: cut-cell metadata scale cap must be finite and at least 1");
            }
        }
        const auto& extension = bc.velocity_extension;
        if (extension.enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: the legacy same-field dry-domain velocity diffusion is retired because it modifies physical momentum; use the separate algebraic level-set advection velocity extension and small-cut aggregation");
        }
    } else {
        if (usesSurfaceStress(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted-ALE SurfaceStress is not yet qualified for current-frame test-function gradients; use Automatic/CurvatureTraction for fitted boundaries");
        }
        if (bc.boundary_marker < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted free surface requires boundary_marker >= 0");
        }
        if (bc.cut_cell_stabilization.enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: cut-cell stabilization is only valid for unfitted level-set free surfaces");
        }
        if (bc.active_domain != FreeSurfaceActiveDomain::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain free-surface volume integration is only valid for unfitted level-set free surfaces");
        }
        if (bc.allow_full_domain_unfitted_free_surface) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Allow_full_domain_unfitted_free_surface is only valid for unfitted level-set free surfaces");
        }
        if (bc.velocity_extension.enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: velocity extension is only valid for unfitted level-set free surfaces");
        }
        if (!options.explicit_legacy_configuration) {
            if (!ale_enabled) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: the qualified "
                    "fitted-ALE free-surface contract requires ALE to be "
                    "enabled");
            }
            if (options.mesh_velocity_source !=
                ALEMeshVelocitySource::CoupledDisplacement) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: the qualified "
                    "fitted-ALE free-surface contract requires mesh velocity "
                    "derived from a coupled mesh-displacement unknown");
            }
            if (bc.normal_kinematic_policy !=
                FreeSurfaceNormalKinematicPolicy::
                    MatchFluidNormalVelocity) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: the qualified "
                    "fitted-ALE free-surface contract requires "
                    "MatchFluidNormalVelocity");
            }
            if (bc.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::Penalty &&
                bc.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::Nitsche) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: the qualified "
                    "fitted-ALE free-surface contract requires explicit "
                    "Penalty or Nitsche normal enforcement");
            }
        }
        if (bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
            !ale_enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted free-surface kinematics require ALE to be enabled");
        }
        if (bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
            bc.normal_kinematic_policy == FreeSurfaceNormalKinematicPolicy::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted free-surface kinematic enforcement requires a normal kinematic policy");
        }
        if (bc.tangential_mesh_policy ==
                FreeSurfaceTangentialMeshPolicy::Prescribed &&
            !ale_enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: prescribed fitted "
                "free-surface tangential mesh velocity requires ALE");
        }
        if (bc.tangential_mesh_policy ==
                FreeSurfaceTangentialMeshPolicy::Prescribed &&
            options.mesh_velocity_source !=
                ALEMeshVelocitySource::CoupledDisplacement) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: prescribed fitted "
                "free-surface tangential mesh velocity requires a coupled "
                "mesh-displacement unknown");
        }
        for (int component = dim; component < 3; ++component) {
            if (!FE::forms::bc::isZeroConstantScalarValue(
                    bc.prescribed_tangential_mesh_velocity[
                        static_cast<std::size_t>(component)])) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed "
                    "tangential mesh velocity must lie in the mesh "
                    "dimension");
            }
        }
    }

    validateFiniteConstantScalar(
        bc.kinematic_penalty, "free-surface kinematic_penalty");
    if (bc.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Penalty) {
        (void)constantScalarValueOrThrow(
            bc.kinematic_penalty,
            "penalty free-surface kinematic_penalty");
        validatePositiveConstantScalar(
            bc.kinematic_penalty,
            "penalty free-surface kinematic_penalty");
    }
    if (bc.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Nitsche &&
        (!std::isfinite(bc.kinematic_nitsche_gamma) ||
         !(bc.kinematic_nitsche_gamma > FE::Real{0.0}))) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics require a finite positive boundary-local kinematic_nitsche_gamma");
    }

    for (std::size_t i = 0; i < bc.contact_lines.size(); ++i) {
        const auto& candidate = bc.contact_lines[i];
        const auto owns_wall_law = [](ContactLineKind model) {
            return model == ContactLineKind::PrescribedAngle ||
                   model == ContactLineKind::DynamicRenE;
        };
        if (!owns_wall_law(contactLineKind(candidate)) ||
            contactLineWallBoundaryMarker(candidate) < 0) {
            continue;
        }
        for (std::size_t j = i + 1; j < bc.contact_lines.size(); ++j) {
            const auto& other = bc.contact_lines[j];
            if (owns_wall_law(contactLineKind(other)) &&
                contactLineWallBoundaryMarker(other) ==
                    contactLineWallBoundaryMarker(candidate)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: duplicate or conflicting contact-line wall laws on wall marker " +
                    std::to_string(contactLineWallBoundaryMarker(candidate)) +
                    "; exactly one contact-line model may own each wall footprint");
            }
        }
    }

    for (const auto& contact_line : bc.contact_lines) {
        if (contactLineKind(contact_line) == ContactLineKind::Pinned) {
            if (isUnfittedLevelSet(bc)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned contact lines are currently supported only for fitted ALE free surfaces");
            }
            if (!ale_enabled) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require ALE to be enabled");
            }
            if (contactLineConstraintMarker(contact_line) < 0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned contact line requires contact_line_marker or wall_boundary_marker >= 0");
            }
        }
        if (contactLineKind(contact_line) == ContactLineKind::PrescribedAngle) {
            const auto angle = constantScalarValueOrThrow(
                contactLineAngleRadians(contact_line),
                "contact-line contact_angle_radians");
            constexpr FE::Real pi = FE::Real{3.14159265358979323846};
            if (!std::isfinite(angle) || !(angle > FE::Real{0.0}) ||
                !(angle < pi)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: contact-line contact_angle_radians must be finite and strictly in (0, pi); complete-wetting endpoints are not a transverse codimension-two contact configuration");
            }
            if (isUnfittedLevelSet(bc)) {
                if (bc.active_domain == FreeSurfaceActiveDomain::None) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require an active liquid side (LevelSetNegative or LevelSetPositive)");
                }
                if (normalizedFreeSurfaceOptionToken(
                        bc.generated_interface_geometry) != "linearcorner") {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles currently support only Generated_interface_geometry=LinearCorner; high-order implicit contact geometry is not yet validated");
                }
                const auto phi_id = resolveLevelSetFieldId(bc, system);
                const auto& phi_record = system.fieldRecord(phi_id);
                if (phi_record.components != 1 || !phi_record.space ||
                    phi_record.space->value_dimension() != 1) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact-angle level-set field must be scalar");
                }
                if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require the level-set field to be an unknown so its normal is coupled");
                }
                (void)owningOperatorTagForField(
                    system,
                    phi_id,
                    "prescribed contact-angle level-set field '" +
                        bc.level_set_field_name + "'");
                if (phi_record.space->polynomial_order() != 1) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles currently require an order-1 level-set space; generated contact geometry is reconstructed from linear corner values");
                }
                if (phi_record.space->continuity() != FE::Continuity::C0) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require a continuous C0 level-set space; discontinuous element-local corner values do not define one wall contact geometry");
                }
                const int generated_marker =
                    generatedUnfittedContactLineMarker(
                        contact_line,
                        bc,
                        system);
                if (contactLineMarker(contact_line) >= 0 &&
                    contactLineMarker(contact_line) != generated_marker) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle uses a generated interface-boundary marker; Contact_line_marker must be omitted or match the generated marker " +
                        std::to_string(generated_marker));
                }
                validateWallNormalDimension(
                    contact_line,
                    dim,
                    "prescribed contact angle");
            } else {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed fitted contact angles are unsupported until a true fitted contact-line (codimension-two) integration entity is available; the condition must not be integrated over the complete free-surface boundary");
            }
        }
        if (contactLineKind(contact_line) == ContactLineKind::DynamicRenE) {
            if (options.viscosity_model) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires literal Newtonian viscosity so accepted sharp-wall slip dissipation can be evaluated from the identical operator stage; constitutive viscosity models are unsupported for this contact law");
            }
            if (!isUnfittedLevelSet(bc)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle is currently supported only for sharp unfitted level-set free surfaces");
            }
            if (bc.active_domain == FreeSurfaceActiveDomain::None) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires an active liquid side (LevelSetNegative or LevelSetPositive)");
            }
            if (bc.active_domain_method !=
                FreeSurfaceActiveDomainMethod::CutVolume) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires Active_domain_method=CutVolume for a sharp liquid domain; SmoothedIndicator remains a diffuse diagnostic path");
            }
            if (bc.active_domain_smoothing_width != FE::Real{0.0}) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle with Active_domain_method=CutVolume requires active_domain_smoothing_width=0 because the sharp operator has no smoothing-width parameter");
            }
            if (normalizedFreeSurfaceOptionToken(
                    bc.generated_interface_geometry) != "linearcorner") {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently supports only Generated_interface_geometry=LinearCorner; high-order implicit contact geometry is not yet validated");
            }

            const auto phi_id = resolveLevelSetFieldId(bc, system);
            const auto& phi_record = system.fieldRecord(phi_id);
            if (phi_record.components != 1 || !phi_record.space ||
                phi_record.space->value_dimension() != 1) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle level-set field must be scalar");
            }
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires the level-set field to be an unknown so the authoritative interface, contact, and sharp wetted-wall geometry can be refreshed from the accepted state");
            }
            if (phi_record.space->polynomial_order() != 1) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires an order-1 level-set space; high-order contact-line geometry is unsupported");
            }
            if (phi_record.space->continuity() != FE::Continuity::C0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires a continuous C0 level-set space; discontinuous element-local corner values do not define one wall contact geometry");
            }

            const auto angle = constantScalarValueOrThrow(
                contactLineAngleRadians(contact_line),
                "DynamicContactAngle equilibrium contact_angle_radians");
            constexpr FE::Real pi = FE::Real{3.14159265358979323846};
            if (!std::isfinite(angle) || !(angle > FE::Real{0.0}) ||
                !(angle < pi)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle equilibrium contact_angle_radians must be finite and strictly in (0, pi); complete-wetting endpoints make the footprint direction singular");
            }
            const auto equilibrium_transverse_sine = std::sin(angle);
            if (!std::isfinite(equilibrium_transverse_sine) ||
                equilibrium_transverse_sine <
                    kMinimumContactTransverseSine) {
                std::ostringstream message;
                message
                    << "IncompressibleNavierStokesVMSModule: "
                       "DynamicContactAngle equilibrium angle is too close "
                       "to a complete-wetting endpoint for the supported "
                       "transverse generated contact geometry"
                    << " contact_angle_radians=" << angle
                    << " sin(theta_e)=" << equilibrium_transverse_sine
                    << " required_sin(theta_e)>="
                    << kMinimumContactTransverseSine
                    << " minimum_transverse_sine="
                    << kMinimumContactTransverseSine;
                throw std::invalid_argument(message.str());
            }
            (void)constantScalarValueOrThrow(
                bc.surface_tension,
                "DynamicContactAngle surface_tension");
            validatePositiveConstantScalar(
                bc.surface_tension,
                "DynamicContactAngle surface_tension");
            (void)constantScalarValueOrThrow(
                contactLineMobility(contact_line),
                "DynamicContactAngle contact-line mobility");
            validatePositiveConstantScalar(
                contactLineMobility(contact_line),
                "DynamicContactAngle contact-line mobility");
            (void)constantScalarValueOrThrow(
                contactLineSlipLength(contact_line),
                "DynamicContactAngle Navier slip_length");
            validatePositiveConstantScalar(
                contactLineSlipLength(contact_line),
                "DynamicContactAngle Navier slip_length");

            const int generated_marker = generatedUnfittedContactLineMarker(
                contact_line,
                bc,
                system);
            if (contactLineMarker(contact_line) >= 0 &&
                contactLineMarker(contact_line) != generated_marker) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle uses a generated interface-boundary marker; Contact_line_marker must be omitted or match the generated marker " +
                    std::to_string(generated_marker));
            }
            (void)normalizedWallNormal(contact_line);
            validateDynamicContactWallEssentialBC(options, contact_line, dim);
            validateDynamicContactPlanarWallMarker(system, contact_line);

        }
    }
}

void validateNavierStokesBoundaryConfiguration(
    const IncompressibleNavierStokesVMSOptions& options,
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    const FE::spaces::FunctionSpace& velocity_space,
    const FE::spaces::FunctionSpace& pressure_space,
    int dim,
    bool pspg_boundary_weak_form_enabled)
{
    using FE::forms::FormExpr;

    const auto generated_active_boundary_for =
        [&system, free_surfaces](int physical_boundary_marker) {
            return sharpActiveBoundaryMarkerFor(
                physical_boundary_marker, free_surfaces, system);
        };

    bool uses_sharp_active_boundary_form = false;
    const auto inspect_boundary_markers =
        [&](const auto& conditions) {
            for (const auto& condition : conditions) {
                uses_sharp_active_boundary_form =
                    uses_sharp_active_boundary_form ||
                    generated_active_boundary_for(
                        condition.boundary_marker).has_value();
            }
        };
    inspect_boundary_markers(options.traction_neumann);
    inspect_boundary_markers(options.traction_robin);
    inspect_boundary_markers(options.pressure_outflow);
    inspect_boundary_markers(options.coupled_outflow_rcr);
    inspect_boundary_markers(options.coupled_outflow_rcrcr);
    inspect_boundary_markers(options.velocity_dirichlet_weak);
    if (pspg_boundary_weak_form_enabled) {
        // PSPG boundary diagnostics are weak boundary forms even when their
        // marker originates from a strong velocity condition. They therefore
        // share the same generated-active quadrature qualification envelope.
        inspect_boundary_markers(options.velocity_dirichlet);
    }
    for (const auto& free_surface : free_surfaces) {
        if (!isUnfittedLevelSet(free_surface) ||
            free_surface.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        uses_sharp_active_boundary_form =
            uses_sharp_active_boundary_form ||
            std::any_of(
                free_surface.contact_lines.begin(),
                free_surface.contact_lines.end(),
                [](const auto& contact_line) {
                    return contactLineKind(contact_line) ==
                           ContactLineKind::DynamicRenE;
                });
    }
    if (uses_sharp_active_boundary_form) {
        if (velocity_space.polynomial_order() != 1 ||
            pressure_space.polynomial_order() != 1) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: sharp generated exterior-boundary forms currently require order-1 velocity and pressure spaces; high-order active-boundary quadrature and entity measures are not qualified");
        }
        for (const auto& free_surface : free_surfaces) {
            if (isUnfittedLevelSet(free_surface) &&
                free_surface.active_domain != FreeSurfaceActiveDomain::None &&
                normalizedFreeSurfaceOptionToken(
                    free_surface.generated_interface_geometry) !=
                    "linearcorner") {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: sharp generated exterior-boundary forms currently require Generated_interface_geometry=LinearCorner; high-order active-boundary quadrature and entity measures are not qualified");
            }
        }
    }

    FE::systems::BoundaryConditionManager velocity_conditions;
    velocity_conditions.install(options.traction_neumann, [&](const auto& bc) {
        for (const auto& value : bc.traction) {
            validateFiniteConstantScalar(value, "traction value");
        }
        return Factories::toTractionBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });
    velocity_conditions.install(options.traction_robin, [&](const auto& bc) {
        validateFiniteConstantScalar(bc.alpha, "Robin coefficient");
        for (const auto& value : bc.rhs) {
            validateFiniteConstantScalar(value, "Robin right-hand side");
        }
        return Factories::toTractionRobinBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });

    const auto u = FE::forms::StateField(
        FE::FieldId{0}, velocity_space, "u_boundary_preflight");
    const auto p = FE::forms::StateField(
        FE::FieldId{1}, pressure_space, "p_boundary_preflight");
    const auto v = FE::forms::TestField(
        FE::FieldId{0}, velocity_space, "v_boundary_preflight");
    const auto q = FE::forms::TestField(
        FE::FieldId{1}, pressure_space, "q_boundary_preflight");
    const auto rho = FormExpr::constant(options.density);

    velocity_conditions.install(options.pressure_outflow, [&](const auto& bc) {
        validateFiniteConstantScalar(bc.pressure, "outflow pressure");
        validateNonnegativeConstantScalar(
            bc.backflow_beta, "outflow backflow coefficient");
        return Factories::toOutflowBC(
            bc,
            u,
            rho,
            generated_active_boundary_for(bc.boundary_marker));
    });

    std::vector<FormExpr> zero_components;
    zero_components.reserve(static_cast<std::size_t>(dim));
    for (int component = 0; component < dim; ++component) {
        zero_components.push_back(FormExpr::constant(FE::Real{0.0}));
    }
    const auto zero_flux = FormExpr::asVector(std::move(zero_components));
    const auto validate_finite_real = [](FE::Real value,
                                         std::string_view label) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(label) + " must be finite");
        }
    };
    for (const auto& bc : options.coupled_outflow_rcr) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "IncompressibleNavierStokesVMSModule coupled RCR preflight");
        for (const auto [value, label] : {
                 std::pair{bc.Rp, std::string_view{"RCR Rp"}},
                 std::pair{bc.C, std::string_view{"RCR C"}},
                 std::pair{bc.Rd, std::string_view{"RCR Rd"}},
                 std::pair{bc.Pd, std::string_view{"RCR Pd"}},
                 std::pair{bc.X0, std::string_view{"RCR initial state"}}}) {
            validate_finite_real(value, label);
        }
        if (bc.Rd == FE::Real{0.0}) {
            throw std::invalid_argument(
                "CoupledRCROutflowBC: Rd must be nonzero");
        }
        validateNonnegativeConstantScalar(
            bc.backflow_beta, "RCR backflow coefficient");
        velocity_conditions.add(
            std::make_unique<FE::forms::bc::ReservedBC>(marker));
    }
    for (const auto& bc : options.coupled_outflow_rcrcr) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "IncompressibleNavierStokesVMSModule coupled RCRCR preflight");
        for (const auto [value, label] : {
                 std::pair{bc.Rp, std::string_view{"RCRCR Rp"}},
                 std::pair{bc.C1, std::string_view{"RCRCR C1"}},
                 std::pair{bc.Rm, std::string_view{"RCRCR Rm"}},
                 std::pair{bc.C2, std::string_view{"RCRCR C2"}},
                 std::pair{bc.Rd, std::string_view{"RCRCR Rd"}},
                 std::pair{bc.Pd, std::string_view{"RCRCR Pd"}},
                 std::pair{bc.P10, std::string_view{"RCRCR P1 initial state"}},
                 std::pair{bc.P20, std::string_view{"RCRCR P2 initial state"}}}) {
            validate_finite_real(value, label);
        }
        if (bc.C1 == FE::Real{0.0} || bc.C2 == FE::Real{0.0}) {
            throw std::invalid_argument(
                "CoupledRCRCROutflowBC: C1 and C2 must be nonzero");
        }
        if (bc.Rm == FE::Real{0.0} || bc.Rd == FE::Real{0.0}) {
            throw std::invalid_argument(
                "CoupledRCRCROutflowBC: Rm and Rd must be nonzero");
        }
        validateNonnegativeConstantScalar(
            bc.backflow_beta, "RCRCR backflow coefficient");
        velocity_conditions.add(
            std::make_unique<FE::forms::bc::ReservedBC>(marker));
    }

    velocity_conditions.install(options.velocity_dirichlet, [&](const auto& bc) {
        for (const auto& value : bc.value) {
            validateFiniteConstantScalar(value, "velocity Dirichlet value");
        }
        return Factories::toVelocityEssentialBC(
            bc, dim, options.velocity_field_name);
    });
    for (const auto& bc : options.velocity_dirichlet_weak) {
        for (const auto& value : bc.value) {
            validateFiniteConstantScalar(
                value, "weak velocity Dirichlet value");
        }
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "IncompressibleNavierStokesVMSModule weak velocity preflight");
        velocity_conditions.add(
            std::make_unique<FE::forms::bc::ReservedBC>(marker));
    }
    for (const auto& bc : free_surfaces) {
        if (bc.implementation == FreeSurfaceImplementation::FittedALE) {
            velocity_conditions.add(
                std::make_unique<FE::forms::bc::ReservedBC>(
                    bc.boundary_marker));
        }
    }
    velocity_conditions.validate();

    FE::systems::BoundaryConditionManager pressure_conditions;
    pressure_conditions.install(options.pressure_dirichlet, [&](const auto& bc) {
        validateFiniteConstantScalar(bc.value, "pressure Dirichlet value");
        return Factories::toPressureEssentialBC(
            bc, options.pressure_field_name);
    });
    pressure_conditions.validate();

    FormExpr momentum_preflight;
    FormExpr continuity_preflight;
    std::vector<
        FE::forms::bc::
            GeneratedBoundaryNitscheTraceFormBinding>
        generated_nitsche_trace_binding_preflight;
    Factories::applyVelocityNitscheBCs(
        momentum_preflight,
        continuity_preflight,
        options,
        dim,
        u,
        p,
        v,
        q,
        FormExpr::constant(options.viscosity),
        generated_active_boundary_for,
        nullptr,
        &generated_nitsche_trace_binding_preflight);
    std::size_t expected_generated_nitsche_trace_bindings = 0u;
    for (const auto& bc : options.velocity_dirichlet_weak) {
        const int marker =
            FE::forms::bc::detail::boundaryMarkerOrThrow(
                bc,
                "IncompressibleNavierStokesVMSModule generated-boundary "
                "Nitsche binding preflight");
        if (generated_active_boundary_for(marker).has_value()) {
            ++expected_generated_nitsche_trace_bindings;
        }
    }
    if (generated_nitsche_trace_binding_preflight.size() !=
        expected_generated_nitsche_trace_bindings) {
        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: generated-boundary "
            "Nitsche binding preflight count differs from the configured "
            "production routes");
    }
}

void warnUnfittedRawCurvatureIfNeeded(const FreeSurfaceBoundary& bc)
{
    if (!isUnfittedLevelSet(bc) ||
        usesSurfaceStress(bc) ||
        FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension) ||
        !bc.use_level_set_curvature ||
        !bc.curvature_field_name.empty()) {
        return;
    }

    FE_LOG_WARNING(
        std::string("IncompressibleNavierStokesVMSModule: unfitted level-set surface tension is using raw level-set curvature") +
        " marker=" + std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " generated_interface_domain_id='" + bc.generated_interface_domain_id + "'" +
        " diagnostic=unfitted_level_set_raw_curvature"
        " recommendation=use zero surface tension, prescribed curvature, or projected/smoothed curvature for verification cases");
}

[[nodiscard]] FE::forms::FormExpr unfittedInterfaceNormal(
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& phi)
{
    auto n = FE::forms::unitNormalFromLevelSet(phi);
    if (bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        return -n;
    }
    return n;
}

void logUnfittedContactLineMeasure(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext*
        cut_context,
    const FreeSurfaceBoundary& bc,
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    int contact_marker)
{
    if (!isUnfittedLevelSet(bc)) {
        return;
    }

    std::size_t rule_count = 0u;
    std::size_t qpoint_count = 0u;
    std::size_t reference_rule_count = 0u;
    std::size_t current_rule_count = 0u;
    FE::Real reference_measure = 0.0;
    FE::Real current_measure = 0.0;
    FE::Real angle_sample_measure = 0.0;
    FE::Real weighted_cos_sum = 0.0;
    FE::Real weighted_angle_sum = 0.0;
    FE::Real weighted_abs_angle_error_sum = 0.0;
    FE::Real weighted_boundary_normal_alignment_sum = 0.0;
    FE::Real max_abs_cos_gap = 0.0;
    FE::Real max_abs_angle_error = 0.0;
    FE::Real min_angle = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_angle = -std::numeric_limits<FE::Real>::infinity();
    std::size_t angle_sample_count = 0u;
    const auto wall_n = normalizedWallNormal(contact_line);
    const auto target_angle = constantScalarValueOrThrow(
        contactLineAngleRadians(contact_line),
        "contact-line contact_angle_radians");
    const auto desired_cos = std::cos(target_angle);
    const bool context_present = cut_context != nullptr;
    const bool marker_available =
        cut_context != nullptr &&
        cut_context->hasGeneratedInterfaceMarker(contact_marker);
    if (marker_available) {
        cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
        const auto rules = cut_context->interfaceRulesForMarker(contact_marker);
        rule_count = rules.size();
        for (const auto* rule : rules) {
            if (rule == nullptr) {
                continue;
            }
            qpoint_count += rule->points.size();
            if (rule->frame == FE::geometry::CutGeometryFrame::Reference) {
                ++reference_rule_count;
                reference_measure += std::abs(rule->measure);
                // Reference normals and d-2 weights cannot be interpreted as
                // physical contact angles or measures before the cell map is
                // applied. StandardAssembler performs that mapping exactly
                // once while consuming the prescribed contact geometry.
                continue;
            }
            ++current_rule_count;
            current_measure += std::abs(rule->measure);
            for (const auto& point : rule->points) {
                const auto weight = std::abs(point.weight);
                if (!(weight > FE::Real{0.0}) || !std::isfinite(weight)) {
                    continue;
                }
                const auto n =
                    activeInterfaceNormalForContactAngle(bc, point.normal);
                if (!(norm3(n) > FE::Real{0.0})) {
                    continue;
                }
                angle_sample_measure += weight;
                // wall_n points out of the liquid and into the solid. The
                // through-liquid angle is therefore acos(-n.wall_n), and the
                // Young geometry residual is n.wall_n + cos(theta).
                const auto normal_dot_wall = clampUnit(dot3(n, wall_n));
                const auto cos_theta = -normal_dot_wall;
                const auto angle = std::acos(cos_theta);
                const auto cos_gap = normal_dot_wall + desired_cos;
                const auto angle_error = angle - target_angle;
                const auto boundary_n =
                    normalizedOrZero(point.boundary_normal);
                weighted_cos_sum += weight * cos_theta;
                weighted_angle_sum += weight * angle;
                weighted_abs_angle_error_sum += weight * std::abs(angle_error);
                weighted_boundary_normal_alignment_sum +=
                    weight * dot3(boundary_n, wall_n);
                max_abs_cos_gap = std::max(max_abs_cos_gap, std::abs(cos_gap));
                max_abs_angle_error =
                    std::max(max_abs_angle_error, std::abs(angle_error));
                min_angle = std::min(min_angle, angle);
                max_angle = std::max(max_angle, angle);
                ++angle_sample_count;
            }
        }
    }

    constexpr FE::Real radians_to_degrees =
        FE::Real{180.0} /
        FE::Real{3.141592653589793238462643383279502884};
    const auto weighted_measure = angle_sample_measure;
    const bool has_angle_samples =
        angle_sample_count > 0u && weighted_measure > FE::Real{0.0};
    const auto mean_cos =
        has_angle_samples ? weighted_cos_sum / weighted_measure
                          : std::numeric_limits<FE::Real>::quiet_NaN();
    const auto mean_angle =
        has_angle_samples ? weighted_angle_sum / weighted_measure
                          : std::numeric_limits<FE::Real>::quiet_NaN();
    const auto mean_abs_angle_error =
        has_angle_samples ? weighted_abs_angle_error_sum / weighted_measure
                          : std::numeric_limits<FE::Real>::quiet_NaN();
    const auto mean_boundary_normal_alignment =
        has_angle_samples
            ? weighted_boundary_normal_alignment_sum / weighted_measure
            : std::numeric_limits<FE::Real>::quiet_NaN();
    if (!has_angle_samples) {
        min_angle = std::numeric_limits<FE::Real>::quiet_NaN();
        max_angle = std::numeric_limits<FE::Real>::quiet_NaN();
    }
    const bool physical_metrics_available =
        reference_rule_count == 0u && current_rule_count > 0u;
    const auto physical_measure = physical_metrics_available
                                      ? current_measure
                                      : std::numeric_limits<FE::Real>::quiet_NaN();

    std::ostringstream msg;
    msg << "IncompressibleNavierStokesVMSModule: prescribed unfitted contact "
           "angle uses generated interface-boundary intersection geometry"
        << " diagnostic=unfitted_prescribed_contact_geometry"
        << " level_set_geometry_owner=accepted_state_wall_aware_repair"
        << " momentum_owner=young_wall_energy"
        << " literal_codimension_two_level_set_residual=retired"
        << " interface_marker=" << bc.interface_marker
        << " wall_boundary_marker=" << contactLineWallBoundaryMarker(contact_line)
        << " contact_line_marker=" << contact_marker
        << " generated_interface_domain_id='"
        << bc.generated_interface_domain_id << "'"
        << " angle_convention=through_liquid"
        << " wall_normal_convention=outward_from_fluid_into_solid"
        << " cut_context=" << (context_present ? "present" : "missing")
        << " marker_status=" << (marker_available ? "available" : "missing")
        << " rules=" << rule_count
        << " reference_rules=" << reference_rule_count
        << " current_rules=" << current_rule_count
        << " qpoints=" << qpoint_count
        << " reference_measure=" << static_cast<double>(reference_measure)
        << " physical_measure=" << static_cast<double>(physical_measure)
        << " physical_metric_status="
        << (physical_metrics_available ? "available"
                                       : "requires_assembly_geometry_mapping")
        << " target_angle_degrees="
        << static_cast<double>(target_angle * radians_to_degrees)
        << " target_through_liquid_cos=" << static_cast<double>(desired_cos)
        << " angle_samples=" << angle_sample_count
        << " mean_cos=" << static_cast<double>(mean_cos)
        << " mean_cos_gap=" << static_cast<double>(mean_cos - desired_cos)
        << " max_abs_cos_gap=" << static_cast<double>(max_abs_cos_gap)
        << " mean_angle_degrees="
        << static_cast<double>(mean_angle * radians_to_degrees)
        << " min_angle_degrees="
        << static_cast<double>(min_angle * radians_to_degrees)
        << " max_angle_degrees="
        << static_cast<double>(max_angle * radians_to_degrees)
        << " mean_abs_angle_error_degrees="
        << static_cast<double>(mean_abs_angle_error * radians_to_degrees)
        << " max_abs_angle_error_degrees="
        << static_cast<double>(max_abs_angle_error * radians_to_degrees)
        << " wall_normal_dot_boundary_normal_mean="
        << static_cast<double>(mean_boundary_normal_alignment);

    FE_LOG_INFO(
        msg.str());
}

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept;
[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept;
[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept;
[[nodiscard]] FE::forms::CutVolumeSide activeDomainSide(
    FreeSurfaceActiveDomain domain) noexcept;

void applyFreeSurfaceCutCellStabilization(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu,
    FE::Real density,
    FE::Real stabilization_epsilon,
    int velocity_components,
    int velocity_polynomial_order,
    int pressure_polynomial_order,
    FE::forms::FormExpr* pressure_stabilization_form = nullptr)
{
    if (!isUnfittedLevelSet(bc) || !bc.cut_cell_stabilization.enabled) {
        return;
    }

    namespace bc_forms = FE::forms::bc;
    const auto& cut = bc.cut_cell_stabilization;
    constexpr int supported_derivative_order = 2;
    const auto derivative_order_label = [](int max_order) -> const char* {
        if (max_order <= 0) {
            return "disabled";
        }
        return max_order > 1 ? "1,2" : "1";
    };
    const bool pressure_stabilization_enabled =
        pressureStabilizationActive(bc);
    const bool pressure_stabilization_incremental =
        pressure_stabilization_enabled &&
        cut.pressure_policy ==
            FreeSurfacePressureStabilizationPolicy::Incremental;
    const int pressure_derivative_order =
        pressure_stabilization_enabled
            ? (pressure_stabilization_incremental
                   ? 1
                   : (pressure_polynomial_order > 1
                          ? supported_derivative_order
                          : 1))
            : 0;
    const int max_derivative_order = pressure_derivative_order;
    const bool has_unsupported_derivative_order =
        pressure_stabilization_enabled &&
        !pressure_stabilization_incremental &&
        pressure_polynomial_order > supported_derivative_order;
    auto cut_scale = cut.use_cut_metadata_scale
        ? FE::forms::cutStabilizationScale()
        : FE::forms::FormExpr::constant(1.0);
    if (cut.use_cut_metadata_scale && cut.cut_metadata_scale_cap.has_value()) {
        cut_scale = FE::forms::min(
            cut_scale,
            FE::forms::FormExpr::constant(*cut.cut_metadata_scale_cap));
    }
    const auto h_f = FE::forms::avg(FE::forms::hNormal());
    const auto h3 = h_f * h_f * h_f;
    const auto h5 = h3 * h_f * h_f;
    // Transient ghost-penalty coefficient. The viscous scale mu alone
    // under-stabilizes inertia-dominated cut supports (rho*h^2/dt >> mu for
    // water-like parameters), so the penalty carries the generalized
    // diffusivity mu_gp = mu + rho*h^2/dt used by transient CutFEM analyses
    // (Schott & Wall; Burman, Fernandez & Massing). Velocity jump penalties
    // scale with mu_gp, pressure jump penalties with 1/mu_gp.
    const auto rho_gp = FE::forms::FormExpr::constant(density);
    const auto mu_gp =
        mu + rho_gp * h_f * h_f / FE::forms::FormExpr::effectiveTimeStep();
    // Dimensionless ghost-penalty calibration for the analytic transient
    // scaling: 0.01*(mu+rho*h^2/dt)*h (gradient jump) and the matching
    // h^3/h^5 variants. The eigenproblem-calibrated alternative was
    // retired: its coercivity-sufficient coefficients measured 2-6 orders
    // above the Newton-conditioning ceiling on the SPHERIC wet-bed cases
    // (the workable small-gamma regime is sub-coercive), and small-cut
    // aggregation now replaces the velocity penalty altogether for
    // vertex-covered (P1/iso-Q2) spaces. See
    // Documentation/plan_ghost_penalty_eigen_calibration_20260611.md.
    constexpr FE::Real kCutPenaltyTransientCalibration{0.01};
    const auto gp_calibration =
        FE::forms::FormExpr::constant(kCutPenaltyTransientCalibration);
    const auto interface_side =
        bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive
            ? "Plus"
            : (bc.active_domain == FreeSurfaceActiveDomain::LevelSetNegative
                   ? "Minus"
                   : "All");
    const auto active_domain_side =
        bc.active_domain == FreeSurfaceActiveDomain::None
            ? "FullDomain"
            : cutVolumeSideName(activeDomainSide(bc.active_domain));

    std::ostringstream oss;
    oss << "IncompressibleNavierStokesVMSModule: cut-cell stabilization "
        << "marker=" << bc.interface_marker
        << " level_set_field='" << bc.level_set_field_name << "'"
        << " interface_side=" << interface_side
        << " active_domain=" << activeDomainName(bc.active_domain)
        << " active_domain_side=" << active_domain_side
        << " Active_domain_method="
        << activeDomainMethodName(bc.active_domain_method)
        << " generated_interface_geometry="
        << bc.generated_interface_geometry
        << " geometry_tangent_policy="
        << bc.geometry_tangent_policy
        << " use_cut_metadata_scale="
        << (cut.use_cut_metadata_scale ? "true" : "false")
        << " cut_metadata_scale_cap=";
    if (cut.cut_metadata_scale_cap.has_value()) {
        oss << *cut.cut_metadata_scale_cap;
    } else {
        oss << "unbounded";
    }
    oss
        << " pressure_stabilization_policy="
        << pressureStabilizationPolicyName(cut.pressure_policy)
        << " pressure_stabilization="
        << (pressure_stabilization_enabled ? "enabled" : "disabled")
        << " pressure_stabilization_form="
        << (pressure_stabilization_enabled
                ? (pressure_stabilization_incremental ? "incremental"
                                                      : "absolute")
                : "disabled")
        << " pressure_disabled_reason="
        << pressureStabilizationDisabledReason(bc)
        << " facet_scope=cut-adjacent"
        << " velocity_polynomial_order=" << velocity_polynomial_order
        << " pressure_polynomial_order=" << pressure_polynomial_order
        << " derivative_orders=" << derivative_order_label(max_derivative_order)
        << " pressure_derivative_orders="
        << derivative_order_label(pressure_derivative_order)
        << " cut_penalty_transient_calibration="
        << kCutPenaltyTransientCalibration
        << " small_cut_aggregation="
        << (bc.small_cut_aggregation ? "true" : "false")
        << " velocity_ghost_penalty_mode=retired_replaced_by_aggregation"
        << " pressure_scaling="
        << (pressure_derivative_order <= 0
                ? "disabled"
                : (pressure_derivative_order > 1
                       ? "0.01*h^3/(mu+rho*h^2/dt),0.01*h^5/(mu+rho*h^2/dt)"
                       : "0.01*h^3/(mu+rho*h^2/dt)"));
    FE_LOG_INFO(oss.str());

    if (has_unsupported_derivative_order) {
        FE_LOG_WARNING(
            "IncompressibleNavierStokesVMSModule: high-order cut-cell "
            "stabilization currently supports derivative_orders=1,2; "
            "higher-normal-derivative penalties above derivative_order=" +
            std::to_string(supported_derivative_order) +
            " are not yet available");
    }

    if (pressure_stabilization_enabled) {
        const auto pressure_penalty = bc_forms::toScalarExpr(
            cut.pressure_gradient_penalty,
            freeSurfaceValueName("ns_free_surface_cut_pressure_penalty", bc));
        const auto stabilized_pressure =
            pressure_stabilization_incremental
                ? FE::forms::FormExpr::effectiveTimeStep() * FE::forms::dt(p)
                : p;
        const auto pressure_jump_p =
            FE::forms::cutAdjacentFacetGradientJump(stabilized_pressure);
        const auto pressure_jump_q =
            FE::forms::cutAdjacentFacetGradientJump(q);
        auto pressure_form =
            FE::forms::cutAdjacentFacetIntegral(
                cut_scale * pressure_penalty * gp_calibration * h3 /
                    (mu_gp + FE::forms::FormExpr::constant(stabilization_epsilon)) *
                    FE::forms::inner(pressure_jump_p, pressure_jump_q),
                bc.interface_marker);

        if (pressure_derivative_order > 1) {
            const auto pressure_second_jump_p =
                FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(
                    stabilized_pressure);
            const auto pressure_second_jump_q =
                FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(q);
            pressure_form =
                pressure_form +
                FE::forms::cutAdjacentFacetIntegral(
                    cut_scale * pressure_penalty * gp_calibration * h5 /
                        (mu_gp + FE::forms::FormExpr::constant(stabilization_epsilon)) *
                        pressure_second_jump_p * pressure_second_jump_q,
                    bc.interface_marker);
        }
        continuity_form = continuity_form + pressure_form;
        if (pressure_stabilization_form != nullptr) {
            *pressure_stabilization_form =
                pressure_stabilization_form->isValid()
                    ? (*pressure_stabilization_form + pressure_form)
                    : pressure_form;
        }
    }
}

[[nodiscard]] FE::forms::FormExpr integrateOnFreeSurface(
    const FE::forms::FormExpr& integrand,
    const FreeSurfaceBoundary& bc,
    bool ale_enabled)
{
    if (isUnfittedLevelSet(bc)) {
        return integrand.dI(bc.interface_marker);
    }
    const auto weighted_integrand =
        useFittedCurrentGeometry(bc, ale_enabled)
            ? integrand * FE::forms::currentMeasure()
            : integrand;
    return weighted_integrand.dExteriorBoundary(
        FE::forms::ExteriorBoundaryMeasure::fullPhysical(
            bc.boundary_marker));
}

struct ActiveVolumeDomain {
    int interface_marker{-1};
    FE::forms::CutVolumeSide side{FE::forms::CutVolumeSide::Negative};
    FreeSurfaceActiveDomainMethod method{FreeSurfaceActiveDomainMethod::CutVolume};
    FE::forms::FormExpr indicator{};
    FE::forms::FormExpr cut_volume_shape_tangent_factor{};
};

[[nodiscard]] FE::forms::FormExpr freeSurfaceLevelSet(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept
{
    switch (method) {
    case FreeSurfaceActiveDomainMethod::CutVolume:
        return "CutVolume";
    case FreeSurfaceActiveDomainMethod::SmoothedIndicator:
        return "SmoothedIndicator";
    }
    return "Unknown";
}

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept
{
    switch (enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return "None";
    case FreeSurfaceKinematicEnforcement::Penalty:
        return "Penalty";
    case FreeSurfaceKinematicEnforcement::Nitsche:
        return "Nitsche";
    }
    return "Unknown";
}

[[nodiscard]] const char* fieldSourceKindName(
    FE::systems::FieldSourceKind source_kind) noexcept
{
    switch (source_kind) {
    case FE::systems::FieldSourceKind::Unknown:
        return "Unknown";
    case FE::systems::FieldSourceKind::PrescribedData:
        return "PrescribedData";
    case FE::systems::FieldSourceKind::DerivedFromUnknown:
        return "DerivedFromUnknown";
    }
    return "Unknown";
}

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative ? "Negative" : "Positive";
}

[[nodiscard]] FE::forms::CutVolumeSide activeDomainSide(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return FE::forms::CutVolumeSide::Negative;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return FE::forms::CutVolumeSide::Positive;
    }
    return FE::forms::CutVolumeSide::Negative;
}

[[nodiscard]] FE::forms::CutVolumeSide oppositeCutVolumeSide(
    FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative
               ? FE::forms::CutVolumeSide::Positive
               : FE::forms::CutVolumeSide::Negative;
}

[[nodiscard]] FE::forms::FormExpr cutVolumeLevelSetShapeTangentFactor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    FE::forms::CutVolumeSide side,
    std::string_view domain_role)
{
    if (!isUnfittedLevelSet(bc) ||
        bc.active_domain == FreeSurfaceActiveDomain::None ||
        bc.active_domain_method != FreeSurfaceActiveDomainMethod::CutVolume ||
        unfittedLevelSetShapeTangentsDisabled()) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (rec.source_kind == FE::systems::FieldSourceKind::PrescribedData) {
        return FE::forms::FormExpr{};
    }
    if (!system.fieldParticipatesInUnknownVector(phi_id)) {
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: unfitted cut-volume active domain "
            << "references a non-prescribed level-set field that is not an unknown; "
            << "no level-set domain tangent will be assembled "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << rec.name << "'"
            << " level_set_source_kind=" << fieldSourceKindName(rec.source_kind)
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " side=" << cutVolumeSideName(side)
            << " domain_role=" << domain_role
            << " diagnostic=unfitted_free_surface_cut_volume_phi_tangent_unavailable";
        FE_LOG_WARNING(oss.str());
        return FE::forms::FormExpr{};
    }

    const auto sign = side == FE::forms::CutVolumeSide::Negative
        ? FE::Real{-1.0}
        : FE::Real{1.0};
    const auto phi =
        FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    const auto dphi =
        FE::forms::FormExpr::trialFunction(*rec.space, "d" + rec.name);
    const auto grad_phi = FE::forms::grad(phi);
    const auto grad_norm =
        FE::forms::sqrt(FE::forms::inner(grad_phi, grad_phi) +
                        FE::forms::FormExpr::constant(FE::Real{1.0e-30}));

    std::ostringstream oss;
    oss << "IncompressibleNavierStokesVMSModule: adding unfitted cut-volume "
        << "level-set Hadamard shape tangent factor "
        << "marker=" << bc.interface_marker
        << " level_set_field='" << rec.name << "'"
        << " level_set_source_kind=" << fieldSourceKindName(rec.source_kind)
        << " Active_domain=" << activeDomainName(bc.active_domain)
        << " Active_domain_method="
        << activeDomainMethodName(bc.active_domain_method)
        << " side=" << cutVolumeSideName(side)
        << " domain_role=" << domain_role
        << " sign=" << sign
        << " experimental_path=unfitted_level_set_shape_tangent"
        << " qualification=Experimental"
        << " diagnostic=unfitted_free_surface_cut_volume_phi_shape_tangent";
    FE_LOG_INFO(oss.str());

    return FE::forms::FormExpr::constant(sign) * dphi / grad_norm;
}

[[nodiscard]] FE::forms::FormExpr activeDomainIndicatorFor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto signed_phi =
        bc.active_domain == FreeSurfaceActiveDomain::LevelSetNegative
            ? FE::forms::FormExpr::constant(bc.level_set_isovalue) - phi
            : phi - FE::forms::FormExpr::constant(bc.level_set_isovalue);
    const auto width = bc.active_domain_smoothing_width > FE::Real{0.0}
        ? FE::forms::FormExpr::constant(bc.active_domain_smoothing_width)
        : FE::forms::h();
    return FE::forms::smoothHeaviside(signed_phi, width);
}

void applyDynamicContactAngleResidual(
    FE::forms::FormExpr& momentum_form,
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& mu,
    int dim,
    FE::forms::FormExpr* conservative_surface_energy_form = nullptr)
{
    using namespace FE::forms;

    for (const auto& contact_line : bc.contact_lines) {
        if (contactLineKind(contact_line) ==
                ContactLineKind::PrescribedAngle &&
            !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
            // Accepted-state wall-aware repair enforces the geometric angle,
            // while the momentum equation still needs the solid--liquid/
            // solid--gas wall force.  SurfaceStress already supplies the
            // dynamic conormal and therefore adds only -gamma*cos(theta_e);
            // CurvatureTraction needs the full cos(theta_d)-cos(theta_e) line
            // force.  If all wall-tangential test traces are essential, either
            // form performs no virtual work and contributes only to the
            // reaction.
            const auto phi_id = resolveLevelSetFieldId(bc, system);
            const auto& phi_record = system.fieldRecord(phi_id);
            const auto phi = StateField(
                phi_id,
                *phi_record.space,
                bc.level_set_field_name);
            const auto n = usesSurfaceStress(bc)
                               ? generatedInterfaceOutwardNormal(bc)
                               : unfittedInterfaceNormal(bc, phi);
            const auto wall_n = wallNormalExpression(contact_line, dim);
            const auto footprint_direction =
                safeNormalize(n - dot(n, wall_n) * wall_n);
            const auto contact_test = dot(v, footprint_direction);
            const auto gamma = FE::forms::bc::toScalarExpr(
                bc.surface_tension,
                freeSurfaceValueName(
                    "ns_prescribed_contact_angle_surface_tension", bc));
            const auto equilibrium_cosine = FormExpr::constant(std::cos(
                constantScalarValueOrThrow(
                    contactLineAngleRadians(contact_line),
                    "PrescribedContactAngle contact_angle_radians")));
            const int contact_marker = generatedUnfittedContactLineMarker(
                contact_line,
                bc,
                system);
            const auto wall_energy_force = usesSurfaceStress(bc)
                ? -gamma * equilibrium_cosine
                : -gamma * (equilibrium_cosine + dot(n, wall_n));
            const auto wall_energy_form = dInterfaceBoundary(
                wall_energy_force * contact_test,
                contact_marker);
            momentum_form = momentum_form + wall_energy_form;
            if (conservative_surface_energy_form != nullptr &&
                usesSurfaceStress(bc)) {
                *conservative_surface_energy_form =
                    conservative_surface_energy_form->isValid()
                        ? (*conservative_surface_energy_form + wall_energy_form)
                        : wall_energy_form;
            }
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: installed prescribed-angle wall-energy momentum term") +
                " interface_marker=" + std::to_string(bc.interface_marker) +
                " contact_marker=" + std::to_string(contact_marker) +
                " surface_tension_form=" + surfaceTensionFormName(bc) +
                " contact_force_discretization=" +
                (usesSurfaceStress(bc)
                     ? "surface_conormal_plus_equilibrium_wall_energy"
                     : "explicit_dynamic_angle_gap") +
                " diagnostic=navier_stokes_prescribed_contact_wall_energy");
            continue;
        }
        if (contactLineKind(contact_line) !=
            ContactLineKind::DynamicRenE) {
            continue;
        }

        const auto phi_id = resolveLevelSetFieldId(bc, system);
        const auto& phi_record = system.fieldRecord(phi_id);
        const auto phi = StateField(
            phi_id,
            *phi_record.space,
            bc.level_set_field_name);
        // SurfaceStress must take its contact geometry from the same mapped
        // generated rule as the surface measure.  The legacy curvature form
        // retains its Q1-gradient normal and full Ren--E angle gap.
        const auto n = usesSurfaceStress(bc)
                           ? generatedInterfaceOutwardNormal(bc)
                           : unfittedInterfaceNormal(bc, phi);
        const auto wall_n = wallNormalExpression(contact_line, dim);
        // n is the outward liquid normal.  Its wall-tangential projection
        // points out of the wetted footprint for either contact orientation.
        const auto footprint_direction =
            safeNormalize(n - dot(n, wall_n) * wall_n);
        const auto contact_velocity = dot(u, footprint_direction);
        const auto contact_test = dot(v, footprint_direction);

        const auto gamma = FE::forms::bc::toScalarExpr(
            bc.surface_tension,
            freeSurfaceValueName(
                "ns_dynamic_contact_angle_surface_tension", bc));
        const auto mobility = FE::forms::bc::toScalarExpr(
            contactLineMobility(contact_line),
            freeSurfaceValueName(
                "ns_dynamic_contact_angle_mobility", bc));
        const auto line_friction = FormExpr::constant(1.0) / mobility;
        const auto equilibrium_cosine = FormExpr::constant(std::cos(
            constantScalarValueOrThrow(
                contactLineAngleRadians(contact_line),
                "DynamicContactAngle equilibrium contact_angle_radians")));
        // With explicit kappa*n traction, the line term contains the complete
        // Ren--E gap cos(theta_e)-cos(theta_d).  With SurfaceStress, its
        // +gamma*cos(theta_d) conormal force is already contained in
        // gamma*P:grad(v), so the separate wall term is only
        // -gamma*cos(theta_e).  Keeping young_gap there double-counts the
        // dynamic-angle force.
        const auto contact_force = usesSurfaceStress(bc)
            ? (line_friction * contact_velocity -
               gamma * equilibrium_cosine)
            : (line_friction * contact_velocity -
               gamma * (equilibrium_cosine + dot(n, wall_n)));
        const int contact_marker = generatedUnfittedContactLineMarker(
            contact_line,
            bc,
            system);
        system.registerGeneratedEmbeddedInterfaceMarker(contact_marker);
        system.addCutIntegrationContextUpdateCallback(
            FE::systems::CutIntegrationContextUpdateCallback{
                .name = "navier_stokes_dynamic_contact_wall_normal:" +
                        std::to_string(contact_marker),
                .callback =
                    [&system, bc, contact_line, contact_marker](
                        const FE::assembly::CutIntegrationContext*
                            cut_context) {
                        if (!validateContactWallNormalCallbackPreflight(
                                system,
                                cut_context,
                                contact_line,
                                contact_marker,
                                cut_context != nullptr &&
                                    system.isSetup(),
                                "dynamic_contact_operator_angle")) {
                            return;
                        }
                        logDynamicContactOperatorAngle(
                            system,
                            cut_context,
                            bc,
                            contact_line,
                            contact_marker);
                    }});
        const auto* current_cut_context =
            system.cutIntegrationContext();
        if (validateContactWallNormalCallbackPreflight(
                system,
                current_cut_context,
                contact_line,
                contact_marker,
                current_cut_context != nullptr &&
                    system.isSetup(),
                "dynamic_contact_operator_angle_initial")) {
            logDynamicContactOperatorAngle(
                system,
                current_cut_context,
                bc,
                contact_line,
                contact_marker);
        }
        momentum_form = momentum_form + dInterfaceBoundary(
            contact_force * contact_test,
            contact_marker);
        if (conservative_surface_energy_form != nullptr &&
            usesSurfaceStress(bc)) {
            // Only the Young wall-energy variation belongs to the
            // conservative split.  The line-friction and wetted-wall Navier
            // terms are dissipative and must remain absent from this
            // diagnostic even away from equilibrium.
            const auto wall_energy_form = dInterfaceBoundary(
                (-gamma * equilibrium_cosine) * contact_test,
                contact_marker);
            *conservative_surface_energy_form =
                conservative_surface_energy_form->isValid()
                    ? (*conservative_surface_energy_form + wall_energy_form)
                    : wall_energy_form;
        }

        const auto slip_length = FE::forms::bc::toScalarExpr(
            contactLineSlipLength(contact_line),
            freeSurfaceValueName(
                "ns_dynamic_contact_angle_slip_length", bc));
        const auto u_tangent = u - dot(u, wall_n) * wall_n;
        const auto v_tangent = v - dot(v, wall_n) * wall_n;
        const int active_wall_marker = generatedActiveBoundaryMarker(
            bc, contactLineWallBoundaryMarker(contact_line), system);
        system.registerGeneratedEmbeddedInterfaceMarker(active_wall_marker);
        momentum_form = momentum_form +
            (mu / slip_length * dot(u_tangent, v_tangent))
                .dExteriorBoundary(
                    FE::forms::ExteriorBoundaryMeasure::
                        generatedActiveSubset(
                            contactLineWallBoundaryMarker(
                                contact_line),
                            active_wall_marker));

        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: installed coupled dissipative Ren--E DynamicContactAngle") +
            " interface_marker=" + std::to_string(bc.interface_marker) +
            " contact_marker=" + std::to_string(contact_marker) +
            " wall_marker=" +
            std::to_string(contactLineWallBoundaryMarker(contact_line)) +
            " convention=theta_through_liquid_n_outward_liquid_nw_fluid_to_solid" +
            " law=xi_Vcl_equals_gamma_cos_thetae_minus_cos_thetad" +
            " surface_tension_form=" + surfaceTensionFormName(bc) +
            " contact_force_discretization=" +
            (usesSurfaceStress(bc)
                 ? "surface_conormal_plus_equilibrium_wall_energy"
                 : "explicit_dynamic_angle_gap") +
            " active_wall_marker=" +
            std::to_string(active_wall_marker) +
            " wetted_wall_domain=sharp_generated_active_boundary" +
            " diagnostic=navier_stokes_dynamic_contact_angle");
    }
}

[[nodiscard]] std::optional<ActiveVolumeDomain> activeVolumeDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system)
{
    std::optional<ActiveVolumeDomain> active_domain;
    for (const auto& bc : free_surfaces) {
        if (bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        if (active_domain.has_value()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: at most one active-domain free surface may restrict Navier-Stokes volume integration");
        }

        const FE::forms::CutVolumeSide side = activeDomainSide(bc.active_domain);
        FE::forms::FormExpr indicator{};
        if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
            indicator = activeDomainIndicatorFor(bc, system);
            FE_LOG_WARNING(
                "IncompressibleNavierStokesVMSModule: Active_domain_method=SmoothedIndicator is diagnostic and not a final benchmark acceptance path");
        }
        active_domain = ActiveVolumeDomain{
            bc.interface_marker,
            side,
            bc.active_domain_method,
            indicator,
            cutVolumeLevelSetShapeTangentFactor(
                bc, system, side, std::string_view("active"))};

        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: active-domain free surface "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << bc.level_set_field_name << "'"
            << " isovalue=" << bc.level_set_isovalue
            << " generated_interface_domain_id='"
            << bc.generated_interface_domain_id << "'"
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " side=" << cutVolumeSideName(side);
        if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
            oss << " smoothing_width="
                << (bc.active_domain_smoothing_width > FE::Real{0.0}
                        ? std::to_string(bc.active_domain_smoothing_width)
                        : std::string("cell_diameter"));
        }
        FE_LOG_INFO(oss.str());
    }
    return active_domain;
}

[[nodiscard]] FE::forms::FormExpr integrateOnActiveVolume(
    const FE::forms::FormExpr& integrand,
    const std::optional<ActiveVolumeDomain>& active_domain)
{
    if (!active_domain.has_value()) {
        return integrand.dx();
    }
    if (active_domain->method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
        return (active_domain->indicator * integrand).dx();
    }
    auto out = integrand.dCutVolume(active_domain->interface_marker,
                                    active_domain->side);
    return out;
}

void appendCutVolumeShapeTangentForm(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& volume_integrand,
    const std::optional<ActiveVolumeDomain>& domain)
{
    if (!domain.has_value() ||
        domain->method != FreeSurfaceActiveDomainMethod::CutVolume ||
        !domain->cut_volume_shape_tangent_factor.isValid()) {
        return;
    }

    auto term =
        (domain->cut_volume_shape_tangent_factor * volume_integrand)
            .dI(domain->interface_marker);
    shape_tangent_form =
        shape_tangent_form.isValid() ? shape_tangent_form + term : term;
}

void appendCutVolumeShapeTangentForm(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& volume_integrand,
    const ActiveVolumeDomain& domain)
{
    appendCutVolumeShapeTangentForm(
        shape_tangent_form,
        volume_integrand,
        std::optional<ActiveVolumeDomain>(domain));
}

[[nodiscard]] FE::forms::FormExpr freeSurfaceLevelSet(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (system.fieldParticipatesInUnknownVector(phi_id)) {
        return FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    }
    return FE::forms::FormExpr::discreteField(phi_id, *rec.space, bc.level_set_field_name);
}

[[nodiscard]] FE::forms::FormExpr freeSurfaceCurvatureField(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (bc.curvature_field_name.empty()) {
        return FE::forms::FormExpr{};
    }

    const auto kappa_id = system.findFieldByName(bc.curvature_field_name);
    if (kappa_id == FE::INVALID_FIELD_ID || !system.hasField(bc.curvature_field_name)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free surface references unknown curvature field '" +
            bc.curvature_field_name + "'");
    }

    const auto& rec = system.fieldRecord(kappa_id);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: curvature field '" +
            bc.curvature_field_name + "' must be scalar");
    }

    if (system.fieldParticipatesInUnknownVector(kappa_id)) {
        return FE::forms::StateField(kappa_id, *rec.space, bc.curvature_field_name);
    }
    return FE::forms::FormExpr::discreteField(
        kappa_id, *rec.space, bc.curvature_field_name);
}

[[nodiscard]] FE::forms::FormExpr unfittedLevelSetNormalSpeedFactor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (rec.source_kind == FE::systems::FieldSourceKind::PrescribedData ||
        !system.fieldParticipatesInUnknownVector(phi_id)) {
        return FE::forms::FormExpr{};
    }
    if (!rec.space || rec.components != 1 || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unfitted free-surface shape tangent requires a scalar level-set field space");
    }

    const auto phi =
        FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    const auto dphi =
        FE::forms::FormExpr::trialFunction(*rec.space, "d" + rec.name);
    const auto grad_phi = FE::forms::grad(phi);
    const auto grad_norm =
        FE::forms::sqrt(FE::forms::inner(grad_phi, grad_phi) +
                        FE::forms::FormExpr::constant(FE::Real{1.0e-30}));
    return FE::forms::FormExpr::constant(FE::Real{-1.0}) * dphi / grad_norm;
}

[[nodiscard]] FE::forms::FormExpr unfittedLevelSetInterfacePointMotion(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (rec.source_kind == FE::systems::FieldSourceKind::PrescribedData ||
        !system.fieldParticipatesInUnknownVector(phi_id)) {
        return FE::forms::FormExpr{};
    }
    if (!rec.space || rec.components != 1 || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unfitted free-surface point-location tangent requires a scalar level-set field space");
    }

    const auto phi =
        FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    const auto dphi =
        FE::forms::FormExpr::trialFunction(*rec.space, "d" + rec.name);
    const auto grad_phi = FE::forms::grad(phi);
    const auto grad_norm =
        FE::forms::sqrt(FE::forms::inner(grad_phi, grad_phi) +
                        FE::forms::FormExpr::constant(FE::Real{1.0e-30}));
    const auto n_level_set = grad_phi / grad_norm;
    const auto normal_speed =
        FE::forms::FormExpr::constant(FE::Real{-1.0}) * dphi / grad_norm;
    return normal_speed * n_level_set;
}

[[nodiscard]] FE::forms::FormExpr unfittedInterfaceMeasureCurvature(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi)
{
    if (!bc.curvature_field_name.empty()) {
        return freeSurfaceCurvatureField(bc, system);
    }
    return FE::forms::meanCurvatureFromLevelSet(phi);
}

[[nodiscard]] FE::forms::FormExpr unfittedTractionCurvature(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi)
{
    FE::forms::FormExpr curvature;
    if (!bc.curvature_field_name.empty()) {
        curvature = freeSurfaceCurvatureField(bc, system);
    } else if (bc.use_level_set_curvature) {
        curvature = FE::forms::meanCurvatureFromLevelSet(phi);
    } else {
        curvature = FE::forms::bc::toScalarExpr(
            bc.curvature,
            freeSurfaceValueName("ns_free_surface_curvature", bc));
    }

    if (bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        return FE::forms::FormExpr::constant(FE::Real{-1.0}) * curvature;
    }
    return curvature;
}

void appendUnfittedInterfaceMeasureShapeTangent(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& residual_integrand,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || !residual_integrand.isValid()) {
        return;
    }
    if (unfittedLevelSetShapeTangentsDisabled()) {
        return;
    }

    const auto normal_speed = unfittedLevelSetNormalSpeedFactor(bc, system);
    if (!normal_speed.isValid()) {
        return;
    }

    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto curvature = unfittedInterfaceMeasureCurvature(bc, system, phi);
    const auto term =
        (curvature * normal_speed * residual_integrand)
            .dI(bc.interface_marker);
    shape_tangent_form =
        shape_tangent_form.isValid() ? shape_tangent_form + term : term;

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: adding unfitted free-surface interface measure shape tangent marker=") +
        std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " curvature_source=" +
        (bc.curvature_field_name.empty() ? "level_set_geometry" : "curvature_field") +
        " experimental_path=unfitted_level_set_shape_tangent"
        " qualification=Experimental"
        " diagnostic=unfitted_free_surface_interface_measure_shape_tangent");
}

[[nodiscard]] FE::forms::FormExpr unfittedInterfaceNormalPointLocationTangent(
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& phi,
    const FE::forms::FormExpr& point_motion)
{
    if (!point_motion.isValid()) {
        return FE::forms::FormExpr{};
    }

    const auto grad_phi = FE::forms::grad(phi);
    const auto grad_norm =
        FE::forms::sqrt(FE::forms::inner(grad_phi, grad_phi) +
                        FE::forms::FormExpr::constant(FE::Real{1.0e-30}));
    const auto n0 = grad_phi / grad_norm;
    const auto* phi_node = phi.node();
    const auto* phi_space =
        phi_node != nullptr ? phi_node->spaceSignature() : nullptr;
    if (phi_space == nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unfitted free-surface normal tangent requires a level-set field space");
    }
    const auto dphi = FE::forms::FormExpr::trialFunction(
        *phi_space,
        "d" + bc.level_set_field_name);
    const auto moved_grad_dphi =
        FE::forms::grad(dphi) + FE::forms::hessian(phi) * point_motion;
    auto dn = (moved_grad_dphi -
               FE::forms::inner(moved_grad_dphi, n0) * n0) / grad_norm;
    if (bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        dn = FE::forms::FormExpr::constant(FE::Real{-1.0}) * dn;
    }
    return dn;
}

void appendUnfittedDynamicStressPointLocationShapeTangent(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& traction_scalar,
    const FE::forms::FormExpr& n,
    const FE::forms::FormExpr& v,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || !traction_scalar.isValid()) {
        return;
    }
    if (unfittedLevelSetShapeTangentsDisabled()) {
        return;
    }

    const auto point_motion = unfittedLevelSetInterfacePointMotion(bc, system);
    if (!point_motion.isValid()) {
        return;
    }
    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto dn = unfittedInterfaceNormalPointLocationTangent(
        bc, phi, point_motion);
    if (!dn.isValid()) {
        return;
    }

    const auto scalar_point =
        FE::forms::directionalDerivativeWrtSpatialCoordinate(
            traction_scalar,
            point_motion);
    if (!scalar_point.isValid()) {
        return;
    }

    const auto v_point = FE::forms::grad(v) * point_motion;
    const auto point_tangent =
        FE::forms::FormExpr::constant(FE::Real{-1.0}) *
        (scalar_point * FE::forms::inner(n, v) +
         traction_scalar *
             (FE::forms::inner(dn, v) + FE::forms::inner(n, v_point)));

    const auto term = point_tangent.dI(bc.interface_marker);
    shape_tangent_form =
        shape_tangent_form.isValid() ? shape_tangent_form + term : term;

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: adding unfitted free-surface interface point-location shape tangent marker=") +
        std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " experimental_path=unfitted_level_set_shape_tangent"
        " qualification=Experimental"
        " diagnostic=unfitted_free_surface_interface_point_location_shape_tangent");
}

void appendUnfittedKinematicPointLocationShapeTangent(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& penalty,
    const FE::forms::FormExpr& relative_velocity,
    const FE::forms::FormExpr& n,
    const FE::forms::FormExpr& v,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || !penalty.isValid()) {
        return;
    }
    if (unfittedLevelSetShapeTangentsDisabled()) {
        return;
    }

    const auto point_motion = unfittedLevelSetInterfacePointMotion(bc, system);
    if (!point_motion.isValid()) {
        return;
    }
    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto dn = unfittedInterfaceNormalPointLocationTangent(
        bc, phi, point_motion);
    if (!dn.isValid()) {
        return;
    }

    const auto u_point =
        FE::forms::directionalDerivativeWrtSpatialCoordinate(
            relative_velocity,
            point_motion);
    if (!u_point.isValid()) {
        return;
    }
    const auto v_point = FE::forms::grad(v) * point_motion;

    const auto u_normal = FE::forms::normalTrace(relative_velocity, n);
    const auto v_normal = FE::forms::normalTrace(v, n);
    const auto u_normal_point =
        FE::forms::inner(u_point, n) + FE::forms::inner(relative_velocity, dn);
    const auto v_normal_point =
        FE::forms::inner(v_point, n) + FE::forms::inner(v, dn);
    const auto point_tangent =
        penalty * (u_normal_point * v_normal + u_normal * v_normal_point);

    const auto term = point_tangent.dI(bc.interface_marker);
    shape_tangent_form =
        shape_tangent_form.isValid() ? shape_tangent_form + term : term;

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: adding unfitted free-surface interface point-location shape tangent marker=") +
        std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " experimental_path=unfitted_level_set_shape_tangent"
        " qualification=Experimental"
        " diagnostic=unfitted_free_surface_interface_point_location_shape_tangent");
}

[[nodiscard]] bool appendUniqueExtraTrialField(
    FE::systems::FormInstallOptions& install,
    FE::FieldId field)
{
    if (field == FE::INVALID_FIELD_ID) {
        return false;
    }
    if (std::find(install.extra_trial_fields.begin(),
                  install.extra_trial_fields.end(),
                  field) != install.extra_trial_fields.end()) {
        return false;
    }
    install.extra_trial_fields.push_back(field);
    return true;
}

[[nodiscard]] bool unfittedFreeSurfaceNeedsLevelSetTrialFieldForNavierStokes(
    const FreeSurfaceBoundary& bc) noexcept
{
    if (!isUnfittedLevelSet(bc)) {
        return false;
    }
    // The legacy curvature-traction contact law depends explicitly on
    // n(phi).  SurfaceStress instead consumes refreshed-frozen interface,
    // contact, and sharp wall geometry from the authoritative cut snapshot;
    // it must not advertise a nonexistent inner-Newton field tangent.
    const bool has_dynamic_contact = std::any_of(
            bc.contact_lines.begin(),
            bc.contact_lines.end(),
            [](const FreeSurfaceContactLine& contact_line) {
                return contactLineKind(contact_line) ==
                       ContactLineKind::DynamicRenE;
            });
    if (has_dynamic_contact && !usesSurfaceStress(bc)) {
        return true;
    }
    if (unfittedLevelSetShapeTangentsDisabled()) {
        return false;
    }
    if (bc.active_domain != FreeSurfaceActiveDomain::None &&
        bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
        return true;
    }
    const bool has_dynamic_interface_stress =
        !FE::forms::bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension);
    if (has_dynamic_interface_stress) {
        return true;
    }
    return bc.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Penalty;
}

void appendUnfittedFreeSurfaceLevelSetTrialFields(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system,
    FE::systems::FormInstallOptions& install)
{
    for (const auto& bc : free_surfaces) {
        const bool shape_tangents_enabled =
            !unfittedLevelSetShapeTangentsDisabled();
        if (!unfittedFreeSurfaceNeedsLevelSetTrialFieldForNavierStokes(bc)) {
            if (isUnfittedLevelSet(bc)) {
                const char* cut_volume_shape_tangent =
                    "not_applicable";
                bool cut_volume_shape_tangent_experimental = false;
                if (bc.active_domain != FreeSurfaceActiveDomain::None &&
                    bc.active_domain_method ==
                        FreeSurfaceActiveDomainMethod::CutVolume &&
                    !shape_tangents_enabled) {
                    cut_volume_shape_tangent = "disabled_by_default";
                } else if (bc.active_domain != FreeSurfaceActiveDomain::None &&
                           bc.active_domain_method ==
                               FreeSurfaceActiveDomainMethod::CutVolume) {
                    const auto phi_id = resolveLevelSetFieldId(bc, system);
                    const auto& rec = system.fieldRecord(phi_id);
                    const bool matrix_only_hadamard =
                        system.fieldParticipatesInUnknownVector(phi_id) &&
                        rec.source_kind !=
                            FE::systems::FieldSourceKind::PrescribedData;
                    cut_volume_shape_tangent =
                        matrix_only_hadamard
                            ? "matrix_only_hadamard"
                            : "not_installed_non_unknown_or_prescribed_level_set";
                    cut_volume_shape_tangent_experimental =
                        matrix_only_hadamard;
                }
                std::ostringstream oss;
                oss << "IncompressibleNavierStokesVMSModule: not adding "
                    << "unfitted free-surface level-set field as Navier-Stokes "
                    << "residual extra trial "
                    << "marker=" << bc.interface_marker
                    << " level_set_field='" << bc.level_set_field_name << "'"
                    << " Active_domain=" << activeDomainName(bc.active_domain)
                    << " Active_domain_method="
                    << activeDomainMethodName(bc.active_domain_method)
                    << " diagnostic=unfitted_free_surface_phi_extra_trial_omitted"
                    << " cut_volume_phi_shape_tangent="
                    << cut_volume_shape_tangent
                    << " reason=no_explicit_residual_level_set_dependence";
                if (cut_volume_shape_tangent_experimental) {
                    oss << " experimental_path=unfitted_level_set_shape_tangent"
                        << " qualification=Experimental";
                }
                FE_LOG_INFO(oss.str());
            }
            continue;
        }
        const auto phi_id = resolveLevelSetFieldId(bc, system);
        const auto& rec = system.fieldRecord(phi_id);
        if (!system.fieldParticipatesInUnknownVector(phi_id)) {
            continue;
        }
        if (!appendUniqueExtraTrialField(install, phi_id)) {
            continue;
        }

        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: adding unfitted free-surface "
            << "level-set field as Navier-Stokes extra trial "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << rec.name << "'"
            << " level_set_source_kind=" << fieldSourceKindName(rec.source_kind)
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " experimental_path=unfitted_level_set_shape_tangent"
            << " qualification=Experimental"
            << " diagnostic=unfitted_free_surface_phi_extra_trial";
        FE_LOG_INFO(oss.str());
    }
}

void appendUnfittedFreeSurfaceCurvatureTrialFields(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system,
    FE::systems::FormInstallOptions& install)
{
    for (const auto& bc : free_surfaces) {
        if (!isUnfittedLevelSet(bc) ||
            usesSurfaceStress(bc) ||
            bc.curvature_field_name.empty() ||
            FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
            continue;
        }

        const auto kappa_id = system.findFieldByName(bc.curvature_field_name);
        if (kappa_id == FE::INVALID_FIELD_ID ||
            !system.fieldParticipatesInUnknownVector(kappa_id)) {
            continue;
        }

        const auto& rec = system.fieldRecord(kappa_id);
        if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: curvature field '" +
                bc.curvature_field_name + "' must be scalar");
        }

        const bool appended = appendUniqueExtraTrialField(install, kappa_id);
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: "
            << (appended ? "added" : "retained")
            << " unfitted free-surface curvature field as Navier-Stokes trial field"
            << " marker=" << bc.interface_marker
            << " curvature_field='" << bc.curvature_field_name << "'"
            << " field_id=" << kappa_id
            << " source_kind=" << fieldSourceKindName(rec.source_kind)
            << " diagnostic=unfitted_free_surface_curvature_trial_field";
        FE_LOG_INFO(oss.str());
    }
}

void applyFreeSurfaceContactLineConstraints(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    int dim)
{
    for (const auto& contact_line : bc.contact_lines) {
        switch (contactLineKind(contact_line)) {
        case ContactLineKind::None:
        case ContactLineKind::PrescribedAngle:
        case ContactLineKind::DynamicRenE:
            continue;
        case ContactLineKind::Pinned:
            break;
        }

        if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require a coupled mesh displacement unknown");
        }

        const int marker = contactLineConstraintMarker(contact_line);
        std::vector<FE::forms::bc::StrongDirichlet> constraints;
        constraints.reserve(static_cast<std::size_t>(dim));
        for (int component = 0; component < dim; ++component) {
            constraints.push_back(FE::forms::bc::StrongDirichlet{
                .field = ale_binding.mesh_displacement_field,
                .boundary_marker = marker,
                .component = component,
                .value = FE::forms::FormExpr::constant(0.0),
                .symbol = "mesh_displacement",
            });
        }
        FE::systems::installStrongDirichlet(system, constraints);
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
                        [&system, bc, contact_line, contact_marker](
                            const FE::assembly::CutIntegrationContext*
                                cut_context) {
                            if (!validateContactWallNormalCallbackPreflight(
                                    system,
                                    cut_context,
                                    contact_line,
                                    contact_marker,
                                    cut_context != nullptr,
                                    "prescribed_contact_geometry")) {
                                return;
                            }
                            logUnfittedContactLineMeasure(
                                system,
                                cut_context,
                                bc,
                                contact_line,
                                contact_marker);
                        }});
            const auto* current_cut_context =
                system.cutIntegrationContext();
            if (validateContactWallNormalCallbackPreflight(
                    system,
                    current_cut_context,
                    contact_line,
                    contact_marker,
                    true,
                    "prescribed_contact_geometry_initial")) {
                logUnfittedContactLineMeasure(
                    system,
                    current_cut_context,
                    bc,
                    contact_line,
                    contact_marker);
            }
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: registered prescribed-angle contact geometry without a level-set residual") +
                " interface_marker=" +
                std::to_string(bc.interface_marker) +
                " contact_marker=" + std::to_string(contact_marker) +
                " level_set_geometry_owner=accepted_state_wall_aware_repair"
                " momentum_owner=young_wall_energy"
                " literal_codimension_two_level_set_residual=retired"
                " diagnostic=navier_stokes_prescribed_contact_geometry");
            continue;
        }

        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: fitted prescribed contact-angle validation did not fail closed");
    }
}

void applyFreeSurfaceBoundary(FE::forms::FormExpr& momentum_form,
                              FE::forms::FormExpr& continuity_form,
                              FE::forms::FormExpr& level_set_shape_tangent_form,
                              const FreeSurfaceBoundary& bc,
                              const FE::systems::FESystem& system,
                              const FE::forms::FormExpr& u,
                              const FE::forms::FormExpr& p,
                              const FE::forms::FormExpr& v,
                              const FE::forms::FormExpr& q,
                              const FE::forms::FormExpr& mesh_velocity,
                              const FE::forms::FormExpr& mu,
                              const IncompressibleNavierStokesVMSOptions& options,
                              bool ale_enabled,
                              int dim,
                              FE::forms::FormExpr* pressure_reference_probe_form = nullptr,
                              FE::forms::FormExpr* tangential_pressure_gradient_probe_form = nullptr,
                              FE::forms::FormExpr* conservative_pressure_form = nullptr,
                              FE::forms::FormExpr* conservative_surface_energy_form = nullptr)
{
    using namespace FE::forms;

    validateFreeSurfaceBoundary(bc, options, ale_enabled, system, dim);
    warnUnfittedRawCurvatureIfNeeded(bc);

    const auto pressure_reference_probe = freeSurfacePressureReferenceProbe();
    const auto tangential_pressure_gradient_probe =
        freeSurfaceTangentialPressureGradientProbe();
    const bool has_dynamic_stress =
        !bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !bc::isZeroConstantScalarValue(bc.surface_tension);
    const bool needs_surface_normal =
        has_dynamic_stress ||
        bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None ||
        tangential_pressure_gradient_probe.has_value();
    if (isUnfittedLevelSet(bc)) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: unfitted free-surface boundary mode marker=") +
            std::to_string(bc.interface_marker) +
            " level_set_field='" + bc.level_set_field_name + "'" +
            " generated_interface_domain_id='" + bc.generated_interface_domain_id + "'" +
            " generated_interface_geometry=" + bc.generated_interface_geometry +
            " geometry_tangent_policy=" + bc.geometry_tangent_policy +
            " level_set_shape_tangents=" + unfittedShapeTangentPolicyName() +
            " active_domain=" + activeDomainName(bc.active_domain) +
            " active_domain_method=" + activeDomainMethodName(bc.active_domain_method) +
            " dynamic_stress=" + (has_dynamic_stress ? "enabled" : "natural_zero") +
            " surface_tension_form=" + surfaceTensionFormName(bc) +
            " curvature_policy=" + freeSurfaceCurvaturePolicyName(bc) +
            " curvature_tangent_policy=" +
            freeSurfaceCurvatureTangentPolicyName(bc, system) +
            " kinematic_enforcement=" + kinematicEnforcementName(bc.kinematic_enforcement) +
            " cut_cell_stabilization=" + (bc.cut_cell_stabilization.enabled ? "enabled" : "disabled") +
            " pressure_stabilization_policy=" +
            pressureStabilizationPolicyName(
                bc.cut_cell_stabilization.pressure_policy) +
            " pressure_stabilization=" +
            (pressureStabilizationActive(bc) ? "active" : "inactive") +
            " pressure_stabilization_disabled_reason=" +
            pressureStabilizationDisabledReason(bc) +
            " velocity_extension=" + (bc.velocity_extension.enabled ? "enabled" : "disabled"));
    }
    const auto p_ext = bc::toScalarExpr(
        bc.external_pressure,
        freeSurfaceValueName("ns_free_surface_external_pressure", bc));
    if (pressure_reference_probe.has_value()) {
        const auto pressure_reference_form =
            integrateOnFreeSurface(
                FormExpr::constant(pressure_reference_probe->value) *
                    (p - p_ext) * q,
                bc,
                ale_enabled);
        continuity_form = continuity_form + pressure_reference_form;
        if (pressure_reference_probe_form != nullptr) {
            *pressure_reference_probe_form =
                pressure_reference_probe_form->isValid()
                    ? (*pressure_reference_probe_form +
                       pressure_reference_form)
                    : pressure_reference_form;
        }
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=free_surface_pressure_reference_probe") +
            " marker=" +
            std::to_string(isUnfittedLevelSet(bc) ? bc.interface_marker
                                                  : bc.boundary_marker) +
            " domain=" + (isUnfittedLevelSet(bc) ? "generated_interface"
                                                  : "boundary") +
            " env=" + pressure_reference_probe->name +
            " raw='" + pressure_reference_probe->raw + "'" +
            " penalty=" + std::to_string(pressure_reference_probe->value) +
            " form=continuity_pressure_trace_reference"
            " reference=free_surface_external_pressure");
    }
    if (!needs_surface_normal) {
        if (isUnfittedLevelSet(bc)) {
            FE_LOG_WARNING(
                std::string("IncompressibleNavierStokesVMSModule: unfitted free surface installs no explicit dI boundary residual marker=") +
                std::to_string(bc.interface_marker) +
                " level_set_field='" + bc.level_set_field_name + "'" +
                " dynamic_stress=natural_zero"
                " kinematic_enforcement=None"
                " diagnostic=unfitted_free_surface_natural_mode");
        }
        return;
    }

    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto n = usesSurfaceStress(bc)
        ? (useFittedCurrentGeometry(bc, ale_enabled)
               ? currentNormal()
               : generatedInterfaceOutwardNormal(bc))
        : (isUnfittedLevelSet(bc)
               ? unfittedInterfaceNormal(bc, phi)
               : (useFittedCurrentGeometry(bc, ale_enabled)
                      ? currentNormal()
                      : FormExpr::normal()));
    const auto gamma = bc::toScalarExpr(
        bc.surface_tension,
        freeSurfaceValueName("ns_free_surface_surface_tension", bc));
    const auto curvature = [&]() {
        if (usesSurfaceStress(bc)) {
            return FormExpr::constant(0.0);
        }
        if (bc::isZeroConstantScalarValue(bc.surface_tension)) {
            return FormExpr::constant(0.0);
        }
        if (isUnfittedLevelSet(bc)) {
            return unfittedTractionCurvature(bc, system, phi);
        }
        if (!isUnfittedLevelSet(bc) && bc.use_current_geometry_curvature) {
            return currentMeanCurvature();
        }
        return bc::toScalarExpr(
            bc.curvature,
            freeSurfaceValueName("ns_free_surface_curvature", bc));
    }();

    if (has_dynamic_stress) {
        if (usesSurfaceStress(bc)) {
            const auto pressure_integrand = p_ext * inner(n, v);
            const auto interface_pressure_form = integrateOnFreeSurface(
                pressure_integrand, bc, ale_enabled);
            if (conservative_pressure_form != nullptr) {
                *conservative_pressure_form =
                    conservative_pressure_form->isValid()
                        ? (*conservative_pressure_form +
                           interface_pressure_form)
                        : interface_pressure_form;
            }
            if (!surfaceStressActive(bc)) {
                // A literal gamma=0 leaves only the external-pressure work.
                // Keep the generated-interface normal required by the
                // SurfaceStress geometry contract, but do not construct the
                // surface projector.  Besides being the exact algebraic
                // specialization, this prevents the zero-coefficient
                // projector/grad(v) tree from reaching the JIT compiler.
                momentum_form = momentum_form + interface_pressure_form;
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: installed pressure-only free-surface stress") +
                    " marker=" + std::to_string(freeSurfaceMarker(bc)) +
                    " surface_tension_form=SurfaceStress" +
                    " normal_source=integration_rule_geometry" +
                    " form=p_external_n_dot_v" +
                    " surface_energy_form=omitted_literal_zero_gamma" +
                    " diagnostic=free_surface_pressure_only_surface_stress");
            } else {
                // Variation of the discrete surface energy gamma*|Gamma_h|.
                // Crucially, n is the geometric normal carried by the same
                // generated-interface quadrature rule as dI, not an
                // independently evaluated Q1-gradient normal.  The positive
                // sign follows div_Gamma(I-n*n)=-kappa*n for the
                // outward-liquid convention.
                const auto projector = FormExpr::identity(dim) - outer(n, n);
                const auto surface_energy_integrand =
                    gamma * inner(projector, grad(v));
                const auto interface_surface_energy_form =
                    integrateOnFreeSurface(
                        surface_energy_integrand, bc, ale_enabled);
                if (conservative_surface_energy_form != nullptr) {
                    *conservative_surface_energy_form =
                        conservative_surface_energy_form->isValid()
                            ? (*conservative_surface_energy_form +
                               interface_surface_energy_form)
                            : interface_surface_energy_form;
                }
                const auto residual_integrand =
                    pressure_integrand + surface_energy_integrand;
                momentum_form = momentum_form + integrateOnFreeSurface(
                    residual_integrand, bc, ale_enabled);
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: installed variational free-surface stress") +
                    " marker=" + std::to_string(freeSurfaceMarker(bc)) +
                    " surface_tension_form=SurfaceStress" +
                    " normal_source=integration_rule_geometry" +
                    " form=gamma_I_minus_n_outer_n_colon_grad_v" +
                    " capillary_balance_method=discrete_energy_volume_stationarity" +
                    " capillary_balance_qualification=prerequisite_only" +
                    " force_projection_applied=0" +
                    " static_geometry_stationarity_required=1" +
                    " diagnostic=free_surface_variational_surface_stress");
            }
        } else {
            const auto traction_scalar = -p_ext - gamma * curvature;
            const auto traction = traction_scalar * n;
            const auto residual_integrand =
                FE::forms::FormExpr::constant(FE::Real{-1.0}) *
                inner(traction, v);
            momentum_form = momentum_form + integrateOnFreeSurface(
                residual_integrand, bc, ale_enabled);
            appendUnfittedInterfaceMeasureShapeTangent(
                level_set_shape_tangent_form,
                residual_integrand,
                bc,
                system);
            appendUnfittedDynamicStressPointLocationShapeTangent(
                level_set_shape_tangent_form,
                traction_scalar,
                n,
                v,
                bc,
                system);
        }
    }

    if (tangential_pressure_gradient_probe.has_value()) {
        const auto grad_q_tangent = grad(q) - dot(grad(q), n) * n;
        const auto grad_p_tangent = grad(p) - dot(grad(p), n) * n;
        const auto pressure_gradient_integrand =
            FormExpr::constant(tangential_pressure_gradient_probe->value) *
            inner(grad_q_tangent, grad_p_tangent);
        const auto pressure_gradient_form =
            integrateOnFreeSurface(pressure_gradient_integrand, bc, ale_enabled);
        continuity_form = continuity_form + pressure_gradient_form;
        if (tangential_pressure_gradient_probe_form != nullptr) {
            *tangential_pressure_gradient_probe_form =
                tangential_pressure_gradient_probe_form->isValid()
                    ? (*tangential_pressure_gradient_probe_form +
                       pressure_gradient_form)
                    : pressure_gradient_form;
        }
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=free_surface_tangential_pressure_gradient_probe") +
            " marker=" +
            std::to_string(isUnfittedLevelSet(bc) ? bc.interface_marker
                                                  : bc.boundary_marker) +
            " domain=" + (isUnfittedLevelSet(bc) ? "generated_interface"
                                                  : "boundary") +
            " env=" + tangential_pressure_gradient_probe->name +
            " raw='" + tangential_pressure_gradient_probe->raw + "'" +
            " scale=" +
            std::to_string(tangential_pressure_gradient_probe->value) +
            " form=continuity_free_surface_tangential_pressure_gradient"
            " reference=free_surface_external_pressure");
    }

    switch (bc.kinematic_enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return;
    case FreeSurfaceKinematicEnforcement::Penalty: {
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        const auto penalty = bc::toScalarExpr(
            bc.kinematic_penalty,
            freeSurfaceValueName("ns_free_surface_kinematic_penalty", bc));
        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto residual_integrand =
            penalty * normal_mismatch * normalTrace(v, n);
        momentum_form = momentum_form + integrateOnFreeSurface(
            residual_integrand, bc, ale_enabled);
        appendUnfittedInterfaceMeasureShapeTangent(
            level_set_shape_tangent_form,
            residual_integrand,
            bc,
            system);
        appendUnfittedKinematicPointLocationShapeTangent(
            level_set_shape_tangent_form,
            penalty,
            u - mesh_velocity,
            n,
            v,
            bc,
            system);
        return;
    }
    case FreeSurfaceKinematicEnforcement::Nitsche: {
        if (isUnfittedLevelSet(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics are only supported on fitted ALE boundaries");
        }
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        if (!(bc.kinematic_nitsche_gamma > 0.0)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics require kinematic_nitsche_gamma > 0");
        }

        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto v_normal = normalTrace(v, n);
        const auto stress_u = FormExpr::constant(2.0) * mu * sym(grad(u));
        const auto stress_v = FormExpr::constant(2.0) * mu * sym(grad(v));
        const auto normal_stress_u = normalTrace(stress_u * n, n);
        const auto normal_stress_v = normalTrace(stress_v * n, n);
        const auto penalty = bc::buildTraceNitschePenalty(
            mu / hNormal(),
            u,
            bc::TraceNitscheOptions{
                .gamma = bc.kinematic_nitsche_gamma,
                .variant = bc.kinematic_nitsche_symmetric
                    ? bc::NitscheVariant::Symmetric
                    : bc::NitscheVariant::Unsymmetric,
                .scale_with_p = bc.kinematic_nitsche_scale_with_p});

        momentum_form = momentum_form + integrateOnFreeSurface(
            (p - normal_stress_u) * v_normal +
            penalty * normal_mismatch * v_normal,
            bc,
            ale_enabled);
        if (bc.kinematic_nitsche_symmetric) {
            momentum_form = momentum_form - integrateOnFreeSurface(
                normal_stress_v * normal_mismatch, bc, ale_enabled);
            continuity_form = continuity_form + integrateOnFreeSurface(
                q * normal_mismatch, bc, ale_enabled);
        } else {
            momentum_form = momentum_form + integrateOnFreeSurface(
                normal_stress_v * normal_mismatch, bc, ale_enabled);
            continuity_form = continuity_form - integrateOnFreeSurface(
                q * normal_mismatch, bc, ale_enabled);
        }
        return;
    }
    }

    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unsupported free-surface kinematic enforcement");
}

[[nodiscard]] const char* tangentialMeshPolicyName(
    FreeSurfaceTangentialMeshPolicy policy) noexcept;

[[nodiscard]] FE::systems::MeshTangentialBoundaryPolicy
systemTangentialMeshPolicy(FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return FE::systems::MeshTangentialBoundaryPolicy::Free;
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return FE::systems::MeshTangentialBoundaryPolicy::SmoothingOnly;
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return FE::systems::MeshTangentialBoundaryPolicy::Prescribed;
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted "
        "free-surface tangential mesh policy");
}

[[nodiscard]] std::string fittedTangentialOperatorSource(
    int boundary_marker,
    FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Fitted free-surface natural tangential state on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "Fitted free-surface tangential surface smoothing on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Fitted free-surface prescribed tangential mesh velocity on "
               "marker " +
               std::to_string(boundary_marker);
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted tangential "
        "mesh policy source");
}

void installFittedFreeSurfaceMeshKinematics(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    const FE::forms::FormExpr& u,
    const IncompressibleNavierStokesVMSOptions& options,
    const FE::systems::FormInstallOptions& base_install_options,
    FE::FieldId velocity_field)
{
    using namespace FE::forms;

    if (bc.implementation != FreeSurfaceImplementation::FittedALE) {
        return;
    }
    const bool install_normal_relation =
        bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
        bc.normal_kinematic_policy ==
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
    const bool install_tangential_smoothing =
        bc.tangential_mesh_policy ==
        FreeSurfaceTangentialMeshPolicy::SmoothingOnly;
    const bool install_tangential_prescription =
        bc.tangential_mesh_policy ==
        FreeSurfaceTangentialMeshPolicy::Prescribed;
    if (!ale_binding.coupled()) {
        return;
    }
    if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh kinematics require a coupled mesh displacement unknown");
    }

    const auto& rec = system.fieldRecord(ale_binding.mesh_displacement_field);
    if (!rec.space) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh displacement field has no function space");
    }

    system.declareMeshTangentialBoundaryPolicy(
        FE::systems::MeshTangentialBoundaryPolicyDeclaration{
            .mesh_displacement_field = ale_binding.mesh_displacement_field,
            .boundary_marker = bc.boundary_marker,
            .policy =
                systemTangentialMeshPolicy(bc.tangential_mesh_policy),
            .owner_component =
                "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary",
        });

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface mesh policy") +
        " marker=" + std::to_string(bc.boundary_marker) +
        " tangential_policy=" +
        tangentialMeshPolicyName(bc.tangential_mesh_policy) +
        " tangential_owner=free_surface_boundary" +
        " tangential_enforcement=" +
        (install_tangential_prescription
             ? "weak_velocity_penalty"
             : install_tangential_smoothing
                   ? "surface_smoothing_functional"
                   : "natural_no_tangential_constraint") +
        " diagnostic=fitted_free_surface_mesh_policy");

    if (!install_normal_relation && !install_tangential_smoothing &&
        !install_tangential_prescription) {
        FE::analysis::BoundaryConditionDescriptor descriptor;
        descriptor.primary_variable =
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field);
        descriptor.boundary_marker = bc.boundary_marker;
        descriptor.trace_kind =
            FE::analysis::TraceKind::TangentialComponent;
        descriptor.enforcement_kind =
            FE::analysis::EnforcementKind::WeakConsistent;
        const auto consumer_source = fittedTangentialOperatorSource(
            bc.boundary_marker, bc.tangential_mesh_policy);
        descriptor.source = consumer_source;
        system.addBoundaryConditionDescriptor(
            std::move(descriptor), options.operator_tag);
        system.bindMeshTangentialBoundaryPolicyConsumer(
            ale_binding.mesh_displacement_field,
            bc.boundary_marker,
            systemTangentialMeshPolicy(bc.tangential_mesh_policy),
            options.operator_tag,
            consumer_source);
        return;
    }

    const auto psi = TestField(
        ale_binding.mesh_displacement_field,
        *rec.space,
        "psi_free_surface_mesh");
    const auto d_mesh = StateField(
        ale_binding.mesh_displacement_field,
        *rec.space,
        "d_mesh_free_surface");
    const auto n = currentNormal();
    FormExpr residual;
    if (install_normal_relation) {
        const auto normal_mismatch = normalTrace(dt(d_mesh) - u, n);
        const auto penalty = [&]() {
            switch (bc.kinematic_enforcement) {
            case FreeSurfaceKinematicEnforcement::Penalty:
                return bc::toScalarExpr(
                    bc.kinematic_penalty,
                    freeSurfaceValueName(
                        "ns_free_surface_mesh_kinematic_penalty", bc));
            case FreeSurfaceKinematicEnforcement::Nitsche:
                return FormExpr::constant(bc.kinematic_nitsche_gamma) /
                       hNormal();
            case FreeSurfaceKinematicEnforcement::None:
                break;
            }
            return FormExpr::constant(0.0);
        }();
        residual = integrateOnFreeSurface(
            penalty * normal_mismatch * normalTrace(psi, n),
            bc,
            /*ale_enabled=*/true);
    }

    if (install_tangential_smoothing) {
        const auto projector =
            FormExpr::identity(rec.components) - outer(n, n);
        const auto tangential_gradient =
            projector * grad(d_mesh) * projector;
        const auto tangential_test_gradient =
            projector * grad(psi) * projector;
        const auto smoothing_weight = bc::toScalarExpr(
            bc.tangential_mesh_penalty,
            freeSurfaceValueName(
                "ns_free_surface_tangential_mesh_smoothing_weight", bc));
        const auto tangential_smoothing_residual =
            integrateOnFreeSurface(
                smoothing_weight *
                    inner(tangential_gradient,
                          tangential_test_gradient),
                bc,
                /*ale_enabled=*/true);
        residual = residual.isValid()
                       ? residual + tangential_smoothing_residual
                       : tangential_smoothing_residual;
    }

    if (install_tangential_prescription) {
        auto target_components = bc::toVectorExpr(
            bc.prescribed_tangential_mesh_velocity,
            rec.components,
            "ns_free_surface_tangential_mesh_velocity",
            bc.boundary_marker,
            bc::ComponentValueNameStyle::Component);
        const auto target =
            FormExpr::asVector(std::move(target_components));
        const auto rate_gap = dt(d_mesh) - target;
        const auto tangential_work =
            inner(rate_gap, psi) -
            normalTrace(rate_gap, n) * normalTrace(psi, n);
        const auto tangential_penalty = bc::toScalarExpr(
            bc.tangential_mesh_penalty,
            freeSurfaceValueName(
                "ns_free_surface_tangential_mesh_penalty", bc));
        const auto tangential_residual = integrateOnFreeSurface(
            tangential_penalty * tangential_work,
            bc,
            /*ale_enabled=*/true);
        residual = residual.isValid()
                       ? residual + tangential_residual
                       : tangential_residual;
    }

    auto install = base_install_options;
    install.compiler_options.use_symbolic_tangent = true;
    ale_binding.configureInstallOptions(install);
    if (install_normal_relation &&
        velocity_field != FE::INVALID_FIELD_ID) {
        install.extra_trial_fields.push_back(velocity_field);
    }

    (void)FE::systems::installFormulation(
        system,
        options.operator_tag,
        {ale_binding.mesh_displacement_field},
        residual,
        install);

    if (install_normal_relation) {
        if (velocity_field == FE::INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted normal "
                "kinematics require a related fluid velocity field");
        }
        const auto mesh_source =
            "Fitted free-surface mesh normal kinematic row on marker " +
            std::to_string(bc.boundary_marker);
        FE::analysis::BoundaryConditionDescriptor mesh_descriptor;
        mesh_descriptor.primary_variable =
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field);
        mesh_descriptor.related_variables.push_back(
            FE::analysis::VariableKey::field(velocity_field));
        mesh_descriptor.boundary_marker = bc.boundary_marker;
        mesh_descriptor.trace_kind =
            FE::analysis::TraceKind::NormalComponent;
        mesh_descriptor.enforcement_kind =
            FE::analysis::EnforcementKind::WeakPenalty;
        mesh_descriptor.source = mesh_source;
        system.addBoundaryConditionDescriptor(
            std::move(mesh_descriptor), options.operator_tag);

        const auto fluid_source =
            "Fitted free-surface fluid normal kinematic row on marker " +
            std::to_string(bc.boundary_marker);
        FE::analysis::BoundaryConditionDescriptor fluid_descriptor;
        fluid_descriptor.primary_variable =
            FE::analysis::VariableKey::field(velocity_field);
        fluid_descriptor.related_variables.push_back(
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field));
        fluid_descriptor.boundary_marker = bc.boundary_marker;
        fluid_descriptor.trace_kind =
            FE::analysis::TraceKind::NormalComponent;
        fluid_descriptor.enforcement_kind =
            bc.kinematic_enforcement ==
                    FreeSurfaceKinematicEnforcement::Nitsche
                ? FE::analysis::EnforcementKind::WeakNitsche
                : FE::analysis::EnforcementKind::WeakPenalty;
        fluid_descriptor.source = fluid_source;
        system.addBoundaryConditionDescriptor(
            std::move(fluid_descriptor), options.operator_tag);
    }

    FE::analysis::BoundaryConditionDescriptor descriptor;
    descriptor.primary_variable =
        FE::analysis::VariableKey::field(
            ale_binding.mesh_displacement_field);
    descriptor.boundary_marker = bc.boundary_marker;
    descriptor.trace_kind =
        FE::analysis::TraceKind::TangentialComponent;
    descriptor.enforcement_kind =
        install_tangential_prescription
            ? FE::analysis::EnforcementKind::WeakPenalty
            : FE::analysis::EnforcementKind::WeakConsistent;
    const auto consumer_source = fittedTangentialOperatorSource(
        bc.boundary_marker, bc.tangential_mesh_policy);
    descriptor.source = consumer_source;
    system.addBoundaryConditionDescriptor(
        std::move(descriptor), options.operator_tag);
    system.bindMeshTangentialBoundaryPolicyConsumer(
        ale_binding.mesh_displacement_field,
        bc.boundary_marker,
        systemTangentialMeshPolicy(bc.tangential_mesh_policy),
        options.operator_tag,
        consumer_source);
}

[[nodiscard]] std::string jsonString(std::string_view value)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() + 2u);
    out.push_back('"');
    for (const unsigned char c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20u) {
                out += "\\u00";
                out.push_back(hex[(c >> 4u) & 0x0fu]);
                out.push_back(hex[c & 0x0fu]);
            } else {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

[[nodiscard]] std::string jsonReal(FE::Real value)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: effective configuration contains a non-finite scalar");
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<FE::Real>::max_digits10)
           << value;
    return stream.str();
}

[[nodiscard]] constexpr const char* jsonBool(bool value) noexcept
{
    return value ? "true" : "false";
}

[[nodiscard]] std::string jsonScalarValue(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value)
{
    if (const auto* literal = std::get_if<FE::Real>(&value)) {
        return jsonReal(*literal);
    }
    if (std::holds_alternative<FE::forms::ScalarCoefficient>(value)) {
        return R"({"kind":"spatial_coefficient"})";
    }
    if (std::holds_alternative<FE::forms::TimeScalarCoefficient>(value)) {
        return R"({"kind":"time_coefficient"})";
    }
    return R"({"kind":"form_expression"})";
}

[[nodiscard]] const char* freeSurfaceImplementationName(
    FreeSurfaceImplementation implementation) noexcept
{
    return implementation == FreeSurfaceImplementation::FittedALE
               ? "FittedALE"
               : "UnfittedLevelSet";
}

[[nodiscard]] const char* freeSurfacePhysicalModelName(
    FreeSurfacePhysicalModel model) noexcept
{
    switch (model) {
    case FreeSurfacePhysicalModel::
        OnePhaseLiquidPrescribedExteriorPressure:
        return "one_phase_liquid_prescribed_exterior_pressure";
    }
    return nullptr;
}

[[nodiscard]] const char* normalKinematicPolicyName(
    FreeSurfaceNormalKinematicPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceNormalKinematicPolicy::None:
        return "None";
    case FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity:
        return "MatchFluidNormalVelocity";
    }
    return "Unknown";
}

[[nodiscard]] const char* tangentialMeshPolicyName(
    FreeSurfaceTangentialMeshPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Free";
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "SmoothingOnly";
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Prescribed";
    }
    return "Unknown";
}

[[nodiscard]] const char* contactLineKindName(ContactLineKind kind) noexcept
{
    switch (kind) {
    case ContactLineKind::None:
        return "None";
    case ContactLineKind::Pinned:
        return "Pinned";
    case ContactLineKind::PrescribedAngle:
        return "PrescribedAngle";
    case ContactLineKind::DynamicRenE:
        return "DynamicRenE";
    }
    return "Unknown";
}

[[nodiscard]] const char* geometryTangentPathName(
    FE::forms::GeometryTangentPath path) noexcept
{
    switch (path) {
    case FE::forms::GeometryTangentPath::Auto:
        return "Auto";
    case FE::forms::GeometryTangentPath::ADReference:
        return "ADReference";
    case FE::forms::GeometryTangentPath::SymbolicRequired:
        return "SymbolicRequired";
    case FE::forms::GeometryTangentPath::SymbolicWithADCheck:
        return "SymbolicWithADCheck";
    }
    return "Unknown";
}

[[nodiscard]] std::string activePhaseSignName(
    FreeSurfaceActiveDomain domain)
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "full_domain";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "negative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "positive";
    }
    return "unknown";
}

[[nodiscard]] std::string contactConfigurationJson(
    const FreeSurfaceContactLine& contact_line)
{
    const auto kind = contactLineKind(contact_line);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"model\":" << jsonString(contactLineKindName(kind));
    if (kind != ContactLineKind::None) {
        out << ",\"wall_boundary_marker\":"
            << contactLineWallBoundaryMarker(contact_line)
            << ",\"contact_line_marker\":"
            << contactLineMarker(contact_line);
    }
    if (kind == ContactLineKind::PrescribedAngle ||
        kind == ContactLineKind::DynamicRenE) {
        const auto wall_normal = normalizedWallNormal(contact_line);
        out << ",\"angle_convention\":\"radians_measured_through_liquid\""
            << ",\"angle_unit\":\"radian\""
            << ",\"equilibrium_angle_radians\":"
            << jsonScalarValue(contactLineAngleRadians(contact_line))
            << ",\"wall_normal_convention\":\"outward_from_liquid_into_solid\""
            << ",\"wall_normal\":[" << jsonReal(wall_normal[0]) << ','
            << jsonReal(wall_normal[1]) << ',' << jsonReal(wall_normal[2])
            << ']';
    }
    if (kind == ContactLineKind::PrescribedAngle) {
        out << ",\"level_set_geometry_owner\":"
            << "\"accepted_state_wall_aware_repair\""
            << ",\"prescribed_angle_operator\":\"wall_aware_geometry_only\"";
    }
    if (kind == ContactLineKind::DynamicRenE) {
        const auto mobility = constantScalarValueOrThrow(
            contactLineMobility(contact_line),
            "effective contact-line mobility");
        out << ",\"law\":\"xi_V_equals_gamma_cos_equilibrium_minus_cos_dynamic\""
            << ",\"mobility\":" << jsonReal(mobility)
            << ",\"mobility_unit\":\"solver_consistent\""
            << ",\"line_friction\":" << jsonReal(FE::Real{1.0} / mobility)
            << ",\"line_friction_unit\":\"reciprocal_mobility\""
            << ",\"slip_length\":"
            << jsonScalarValue(contactLineSlipLength(contact_line))
            << ",\"slip_length_unit\":\"length\"";
    }
    out << '}';
    return out.str();
}

[[nodiscard]] bool freeSurfaceConfigurationLess(
    const FreeSurfaceBoundary& lhs,
    const FreeSurfaceBoundary& rhs)
{
    return std::tie(lhs.implementation,
                    lhs.boundary_marker,
                    lhs.interface_marker,
                    lhs.level_set_field_name,
                    lhs.generated_interface_domain_id) <
           std::tie(rhs.implementation,
                    rhs.boundary_marker,
                    rhs.interface_marker,
                    rhs.level_set_field_name,
                    rhs.generated_interface_domain_id);
}

[[nodiscard]] bool contactConfigurationLess(
    const FreeSurfaceContactLine& lhs,
    const FreeSurfaceContactLine& rhs)
{
    return std::tuple{contactLineKind(lhs),
                      contactLineWallBoundaryMarker(lhs),
                      contactLineMarker(lhs)} <
           std::tuple{contactLineKind(rhs),
                      contactLineWallBoundaryMarker(rhs),
                      contactLineMarker(rhs)};
}

struct TangentialPolicyProvenance {
    const FE::systems::MeshTangentialBoundaryPolicyDeclaration*
        declaration{nullptr};
    const FE::analysis::BoundaryConditionDescriptor* consumed_operator{
        nullptr};
};

[[nodiscard]] TangentialPolicyProvenance
tangentialPolicyProvenance(
    const FreeSurfaceBoundary& boundary,
    const FE::systems::FESystem& system)
{
    if (boundary.implementation !=
        FreeSurfaceImplementation::FittedALE) {
        return {};
    }
    const auto displacement = system.meshMotionField(
        FE::systems::MeshMotionFieldRole::Displacement);
    if (!displacement.has_value()) {
        return {};
    }
    const auto declarations =
        system.meshTangentialBoundaryPolicies();
    const auto declaration = std::find_if(
        declarations.begin(),
        declarations.end(),
        [&](const auto& candidate) {
            return candidate.mesh_displacement_field == *displacement &&
                   candidate.boundary_marker ==
                       boundary.boundary_marker &&
                   candidate.policy == systemTangentialMeshPolicy(
                                           boundary.tangential_mesh_policy);
        });
    if (declaration == declarations.end()) {
        return {};
    }
    if (!declaration->consumer_bound ||
        !system.hasOperator(declaration->consumer_operator_tag) ||
        declaration->consumer_source != fittedTangentialOperatorSource(
            boundary.boundary_marker,
            boundary.tangential_mesh_policy)) {
        return TangentialPolicyProvenance{
            .declaration = &*declaration,
        };
    }

    const auto& descriptors =
        system.boundaryConditionDescriptors();
    const auto matches_consumer = [&](const auto& candidate) {
        return candidate.primary_variable ==
                   FE::analysis::VariableKey::field(
                       declaration->mesh_displacement_field) &&
               candidate.boundary_marker ==
                   declaration->boundary_marker &&
               candidate.trace_kind ==
                   FE::analysis::TraceKind::TangentialComponent &&
               candidate.enforcement_kind ==
                   (boundary.tangential_mesh_policy ==
                            FreeSurfaceTangentialMeshPolicy::Prescribed
                        ? FE::analysis::EnforcementKind::WeakPenalty
                        : FE::analysis::EnforcementKind::
                              WeakConsistent) &&
               candidate.source == declaration->consumer_source;
    };
    const auto consumed = std::find_if(
        descriptors.begin(),
        descriptors.end(),
        matches_consumer);
    const auto consumed_count = std::count_if(
        descriptors.begin(),
        descriptors.end(),
        matches_consumer);
    return TangentialPolicyProvenance{
        .declaration = &*declaration,
        .consumed_operator =
            consumed_count == 1u ? &*consumed : nullptr,
    };
}

[[nodiscard]] EffectiveConfigurationArtifact makeEffectiveConfigurationArtifact(
    const IncompressibleNavierStokesVMSOptions& options,
    std::vector<FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    int dimension,
    bool effective_enable_vms)
{
    std::sort(free_surfaces.begin(), free_surfaces.end(),
              freeSurfaceConfigurationLess);

    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"artifact_schema_version\":3"
        << ",\"component\":\"incompressible_navier_stokes_free_surface\""
        << ",\"configuration_schema\":{\"input_version\":"
        << options.input_configuration_schema_version
        << ",\"effective_version\":"
        << IncompressibleNavierStokesVMSOptions::
               current_configuration_schema_version
        << ",\"migration_mode\":"
        << jsonString(options.explicit_legacy_configuration
                          ? "explicit_legacy"
                          : "current")
        << '}'
        << ",\"capability_label\":"
        << jsonString(options.explicit_legacy_configuration
                          ? "legacy_diagnostic"
                          : "one_phase_liquid_sharp_interface")
        << ",\"physical_model\":";
    if (options.explicit_legacy_configuration) {
        out << "null";
    } else {
        out << "{\"name\":"
            << jsonString(freeSurfacePhysicalModelName(
                   options.free_surface_physical_model))
            << ",\"liquid_phase_count\":1"
            << ",\"liquid_velocity_field_count\":1"
            << ",\"liquid_pressure_field_count\":1"
            << ",\"material_density_state_count\":1"
            << ",\"material_viscosity_state_count\":1"
            << ",\"exterior_pressure_mode\":"
               "\"prescribed_scalar_traction_reference\""
            << ",\"exterior_momentum_solved\":false"
            << ",\"exterior_pressure_field_solved\":false"
            << ",\"incompressible_two_fluid_implemented\":false"
            << ",\"gas_dynamics_implemented\":false}";
    }
    out
        << ",\"units\":{\"system\":\"consistent_solver_units\",\"angle\":\"radian\",\"length\":\"solver_length\",\"pressure\":\"solver_pressure\",\"surface_tension\":\"force_per_length\"}"
        << ",\"fields\":{\"velocity\":"
        << jsonString(options.velocity_field_name)
        << ",\"pressure\":" << jsonString(options.pressure_field_name)
        << ",\"operator\":" << jsonString(options.operator_tag)
        << ",\"dimension\":" << dimension << '}'
        << ",\"ale\":{\"enabled\":" << jsonBool(options.enable_ale)
        << ",\"mesh_velocity_source\":"
        << jsonString(options.mesh_velocity_source ==
                              ALEMeshVelocitySource::CoupledDisplacement
                          ? "CoupledDisplacement"
                          : "PrescribedData")
        << ",\"mesh_velocity_field\":"
        << jsonString(options.mesh_velocity_field_name)
        << ",\"mesh_displacement_field\":"
        << jsonString(options.mesh_displacement_field_name)
        << ",\"geometry_tangent_path\":"
        << jsonString(geometryTangentPathName(options.moving_mesh_tangent_path))
        << '}'
        << ",\"generic_velocity_nitsche\":{\"gamma\":"
        << jsonReal(options.nitsche_gamma)
        << ",\"symmetric\":" << jsonBool(options.nitsche_symmetric)
        << ",\"scale_with_polynomial_order\":"
        << jsonBool(options.nitsche_scale_with_p)
        << ",\"generated_active_boundary_minimum_energy_ratio\":"
        << jsonReal(
               options
                   .generated_boundary_nitsche_minimum_energy_ratio)
        << '}'
        << ",\"stabilization\":{\"vms_enabled\":"
        << jsonBool(effective_enable_vms)
        << ",\"ct_m\":" << jsonReal(options.ct_m)
        << ",\"ct_c\":" << jsonReal(options.ct_c)
        << ",\"epsilon\":" << jsonReal(options.stabilization_epsilon)
        << '}'
        << ",\"maintenance_policy\":{\"owner_component\":\"level_set_transport\",\"coupling\":\"one_way_velocity_to_extension_to_level_set\"}"
        << ",\"extension_guards\":{\"physical_momentum_dry_extension_allowed\":false,\"auxiliary_extension_owner\":\"level_set_transport\",\"external_owner_required\":true}"
        << ",\"free_surfaces\":[";

    for (std::size_t surface_index = 0;
         surface_index < free_surfaces.size(); ++surface_index) {
        if (surface_index != 0u) {
            out << ',';
        }
        const auto& boundary = free_surfaces[surface_index];
        const auto tangential_provenance =
            tangentialPolicyProvenance(boundary, system);
        auto contacts = boundary.contact_lines;
        std::sort(contacts.begin(), contacts.end(), contactConfigurationLess);

        out << "{\"implementation\":"
            << jsonString(freeSurfaceImplementationName(boundary.implementation))
            << ",\"boundary_marker\":" << boundary.boundary_marker
            << ",\"interface_marker\":" << boundary.interface_marker
            << ",\"level_set_field\":"
            << jsonString(boundary.level_set_field_name)
            << ",\"generated_interface_domain\":"
            << jsonString(boundary.generated_interface_domain_id)
            << ",\"generated_interface_geometry\":"
            << jsonString(boundary.generated_interface_geometry)
            << ",\"geometry_tangent_policy\":"
            << jsonString(boundary.geometry_tangent_policy)
            << ",\"level_set_isovalue\":"
            << jsonReal(boundary.level_set_isovalue)
            << ",\"active_domain\":"
            << jsonString(activeDomainName(boundary.active_domain))
            << ",\"active_phase_sign\":"
            << jsonString(activePhaseSignName(boundary.active_domain))
            << ",\"active_domain_method\":"
            << jsonString(activeDomainMethodName(boundary.active_domain_method))
            << ",\"active_domain_smoothing_width\":"
            << jsonReal(boundary.active_domain_smoothing_width)
            << ",\"smoothing_width_unit\":\"length\""
            << ",\"allow_full_domain_unfitted_free_surface\":"
            << jsonBool(boundary.allow_full_domain_unfitted_free_surface)
            << ",\"external_pressure\":"
            << jsonScalarValue(boundary.external_pressure)
            << ",\"surface_tension\":"
            << jsonScalarValue(boundary.surface_tension)
            << ",\"surface_tension_form_requested\":";
        switch (boundary.surface_tension_form) {
        case FreeSurfaceSurfaceTensionForm::Automatic:
            out << "\"Automatic\"";
            break;
        case FreeSurfaceSurfaceTensionForm::CurvatureTraction:
            out << "\"CurvatureTraction\"";
            break;
        case FreeSurfaceSurfaceTensionForm::SurfaceStress:
            out << "\"SurfaceStress\"";
            break;
        }
        out << ",\"surface_tension_form_effective\":"
            << jsonString(surfaceTensionFormName(boundary));
        if (boundary.implementation ==
            FreeSurfaceImplementation::FittedALE) {
            out << ",\"fitted_surface_contact_capability\":{"
                << "\"qualification\":"
                << jsonString(
                       options.explicit_legacy_configuration
                           ? "unqualified_explicit_legacy"
                           : "supported_configuration_envelope")
                << ",\"supported_requests\":{"
                << "\"surface_tension_form\":["
                << "\"Automatic\",\"CurvatureTraction\"],"
                << "\"contact_line_model\":[\"None\",\"Pinned\"]}"
                << ",\"exclusion_disposition\":"
                << "\"fail_closed_before_system_mutation\""
                << ",\"exclusions\":[{"
                << "\"feature\":\"surface_tension_form\","
                << "\"value\":\"SurfaceStress\","
                << "\"reason_code\":"
                << "\"fitted_surface_stress_current_frame_gradient_unqualified\""
                << "},{\"feature\":\"contact_line_model\","
                << "\"value\":\"PrescribedAngle\","
                << "\"reason_code\":"
                << "\"fitted_contact_line_codimension_two_unavailable\""
                << "},{\"feature\":\"contact_line_model\","
                << "\"value\":\"DynamicRenE\","
                << "\"reason_code\":"
                << "\"dynamic_contact_requires_sharp_unfitted_level_set\""
                << "}]}";
        }
        out << ",\"curvature_policy\":"
            << jsonString(freeSurfaceCurvaturePolicyName(boundary))
            << ",\"curvature_tangent_policy\":"
            << jsonString(
                   freeSurfaceCurvatureTangentPolicyName(boundary, system))
            << ",\"kinematic\":{\"normal_policy\":"
            << jsonString(normalKinematicPolicyName(
                   boundary.normal_kinematic_policy))
            << ",\"tangential_mesh_policy\":"
            << jsonString(tangentialMeshPolicyName(
                   boundary.tangential_mesh_policy))
            << ",\"prescribed_tangential_mesh_velocity\":["
            << jsonScalarValue(
                   boundary.prescribed_tangential_mesh_velocity[0])
            << ','
            << jsonScalarValue(
                   boundary.prescribed_tangential_mesh_velocity[1])
            << ','
            << jsonScalarValue(
                   boundary.prescribed_tangential_mesh_velocity[2])
            << ']'
            << ",\"tangential_mesh_penalty\":"
            << jsonScalarValue(boundary.tangential_mesh_penalty)
            << ",\"tangential_mesh_owner\":";
        if (tangential_provenance.declaration != nullptr) {
            out << jsonString(
                tangential_provenance.declaration->owner_component);
        } else {
            out << "null";
        }
        out << ",\"policy_consumed\":"
            << jsonBool(
                   tangential_provenance.consumed_operator != nullptr)
            << ",\"operator_tag\":";
        if (tangential_provenance.consumed_operator != nullptr) {
            out << jsonString(
                tangential_provenance.declaration
                    ->consumer_operator_tag);
        } else {
            out << "null";
        }
        out << ",\"operator_source\":";
        if (tangential_provenance.consumed_operator != nullptr) {
            out << jsonString(
                tangential_provenance.consumed_operator->source);
        } else {
            out << "null";
        }
        out << ",\"policy_qualification\":"
            << jsonString(
                   boundary.implementation !=
                           FreeSurfaceImplementation::FittedALE
                       ? "not_applicable"
                       : options.explicit_legacy_configuration
                             ? "unqualified_explicit_legacy"
                             : "supported_configuration_envelope")
            << ",\"enforcement\":"
            << jsonString(kinematicEnforcementName(
                   boundary.kinematic_enforcement))
            << ",\"penalty\":"
            << jsonScalarValue(boundary.kinematic_penalty)
            << ",\"nitsche\":{\"gamma\":"
            << jsonReal(boundary.kinematic_nitsche_gamma)
            << ",\"symmetric\":"
            << jsonBool(boundary.kinematic_nitsche_symmetric)
            << ",\"scale_with_polynomial_order\":"
            << jsonBool(boundary.kinematic_nitsche_scale_with_p) << "}}"
            << ",\"stabilization\":{\"enabled\":"
            << jsonBool(boundary.cut_cell_stabilization.enabled)
            << ",\"small_cut_aggregation\":"
            << jsonBool(boundary.small_cut_aggregation)
            << ",\"pressure_policy\":"
            << jsonString(pressureStabilizationPolicyName(
                   boundary.cut_cell_stabilization.pressure_policy))
            << ",\"pressure_gradient_penalty\":"
            << jsonScalarValue(
                   boundary.cut_cell_stabilization.pressure_gradient_penalty)
            << ",\"use_cut_metadata_scale\":"
            << jsonBool(
                   boundary.cut_cell_stabilization.use_cut_metadata_scale)
            << ",\"cut_metadata_scale_cap\":";
        if (boundary.cut_cell_stabilization.cut_metadata_scale_cap.has_value()) {
            out << jsonReal(
                *boundary.cut_cell_stabilization.cut_metadata_scale_cap);
        } else {
            out << "null";
        }
        out << ",\"aggregation_guards\":{\"maximum_root_path_length\":"
            << boundary.small_cut_aggregation_guards
                   .maximum_root_path_length
            << ",\"maximum_reference_extrapolation_distance\":"
            << jsonReal(
                   boundary.small_cut_aggregation_guards
                       .maximum_reference_extrapolation_distance)
            << ",\"maximum_absolute_coefficient\":"
            << jsonReal(
                   boundary.small_cut_aggregation_guards
                       .maximum_absolute_coefficient)
            << ",\"maximum_row_l1_norm\":"
            << jsonReal(
                   boundary.small_cut_aggregation_guards
                       .maximum_row_l1_norm)
            << '}';
        out << '}'
            << ",\"pruning\":{\"decision_owner\":\"authoritative_geometry_snapshot\",\"fallback_to_whole_face\":false}"
            << ",\"legacy_dry_velocity_diffusion\":{\"enabled\":"
            << jsonBool(boundary.velocity_extension.enabled)
            << ",\"diffusivity\":"
            << jsonScalarValue(boundary.velocity_extension.diffusivity)
            << ",\"production_allowed\":false}"
            << ",\"contact_lines\":[";
        for (std::size_t contact_index = 0;
             contact_index < contacts.size(); ++contact_index) {
            if (contact_index != 0u) {
                out << ',';
            }
            out << contactConfigurationJson(contacts[contact_index]);
        }
        out << "]}";
    }
    out << "]}";

    return EffectiveConfigurationArtifact{
        .component = "incompressible_navier_stokes_free_surface",
        .json = out.str(),
    };
}

} // namespace

void IncompressibleNavierStokesVMSModule::registerOn(FE::systems::FESystem& system) const
{
    if (options_.mesh_velocity_source !=
            ALEMeshVelocitySource::PrescribedData &&
        options_.mesh_velocity_source !=
            ALEMeshVelocitySource::CoupledDisplacement) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "unsupported ALE mesh-velocity source");
    }
    if (freeSurfacePhysicalModelName(options_.free_surface_physical_model) ==
        nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "unsupported_free_surface_physical_model");
    }
    effective_configuration_artifact_.reset();
    const auto current_schema =
        IncompressibleNavierStokesVMSOptions::
            current_configuration_schema_version;
    if (options_.input_configuration_schema_version != current_schema &&
        options_.input_configuration_schema_version != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: unsupported free-surface configuration schema version " +
            std::to_string(options_.input_configuration_schema_version));
    }
    if (options_.input_configuration_schema_version == 1 &&
        !options_.explicit_legacy_configuration) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: schema version 1 requires explicit_legacy_configuration=true and cannot inherit the current capability label");
    }
    if (options_.input_configuration_schema_version == current_schema &&
        options_.explicit_legacy_configuration) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: explicit legacy behavior requires input configuration schema version 1");
    }
    if (!velocity_space_) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null velocity_space");
    }
    if (!pressure_space_) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null pressure_space");
    }
    if (options_.operator_tag.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: operator_tag must be non-empty");
    }

    const int dim = velocity_space_->value_dimension();
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: velocity space must have 1..3 components");
    }
    if (options_.rotating_frame_coriolis_enabled && dim != 3) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: rotating-frame Coriolis forcing requires a 3D velocity space");
    }
    if (pressure_space_->value_dimension() != 1) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: pressure space must be scalar");
    }
    if (!(options_.density > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: density must be > 0");
    }
    if (options_.viscosity_model == nullptr && !(options_.viscosity > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: viscosity must be > 0 when viscosity_model is not provided");
    }
    const bool nitsche_energy_diagnostic_requested =
        symmetricNitscheEnergyDiagnosticEnabled();
    if (nitsche_energy_diagnostic_requested &&
        options_
                .symmetric_nitsche_energy_qualification_scope !=
            SymmetricNitscheEnergyQualificationScope::
                JointLowLevelPrerequisite) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: symmetric Nitsche energy diagnostic requires explicit joint_low_level_prerequisite scope");
    }
    if (nitsche_energy_diagnostic_requested &&
        options_.viscosity_model != nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: symmetric Nitsche energy diagnostic requires constant viscosity; viscosity_model is unsupported");
    }
    const auto vms_override = navierStokesVmsDiagnosticOverride();
    const bool enable_vms =
        vms_override.has_value() ? vms_override->value : options_.enable_vms;
    const auto pspg_pressure_gradient_scale_override =
        navierStokesPspgPressureGradientScale();
    const FE::Real pspg_pressure_gradient_scale =
        pspg_pressure_gradient_scale_override.has_value()
            ? pspg_pressure_gradient_scale_override->value
            : FE::Real{1.0};
    const auto pspg_nonpressure_residual_scale_override =
        navierStokesPspgNonpressureResidualScale();
    const FE::Real pspg_nonpressure_residual_scale =
        pspg_nonpressure_residual_scale_override.has_value()
            ? pspg_nonpressure_residual_scale_override->value
            : FE::Real{1.0};
    const auto pspg_pressure_gradient_cut_volume_scale_cap_override =
        navierStokesPspgPressureGradientCutVolumeScaleCap();
    const FE::Real pspg_pressure_gradient_cut_volume_scale_cap =
        pspg_pressure_gradient_cut_volume_scale_cap_override.has_value()
            ? pspg_pressure_gradient_cut_volume_scale_cap_override->value
            : FE::Real{1.0};
    const bool pspg_continuity_full_cell_support =
        navierStokesPspgContinuityFullCellSupportDiagnosticEnabled();
    const auto pspg_pressure_gradient_form_override =
        navierStokesPspgPressureGradientForm();
    const PspgPressureGradientForm pspg_pressure_gradient_form =
        pspg_pressure_gradient_form_override.has_value()
            ? pspg_pressure_gradient_form_override->value
            : PspgPressureGradientForm::Absolute;
    const auto pspg_boundary_pressure_gradient_scale_override =
        navierStokesPspgBoundaryPressureGradientScale();
    const FE::Real pspg_boundary_pressure_gradient_scale =
        pspg_boundary_pressure_gradient_scale_override.has_value()
            ? pspg_boundary_pressure_gradient_scale_override->value
            : FE::Real{0.0};
    const auto pspg_boundary_pressure_flux_scale_override =
        navierStokesPspgBoundaryPressureFluxScale();
    const FE::Real pspg_boundary_pressure_flux_scale =
        pspg_boundary_pressure_flux_scale_override.has_value()
            ? pspg_boundary_pressure_flux_scale_override->value
            : FE::Real{0.0};
    const auto pspg_boundary_tangential_pressure_gradient_scale_override =
        navierStokesPspgBoundaryTangentialPressureGradientScale();
    const FE::Real pspg_boundary_tangential_pressure_gradient_scale =
        pspg_boundary_tangential_pressure_gradient_scale_override.has_value()
            ? pspg_boundary_tangential_pressure_gradient_scale_override->value
            : FE::Real{0.0};
    const auto pspg_boundary_tangential_momentum_residual_scale_override =
        navierStokesPspgBoundaryTangentialMomentumResidualScale();
    const FE::Real pspg_boundary_tangential_momentum_residual_scale =
        pspg_boundary_tangential_momentum_residual_scale_override.has_value()
            ? pspg_boundary_tangential_momentum_residual_scale_override->value
            : FE::Real{0.0};
    if (vms_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_vms_override") +
            " env=" + vms_override->name +
            " raw='" + vms_override->raw + "'" +
            " option_enable_vms=" + (options_.enable_vms ? "1" : "0") +
            " effective_enable_vms=" + (enable_vms ? "1" : "0"));
    }
    if (pspg_pressure_gradient_scale_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_pressure_gradient_scale") +
            " env=" + pspg_pressure_gradient_scale_override->name +
            " raw='" + pspg_pressure_gradient_scale_override->raw + "'" +
            " scale=" + std::to_string(
                static_cast<double>(pspg_pressure_gradient_scale)));
    }
    if (pspg_nonpressure_residual_scale_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_nonpressure_residual_scale") +
            " env=" + pspg_nonpressure_residual_scale_override->name +
            " raw='" + pspg_nonpressure_residual_scale_override->raw + "'" +
            " scale=" + std::to_string(
                static_cast<double>(pspg_nonpressure_residual_scale)));
    }
    if (pspg_pressure_gradient_cut_volume_scale_cap_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_pressure_gradient_cut_volume_scale_cap") +
            " env=" +
            pspg_pressure_gradient_cut_volume_scale_cap_override->name +
            " raw='" +
            pspg_pressure_gradient_cut_volume_scale_cap_override->raw + "'" +
            " cap=" + std::to_string(
                static_cast<double>(
                    pspg_pressure_gradient_cut_volume_scale_cap)));
    }
    if (pspg_pressure_gradient_form_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_pressure_gradient_form") +
            " env=" + pspg_pressure_gradient_form_override->name +
            " raw='" + pspg_pressure_gradient_form_override->raw + "'" +
            " form=" +
            pspgPressureGradientFormName(pspg_pressure_gradient_form));
    }
    if (pspg_boundary_pressure_gradient_scale_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_pressure_gradient_scale") +
            " env=" + pspg_boundary_pressure_gradient_scale_override->name +
            " raw='" + pspg_boundary_pressure_gradient_scale_override->raw + "'" +
            " scale=" + std::to_string(
                static_cast<double>(
                    pspg_boundary_pressure_gradient_scale)));
    }
    if (pspg_boundary_pressure_flux_scale_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_pressure_flux_scale") +
            " env=" + pspg_boundary_pressure_flux_scale_override->name +
            " raw='" + pspg_boundary_pressure_flux_scale_override->raw + "'" +
            " scale=" + std::to_string(
                static_cast<double>(
                    pspg_boundary_pressure_flux_scale)));
    }
    if (pspg_boundary_tangential_pressure_gradient_scale_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_tangential_pressure_gradient_scale") +
            " env=" +
            pspg_boundary_tangential_pressure_gradient_scale_override->name +
            " raw='" +
            pspg_boundary_tangential_pressure_gradient_scale_override->raw +
            "'" +
            " scale=" + std::to_string(
                static_cast<double>(
                    pspg_boundary_tangential_pressure_gradient_scale)));
    }
    if (pspg_boundary_tangential_momentum_residual_scale_override.has_value()) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_tangential_momentum_residual_scale") +
            " env=" +
            pspg_boundary_tangential_momentum_residual_scale_override->name +
            " raw='" +
            pspg_boundary_tangential_momentum_residual_scale_override->raw +
            "'" +
            " scale=" + std::to_string(
                static_cast<double>(
                    pspg_boundary_tangential_momentum_residual_scale)));
    }
    if (enable_vms && !(options_.stabilization_epsilon > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: stabilization_epsilon must be > 0 when VMS is enabled");
    }

    FE::systems::FieldSpec u_spec;
    u_spec.name = options_.velocity_field_name;
    u_spec.space = velocity_space_;
    u_spec.components = dim;

    FE::systems::FieldSpec p_spec;
    p_spec.name = options_.pressure_field_name;
    p_spec.space = pressure_space_;
    p_spec.components = 1;

    std::optional<FE::systems::FieldSpec> body_force_spec;
    if (!options_.body_force_field_name.empty()) {
        FE::systems::FieldSpec source_spec;
        source_spec.name = options_.body_force_field_name;
        source_spec.space = velocity_space_;
        source_spec.components = dim;
        source_spec.source_kind =
            FE::systems::FieldSourceKind::PrescribedData;
        body_force_spec = std::move(source_spec);
    }

    const FE::systems::ALEBindingOptions ale_options{
        .enabled = options_.enable_ale,
        .dimension = dim,
        .mesh_velocity_source =
            options_.mesh_velocity_source ==
                    ALEMeshVelocitySource::CoupledDisplacement
                ? FE::systems::ALEMeshVelocitySource::CoupledDisplacement
                : FE::systems::ALEMeshVelocitySource::PrescribedData,
        .geometry_tangent_path = options_.moving_mesh_tangent_path,
        .mesh_velocity_field_name = options_.mesh_velocity_field_name,
        .mesh_displacement_field_name =
            options_.mesh_displacement_field_name,
        .mesh_velocity_space = options_.mesh_velocity_space
            ? options_.mesh_velocity_space
            : velocity_space_,
        .mesh_displacement_space = velocity_space_,
        .auto_register_mesh_velocity_field =
            options_.auto_register_mesh_velocity_field,
        .auto_register_mesh_displacement_field =
            options_.auto_register_mesh_displacement_field,
    };

    // Preflight every free-surface field, policy, ownership relation, and
    // generated-marker request before adding Navier--Stokes fields, forms, or
    // constraints.  A rejected configuration must leave the system definition
    // unchanged.
    std::vector<FreeSurfaceBoundary> effective_free_surfaces;
    effective_free_surfaces.reserve(options_.free_surface.size());
    std::set<int> prospective_fitted_normal_markers;
    for (const auto& bc : options_.free_surface) {
        auto effective_bc = withResolvedInterfaceMarker(bc, system);
        if (effective_bc.active_domain_method ==
                FreeSurfaceActiveDomainMethod::SmoothedIndicator &&
            !options_.explicit_legacy_configuration) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: SmoothedIndicator is an explicit legacy diagnostic and requires configuration schema version 1 with legacy behavior enabled");
        }
        validateFreeSurfaceBoundary(effective_bc,
                                    options_,
                                    options_.enable_ale,
                                    system,
                                    dim);
        if (effective_bc.implementation ==
                FreeSurfaceImplementation::FittedALE &&
            options_.enable_ale &&
            options_.mesh_velocity_source ==
                ALEMeshVelocitySource::CoupledDisplacement) {
            if (!system.meshTangentialBoundaryPolicyHistory().empty()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface tangential policies cannot be installed "
                    "after accepted tangential-policy history has begun");
            }
            auto displacement_field = system.meshMotionField(
                FE::systems::MeshMotionFieldRole::Displacement);
            if (!displacement_field.has_value()) {
                const auto named = system.findFieldByName(
                    options_.mesh_displacement_field_name);
                if (named != FE::INVALID_FIELD_ID) {
                    displacement_field = named;
                }
            }
            const bool installs_normal_relation =
                effective_bc.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::None &&
                effective_bc.normal_kinematic_policy ==
                    FreeSurfaceNormalKinematicPolicy::
                        MatchFluidNormalVelocity;
            if (installs_normal_relation &&
                !system.meshNormalBoundaryConstraintHistory().empty()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface normal relations cannot be installed "
                    "after accepted normal-constraint history has begun");
            }
            if (installs_normal_relation &&
                !prospective_fitted_normal_markers
                     .insert(effective_bc.boundary_marker)
                     .second) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: duplicate fitted "
                    "free-surface normal relation on marker " +
                    std::to_string(effective_bc.boundary_marker));
            }
            if (installs_normal_relation &&
                displacement_field.has_value()) {
                const auto& displacement_record =
                    system.fieldRecord(*displacement_field);
                if (displacement_record.space == nullptr ||
                    velocity_space_->polynomial_order() >
                        displacement_record.space->polynomial_order()) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: fitted-ALE "
                        "normal measurement quadrature requires velocity "
                        "order " +
                        std::to_string(
                            velocity_space_->polynomial_order()) +
                        " <= displacement-primary order " +
                        (displacement_record.space != nullptr
                             ? std::to_string(
                                   displacement_record.space
                                       ->polynomial_order())
                             : std::string("unavailable")) +
                        " on marker " +
                        std::to_string(
                            effective_bc.boundary_marker));
                }
                const auto normal_conflict = std::find_if(
                    system.meshNormalBoundaryConstraints().begin(),
                    system.meshNormalBoundaryConstraints().end(),
                    [&](const auto& declaration) {
                        return declaration.boundary_marker ==
                                   effective_bc.boundary_marker &&
                               declaration.mesh_displacement_field ==
                                   *displacement_field;
                    });
                if (normal_conflict !=
                    system.meshNormalBoundaryConstraints().end()) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: fitted "
                        "free-surface normal relation on marker " +
                        std::to_string(effective_bc.boundary_marker) +
                        " conflicts with existing mesh-motion owner '" +
                        normal_conflict->owner_component + "'");
                }
            }
            const auto conflict = std::find_if(
                system.meshTangentialBoundaryPolicies().begin(),
                system.meshTangentialBoundaryPolicies().end(),
                [&](const auto& declaration) {
                    return declaration.boundary_marker ==
                               effective_bc.boundary_marker &&
                           (!displacement_field.has_value() ||
                            declaration.mesh_displacement_field ==
                                *displacement_field);
                });
            if (conflict !=
                system.meshTangentialBoundaryPolicies().end()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface tangential policy on marker " +
                    std::to_string(effective_bc.boundary_marker) +
                    " conflicts with existing mesh-motion owner '" +
                    conflict->owner_component + "'");
            }
        }
        effective_free_surfaces.push_back(std::move(effective_bc));
    }
    validateGeneratedFreeSurfaceMarkerUniqueness(
        effective_free_surfaces, system);
    validateActiveDomainPressureConstraints(
        system,
        options_,
        effective_free_surfaces);
    const auto active_pressure_domain =
        activePressureDomainFor(effective_free_surfaces);
    if (!(options_
              .generated_boundary_nitsche_minimum_energy_ratio >
          FE::Real{0.0}) ||
        !(options_
              .generated_boundary_nitsche_minimum_energy_ratio <
          FE::Real{1.0}) ||
        !std::isfinite(
            options_
                .generated_boundary_nitsche_minimum_energy_ratio)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "generated_boundary_nitsche_minimum_energy_ratio must be "
            "finite and strictly between zero and one");
    }
    if (!options_.velocity_dirichlet_weak.empty() &&
        (!(options_.nitsche_gamma > FE::Real{0.0}) ||
         !std::isfinite(options_.nitsche_gamma))) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "nitsche_gamma must be finite and > 0 when weak velocity "
            "Dirichlet conditions are configured");
    }
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const bool native_mesh_available_for_trace_certificate =
        system.mesh() != nullptr;
#else
    const bool native_mesh_available_for_trace_certificate =
        false;
#endif
    const bool has_generated_boundary_nitsche_route =
        std::any_of(
            options_.velocity_dirichlet_weak.begin(),
            options_.velocity_dirichlet_weak.end(),
            [&](const auto& bc) {
                return sharpActiveBoundaryMarkerFor(
                           FE::forms::bc::detail::
                               boundaryMarkerOrThrow(
                                   bc,
                                   "IncompressibleNavierStokesVMSModule::"
                                   "registerOn generated-boundary "
                                   "Nitsche trace preflight"),
                           effective_free_surfaces,
                           system)
                    .has_value();
            });

    requireDistinctFieldNames({
        {"velocity", u_spec.name},
        {"pressure", p_spec.name},
        {"body-force data",
         body_force_spec.has_value() ? std::string_view(body_force_spec->name)
                                     : std::string_view{}},
        {"mesh velocity",
         options_.enable_ale
             ? std::string_view(ale_options.mesh_velocity_field_name)
             : std::string_view{}},
        {"mesh displacement",
         options_.enable_ale &&
                 ale_options.mesh_velocity_source ==
                     FE::systems::ALEMeshVelocitySource::CoupledDisplacement
             ? std::string_view(ale_options.mesh_displacement_field_name)
             : std::string_view{}},
    });
    validateCompatibleField(
        system,
        u_spec,
        FE::systems::FieldSourceKind::Unknown,
        /*allow_missing=*/true,
        "IncompressibleNavierStokesVMSModule::registerOn velocity");
    validateCompatibleField(
        system,
        p_spec,
        FE::systems::FieldSourceKind::Unknown,
        /*allow_missing=*/true,
        "IncompressibleNavierStokesVMSModule::registerOn pressure");
    if (body_force_spec.has_value()) {
        validateCompatibleField(
            system,
            *body_force_spec,
            FE::systems::FieldSourceKind::PrescribedData,
            options_.auto_register_body_force_field,
            "IncompressibleNavierStokesVMSModule::registerOn momentum source");
    }
    FE::systems::validateALEBinding(system, ale_options);
    if (ale_options.mesh_velocity_source !=
        FE::systems::ALEMeshVelocitySource::CoupledDisplacement) {
        for (const auto& bc : effective_free_surfaces) {
            for (const auto& contact_line : bc.contact_lines) {
                if (contactLineKind(contact_line) == ContactLineKind::Pinned) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require ALE mesh velocity to be derived from a coupled mesh displacement unknown");
                }
            }
        }
    }
    validateNavierStokesBoundaryConfiguration(
        options_,
        effective_free_surfaces,
        system,
        *velocity_space_,
        *pressure_space_,
        dim,
        pspg_boundary_pressure_gradient_scale > FE::Real{0.0} ||
            pspg_boundary_pressure_flux_scale > FE::Real{0.0} ||
            pspg_boundary_tangential_pressure_gradient_scale >
                FE::Real{0.0} ||
            pspg_boundary_tangential_momentum_residual_scale >
                FE::Real{0.0});
    if (has_generated_boundary_nitsche_route) {
        if (!active_pressure_domain.has_value() ||
            active_pressure_domain->boundary
                    ->active_domain_method !=
                FreeSurfaceActiveDomainMethod::CutVolume ||
            !active_pressure_domain->boundary
                 ->small_cut_aggregation) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche routes require CutVolume "
                "small-cut aggregation and aggregate-trace certification");
        }
        if (!native_mesh_available_for_trace_certificate) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche routes require a native mesh "
                "for aggregate-trace certification");
        }
        if (options_.viscosity_model != nullptr ||
            !(options_.viscosity > FE::Real{0.0}) ||
            !std::isfinite(options_.viscosity)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "certified generated-boundary Nitsche routes require "
                "finite positive constant Newtonian viscosity");
        }
        const bool supported_trace_space =
            isSupportedGeneratedBoundaryTraceSpace(
                *velocity_space_, dim);
        const auto existing_velocity =
            system.findFieldByName(
                options_.velocity_field_name);
        bool supported_existing_velocity = true;
        if (existing_velocity != FE::INVALID_FIELD_ID) {
            const auto& record =
                system.fieldRecord(existing_velocity);
            supported_existing_velocity =
                record.space &&
                record.components == dim &&
                isSupportedGeneratedBoundaryTraceSpace(
                    *record.space, dim);
        }
        if (!supported_trace_space) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "certified generated-boundary Nitsche routes require "
                "an affine P1 Product H1 velocity space on Triangle3 "
                "or Tetra4 cells");
        }
        if (!supported_existing_velocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "the existing velocity field is outside the affine P1 "
                "Product H1 Triangle3/Tetra4 generated-boundary trace "
                "certificate envelope");
        }
    }
    validateFreeSurfaceContactGeometryPreflight(
        effective_free_surfaces, system);
    validateFreeSurfaceDiscreteFunctionalPreflight(
        effective_free_surfaces, system);

    const FE::FieldId u_id = ensureCompatibleUnknownField(
        system,
        std::move(u_spec),
        "IncompressibleNavierStokesVMSModule::registerOn velocity");

    const FE::FieldId p_id = ensureCompatibleUnknownField(
        system,
        std::move(p_spec),
        "IncompressibleNavierStokesVMSModule::registerOn pressure");

    FE::FieldId body_force_field_id = FE::INVALID_FIELD_ID;
    if (body_force_spec.has_value()) {
        body_force_field_id = ensureCompatiblePrescribedField(
            system,
            std::move(*body_force_spec),
            options_.auto_register_body_force_field,
            "IncompressibleNavierStokesVMSModule::registerOn momentum source");
    }

    declareFreeSurfaceDiscreteFunctionals(
        effective_free_surfaces,
        system,
        u_id,
        options_.density,
        options_.body_force,
        options_.viscosity,
        options_.viscosity_model == nullptr);

    const auto generated_active_boundary_for =
        [&system, &effective_free_surfaces](int physical_boundary_marker)
            -> std::optional<int> {
            const auto marker = sharpActiveBoundaryMarkerFor(
                physical_boundary_marker,
                effective_free_surfaces,
                system);
            if (marker.has_value()) {
                system.registerGeneratedEmbeddedInterfaceMarker(*marker);
            }
            return marker;
        };
    const auto integrate_on_physical_boundary =
        [&generated_active_boundary_for](const FE::forms::FormExpr& integrand,
                                         int physical_boundary_marker) {
            const auto marker = generated_active_boundary_for(
                physical_boundary_marker);
            const auto measure =
                marker.has_value()
                    ? FE::forms::ExteriorBoundaryMeasure::
                          generatedActiveSubset(
                              physical_boundary_marker,
                              *marker)
                    : FE::forms::ExteriorBoundaryMeasure::
                          fullPhysical(
                              physical_boundary_marker);
            return integrand.dExteriorBoundary(measure);
        };
    struct PendingSmallCutAggregation {
        FE::geometry::CutIntegrationSide side{
            FE::geometry::CutIntegrationSide::Negative};
        int interface_marker{-1};
        FE::constraints::SmallCutAggregationGuardOptions guards{};
    };
    std::optional<PendingSmallCutAggregation> pending_small_cut_aggregation;
    if (active_pressure_domain.has_value() &&
        active_pressure_domain->boundary->active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume) {
        auto& gauge_registry = system.gaugeRegistry();
        constexpr std::string_view free_surface_pressure_anchor =
            "Unfitted CutVolume embedded free-surface natural traction anchors absolute pressure";
        const bool anchor_already_registered = std::any_of(
            gauge_registry.anchoring().begin(),
            gauge_registry.anchoring().end(),
            [&](const FE::gauge::AnchoringEvidence& evidence) {
                return evidence.field == p_id &&
                       evidence.family ==
                           FE::gauge::NullspaceModeFamily::ScalarConstant &&
                       evidence.source == free_surface_pressure_anchor;
            });
        if (!anchor_already_registered) {
            gauge_registry.addAnchoring(FE::gauge::AnchoringEvidence{
                .field = p_id,
                .component = -1,
                .region = -1,
                .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
                .verdict = FE::gauge::AnchoringVerdict::Anchored,
                .source = std::string(free_surface_pressure_anchor),
            });
        }

        const auto& pressure_anchor_boundary =
            *active_pressure_domain->boundary;
        auto pressure_anchor_measure_guard =
            std::make_shared<CutVolumePressureAnchorMeasureGuard>(
                pressure_anchor_boundary.interface_marker,
                pressure_anchor_boundary.level_set_field_name,
                pressure_anchor_boundary.generated_interface_domain_id);
        system.addGlobalKernel(options_.operator_tag,
                               pressure_anchor_measure_guard);
        system.addCutIntegrationContextUpdateCallback(
            FE::systems::CutIntegrationContextUpdateCallback{
                .name = "navier_stokes_cut_volume_pressure_anchor_measure:" +
                        std::to_string(
                            pressure_anchor_boundary.interface_marker),
                .callback =
                    [&system, pressure_anchor_measure_guard](
                        const FE::assembly::CutIntegrationContext* context) {
                        if (!coordinateCutContextCallbackLocalPhase(
                                system,
                                std::exception_ptr{},
                                context != nullptr,
                                "cut_volume_pressure_anchor_context")) {
                            // Use the same null-candidate diagnostic on every
                            // rank without letting a rank-local non-null
                            // candidate enter the measure reductions.
                            pressure_anchor_measure_guard
                                ->validateContextUpdate(system, nullptr);
                            throw std::runtime_error(
                                "IncompressibleNavierStokesVMSModule: "
                                "CutVolume pressure-anchor null context was "
                                "unexpectedly accepted");
                        }
                        pressure_anchor_measure_guard->validateContextUpdate(
                            system, context);
                    },
            });
    }

    if (active_pressure_domain.has_value() &&
        active_pressure_domain->boundary->active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume) {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        const bool native_mesh_available = system.mesh() != nullptr;
#else
        const bool native_mesh_available = false;
#endif
        if (!native_mesh_available) {
            FE_LOG_WARNING(
                "IncompressibleNavierStokesVMSModule: skipping active-domain "
                "inactive velocity/pressure constraints because native mesh support "
                "is unavailable in this FESystem");
        } else {
            const auto side =
                active_pressure_domain->active_domain ==
                        FreeSurfaceActiveDomain::LevelSetPositive
                    ? FE::constraints::LevelSetConstraintSide::Positive
                    : FE::constraints::LevelSetConstraintSide::Negative;
            const bool constrain_inactive_velocity =
                constrainInactiveActiveDomainVelocity(
                    *active_pressure_domain->boundary);
            if (constrain_inactive_velocity) {
                system.addSystemConstraint(
                    std::make_unique<
                        FE::constraints::LevelSetActiveSideVertexDirichletConstraint>(
                        u_id,
                        active_pressure_domain->boundary->level_set_field_name,
                        side,
                        active_pressure_domain->boundary->level_set_isovalue,
                        FE::Real{0.0},
                        active_pressure_domain->boundary->interface_marker));
            } else {
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: inactive active-domain velocity constraints disabled") +
                    " marker=" +
                    std::to_string(
                        active_pressure_domain->boundary->interface_marker) +
                    " level_set_field='" +
                    active_pressure_domain->boundary->level_set_field_name +
                    "' reason=velocity_extension_enabled");
            }
            system.addSystemConstraint(
                std::make_unique<
                    FE::constraints::LevelSetActiveSideVertexDirichletConstraint>(
                    p_id,
                    active_pressure_domain->boundary->level_set_field_name,
                    side,
                    active_pressure_domain->boundary->level_set_isovalue,
                    FE::Real{0.0},
                    active_pressure_domain->boundary->interface_marker));
            if (active_pressure_domain->boundary->small_cut_aggregation) {
                const auto aggregation_side =
                    active_pressure_domain->active_domain ==
                            FreeSurfaceActiveDomain::LevelSetPositive
                        ? FE::geometry::CutIntegrationSide::Positive
                        : FE::geometry::CutIntegrationSide::Negative;
                // Defer aggregation until every strong velocity/pressure
                // constraint has been registered below.  Small-cut
                // aggregation then observes essential data component by
                // component: a normal-only impermeability condition replaces
                // only the normal line, while tangential slip DOFs retain
                // their aggregation support.  The former marker-wide
                // exclusion incorrectly removed those tangential lines.
                pending_small_cut_aggregation = PendingSmallCutAggregation{
                    .side = aggregation_side,
                    .interface_marker =
                        active_pressure_domain->boundary->interface_marker,
                    .guards = {
                        .maximum_root_path_length =
                            active_pressure_domain->boundary
                                ->small_cut_aggregation_guards
                                .maximum_root_path_length,
                        .maximum_reference_extrapolation_distance =
                            active_pressure_domain->boundary
                                ->small_cut_aggregation_guards
                                .maximum_reference_extrapolation_distance,
                        .maximum_absolute_coefficient =
                            active_pressure_domain->boundary
                                ->small_cut_aggregation_guards
                                .maximum_absolute_coefficient,
                        .maximum_row_l1_norm =
                            active_pressure_domain->boundary
                                ->small_cut_aggregation_guards
                                .maximum_row_l1_norm,
                    },
                };
            }
        }
    }

    if (!options_.node_pressure_constraints.values.empty()) {
        std::vector<FE::constraints::VertexDirichletValue> values;
        values.reserve(options_.node_pressure_constraints.values.size());
        for (const auto& in : options_.node_pressure_constraints.values) {
            values.push_back(FE::constraints::VertexDirichletValue{in.node_id, in.pressure});
        }

        FE::constraints::VertexIdMode mode = FE::constraints::VertexIdMode::GlobalVertexGid;
        switch (options_.node_pressure_constraints.id_type) {
        case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::GlobalVertexGid:
            mode = FE::constraints::VertexIdMode::GlobalVertexGid;
            break;
        case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::LocalVertexId:
            mode = FE::constraints::VertexIdMode::LocalVertexId;
            break;
        }

        system.addSystemConstraint(
            std::make_unique<FE::constraints::VertexDirichletConstraint>(p_id, std::move(values), mode));
    }

    using namespace svmp::FE::forms;

    const auto u = StateField(u_id, *velocity_space_, options_.velocity_field_name);
    const auto p = StateField(p_id, *pressure_space_, options_.pressure_field_name);

    const auto v = TestField(u_id, *velocity_space_, "v");
    const auto q = TestField(p_id, *pressure_space_, "q");

    const auto ale_binding =
        FE::systems::resolveALEBinding(system, ale_options);

    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
        normal_declarations;
    if (ale_binding.coupled()) {
        normal_declarations.reserve(effective_free_surfaces.size());
        for (const auto& boundary : effective_free_surfaces) {
            const bool installs_normal_relation =
                boundary.implementation ==
                    FreeSurfaceImplementation::FittedALE &&
                boundary.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::None &&
                boundary.normal_kinematic_policy ==
                    FreeSurfaceNormalKinematicPolicy::
                        MatchFluidNormalVelocity;
            if (!installs_normal_relation) {
                continue;
            }
            normal_declarations.push_back(
                FE::systems::MeshNormalBoundaryConstraintDeclaration{
                    .mesh_displacement_field =
                        ale_binding.mesh_displacement_field,
                    .boundary_marker = boundary.boundary_marker,
                    .quantity = FE::systems::MeshNormalBoundaryQuantity::
                        MeshVelocityTrace,
                    .target_kind =
                        FE::systems::MeshNormalBoundaryTargetKind::
                            FluidNormalVelocity,
                    .target_expression =
                        normalTrace(u, currentNormal()),
                    .enforcement_kind =
                        FE::analysis::EnforcementKind::WeakPenalty,
                    .related_velocity_field = u_id,
                    .owner_component =
                        "IncompressibleNavierStokesVMSModule."
                        "FreeSurfaceBoundary",
                });
        }
    }

    if (!system.hasOperator(options_.operator_tag)) {
        system.addOperator(options_.operator_tag);
    }

    const auto active_volume_domain =
        activeVolumeDomainFor(effective_free_surfaces, system);

    const auto rho = FormExpr::constant(options_.density);

    // Body force/source acceleration. The optional field is evaluated as
    // prescribed data, so it contributes to both Galerkin forcing and VMS
    // strong residual without adding unknowns to the system.
    std::vector<FormExpr> f_comp;
    f_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        f_comp.push_back(FormExpr::constant(options_.body_force[static_cast<std::size_t>(d)]));
    }
    FormExpr f = FormExpr::asVector(std::move(f_comp));
    if (options_.has_body_force_spacetime) {
        std::vector<FormExpr> source_comp;
        source_comp.reserve(static_cast<std::size_t>(dim));
        for (int d = 0; d < dim; ++d) {
            source_comp.push_back(bc::toScalarExpr(
                options_.body_force_spacetime[static_cast<std::size_t>(d)],
                "ns_body_force_spacetime_" + std::to_string(d)));
        }
        f = f + FormExpr::asVector(std::move(source_comp));
    }
    if (body_force_field_id != FE::INVALID_FIELD_ID) {
        f = f + StateField(
                    body_force_field_id,
                    *velocity_space_,
                    options_.body_force_field_name);
    }
    if (options_.rotating_frame_coriolis_enabled) {
        std::vector<FormExpr> omega_comp;
        omega_comp.reserve(3u);
        for (int d = 0; d < 3; ++d) {
            omega_comp.push_back(bc::toScalarExpr(
                options_.rotating_frame_angular_velocity[static_cast<std::size_t>(d)],
                "ns_rotating_frame_angular_velocity_" + std::to_string(d)));
        }
        f = f + FormExpr::constant(-2.0) *
                    cross(FormExpr::asVector(std::move(omega_comp)), u);
    }

    const auto eps_for_mu = sym(grad(u));
    const auto gamma_for_mu =
        sqrt(FormExpr::constant(2.0) * inner(eps_for_mu, eps_for_mu));
    FormExpr mu;
    if (options_.viscosity_model) {
        // Variable viscosity remains a tagged constitutive expression so
        // installFormulation() can publish the law from the residual DAG.
        std::shared_ptr<const FE::forms::ConstitutiveModel> viscosity_model =
            options_.viscosity_model;
        auto viscosity_metadata = FE::analysis::dynamicViscosityMetadata(
            FE::INVALID_FIELD_ID,
            options_.viscosity,
            options_.viscosity_model);
        viscosity_model = FE::constitutive::withConstitutiveLawMetadata(
            std::move(viscosity_model),
            0u,
            std::move(viscosity_metadata));
        mu = constitutive(std::move(viscosity_model), gamma_for_mu).out(0);
    } else {
        mu = FormExpr::constant(options_.viscosity);
    }

    // ALE uses relative convection u - w_mesh. Static/default paths remain unchanged.
    const auto zero = zeroVector(dim);
    const auto w_mesh = meshVelocity();
    const auto mesh_velocity = options_.enable_ale ? w_mesh : zero;
    const auto a = options_.enable_convection
                       ? (options_.enable_ale ? (u - w_mesh) : u)
                       : zero;
    const bool include_mcv =
        options_.enable_ale && options_.include_moving_control_volume_transient;
    const auto moving_volume_strong =
        include_mcv ? (div(mesh_velocity) * u) : zero;

    // Strong momentum residual (full, including dt(u)):
    //   R_m = rho*(dt(u) + grad(u)*a + chi*div(w_mesh)*u - f)
    //         + grad(p) - div(2 mu sym(grad(u)))
    // with a = u - w_mesh for ALE and chi set by the moving-control-volume option.
    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto pressure_gradient_residual = grad(p);
    const auto pspg_pressure_gradient_pressure =
        pspg_pressure_gradient_form == PspgPressureGradientForm::Incremental
            ? FormExpr::effectiveTimeStep() * dt(p)
            : p;
    const auto pspg_pressure_gradient_residual =
        grad(pspg_pressure_gradient_pressure);
    const auto r_m_without_pressure =
        rho * (dt(u) + grad(u) * a + moving_volume_strong - f) - div(stress);
    const auto r_m = r_m_without_pressure + pressure_gradient_residual;

    // Galerkin terms.
    const auto inertia = rho * inner(dt(u), v);
    const auto moving_volume =
        include_mcv ? rho * div(w_mesh) * inner(u, v) : FormExpr::constant(0.0);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    const auto galerkin_momentum_integrand =
        inertia + moving_volume + convection + viscous + pressure + forcing;
    const auto galerkin_continuity_integrand = q * div(u);

    FormExpr active_momentum_integrand = galerkin_momentum_integrand;
    FormExpr active_continuity_integrand = galerkin_continuity_integrand;
    FormExpr vms_pspg_continuity_integrand;
    FormExpr vms_pspg_pressure_gradient_integrand;
    FormExpr vms_pspg_nonpressure_integrand;
    FormExpr vms_pspg_boundary_pressure_gradient_form;
    FormExpr vms_pspg_boundary_pressure_flux_form;
    FormExpr vms_pspg_boundary_tangential_pressure_gradient_form;
    FormExpr vms_pspg_boundary_tangential_momentum_residual_form;

    if (enable_vms) {
        // Residual-based VMS with static subscales:
        //   u' = -tau_M * R_m
        //   p' = -tau_C * (div u)
        // and coarse-scale stabilization terms assembled from (u', p').
        const auto eps = FormExpr::constant(options_.stabilization_epsilon);
        const auto dt_step = FormExpr::effectiveTimeStep();
        const auto ct_m = FormExpr::constant(options_.ct_m);
        const auto ct_c = FormExpr::constant(options_.ct_c);

        // Element metric tensor Kxi = J^{-T} J^{-1}. FE Forms exposes Jinv()
        // with the active physical dimension, so 2D contractions do not include
        // a dummy frame-thickness component.
        const auto Jinv_expr = Jinv();
        const auto K = transpose(Jinv_expr) * Jinv_expr;
        const auto nu = mu / rho;

        // Legacy-inspired tau_M (stored here as tau_M/rho, matching legacy fluid.cpp naming).
        const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
        const auto kU = inner(a, K * a);
        const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
        const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));

        const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

        const auto u_sub = -tau_m * r_m;
        const auto p_sub = -tau_c * div(u);

        // Advection velocity for convection-related terms (disabled for Stokes).
        const auto u_adv = options_.enable_convection ? (u + u_sub - mesh_velocity) : a;
        const auto p_adv = p + p_sub;

        // Momentum: Galerkin + VMS (SUPG-like) + pressure-subscale (LSIC-like).
        const auto convection_adv = rho * inner(grad(u) * u_adv, v);
        const auto pressure_adv = -p_adv * div(v);
        // Legacy-style full VMS: use the subscale-augmented advection velocity in the
        // test-function stabilization term and include the tauB-based cross-stress closure.
        const auto supg = -rho * inner(grad(v) * u_adv, u_sub);

        // tauB cross-stress closure (legacy fluid.cpp):
        //   tauB = rho / sqrt( u'^T Kxi u' )
        // and adds + (u' · ∇v) · ( tauB * (u' · ∇)u ).
        FormExpr cross_stress = FormExpr::constant(0.0);
        if (options_.enable_convection) {
            const auto tau_b = rho / sqrt(inner(u_sub, K * u_sub) + eps);
            const auto rV_tau = tau_b * (grad(u) * u_sub); // (tauB * (u'·∇)u)
            cross_stress = inner(grad(v) * u_sub, rV_tau);
        }

        active_momentum_integrand =
            inertia + moving_volume + convection_adv + viscous + pressure_adv +
            forcing + supg + cross_stress;

        // Continuity: Galerkin + VMS (PSPG-like).
        const auto pspg_pressure_gradient_scale_expr =
            FormExpr::constant(pspg_pressure_gradient_scale);
        FormExpr pspg_pressure_gradient_support_scale =
            FormExpr::constant(1.0);
        if (pspg_pressure_gradient_cut_volume_scale_cap_override.has_value()) {
            if (active_volume_domain.has_value() &&
                active_volume_domain->method ==
                    FreeSurfaceActiveDomainMethod::CutVolume &&
                pspg_pressure_gradient_cut_volume_scale_cap > FE::Real{1.0}) {
                const auto fraction_floor = FormExpr::constant(1.0e-12);
                pspg_pressure_gradient_support_scale =
                    FE::forms::min(
                        FormExpr::constant(
                            pspg_pressure_gradient_cut_volume_scale_cap),
                        FormExpr::constant(1.0) /
                            FE::forms::max(
                                FE::forms::cutVolumeFraction(),
                                fraction_floor));
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_pressure_gradient_cut_volume_support_scale") +
                    " status=installed"
                    " form=cut_volume_fraction_inverse_cap"
                    " cap=" + std::to_string(
                        static_cast<double>(
                            pspg_pressure_gradient_cut_volume_scale_cap)));
            } else {
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_pressure_gradient_cut_volume_support_scale") +
                    " status=skipped"
                    " reason=no_cut_volume_active_domain_or_cap_leq_one"
                    " cap=" + std::to_string(
                        static_cast<double>(
                            pspg_pressure_gradient_cut_volume_scale_cap)));
            }
        }
        const auto pspg_continuity_momentum_residual =
            FormExpr::constant(pspg_nonpressure_residual_scale) *
            r_m_without_pressure +
            pspg_pressure_gradient_scale_expr *
            pspg_pressure_gradient_support_scale *
            pspg_pressure_gradient_residual;
        vms_pspg_pressure_gradient_integrand =
            pspg_pressure_gradient_scale_expr *
            pspg_pressure_gradient_support_scale *
            inner(grad(q), tau_m * pspg_pressure_gradient_residual);
        vms_pspg_nonpressure_integrand =
            FormExpr::constant(pspg_nonpressure_residual_scale) *
            inner(grad(q), tau_m * r_m_without_pressure);
        vms_pspg_continuity_integrand =
            vms_pspg_pressure_gradient_integrand +
            vms_pspg_nonpressure_integrand;
        if (pspg_continuity_full_cell_support) {
            active_continuity_integrand = galerkin_continuity_integrand;
            FE_LOG_INFO(
                "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_continuity_volume_support"
                " status=installed"
                " form=full_cell_vms_pspg_plus_active_galerkin"
                " qualification=DiagnosticOnly"
                " env=SVMP_NS_PSPG_CONTINUITY_FULL_CELL_SUPPORT");
        } else {
            active_continuity_integrand =
                galerkin_continuity_integrand + vms_pspg_nonpressure_integrand;
        }

        if (pspg_boundary_pressure_gradient_scale > FE::Real{0.0}) {
            std::vector<int> wall_markers;
            auto append_marker = [&wall_markers](int marker) {
                if (marker < 0) {
                    return;
                }
                if (std::find(wall_markers.begin(), wall_markers.end(), marker) ==
                    wall_markers.end()) {
                    wall_markers.push_back(marker);
                }
            };
            for (const auto& bc : options_.velocity_dirichlet) {
                append_marker(bc.boundary_marker);
            }
            for (const auto& bc : options_.velocity_dirichlet_weak) {
                append_marker(bc.boundary_marker);
            }

            if (wall_markers.empty()) {
                FE_LOG_WARNING(
                    "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_pressure_gradient"
                    " status=skipped"
                    " reason=no_velocity_dirichlet_markers");
            } else {
                std::sort(wall_markers.begin(), wall_markers.end());
                const auto n_wall = FormExpr::normal();
                const auto boundary_pressure_gradient_integrand =
                    FormExpr::constant(pspg_boundary_pressure_gradient_scale) *
                    h() * tau_m *
                    dot(grad(q), n_wall) *
                    dot(pspg_pressure_gradient_residual, n_wall);
                std::ostringstream marker_stream;
                for (std::size_t i = 0; i < wall_markers.size(); ++i) {
                    const int marker = wall_markers[i];
                    auto boundary_form = integrate_on_physical_boundary(
                        boundary_pressure_gradient_integrand, marker);
                    vms_pspg_boundary_pressure_gradient_form =
                        vms_pspg_boundary_pressure_gradient_form.isValid()
                            ? (vms_pspg_boundary_pressure_gradient_form +
                               boundary_form)
                            : boundary_form;
                    if (i > 0) {
                        marker_stream << "|";
                    }
                    marker_stream << marker;
                }
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_pressure_gradient") +
                    " status=installed"
                    " form=wall_normal_pressure_gradient"
                    " pressure_gradient_form=" +
                    pspgPressureGradientFormName(
                        pspg_pressure_gradient_form) +
                    " scale=" + std::to_string(
                        static_cast<double>(
                            pspg_boundary_pressure_gradient_scale)) +
                    " boundary_markers=" + marker_stream.str());
            }
        }

        if (pspg_boundary_pressure_flux_scale > FE::Real{0.0}) {
            std::vector<int> wall_markers;
            auto append_marker = [&wall_markers](int marker) {
                if (marker < 0) {
                    return;
                }
                if (std::find(wall_markers.begin(), wall_markers.end(), marker) ==
                    wall_markers.end()) {
                    wall_markers.push_back(marker);
                }
            };
            for (const auto& bc : options_.velocity_dirichlet) {
                append_marker(bc.boundary_marker);
            }
            for (const auto& bc : options_.velocity_dirichlet_weak) {
                append_marker(bc.boundary_marker);
            }

            if (wall_markers.empty()) {
                FE_LOG_WARNING(
                    "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_pressure_flux"
                    " status=skipped"
                    " reason=no_velocity_dirichlet_markers");
            } else {
                std::sort(wall_markers.begin(), wall_markers.end());
                const auto n_wall = FormExpr::normal();
                const auto boundary_pressure_flux_integrand =
                    FormExpr::constant(
                        -pspg_boundary_pressure_flux_scale) *
                    q * tau_m *
                    dot(pspg_pressure_gradient_residual, n_wall);
                std::ostringstream marker_stream;
                for (std::size_t i = 0; i < wall_markers.size(); ++i) {
                    const int marker = wall_markers[i];
                    auto boundary_form = integrate_on_physical_boundary(
                        boundary_pressure_flux_integrand, marker);
                    vms_pspg_boundary_pressure_flux_form =
                        vms_pspg_boundary_pressure_flux_form.isValid()
                            ? (vms_pspg_boundary_pressure_flux_form +
                               boundary_form)
                            : boundary_form;
                    if (i > 0) {
                        marker_stream << "|";
                    }
                    marker_stream << marker;
                }
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_pressure_flux") +
                    " status=installed"
                    " form=wall_pressure_flux"
                    " pressure_gradient_form=" +
                    pspgPressureGradientFormName(
                        pspg_pressure_gradient_form) +
                    " scale=" + std::to_string(
                        static_cast<double>(
                            pspg_boundary_pressure_flux_scale)) +
                    " boundary_markers=" + marker_stream.str());
            }
        }

        if (pspg_boundary_tangential_pressure_gradient_scale > FE::Real{0.0}) {
            std::vector<int> wall_markers;
            auto append_marker = [&wall_markers](int marker) {
                if (marker < 0) {
                    return;
                }
                if (std::find(wall_markers.begin(), wall_markers.end(), marker) ==
                    wall_markers.end()) {
                    wall_markers.push_back(marker);
                }
            };
            for (const auto& bc : options_.velocity_dirichlet) {
                append_marker(bc.boundary_marker);
            }
            for (const auto& bc : options_.velocity_dirichlet_weak) {
                append_marker(bc.boundary_marker);
            }

            if (wall_markers.empty()) {
                FE_LOG_WARNING(
                    "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_tangential_pressure_gradient"
                    " status=skipped"
                    " reason=no_velocity_dirichlet_markers");
            } else {
                std::sort(wall_markers.begin(), wall_markers.end());
                const auto n_wall = FormExpr::normal();
                const auto grad_q_tangent =
                    grad(q) - dot(grad(q), n_wall) * n_wall;
                const auto pspg_pressure_gradient_tangent =
                    pspg_pressure_gradient_residual -
                    dot(pspg_pressure_gradient_residual, n_wall) * n_wall;
                const auto boundary_tangential_pressure_gradient_integrand =
                    FormExpr::constant(
                        pspg_boundary_tangential_pressure_gradient_scale) *
                    h() * tau_m *
                    dot(
                        grad_q_tangent,
                        pspg_pressure_gradient_tangent);
                std::ostringstream marker_stream;
                for (std::size_t i = 0; i < wall_markers.size(); ++i) {
                    const int marker = wall_markers[i];
                    auto boundary_form = integrate_on_physical_boundary(
                        boundary_tangential_pressure_gradient_integrand,
                        marker);
                    vms_pspg_boundary_tangential_pressure_gradient_form =
                        vms_pspg_boundary_tangential_pressure_gradient_form.isValid()
                            ? (vms_pspg_boundary_tangential_pressure_gradient_form +
                               boundary_form)
                            : boundary_form;
                    if (i > 0) {
                        marker_stream << "|";
                    }
                    marker_stream << marker;
                }
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_tangential_pressure_gradient") +
                    " status=installed"
                    " form=wall_tangential_pressure_gradient"
                    " pressure_gradient_form=" +
                    pspgPressureGradientFormName(
                        pspg_pressure_gradient_form) +
                    " scale=" + std::to_string(
                        static_cast<double>(
                            pspg_boundary_tangential_pressure_gradient_scale)) +
                    " boundary_markers=" + marker_stream.str());
            }
        }

        if (pspg_boundary_tangential_momentum_residual_scale > FE::Real{0.0}) {
            std::vector<int> wall_markers;
            auto append_marker = [&wall_markers](int marker) {
                if (marker < 0) {
                    return;
                }
                if (std::find(wall_markers.begin(), wall_markers.end(), marker) ==
                    wall_markers.end()) {
                    wall_markers.push_back(marker);
                }
            };
            for (const auto& bc : options_.velocity_dirichlet) {
                append_marker(bc.boundary_marker);
            }
            for (const auto& bc : options_.velocity_dirichlet_weak) {
                append_marker(bc.boundary_marker);
            }

            if (wall_markers.empty()) {
                FE_LOG_WARNING(
                    "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_tangential_momentum_residual"
                    " status=skipped"
                    " reason=no_velocity_dirichlet_markers");
            } else {
                std::sort(wall_markers.begin(), wall_markers.end());
                const auto n_wall = FormExpr::normal();
                const auto grad_q_tangent =
                    grad(q) - dot(grad(q), n_wall) * n_wall;
                const auto pspg_momentum_residual_tangent =
                    pspg_continuity_momentum_residual -
                    dot(pspg_continuity_momentum_residual, n_wall) * n_wall;
                const auto boundary_tangential_momentum_residual_integrand =
                    FormExpr::constant(
                        pspg_boundary_tangential_momentum_residual_scale) *
                    h() *
                    dot(
                        grad_q_tangent,
                        tau_m * pspg_momentum_residual_tangent);
                std::ostringstream marker_stream;
                for (std::size_t i = 0; i < wall_markers.size(); ++i) {
                    const int marker = wall_markers[i];
                    auto boundary_form = integrate_on_physical_boundary(
                        boundary_tangential_momentum_residual_integrand,
                        marker);
                    vms_pspg_boundary_tangential_momentum_residual_form =
                        vms_pspg_boundary_tangential_momentum_residual_form.isValid()
                            ? (vms_pspg_boundary_tangential_momentum_residual_form +
                               boundary_form)
                            : boundary_form;
                    if (i > 0) {
                        marker_stream << "|";
                    }
                    marker_stream << marker;
                }
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_boundary_tangential_momentum_residual") +
                    " status=installed"
                    " form=wall_tangential_momentum_residual"
                    " pressure_gradient_form=" +
                    pspgPressureGradientFormName(
                        pspg_pressure_gradient_form) +
                    " pressure_gradient_scale=" + std::to_string(
                        static_cast<double>(pspg_pressure_gradient_scale)) +
                    " scale=" + std::to_string(
                        static_cast<double>(
                            pspg_boundary_tangential_momentum_residual_scale)) +
                    " boundary_markers=" + marker_stream.str());
            }
        }
    }

    FormExpr momentum_form =
        integrateOnActiveVolume(active_momentum_integrand, active_volume_domain);
    FormExpr continuity_form =
        integrateOnActiveVolume(active_continuity_integrand, active_volume_domain);
    FormExpr vms_pspg_pressure_gradient_form;
    if (vms_pspg_pressure_gradient_integrand.isValid()) {
        vms_pspg_pressure_gradient_form =
            pspg_continuity_full_cell_support
                ? vms_pspg_pressure_gradient_integrand.dx()
                : integrateOnActiveVolume(
                      vms_pspg_pressure_gradient_integrand,
                      active_volume_domain);
    }
    if (pspg_continuity_full_cell_support &&
        vms_pspg_nonpressure_integrand.isValid()) {
        continuity_form = continuity_form + vms_pspg_nonpressure_integrand.dx();
    }
    FormExpr active_continuity_diagnostic_form = continuity_form;
    if (vms_pspg_pressure_gradient_form.isValid()) {
        active_continuity_diagnostic_form =
            active_continuity_diagnostic_form + vms_pspg_pressure_gradient_form;
    }
    if (vms_pspg_boundary_pressure_gradient_form.isValid()) {
        continuity_form =
            continuity_form + vms_pspg_boundary_pressure_gradient_form;
    }
    if (vms_pspg_boundary_pressure_flux_form.isValid()) {
        continuity_form =
            continuity_form + vms_pspg_boundary_pressure_flux_form;
    }
    if (vms_pspg_boundary_tangential_pressure_gradient_form.isValid()) {
        continuity_form =
            continuity_form +
            vms_pspg_boundary_tangential_pressure_gradient_form;
    }
    if (vms_pspg_boundary_tangential_momentum_residual_form.isValid()) {
        continuity_form =
            continuity_form +
            vms_pspg_boundary_tangential_momentum_residual_form;
    }
    FormExpr pressure_ghost_penalty_form;
    FormExpr free_surface_pressure_reference_probe_form;
    FormExpr free_surface_tangential_pressure_gradient_probe_form;
    FormExpr free_surface_conservative_pressure_form;
    FormExpr free_surface_conservative_surface_energy_form;
    FormExpr level_set_shape_tangent_form;
    const bool conservative_balance_diagnostic_requested =
        freeSurfaceConservativeBalanceDiagnosticEnabled();
    const bool conservative_balance_has_surface_energy =
        std::any_of(
            effective_free_surfaces.begin(),
            effective_free_surfaces.end(),
            [](const FreeSurfaceBoundary& bc) {
                return usesSurfaceStress(bc) &&
                       FE::forms::bc::isConstantScalarValue(
                           bc.surface_tension) &&
                       !FE::forms::bc::isZeroConstantScalarValue(
                           bc.surface_tension);
            });
    const bool conservative_balance_all_surface_stress =
        !effective_free_surfaces.empty() &&
        std::all_of(
            effective_free_surfaces.begin(),
            effective_free_surfaces.end(),
            [](const FreeSurfaceBoundary& bc) {
                return usesSurfaceStress(bc) &&
                       FE::forms::bc::isConstantScalarValue(
                           bc.surface_tension);
            });
    const bool conservative_balance_diagnostic_supported =
        conservative_balance_diagnostic_requested &&
        conservative_balance_has_surface_energy &&
        conservative_balance_all_surface_stress;
    if (conservative_balance_diagnostic_supported) {
        // The pressure part is the weak pressure work on the active liquid.
        // Prescribed exterior-pressure work is appended on each free surface.
        free_surface_conservative_pressure_form =
            integrateOnActiveVolume(pressure, active_volume_domain);
    } else if (conservative_balance_diagnostic_requested) {
        FE_LOG_WARNING(
            "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_free_surface_conservative_balance_operators"
            " status=skipped"
            " reason=requires_all_free_surfaces_surface_stress_and_nonzero_constant_surface_tension");
    }
    appendCutVolumeShapeTangentForm(
        level_set_shape_tangent_form,
        active_momentum_integrand,
        active_volume_domain);
    appendCutVolumeShapeTangentForm(
        level_set_shape_tangent_form,
        active_continuity_integrand,
        active_volume_domain);
    if (!pspg_continuity_full_cell_support &&
        vms_pspg_pressure_gradient_integrand.isValid()) {
        appendCutVolumeShapeTangentForm(
            level_set_shape_tangent_form,
            vms_pspg_pressure_gradient_integrand,
            active_volume_domain);
    }

    // ---------------------------------------------------------------------
    // Boundary conditions (installer + factories)
    // ---------------------------------------------------------------------

    if (!options_.coupled_outflow_rcr.empty() || !options_.coupled_outflow_rcrcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, options_.jit_policy);
    }

    FE::systems::BoundaryConditionManager bc_manager;

    // Weak velocity Dirichlet is applied directly to the Forms residual (affects both momentum and continuity).
    // Reserve the marker here so validate() catches conflicts with other BC types.
    bc_manager.install(options_.velocity_dirichlet_weak, Factories::reserveMarker);

    auto free_surface_kinematics_install =
        physicsInstallOptions(options_.jit_policy);
    for (const auto& effective_bc : effective_free_surfaces) {
        applyFreeSurfaceContactLineConstraints(system, effective_bc, ale_binding, dim);
        registerFreeSurfacePrescribedAngleGeometry(system, effective_bc);
        applyDynamicContactAngleResidual(
            momentum_form,
            system,
            effective_bc,
            u,
            v,
            mu,
            dim,
            conservative_balance_diagnostic_supported
                ? &free_surface_conservative_surface_energy_form
                : nullptr);
        if (effective_bc.implementation == FreeSurfaceImplementation::FittedALE) {
            bc_manager.add(std::make_unique<FE::forms::bc::ReservedBC>(effective_bc.boundary_marker));
        }
        applyFreeSurfaceBoundary(
            momentum_form,
            continuity_form,
            level_set_shape_tangent_form,
            effective_bc,
            system,
            u,
            p,
            v,
            q,
            mesh_velocity,
            mu,
            options_,
            options_.enable_ale,
            dim,
            pressureRowContributionDiagnosticEnabled()
                ? &free_surface_pressure_reference_probe_form
                : nullptr,
            pressureRowContributionDiagnosticEnabled()
                ? &free_surface_tangential_pressure_gradient_probe_form
                : nullptr,
            conservative_balance_diagnostic_supported
                ? &free_surface_conservative_pressure_form
                : nullptr,
            conservative_balance_diagnostic_supported
                ? &free_surface_conservative_surface_energy_form
                : nullptr);
        installFittedFreeSurfaceMeshKinematics(
            system,
            effective_bc,
            ale_binding,
            u,
            options_,
            free_surface_kinematics_install,
            u_id);
        applyFreeSurfaceCutCellStabilization(
            momentum_form,
            continuity_form,
            effective_bc,
            u,
            p,
            v,
            q,
            mu,
            options_.density,
            options_.stabilization_epsilon,
            dim,
            velocity_space_->polynomial_order(),
            pressure_space_->polynomial_order(),
            pressureRowContributionDiagnosticEnabled()
                ? &pressure_ghost_penalty_form
                : nullptr);
    }

    bc_manager.install(options_.traction_neumann, [&](const auto& bc) {
        return Factories::toTractionBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.traction_robin, [&](const auto& bc) {
        return Factories::toTractionRobinBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.pressure_outflow, [&](const auto& bc) {
        return Factories::toOutflowBC(
            bc, u, rho, generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.coupled_outflow_rcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(
            bc,
            system,
            u,
            rho,
            generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.coupled_outflow_rcrcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(
            bc,
            system,
            u,
            rho,
            generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.velocity_dirichlet,
                       [&](const auto& bc) {
        return Factories::toVelocityEssentialBC(bc, dim, options_.velocity_field_name);
    });

    bc_manager.applyAll(system, momentum_form, u, v, u_id);

    FE::systems::BoundaryConditionManager p_bc_manager;
    p_bc_manager.install(options_.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, options_.pressure_field_name); });
    p_bc_manager.applyAll(system, p_id);

    if (pending_small_cut_aggregation.has_value()) {
        // Mixed problems must aggregate both spaces: leaving the small-cut
        // pressures free while their conjugate velocities are slaved breaks
        // the local saddle point.  Registration after all essential
        // constraints is deliberate. SmallCutAggregationConstraint gathers
        // preconstrained component facts communicator-wide before installing
        // canonical lines, so even owner/non-owner boundary visibility cannot
        // turn a partial strong condition into an owner-wins aggregation line.
        system.addSystemConstraint(
            std::make_unique<FE::constraints::SmallCutAggregationConstraint>(
                u_id,
                pending_small_cut_aggregation->side,
                pending_small_cut_aggregation->interface_marker,
                std::vector<int>{},
                std::vector<FE::GlobalIndex>{},
                pending_small_cut_aggregation->guards));
        system.addSystemConstraint(
            std::make_unique<FE::constraints::SmallCutAggregationConstraint>(
                p_id,
                pending_small_cut_aggregation->side,
                pending_small_cut_aggregation->interface_marker,
                std::vector<int>{},
                std::vector<FE::GlobalIndex>{},
                pending_small_cut_aggregation->guards));
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: small-cut "
                        "aggregation enabled for unfitted free surface") +
            " marker=" +
            std::to_string(
                pending_small_cut_aggregation->interface_marker) +
            " registration_order=after_strong_constraints"
            " component_precedence=per_dof"
            " maximum_root_path_length=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_root_path_length) +
            " maximum_reference_extrapolation_distance=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_reference_extrapolation_distance) +
            " maximum_absolute_coefficient=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_absolute_coefficient) +
            " maximum_row_l1_norm=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_row_l1_norm) +
            " diagnostic=small_cut_aggregation_registration"
            " velocity_ghost_penalty=skipped_by_aggregation");
    }

    Factories::VelocityNitscheEnergyForms nitsche_energy_forms;
    std::vector<
        FE::forms::bc::
            GeneratedBoundaryNitscheTraceFormBinding>
        generated_nitsche_trace_bindings;
    Factories::applyVelocityNitscheBCs(
        momentum_form,
        continuity_form,
        options_,
        dim,
        u,
        p,
        v,
        q,
        mu,
        generated_active_boundary_for,
        nitsche_energy_diagnostic_requested
            ? &nitsche_energy_forms
            : nullptr,
        pending_small_cut_aggregation.has_value()
            ? &generated_nitsche_trace_bindings
            : nullptr);
    std::size_t expected_generated_nitsche_trace_bindings = 0u;
    for (const auto& bc : options_.velocity_dirichlet_weak) {
        const int physical_marker =
            FE::forms::bc::detail::boundaryMarkerOrThrow(
                bc,
                "IncompressibleNavierStokesVMSModule::registerOn "
                "generated-boundary Nitsche trace policy");
        const auto generated_marker =
            generated_active_boundary_for(physical_marker);
        if (!generated_marker.has_value()) {
            continue;
        }
        if (!pending_small_cut_aggregation.has_value()) {
            throw std::logic_error(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche route reached installation "
                "without its preflighted small-cut aggregation policy");
        }
        ++expected_generated_nitsche_trace_bindings;
        if (options_.viscosity_model != nullptr) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                    "certified generated-boundary Nitsche routes require "
                    "constant Newtonian viscosity");
        }
    }
    if (generated_nitsche_trace_bindings.size() !=
        expected_generated_nitsche_trace_bindings) {
        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "generated-boundary Nitsche form binding count differs "
            "from the certified production routes");
    }

    // Install the complete residual (momentum + continuity) via the unified
    // installFormulation() entry point.  It auto-detects the two-field mixed
    // structure and sets up per-block Jacobian kernels with optimal assembly.
    const auto residual = momentum_form + continuity_form;

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    if (!options_.viscosity_model) {
        install.recordDynamicViscosity(u_id, options_.viscosity);
    }
    if (pending_small_cut_aggregation.has_value()) {
        install.generated_boundary_nitsche_trace_requests.reserve(
            generated_nitsche_trace_bindings.size());
        for (auto& binding :
             generated_nitsche_trace_bindings) {
            install.generated_boundary_nitsche_trace_requests.push_back(
                FE::systems::
                    GeneratedBoundaryNitscheTraceInstallRequest{
                        .binding = std::move(binding),
                        .volume_interface_marker =
                            pending_small_cut_aggregation
                                ->interface_marker,
                        .minimum_symmetric_energy_ratio =
                            options_
                                .generated_boundary_nitsche_minimum_energy_ratio,
                    });
        }
    }
    appendUnfittedFreeSurfaceLevelSetTrialFields(
        effective_free_surfaces, system, install);
    appendUnfittedFreeSurfaceCurvatureTrialFields(
        effective_free_surfaces, system, install);
    ale_binding.configureInstallOptions(install);
    (void)FE::systems::installFormulation(
        system, options_.operator_tag, {u_id, p_id}, residual, install);

    if (nitsche_energy_diagnostic_requested &&
        !nitsche_energy_forms.symmetric_boundaries.empty()) {
        const auto bulk_viscous_form =
            integrateOnActiveVolume(viscous, active_volume_domain);
        const auto bulk_plus_consistency_form =
            bulk_viscous_form +
            nitsche_energy_forms.symmetric_consistency;
        const auto symmetric_operator_form =
            bulk_plus_consistency_form +
            nitsche_energy_forms.penalty;
        const auto energy_norm_form =
            bulk_viscous_form + nitsche_energy_forms.penalty;
        const auto install_velocity_diagnostic =
            [&](std::string_view operator_tag,
                const FormExpr& diagnostic_form) {
                auto diagnostic_install = install;
                diagnostic_install.source_component_tag =
                    std::string(operator_tag);
                diagnostic_install.extra_trial_fields.clear();
                diagnostic_install
                    .generated_boundary_nitsche_trace_requests
                    .clear();
                (void)FE::systems::installFormulation(
                    system,
                    std::string(operator_tag),
                    {u_id},
                    diagnostic_form,
                    diagnostic_install);
            };
        install_velocity_diagnostic(
            SymmetricNitscheEnergyDiagnosticOperators::bulk_viscous,
            bulk_viscous_form);
        install_velocity_diagnostic(
            SymmetricNitscheEnergyDiagnosticOperators::
                bulk_plus_consistency,
            bulk_plus_consistency_form);
        install_velocity_diagnostic(
            SymmetricNitscheEnergyDiagnosticOperators::symmetric_operator,
            symmetric_operator_form);
        install_velocity_diagnostic(
            SymmetricNitscheEnergyDiagnosticOperators::energy_norm,
            energy_norm_form);
        const auto generated_sharp_boundary_count =
            static_cast<std::size_t>(std::count_if(
                nitsche_energy_forms.symmetric_boundaries.begin(),
                nitsche_energy_forms.symmetric_boundaries.end(),
                [](const auto& route) {
                    return route.generated_active_boundary_marker
                        .has_value();
                }));
        std::string boundary_routes;
        for (const auto& route :
             nitsche_energy_forms.symmetric_boundaries) {
            if (!boundary_routes.empty()) {
                boundary_routes += ",";
            }
            boundary_routes +=
                std::to_string(route.physical_boundary_marker) +
                "->";
            boundary_routes +=
                route.generated_active_boundary_marker.has_value()
                    ? std::to_string(
                          *route.generated_active_boundary_marker)
                    : "full";
        }
        FE_LOG_INFO(
            std::string(
                "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_symmetric_nitsche_energy_operators") +
            " status=installed"
            " symmetric_boundary_count=" +
            std::to_string(
                nitsche_energy_forms.symmetric_boundaries.size()) +
            " generated_sharp_boundary_count=" +
            std::to_string(generated_sharp_boundary_count) +
            " full_boundary_count=" +
            std::to_string(
                nitsche_energy_forms.symmetric_boundaries.size() -
                generated_sharp_boundary_count) +
            " boundary_routes='" + boundary_routes + "'" +
            " bulk_operator='" +
            std::string(
                SymmetricNitscheEnergyDiagnosticOperators::bulk_viscous) +
            "' bulk_plus_consistency_operator='" +
            std::string(
                SymmetricNitscheEnergyDiagnosticOperators::
                    bulk_plus_consistency) +
            "' symmetric_operator='" +
            std::string(
                SymmetricNitscheEnergyDiagnosticOperators::
                    symmetric_operator) +
            "' energy_norm_operator='" +
            std::string(
                SymmetricNitscheEnergyDiagnosticOperators::energy_norm) +
            "' scope=velocity_viscous_block_only"
            " transient_pressure_convection_excluded=1");
    } else if (nitsche_energy_diagnostic_requested) {
        FE_LOG_WARNING(
            "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_symmetric_nitsche_energy_operators"
            " status=skipped"
            " reason=no_symmetric_weak_velocity_boundary");
    }

    if (vms_pspg_pressure_gradient_form.isValid()) {
        auto direct_pspg_install = install;
        direct_pspg_install
            .generated_boundary_nitsche_trace_requests
            .clear();
        direct_pspg_install.source_component_tag =
            "navier_stokes_vms_pspg_pressure_gradient";
        if (std::find(direct_pspg_install.extra_trial_fields.begin(),
                      direct_pspg_install.extra_trial_fields.end(),
                      u_id) == direct_pspg_install.extra_trial_fields.end()) {
            direct_pspg_install.extra_trial_fields.push_back(u_id);
        }
        (void)FE::systems::installFormulation(
            system,
            options_.operator_tag,
            {p_id},
            vms_pspg_pressure_gradient_form,
            direct_pspg_install);
    }

    if (conservative_balance_diagnostic_supported) {
        FE_THROW_IF(
            !free_surface_conservative_pressure_form.isValid() ||
                !free_surface_conservative_surface_energy_form.isValid(),
            FE::InvalidArgumentException,
            "IncompressibleNavierStokesVMSModule: requested free-surface conservative-balance diagnostics did not construct both pressure and surface-energy virtual-work forms");

        const auto conservative_balance_form =
            free_surface_conservative_pressure_form +
            free_surface_conservative_surface_energy_form;
        // This deliberately contains only the pressure/velocity adjoint pair.
        // With pressure = -p*div(v), the two blocks are
        //   -int p div(v)  and  -int q div(u),
        // hence the assembled matrix is symmetric with identically zero
        // velocity-velocity and pressure-pressure blocks.  That symmetry is
        // the backend-independent transpose action used by the matrix-free
        // LSQR representability diagnostic in NewtonSolver.
        const auto pressure_representability_pair_form =
            integrateOnActiveVolume(pressure, active_volume_domain) -
            integrateOnActiveVolume(q * div(u), active_volume_domain);
        const auto install_diagnostic =
            [&](std::string_view operator_tag,
                const FormExpr& diagnostic_form) {
                auto diagnostic_install = install;
                diagnostic_install
                    .generated_boundary_nitsche_trace_requests
                    .clear();
                diagnostic_install.source_component_tag =
                    std::string(operator_tag);
                (void)FE::systems::installFormulation(
                    system,
                    std::string(operator_tag),
                    {u_id},
                    diagnostic_form,
                    diagnostic_install);
            };
        install_diagnostic(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_virtual_work,
            free_surface_conservative_pressure_form);
        install_diagnostic(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                surface_energy_virtual_work,
            free_surface_conservative_surface_energy_form);
        install_diagnostic(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                conservative_balance,
            conservative_balance_form);

        {
            auto diagnostic_install = install;
            diagnostic_install
                .generated_boundary_nitsche_trace_requests
                .clear();
            diagnostic_install.source_component_tag = std::string(
                FreeSurfaceConservativeBalanceDiagnosticOperators::
                    pressure_representability_pair);
            // The pair is intentionally only a pressure/velocity operator.
            // Generated geometry is frozen external state for this
            // diagnostic; inherited level-set/curvature/ALE extra trial
            // fields would add blocks that are outside [0,G;G^T,0].
            diagnostic_install.extra_trial_fields.clear();
            (void)FE::systems::installFormulation(
                system,
                std::string(
                    FreeSurfaceConservativeBalanceDiagnosticOperators::
                        pressure_representability_pair),
                {u_id, p_id},
                pressure_representability_pair_form,
                diagnostic_install);
        }

        FE_LOG_INFO(
            std::string(
                "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_free_surface_conservative_balance_operators") +
            " status=installed" +
            " pressure_operator='" +
            std::string(
                FreeSurfaceConservativeBalanceDiagnosticOperators::
                    pressure_virtual_work) +
            "' surface_energy_operator='" +
            std::string(
                FreeSurfaceConservativeBalanceDiagnosticOperators::
                    surface_energy_virtual_work) +
            "' balance_operator='" +
            std::string(
                FreeSurfaceConservativeBalanceDiagnosticOperators::
                    conservative_balance) +
            "' pressure_representability_pair_operator='" +
            std::string(
                FreeSurfaceConservativeBalanceDiagnosticOperators::
                    pressure_representability_pair) +
            "' pressure_terms=active_volume_minus_p_div_v_plus_external_pressure_n_dot_v" +
            " surface_energy_terms=surface_area_variation_plus_young_wall_energy" +
            " excluded_terms=line_friction_and_wetted_wall_navier_dissipation" +
            " scope=pressure_and_surface_energy_first_variations_only" +
            " total_momentum_equilibrium_claimed=0" +
            " qualification=diagnostic_only_no_time_discrete_energy_claim");
    }

    if (pressureRowContributionDiagnosticEnabled()) {
        auto diagnostic_install = install;
        diagnostic_install
            .generated_boundary_nitsche_trace_requests
            .clear();
        if (std::find(diagnostic_install.extra_trial_fields.begin(),
                      diagnostic_install.extra_trial_fields.end(),
                      u_id) == diagnostic_install.extra_trial_fields.end()) {
            diagnostic_install.extra_trial_fields.push_back(u_id);
        }
        FE_LOG_INFO(
            "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pressure_row_contribution_operators"
            " installing=1"
            " ops=equations_diagnostic_ns_galerkin_continuity|equations_diagnostic_ns_active_continuity|equations_diagnostic_ns_vms_pspg|equations_diagnostic_ns_vms_pspg_pressure_gradient|equations_diagnostic_ns_vms_pspg_nonpressure|equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient|equations_diagnostic_ns_vms_pspg_boundary_pressure_flux|equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient|equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual|equations_diagnostic_ns_pressure_ghost_penalty|equations_diagnostic_ns_free_surface_pressure_reference_probe|equations_diagnostic_ns_free_surface_tangential_pressure_gradient_probe");
        (void)FE::systems::installFormulation(
            system,
            "equations_diagnostic_ns_galerkin_continuity",
            {p_id},
            integrateOnActiveVolume(
                galerkin_continuity_integrand,
                active_volume_domain),
            diagnostic_install);
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_active_continuity",
                {p_id},
                active_continuity_diagnostic_form,
                diagnostic_install);
        if (vms_pspg_continuity_integrand.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg",
                {p_id},
                pspg_continuity_full_cell_support
                    ? vms_pspg_continuity_integrand.dx()
                    : integrateOnActiveVolume(
                          vms_pspg_continuity_integrand,
                          active_volume_domain),
                diagnostic_install);
        }
        if (vms_pspg_pressure_gradient_integrand.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg_pressure_gradient",
                {p_id},
                pspg_continuity_full_cell_support
                    ? vms_pspg_pressure_gradient_integrand.dx()
                    : integrateOnActiveVolume(
                          vms_pspg_pressure_gradient_integrand,
                          active_volume_domain),
                diagnostic_install);
        }
        if (vms_pspg_nonpressure_integrand.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg_nonpressure",
                {p_id},
                pspg_continuity_full_cell_support
                    ? vms_pspg_nonpressure_integrand.dx()
                    : integrateOnActiveVolume(
                          vms_pspg_nonpressure_integrand,
                          active_volume_domain),
                diagnostic_install);
        }
        if (vms_pspg_boundary_pressure_gradient_form.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient",
                {p_id},
                vms_pspg_boundary_pressure_gradient_form,
                diagnostic_install);
        }
        if (vms_pspg_boundary_pressure_flux_form.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux",
                {p_id},
                vms_pspg_boundary_pressure_flux_form,
                diagnostic_install);
        }
        if (vms_pspg_boundary_tangential_pressure_gradient_form.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient",
                {p_id},
                vms_pspg_boundary_tangential_pressure_gradient_form,
                diagnostic_install);
        }
        if (vms_pspg_boundary_tangential_momentum_residual_form.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual",
                {p_id},
                vms_pspg_boundary_tangential_momentum_residual_form,
                diagnostic_install);
        }
        if (pressure_ghost_penalty_form.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_pressure_ghost_penalty",
                {p_id},
                pressure_ghost_penalty_form,
                diagnostic_install);
        }
        if (free_surface_pressure_reference_probe_form.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_free_surface_pressure_reference_probe",
                {p_id},
                free_surface_pressure_reference_probe_form,
                diagnostic_install);
        }
        if (free_surface_tangential_pressure_gradient_probe_form.isValid()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_free_surface_tangential_pressure_gradient_probe",
                {p_id},
                free_surface_tangential_pressure_gradient_probe_form,
                diagnostic_install);
        }
    }

    if (level_set_shape_tangent_form.isValid()) {
        std::array<FE::FieldId, 2> test_fields{{u_id, p_id}};
        std::vector<FE::FieldId> phi_trial_fields;
        for (const auto& bc : effective_free_surfaces) {
            if (!isUnfittedLevelSet(bc)) {
                continue;
            }
            const auto phi_id = resolveLevelSetFieldId(bc, system);
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                continue;
            }
            if (std::find(phi_trial_fields.begin(),
                          phi_trial_fields.end(),
                          phi_id) == phi_trial_fields.end()) {
                phi_trial_fields.push_back(phi_id);
            }
        }
        if (!phi_trial_fields.empty()) {
            (void)FE::systems::installMixedBilinear(
                system,
                options_.operator_tag,
                std::span<const FE::FieldId>(test_fields.data(),
                                             test_fields.size()),
                std::span<const FE::FieldId>(phi_trial_fields.data(),
                                             phi_trial_fields.size()),
                level_set_shape_tangent_form,
                install);
        }
    }

    for (auto& declaration : normal_declarations) {
        const auto marker = declaration.boundary_marker;
        const auto displacement_field =
            declaration.mesh_displacement_field;
        system.declareMeshNormalBoundaryConstraint(
            std::move(declaration));
        system.bindMeshNormalBoundaryConstraintConsumer(
            displacement_field,
            marker,
            options_.operator_tag,
            "Fitted free-surface mesh normal kinematic row on marker " +
                std::to_string(marker),
            "Fitted free-surface fluid normal kinematic row on marker " +
                std::to_string(marker));
        system.registerFittedALENormalOperatorStageMeasurement(
            displacement_field, marker);
    }

    effective_configuration_artifact_ = makeEffectiveConfigurationArtifact(
        options_, effective_free_surfaces, system, dim, enable_vms);
}

std::optional<EffectiveConfigurationArtifact>
IncompressibleNavierStokesVMSModule::effectiveConfigurationArtifact() const
{
    return effective_configuration_artifact_;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
