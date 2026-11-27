/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/CertificateEnvironment.h"

namespace svmp::Physics::formulations::navier_stokes::free_surface {
namespace {

namespace controls = FE::analysis::constraint_range_environment;
using FE::environment::EnvironmentNameRegistry;
using FE::environment::NameDeclaration;

bool reassembleEachSample() noexcept
{
    static const std::vector<std::string> names{
        controls::reassemble,
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY",
        "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY"};
    return environmentRegistry().firstEnabled(names);
}

std::string_view physicalReason(std::string_view numerical) noexcept
{
    static constexpr std::pair<std::string_view, std::string_view> names[] = {
        {"constant_field_action_nonfinite", "constant_pressure_action_nonfinite"},
        {"constant_field_action_outside_primary_trace", "constant_pressure_action_outside_velocity_trace"},
        {"constant_field_direction_zero_or_nonfinite", "constant_pressure_direction_zero_or_nonfinite"},
        {"constant_field_kkt_unavailable", "constant_pressure_kkt_unavailable"},
        {"constant_field_metrics_nonfinite", "constant_pressure_metrics_nonfinite"},
        {"constant_field_multiplier_nonfinite", "constant_pressure_multiplier_nonfinite"},
        {"constant_field_projection_nonfinite", "constant_pressure_projection_nonfinite"},
        {"constant_field_residual_nonfinite", "constant_pressure_residual_nonfinite"},
        {"invalid_primary_or_constraint_range", "invalid_velocity_or_pressure_range"},
        {"mixed_primary_constraint_fields_not_identifiable", "mixed_velocity_pressure_fields_not_identifiable"},
        {"nonzero_constraint_field_inhomogeneity", "nonzero_pressure_constraint_inhomogeneity"},
        {"field_constraints_do_not_preserve_constants", "pressure_constraints_do_not_preserve_constants"},
        {"constraint_correction_breakdown", "pressure_correction_breakdown"},
        {"constraint_correction_increased_residual", "pressure_correction_increased_residual"},
        {"constraint_correction_nonfinite", "pressure_correction_nonfinite"},
        {"constraint_correction_not_stationary", "pressure_correction_not_stationary"},
        {"constraint_pair_or_load_unavailable", "pressure_pair_or_load_unavailable"},
        {"constraint_solution_storage_unavailable", "pressure_solution_storage_unavailable"},
        {"constraint_space_unit_coefficients_do_not_represent_constant", "pressure_space_unit_coefficients_do_not_represent_constant"},
        {"solution_outside_constraint_subspace", "solution_outside_pressure_subspace"},
        {"unit_constraint_field_nonfinite", "unit_pressure_field_nonfinite"},
    };
    for (const auto& [key, name] : names) {
        if (key == numerical) {
            return name;
        }
    }
    return numerical;
}

constexpr FE::analysis::ConstraintRangeRecordVocabulary certificate_records{
    .sample_heading = "NewtonSolver: free-surface conservative balance diagnostic=free_surface_conservative_balance",
    .range_gate_heading = "NewtonSolver: accepted static pressure-representability distance gate diagnostic=free_surface_pressure_representability_distance_gate",
    .constant_gate_heading = "NewtonSolver: accepted static constant-pressure KKT distance gate diagnostic=free_surface_constant_pressure_kkt_distance_gate",
    .initializer_heading = "NewtonSolver: static compatible free-surface pressure initializer diagnostic=free_surface_static_compatible_pressure_initializer",
    .assembly_source = "free_surface_conservative_balance",
    .initializer_phase = "static_compatible_pressure_initializer",
    .range = "pressure_representability",
    .constant = "constant_pressure",
    .field = "pressure",
    .multiplier = "volume_multiplier",
    .normalization = "pressure_plus_physical_potential_norms",
    .constant_fit = "closed_form_one_dimensional_pressure_trace",
    .load = "prescribed_external_pressure_plus_surface_area_variation_plus_young_wall_energy_plus_gravitational_potential",
    .scope = " scope=pressure_and_physical_potential_first_variations_only contract=instantaneous_constrained_velocity_test_virtual_work",
    .exclusions = " excludes=line_friction_and_wetted_wall_navier_dissipation",
    .sample_claims = " total_momentum_equilibrium_claimed=0 discrete_energy_theorem_claimed=0",
    .initializer_scope = " force_projection_applied=0 balanced_force_evidence=0 total_momentum_equilibrium_claimed=0 production_capillary_operator_changed=0",
    .work_norms = {"pressure_virtual_work_norm", "surface_energy_virtual_work_norm",
                   "gravitational_potential_virtual_work_norm", "physical_potential_virtual_work_norm",
                   "pressure_representability_load_virtual_work_norm", "conservative_balance_norm"},
    .reason_name = physicalReason};

} // namespace

const EnvironmentNameRegistry& environmentRegistry()
{
    using enum FE::environment::NameKind;
    using enum FE::environment::ValueType;
    using enum FE::environment::RunStatus;
    // Stored in existing effective_configuration.json records as the
    // numerical service to which the formulation sends the resolved method.
    constexpr auto kCertificate = "FE/Analysis/ConstraintRangeCertificate";
    static constexpr NameDeclaration aliases[] = {
        {"SVMP_KEEP_DRY_CELL_POINTS", Performance, "FE/LevelSet/LevelSetGeometryService", Flag, "off", Reference,
         "keeps every quadrature point of full cells in the cut snapshot"},
        {"SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off", Reference,
         "second spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"},
        {"SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY", Performance, kCertificate, Flag,
         "off", Reference,
         "second spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY"},
        {"SVMP_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate, Real,
         "no gate", NonReference,
         "second spelling of SVMP_NS_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE"},
        {"SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off", Reference,
         "fourth spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"},
        {"SVMP_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate,
         Real, "no gate", NonReference,
         "second spelling of SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE"},
        {"SVMP_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER", NumericalOverride, kCertificate, Flag,
         "off", NonReference,
         "second spelling of SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER"},
        {"SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off",
         Reference,
         "logs the constraint-range certificate (representability and constant-field KKT distances); "
         "the first set spelling of the four decides"},
        {"SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY", Performance, kCertificate, Flag,
         "off", Reference, "recomputes the certificate's pair matrix and least-squares solve for every sample"},
        {"SVMP_NS_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate,
         Real, "no gate", NonReference,
         "accepts a converged step only if the constant-field KKT distance is at most this value"},
        {"SVMP_NS_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off", Reference,
         "third spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"},
        {"SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate,
         Real, "no gate", NonReference,
         "accepts a converged step only if the representability distance is at most this value"},
        {"SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER", NumericalOverride, kCertificate, Flag,
         "off", NonReference, "replaces the initial guess by the static-compatible least-squares correction"},
    };
    static const EnvironmentNameRegistry registry(EnvironmentNameRegistry::solver(), aliases);
    return registry;
}

bool keepFullCellPointsFromEnvironment()
{
    const char* value = environmentRegistry().value("SVMP_KEEP_DRY_CELL_POINTS");
    return value && value[0] != '\0' && value[0] != '0';
}

const std::vector<std::string>& conservativeBalanceDiagnosticSwitches()
{
    static const std::vector<std::string> names{
        controls::observe,
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        "SVMP_NS_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC",
        "SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC"};
    return names;
}

FE::analysis::NewtonAnalysisMethod newtonAnalysisMethodFromEnvironment()
{
    static const std::vector<std::string> range_names{
        controls::range_distance,
        "SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE",
        "SVMP_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE"};
    static const std::vector<std::string> kkt_names{
        controls::constant_kkt_distance,
        "SVMP_NS_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE",
        "SVMP_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE"};
    static const std::vector<std::string> initialize_names{
        controls::initialize,
        "SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER",
        "SVMP_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER"};

    auto method = FE::analysis::newtonAnalysisMethodFromEnvironment();
    const auto& names = environmentRegistry();
    method.constraint_range = {
        .observe = names.firstEnabled(conservativeBalanceDiagnosticSwitches()),
        .max_range_distance = names.firstReal(range_names),
        .max_constant_field_kkt_distance = names.firstReal(kkt_names),
        .initialize_constraint_field = names.firstEnabled(initialize_names),
        .reassemble_each_sample = reassembleEachSample,
        .record_vocabulary = &certificate_records};
    return method;
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
