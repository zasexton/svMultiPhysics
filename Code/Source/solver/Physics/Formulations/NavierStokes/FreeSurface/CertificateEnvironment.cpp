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
        .reassemble_each_sample = reassembleEachSample};
    return method;
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
