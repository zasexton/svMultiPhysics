#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceBoundarySchema.h"
#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidInterface.h"
#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/TemporalValues.h"
#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"
#include "Physics/Options/BoundaryDataReaders.h"
#include "Physics/Options/InputReaders.h"
#include "Physics/Options/NavierStokesRegistration.h"

#include "FE/Core/Logger.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Interfaces/MaterialInterfaceTransportVelocity.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellTopology.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace {

// The generic input readers and the boundary data readers (decision D52).
using namespace svmp::Physics::options;

svmp::FE::forms::GeometryTangentPath parse_geometry_tangent_path(std::string_view raw,
                                                                 std::string_view context)
{
  const auto path = lower_copy(trim_copy(std::string(raw)));
  if (path == "symbolic" || path == "symbolic_required" ||
      path == "symbolicrequired" || path == "required") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicRequired;
  }
  if (path == "ad" || path == "ad_reference" || path == "adreference" ||
      path == "reference_ad" || path == "referencead") {
    return svmp::FE::forms::GeometryTangentPath::ADReference;
  }
  if (path == "symbolic_ad_check" || path == "symbolic_with_ad_check" ||
      path == "symbolicadcheck" || path == "symbolicwithadcheck" ||
      path == "check" || path == "parity_check" || path == "paritycheck") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck;
  }
  if (path == "auto") {
    return svmp::FE::forms::GeometryTangentPath::Auto;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of 'symbolic', 'ad', 'symbolic_ad_check', or 'auto'.");
}

void apply_node_pressure_constraints(
    const svmp::Physics::EquationModuleInput& input,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  if (!input.node_pressure_constraints.has_value()) {
    return;
  }

  const auto& node_constraints = *input.node_pressure_constraints;
  options.node_pressure_constraints.id_type =
      parse_node_pressure_constraint_id_type(node_constraints.id_type);
  options.node_pressure_constraints.values =
      read_node_pressure_constraints_csv(node_constraints.values_file_path);
}

void apply_fluid_momentum_source_params(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  for (const auto key : {"Momentum_source_field_name", "MomentumSourceFieldName",
                         "Body_force_field_name", "BodyForceFieldName",
                         "Body_force_field", "BodyForceField"}) {
    if (const auto field_name = get_defined_string(params, key)) {
      options.body_force_field_name = *field_name;
      break;
    }
  }
  for (const auto key : {"Auto_register_momentum_source_field", "AutoRegisterMomentumSourceField",
                         "Auto_register_body_force_field", "AutoRegisterBodyForceField"}) {
    if (const auto auto_register = get_defined_bool(params, key)) {
      options.auto_register_body_force_field = *auto_register;
      break;
    }
  }
}

void apply_fluid_properties(const svmp::Physics::DomainInput& domain,
                            svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (const auto rho = get_defined_double(domain.params, "Density")) {
    options.density = static_cast<svmp::FE::Real>(*rho);
  } else if (const auto rho2 = get_defined_double(domain.params, "Fluid_density")) {
    options.density = static_cast<svmp::FE::Real>(*rho2);
  }

  if (const auto fx = get_defined_double(domain.params, "Force_x")) {
    options.body_force[0] = static_cast<svmp::FE::Real>(*fx);
  }
  if (const auto fy = get_defined_double(domain.params, "Force_y")) {
    options.body_force[1] = static_cast<svmp::FE::Real>(*fy);
  }
  if (const auto fz = get_defined_double(domain.params, "Force_z")) {
    options.body_force[2] = static_cast<svmp::FE::Real>(*fz);
  }
  apply_fluid_momentum_source_params(domain.params, options);

  if (const auto enabled = get_defined_bool(domain.params, "Hydrostatic_pressure_initialization")) {
    options.hydrostatic_pressure_initialization.enabled = *enabled;
  }
  if (const auto reference = get_defined_double(domain.params, "Hydrostatic_pressure_reference")) {
    options.hydrostatic_pressure_initialization.reference_pressure =
        static_cast<svmp::FE::Real>(*reference);
  }
  if (const auto reference_point =
          get_defined_string(domain.params, "Hydrostatic_pressure_reference_point")) {
    options.hydrostatic_pressure_initialization.reference_point =
        parse_real_vector3(*reference_point, "Hydrostatic_pressure_reference_point");
  }
  if (const auto field_name = get_defined_string(domain.params, "Hydrostatic_pressure_field_name")) {
    options.hydrostatic_pressure_initialization.field_name = *field_name;
  }

  const auto* model_param = find_param(domain.params, "Viscosity.model");
  if (!model_param || !model_param->defined || trim_copy(model_param->value).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] <Viscosity model=\"...\"> is required for the new solver Navier-Stokes module. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto model_raw = trim_copy(model_param->value);
  const auto model = lower_copy(model_raw);

  if (model == "constant") {
    const auto* mu_param = find_param(domain.params, "Viscosity.Value");
    const double mu = mu_param ? parse_double(mu_param->value, "Viscosity/Value") : 0.0;
    if (!(mu > 0.0)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Constant viscosity must be > 0 for the new solver Navier-Stokes module. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    options.viscosity = static_cast<svmp::FE::Real>(mu);
    options.viscosity_model.reset();
    return;
  }

  if (model == "carreau-yasuda" || model == "carreau_yasuda") {
    const auto* p_mu_inf = find_param(domain.params, "Viscosity.Limiting_high_shear_rate_viscosity");
    const auto* p_mu0 = find_param(domain.params, "Viscosity.Limiting_low_shear_rate_viscosity");
    const auto* p_lambda = find_param(domain.params, "Viscosity.Shear_rate_tensor_multiplier");
    const auto* p_n = find_param(domain.params, "Viscosity.Power_law_index");
    const auto* p_a = find_param(domain.params, "Viscosity.Shear_rate_tensor_exponent");

    const auto mu_inf = static_cast<svmp::FE::Real>(
        p_mu_inf ? parse_double(p_mu_inf->value, "Viscosity/Limiting_high_shear_rate_viscosity") : 0.0);
    const auto mu0 = static_cast<svmp::FE::Real>(
        p_mu0 ? parse_double(p_mu0->value, "Viscosity/Limiting_low_shear_rate_viscosity") : 0.0);
    const auto lambda = static_cast<svmp::FE::Real>(
        p_lambda ? parse_double(p_lambda->value, "Viscosity/Shear_rate_tensor_multiplier") : 0.0);
    const auto n =
        static_cast<svmp::FE::Real>(p_n ? parse_double(p_n->value, "Viscosity/Power_law_index") : 0.0);
    const auto a = static_cast<svmp::FE::Real>(
        p_a ? parse_double(p_a->value, "Viscosity/Shear_rate_tensor_exponent") : 0.0);

    try {
      options.viscosity_model = std::make_shared<svmp::Physics::materials::fluid::CarreauYasudaViscosity>(
          mu0, mu_inf, lambda, n, a);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("[svMultiPhysics::Physics] Invalid Carreau-Yasuda viscosity parameters for the new solver "
                      "Navier-Stokes module: ") +
          e.what() + ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (p_mu0 && p_mu0->defined) {
      options.viscosity = mu0;
    }
    return;
  }

  if (model == "cassons" || model == "casson") {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Cassons viscosity model is not supported by the new solver Navier-Stokes module yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Fluid viscosity model '" + model_raw +
      "' is not supported by the new solver Navier-Stokes module. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> "
      "to use the legacy solver.");
}

void apply_fluid_moving_domain_params(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  namespace ns = svmp::Physics::formulations::navier_stokes;

  constexpr std::array<std::string_view, 1> kAleKeys = {
      "Enable_ALE",
  };
  for (const auto key : kAleKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.enable_ale = *value;
    }
  }

  constexpr std::array<std::string_view, 1> kMeshVelocityFieldKeys = {
      "Mesh_velocity_field",
  };
  for (const auto key : kMeshVelocityFieldKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.mesh_velocity_field_name = *value;
    }
  }

  constexpr std::array<std::string_view, 1> kMeshVelocitySourceKeys = {
      "Mesh_velocity_source",
  };
  for (const auto key : kMeshVelocitySourceKeys) {
    if (const auto value = get_defined_string(params, key)) {
      const auto source = lower_copy(trim_copy(*value));
      if (source == "prescribed" || source == "prescribed_data" ||
          source == "data" || source == "mesh_motion_data") {
        options.mesh_velocity_source = ns::ALEMeshVelocitySource::PrescribedData;
      } else if (source == "coupled" || source == "coupled_displacement" ||
                 source == "derived" || source == "derived_from_displacement" ||
                 source == "monolithic") {
        options.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
      } else {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Mesh_velocity_source must be one of "
            "'prescribed_data' or 'coupled_displacement'.");
      }
    }
  }

  constexpr std::array<std::string_view, 2> kMeshDisplacementFieldKeys = {
      "Mesh_displacement_field",
      "MeshDisplacementField",
  };
  for (const auto key : kMeshDisplacementFieldKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.mesh_displacement_field_name = *value;
    }
  }

  constexpr std::array<std::string_view, 1> kAutoRegisterMeshDisplacementKeys = {
      "Auto_register_mesh_displacement_field",
  };
  for (const auto key : kAutoRegisterMeshDisplacementKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.auto_register_mesh_displacement_field = *value;
    }
  }

  constexpr std::array<std::string_view, 2> kMovingMeshTangentPathKeys = {
      "MovingMeshTangentPath",
      "Moving_mesh_tangent_path",
  };
  for (const auto key : kMovingMeshTangentPathKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.moving_mesh_tangent_path =
          parse_geometry_tangent_path(*value, key);
    }
  }
}

void apply_fluid_moving_domain_options(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::Physics::DomainInput& domain,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  apply_fluid_moving_domain_params(input.equation_params, options);
  apply_fluid_moving_domain_params(domain.params, options);
  if (trim_copy(options.mesh_velocity_field_name).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mesh_velocity_field must be non-empty when configuring Navier-Stokes ALE.");
  }
  if (options.mesh_velocity_source ==
          svmp::Physics::formulations::navier_stokes::ALEMeshVelocitySource::CoupledDisplacement &&
      trim_copy(options.mesh_displacement_field_name).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mesh_displacement_field must be non-empty when configuring coupled ALE.");
  }
}

void apply_free_surface_schema_options(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::
        IncompressibleNavierStokesVMSOptions& options)
{
  const auto contract = resolve_free_surface_schema_contract(params);
  options.input_configuration_schema_version = contract.version;
  options.explicit_legacy_configuration = contract.explicit_legacy;
}

std::string normalized_token(std::string_view raw)
{
  auto out = lower_copy(trim_copy(std::string(raw)));
  out.erase(std::remove_if(out.begin(), out.end(), [](unsigned char ch) {
              return ch == '_' || ch == '-' || std::isspace(ch);
            }),
            out.end());
  return out;
}

bool is_free_surface_type(std::string_view raw)
{
  return normalized_token(raw) == "freesurface";
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation
parse_free_surface_implementation(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "fitted" || token == "fittedale" || token == "ale") {
    return FreeSurfaceImplementation::FittedALE;
  }
  if (token == "unfitted" || token == "unfittedlevelset" ||
      token == "levelset" || token == "embeddedlevelset") {
    return FreeSurfaceImplementation::UnfittedLevelSet;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of FittedALE or UnfittedLevelSet.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceKinematicEnforcement
parse_free_surface_kinematic_enforcement(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceKinematicEnforcement;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none") {
    return FreeSurfaceKinematicEnforcement::None;
  }
  if (token == "penalty" || token == "nitsche") {
    // The fitted Penalty and Nitsche enforcements imposed the kinematic
    // relation on the fluid rows as well.
    throw std::runtime_error(
        "[svMultiPhysics::Physics] " + std::string(context) + "=" +
        std::string(raw) + " (" +
        (token == "penalty" ? "Penalty" : "Nitsche") +
        ") was retired (decision D47 Q9); use MeshNitsche.");
  }
  if (token == "meshnitsche") {
    return FreeSurfaceKinematicEnforcement::MeshNitsche;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be None or MeshNitsche.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomain
parse_free_surface_active_domain(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomain;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none" || token == "disabled" ||
      token == "off" || token == "inactive") {
    return FreeSurfaceActiveDomain::None;
  }
  if (token == "levelsetnegative" || token == "negative" ||
      token == "negativelevelset" || token == "phinegative") {
    return FreeSurfaceActiveDomain::LevelSetNegative;
  }
  if (token == "levelsetpositive" || token == "positive" ||
      token == "positivelevelset" || token == "phipositive") {
    return FreeSurfaceActiveDomain::LevelSetPositive;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of None, LevelSetNegative, or LevelSetPositive.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomainMethod
parse_free_surface_active_domain_method(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomainMethod;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "cutvolume" || token == "cutcellvolume" ||
      token == "exactcutvolume") {
    return FreeSurfaceActiveDomainMethod::CutVolume;
  }
  if (token == "smoothedindicator" || token == "smoothindicator" ||
      token == "indicator") {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] " + std::string(context) + "=" +
        std::string(raw) +
        " (SmoothedIndicator) was retired (decision D42); use CutVolume.");
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be CutVolume.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceSurfaceTensionForm
parse_free_surface_surface_tension_form(std::string_view raw,
                                       std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceSurfaceTensionForm;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "automatic" || token == "auto") {
    return FreeSurfaceSurfaceTensionForm::Automatic;
  }
  if (token == "curvaturetraction" || token == "explicitcurvature" ||
      token == "younglaplacetraction") {
    // The supplied- or raw-curvature traction was used by no accepted
    // benchmark; SurfaceStress is the variation of the discrete area.
    throw std::runtime_error(
        "[svMultiPhysics::Physics] " + std::string(context) + "=" +
        std::string(raw) +
        " (CurvatureTraction) was retired (decision D47 Q2); use "
        "SurfaceStress.");
  }
  if (token == "generatedcurvaturetraction" ||
      token == "generatedgeometrycurvaturetraction" ||
      token == "generatednormalcurvaturetraction") {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] " + std::string(context) + "=" +
        std::string(raw) +
        " (GeneratedCurvatureTraction) was retired (decision D42); use "
        "SurfaceStress.");
  }
  if (token == "kinematicareagradienttraction" ||
      token == "totalenergygradienttraction" ||
      token == "discreteenergygradienttraction") {
    // The projected-curvature traction was retired with its curvature
    // projection; SurfaceStress is the variation of the discrete area.
    throw std::runtime_error(
        "[svMultiPhysics::Physics] " + std::string(context) + "=" +
        std::string(raw) +
        " (KinematicAreaGradientTraction) was retired (decision D48); use "
        "SurfaceStress.");
  }
  if (token == "surfacestress" || token == "surfaceenergy" ||
      token == "laplacebeltrami" || token == "variational") {
    return FreeSurfaceSurfaceTensionForm::SurfaceStress;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be Automatic or SurfaceStress.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceSurfaceTensionSemiImplicit
parse_free_surface_surface_tension_semi_implicit(std::string_view raw,
                                                 std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::
      FreeSurfaceSurfaceTensionSemiImplicit;
  const auto token = normalized_token(raw);
  if (token == "none" || token == "off" || token == "disabled") {
    return FreeSurfaceSurfaceTensionSemiImplicit::None;
  }
  if (token == "normalincrement" || token == "laggednormalincrement") {
    return FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be None or NormalIncrement.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceNormalKinematicPolicy
parse_free_surface_normal_kinematic_policy(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceNormalKinematicPolicy;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "matchfluidnormalvelocity" ||
      token == "matchfluid" || token == "fluidnormalvelocity") {
    return FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
  }
  if (token == "none" || token == "disabled" || token == "off") {
    return FreeSurfaceNormalKinematicPolicy::None;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of MatchFluidNormalVelocity or None.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceTangentialMeshPolicy
parse_free_surface_tangential_mesh_policy(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceTangentialMeshPolicy;
  const auto token = normalized_token(raw);
  if (token == "free" || token == "freetangential" || token == "unconstrained") {
    return FreeSurfaceTangentialMeshPolicy::Free;
  }
  if (token.empty() || token == "smoothing" || token == "smoothingonly" ||
      token == "meshsmoothing" || token == "smooth") {
    return FreeSurfaceTangentialMeshPolicy::SmoothingOnly;
  }
  if (token == "prescribed" || token == "prescribedtangential" ||
      token == "prescribedvelocity") {
    return FreeSurfaceTangentialMeshPolicy::Prescribed;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of Free, SmoothingOnly, or Prescribed.");
}

enum class ParsedFreeSurfaceContactModel : std::uint8_t {
  None,
  Pinned,
  PrescribedAngle,
  DynamicRenE
};

ParsedFreeSurfaceContactModel
parse_free_surface_contact_line_model(std::string_view raw, std::string_view context)
{
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none" || token == "disabled" || token == "off") {
    return ParsedFreeSurfaceContactModel::None;
  }
  if (token == "pinned" || token == "fixed" || token == "fixedposition") {
    return ParsedFreeSurfaceContactModel::Pinned;
  }
  if (token == "prescribedcontactangle" || token == "contactangle" ||
      token == "prescribedangle") {
    return ParsedFreeSurfaceContactModel::PrescribedAngle;
  }
  if (token == "dynamiccontactangle" || token == "dynamicangle" ||
      token == "dynamicrene" || token == "rene") {
    return ParsedFreeSurfaceContactModel::DynamicRenE;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of None, Pinned, PrescribedAngle, or DynamicRenE.");
}

enum class ParsedFreeSurfaceWallSlipModel : std::uint8_t {
  None,
  Navier
};

ParsedFreeSurfaceWallSlipModel
parse_free_surface_wall_slip_model(std::string_view raw, std::string_view context)
{
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none" || token == "noslip" ||
      token == "disabled" || token == "off") {
    return ParsedFreeSurfaceWallSlipModel::None;
  }
  if (token == "navier" || token == "navierslip" || token == "slip") {
    return ParsedFreeSurfaceWallSlipModel::Navier;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of None or Navier.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfacePressureStabilizationPolicy
parse_free_surface_pressure_stabilization_policy(std::string_view raw,
                                                 std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfacePressureStabilizationPolicy;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "enabled" || token == "enable" ||
      token == "on" || token == "true") {
    return FreeSurfacePressureStabilizationPolicy::Enabled;
  }
  if (token == "incremental" || token == "pressureincrement" ||
      token == "incrementalpressure" || token == "incrementalenabled") {
    return FreeSurfacePressureStabilizationPolicy::Incremental;
  }
  if (token == "disabled" || token == "disable" ||
      token == "off" || token == "false") {
    return FreeSurfacePressureStabilizationPolicy::Disabled;
  }
  if (token == "disabledforrefreshedfrozenhighorder" ||
      token == "disableforrefreshedfrozenhighorder" ||
      token == "refreshedfrozenhighorderdisabled" ||
      token == "highorderrefreshedfrozendisabled") {
    return FreeSurfacePressureStabilizationPolicy::
        DisabledForRefreshedFrozenHighOrder;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of Enabled, Incremental, Disabled, or DisabledForRefreshedFrozenHighOrder.");
}

void reject_duplicate_aliases(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view property)
{
  if (defined_parameter_count(params, keys) > 1u) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration defines multiple aliases for " +
        std::string(property) + ". Specify exactly one spelling.");
  }
}

void apply_fluid_rotating_frame_coriolis(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::Physics::DomainInput& domain,
    int dim,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using Options = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  const auto enabled = first_defined_bool(
      input.equation_params,
      {"Rotating_frame_coriolis", "RotatingFrameCoriolis",
       "Enable_rotating_frame_coriolis", "EnableRotatingFrameCoriolis"})
      .value_or(first_defined_bool(
          domain.params,
          {"Rotating_frame_coriolis", "RotatingFrameCoriolis",
           "Enable_rotating_frame_coriolis", "EnableRotatingFrameCoriolis"})
          .value_or(false));

  const auto equation_omega = first_defined_string(
      input.equation_params,
      {"Rotating_frame_angular_velocity", "RotatingFrameAngularVelocity",
       "Angular_velocity", "AngularVelocity"});
  const auto domain_omega = first_defined_string(
      domain.params,
      {"Rotating_frame_angular_velocity", "RotatingFrameAngularVelocity",
       "Angular_velocity", "AngularVelocity"});
  const auto equation_omega_file = first_defined_string(
      input.equation_params,
      {"Rotating_frame_angular_velocity_temporal_values_file_path",
       "RotatingFrameAngularVelocityTemporalValuesFilePath",
       "Angular_velocity_temporal_values_file_path",
       "AngularVelocityTemporalValuesFilePath"});
  const auto domain_omega_file = first_defined_string(
      domain.params,
      {"Rotating_frame_angular_velocity_temporal_values_file_path",
       "RotatingFrameAngularVelocityTemporalValuesFilePath",
       "Angular_velocity_temporal_values_file_path",
       "AngularVelocityTemporalValuesFilePath"});

  if (equation_omega && equation_omega_file) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Specify either Rotating_frame_angular_velocity or "
        "Rotating_frame_angular_velocity_temporal_values_file_path, not both.");
  }
  if (domain_omega && domain_omega_file) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Specify either Rotating_frame_angular_velocity or "
        "Rotating_frame_angular_velocity_temporal_values_file_path in a domain, not both.");
  }

  const bool equation_has_omega = equation_omega || equation_omega_file;
  const auto omega = equation_omega ? equation_omega : (!equation_has_omega ? domain_omega : std::nullopt);
  const auto omega_file =
      equation_omega_file ? equation_omega_file : (!equation_has_omega ? domain_omega_file : std::nullopt);

  if (!enabled && !omega && !omega_file) {
    return;
  }
  if (dim != 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Rotating-frame Coriolis forcing requires a 3D Navier-Stokes mesh.");
  }

  options.rotating_frame_coriolis_enabled = true;
  if (omega) {
    const auto parsed = parse_real_vector3(*omega, "Rotating_frame_angular_velocity");
    for (std::size_t d = 0; d < parsed.size(); ++d) {
      options.rotating_frame_angular_velocity[d] = Options::ScalarValue{parsed[d]};
    }
  }
  if (omega_file) {
    auto temporal = svmp::Physics::readTemporalValuesFile(
        *omega_file, /*num_components=*/3, svmp::Physics::TemporalEndBehavior::Clamp);
    for (int d = 0; d < 3; ++d) {
      const int comp = d;
      options.rotating_frame_angular_velocity[static_cast<std::size_t>(d)] =
          svmp::FE::forms::TimeScalarCoefficient(
              [temporal, comp](svmp::FE::Real /*x*/,
                               svmp::FE::Real /*y*/,
                               svmp::FE::Real /*z*/,
                               svmp::FE::Real t) -> svmp::FE::Real {
                return temporal->interpolate(t, comp);
              });
    }
  }
}

void apply_fluid_momentum_source_spacetime_file(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::Physics::DomainInput& domain,
    int dim,
    MarkerCommunicator comm,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  const auto path = first_defined_string(
      input.equation_params,
      {"Momentum_source_temporal_and_spatial_values_file_path",
       "MomentumSourceTemporalAndSpatialValuesFilePath",
       "Body_force_temporal_and_spatial_values_file_path",
       "BodyForceTemporalAndSpatialValuesFilePath"});
  const auto domain_path = first_defined_string(
      domain.params,
      {"Momentum_source_temporal_and_spatial_values_file_path",
       "MomentumSourceTemporalAndSpatialValuesFilePath",
       "Body_force_temporal_and_spatial_values_file_path",
       "BodyForceTemporalAndSpatialValuesFilePath"});

  const auto source_path = path ? path : domain_path;
  if (!source_path || source_path->empty()) {
    return;
  }
  if (!input.mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Time/spatial Navier-Stokes momentum source requires a mesh.");
  }

  auto data = read_temporal_and_spatial_values_file(
      *input.mesh,
      /*boundary_marker=*/-1,
      *source_path,
      comm);
  options.has_body_force_spacetime = true;
  for (int d = 0; d < dim; ++d) {
    if (d < data->dof) {
      const int comp = d;
      options.body_force_spacetime[static_cast<std::size_t>(d)] =
          svmp::FE::forms::TimeScalarCoefficient(
              [data, comp](svmp::FE::Real x,
                           svmp::FE::Real y,
                           svmp::FE::Real z,
                           svmp::FE::Real t) -> svmp::FE::Real {
                const std::array<svmp::FE::Real, 3> p{x, y, z};
                return data->interpolateSpatial(p, t, comp);
              });
    } else {
      options.body_force_spacetime[static_cast<std::size_t>(d)] =
          svmp::Physics::formulations::navier_stokes::
              IncompressibleNavierStokesVMSOptions::ScalarValue{0.0};
    }
  }
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation
free_surface_implementation_from_params(const svmp::Physics::ParameterMap& params)
{
  const auto raw = first_defined_string(
      params,
      {"Implementation"});
  return parse_free_surface_implementation(raw.value_or("FittedALE"), "Free-surface Implementation");
}

void append_free_surface_contact_line(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary& fs)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;
  using ContactLine = IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

  constexpr std::string_view contact_keys[]{
      "Contact_line_model",
      "Contact_angle_radians",
      "Contact_angle_degrees",
      "Contact_line_wall_markers",
      "Contact_line_wall_marker",
      "Contact_line_wall_face",
      "Contact_line_marker",
      "Contact_line_wall_normal",
      "Contact_line_mobility",
      "Wall_slip_model",
      "Wall_slip_length",
      "Contact_line_law_enforcement"};

  const auto is_known_contact_key = [&](std::string_view key) {
    return std::find(std::begin(contact_keys), std::end(contact_keys), key) !=
           std::end(contact_keys);
  };
  const auto looks_like_contact_key = [](std::string_view key) {
    const auto token = normalized_token(key);
    const auto starts_with = [&](std::string_view prefix) {
      return token.rfind(prefix, 0) == 0;
    };
    return starts_with("contact") || starts_with("freesurfacecontact") ||
           starts_with("prescribedcontact") ||
           starts_with("wallboundarymarker") ||
           starts_with("wallboundaryface") || starts_with("wallnormal") ||
           starts_with("wallslip") || token == "mobility" ||
           token == "sliplength";
  };
  for (const auto& [key, value] : params) {
    if (value.defined && looks_like_contact_key(key) &&
        !is_known_contact_key(key)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Unknown free-surface contact key '" +
          key + "'. Contact configuration is fail-closed.");
    }
  }

  const bool contact_requested = std::any_of(
      std::begin(contact_keys), std::end(contact_keys), [&](std::string_view key) {
        const auto* value = find_param(params, key);
        return value != nullptr && value->defined;
      });
  if (!contact_requested) {
    return;
  }

  reject_duplicate_aliases(
      params,
      {"Contact_line_model"},
      "contact-line model");
  reject_duplicate_aliases(
      params,
      {"Contact_angle_radians"},
      "contact angle in radians");
  reject_duplicate_aliases(
      params,
      {"Contact_angle_degrees"},
      "contact angle in degrees");
  reject_duplicate_aliases(
      params,
      {"Contact_line_wall_markers",
       "Contact_line_wall_marker"},
      "contact wall marker(s)");
  reject_duplicate_aliases(
      params,
      {"Contact_line_wall_face"},
      "contact wall face name(s)");
  reject_duplicate_aliases(
      params,
      {"Contact_line_marker"},
      "contact-line marker");
  reject_duplicate_aliases(
      params,
      {"Contact_line_wall_normal"},
      "contact wall normal(s)");
  reject_duplicate_aliases(
      params,
      {"Contact_line_mobility"},
      "contact-line mobility");
  reject_duplicate_aliases(
      params,
      {"Wall_slip_model"},
      "contact wall-slip model");
  reject_duplicate_aliases(
      params,
      {"Wall_slip_length"},
      "contact wall-slip length");

  if (any_parameter_defined(
          params,
          {"Contact_line_wall_face"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Contact wall face names are not supported by the typed free-surface contact configuration; provide explicit wall marker ids.");
  }

  const auto model = first_defined_string(
      params,
      {"Contact_line_model"});
  const auto angle_radians = first_defined_double(
      params,
      {"Contact_angle_radians"});
  const auto angle_degrees = first_defined_double(
      params,
      {"Contact_angle_degrees"});

  if (!model.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration requires an explicit Contact_line_model (None, Pinned, PrescribedAngle, or DynamicRenE).");
  }
  if (angle_radians.has_value() && angle_degrees.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration must specify the contact angle in either radians or degrees, not both.");
  }

  const auto parsed_model = parse_free_surface_contact_line_model(
      *model, "Free-surface Contact_line_model");

  const bool has_angle = angle_radians.has_value() || angle_degrees.has_value();
  const bool has_wall_marker = any_parameter_defined(
      params,
      {"Contact_line_wall_markers",
       "Contact_line_wall_marker"});
  const bool has_wall_normal = any_parameter_defined(
      params,
      {"Contact_line_wall_normal"});
  const bool has_mobility = any_parameter_defined(
      params,
      {"Contact_line_mobility"});
  const bool has_slip_model = any_parameter_defined(
      params,
      {"Wall_slip_model"});
  const bool has_slip_length = any_parameter_defined(
      params,
      {"Wall_slip_length"});
  const bool has_contact_marker = any_parameter_defined(
      params, {"Contact_line_marker"});
  const auto slip_model = first_defined_string(
      params,
      {"Wall_slip_model"});

  if (parsed_model == ParsedFreeSurfaceContactModel::None) {
    if (has_angle || has_wall_marker || has_wall_normal || has_mobility ||
        has_slip_model || has_slip_length ||
        has_contact_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=None does not accept contact-angle, wall, mobility, slip, or contact-marker parameters.");
    }
    fs.contact_lines.push_back(ContactLine{
        .configuration = ContactLine::None{},
    });
    return;
  }
  if (parsed_model == ParsedFreeSurfaceContactModel::Pinned) {
    if (!has_wall_marker && !has_contact_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=Pinned requires Contact_line_marker or a contact wall marker.");
    }
    if (has_angle || has_wall_normal || has_mobility ||
        has_slip_model || has_slip_length) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=Pinned accepts only its contact-line or wall marker.");
    }
  } else {
    if (!has_angle || !has_wall_marker || !has_wall_normal) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] PrescribedAngle and DynamicRenE require an explicit angle, wall marker(s), and wall normal(s).");
    }
    if (parsed_model == ParsedFreeSurfaceContactModel::PrescribedAngle) {
      if (has_mobility) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle does not accept contact-line mobility.");
      }
      if (has_slip_model != has_slip_length) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle Navier slip requires both Wall_slip_model and Wall_slip_length, or neither.");
      }
      if (has_slip_model &&
          parse_free_surface_wall_slip_model(
              *slip_model, "Free-surface Wall_slip_model") !=
              ParsedFreeSurfaceWallSlipModel::Navier) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle wall slip requires Wall_slip_model=Navier.");
      }
    } else if (!has_mobility || !has_slip_model || !has_slip_length) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] DynamicRenE requires explicit mobility, wall-slip model, and wall-slip length parameters.");
    }
  }

  const auto wall_markers = first_defined_int_entries(
      params,
      {"Contact_line_wall_markers",
       "Contact_line_wall_marker"},
      "Free-surface Contact_line_wall_marker");
  const int contact_marker = first_defined_int(
      params, {"Contact_line_marker"})
                                 .value_or(-1);

  IncompressibleNavierStokesVMSOptions::ScalarValue contact_angle{
      svmp::FE::Real{1.57079632679489661923}};
  if (angle_radians.has_value()) {
    contact_angle = static_cast<svmp::FE::Real>(*angle_radians);
  } else if (angle_degrees.has_value()) {
    constexpr double pi = 3.14159265358979323846;
    contact_angle =
        static_cast<svmp::FE::Real>((*angle_degrees) * pi / 180.0);
  }

  const auto wall_normals = first_defined_vector3_entries(
      params,
      {"Contact_line_wall_normal"},
      "Free-surface Contact_line_wall_normal");

  IncompressibleNavierStokesVMSOptions::ScalarValue mobility{
      svmp::FE::Real{0.0}};
  bool strong_contact_law = false;
  if (const auto enforcement = first_defined_string(
          params, {"Contact_line_law_enforcement"})) {
    const auto token = normalized_token(*enforcement);
    if (token == "strong") {
      strong_contact_law = true;
    } else if (token != "weak") {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_law_enforcement must be Weak or Strong.");
    }
  }
  if (const auto parsed_mobility = first_defined_double(
          params,
          {"Contact_line_mobility"})) {
    mobility = static_cast<svmp::FE::Real>(*parsed_mobility);
  }

  if (parsed_model == ParsedFreeSurfaceContactModel::DynamicRenE &&
      (!slip_model.has_value() ||
       parse_free_surface_wall_slip_model(
           *slip_model, "Free-surface Wall_slip_model") !=
           ParsedFreeSurfaceWallSlipModel::Navier)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] DynamicRenE requires Wall_slip_model=Navier.");
  }

  IncompressibleNavierStokesVMSOptions::ScalarValue slip_length{
      svmp::FE::Real{0.0}};
  if (const auto parsed_slip_length = first_defined_double(
          params,
          {"Wall_slip_length"})) {
    slip_length = static_cast<svmp::FE::Real>(*parsed_slip_length);
  }

  const std::size_t contact_line_count =
      wall_markers.has_value() ? wall_markers->size() : 1u;
  if (contact_line_count > 1u && has_contact_marker) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] A single Contact_line_marker cannot be shared by multiple contact walls; omit it to use generated markers.");
  }
  if (wall_normals.has_value() && wall_normals->size() != 1u &&
      wall_normals->size() != contact_line_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface Contact_line_wall_normals must contain either one vector or one vector per wall marker.");
  }
  for (std::size_t i = 0; i < contact_line_count; ++i) {
    const int wall_marker = wall_markers.has_value()
                                ? (*wall_markers)[i]
                                : -1;
    std::array<IncompressibleNavierStokesVMSOptions::ScalarValue, 3>
        wall_normal{svmp::FE::Real{0.0},
                    svmp::FE::Real{0.0},
                    svmp::FE::Real{0.0}};
    if (wall_normals.has_value()) {
      const auto& normal =
          wall_normals->size() == 1u ? (*wall_normals)[0] : (*wall_normals)[i];
      wall_normal = {normal[0], normal[1], normal[2]};
    }

    if (strong_contact_law &&
        parsed_model != ParsedFreeSurfaceContactModel::DynamicRenE) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_law_enforcement=Strong requires "
          "Contact_line_model=DynamicRenE.");
    }
    switch (parsed_model) {
    case ParsedFreeSurfaceContactModel::None:
      throw std::logic_error("None contact configuration expanded unexpectedly");
    case ParsedFreeSurfaceContactModel::Pinned:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::Pinned{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = contact_marker,
          },
      });
      break;
    case ParsedFreeSurfaceContactModel::PrescribedAngle:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::PrescribedAngle{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = contact_marker,
              .contact_angle_radians = contact_angle,
              .wall_normal = wall_normal,
              .slip_length = has_slip_length
                  ? std::optional<
                        IncompressibleNavierStokesVMSOptions::ScalarValue>{
                        slip_length}
                  : std::nullopt,
          },
      });
      break;
    case ParsedFreeSurfaceContactModel::DynamicRenE:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::DynamicRenE{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = contact_marker,
              .equilibrium_contact_angle_radians = contact_angle,
              .wall_normal = wall_normal,
              .mobility = mobility,
              .slip_length = slip_length,
              .strong_law = strong_contact_law,
          },
      });
      break;
    }
  }
}

void append_free_surface_bc(
    const svmp::Physics::BoundaryConditionInput& bc,
    bool is_steady,
    bool has_temp_spat,
    bool has_other_files,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation;
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceKinematicEnforcement;
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceTangentialMeshPolicy;
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (!is_steady) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "' currently supports only steady scalar parameters in the new solver Navier-Stokes input translator.");
  }
  if (has_temp_spat || has_other_files) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "' does not support spatial/temporal files in the new solver Navier-Stokes input translator.");
  }

  IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary fs{};
  // Kept for the provenance of the resolved values (effective-configuration
  // artifact, decision D47).
  fs.input = svmp::Physics::formulations::navier_stokes::optionInputFromParameters(bc.params);
  const bool explicit_legacy_configuration =
      options.input_configuration_schema_version == 1 &&
      options.explicit_legacy_configuration;
  fs.implementation = free_surface_implementation_from_params(bc.params);
  if (const auto active_domain = first_defined_string(
          bc.params,
          {"Active_domain"})) {
    fs.active_domain =
        parse_free_surface_active_domain(*active_domain, "Free-surface Active_domain");
  }
  if (const auto active_domain_method = first_defined_string(
          bc.params,
          {"Active_domain_method"})) {
    fs.active_domain_method = parse_free_surface_active_domain_method(
        *active_domain_method, "Free-surface Active_domain_method");
  }
  if (const auto active_domain_smoothing_width = first_defined_double(
          bc.params,
          {"Active_domain_smoothing_width"})) {
    if (*active_domain_smoothing_width < 0.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free_surface Active_domain_smoothing_width must be nonnegative.");
    }
    fs.active_domain_smoothing_width =
        static_cast<svmp::FE::Real>(*active_domain_smoothing_width);
  }
  if (const auto allow_full_domain = first_defined_bool(
          bc.params,
          {"Allow_full_domain_unfitted_free_surface"})) {
    fs.allow_full_domain_unfitted_free_surface = *allow_full_domain;
  }
  if (fs.implementation == FreeSurfaceImplementation::FittedALE) {
    fs.boundary_marker = bc.boundary_marker;
  } else {
    const auto marker = first_defined_int(bc.params, {"Interface_marker"});
    if (marker.has_value()) {
      fs.interface_marker = *marker;
    }
    if (const auto field_name = first_defined_string(
            bc.params,
            {"Level_set_field_name"})) {
      fs.level_set_field_name = *field_name;
    }
    if (const auto domain_id = first_defined_string(
            bc.params,
            {"Generated_interface_domain_id"})) {
      fs.generated_interface_domain_id = *domain_id;
    }
    if (const auto geometry = first_defined_string(
            bc.params,
            {"Generated_interface_geometry"})) {
      fs.generated_interface_geometry = *geometry;
    }
    if (const auto tangent_policy = first_defined_string(
            bc.params,
            {"Geometry_tangent_policy"})) {
      fs.geometry_tangent_policy = *tangent_policy;
    }
    if (const auto isovalue = first_defined_double(
            bc.params,
            {"Level_set_isovalue"})) {
      fs.level_set_isovalue = static_cast<svmp::FE::Real>(*isovalue);
    }
  }

  if (const auto pressure = first_defined_double(
          bc.params,
          {"External_pressure"})) {
    fs.external_pressure = IncompressibleNavierStokesVMSOptions::ScalarValue{
        static_cast<svmp::FE::Real>(*pressure)};
  }
  if (const auto surface_tension = first_defined_double(
          bc.params,
          {"Surface_tension"})) {
    fs.surface_tension = IncompressibleNavierStokesVMSOptions::ScalarValue{
        static_cast<svmp::FE::Real>(*surface_tension)};
  }
  if (const auto surface_tension_form = first_defined_string(
          bc.params,
          {"Surface_tension_form"})) {
    fs.surface_tension_form = parse_free_surface_surface_tension_form(
        *surface_tension_form, "Free-surface Surface_tension_form");
  }
  if (const auto semi_implicit = first_defined_string(
          bc.params,
          {"Surface_tension_semi_implicit"})) {
    fs.surface_tension_semi_implicit =
        parse_free_surface_surface_tension_semi_implicit(
            *semi_implicit, "Free-surface Surface_tension_semi_implicit");
  }
  // Decision D47 Q2 retired the CurvatureTraction route with its supplied
  // curvature and its raw level-set curvature, and the opt-in of fitted
  // SurfaceStress.  The keys stay known to Parameters, which reports the
  // restated no-op values Use_level_set_curvature=false and
  // Allow_fitted_surface_stress=true as input deprecations; every other
  // value fails here.
  if (const auto allow_fitted_surface_stress = first_defined_bool(
          bc.params,
          {"Allow_fitted_surface_stress"});
      allow_fitted_surface_stress.has_value() &&
      !*allow_fitted_surface_stress) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Allow_fitted_surface_stress=false was retired (decision D47 "
        "Q2); fitted SurfaceStress needs no opt-in.  Remove the key and use "
        "Surface_tension_form=SurfaceStress.");
  }
  if (first_defined_string(bc.params, {"Curvature"}).has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Curvature (the supplied curvature of the CurvatureTraction "
        "route) was retired (decision D47 Q2); remove it and use "
        "Surface_tension_form=SurfaceStress.");
  }
  if (first_defined_bool(bc.params, {"Use_level_set_curvature"})
          .value_or(false)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Use_level_set_curvature=true (the raw level-set curvature of "
        "the CurvatureTraction route) was retired (decision D47 Q2); use "
        "Surface_tension_form=SurfaceStress.");
  }
  // Decision D48 retired KinematicAreaGradientTraction, the only reader of
  // the projected curvature field of a boundary.  Every spelling of the key
  // fails whenever it is given, whatever its value; the keys stay known to
  // Parameters.
  for (const std::string_view key :
       {"Curvature_field", "CurvatureField", "Curvature_field_name",
        "CurvatureFieldName", "Projected_curvature_field",
        "ProjectedCurvatureField", "Free_surface_curvature_field",
        "FreeSurfaceCurvatureField"}) {
    if (const auto* parameter = find_param(bc.params, key);
        parameter != nullptr && parameter->defined) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free_surface BC '" + bc.name + "': " +
          std::string(key) +
          " (the projected curvature field of the "
          "KinematicAreaGradientTraction route) was retired (decision D48); "
          "remove it and use Surface_tension_form=SurfaceStress.");
    }
  }
  // Fitted CurvatureTraction with pointwise current-geometry curvature was
  // withdrawn by D5 (identically zero on affine faces) and retired by D42.
  if (first_defined_bool(
          bc.params,
          {"Use_current_geometry_curvature"})
          .value_or(false)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Use_current_geometry_curvature=true (pointwise current-geometry "
        "curvature of the fitted CurvatureTraction) was retired (decision "
        "D42); it is identically zero on affine faces.  Use "
        "Surface_tension_form=SurfaceStress, the default.");
  }

  if (const auto normal_policy = first_defined_string(
          bc.params,
          {"Normal_kinematic_policy"})) {
    fs.normal_kinematic_policy = parse_free_surface_normal_kinematic_policy(
        *normal_policy, "Free-surface Normal_kinematic_policy");
  }
  const auto tangential_policy = first_defined_string(
          bc.params,
          {"Tangential_mesh_policy"});
  if (tangential_policy.has_value()) {
    fs.tangential_mesh_policy = parse_free_surface_tangential_mesh_policy(
        *tangential_policy, "Free-surface Tangential_mesh_policy");
  }
  const auto tangential_velocity = first_defined_string(
          bc.params,
          {"Prescribed_tangential_mesh_velocity"});
  if (tangential_velocity.has_value()) {
    const auto v = parse_real_vector3(
        *tangential_velocity, "Free-surface Prescribed_tangential_mesh_velocity");
    fs.prescribed_tangential_mesh_velocity = {
        IncompressibleNavierStokesVMSOptions::ScalarValue{v[0]},
        IncompressibleNavierStokesVMSOptions::ScalarValue{v[1]},
        IncompressibleNavierStokesVMSOptions::ScalarValue{v[2]}};
    if (fs.tangential_mesh_policy != FreeSurfaceTangentialMeshPolicy::Prescribed) {
      if (tangential_policy.has_value() && !explicit_legacy_configuration) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Prescribed tangential mesh velocity "
            "conflicts with an explicit non-Prescribed tangential policy.");
      }
      fs.tangential_mesh_policy = FreeSurfaceTangentialMeshPolicy::Prescribed;
    }
  }
  const auto tangential_mesh_penalty = first_defined_double(
      bc.params,
      {"Tangential_mesh_penalty"});
  if (tangential_mesh_penalty.has_value()) {
    if (!std::isfinite(*tangential_mesh_penalty) ||
        !(*tangential_mesh_penalty > 0.0)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface tangential mesh penalty "
          "must be finite and positive.");
    }
    fs.tangential_mesh_penalty =
        IncompressibleNavierStokesVMSOptions::ScalarValue{
            static_cast<svmp::FE::Real>(*tangential_mesh_penalty)};
  }
  if (tangential_mesh_penalty.has_value() &&
      fs.tangential_mesh_policy !=
          FreeSurfaceTangentialMeshPolicy::Prescribed &&
      fs.tangential_mesh_policy !=
          FreeSurfaceTangentialMeshPolicy::SmoothingOnly &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Tangential_mesh_penalty requires "
        "Tangential_mesh_policy=Prescribed or SmoothingOnly; Free has no "
        "tangential boundary row, and unused tangential settings are "
        "accepted only by the explicit schema-1 legacy mode.");
  }

  const auto enforcement = first_defined_string(
      bc.params,
      {"Kinematic_enforcement"});
  if (enforcement.has_value()) {
    fs.kinematic_enforcement =
        parse_free_surface_kinematic_enforcement(
            *enforcement, "Free-surface Kinematic_enforcement");
  } else if (fs.implementation == FreeSurfaceImplementation::FittedALE &&
             !explicit_legacy_configuration) {
    // Decision D35: MeshNitsche is the qualified default for fitted ALE
    // free surfaces.  The qualified contract rejected an omitted
    // Kinematic_enforcement before, so no accepted input changes; the
    // explicit schema-1 legacy mode keeps its None default.
    fs.kinematic_enforcement = FreeSurfaceKinematicEnforcement::MeshNitsche;
  }
  // Kinematic_penalty set the penalty of the Penalty enforcement (the
  // schema-1 legacy mode promoted it to Penalty); both were retired by
  // decision D47 Q9, so the key fails whenever it is given.
  if (first_defined_string(bc.params, {"Kinematic_penalty"}).has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Kinematic_penalty (the penalty of the Penalty kinematic "
        "enforcement) was retired (decision D47 Q9); remove it and use "
        "Kinematic_enforcement=MeshNitsche.");
  }
  const auto kinematic_nitsche_gamma = first_defined_double(
          bc.params,
          {"Kinematic_nitsche_gamma"});
  if (kinematic_nitsche_gamma.has_value()) {
    fs.kinematic_nitsche_gamma =
        static_cast<svmp::FE::Real>(*kinematic_nitsche_gamma);
  }
  // Kinematic_nitsche_symmetric and Kinematic_nitsche_scale_with_p, which
  // no deck, runner or test set, were removed (decision D47); the Nitsche
  // enforcement they configured was retired by D47 Q9.
  if (kinematic_nitsche_gamma.has_value() &&
      fs.kinematic_enforcement != FreeSurfaceKinematicEnforcement::MeshNitsche &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Kinematic_nitsche_gamma requires Kinematic_enforcement=MeshNitsche; an unused gamma is accepted only by the explicit schema-1 legacy mode.");
  }

  const auto small_cut_aggregation = first_defined_bool(
          bc.params,
          {"Small_cut_aggregation"});
  if (small_cut_aggregation.has_value()) {
    fs.small_cut_aggregation = *small_cut_aggregation;
  }

  bool aggregation_guard_suboption_present = false;
  if (const auto maximum_root_path = first_defined_int(
          bc.params,
          {"Small_cut_aggregation_maximum_root_path_length"})) {
    aggregation_guard_suboption_present = true;
    if (*maximum_root_path <= 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum root path length must be positive.");
    }
    fs.small_cut_aggregation_guards.maximum_root_path_length =
        static_cast<std::size_t>(*maximum_root_path);
  }
  if (const auto maximum_extrapolation = first_defined_double(
          bc.params,
          {"Small_cut_aggregation_maximum_reference_extrapolation_distance"})) {
    aggregation_guard_suboption_present = true;
    if (!std::isfinite(*maximum_extrapolation) ||
        *maximum_extrapolation < 0.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum reference extrapolation distance must be finite and "
          "nonnegative.");
    }
    fs.small_cut_aggregation_guards
        .maximum_reference_extrapolation_distance =
        static_cast<svmp::FE::Real>(*maximum_extrapolation);
  }
  if (const auto maximum_coefficient = first_defined_double(
          bc.params,
          {"Small_cut_aggregation_maximum_absolute_coefficient"})) {
    aggregation_guard_suboption_present = true;
    if (!std::isfinite(*maximum_coefficient) ||
        *maximum_coefficient < 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum absolute coefficient must be finite and at least 1.");
    }
    fs.small_cut_aggregation_guards.maximum_absolute_coefficient =
        static_cast<svmp::FE::Real>(*maximum_coefficient);
  }
  if (const auto maximum_row_norm = first_defined_double(
          bc.params,
          {"Small_cut_aggregation_maximum_row_l1_norm"})) {
    aggregation_guard_suboption_present = true;
    if (!std::isfinite(*maximum_row_norm) || *maximum_row_norm < 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum row L1 norm must be finite and at least 1.");
    }
    fs.small_cut_aggregation_guards.maximum_row_l1_norm =
        static_cast<svmp::FE::Real>(*maximum_row_norm);
  }
  if (const auto rootless_fallback = first_defined_bool(
          bc.params,
          {"Small_cut_aggregation_rootless_fallback"})) {
    // Opt-in fail-safe for candidates without a root inside the guards
    // (SmallCutAggregationGuardOptions::rootless_fallback, D33).
    aggregation_guard_suboption_present = true;
    fs.small_cut_aggregation_guards.rootless_fallback = *rootless_fallback;
  }
  if (fs.small_cut_aggregation_guards.maximum_row_l1_norm <
      fs.small_cut_aggregation_guards.maximum_absolute_coefficient) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
        "maximum row L1 norm must be no smaller than the maximum absolute "
        "coefficient.");
  }
  if (!fs.small_cut_aggregation && aggregation_guard_suboption_present &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Small_cut_aggregation=false cannot be "
        "combined with aggregation guard settings; unused guards are "
        "accepted only by the explicit schema-1 legacy mode.");
  }

  const auto cut_cell_stabilization_enabled = first_defined_bool(
          bc.params,
          {"Enable_cut_cell_stabilization"});
  const bool cut_cell_stabilization_explicitly_disabled =
      cut_cell_stabilization_enabled.has_value() &&
      !*cut_cell_stabilization_enabled;
  bool cut_cell_suboption_present = false;
  if (cut_cell_stabilization_enabled.has_value()) {
    fs.cut_cell_stabilization.enabled = *cut_cell_stabilization_enabled;
  }
  if (first_defined_double(
          bc.params,
          {"Cut_cell_velocity_gradient_penalty"})) {
    if (!explicit_legacy_configuration) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Cut-cell velocity ghost-penalty settings are retired; small-cut aggregation replaces them. Only explicit schema-1 legacy input may retain an ignored archived value.");
    }
    std::cerr << "[svMultiPhysics::Physics] WARNING: "
                 "Cut_cell_velocity_gradient_penalty is deprecated and "
                 "ignored by explicit schema-1 legacy mode; small-cut aggregation replaces the velocity "
                 "ghost penalty." << std::endl;
  }
  if (const auto pressure_penalty = first_defined_double(
          bc.params,
          {"Cut_cell_pressure_gradient_penalty"})) {
    cut_cell_suboption_present = true;
    fs.cut_cell_stabilization.pressure_gradient_penalty =
        IncompressibleNavierStokesVMSOptions::ScalarValue{
            static_cast<svmp::FE::Real>(*pressure_penalty)};
    if (!cut_cell_stabilization_explicitly_disabled) {
      fs.cut_cell_stabilization.enabled = true;
    }
  }
  if (const auto pressure_policy = first_defined_string(
          bc.params,
          {"Cut_cell_pressure_stabilization_policy"})) {
    cut_cell_suboption_present = true;
    fs.cut_cell_stabilization.pressure_policy =
        parse_free_surface_pressure_stabilization_policy(
            *pressure_policy,
            "Free-surface Cut_cell_pressure_stabilization_policy");
  }
  if (const auto use_cut_scale = first_defined_bool(
          bc.params,
          {"Use_cut_metadata_scale"})) {
    cut_cell_suboption_present = true;
    fs.cut_cell_stabilization.use_cut_metadata_scale = *use_cut_scale;
  }
  if (const auto cut_scale_cap = first_defined_double(
          bc.params,
          {"Cut_cell_metadata_scale_cap"})) {
    cut_cell_suboption_present = true;
    if (!std::isfinite(*cut_scale_cap) || *cut_scale_cap < 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface "
          "Cut_cell_metadata_scale_cap must be finite and at least 1.");
    }
    fs.cut_cell_stabilization.cut_metadata_scale_cap =
        static_cast<svmp::FE::Real>(*cut_scale_cap);
  }
  if (cut_cell_stabilization_explicitly_disabled &&
      cut_cell_suboption_present && !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Enable_cut_cell_stabilization=false cannot be combined with cut-cell stabilization suboptions; unused stabilization settings are accepted only by the explicit schema-1 legacy mode.");
  }
  if (!cut_cell_stabilization_enabled.has_value() &&
      cut_cell_suboption_present) {
    fs.cut_cell_stabilization.enabled = true;
  }
  if (first_defined_int(
          bc.params,
          {"Cut_cell_velocity_max_derivative_order"})) {
    if (!explicit_legacy_configuration) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Cut-cell velocity derivative-order settings are retired with the velocity ghost penalty. Only explicit schema-1 legacy input may retain an ignored archived value.");
    }
    std::cerr << "[svMultiPhysics::Physics] WARNING: "
                 "Cut_cell_velocity_max_derivative_order is deprecated and "
                 "ignored by explicit schema-1 legacy mode (velocity ghost penalty retired)." << std::endl;
  }

  // The legacy same-field dry-velocity diffusion (Velocity_extension) was
  // retired with decision D42; it modified the physical momentum rows.
  // Decks may still state it as disabled; an explicit schema-1 legacy deck
  // may also keep an ignored archived diffusivity next to that.
  const auto velocity_extension_enabled = first_defined_bool(
      bc.params,
      {"Enable_velocity_extension"});
  const bool velocity_extension_explicitly_disabled =
      velocity_extension_enabled.has_value() &&
      !*velocity_extension_enabled;
  const bool velocity_extension_diffusivity_present =
      first_defined_double(
          bc.params,
          {"Velocity_extension_diffusivity"})
          .has_value();
  if (velocity_extension_enabled.value_or(false) ||
      (velocity_extension_diffusivity_present &&
       !velocity_extension_explicitly_disabled)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': the legacy same-field dry-velocity diffusion "
        "(Enable_velocity_extension / Velocity_extension_diffusivity) was "
        "retired (decision D42) because it modified the physical momentum; "
        "the level-set advection velocity extension and small-cut "
        "aggregation replace it.  Remove these keys or set "
        "Enable_velocity_extension=false.");
  }
  if (velocity_extension_diffusivity_present &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Velocity_extension_diffusivity belongs to the retired legacy "
        "dry-velocity diffusion (decision D42); only explicit schema-1 "
        "legacy input may retain an ignored archived value.");
  }

  append_free_surface_contact_line(bc.params, fs);
  options.free_surface.push_back(std::move(fs));
}

svmp::Physics::formulations::navier_stokes::
    IncompressibleNavierStokesVMSOptions
translate_fitted_surface_contact_capability(
    const svmp::Physics::EquationModuleInput& input)
{
  using svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions;

  IncompressibleNavierStokesVMSOptions options{};
  apply_free_surface_schema_options(input.equation_params, options);
  for (const auto& bc : input.boundary_conditions) {
    const auto* bc_type = find_param(bc.params, "Type");
    if (bc_type == nullptr || !bc_type->defined ||
        !is_free_surface_type(bc_type->value)) {
      continue;
    }

    const auto* time_dependence =
        find_param(bc.params, "Time_dependence");
    const auto time_value =
        time_dependence != nullptr && time_dependence->defined
            ? lower_copy(trim_copy(time_dependence->value))
            : std::string{"steady"};
    const bool is_steady = time_value.empty() || time_value == "steady";
    const bool has_temporal_spatial_file = has_nonempty_defined(
        bc.params, "Temporal_and_spatial_values_file_path");
    const bool has_other_files =
        has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") ||
        has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path") ||
        has_nonempty_defined(bc.params, "Fourier_coefficients_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_profile_file_path");
    append_free_surface_bc(
        bc,
        is_steady,
        has_temporal_spatial_file,
        has_other_files,
        options);
  }
  return options;
}

void validate_fitted_surface_contact_capability(
    const svmp::Physics::formulations::navier_stokes::
        IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::
      FreeSurfaceImplementation;
  using ContactLine = svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

  for (const auto& boundary : options.free_surface) {
    if (boundary.implementation != FreeSurfaceImplementation::FittedALE) {
      continue;
    }
    // Fitted SurfaceStress needs no opt-in (decision D47 Q2(b)); the module
    // checks its frame, its surface tension and its contact lines.
    for (const auto& contact_line : boundary.contact_lines) {
      if (std::holds_alternative<ContactLine::PrescribedAngle>(
              contact_line.configuration)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed fitted contact angles are unsupported until a true fitted contact-line (codimension-two) integration entity is available; the condition must not be integrated over the complete free-surface boundary");
      }
      if (std::holds_alternative<ContactLine::DynamicRenE>(
              contact_line.configuration)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle is currently supported only for sharp unfitted level-set free surfaces");
      }
    }
  }
}

void apply_fluid_bcs(const svmp::Physics::EquationModuleInput& input,
                     const svmp::Physics::DomainInput& domain,
                     MarkerCommunicator comm,
                     svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation;

  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes BC translation received null mesh.");
  }

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes BC translation: " +
        std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  svmp::FE::Real backflow_beta = 0.0;
  if (const auto* p = find_param(domain.params, "Backflow_stabilization_coefficient")) {
    if (!trim_copy(p->value).empty()) {
      backflow_beta = static_cast<svmp::FE::Real>(parse_double(p->value, "Backflow_stabilization_coefficient"));
    }
  }

  for (const auto& bc : input.boundary_conditions) {
    const auto* bc_type_raw = find_param(bc.params, "Type");
    const std::string bc_type = bc_type_raw ? trim_copy(bc_type_raw->value) : std::string{};
    const std::string bc_type_lc = lower_copy(bc_type);
    const bool free_surface_type = is_free_surface_type(bc_type);
    const bool unfitted_free_surface =
        free_surface_type &&
        free_surface_implementation_from_params(bc.params) == FreeSurfaceImplementation::UnfittedLevelSet;

    if (bc.boundary_marker == svmp::INVALID_LABEL && !unfitted_free_surface) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* time_dep = find_param(bc.params, "Time_dependence");
    const std::string time_value_raw =
        (time_dep && time_dep->defined) ? trim_copy(time_dep->value) : std::string("Steady");
    const std::string time_value_lc = lower_copy(time_value_raw);

    const bool is_steady = time_value_lc.empty() || time_value_lc == "steady";
    const bool is_general = time_value_lc == "general";
    const bool is_unsteady = time_value_lc == "unsteady";
    const bool is_resistance = time_value_lc == "resistance";
    const bool is_rcr = (time_value_lc == "rcr" || time_value_lc == "windkessel");
    const bool is_rcrcr =
        (time_value_lc == "rcrcr" || time_value_lc == "windkessel2c" || time_value_lc == "windkessel_2c");

    const bool has_temp_spat = has_nonempty_defined(bc.params, "Temporal_and_spatial_values_file_path");
    const bool has_other_files = has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") || has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path") ||
        has_nonempty_defined(bc.params, "Fourier_coefficients_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_profile_file_path");

    if (is_steady && (has_temp_spat || has_other_files)) {
      throw std::runtime_error(
        "[svMultiPhysics::Physics] Spatial/temporal boundary-condition files are not supported for the new solver "
        "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (free_surface_type) {
      append_free_surface_bc(bc, is_steady, has_temp_spat, has_other_files, options);
      continue;
    }

    const auto* value_param = find_param(bc.params, "Value");
    const svmp::FE::Real magnitude =
        static_cast<svmp::FE::Real>(value_param ? parse_double(value_param->value, "Add_BC/Value") : 0.0);

    std::vector<int> effective_dir{};
    if (const auto* dir_param = find_param(bc.params, "Effective_direction")) {
      const auto s = trim_copy(dir_param->value);
      if (!s.empty()) {
        effective_dir = parse_int_list(s);
      }
    }

    if (bc_type_lc == "dir" || bc_type_lc == "dirichlet") {
      if (!is_steady && !is_general && !is_unsteady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Only Steady, General, and Unsteady boundary conditions are supported for the new solver "
            "Navier-Stokes module Dirichlet BCs (got Time_dependence='" +
            time_value_raw + "'). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }

      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        // Legacy inputs sometimes set this flag for clarity; the new solver applies
        // velocity Dirichlet along the boundary normal by default when Effective_direction is unset.
        // Accept the flag for Dirichlet to improve legacy compatibility.
      }

      const auto* weak_param = find_param(bc.params, "Weakly_applied");
      const bool weak = weak_param && weak_param->defined && parse_bool_relaxed(weak_param->value);

      if (is_general) {
        if (!has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Navier-Stokes Dirichlet BC currently supports only "
              "<Temporal_and_spatial_values_file_path>. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }
        if (weak) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BCs from temporal/spatial files are only supported as strong "
              "Dirichlet (Weakly_applied=false). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto* file_param = find_param(bc.params, "Temporal_and_spatial_values_file_path");
        const std::string file_path =
            (file_param && file_param->defined) ? trim_copy(file_param->value) : std::string{};
        if (file_path.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BC is missing Temporal_and_spatial_values_file_path.");
        }

        auto data = read_temporal_and_spatial_values_file(
            *input.mesh, bc.boundary_marker, file_path, comm);

        IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
        dir.boundary_marker = bc.boundary_marker;

        for (int d = 0; d < dim; ++d) {
          if (d < data->dof) {
            const int comp = d;
            dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
                [data, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real t) -> svmp::FE::Real {
                  const std::array<svmp::FE::Real, 3> p{x, y, z};
                  const auto node = data->findNodeIndex(p);
                  return data->interpolate(node, t, comp);
                });
          } else {
            dir.value[static_cast<std::size_t>(d)] = IncompressibleNavierStokesVMSOptions::ScalarValue{0.0};
          }
        }

        options.velocity_dirichlet.push_back(std::move(dir));
        continue;
      }

      if (is_unsteady) {
        // Unsteady Dirichlet BC: time-varying magnitude from a .flow file + spatial profile (Flat/Parabolic).
        const auto* file_param = find_param(bc.params, "Temporal_values_file_path");
        const std::string flow_file =
            (file_param && file_param->defined) ? trim_copy(file_param->value) : std::string{};
        if (flow_file.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Unsteady Dirichlet BC '" + bc.name +
              "' is missing <Temporal_values_file_path>.");
        }

        auto temporal = svmp::Physics::readTemporalValuesFile(
            flow_file, /*num_components=*/1, svmp::Physics::TemporalEndBehavior::Periodic);

        const auto* impose_flux_param_u = find_param(bc.params, "Impose_flux");
        const bool impose_flux_u = impose_flux_param_u ? parse_bool_relaxed(impose_flux_param_u->value) : false;

        const auto* profile_param_u = find_param(bc.params, "Profile");
        const std::string profile_raw_u =
            (profile_param_u && profile_param_u->defined) ? trim_copy(profile_param_u->value) : std::string("Flat");
        const std::string profile_lc_u = lower_copy(profile_raw_u);

        InletProfileType profile_u = InletProfileType::Flat;
        if (profile_lc_u == "flat") {
          profile_u = InletProfileType::Flat;
        } else if (profile_lc_u == "parabolic") {
          profile_u = InletProfileType::Parabolic;
        } else {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='" + profile_raw_u +
              "' for Unsteady BC '" + bc.name + "'. Supported profiles: Flat, Parabolic.");
        }

        int active_count_u = 0;
        std::array<int, 3> active_u{0, 0, 0};
        for (int d = 0; d < dim; ++d) {
          const int flag = (static_cast<std::size_t>(d) < effective_dir.size() && effective_dir[static_cast<std::size_t>(d)] != 0) ? 1 : 0;
          active_u[static_cast<std::size_t>(d)] = flag;
          active_count_u += flag;
        }
        const bool use_normal_u = (active_count_u == 0 || active_count_u == dim);

        auto ctx_u = std::make_shared<InletProfileContext>();
        ctx_u->dim = dim;
        ctx_u->profile = profile_u;
        ctx_u->use_normal_direction = use_normal_u;
        ctx_u->active_components = active_u;

        const auto g_u =
            global_marker_geometry(*input.mesh, bc.boundary_marker, comm);
        if (!(g_u.area > 0.0)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
              " has zero area; cannot apply Unsteady Dirichlet BC '" + bc.name + "'.");
        }

        if (use_normal_u) {
          const Vec3d n = normalized(g_u.normal_sum);
          if (!(norm2(n) > 0.0)) {
            throw std::runtime_error(
                "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
                " has a degenerate normal; cannot apply Unsteady Dirichlet BC '" + bc.name + "'.");
          }
          ctx_u->normal = n;
        }

        if (profile_u == InletProfileType::Parabolic) {
          ctx_u->parabolic = build_parabolic_profile_data(
              *input.mesh, bc.boundary_marker, comm);
        }

      double normalization_u = 1.0;
      if (impose_flux_u) {
        if (profile_u == InletProfileType::Flat) {
          normalization_u = g_u.area;
        } else {
            normalization_u = integrate_parabolic_weight_over_marker(
                *input.mesh, bc.boundary_marker, *ctx_u->parabolic, comm);
          }
          if (!(normalization_u > 0.0)) {
            throw std::runtime_error(
                "[svMultiPhysics::Physics] Failed to compute positive normalization for <Impose_flux> on Unsteady "
                "Dirichlet BC '" + bc.name + "'.");
        }
      }

        // Set scale=1/normalization; the time-dependent flow rate is multiplied in the callback.
        ctx_u->scale = 1.0 / normalization_u;

        if (navierStokesTraceEnabled()) {
          const auto local_g = local_marker_geometry(*input.mesh, bc.boundary_marker);
          const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
          std::ostringstream oss;
          oss << "NavierStokes BC setup: unsteady Dirichlet marker=" << bc.boundary_marker
              << " name='" << bc.name << "'"
              << " profile=" << ((profile_u == InletProfileType::Parabolic) ? "Parabolic" : "Flat")
              << " impose_flux=" << (impose_flux_u ? 1 : 0)
              << " local_faces=" << local_faces.size()
              << " local_area=" << local_g.area
              << " global_area=" << g_u.area
              << " normalization=" << normalization_u
              << " scale=" << ctx_u->scale
              << " center=(" << ((g_u.area > 0.0) ? g_u.center_sum.x / g_u.area : 0.0)
              << "," << ((g_u.area > 0.0) ? g_u.center_sum.y / g_u.area : 0.0)
              << "," << ((g_u.area > 0.0) ? g_u.center_sum.z / g_u.area : 0.0) << ")"
              << " normal=(" << ctx_u->normal.x << "," << ctx_u->normal.y << "," << ctx_u->normal.z << ")"
              << " perim_points=" << (ctx_u->parabolic.has_value() ? ctx_u->parabolic->perimeter_unit_dirs.size() : 0u);
          navierStokesTraceLog(oss.str());
        }

        IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir_u{};
        dir_u.boundary_marker = bc.boundary_marker;
        dir_u.active_components = use_normal_u
                                      ? std::array<bool, 3>{true, true, true}
                                      : active_components_from_flags(active_u, dim);

        for (int d = 0; d < dim; ++d) {
          const int comp = d;
          dir_u.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
              [ctx_u, comp, temporal](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real t) -> svmp::FE::Real {
                const double flow_rate = temporal->interpolate(static_cast<double>(t));
                const Vec3d p{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
                // componentValue = scale * weight(x) * direction_factor
                // With scale = 1/normalization, this gives flow_rate / normalization * weight * direction
                return static_cast<svmp::FE::Real>(flow_rate * ctx_u->componentValue(comp, p));
              });
        }

        if (weak) {
          options.velocity_dirichlet_weak.push_back(std::move(dir_u));
        } else {
          options.velocity_dirichlet.push_back(std::move(dir_u));
        }
        continue;
      }

      IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
      dir.boundary_marker = bc.boundary_marker;

      // Legacy-style Dirichlet velocity BCs:
      // - default direction: boundary outward normal
      // - profile: Flat or Parabolic
      // - Impose_flux: if true, normalize profile so ∫ profile ds = 1 and interpret Value as flow rate
      const auto* impose_flux_param = find_param(bc.params, "Impose_flux");
      const bool impose_flux = impose_flux_param ? parse_bool_relaxed(impose_flux_param->value) : false;

      const auto* profile_param = find_param(bc.params, "Profile");
      const std::string profile_raw =
          (profile_param && profile_param->defined) ? trim_copy(profile_param->value) : std::string("Flat");
      const std::string profile_lc = lower_copy(profile_raw);

      InletProfileType profile = InletProfileType::Flat;
      if (profile_lc == "flat") {
        profile = InletProfileType::Flat;
      } else if (profile_lc == "parabolic") {
        profile = InletProfileType::Parabolic;
      } else if (profile_lc == "user_defined" || profile_lc == "user-defined" || profile_lc == "userdefined") {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Dirichlet BC Profile='User_defined' is not supported for the new solver "
            "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      } else {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='" + profile_raw +
            "' for the new solver Navier-Stokes module. Supported profiles: Flat, Parabolic.");
      }

      // Common case: no-slip wall (Value=0) and other zero Dirichlet constraints.
      // Avoid expensive global geometry computations (MPI allgathers) when the imposed value is zero.
      if (magnitude == static_cast<svmp::FE::Real>(0.0)) {
        if (!effective_dir.empty()) {
          std::array<int, 3> active_zero{0, 0, 0};
          int active_count_zero = 0;
          for (int d = 0; d < dim; ++d) {
            const int flag =
                (static_cast<std::size_t>(d) < effective_dir.size() &&
                 effective_dir[static_cast<std::size_t>(d)] != 0)
                    ? 1
                    : 0;
            active_zero[static_cast<std::size_t>(d)] = flag;
            active_count_zero += flag;
          }
          if (active_count_zero > 0 && active_count_zero < dim) {
            dir.active_components = active_components_from_flags(active_zero, dim);
          }
        }
        if (weak) {
          options.velocity_dirichlet_weak.push_back(std::move(dir));
        } else {
          options.velocity_dirichlet.push_back(std::move(dir));
        }
        continue;
      }

      int active_count = 0;
      std::array<int, 3> active{0, 0, 0};
      for (int d = 0; d < dim; ++d) {
        const int flag = (static_cast<std::size_t>(d) < effective_dir.size() && effective_dir[static_cast<std::size_t>(d)] != 0) ? 1 : 0;
        active[static_cast<std::size_t>(d)] = flag;
        active_count += flag;
      }
      const bool use_normal_direction = (active_count == 0 || active_count == dim);

      auto ctx = std::make_shared<InletProfileContext>();
      ctx->dim = dim;
      ctx->profile = profile;
      ctx->use_normal_direction = use_normal_direction;
      ctx->active_components = active;

      // Compute normal/area/center and profile normalization as needed.
      const auto g =
          global_marker_geometry(*input.mesh, bc.boundary_marker, comm);
      if (!(g.area > 0.0)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
            " has zero area; cannot apply Dirichlet BC '" + bc.name + "'.");
      }

      if (use_normal_direction) {
        const Vec3d n = normalized(g.normal_sum);
        if (!(norm2(n) > 0.0)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
              " has a degenerate normal; cannot apply Dirichlet BC '" + bc.name + "'.");
        }
        ctx->normal = n;
      }

      if (profile == InletProfileType::Parabolic) {
        ctx->parabolic = build_parabolic_profile_data(
            *input.mesh, bc.boundary_marker, comm);
      }

      double normalization = 1.0;
      if (impose_flux) {
        if (profile == InletProfileType::Flat) {
          normalization = g.area;
        } else {
          normalization = integrate_parabolic_weight_over_marker(
              *input.mesh, bc.boundary_marker, *ctx->parabolic, comm);
        }
        if (!(normalization > 0.0)) {
          const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
          const std::size_t perim_points =
              (ctx->parabolic.has_value()) ? ctx->parabolic->perimeter_unit_dirs.size() : 0u;
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Failed to compute a positive normalization for <Impose_flux>true</Impose_flux> "
              "on Dirichlet BC '" + bc.name + "' (marker=" + std::to_string(bc.boundary_marker) +
              ", normalization=" + std::to_string(normalization) + ", area=" + std::to_string(g.area) +
              ", perim_points=" + std::to_string(perim_points) + ", local_faces=" + std::to_string(local_faces.size()) +
              ").");
        }
      }

      ctx->scale = static_cast<double>(magnitude) / normalization;

      if (navierStokesTraceEnabled()) {
        const auto local_g = local_marker_geometry(*input.mesh, bc.boundary_marker);
        const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
        std::ostringstream oss;
        oss << "NavierStokes BC setup: steady Dirichlet marker=" << bc.boundary_marker
            << " name='" << bc.name << "'"
            << " profile=" << ((profile == InletProfileType::Parabolic) ? "Parabolic" : "Flat")
            << " impose_flux=" << (impose_flux ? 1 : 0)
            << " magnitude=" << magnitude
            << " local_faces=" << local_faces.size()
            << " local_area=" << local_g.area
            << " global_area=" << g.area
            << " normalization=" << normalization
            << " scale=" << ctx->scale
            << " center=(" << ((g.area > 0.0) ? g.center_sum.x / g.area : 0.0)
            << "," << ((g.area > 0.0) ? g.center_sum.y / g.area : 0.0)
            << "," << ((g.area > 0.0) ? g.center_sum.z / g.area : 0.0) << ")"
            << " normal=(" << ctx->normal.x << "," << ctx->normal.y << "," << ctx->normal.z << ")"
            << " perim_points=" << (ctx->parabolic.has_value() ? ctx->parabolic->perimeter_unit_dirs.size() : 0u);
        navierStokesTraceLog(oss.str());
      }

      for (int d = 0; d < dim; ++d) {
        const int comp = d;
        dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
            [ctx, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real /*t*/) -> svmp::FE::Real {
              const Vec3d p{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
              return static_cast<svmp::FE::Real>(ctx->componentValue(comp, p));
            });
      }
      dir.active_components = use_normal_direction
                                  ? std::array<bool, 3>{true, true, true}
                                  : active_components_from_flags(active, dim);

      if (weak) {
        options.velocity_dirichlet_weak.push_back(std::move(dir));
      } else {
        options.velocity_dirichlet.push_back(std::move(dir));
      }
      continue;
    }

    if (bc_type_lc == "neu" || bc_type_lc == "neumann") {
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }

      if (is_steady) {
        IncompressibleNavierStokesVMSOptions::PressureOutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.pressure = IncompressibleNavierStokesVMSOptions::ScalarValue{magnitude};
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.pressure_outflow.push_back(std::move(out));
        continue;
      }

      if (is_resistance) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Resistance outflow BCs do not support spatial/temporal files for the new "
              "solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = 0.0;
        out.C = 0.0;
        out.Rd = static_cast<svmp::FE::Real>(magnitude);
        out.Pd = 0.0;
        out.X0 = 0.0;
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      if (is_rcr) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BCs do not support spatial/temporal files for the new solver "
              "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto Rp = get_defined_double(bc.params, "RCR.Proximal_resistance");
        const auto C = get_defined_double(bc.params, "RCR.Capacitance");
        const auto Rd = get_defined_double(bc.params, "RCR.Distal_resistance");
        const auto Pd = get_defined_double(bc.params, "RCR.Distal_pressure");
        const auto X0 = get_defined_double(bc.params, "RCR.Initial_pressure");

        if (!Rp.has_value() || !C.has_value() || !Rd.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BC '" + bc.name + "' is missing required <RCR_values> entries "
              "(Proximal_resistance, Capacitance, Distal_resistance).");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = static_cast<svmp::FE::Real>(*Rp);
        out.C = static_cast<svmp::FE::Real>(*C);
        out.Rd = static_cast<svmp::FE::Real>(*Rd);
        out.Pd = static_cast<svmp::FE::Real>(Pd.value_or(0.0));
        out.X0 = static_cast<svmp::FE::Real>(X0.value_or(0.0));
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      if (is_rcrcr) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCRCR outflow BCs do not support spatial/temporal files for the new solver "
              "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto Rp = get_defined_double(bc.params, "RCRCR.Proximal_resistance");
        const auto C1 = get_defined_double(bc.params, "RCRCR.Proximal_capacitance");
        const auto Rm = get_defined_double(bc.params, "RCRCR.Intermediate_resistance");
        const auto C2 = get_defined_double(bc.params, "RCRCR.Distal_capacitance");
        const auto Rd = get_defined_double(bc.params, "RCRCR.Distal_resistance");
        const auto Pd = get_defined_double(bc.params, "RCRCR.Distal_pressure");
        const auto P10 = get_defined_double(bc.params, "RCRCR.Initial_pressure_1");
        const auto P20 = get_defined_double(bc.params, "RCRCR.Initial_pressure_2");

        if (!Rp.has_value() || !C1.has_value() || !Rm.has_value() || !C2.has_value() || !Rd.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCRCR outflow BC '" + bc.name +
              "' is missing required <RCRCR_values> entries "
              "(Proximal_resistance, Proximal_capacitance, Intermediate_resistance, Distal_capacitance, "
              "Distal_resistance).");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = static_cast<svmp::FE::Real>(*Rp);
        out.C1 = static_cast<svmp::FE::Real>(*C1);
        out.Rm = static_cast<svmp::FE::Real>(*Rm);
        out.C2 = static_cast<svmp::FE::Real>(*C2);
        out.Rd = static_cast<svmp::FE::Real>(*Rd);
        out.Pd = static_cast<svmp::FE::Real>(Pd.value_or(0.0));
        out.P10 = static_cast<svmp::FE::Real>(P10.value_or(Pd.value_or(0.0)));
        out.P20 = static_cast<svmp::FE::Real>(P20.value_or(Pd.value_or(0.0)));
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcrcr.push_back(std::move(out));
        continue;
      }

      throw std::runtime_error(
          "[svMultiPhysics::Physics] Neumann BC Time_dependence='" + time_value_raw +
          "' is not supported for the new solver Navier-Stokes module. Supported: Steady, Resistance, RCR, RCRCR. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (bc_type_lc == "trac" || bc_type_lc == "traction") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady traction BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      IncompressibleNavierStokesVMSOptions::TractionNeumannBC trac{};
      trac.boundary_marker = bc.boundary_marker;
      fill_vector(trac.traction, dim, effective_dir, magnitude);
      options.traction_neumann.push_back(std::move(trac));
      continue;
    }

    if (bc_type_lc == "rbn" || bc_type_lc == "robin") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady Robin BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }
      const auto* stiff_param = find_param(bc.params, "Stiffness");
      const svmp::FE::Real stiff =
          static_cast<svmp::FE::Real>(stiff_param ? parse_double(stiff_param->value, "Add_BC/Stiffness") : 0.0);

      IncompressibleNavierStokesVMSOptions::TractionRobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.alpha = IncompressibleNavierStokesVMSOptions::ScalarValue{stiff};
      fill_vector(robin.rhs, dim, effective_dir, magnitude);
      options.traction_robin.push_back(std::move(robin));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" + bc_type +
        "' is not supported for the new solver Navier-Stokes module. Supported types: Dir, Dirichlet, Neu, Neumann, "
        "Trac, Traction, Robin, Rbn, Free_surface. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

TranslatedIncompressibleTwoFluidInput
translate_incompressible_two_fluid_input(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::FE::systems::FESystem& system)
{
  using Options = svmp::Physics::formulations::navier_stokes::
      IncompressibleTwoFluidOptions;
  using BoundaryOptions = svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions;

  if (!input.mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Incompressible two-fluid module factory received null mesh.");
  }
  if (input.body_force_block_count != 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_body_force_configuration");
  }
  if (!input.domains.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_explicit_domain_configuration");
  }
  if (!trim_copy(input.module_options).empty() ||
      !trim_copy(input.module_options_file_path).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_module_options");
  }
  if (!input.outputs.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_output_configuration");
  }
  if (!input.nested_configuration_blocks.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_equation_block:" +
        input.nested_configuration_blocks.front());
  }
  if (!input.default_domain.nested_configuration_blocks.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_default_domain_block:" +
        input.default_domain.nested_configuration_blocks.front());
  }

  const int dimension = input.mesh->dim();
  if (dimension != 2 && dimension != 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Incompressible two-fluid spaces require dimension two or three.");
  }
  const auto element_type = infer_base_element_type(*input.mesh);
  const bool supported_cell =
      (dimension == 2 && element_type == svmp::FE::ElementType::Triangle3) ||
      (dimension == 3 && element_type == svmp::FE::ElementType::Tetra4);
  const int order =
      resolve_element_order(input, infer_polynomial_order(*input.mesh));
  const bool taylor_hood = first_defined_bool(
      input.equation_params,
      {"Use_taylor_hood_type_basis"}).value_or(false);
  if (!supported_cell || order != 1 || taylor_hood) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Incompressible two-fluid input requires affine equal-order P1 Triangle3 or Tetra4 spaces.");
  }

  Options options;
  if (input.node_pressure_constraints.has_value()) {
    const auto& gauge_input = *input.node_pressure_constraints;
    (void)parse_node_pressure_constraint_id_type(gauge_input.id_type);
    const auto values =
        read_node_pressure_constraints_csv(gauge_input.values_file_path);
    if (values.size() != 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_two_fluid_shared_pressure_gauge_count");
    }
    options.shared_pressure_gauge =
        svmp::Physics::formulations::navier_stokes::
            IncompressibleTwoFluidSharedPressureGauge{
            .vertex_gid = values.front().node_id,
            .pressure = values.front().pressure,
        };
  }
  options.level_set_field_name = require_two_fluid_string(
      input.equation_params,
      {"Level_set_field_name"},
      "Level_set_field_name");
  options.generated_interface_domain_id = require_two_fluid_string(
      input.equation_params,
      {"Generated_interface_domain_id"},
      "Generated_interface_domain_id");
  options.interface_marker = require_two_fluid_int(
      input.equation_params,
      {"Material_interface_marker"},
      "Material_interface_marker");
  options.negative_phase.density = require_two_fluid_real(
      input.equation_params,
      {"Negative_phase_density"},
      "Negative_phase_density");
  options.negative_phase.viscosity = require_two_fluid_real(
      input.equation_params,
      {"Negative_phase_dynamic_viscosity"},
      "Negative_phase_dynamic_viscosity");
  options.positive_phase.density = require_two_fluid_real(
      input.equation_params,
      {"Positive_phase_density"},
      "Positive_phase_density");
  options.positive_phase.viscosity = require_two_fluid_real(
      input.equation_params,
      {"Positive_phase_dynamic_viscosity"},
      "Positive_phase_dynamic_viscosity");
  options.surface_tension = require_two_fluid_real(
      input.equation_params,
      {"Two_fluid_surface_tension"},
      "Two_fluid_surface_tension");
  options.interface_nitsche_gamma = require_two_fluid_real(
      input.equation_params,
      {"Two_fluid_interface_nitsche_gamma"},
      "Two_fluid_interface_nitsche_gamma");
  if (any_parameter_defined(
          input.equation_params,
          {"Prescribed_pressure_jump"})) {
    options.prescribed_pressure_jump = require_two_fluid_real(
        input.equation_params,
        {"Prescribed_pressure_jump"},
        "Prescribed_pressure_jump");
  }
  if (any_parameter_defined(
          input.equation_params,
          {"Prescribed_viscous_traction_jump_x",
           "Prescribed_viscous_traction_jump_y",
           "Prescribed_viscous_traction_jump_z"})) {
    if (dimension == 2 &&
        any_parameter_defined(
            input.equation_params,
            {"Prescribed_viscous_traction_jump_z"})) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_two_fluid_out_of_plane_prescribed_viscous_traction_jump");
    }
    std::array<svmp::FE::Real, 3> target{};
    target[0] = require_two_fluid_real(
        input.equation_params,
        {"Prescribed_viscous_traction_jump_x"},
        "Prescribed_viscous_traction_jump_x");
    target[1] = require_two_fluid_real(
        input.equation_params,
        {"Prescribed_viscous_traction_jump_y"},
        "Prescribed_viscous_traction_jump_y");
    if (dimension == 3) {
      target[2] = require_two_fluid_real(
          input.equation_params,
          {"Prescribed_viscous_traction_jump_z"},
          "Prescribed_viscous_traction_jump_z");
    }
    options.prescribed_viscous_traction_jump = target;
  }
  options.require_conservative_phase_momentum_reconciliation = true;
  if (options.interface_marker < 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Material_interface_marker must be nonnegative for incompressible two-fluid input.");
  }
  if (any_parameter_defined(
          input.equation_params, {"Operator_tag", "OperatorTag"})) {
    options.operator_tag = require_two_fluid_string(
        input.equation_params,
        {"Operator_tag", "OperatorTag"},
        "Operator_tag");
  }
  options.enable_convection = input.equation_type != "stokes";
  options.jit_policy =
      svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);

  for (std::size_t component = 0u; component < options.body_force.size();
       ++component) {
    static constexpr std::array<std::string_view, 3> force_keys{
        "Force_x", "Force_y", "Force_z"};
    if (const auto value = get_defined_double(
            input.default_domain.params, force_keys[component])) {
      options.body_force[component] = static_cast<svmp::FE::Real>(*value);
    }
  }

  BoundaryOptions translated_boundaries;
  apply_fluid_moving_domain_options(
      input, input.default_domain, translated_boundaries);
  if (translated_boundaries.enable_ale) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_moving_mesh");
  }
  apply_fluid_momentum_source_params(
      input.equation_params, translated_boundaries);
  apply_fluid_momentum_source_params(
      input.default_domain.params, translated_boundaries);
  if (!translated_boundaries.body_force_field_name.empty() ||
      any_parameter_defined(
          input.equation_params,
          {"Momentum_source_temporal_and_spatial_values_file_path",
           "MomentumSourceTemporalAndSpatialValuesFilePath",
           "Body_force_temporal_and_spatial_values_file_path",
           "BodyForceTemporalAndSpatialValuesFilePath"}) ||
      any_parameter_defined(
          input.default_domain.params,
          {"Momentum_source_temporal_and_spatial_values_file_path",
           "MomentumSourceTemporalAndSpatialValuesFilePath",
           "Body_force_temporal_and_spatial_values_file_path",
           "BodyForceTemporalAndSpatialValuesFilePath"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_nonconstant_momentum_source");
  }
  reject_unsupported_two_fluid_equation_parameters(
      input.equation_params);
  reject_unsupported_two_fluid_default_domain_parameters(
      input.default_domain.params);
  apply_fluid_bcs(
      input,
      input.default_domain,
      markerCommunicator(system),
      translated_boundaries);
  if (!translated_boundaries.velocity_dirichlet_weak.empty() ||
      !translated_boundaries.pressure_dirichlet.empty() ||
      !translated_boundaries.traction_neumann.empty() ||
      !translated_boundaries.traction_robin.empty() ||
      !translated_boundaries.pressure_outflow.empty() ||
      !translated_boundaries.free_surface.empty() ||
      !translated_boundaries.coupled_outflow_rcr.empty() ||
      !translated_boundaries.coupled_outflow_rcrcr.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_boundary_condition");
  }
  reject_unsupported_two_fluid_boundary_parameters(
      input.boundary_conditions);
  options.shared_velocity_dirichlet =
      std::move(translated_boundaries.velocity_dirichlet);

  auto velocity_space = svmp::FE::spaces::VectorSpace(
      svmp::FE::spaces::SpaceType::H1,
      element_type,
      1,
      dimension);
  auto pressure_space =
      svmp::FE::spaces::SpaceFactory::create_h1(element_type, 1);
  return TranslatedIncompressibleTwoFluidInput{
      .velocity_space = std::move(velocity_space),
      .pressure_space = std::move(pressure_space),
      .options = std::move(options),
  };
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_incompressible_two_fluid_from_input(
    const svmp::Physics::EquationModuleInput& input,
    svmp::FE::systems::FESystem& system)
{
  using Module = svmp::Physics::formulations::navier_stokes::
      IncompressibleTwoFluidModule;
  auto translated = translate_incompressible_two_fluid_input(input, system);
  auto module = std::make_unique<Module>(
      translated.velocity_space,
      translated.pressure_space,
      translated.velocity_space,
      translated.pressure_space,
      std::move(translated.options));
  module->registerOn(system);
  return module;
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_navier_stokes_from_input(const svmp::Physics::EquationModuleInput& input,
                                svmp::FE::systems::FESystem& system)
{
  svmp::Physics::formulations::navier_stokes::
      preflightFittedSurfaceContactCapability(input);
  const auto physical_model =
      validate_and_resolve_free_surface_physical_model(input);
  if (physical_model ==
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    return create_incompressible_two_fluid_from_input(input, system);
  }
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes module factory received null mesh.");
  }

  const auto& domain = select_single_domain(input, "Navier-Stokes");

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes spaces: " + std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int vel_order = resolve_element_order(input, infer_polynomial_order(*input.mesh));

  bool taylor_hood = false;
  if (const auto* p = find_param(input.equation_params, "Use_taylor_hood_type_basis"); p && p->defined) {
    taylor_hood = parse_bool_relaxed(p->value);
  }
  const int p_order = taylor_hood ? std::max(1, vel_order - 1) : vel_order;

  auto velocity_space =
      svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, element_type, vel_order, dim);
  auto pressure_space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, p_order);

  svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions options{};
  options.velocity_field_name = "Velocity";
  options.pressure_field_name = "Pressure";
  options.free_surface_physical_model =
      svmp::Physics::formulations::navier_stokes::
          FreeSurfacePhysicalModel::
              OnePhaseLiquidPrescribedExteriorPressure;
  if (const auto operator_tag = first_defined_string(
          input.equation_params, {"Operator_tag", "OperatorTag"})) {
    options.operator_tag = *operator_tag;
  }
  options.enable_convection = (input.equation_type != "stokes");
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);
  options.conservative_balance_diagnostics =
      input.derived_requests.conservative_balance_diagnostics;

  apply_free_surface_schema_options(input.equation_params, options);
  apply_fluid_moving_domain_options(input, domain, options);
  apply_fluid_momentum_source_params(input.equation_params, options);
  apply_fluid_properties(domain, options);
  apply_fluid_momentum_source_spacetime_file(
      input, domain, dim, markerCommunicator(system), options);
  apply_fluid_rotating_frame_coriolis(input, domain, dim, options);
  apply_node_pressure_constraints(input, options);
  apply_fluid_bcs(input, domain, markerCommunicator(system), options);

  auto module = std::make_unique<svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule>(
      std::move(velocity_space), std::move(pressure_space), std::move(options));
  module->registerOn(system);
  return module;
}

} // namespace

SVMP_REGISTER_EQUATION("fluid", &create_navier_stokes_from_input);
SVMP_REGISTER_EQUATION("stokes", &create_navier_stokes_from_input);

namespace svmp::Physics::formulations::navier_stokes {

void preflightFittedSurfaceContactCapability(
    const svmp::Physics::EquationModuleInput& input)
{
  if (input.equation_type != "fluid" && input.equation_type != "stokes") {
    throw std::invalid_argument(
        "preflightFittedSurfaceContactCapability requires a fluid or stokes equation input");
  }
  const auto physical_model =
      validate_and_resolve_free_surface_physical_model(input);
  if (physical_model ==
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    return;
  }
  validate_fitted_surface_contact_capability(
      translate_fitted_surface_contact_capability(input));
}

std::optional<IncompressibleTwoFluidDependency>
incompressibleTwoFluidDependency(
    const svmp::Physics::EquationModuleInput& input,
    const FE::systems::FESystem& system)
{
  if (input.equation_type != "fluid" && input.equation_type != "stokes") {
    throw std::invalid_argument(
        "incompressibleTwoFluidDependency requires a fluid or stokes equation input");
  }
  if (validate_and_resolve_free_surface_physical_model(input) !=
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    return std::nullopt;
  }
  auto translated =
      translate_incompressible_two_fluid_input(input, system);
  const auto& options = translated.options;
  if (!input.mesh || !translated.velocity_space ||
      !translated.pressure_space) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] incomplete incompressible two-fluid dependency");
  }
  return IncompressibleTwoFluidDependency{
      .mesh = input.mesh.get(),
      .mesh_name = input.mesh_name,
      .dimension = translated.velocity_space->value_dimension(),
      .interface_marker = options.interface_marker,
      .level_set_field_name = options.level_set_field_name,
      .negative_velocity_field_name =
          options.negative_phase.velocity_field_name,
      .positive_velocity_field_name =
          options.positive_phase.velocity_field_name,
      .negative_pressure_field_name =
          options.negative_phase.pressure_field_name,
      .positive_pressure_field_name =
          options.positive_phase.pressure_field_name,
      .operator_tag = options.operator_tag,
      .generated_interface_domain_id =
          options.generated_interface_domain_id,
  };
}

void preRegisterIncompressibleTwoFluidDependencyFields(
    const svmp::Physics::EquationModuleInput& input,
    FE::systems::FESystem& system)
{
  if (validate_and_resolve_free_surface_physical_model(input) !=
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    throw std::invalid_argument(
        "preRegisterIncompressibleTwoFluidDependencyFields requires incompressible two-fluid input");
  }
  auto translated =
      translate_incompressible_two_fluid_input(input, system);
  preflightTwoFluidTranslationBeforeMutation(translated, system);

  const auto& options = translated.options;
  const int dimension = translated.velocity_space->value_dimension();
  const auto u_negative = ensureTwoFluidUnknownField(
      system,
      options.negative_phase.velocity_field_name,
      translated.velocity_space,
      dimension);
  const auto u_positive = ensureTwoFluidUnknownField(
      system,
      options.positive_phase.velocity_field_name,
      translated.velocity_space,
      dimension);
  const auto p_negative = ensureTwoFluidUnknownField(
      system,
      options.negative_phase.pressure_field_name,
      translated.pressure_space,
      1);
  const auto p_positive = ensureTwoFluidUnknownField(
      system,
      options.positive_phase.pressure_field_name,
      translated.pressure_space,
      1);
  (void)p_negative;
  (void)p_positive;

  const auto level_set =
      system.findFieldByName(options.level_set_field_name);
  FE::interfaces::GeneratedInterfaceMarkerKey marker_key{};
  marker_key.source =
      FE::interfaces::LevelSetInterfaceSource::fromField(level_set);
  marker_key.domain_id = options.generated_interface_domain_id;
  marker_key.isovalue = options.level_set_isovalue;
  marker_key.requested_marker = options.interface_marker;
  const int interface_marker =
      FE::interfaces::stableGeneratedInterfaceMarker(marker_key);

  const IncompressibleTwoFluidInterfaceParameters parameters{
      .dimension = dimension,
      .interface_marker = interface_marker,
      .negative_density = options.negative_phase.density,
      .positive_density = options.positive_phase.density,
      .negative_viscosity = options.negative_phase.viscosity,
      .positive_viscosity = options.positive_phase.viscosity,
      .nitsche_gamma = options.interface_nitsche_gamma,
      .surface_tension = options.surface_tension,
      .include_transient_penalty =
          options.include_transient_interface_penalty,
      .prescribed_pressure_jump = options.prescribed_pressure_jump,
      .prescribed_viscous_traction_jump =
          options.prescribed_viscous_traction_jump,
  };
  const auto weights = incompressibleTwoFluidInterfaceWeights(parameters);
  system.declareMaterialInterfaceTransportVelocity(
      FE::interfaces::MaterialInterfaceTransportVelocityDeclaration{
          .dimension = dimension,
          .interface_marker = interface_marker,
          .level_set_field = level_set,
          .negative_velocity_field = u_negative,
          .positive_velocity_field = u_positive,
          .level_set_isovalue = options.level_set_isovalue,
          .negative_trace_weight = weights.negative_complement,
          .positive_trace_weight = weights.positive_complement,
          .geometry_domain_id = options.generated_interface_domain_id,
          .owner_component = "incompressible_two_fluid",
      });
}

FE::FieldId preRegisterPrimaryVelocityField(
    const svmp::Physics::EquationModuleInput& input,
    FE::systems::FESystem& system)
{
  preflightFittedSurfaceContactCapability(input);
  if (validate_and_resolve_free_surface_physical_model(input) ==
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    throw std::invalid_argument(
        "Incompressible two-fluid input owns separate phase velocities and cannot pre-register a field named 'Velocity'");
  }
  if (input.equation_type != "fluid" && input.equation_type != "stokes") {
    throw std::invalid_argument(
        "preRegisterPrimaryVelocityField requires a fluid or stokes equation input");
  }
  if (!input.mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Navier-Stokes velocity pre-registration received null mesh.");
  }

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes velocity pre-registration: " +
        std::to_string(dim));
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int velocity_order =
      resolve_element_order(input, infer_polynomial_order(*input.mesh));
  auto velocity_space = svmp::FE::spaces::VectorSpace(
      svmp::FE::spaces::SpaceType::H1,
      element_type,
      velocity_order,
      dim);

  FE::systems::FieldSpec velocity_spec;
  velocity_spec.name = "Velocity";
  velocity_spec.space = std::move(velocity_space);
  velocity_spec.components = dim;
  const auto existing = system.findFieldByName(velocity_spec.name);
  if (existing == FE::INVALID_FIELD_ID) {
    return system.addField(std::move(velocity_spec));
  }

  const auto& record = system.fieldRecord(existing);
  if (record.source_kind != FE::systems::FieldSourceKind::Unknown ||
      record.components != velocity_spec.components || !record.space ||
      !velocity_spec.space ||
      record.space->space_type() != velocity_spec.space->space_type() ||
      record.space->field_type() != velocity_spec.space->field_type() ||
      record.space->value_dimension() !=
          velocity_spec.space->value_dimension() ||
      record.space->topological_dimension() !=
          velocity_spec.space->topological_dimension() ||
      record.space->polynomial_order() !=
          velocity_spec.space->polynomial_order() ||
      record.space->element_type() != velocity_spec.space->element_type()) {
    throw std::invalid_argument(
        "Navier-Stokes future velocity pre-registration found an incompatible existing field 'Velocity'");
  }
  return existing;
}

void forceLink_NavierStokesRegister() {}

} // namespace svmp::Physics::formulations::navier_stokes
