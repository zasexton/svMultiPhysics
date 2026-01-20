#include "Application/Translators/MaterialTranslator.h"

#include "Parameters.h"
#include "Physics/Formulations/Poisson/PoissonModule.h"

#include <stdexcept>

namespace application {
namespace translators {

void MaterialTranslator::applyThermalProperties(
    const std::vector<const DomainParameters*>& domain_params,
    svmp::Physics::formulations::poisson::PoissonOptions& options)
{
  using svmp::Physics::formulations::poisson::PoissonOptions;

  const DomainParameters* domain = nullptr;
  for (const auto* d : domain_params) {
    if (d) {
      if (domain != nullptr) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Multiple <Domain> blocks are not supported for the new solver Poisson module "
            "yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      domain = d;
    }
  }

  if (!domain) {
    return;
  }

  if (domain->anisotropic_conductivity.defined() && !domain->anisotropic_conductivity.value().empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Anisotropic_conductivity is not supported for the new solver Poisson module yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  if (domain->conductivity.defined()) {
    options.diffusion = static_cast<svmp::FE::Real>(domain->conductivity.value());
  } else if (domain->isotropic_conductivity.defined()) {
    options.diffusion = static_cast<svmp::FE::Real>(domain->isotropic_conductivity.value());
  }

  if (domain->source_term.defined()) {
    options.source = static_cast<svmp::FE::Real>(domain->source_term.value());
  }
}

} // namespace translators
} // namespace application
