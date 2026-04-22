#include "LinearAlgebra.h"
#include "consts.h"

#include <map>
#include <string>

const std::map<std::string, consts::LinearAlgebraType> LinearAlgebra::name_to_type = {
    {"none", consts::LinearAlgebraType::none},
    {"fsils", consts::LinearAlgebraType::fsils},
    {"petsc", consts::LinearAlgebraType::petsc},
    {"trilinos", consts::LinearAlgebraType::trilinos},
};

const std::map<consts::LinearAlgebraType, std::string> LinearAlgebra::type_to_name = {
    {consts::LinearAlgebraType::none, "none"},
    {consts::LinearAlgebraType::fsils, "fsils"},
    {consts::LinearAlgebraType::petsc, "petsc"},
    {consts::LinearAlgebraType::trilinos, "trilinos"},
};

void LinearAlgebra::check_equation_compatibility(
    const consts::EquationType,
    const consts::LinearAlgebraType,
    const consts::LinearAlgebraType)
{
}

LinearAlgebra::LinearAlgebra() = default;

LinearAlgebra* LinearAlgebraFactory::create_interface(consts::LinearAlgebraType)
{
  return nullptr;
}

namespace ustruct {

bool constitutive_model_is_valid(consts::ConstitutiveModelType)
{
  return true;
}

} // namespace ustruct
