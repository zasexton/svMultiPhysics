#include "LinearAlgebra.h"
#include "consts.h"

#include <map>
#include <string>

namespace {

class NoopLinearAlgebra final : public LinearAlgebra {
 public:
  explicit NoopLinearAlgebra(consts::LinearAlgebraType interface_type_in)
  {
    interface_type = interface_type_in;
  }

  void alloc(ComMod&, eqType&) override {}

  void assemble(ComMod&,
                const int,
                const Vector<int>&,
                const Array3<double>&,
                const Array<double>&) override {}

  void check_options(const consts::PreconditionerType,
                     const consts::LinearAlgebraType) override {}

  void initialize(ComMod&, eqType&) override {}

  void set_assembly(consts::LinearAlgebraType assembly_type_in) override
  {
    assembly_type = assembly_type_in;
  }

  void set_preconditioner(consts::PreconditionerType prec_type) override
  {
    preconditioner_type = prec_type;
  }

  void solve(ComMod&, eqType&, const Vector<int>&, const Vector<double>&) override {}
};

} // namespace

const std::map<std::string, consts::LinearAlgebraType> LinearAlgebra::name_to_type = {
    {"none", consts::LinearAlgebraType::none},
    {"fsils", consts::LinearAlgebraType::fsils},
    {"petsc", consts::LinearAlgebraType::petsc},
    {"trilinos", consts::LinearAlgebraType::trilinos},
    {"eigen", consts::LinearAlgebraType::eigen},
};

const std::map<consts::LinearAlgebraType, std::string> LinearAlgebra::type_to_name = {
    {consts::LinearAlgebraType::none, "none"},
    {consts::LinearAlgebraType::fsils, "fsils"},
    {consts::LinearAlgebraType::petsc, "petsc"},
    {consts::LinearAlgebraType::trilinos, "trilinos"},
    {consts::LinearAlgebraType::eigen, "eigen"},
};

void LinearAlgebra::check_equation_compatibility(
    const consts::EquationType,
    const consts::LinearAlgebraType,
    const consts::LinearAlgebraType)
{
}

LinearAlgebra::LinearAlgebra() = default;

LinearAlgebra* LinearAlgebraFactory::create_interface(consts::LinearAlgebraType interface_type)
{
  return new NoopLinearAlgebra(interface_type);
}

namespace ustruct {

bool constitutive_model_is_valid(consts::ConstitutiveModelType)
{
  return true;
}

} // namespace ustruct
