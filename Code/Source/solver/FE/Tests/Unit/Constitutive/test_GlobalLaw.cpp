/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constitutive/GlobalLaw.h"
#include "Constitutive/ModelCRTP.h"

#include "Systems/FESystem.h"

#include "Sparsity/SparsityPattern.h"

#include "Spaces/H1Space.h"

#include "Dofs/DofHandler.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>

namespace svmp {
namespace FE {
namespace constitutive {
namespace test {

class IdentityModel final : public ModelCRTP<IdentityModel> {
public:
    template <class Scalar, class Workspace>
    [[nodiscard]] forms::Value<Scalar> evaluateImpl(const forms::Value<Scalar>& input,
                                                    int /*dim*/,
                                                    Workspace& /*workspace*/) const
    {
        return input;
    }
};

class ToyGlobalLaw final : public GlobalLaw {
public:
    ToyGlobalLaw(Real vec_value, Real mat_value, std::shared_ptr<const forms::ConstitutiveModel> model)
        : vec_value_(vec_value), mat_value_(mat_value), model_(std::move(model))
    {
    }

    [[nodiscard]] std::string name() const override { return "ToyGlobalLaw"; }

    [[nodiscard]] std::vector<std::shared_ptr<const forms::ConstitutiveModel>> pointwiseModels() const override
    {
        if (!model_) return {};
        return {model_};
    }

    void addSparsityCouplings(const systems::FESystem& /*system*/,
                              sparsity::SparsityPattern& pattern) const override
    {
        pattern.addEntry(0, 1);
    }

    [[nodiscard]] assembly::AssemblyResult assemble(const systems::FESystem& /*system*/,
                                                    const systems::AssemblyRequest& request,
                                                    const systems::SystemStateView& /*state*/,
                                                    assembly::GlobalSystemView* matrix_out,
                                                    assembly::GlobalSystemView* vector_out) const override
    {
        assembly::AssemblyResult r;
        if (request.want_vector && vector_out != nullptr) {
            vector_out->addVectorEntry(0, vec_value_);
        }
        if (request.want_matrix && matrix_out != nullptr) {
            matrix_out->addMatrixEntry(0, 1, mat_value_);
        }
        return r;
    }

private:
    Real vec_value_{0.0};
    Real mat_value_{0.0};
    std::shared_ptr<const forms::ConstitutiveModel> model_{};
};

static dofs::MeshTopologyInfo makeSingleTetraTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

TEST(GlobalLawTest, EmitsPointwiseModelsAndGlobalKernel)
{
    auto model = std::make_shared<IdentityModel>();
    auto law = std::make_shared<ToyGlobalLaw>(1.0, 2.0, model);

    const auto emitted = law->emit();
    ASSERT_EQ(emitted.pointwise_models.size(), 1u);
    ASSERT_EQ(emitted.global_kernels.size(), 1u);
    EXPECT_EQ(emitted.global_kernels[0]->name(), "ToyGlobalLaw");
}

TEST(GlobalLawTest, AssemblesAndAugmentsSparsityViaAdapter)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    systems::FESystem system(mesh);

    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    systems::FieldSpec field;
    field.name = "u";
    field.space = space;
    field.components = 1;
    (void)system.addField(field);

    system.addOperator("op");

    auto model = std::make_shared<IdentityModel>();
    auto law = std::make_shared<ToyGlobalLaw>(3.0, 4.0, model);
    installGlobalLawKernels(system, "op", law);

    systems::SetupInputs inputs;
    inputs.topology_override = makeSingleTetraTopology();
    system.setup({}, inputs);

    EXPECT_TRUE(system.sparsity("op").hasEntry(0, 1));

    assembly::DenseMatrixView jac(system.dofHandler().getNumDofs());
    assembly::DenseVectorView rhs(system.dofHandler().getNumDofs());
    systems::SystemStateView state;

    systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.zero_outputs = true;

    (void)system.assemble(req, state, &jac, &rhs);

    EXPECT_DOUBLE_EQ(rhs.getVectorEntry(0), 3.0);
    EXPECT_DOUBLE_EQ(jac.getMatrixEntry(0, 1), 4.0);
}

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp

