#pragma once

#include "Assembly/TimeIntegrationContext.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/VertexFieldInitialization.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

namespace svmp::FE::test {

// Every rank integrates the same reference triangle with independently owned
// backend rows. This isolates cell gathers from inter-rank residual assembly.
template <typename MakeVector>
void checkDistributedFieldCoefficients(int rank, int ranks, MakeVector make_vector)
{
    const GlobalIndex first = 3 * rank / ranks;
    const GlobalIndex last = 3 * (rank + 1) / ranks;
    std::vector<GlobalIndex> ghosts;
    for (GlobalIndex row = 0; row < 3; ++row) {
        if (row < first || row >= last) ghosts.push_back(row);
    }
    auto current = make_vector(first, last - first, ghosts);
    auto previous = make_vector(first, last - first, ghosts);
    auto older = make_vector(first, last - first, ghosts);
    auto base = std::make_shared<MeshBase>();
    CellShape triangle{};
    triangle.family = CellFamily::Triangle;
    triangle.num_corners = 3;
    triangle.order = 1;
    base->build_from_arrays(2, std::vector<real_t>{0, 0, 1, 0, 0, 1},
                            std::vector<offset_t>{0, 3}, std::vector<index_t>{0, 1, 2}, {triangle});
    base->finalize();
    auto mesh = create_mesh(std::move(base), MeshComm::self());
    systems::FESystem system(mesh);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    const auto field = system.addField({.name = "c", .space = space});
    system.addOperator("op");
    const auto c = forms::FormExpr::trialFunction(*space, "c");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    const auto residual = (forms::dt(c) * v + forms::inner(forms::grad(c), forms::grad(v))).dx();
    auto ir = forms::FormCompiler{}.compileResidual(residual);
    system.addCellKernel("op", field, field,
        std::make_shared<forms::NonlinearFormKernel>(std::move(ir), forms::ADMode::Forward));
    system.setup();

    current->set(7);
    previous->set(4);
    older->set(1);
    current->updateGhosts();
    previous->updateGhosts();
    older->updateGhosts();
    assembly::TimeIntegrationContext time;
    time.dt1 = assembly::TimeDerivativeStencil{.order = 1, .a = {1.5, -2.0, 0.5}};
    systems::SystemStateView state;
    state.u_vector = current.get();
    state.u_prev_vector = previous.get();
    state.u_prev2_vector = older.get();
    state.time_integration = &time;
    assembly::DenseVectorView result(3);
    systems::AssemblyRequest request;
    request.op = "op";
    request.want_vector = true;
    const auto assembled = system.assemble(request, state, nullptr, &result);
    EXPECT_TRUE(assembled.success) << assembled.error_message;
    // grad(c) = 0 and dc/dt = 3. Each linear test function integrates to 1/6.
    for (GlobalIndex row = 0; row < 3; ++row) {
        EXPECT_NEAR(result.getVectorEntry(row), 0.5, 2e-14);
    }

    system.searchAccess()->build();
    const auto value = system.evaluateFieldAtPoint(field, state, {0.2, 0.3, 0});
    EXPECT_TRUE(value.has_value());
    if (value) EXPECT_DOUBLE_EQ((*value)[0], 7);
    std::array<double, 3> nodal{};
    EXPECT_TRUE(system.evaluateFieldAtVertices(field, state, 3, nodal));
    EXPECT_EQ(nodal, (std::array<double, 3>{7, 7, 7}));

    EXPECT_EQ(systems::initializeFieldFromMeshVertexField(system, *current, field, "c", 11),
              current->ownedGlobalRows().size());
    current->updateGhosts();
    auto view = current->createGhostedReadView();
    for (GlobalIndex row = 0; row < 3; ++row) {
        EXPECT_DOUBLE_EQ(view->getVectorEntry(row), 11);
    }

    // A rank owning no rows still participates in the vector finalization.
    const bool empty = current->ownedGlobalRows().empty();
    const std::vector<GlobalIndex> rows = empty ? std::vector<GlobalIndex>{}
                                                : std::vector<GlobalIndex>{0, 1, 2};
    const std::vector<Real> values = empty ? std::vector<Real>{} : std::vector<Real>{2, 3, 4};
    systems::writeVectorEntries(*current, rows, values, "no entries", "no view");
    current->updateGhosts();
    for (GlobalIndex row = 0; row < 3; ++row) {
        EXPECT_DOUBLE_EQ(view->getVectorEntry(row), 2 + row);
    }
}

} // namespace svmp::FE::test
