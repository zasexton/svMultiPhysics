/**
 * @file test_AuxiliaryInputRegistry.cpp
 * @brief Unit tests for AuxiliaryInputRegistry — generalized auxiliary input infrastructure
 */

#include <gtest/gtest.h>

#include "Systems/AuxiliaryInputRegistry.h"

#include <cmath>
#include <string>
#include <vector>

using svmp::FE::Real;
using namespace svmp::FE::systems;

// ---------------------------------------------------------------------------
//  Helper: make a simple scalar input spec
// ---------------------------------------------------------------------------

static AuxiliaryInputSpec makeScalarInput(const std::string& name,
                                           AuxiliaryInputProducer producer =
                                               AuxiliaryInputProducer::DirectUserData)
{
    AuxiliaryInputSpec spec;
    spec.name = name;
    spec.size = 1;
    spec.producer = producer;
    return spec;
}

// ---------------------------------------------------------------------------
//  Registration and access
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, RegisterAndAccess)
{
    AuxiliaryInputRegistry reg;

    auto slot = reg.registerInput(makeScalarInput("Q"));
    EXPECT_EQ(slot, 0u);
    EXPECT_EQ(reg.inputCount(), 1u);
    EXPECT_EQ(reg.totalSize(), 1u);
    EXPECT_TRUE(reg.hasInput("Q"));
    EXPECT_FALSE(reg.hasInput("P"));
}

TEST(AuxiliaryInputRegistry, MultiComponentInput)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "vel";
    spec.size = 3;
    spec.component_names = {"vx", "vy", "vz"};
    spec.producer = AuxiliaryInputProducer::SampledStateField;

    auto slot = reg.registerInput(spec);
    EXPECT_EQ(slot, 0u);
    EXPECT_EQ(reg.totalSize(), 3u);

    auto vals = reg.valuesOf("vel");
    EXPECT_EQ(vals.size(), 3u);
}

TEST(AuxiliaryInputRegistry, MultipleInputsSlotAssignment)
{
    AuxiliaryInputRegistry reg;

    auto s1 = reg.registerInput(makeScalarInput("A"));
    AuxiliaryInputSpec bspec;
    bspec.name = "B";
    bspec.size = 3;
    auto s2 = reg.registerInput(bspec);
    auto s3 = reg.registerInput(makeScalarInput("C"));

    EXPECT_EQ(s1, 0u);
    EXPECT_EQ(s2, 1u); // after A's 1 value
    EXPECT_EQ(s3, 4u); // after B's 3 values
    EXPECT_EQ(reg.totalSize(), 5u);

    EXPECT_EQ(reg.slotOf("A"), 0u);
    EXPECT_EQ(reg.slotOf("B"), 1u);
    EXPECT_EQ(reg.slotOf("C"), 4u);
}

TEST(AuxiliaryInputRegistry, DuplicateNameThrows)
{
    AuxiliaryInputRegistry reg;
    reg.registerInput(makeScalarInput("Q"));
    EXPECT_THROW(reg.registerInput(makeScalarInput("Q")),
                 svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryInputRegistry, GetSetScalar)
{
    AuxiliaryInputRegistry reg;
    reg.registerInput(makeScalarInput("Q"));

    EXPECT_DOUBLE_EQ(reg.get("Q"), 0.0);

    reg.set("Q", 42.0);
    EXPECT_DOUBLE_EQ(reg.get("Q"), 42.0);
}

TEST(AuxiliaryInputRegistry, MutableValuesOfWritesThrough)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "v";
    spec.size = 2;
    reg.registerInput(spec);

    auto mv = reg.mutableValuesOf("v");
    mv[0] = 10.0;
    mv[1] = 20.0;

    auto rv = reg.valuesOf("v");
    EXPECT_DOUBLE_EQ(rv[0], 10.0);
    EXPECT_DOUBLE_EQ(rv[1], 20.0);
}

TEST(AuxiliaryInputRegistry, AllValuesFlat)
{
    AuxiliaryInputRegistry reg;
    reg.registerInput(makeScalarInput("A"));
    reg.registerInput(makeScalarInput("B"));

    reg.set("A", 1.0);
    reg.set("B", 2.0);

    auto all = reg.all();
    ASSERT_EQ(all.size(), 2u);
    EXPECT_DOUBLE_EQ(all[0], 1.0);
    EXPECT_DOUBLE_EQ(all[1], 2.0);
}

TEST(AuxiliaryInputRegistry, InputNames)
{
    AuxiliaryInputRegistry reg;
    reg.registerInput(makeScalarInput("alpha"));
    reg.registerInput(makeScalarInput("beta"));
    reg.registerInput(makeScalarInput("gamma"));

    auto names = reg.inputNames();
    ASSERT_EQ(names.size(), 3u);
    EXPECT_EQ(names[0], "alpha");
    EXPECT_EQ(names[1], "beta");
    EXPECT_EQ(names[2], "gamma");
}

TEST(AuxiliaryInputRegistry, SpecRetrieval)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "Q";
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.boundary_marker = 3;
    spec.requires_mpi_reduction = true;
    reg.registerInput(spec);

    const auto& retrieved = reg.specOf("Q");
    EXPECT_EQ(retrieved.name, "Q");
    EXPECT_EQ(retrieved.producer, AuxiliaryInputProducer::BoundaryReduction);
    EXPECT_EQ(retrieved.boundary_marker, 3);
    EXPECT_TRUE(retrieved.requires_mpi_reduction);
}

// ---------------------------------------------------------------------------
//  Producer types coverage
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, AllProducerTypesRegisterable)
{
    AuxiliaryInputRegistry reg;

    auto mkSpec = [](const std::string& name, AuxiliaryInputProducer p) {
        AuxiliaryInputSpec s;
        s.name = name;
        s.size = 1;
        s.producer = p;
        return s;
    };

    reg.registerInput(mkSpec("a", AuxiliaryInputProducer::BoundaryReduction));
    reg.registerInput(mkSpec("b", AuxiliaryInputProducer::FormulationCallback));
    reg.registerInput(mkSpec("c", AuxiliaryInputProducer::ParameterDerived));
    reg.registerInput(mkSpec("d", AuxiliaryInputProducer::DirectUserData));
    reg.registerInput(mkSpec("e", AuxiliaryInputProducer::AuxiliaryOutput));
    reg.registerInput(mkSpec("f", AuxiliaryInputProducer::SampledStateField));
    reg.registerInput(mkSpec("g", AuxiliaryInputProducer::CoupledField));
    reg.registerInput(mkSpec("h", AuxiliaryInputProducer::CellAverage));
    reg.registerInput(mkSpec("i", AuxiliaryInputProducer::CellSample));
    reg.registerInput(mkSpec("j", AuxiliaryInputProducer::DomainAverage));
    reg.registerInput(mkSpec("k", AuxiliaryInputProducer::DomainIntegral));
    reg.registerInput(mkSpec("l", AuxiliaryInputProducer::SampledBoundaryTrace));
    reg.registerInput(mkSpec("m", AuxiliaryInputProducer::CoupledBoundaryTrace));
    reg.registerInput(mkSpec("n", AuxiliaryInputProducer::SampledBoundaryReduction));
    reg.registerInput(mkSpec("o", AuxiliaryInputProducer::CoupledBoundaryReduction));

    EXPECT_EQ(reg.inputCount(), 15u);
}

// ---------------------------------------------------------------------------
//  Evaluation lifecycle
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, CallbackEvaluation)
{
    AuxiliaryInputRegistry reg;

    auto spec = makeScalarInput("Q", AuxiliaryInputProducer::BoundaryReduction);
    spec.update_schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep;

    int call_count = 0;
    reg.registerInput(spec, [&](Real time, Real dt, std::span<Real> out) {
        out[0] = time * 10.0;
        call_count++;
    });

    reg.evaluate(1.0, 0.1);
    EXPECT_EQ(call_count, 1);
    EXPECT_DOUBLE_EQ(reg.get("Q"), 10.0);

    // Second eval without invalidation: already clean, skip.
    reg.evaluate(2.0, 0.1);
    EXPECT_EQ(call_count, 1);

    // Invalidate and re-evaluate.
    reg.invalidateAll();
    reg.evaluate(2.0, 0.1);
    EXPECT_EQ(call_count, 2);
    EXPECT_DOUBLE_EQ(reg.get("Q"), 20.0);
}

TEST(AuxiliaryInputRegistry, OnceAtSetupSchedule)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec = makeScalarInput("const_param");
    spec.update_schedule = AuxiliaryInputUpdateSchedule::OnceAtSetup;

    int call_count = 0;
    reg.registerInput(spec, [&](Real, Real, std::span<Real> out) {
        out[0] = 42.0;
        call_count++;
    });

    reg.evaluate(0.0, 0.0);
    EXPECT_EQ(call_count, 1);
    EXPECT_DOUBLE_EQ(reg.get("const_param"), 42.0);

    // Subsequent evaluations are skipped (even after invalidateAll).
    reg.invalidateAll();
    reg.evaluate(1.0, 0.1);
    EXPECT_EQ(call_count, 1); // Not called again
}

TEST(AuxiliaryInputRegistry, EachNonlinearIterationSchedule)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec = makeScalarInput("coupled");
    spec.update_schedule = AuxiliaryInputUpdateSchedule::EachNonlinearIteration;

    int call_count = 0;
    reg.registerInput(spec, [&](Real, Real, std::span<Real> out) {
        out[0] = static_cast<Real>(call_count + 1);
        call_count++;
    });

    // First eval
    reg.evaluate(0.0, 0.1);
    EXPECT_EQ(call_count, 1);

    // Nonlinear iteration: re-evaluates even when not dirty.
    reg.evaluate(0.0, 0.1, /*is_nonlinear_iteration=*/true);
    EXPECT_EQ(call_count, 2);
}

TEST(AuxiliaryInputRegistry, ManualSchedule)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec = makeScalarInput("manual");
    spec.update_schedule = AuxiliaryInputUpdateSchedule::Manual;

    int call_count = 0;
    reg.registerInput(spec, [&](Real, Real, std::span<Real> out) {
        out[0] = 99.0;
        call_count++;
    });

    // First eval: dirty by default, so callback runs.
    reg.evaluate(0.0, 0.1);
    EXPECT_EQ(call_count, 1);

    // Not dirty anymore.
    reg.evaluate(0.0, 0.1);
    EXPECT_EQ(call_count, 1);

    // Mark dirty explicitly.
    reg.markDirty("manual");
    reg.evaluate(0.0, 0.1);
    EXPECT_EQ(call_count, 2);
}

// ---------------------------------------------------------------------------
//  Dependency ordering
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, DependencyOrdering)
{
    AuxiliaryInputRegistry reg;

    reg.registerInput(makeScalarInput("C")); // index 0
    reg.registerInput(makeScalarInput("A")); // index 1
    reg.registerInput(makeScalarInput("B")); // index 2

    // C depends on A and B.  A depends on B.
    reg.addDependency("C", "A");
    reg.addDependency("C", "B");
    reg.addDependency("A", "B");

    auto order = reg.evaluationOrder();
    ASSERT_EQ(order.size(), 3u);

    // B (index 2) must come before A (index 1), which must come before C (index 0).
    std::size_t pos_B = 0, pos_A = 0, pos_C = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (order[i] == 0) pos_C = i;
        if (order[i] == 1) pos_A = i;
        if (order[i] == 2) pos_B = i;
    }
    EXPECT_LT(pos_B, pos_A);
    EXPECT_LT(pos_A, pos_C);
}

TEST(AuxiliaryInputRegistry, DependencyOrderingAffectsEvaluation)
{
    AuxiliaryInputRegistry reg;

    std::vector<std::string> eval_order;

    reg.registerInput(makeScalarInput("C"), [&](Real, Real, std::span<Real> out) {
        eval_order.push_back("C");
        out[0] = 30.0;
    });
    reg.registerInput(makeScalarInput("A"), [&](Real, Real, std::span<Real> out) {
        eval_order.push_back("A");
        out[0] = 10.0;
    });

    reg.addDependency("C", "A");

    reg.evaluate(0.0, 0.1);

    ASSERT_EQ(eval_order.size(), 2u);
    EXPECT_EQ(eval_order[0], "A"); // A before C
    EXPECT_EQ(eval_order[1], "C");
}

// ---------------------------------------------------------------------------
//  AuxiliaryOutput producer (input from another model's output)
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, AuxiliaryOutputProducer)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "model_output";
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::AuxiliaryOutput;
    spec.source_output_name = "ionic_model.voltage";

    reg.registerInput(spec, [](Real, Real, std::span<Real> out) {
        out[0] = -80.0; // simulated output
    });

    reg.evaluate(0.0, 0.1);
    EXPECT_DOUBLE_EQ(reg.get("model_output"), -80.0);
    EXPECT_EQ(reg.specOf("model_output").source_output_name, "ionic_model.voltage");
}

// ---------------------------------------------------------------------------
//  Field stage selection
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, FieldStageSelection)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "field_sample";
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::SampledStateField;
    spec.field_stage = AuxiliaryFieldStage::Committed;
    reg.registerInput(spec);

    EXPECT_EQ(reg.specOf("field_sample").field_stage, AuxiliaryFieldStage::Committed);
}

// ---------------------------------------------------------------------------
//  Clear
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, Clear)
{
    AuxiliaryInputRegistry reg;
    reg.registerInput(makeScalarInput("A"));
    reg.registerInput(makeScalarInput("B"));
    reg.set("A", 1.0);

    reg.clear();
    EXPECT_EQ(reg.inputCount(), 0u);
    EXPECT_EQ(reg.totalSize(), 0u);
    EXPECT_FALSE(reg.hasInput("A"));
}

// ---------------------------------------------------------------------------
//  Debug inspection
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, DebugDump)
{
    AuxiliaryInputRegistry reg;
    reg.registerInput(makeScalarInput("Q"));
    reg.set("Q", 3.14);

    auto dump = reg.debugDump();
    EXPECT_NE(dump.find("Q"), std::string::npos);
    EXPECT_NE(dump.find("3.14"), std::string::npos);
}

// ---------------------------------------------------------------------------
//  Boundary-specific inputs
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, BoundaryReductionInput)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "flow_rate";
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.boundary_marker = 5;
    spec.requires_mpi_reduction = true;
    spec.update_schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep;

    reg.registerInput(spec, [](Real, Real, std::span<Real> out) {
        out[0] = 100.0; // simulated boundary integral
    });

    reg.evaluate(0.0, 0.1);
    EXPECT_DOUBLE_EQ(reg.get("flow_rate"), 100.0);
    EXPECT_EQ(reg.specOf("flow_rate").boundary_marker, 5);
    EXPECT_TRUE(reg.specOf("flow_rate").requires_mpi_reduction);
}

TEST(AuxiliaryInputRegistry, SampledBoundaryReductionInput)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "pressure_avg";
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::SampledBoundaryReduction;
    spec.boundary_marker = 2;
    reg.registerInput(spec);

    EXPECT_EQ(reg.specOf("pressure_avg").producer,
              AuxiliaryInputProducer::SampledBoundaryReduction);
}

// ---------------------------------------------------------------------------
//  Parameter-derived and callback providers
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, ParameterDerivedInput)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "viscosity";
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::ParameterDerived;
    spec.update_schedule = AuxiliaryInputUpdateSchedule::OnceAtSetup;

    reg.registerInput(spec, [](Real, Real, std::span<Real> out) {
        out[0] = 0.04; // constant parameter
    });

    reg.evaluate(0.0, 0.0);
    EXPECT_DOUBLE_EQ(reg.get("viscosity"), 0.04);
}

// ---------------------------------------------------------------------------
//  Block-valued inputs (size > 1)
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, BlockValuedInput)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "coupling_state";
    spec.size = 4;
    spec.component_names = {"P_d", "P_p", "Q_d", "Q_p"};
    spec.producer = AuxiliaryInputProducer::AuxiliaryOutput;

    reg.registerInput(spec, [](Real, Real, std::span<Real> out) {
        out[0] = 1.0;
        out[1] = 2.0;
        out[2] = 3.0;
        out[3] = 4.0;
    });

    reg.evaluate(0.0, 0.1);

    auto vals = reg.valuesOf("coupling_state");
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_DOUBLE_EQ(vals[0], 1.0);
    EXPECT_DOUBLE_EQ(vals[3], 4.0);
}

// ---------------------------------------------------------------------------
//  Entity-local inputs
// ---------------------------------------------------------------------------

TEST(AuxiliaryInputRegistry, EntityLocalRegistration)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "node_voltage";
    spec.size = 1;
    spec.entity_count = 5; // 5 nodes
    spec.producer = AuxiliaryInputProducer::SampledStateField;

    auto slot = reg.registerEntityInput(spec,
        [](Real, Real, std::size_t entity_idx, std::span<Real> out) {
            out[0] = static_cast<Real>(entity_idx) * 10.0;
        });

    EXPECT_EQ(slot, 0u);
    EXPECT_EQ(reg.totalSize(), 5u); // 5 entities × 1 component
    EXPECT_TRUE(reg.isEntityLocal("node_voltage"));
}

TEST(AuxiliaryInputRegistry, EntityLocalEvaluation)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "V";
    spec.size = 1;
    spec.entity_count = 3;

    reg.registerEntityInput(spec,
        [](Real, Real, std::size_t e, std::span<Real> out) {
            out[0] = static_cast<Real>(e + 1) * 100.0; // 100, 200, 300
        });

    reg.evaluate(0.0, 0.1);

    // Full view
    auto all = reg.valuesOf("V");
    ASSERT_EQ(all.size(), 3u);
    EXPECT_DOUBLE_EQ(all[0], 100.0);
    EXPECT_DOUBLE_EQ(all[1], 200.0);
    EXPECT_DOUBLE_EQ(all[2], 300.0);

    // Per-entity access
    auto e0 = reg.valuesOf("V", 0);
    ASSERT_EQ(e0.size(), 1u);
    EXPECT_DOUBLE_EQ(e0[0], 100.0);

    auto e2 = reg.valuesOf("V", 2);
    EXPECT_DOUBLE_EQ(e2[0], 300.0);
}

TEST(AuxiliaryInputRegistry, EntityLocalMultiComponent)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "stress";
    spec.size = 3;
    spec.entity_count = 2;

    reg.registerEntityInput(spec,
        [](Real, Real, std::size_t e, std::span<Real> out) {
            out[0] = static_cast<Real>(e) * 1.0;
            out[1] = static_cast<Real>(e) * 2.0;
            out[2] = static_cast<Real>(e) * 3.0;
        });

    reg.evaluate(0.0, 0.1);

    // Entity 1: [1.0, 2.0, 3.0]
    auto e1 = reg.valuesOf("stress", 1);
    ASSERT_EQ(e1.size(), 3u);
    EXPECT_DOUBLE_EQ(e1[0], 1.0);
    EXPECT_DOUBLE_EQ(e1[1], 2.0);
    EXPECT_DOUBLE_EQ(e1[2], 3.0);

    // Total storage
    EXPECT_EQ(reg.totalSize(), 6u);
}

TEST(AuxiliaryInputRegistry, EntityLocalOutOfRangeThrows)
{
    AuxiliaryInputRegistry reg;

    AuxiliaryInputSpec spec;
    spec.name = "V";
    spec.size = 1;
    spec.entity_count = 3;
    reg.registerEntityInput(spec, [](Real, Real, std::size_t, std::span<Real>) {});

    EXPECT_THROW(reg.valuesOf("V", 3), svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryInputRegistry, GlobalInputIgnoresEntityIndex)
{
    AuxiliaryInputRegistry reg;

    reg.registerInput(makeScalarInput("Q"));
    reg.set("Q", 42.0);

    // Global input: entity_index is ignored.
    EXPECT_FALSE(reg.isEntityLocal("Q"));
    auto val = reg.valuesOf("Q", 999);
    EXPECT_DOUBLE_EQ(val[0], 42.0);
}

TEST(AuxiliaryInputRegistry, MixedGlobalAndEntityLocal)
{
    AuxiliaryInputRegistry reg;

    // Global input first
    reg.registerInput(makeScalarInput("k"));
    reg.set("k", 5.0);

    // Entity-local input second
    AuxiliaryInputSpec spec;
    spec.name = "V";
    spec.size = 1;
    spec.entity_count = 2;
    reg.registerEntityInput(spec,
        [](Real, Real, std::size_t e, std::span<Real> out) {
            out[0] = static_cast<Real>(e + 1);
        });

    reg.evaluate(0.0, 0.1);

    // Global: slot 0
    EXPECT_DOUBLE_EQ(reg.get("k"), 5.0);
    // Entity-local: slots 1-2
    EXPECT_DOUBLE_EQ(reg.valuesOf("V", 0)[0], 1.0);
    EXPECT_DOUBLE_EQ(reg.valuesOf("V", 1)[0], 2.0);
}
