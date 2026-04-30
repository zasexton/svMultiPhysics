#include "Coupling/PartitionedCouplingBuilder.h"
#include "Core/FEException.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

CouplingValueDescriptor vectorDescriptor(int components)
{
    return CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = components,
    };
}

CouplingFieldRequirement fieldRequirement(std::string participant,
                                          std::string field,
                                          CouplingValueDescriptor value)
{
    return CouplingFieldRequirement{
        .field = CouplingFieldUse{
            .participant_name = std::move(participant),
            .field_name = std::move(field),
        },
        .value = std::move(value),
    };
}

} // namespace

TEST(PartitionedCouplingBuilder, BuildsFieldExchangeDeclarations)
{
    PartitionedCouplingBuilder builder("fsi_wall");

    CouplingTransferDeclaration transfer;
    transfer.kind = CouplingTransferKind::Identity;

    builder.exchange(
               "solid_displacement",
               CouplingFieldUse{
                   .participant_name = "solid",
                   .field_name = "displacement",
               },
               CouplingFieldUse{
                   .participant_name = "fluid",
                   .field_name = "velocity",
               })
        .sharedInterface("wall")
        .value(vectorDescriptor(3))
        .transfer(transfer)
        .strategy(CouplingPartitionedStrategyDeclaration{
            .solve_strategy =
                CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
            .relaxation_strategy =
                CouplingPartitionedRelaxationStrategy::Constant,
            .convergence_norm =
                CouplingPartitionedConvergenceNorm::ExchangeIncrement,
            .relaxation_factor = 0.5,
            .max_iterations = 4,
        })
        .producerTemporal(CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Accepted,
        })
        .consumerTemporal(CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Current,
        });

    const auto& declarations = builder.declarations();
    ASSERT_EQ(declarations.size(), 1u);
    const auto& exchange = declarations.front();

    EXPECT_EQ(exchange.producer_port.contract_instance_name, "fsi_wall");
    EXPECT_EQ(exchange.producer_port.port_name,
              "solid_displacement.producer");
    EXPECT_EQ(exchange.consumer_port.contract_instance_name, "fsi_wall");
    EXPECT_EQ(exchange.consumer_port.port_name,
              "solid_displacement.consumer");
    ASSERT_TRUE(exchange.producer.has_value());
    EXPECT_EQ(exchange.producer->kind, CouplingEndpointKind::Field);
    ASSERT_TRUE(exchange.producer->participant_name.has_value());
    EXPECT_EQ(*exchange.producer->participant_name, "solid");
    EXPECT_EQ(exchange.producer->endpoint_name, "displacement");
    EXPECT_EQ(exchange.producer->temporal.slot, CouplingTemporalSlot::Accepted);
    ASSERT_TRUE(exchange.consumer.has_value());
    EXPECT_EQ(exchange.consumer->kind, CouplingEndpointKind::Field);
    ASSERT_TRUE(exchange.consumer->participant_name.has_value());
    EXPECT_EQ(*exchange.consumer->participant_name, "fluid");
    EXPECT_EQ(exchange.consumer->endpoint_name, "velocity");
    EXPECT_EQ(exchange.consumer->temporal.slot, CouplingTemporalSlot::Current);
    ASSERT_TRUE(exchange.shared_region_name.has_value());
    EXPECT_EQ(*exchange.shared_region_name, "wall");
    EXPECT_EQ(exchange.value.rank, CouplingValueRank::Vector);
    EXPECT_EQ(exchange.value.components, 3);
    EXPECT_EQ(exchange.transfer.kind, CouplingTransferKind::Identity);
    EXPECT_EQ(exchange.strategy.solve_strategy,
              CouplingPartitionedSolveStrategy::StaggeredFixedPoint);
    EXPECT_EQ(exchange.strategy.relaxation_strategy,
              CouplingPartitionedRelaxationStrategy::Constant);
    EXPECT_EQ(exchange.strategy.convergence_norm,
              CouplingPartitionedConvergenceNorm::ExchangeIncrement);
    EXPECT_EQ(exchange.strategy.max_iterations, 4);
}

TEST(PartitionedCouplingBuilder, BuildsGeneratedExchangePortNames)
{
    PartitionedCouplingBuilder builder("fsi_wall");
    builder
        .exchange(CouplingGeneratedNameRequest{
                      .contract_name = "fsi_wall",
                      .relation_name = "fsi_interface",
                      .local_name = "solid_motion",
                  },
                  CouplingFieldUse{
                      .participant_name = "solid",
                      .field_name = "displacement",
                  },
                  CouplingFieldUse{
                      .participant_name = "fluid",
                      .field_name = "mesh_displacement",
                  })
        .sharedInterface("wall");

    const auto& declarations = builder.declarations();
    ASSERT_EQ(declarations.size(), 1u);
    EXPECT_EQ(declarations.front().producer_port.contract_instance_name,
              "fsi_wall");
    EXPECT_EQ(declarations.front().producer_port.port_name,
              "fsi_wall.fsi_interface.solid_motion.producer");
    EXPECT_EQ(declarations.front().consumer_port.port_name,
              "fsi_wall.fsi_interface.solid_motion.consumer");

    EXPECT_THROW(static_cast<void>(
                     builder.exchange(CouplingGeneratedNameRequest{
                                          .contract_name = "fsi.wall",
                                          .relation_name = "fsi_interface",
                                          .local_name = "bad_motion",
                                      },
                                      CouplingFieldUse{
                                          .participant_name = "solid",
                                          .field_name = "displacement",
                                      },
                                      CouplingFieldUse{
                                          .participant_name = "fluid",
                                          .field_name = "mesh_displacement",
                                      })),
                 InvalidArgumentException);
}

TEST(PartitionedCouplingBuilder, BuildsGenericEndpointExchangeDeclarations)
{
    PartitionedCouplingBuilder builder("junction");

    auto handle =
        builder.exchange(
                   "branch_pressure",
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::AuxiliaryOutput,
                       .participant_name = "junction_state",
                       .endpoint_name = "pressure",
                   },
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::RegionData,
                       .participant_name = "branch",
                       .endpoint_name = "outlet_pressure",
                   })
            .producerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "junction_state",
                .region_name = "node",
            })
            .consumerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "branch",
                .region_name = "outlet",
            })
            .value(vectorDescriptor(2));

    const auto& exchange = handle.declaration();
    ASSERT_TRUE(exchange.producer.has_value());
    EXPECT_EQ(exchange.producer->kind, CouplingEndpointKind::AuxiliaryOutput);
    ASSERT_TRUE(exchange.consumer.has_value());
    EXPECT_EQ(exchange.consumer->kind, CouplingEndpointKind::RegionData);
    ASSERT_TRUE(exchange.producer_region.has_value());
    EXPECT_EQ(exchange.producer_region->region_name, "node");
    ASSERT_TRUE(exchange.consumer_region.has_value());
    EXPECT_EQ(exchange.consumer_region->region_name, "outlet");
    EXPECT_EQ(exchange.value.rank, CouplingValueRank::Vector);
    EXPECT_EQ(exchange.value.components, 2);

    auto declarations = builder.takeDeclarations();
    ASSERT_EQ(declarations.size(), 1u);
    EXPECT_TRUE(builder.declarations().empty());
    EXPECT_EQ(declarations.front().producer_port.port_name,
              "branch_pressure.producer");
}

TEST(PartitionedCouplingBuilder, AllowsExplicitPortNameOverrides)
{
    PartitionedCouplingBuilder builder("fsi_wall");

    builder
        .exchange("solid_motion",
                  CouplingFieldUse{
                      .participant_name = "solid",
                      .field_name = "displacement",
                  },
                  CouplingFieldUse{
                      .participant_name = "fluid",
                      .field_name = "mesh_displacement",
                  })
        .producerPort("solid_displacement")
        .consumerPort("fluid_displacement");

    const auto& declarations = builder.declarations();
    ASSERT_EQ(declarations.size(), 1u);
    EXPECT_EQ(declarations.front().producer_port.contract_instance_name,
              "fsi_wall");
    EXPECT_EQ(declarations.front().producer_port.port_name,
              "solid_displacement");
    EXPECT_EQ(declarations.front().consumer_port.contract_instance_name,
              "fsi_wall");
    EXPECT_EQ(declarations.front().consumer_port.port_name,
              "fluid_displacement");
}

TEST(PartitionedCouplingBuilder, InfersFieldExchangeValueDescriptors)
{
    PartitionedCouplingBuilder builder("thermal");
    builder
        .addFieldRequirement(fieldRequirement(
            "wall",
            "temperature",
            CouplingValueDescriptor{
                .rank = CouplingValueRank::Scalar,
                .components = 1,
            }))
        .addFieldRequirement(fieldRequirement(
            "solid",
            "temperature",
            CouplingValueDescriptor{
                .rank = CouplingValueRank::Scalar,
                .components = 1,
            }))
        .addFieldRequirement(fieldRequirement(
            "fluid",
            "velocity",
            vectorDescriptor(3)))
        .addFieldRequirement(fieldRequirement(
            "mesh",
            "velocity",
            vectorDescriptor(3)));

    static_cast<void>(builder.exchange(
        "temperature",
        CouplingFieldUse{
            .participant_name = "wall",
            .field_name = "temperature",
        },
        CouplingFieldUse{
            .participant_name = "solid",
            .field_name = "temperature",
        }));
    static_cast<void>(builder.exchange(
        "velocity",
        CouplingFieldUse{
            .participant_name = "fluid",
            .field_name = "velocity",
        },
        CouplingFieldUse{
            .participant_name = "mesh",
            .field_name = "velocity",
        }));

    const auto& declarations = builder.declarations();
    ASSERT_EQ(declarations.size(), 2u);
    EXPECT_EQ(declarations[0].value.rank, CouplingValueRank::Scalar);
    EXPECT_EQ(declarations[0].value.components, 1);
    EXPECT_EQ(declarations[1].value.rank, CouplingValueRank::Vector);
    EXPECT_EQ(declarations[1].value.components, 3);
}

TEST(PartitionedCouplingBuilder, RejectsIncompatibleInferredValueDescriptors)
{
    PartitionedCouplingBuilder builder("bad_exchange");
    builder
        .addFieldRequirement(fieldRequirement(
            "left",
            "primary",
            vectorDescriptor(3)))
        .addFieldRequirement(fieldRequirement(
            "right",
            "primary",
            CouplingValueDescriptor{
                .rank = CouplingValueRank::Scalar,
                .components = 1,
            }));

    EXPECT_THROW(static_cast<void>(
                     builder.exchange(
                         "primary",
                         CouplingFieldUse{
                             .participant_name = "left",
                             .field_name = "primary",
                         },
                         CouplingFieldUse{
                             .participant_name = "right",
                             .field_name = "primary",
                         })),
                 InvalidArgumentException);
}

TEST(PartitionedCouplingBuilder, BuildsGroupHints)
{
    PartitionedCouplingBuilder builder("junction");
    builder.group("outlet_branches", {"branch_a", "branch_b", "branch_c"});

    const auto& hints = builder.groupHints();
    ASSERT_EQ(hints.size(), 1u);
    EXPECT_EQ(hints.front().name, "outlet_branches");
    ASSERT_EQ(hints.front().participant_names.size(), 3u);
    EXPECT_EQ(hints.front().participant_names[0], "branch_a");
    EXPECT_EQ(hints.front().participant_names[1], "branch_b");
    EXPECT_EQ(hints.front().participant_names[2], "branch_c");

    auto moved = builder.takeGroupHints();
    ASSERT_EQ(moved.size(), 1u);
    EXPECT_TRUE(builder.groupHints().empty());
    EXPECT_EQ(moved.front().name, "outlet_branches");
}

TEST(PartitionedCouplingBuilder, RejectsMissingContractOrExchangeNames)
{
    EXPECT_THROW(static_cast<void>(PartitionedCouplingBuilder("")),
                 InvalidArgumentException);

    PartitionedCouplingBuilder builder("contract");
    EXPECT_THROW(static_cast<void>(
                     builder.exchange(
                         "",
                         CouplingFieldUse{
                             .participant_name = "left",
                             .field_name = "primary",
                         },
                         CouplingFieldUse{
                             .participant_name = "right",
                             .field_name = "primary",
                         })),
                 InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(builder.group("", {"left", "right"})),
                 InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(builder.group("empty", {})),
                 InvalidArgumentException);
}
