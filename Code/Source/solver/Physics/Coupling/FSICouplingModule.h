/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_COUPLING_FSI_COUPLING_MODULE_H
#define SVMP_PHYSICS_COUPLING_FSI_COUPLING_MODULE_H

/**
 * @file FSICouplingModule.h
 * @brief Physics-side FSI coupling contract declaration.
 */

#include "FE/Coupling/CouplingContract.h"
#include "FE/Spaces/FunctionSpace.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace coupling {

struct FSILagrangeMultiplierOptions {
    bool enabled{false};
    std::string contract_field_namespace;
    std::optional<std::string> system_participant_name;
    std::string field_name{"lambda"};
    std::shared_ptr<const FE::spaces::FunctionSpace> space;
    int components{0};
    std::optional<std::string> shared_region_name;
    bool require_mortar_compatible_trace{true};
};

struct FSICouplingOptions {
    FE::coupling::CouplingMode mode{FE::coupling::CouplingMode::Monolithic};
    std::string contract_name{"fsi"};
    std::string fluid_name{"fluid"};
    std::string solid_name{"solid"};
    std::optional<std::string> mesh_name;
    std::string interface_name{"interface"};

    std::string fluid_velocity_field{"velocity"};
    std::string fluid_pressure_field{"pressure"};
    std::string solid_displacement_field{"displacement"};
    std::optional<std::string> solid_velocity_field{std::string{"velocity"}};
    std::optional<std::string> mesh_displacement_field{std::string{"displacement"}};
    bool use_solid_displacement_derivative{false};
    int interface_components{3};

    FSILagrangeMultiplierOptions multiplier{};
    FE::coupling::CouplingTransferDeclaration solid_to_fluid_transfer{};
    FE::coupling::CouplingTransferDeclaration fluid_to_solid_transfer{};
};

class FSICouplingModule final : public FE::coupling::CouplingContract {
public:
    explicit FSICouplingModule(FSICouplingOptions options = {});

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] FE::coupling::CouplingContractDeclaration declare() const override;

    void validate(const FE::coupling::CouplingContext& ctx) const override;

    [[nodiscard]] std::vector<FE::coupling::CouplingExchangeDeclaration>
    buildPartitionedExchangeDeclarations(const FE::coupling::CouplingContext& ctx) const override;

private:
    FSICouplingOptions options_{};
};

} // namespace coupling
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_COUPLING_FSI_COUPLING_MODULE_H
