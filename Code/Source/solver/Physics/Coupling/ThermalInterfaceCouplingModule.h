/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_COUPLING_THERMAL_INTERFACE_COUPLING_MODULE_H
#define SVMP_PHYSICS_COUPLING_THERMAL_INTERFACE_COUPLING_MODULE_H

/**
 * @file ThermalInterfaceCouplingModule.h
 * @brief Physics-side thermal interface coupling contract declaration.
 */

#include "FE/Coupling/DefinitionBackedCouplingContract.h"

#include <string>

namespace svmp {
namespace Physics {
namespace coupling {

struct ThermalInterfaceCouplingOptions {
    FE::coupling::CouplingMode mode{FE::coupling::CouplingMode::Monolithic};
    std::string contract_name{"thermal_interface"};
    std::string side_a_name{"side_a"};
    std::string side_b_name{"side_b"};
    std::string interface_name{"interface"};

    std::string side_a_temperature_field{"temperature"};
    std::string side_b_temperature_field{"temperature"};
    std::string side_a_heat_flux_field{"heat_flux"};
    std::string side_b_heat_flux_field{"heat_flux"};
    int temperature_components{1};
    int heat_flux_components{1};

    FE::coupling::CouplingTransferDeclaration temperature_transfer{};
    FE::coupling::CouplingTransferDeclaration heat_flux_transfer{};
};

class ThermalInterfaceCouplingModule final
    : public FE::coupling::DefinitionBackedCouplingContract {
public:
    explicit ThermalInterfaceCouplingModule(
        ThermalInterfaceCouplingOptions options = {});

    [[nodiscard]] std::string name() const override;

    void validate(const FE::coupling::CouplingContext& ctx) const override;

protected:
    [[nodiscard]] std::string contractInstanceName() const override;
    void define(FE::coupling::CouplingDefinitionBuilder& builder) const override;

private:
    ThermalInterfaceCouplingOptions options_{};
};

} // namespace coupling
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_COUPLING_THERMAL_INTERFACE_COUPLING_MODULE_H
