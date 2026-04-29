#include "Coupling/CouplingRegistry.h"

#include "Core/FEException.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace svmp::FE::coupling;
using svmp::FE::InvalidArgumentException;

namespace {

class DummyContract final : public CouplingContract {
public:
    std::string name() const override { return "dummy"; }

    CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "dummy_instance";
        return declaration;
    }
};

} // namespace

TEST(CouplingRegistry, RegistersAndCreatesContracts)
{
    CouplingRegistry registry;
    registry.registerContract("dummy", [] {
        return std::make_unique<DummyContract>();
    });

    EXPECT_TRUE(registry.contains("dummy"));
    EXPECT_EQ(registry.names().size(), 1u);
    const auto contract = registry.create("dummy");
    ASSERT_NE(contract, nullptr);
    EXPECT_EQ(contract->name(), "dummy");
}

TEST(CouplingRegistry, RejectsDuplicateContractTypes)
{
    CouplingRegistry registry;
    registry.registerContract("dummy", [] {
        return std::make_unique<DummyContract>();
    });

    EXPECT_THROW(registry.registerContract("dummy", [] {
        return std::make_unique<DummyContract>();
    }), InvalidArgumentException);
}

TEST(CouplingRegistry, UnknownContractLookupFails)
{
    CouplingRegistry registry;
    EXPECT_FALSE(registry.contains("missing"));
    EXPECT_THROW(static_cast<void>(registry.create("missing")), InvalidArgumentException);
}
