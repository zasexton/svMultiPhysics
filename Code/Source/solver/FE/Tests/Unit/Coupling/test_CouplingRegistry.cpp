#include "Coupling/CouplingRegistry.h"

#include "Core/FEException.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace svmp::FE::coupling;
using svmp::FE::InvalidArgumentException;

namespace {

class DummyContract : public CouplingContract {
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

class MonolithicContract final : public DummyContract {
public:
    std::string name() const override { return "monolithic"; }
    bool supportsMonolithicLowering() const override { return true; }
};

class PartitionedContract final : public DummyContract {
public:
    std::string name() const override { return "partitioned"; }
    bool supportsPartitionedLowering() const override { return true; }
};

class HybridContract final : public DummyContract {
public:
    std::string name() const override { return "hybrid"; }
    bool supportsMonolithicLowering() const override { return true; }
    bool supportsPartitionedLowering() const override { return true; }
};

class MismatchedRegisteredContract final : public DummyContract {
public:
    std::string name() const override { return "actual_contract_name"; }
};

bool containsName(const std::vector<std::string>& names, const std::string& name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

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

TEST(CouplingRegistry, FiltersContractsBySupportedMode)
{
    CouplingRegistry registry;
    registry.registerContract("monolithic", [] {
        return std::make_unique<MonolithicContract>();
    });
    registry.registerContract("partitioned", [] {
        return std::make_unique<PartitionedContract>();
    });
    registry.registerContract("hybrid", [] {
        return std::make_unique<HybridContract>();
    });

    EXPECT_TRUE(registry.supportsMode("monolithic", CouplingMode::Monolithic));
    EXPECT_FALSE(registry.supportsMode("monolithic", CouplingMode::Partitioned));
    EXPECT_TRUE(registry.supportsMode("partitioned", CouplingMode::Partitioned));
    EXPECT_FALSE(registry.supportsMode("partitioned", CouplingMode::Monolithic));
    EXPECT_TRUE(registry.supportsMode("hybrid", CouplingMode::Monolithic));
    EXPECT_TRUE(registry.supportsMode("hybrid", CouplingMode::Partitioned));

    const auto monolithic = registry.namesSupporting(CouplingMode::Monolithic);
    EXPECT_TRUE(containsName(monolithic, "monolithic"));
    EXPECT_TRUE(containsName(monolithic, "hybrid"));
    EXPECT_FALSE(containsName(monolithic, "partitioned"));

    const auto partitioned = registry.namesSupporting(CouplingMode::Partitioned);
    EXPECT_TRUE(containsName(partitioned, "partitioned"));
    EXPECT_TRUE(containsName(partitioned, "hybrid"));
    EXPECT_FALSE(containsName(partitioned, "monolithic"));
}

TEST(CouplingRegistry, ValidatesDeclarationTypesAndInstanceNames)
{
    CouplingRegistry registry;
    registry.registerContract("dummy", [] {
        return std::make_unique<DummyContract>();
    });
    registry.registerContract("mismatched_key", [] {
        return std::make_unique<MismatchedRegisteredContract>();
    });

    CouplingContractDeclaration valid;
    valid.contract_type = "dummy";
    valid.contract_name = "valid_instance";
    const std::array<CouplingContractDeclaration, 1> valid_declarations{valid};
    EXPECT_TRUE(registry.validateDeclarations(
                            std::span<const CouplingContractDeclaration>(
                                valid_declarations))
                    .ok());

    auto unknown = valid;
    unknown.contract_type = "missing";
    unknown.contract_name = "unknown_instance";
    auto mismatched = valid;
    mismatched.contract_type = "mismatched_key";
    mismatched.contract_name = "mismatched_instance";
    auto duplicate = valid;
    duplicate.contract_name = valid.contract_name;
    const std::array<CouplingContractDeclaration, 4> invalid_declarations{
        valid,
        unknown,
        mismatched,
        duplicate,
    };

    const auto validation = registry.validateDeclarations(
        std::span<const CouplingContractDeclaration>(invalid_declarations));
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("declaration type is not registered"), std::string::npos);
    EXPECT_NE(text.find("does not match the registered contract name"),
              std::string::npos);
    EXPECT_NE(text.find("duplicate coupling contract instance name"),
              std::string::npos);
}
