#include "Coupling/CouplingDeclaration.h"
#include "Coupling/CouplingContract.h"
#include "Coupling/CouplingGraph.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <span>

using namespace svmp::FE::coupling;
using svmp::FE::InvalidArgumentException;

namespace {

std::shared_ptr<const svmp::FE::spaces::FunctionSpace> scalarSpace()
{
    return std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Triangle3, 1);
}

CouplingContractDeclaration minimalDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "surface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.shared_regions.push_back({
        .shared_region_name = "interface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    return declaration;
}

class MissingFieldContract final : public CouplingContract {
public:
    std::string name() const override { return "missing_field_contract"; }

    CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "missing_field_instance";
        declaration.participants.push_back({.participant_name = "left"});
        declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
        return declaration;
    }
};

class MismatchedTypeContract final : public CouplingContract {
public:
    std::string name() const override { return "actual_type"; }

    CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = "declared_type";
        declaration.contract_name = "mismatched_instance";
        return declaration;
    }
};

} // namespace

TEST(CouplingContractValidation, AcceptsMinimalTwoParticipantDeclaration)
{
    const auto declaration = minimalDeclaration();
    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());
}

TEST(CouplingContractValidation, RejectsEmptyContractNames)
{
    auto declaration = minimalDeclaration();
    declaration.contract_name.clear();

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("configured contract name"),
              std::string::npos);
}

TEST(CouplingContractValidation, RejectsDuplicateRequirements)
{
    auto declaration = minimalDeclaration();
    declaration.participants.push_back({.participant_name = "left"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "surface",
    });
    declaration.shared_regions.push_back({.shared_region_name = "interface"});

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("duplicate participant requirement"), std::string::npos);
    EXPECT_NE(text.find("duplicate field requirement"), std::string::npos);
    EXPECT_NE(text.find("duplicate participant-local region requirement"), std::string::npos);
    EXPECT_NE(text.find("duplicate shared-region requirement"), std::string::npos);
}

TEST(CouplingContractValidation, ValidatesAdditionalFieldAttachmentRules)
{
    auto declaration = minimalDeclaration();
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "lambda",
        .components = -1,
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "surface_field",
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("component count"), std::string::npos);
    EXPECT_NE(text.find("exactly one region attachment"), std::string::npos);
}

TEST(CouplingContractValidation, ValidatesAdditionalFieldSpaceAndComponents)
{
    auto declaration = minimalDeclaration();
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "missing_space",
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "bad_components",
        .space = scalarSpace(),
        .components = 2,
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("function space"), std::string::npos);
    EXPECT_NE(text.find("match the function space"), std::string::npos);

    declaration.additional_fields.clear();
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "inferred_components",
        .space = scalarSpace(),
        .components = 0,
    });

    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());
}

TEST(CouplingContractValidation, ValidatesTemporalRequirementsAndExchangeShape)
{
    auto declaration = minimalDeclaration();
    declaration.temporal_requirements.push_back({
        .quantity = CouplingTemporalQuantity::FieldDerivative,
        .derivative_order = 0,
    });
    declaration.partitioned_exchange_declarations.push_back({
        .producer_port = {.contract_instance_name = "generic_instance", .port_name = "out"},
        .consumer_port = {.contract_instance_name = "generic_instance", .port_name = "in"},
        .value = {.rank = CouplingValueRank::MixedBlock, .components = 2},
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("positive derivative order"), std::string::npos);
    EXPECT_NE(text.find("mixed block values require component layout"), std::string::npos);
}

TEST(CouplingContractValidation, CouplingGraphRejectsDuplicateInstances)
{
    auto a = minimalDeclaration();
    auto b = minimalDeclaration();
    b.contract_type = "other_generic";

    CouplingGraph graph;
    CouplingContext context;
    const std::array<CouplingContractDeclaration, 2> declarations{a, b};
    const auto validation =
        graph.buildDeclarationGraph(context, std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("duplicate coupling contract instance name"),
              std::string::npos);
}

TEST(CouplingContractValidation, DefaultContractValidationChecksDeclarationGraph)
{
    const MissingFieldContract contract;

    EXPECT_THROW(contract.validate(CouplingContext{}), InvalidArgumentException);
}

TEST(CouplingContractValidation, DefaultContractValidationRejectsTypeMismatch)
{
    const MismatchedTypeContract contract;

    EXPECT_THROW(contract.validate(CouplingContext{}), InvalidArgumentException);
}
