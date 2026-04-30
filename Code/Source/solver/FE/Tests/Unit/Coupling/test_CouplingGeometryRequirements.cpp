#include "Coupling/CouplingGeometryRequirements.h"
#include "Coupling/CouplingGraph.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* geometrySystemToken()
{
    return reinterpret_cast<const systems::FESystem*>(1);
}

CouplingContext geometryContext()
{
    const auto* system = geometrySystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    const CouplingRegionRef surface{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 4,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "system",
        .system = system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = 1,
    });
    builder.addRegion(surface);
    builder.addSharedRegion(SharedRegionRef{
        .name = "surface_pair",
        .required_region_kind = CouplingRegionKind::Boundary,
        .participant_regions = {surface},
    });
    return builder.build();
}

CouplingGeometryTerminalRequirement boundaryNormalRequirement()
{
    return CouplingGeometryTerminalRequirement{
        .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
        .scope = CouplingGeometryTerminalScope{
            .participant_name = "left",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
            },
            .location = CouplingGeometryTerminalLocationDeclaration{
                .region_kind = CouplingRegionKind::Boundary,
                .coordinate_configuration = forms::GeometryConfiguration::Current,
            },
        },
    };
}

CouplingGeometryTerminalAvailability boundaryGeometryAvailability()
{
    CouplingGeometryTerminalAvailability availability;
    availability.supported_quantities = {
        CouplingGeometryTerminalQuantity::CurrentNormal,
        CouplingGeometryTerminalQuantity::CurrentMeasure,
        CouplingGeometryTerminalQuantity::MeshDisplacement,
    };
    availability.supported_domains = {
        analysis::DomainKind::Cell,
        analysis::DomainKind::Boundary,
    };
    return availability;
}

CouplingContractDeclaration geometryDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.geometry_requirements.push_back(boundaryNormalRequirement());
    return declaration;
}

} // namespace

TEST(CouplingGeometryRequirements, MapsRegionKindsToAnalysisDomains)
{
    EXPECT_EQ(toAnalysisDomainKind(CouplingRegionKind::Domain),
              analysis::DomainKind::Cell);
    EXPECT_EQ(toAnalysisDomainKind(CouplingRegionKind::Boundary),
              analysis::DomainKind::Boundary);
    EXPECT_EQ(toAnalysisDomainKind(CouplingRegionKind::InterfaceFace),
              analysis::DomainKind::InterfaceFace);
    EXPECT_FALSE(toAnalysisDomainKind(CouplingRegionKind::UserDefined).has_value());
}

TEST(CouplingGeometryRequirements, SummarizesQuantitiesDomainsAndConfigurations)
{
    const std::array<CouplingGeometryTerminalRequirement, 1> requirements{
        boundaryNormalRequirement(),
    };

    const auto summary = summarizeGeometryTerminalRequirements(
        geometryContext(),
        std::span<const CouplingGeometryTerminalRequirement>(requirements));

    ASSERT_EQ(summary.quantities.size(), 1u);
    EXPECT_EQ(summary.quantities[0], CouplingGeometryTerminalQuantity::CurrentNormal);
    ASSERT_EQ(summary.domains.size(), 1u);
    EXPECT_EQ(summary.domains[0], analysis::DomainKind::Boundary);
    EXPECT_TRUE(summary.requires_current_configuration);
}

TEST(CouplingGeometryRequirements, AcceptsSupportedBoundaryGeometryRequirement)
{
    const std::array<CouplingGeometryTerminalRequirement, 1> requirements{
        boundaryNormalRequirement(),
    };

    const auto validation = validateGeometryTerminalRequirements(
        geometryContext(),
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        boundaryGeometryAvailability());

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGeometryRequirements, RejectsUnsupportedQuantityAndDomain)
{
    auto requirement = boundaryNormalRequirement();
    requirement.scope.location->region_kind = CouplingRegionKind::InterfaceFace;

    CouplingGeometryTerminalAvailability availability;
    availability.supported_quantities = {CouplingGeometryTerminalQuantity::MeshDisplacement};
    availability.supported_domains = {analysis::DomainKind::Cell};

    const std::array<CouplingGeometryTerminalRequirement, 1> requirements{requirement};
    const auto validation = validateGeometryTerminalRequirements(
        geometryContext(),
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        availability);

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("quantity is not available"), std::string::npos);
    EXPECT_NE(text.find("domain is not available"), std::string::npos);
}

TEST(CouplingGeometryRequirements, RejectsUserDefinedRegionWithoutConcreteDomain)
{
    auto requirement = boundaryNormalRequirement();
    requirement.scope.location->region_kind = CouplingRegionKind::UserDefined;

    CouplingGeometryTerminalAvailability availability;
    availability.supported_quantities = {CouplingGeometryTerminalQuantity::CurrentNormal};
    availability.supported_domains = {analysis::DomainKind::Boundary};

    const std::array<CouplingGeometryTerminalRequirement, 1> requirements{requirement};
    const auto validation = validateGeometryTerminalRequirements(
        geometryContext(),
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        availability);

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("unsupported region kind"),
              std::string::npos);
}

TEST(CouplingGeometryRequirements, RejectsMissingOwnerScope)
{
    const std::array<CouplingGeometryTerminalRequirement, 1> requirements{
        CouplingGeometryTerminalRequirement{
            .quantity = CouplingGeometryTerminalQuantity::CurrentJacobian,
        },
    };

    CouplingGeometryTerminalAvailability availability;
    availability.supported_quantities = {CouplingGeometryTerminalQuantity::CurrentJacobian};
    availability.supported_domains = {analysis::DomainKind::Cell};

    const auto validation = validateGeometryTerminalRequirements(
        geometryContext(),
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        availability);

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires owner scope"),
              std::string::npos);
}

TEST(CouplingGeometryRequirements, CouplingGraphValidatesAggregatedDeclarations)
{
    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{geometryDeclaration()};
    const auto declaration_validation = graph.buildDeclarationGraph(
        geometryContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(declaration_validation.ok()) << formatDiagnostics(declaration_validation);

    EXPECT_TRUE(graph.validateGeometryTerminalRequirements(
                         geometryContext(),
                         boundaryGeometryAvailability())
                    .ok());

    CouplingGeometryTerminalAvailability unsupported = boundaryGeometryAvailability();
    unsupported.supports_current_configuration = false;
    const auto validation = graph.validateGeometryTerminalRequirements(
        geometryContext(),
        unsupported);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "current-configuration geometry terminal is unavailable"),
              std::string::npos);
}

TEST(CouplingGeometryRequirements, CouplingGraphRequiresInstalledGeometryTerminalsDeclared)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "geometry_form";
    metadata.geometry_terminals.push_back(
        CouplingFormGeometryTerminalProvenance{
            .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
            .location = CouplingGeometryTerminalLocationProvenance{
                .region_kind = CouplingRegionKind::Boundary,
                .marker = 4,
            },
            .analysis_domain = analysis::DomainKind::Boundary,
            .provider = "forms",
            .normal_available = true,
        });

    const auto run_validation = [&]() {
        CouplingGraph graph;
        const std::array<CouplingContractDeclaration, 1> declarations{declaration};
        const std::array<CouplingFormAnalysisMetadata, 1> installed{metadata};
        return graph.buildFinalizedGraph(
            geometryContext(),
            std::span<const CouplingContractDeclaration>(declarations),
            std::span<const CouplingFormAnalysisMetadata>(installed));
    };

    const auto missing = run_validation();
    EXPECT_FALSE(missing.ok());
    EXPECT_NE(formatDiagnostics(missing).find("current_normal"),
              std::string::npos);

    declaration.geometry_requirements.push_back(
        CouplingGeometryTerminalRequirement{
            .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
        });
    const auto wrong_domain = run_validation();
    EXPECT_FALSE(wrong_domain.ok());
    EXPECT_NE(formatDiagnostics(wrong_domain).find("Boundary"),
              std::string::npos);

    declaration.geometry_requirements[0].scope.location =
        CouplingGeometryTerminalLocationDeclaration{
            .region_kind = CouplingRegionKind::Boundary,
        };
    EXPECT_TRUE(run_validation().ok());
}
