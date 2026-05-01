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
    CouplingRegionRef surface{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 4,
    };
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    surface.logical_region = svmp::search::LogicalInterfaceRegionId{
        .persistent_id = "left_surface",
        .name = "surface",
    };
#endif

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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
svmp::search::InterfaceRevisionSnapshot geometryRevisionSnapshot(
    std::uint64_t offset)
{
    return svmp::search::InterfaceRevisionSnapshot{
        .configuration = svmp::Configuration::Current,
        .geometry_revision = 300 + offset,
        .reference_geometry_revision = 400 + offset,
        .current_geometry_revision = 500 + offset,
        .topology_revision = 600 + offset,
        .ownership_revision = 700 + offset,
        .numbering_revision = 800 + offset,
        .field_layout_revision = 900 + offset,
        .label_revision = 1000 + offset,
        .active_configuration_epoch = 1100 + offset,
    };
}
#endif

CouplingContext boundaryInterfaceGeometryContext()
{
    const auto* left_system = geometrySystemToken();
    const auto* right_system =
        reinterpret_cast<const systems::FESystem*>(2);
    const auto space =
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingRegionRef boundary{
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 4,
        .coordinate_configuration = CouplingCoordinateConfiguration::Current,
        .geometry_revision = 41,
    };
    CouplingRegionRef left_interface{
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 7,
        .side = CouplingInterfaceSide::Minus,
        .coordinate_configuration = CouplingCoordinateConfiguration::Current,
    };
    CouplingRegionRef right_interface{
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 8,
        .side = CouplingInterfaceSide::Plus,
        .coordinate_configuration = CouplingCoordinateConfiguration::Current,
    };
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    left_interface.logical_region = svmp::search::LogicalInterfaceRegionId{
        .persistent_id = "left_interface",
        .name = "left_interface",
        .physical_label = 7,
    };
    right_interface.logical_region = svmp::search::LogicalInterfaceRegionId{
        .persistent_id = "right_interface",
        .name = "right_interface",
        .physical_label = 8,
    };
    left_interface.revision_snapshot = geometryRevisionSnapshot(1);
    right_interface.revision_snapshot = geometryRevisionSnapshot(2);
#else
    left_interface.geometry_revision = 301;
    right_interface.geometry_revision = 302;
#endif

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = 1,
    });
    builder.addRegion(boundary);
    builder.addRegion(left_interface);
    builder.addRegion(right_interface);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {left_interface, right_interface},
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

CouplingGeometryTerminalRequirement geometryRequirement(
    CouplingGeometryTerminalQuantity quantity,
    CouplingRegionKind region_kind,
    forms::GeometryConfiguration configuration =
        forms::GeometryConfiguration::Reference)
{
    return CouplingGeometryTerminalRequirement{
        .quantity = quantity,
        .scope = CouplingGeometryTerminalScope{
            .participant_name = "left",
            .location = CouplingGeometryTerminalLocationDeclaration{
                .region_kind = region_kind,
                .coordinate_configuration = configuration,
            },
        },
    };
}

CouplingGeometryTerminalAvailability broadGeometryAvailability()
{
    return CouplingGeometryTerminalAvailability{
        .supported_quantities = {
            CouplingGeometryTerminalQuantity::MeshDisplacement,
            CouplingGeometryTerminalQuantity::Coordinate,
            CouplingGeometryTerminalQuantity::Jacobian,
            CouplingGeometryTerminalQuantity::JacobianInverse,
            CouplingGeometryTerminalQuantity::Normal,
            CouplingGeometryTerminalQuantity::CurrentMeasure,
            CouplingGeometryTerminalQuantity::SurfaceJacobian,
            CouplingGeometryTerminalQuantity::CellDiameter,
            CouplingGeometryTerminalQuantity::CellVolume,
            CouplingGeometryTerminalQuantity::FacetArea,
            CouplingGeometryTerminalQuantity::CellDomainId,
        },
        .supported_domains = {
            analysis::DomainKind::Cell,
            analysis::DomainKind::Boundary,
            analysis::DomainKind::InterfaceFace,
        },
    };
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
    EXPECT_FALSE(toAnalysisDomainKind(CouplingRegionKind::Curve).has_value());
    EXPECT_FALSE(toAnalysisDomainKind(CouplingRegionKind::Point).has_value());
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

TEST(CouplingGeometryRequirements, RejectsLocationOnlySharedRegionOwnerScope)
{
    const std::array<CouplingGeometryTerminalRequirement, 1> requirements{
        CouplingGeometryTerminalRequirement{
            .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
            .scope = CouplingGeometryTerminalScope{
                .location = CouplingGeometryTerminalLocationDeclaration{
                    .region_kind = CouplingRegionKind::Boundary,
                    .shared_region_name = "surface_pair",
                },
            },
        },
    };

    const auto validation = validateGeometryTerminalRequirements(
        geometryContext(),
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        boundaryGeometryAvailability());

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires owner scope"),
              std::string::npos);
}

TEST(CouplingGeometryRequirements, RejectsConflictingOwnerAndRegionParticipants)
{
    auto requirement = boundaryNormalRequirement();
    requirement.scope.participant_name = "right";

    const std::array<CouplingGeometryTerminalRequirement, 1> requirements{requirement};
    const auto validation = validateGeometryTerminalRequirements(
        geometryContext(),
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        boundaryGeometryAvailability());

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("conflicts with region participant"),
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

TEST(CouplingGeometryRequirements, GeometryFixtureCoversCoreQuantitiesAndAvailability)
{
    std::vector<CouplingGeometryTerminalRequirement> requirements{
        geometryRequirement(CouplingGeometryTerminalQuantity::MeshDisplacement,
                            CouplingRegionKind::Domain),
        geometryRequirement(CouplingGeometryTerminalQuantity::Coordinate,
                            CouplingRegionKind::Domain),
        geometryRequirement(CouplingGeometryTerminalQuantity::Jacobian,
                            CouplingRegionKind::Domain),
        geometryRequirement(CouplingGeometryTerminalQuantity::JacobianInverse,
                            CouplingRegionKind::Domain),
        geometryRequirement(CouplingGeometryTerminalQuantity::Normal,
                            CouplingRegionKind::InterfaceFace,
                            forms::GeometryConfiguration::Current),
        geometryRequirement(CouplingGeometryTerminalQuantity::CurrentMeasure,
                            CouplingRegionKind::Boundary,
                            forms::GeometryConfiguration::Current),
        geometryRequirement(CouplingGeometryTerminalQuantity::SurfaceJacobian,
                            CouplingRegionKind::Boundary),
        geometryRequirement(CouplingGeometryTerminalQuantity::CellDiameter,
                            CouplingRegionKind::Domain),
        geometryRequirement(CouplingGeometryTerminalQuantity::CellVolume,
                            CouplingRegionKind::Domain),
        geometryRequirement(CouplingGeometryTerminalQuantity::FacetArea,
                            CouplingRegionKind::Boundary),
        geometryRequirement(CouplingGeometryTerminalQuantity::CellDomainId,
                            CouplingRegionKind::Domain),
    };

    const auto context = boundaryInterfaceGeometryContext();
    const auto summary = summarizeGeometryTerminalRequirements(
        context,
        std::span<const CouplingGeometryTerminalRequirement>(requirements));
    EXPECT_EQ(summary.quantities.size(), requirements.size());
    EXPECT_TRUE(summary.requires_reference_configuration);
    EXPECT_TRUE(summary.requires_current_configuration);
    ASSERT_EQ(summary.domains.size(), 3u);
    EXPECT_EQ(summary.domains[0], analysis::DomainKind::Cell);
    EXPECT_EQ(summary.domains[1], analysis::DomainKind::InterfaceFace);
    EXPECT_EQ(summary.domains[2], analysis::DomainKind::Boundary);

    const auto availability = broadGeometryAvailability();
    EXPECT_TRUE(validateGeometryTerminalRequirements(
                    context,
                    std::span<const CouplingGeometryTerminalRequirement>(
                        requirements),
                    availability)
                    .ok());

    auto missing_quantity = availability;
    missing_quantity.supported_quantities.pop_back();
    const auto quantity_validation = validateGeometryTerminalRequirements(
        context,
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        missing_quantity);
    EXPECT_FALSE(quantity_validation.ok());
    EXPECT_NE(formatDiagnostics(quantity_validation).find("quantity is not available"),
              std::string::npos);

    auto cell_only = availability;
    cell_only.supported_domains = {analysis::DomainKind::Cell};
    const auto domain_validation = validateGeometryTerminalRequirements(
        context,
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        cell_only);
    EXPECT_FALSE(domain_validation.ok());
    EXPECT_NE(formatDiagnostics(domain_validation).find("domain is not available"),
              std::string::npos);

    auto no_current = availability;
    no_current.supports_current_configuration = false;
    const auto current_validation = validateGeometryTerminalRequirements(
        context,
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        no_current);
    EXPECT_FALSE(current_validation.ok());
    EXPECT_NE(formatDiagnostics(current_validation).find(
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

TEST(CouplingGeometryRequirements, GeometryFixtureValidatesBoundaryAndInterfaceProvenance)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "geometry_fixture";
    declaration.geometry_requirements.push_back(
        CouplingGeometryTerminalRequirement{
            .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
            .scope = CouplingGeometryTerminalScope{
                .region = CouplingRegionEndpointDeclaration{
                    .participant_name = "left",
                    .region_name = "surface",
                },
                .location = CouplingGeometryTerminalLocationDeclaration{
                    .region_kind = CouplingRegionKind::Boundary,
                    .coordinate_configuration =
                        forms::GeometryConfiguration::Current,
                    .quadrature_policy_key = 41,
                },
            },
        });
    declaration.geometry_requirements.push_back(
        CouplingGeometryTerminalRequirement{
            .quantity = CouplingGeometryTerminalQuantity::SurfaceJacobian,
            .scope = CouplingGeometryTerminalScope{
                .region = CouplingRegionEndpointDeclaration{
                    .participant_name = "left",
                    .region_name = "interface",
                    .shared_region_name = "interface",
                },
                .location = CouplingGeometryTerminalLocationDeclaration{
                    .region_kind = CouplingRegionKind::InterfaceFace,
                    .shared_region_name = "interface",
                    .side = CouplingInterfaceSide::Minus,
                    .coordinate_configuration =
                        forms::GeometryConfiguration::Current,
                    .transform_from_configuration =
                        forms::GeometryConfiguration::Reference,
                    .transform_to_configuration =
                        forms::GeometryConfiguration::Current,
                    .quadrature_policy_key = 73,
                },
            },
        });

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "geometry_fixture_form";
    metadata.geometry_terminals.push_back(
        CouplingFormGeometryTerminalProvenance{
            .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
            .location = CouplingGeometryTerminalLocationProvenance{
                .region_kind = CouplingRegionKind::Boundary,
                .marker = 4,
                .coordinate_configuration =
                    forms::GeometryConfiguration::Current,
                .geometry_revision = 41,
                .quadrature_policy_key = 41,
            },
            .analysis_domain = analysis::DomainKind::Boundary,
            .owner = CouplingGeometryTerminalOwnerProvenance{
                .participant_name = "left",
                .system_name = "left_system",
                .region_name = "surface",
            },
            .provider = "forms",
            .normal_available = true,
        });
    metadata.geometry_terminals.push_back(
        CouplingFormGeometryTerminalProvenance{
            .quantity = CouplingGeometryTerminalQuantity::SurfaceJacobian,
            .location = CouplingGeometryTerminalLocationProvenance{
                .region_kind = CouplingRegionKind::InterfaceFace,
                .shared_region_name = "interface",
                .marker = 7,
                .side = CouplingInterfaceSide::Minus,
                .coordinate_configuration =
                    forms::GeometryConfiguration::Current,
                .transform_from_configuration =
                    forms::GeometryConfiguration::Reference,
                .transform_to_configuration =
                    forms::GeometryConfiguration::Current,
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
                .logical_region = svmp::search::LogicalInterfaceRegionId{
                    .persistent_id = "left_interface",
                    .name = "left_interface",
                    .physical_label = 7,
                },
                .geometry_revision = geometryRevisionSnapshot(1).geometry_revision,
#else
                .geometry_revision = 301,
#endif
                .quadrature_policy_key = 73,
            },
            .analysis_domain = analysis::DomainKind::InterfaceFace,
            .owner = CouplingGeometryTerminalOwnerProvenance{
                .participant_name = "left",
                .system_name = "left_system",
                .region_name = "interface",
                .shared_region_name = "interface",
            },
            .provider = "forms",
            .gradient_or_jacobian_available = true,
        });

    const auto validate_metadata =
        [&](const CouplingFormAnalysisMetadata& form) {
            CouplingGraph graph;
            const std::array<CouplingContractDeclaration, 1> declarations{
                declaration};
            const std::array<CouplingFormAnalysisMetadata, 1> installed{form};
            return graph.buildFinalizedGraph(
                boundaryInterfaceGeometryContext(),
                std::span<const CouplingContractDeclaration>(declarations),
                std::span<const CouplingFormAnalysisMetadata>(installed));
        };

    EXPECT_TRUE(validate_metadata(metadata).ok());

    auto wrong_domain = metadata;
    wrong_domain.geometry_terminals[1].analysis_domain =
        analysis::DomainKind::Boundary;
    EXPECT_FALSE(validate_metadata(wrong_domain).ok());

    auto wrong_side = metadata;
    wrong_side.geometry_terminals[1].location.side = CouplingInterfaceSide::Plus;
    EXPECT_FALSE(validate_metadata(wrong_side).ok());

    auto wrong_revision = metadata;
    wrong_revision.geometry_terminals[1].location.geometry_revision += 1;
    EXPECT_FALSE(validate_metadata(wrong_revision).ok());

    auto wrong_quadrature = metadata;
    wrong_quadrature.geometry_terminals[1].location.quadrature_policy_key += 1;
    EXPECT_FALSE(validate_metadata(wrong_quadrature).ok());

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto wrong_logical_region = metadata;
    wrong_logical_region.geometry_terminals[1]
        .location.logical_region->persistent_id = "other_interface";
    EXPECT_FALSE(validate_metadata(wrong_logical_region).ok());
#endif
}

TEST(CouplingGeometryRequirements, CouplingGraphValidatesGeometryTerminalLocationProvenance)
{
    CouplingContractDeclaration declaration = geometryDeclaration();

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "boundary_normal_form";
    metadata.geometry_terminals.push_back(
        CouplingFormGeometryTerminalProvenance{
            .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
            .location = CouplingGeometryTerminalLocationProvenance{
                .region_kind = CouplingRegionKind::Boundary,
                .marker = 4,
                .coordinate_configuration =
                    forms::GeometryConfiguration::Current,
            },
            .analysis_domain = analysis::DomainKind::Boundary,
            .owner = CouplingGeometryTerminalOwnerProvenance{
                .participant_name = "left",
                .system_name = "system",
                .region_name = "surface",
            },
            .provider = "forms",
            .normal_available = true,
        });
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    metadata.geometry_terminals[0].location.logical_region =
        svmp::search::LogicalInterfaceRegionId{
            .persistent_id = "left_surface",
            .name = "surface",
        };
#endif

    const auto validate_metadata = [&](const CouplingFormAnalysisMetadata& form) {
        CouplingGraph graph;
        const std::array<CouplingContractDeclaration, 1> declarations{declaration};
        const std::array<CouplingFormAnalysisMetadata, 1> installed{form};
        return graph.buildFinalizedGraph(
            geometryContext(),
            std::span<const CouplingContractDeclaration>(declarations),
            std::span<const CouplingFormAnalysisMetadata>(installed));
    };

    EXPECT_TRUE(validate_metadata(metadata).ok());

    auto wrong_marker = metadata;
    wrong_marker.geometry_terminals[0].location.marker = 5;
    EXPECT_FALSE(validate_metadata(wrong_marker).ok());

    auto wrong_configuration = metadata;
    wrong_configuration.geometry_terminals[0].location.coordinate_configuration =
        forms::GeometryConfiguration::Reference;
    EXPECT_FALSE(validate_metadata(wrong_configuration).ok());

    auto wrong_owner = metadata;
    wrong_owner.geometry_terminals[0].owner->participant_name = "right";
    EXPECT_FALSE(validate_metadata(wrong_owner).ok());

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto wrong_logical_region = metadata;
    wrong_logical_region.geometry_terminals[0].location.logical_region =
        svmp::search::LogicalInterfaceRegionId{
            .persistent_id = "other_surface",
            .name = "surface",
        };
    EXPECT_FALSE(validate_metadata(wrong_logical_region).ok());
#endif
}
