/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceFactory.h"

#include "Assembly/Assembler.h"

namespace svmp {
namespace FE {
namespace spaces {

namespace {

int require_space_request_order(const elements::ElementRequest& req,
                                const char* message) {
    if (!req.order.has_value()) {
        FE_THROW(InvalidArgumentException, message);
    }
    return *req.order;
}

} // namespace

std::shared_ptr<FunctionSpace> SpaceFactory::create(SpaceType type,
                                                    ElementType element_type,
                                                    int order) {
    switch (type) {
        case SpaceType::H1:
            return create_h1(element_type, order);
        case SpaceType::C1:
            return create_c1(element_type, order);
        case SpaceType::L2:
            return create_l2(element_type, order);
        case SpaceType::HCurl:
            return create_hcurl(element_type, order);
        case SpaceType::HDiv:
            return create_hdiv(element_type, order);
        case SpaceType::Product:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create: Product spaces require component count");
        case SpaceType::GenericBasis:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create: GenericBasisSpace requires SpaceFactory::create(const SpaceRequest&)");
        case SpaceType::Mixed:
        case SpaceType::Trace:
        case SpaceType::Mortar:
        case SpaceType::Composite:
        case SpaceType::Enriched:
        case SpaceType::Adaptive:
            FE_THROW(NotImplementedException,
                     "SpaceFactory::create: use specialized constructors for composite/enriched/adaptive spaces");
        default:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create: unknown SpaceType");
    }
}

std::shared_ptr<FunctionSpace> SpaceFactory::create(const SpaceRequest& req) {
    switch (req.space_type) {
        case SpaceType::H1:
            return std::make_shared<H1Space>(req.element);
        case SpaceType::C1:
            return create_c1(
                req.element.element_type,
                require_space_request_order(req.element,
                                            "SpaceFactory::create(SpaceRequest): C1 requests require an explicit order"));
        case SpaceType::L2:
            return std::make_shared<L2Space>(req.element);
        case SpaceType::HCurl:
            return std::make_shared<HCurlSpace>(req.element);
        case SpaceType::HDiv:
            return std::make_shared<HDivSpace>(req.element);
        case SpaceType::GenericBasis: {
            auto element = elements::ElementFactory::create(req.element);
            return std::make_shared<GenericBasisSpace>(std::move(element));
        }
        case SpaceType::Product:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create(SpaceRequest): Product spaces require component count");
        case SpaceType::Mixed:
        case SpaceType::Trace:
        case SpaceType::Mortar:
        case SpaceType::Composite:
        case SpaceType::Enriched:
        case SpaceType::Adaptive:
            FE_THROW(NotImplementedException,
                     "SpaceFactory::create(SpaceRequest): use specialized constructors for composite/enriched/adaptive spaces");
        default:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create(SpaceRequest): unknown SpaceType");
    }
}

std::shared_ptr<ProductSpace> SpaceFactory::create_vector_h1(ElementType element_type,
                                                             int order,
                                                             int components) {
    auto base = create_h1(element_type, order);
    return std::make_shared<ProductSpace>(base, components);
}

ElementType inferUniformElementType(const assembly::IMeshAccess& mesh, int domain_id)
{
    FE_THROW_IF(mesh.numCells() <= 0, InvalidArgumentException,
                "inferUniformElementType: mesh has no cells");

    ElementType element_type = ElementType::Unknown;
    bool have_type = false;
    bool matched_domain = false;

    mesh.forEachCell([&](GlobalIndex cell_id) {
        if (domain_id >= 0 && mesh.getCellDomainId(cell_id) != domain_id) {
            return;
        }
        matched_domain = true;

        const ElementType cell_type = mesh.getCellType(cell_id);
        if (!have_type) {
            element_type = cell_type;
            have_type = true;
            return;
        }

        if (cell_type != element_type) {
            FE_THROW(InvalidArgumentException,
                     "inferUniformElementType: mesh has mixed element types; pass ElementType explicitly or use domain_id");
        }
    });

    if (domain_id >= 0 && !matched_domain) {
        FE_THROW(InvalidArgumentException,
                 "inferUniformElementType: domain_id did not match any cells");
    }

    FE_THROW_IF(!have_type, InvalidArgumentException,
                "inferUniformElementType: mesh iteration yielded no cells");
    FE_THROW_IF(element_type == ElementType::Unknown, InvalidArgumentException,
                "inferUniformElementType: mesh element type is Unknown");

    return element_type;
}

std::shared_ptr<FunctionSpace> Space(SpaceType type,
                                     const assembly::IMeshAccess& mesh,
                                     int order,
                                     int components,
                                     int domain_id)
{
    const ElementType element_type = inferUniformElementType(mesh, domain_id);
    return Space(type, element_type, order, components);
}

std::shared_ptr<FunctionSpace> Space(SpaceType type,
                                     const std::shared_ptr<const assembly::IMeshAccess>& mesh,
                                     int order,
                                     int components,
                                     int domain_id)
{
    FE_CHECK_NOT_NULL(mesh.get(), "Space(SpaceType,mesh,...): mesh");
    return Space(type, *mesh, order, components, domain_id);
}

std::shared_ptr<FunctionSpace> VectorSpace(SpaceType type,
                                           const assembly::IMeshAccess& mesh,
                                           int order,
                                           int components,
                                           int domain_id)
{
    return Space(type, mesh, order, components, domain_id);
}

std::shared_ptr<FunctionSpace> VectorSpace(SpaceType type,
                                           const std::shared_ptr<const assembly::IMeshAccess>& mesh,
                                           int order,
                                           int components,
                                           int domain_id)
{
    FE_CHECK_NOT_NULL(mesh.get(), "VectorSpace(SpaceType,mesh,...): mesh");
    return VectorSpace(type, *mesh, order, components, domain_id);
}

} // namespace spaces
} // namespace FE
} // namespace svmp
