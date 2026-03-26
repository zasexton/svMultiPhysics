/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/FormExprScanner.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void scanNode(const forms::FormExprNode& node, FormExprScanResult& result) {
    using FT = forms::FormExprType;

    switch (node.type()) {
        case FT::TimeDerivative:
            result.has_time_derivative = true;
            break;
        case FT::CellDiameter:
            result.has_cell_diameter = true;
            break;
        case FT::Jump:
            result.has_jump = true;
            break;
        case FT::Average:
            result.has_average = true;
            break;
        case FT::CellIntegral:
            result.has_cell_integral = true;
            break;
        case FT::BoundaryIntegral: {
            result.has_boundary_integral = true;
            auto marker = node.boundaryMarker();
            if (marker && *marker >= 0) {
                if (std::find(result.boundary_markers.begin(),
                              result.boundary_markers.end(), *marker)
                    == result.boundary_markers.end()) {
                    result.boundary_markers.push_back(*marker);
                }
            }
            break;
        }
        case FT::InteriorFaceIntegral:
            result.has_interior_face_integral = true;
            break;
        case FT::InterfaceIntegral: {
            result.has_interface_integral = true;
            auto marker = node.interfaceMarker();
            if (marker && *marker >= 0) {
                if (std::find(result.interface_markers.begin(),
                              result.interface_markers.end(), *marker)
                    == result.interface_markers.end()) {
                    result.interface_markers.push_back(*marker);
                }
            }
            break;
        }
        case FT::BoundaryIntegralSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.boundary_functional_names.begin(),
                              result.boundary_functional_names.end(), s)
                    == result.boundary_functional_names.end()) {
                    result.boundary_functional_names.push_back(std::move(s));
                }
            }
            break;
        }
        case FT::AuxiliaryStateSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.auxiliary_state_names.begin(),
                              result.auxiliary_state_names.end(), s)
                    == result.auxiliary_state_names.end()) {
                    result.auxiliary_state_names.push_back(std::move(s));
                }
            }
            break;
        }
        case FT::AuxiliaryInputSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.auxiliary_input_names.begin(),
                              result.auxiliary_input_names.end(), s)
                    == result.auxiliary_input_names.end()) {
                    result.auxiliary_input_names.push_back(std::move(s));
                }
            }
            break;
        }
        case FT::AuxiliaryOutputSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.auxiliary_output_names.begin(),
                              result.auxiliary_output_names.end(), s)
                    == result.auxiliary_output_names.end()) {
                    result.auxiliary_output_names.push_back(std::move(s));
                }
            }
            break;
        }
        default:
            break;
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            scanNode(*child, result);
        }
    }
}

} // namespace

FormExprScanResult scanFormExpr(const forms::FormExprNode& root) {
    FormExprScanResult result;
    scanNode(root, result);
    return result;
}

std::vector<DomainKind> FormExprScanResult::activeDomains() const {
    std::vector<DomainKind> domains;
    if (has_cell_integral) domains.push_back(DomainKind::Cell);
    if (has_boundary_integral) domains.push_back(DomainKind::Boundary);
    if (has_interior_face_integral) domains.push_back(DomainKind::InteriorFace);
    if (has_interface_integral) domains.push_back(DomainKind::InterfaceFace);
    if (!boundary_functional_names.empty() || !auxiliary_state_names.empty()) {
        // Presence of coupled-boundary symbols implies a coupled boundary domain
        if (std::find(domains.begin(), domains.end(), DomainKind::CoupledBoundary)
            == domains.end()) {
            domains.push_back(DomainKind::CoupledBoundary);
        }
    }
    // If nothing was detected (e.g., no explicit integral wrappers), default to Cell
    if (domains.empty()) domains.push_back(DomainKind::Cell);
    return domains;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
