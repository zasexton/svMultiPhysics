/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SymmetricTetrahedronQuadrature.h"
#include "TetrahedronQuadrature.h"

#include <utility>

namespace svmp {
namespace FE {
namespace quadrature {

SymmetricTetrahedronQuadrature::SymmetricTetrahedronQuadrature(int requested_order)
    : QuadratureRule(svmp::CellFamily::Tetra, 3) {

    if (requested_order < 1) {
        throw FEException("SymmetricTetrahedronQuadrature: order must be >= 1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (requested_order > max_order()) {
        throw FEException("SymmetricTetrahedronQuadrature: order > 14 not supported; use TetrahedronQuadrature",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Tabulated symmetric tetra rules are only validated up to max_tabulated_order().
    // For higher orders we fall back to the (robust) Duffy-based tensor rule.
    if (requested_order > max_tabulated_order()) {
        TetrahedronQuadrature fallback(requested_order);
        set_order(fallback.order());
        set_data(std::vector<QuadPoint>(fallback.points().begin(), fallback.points().end()),
                 std::vector<Real>(fallback.weights().begin(), fallback.weights().end()));
        return;
    }

    switch (requested_order) {
        case 1:  initialize_order_1(); break;
        case 2:  initialize_order_2(); break;
        case 3:  initialize_order_3(); break;
        case 4:  initialize_order_4(); break;
        case 5:  initialize_order_5(); break;
        case 6:  initialize_order_6(); break;
        case 7:  initialize_order_7(); break;
        case 8:  initialize_order_8(); break;
        default: initialize_order_8(); break;
    }

    set_data(std::move(pts_), std::move(wts_));
}

void SymmetricTetrahedronQuadrature::add_1_point(Real weight) {
    pts_.push_back(QuadPoint{Real(0.25), Real(0.25), Real(0.25)});
    wts_.push_back(weight);
}

void SymmetricTetrahedronQuadrature::add_4_symmetric_points(Real a, Real weight) {
    const Real b = (Real(1) - a) / Real(3);

    pts_.push_back(QuadPoint{b, b, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, b, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, a, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, b, a}); wts_.push_back(weight);
}

void SymmetricTetrahedronQuadrature::add_6_symmetric_points(Real a, Real weight) {
    const Real b = (Real(1) - Real(2) * a) / Real(2);

    pts_.push_back(QuadPoint{a, b, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, a, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, b, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, a, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, b, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, a, a}); wts_.push_back(weight);
}

void SymmetricTetrahedronQuadrature::add_12_symmetric_points(Real a, Real b, Real weight) {
    const Real c = Real(1) - Real(2) * a - b;

    pts_.push_back(QuadPoint{a, b, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, c, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, a, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, a, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, c, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, b, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, a, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, c, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, a, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, a, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, b, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, a, a}); wts_.push_back(weight);
}

void SymmetricTetrahedronQuadrature::add_24_symmetric_points(Real a, Real b, Real c, Real weight) {
    const Real d = Real(1) - a - b - c;

    pts_.push_back(QuadPoint{b, c, d}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, d, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, b, d}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, d, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{d, b, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{d, c, b}); wts_.push_back(weight);

    pts_.push_back(QuadPoint{a, c, d}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, d, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, a, d}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, d, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{d, a, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{d, c, a}); wts_.push_back(weight);

    pts_.push_back(QuadPoint{a, b, d}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, d, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, a, d}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, d, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{d, a, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{d, b, a}); wts_.push_back(weight);

    pts_.push_back(QuadPoint{a, b, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{a, c, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, a, c}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{b, c, a}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, a, b}); wts_.push_back(weight);
    pts_.push_back(QuadPoint{c, b, a}); wts_.push_back(weight);
}

// Keast rules (from "Moderate Degree Tetrahedral Quadrature Formulas", P. Keast, 1986)
// Weights are scaled for reference tetrahedron with volume 1/6.

void SymmetricTetrahedronQuadrature::initialize_order_1() {
    set_order(1);
    pts_.reserve(1);
    wts_.reserve(1);
    add_1_point(Real(1.0 / 6.0));
}

void SymmetricTetrahedronQuadrature::initialize_order_2() {
    set_order(2);
    pts_.reserve(4);
    wts_.reserve(4);
    add_4_symmetric_points(Real(0.58541019662496845446137605030969), Real(1.0 / 24.0));
}

void SymmetricTetrahedronQuadrature::initialize_order_3() {
    set_order(3);
    pts_.reserve(5);
    wts_.reserve(5);
    add_1_point(Real(-4.0 / 30.0));
    add_4_symmetric_points(Real(0.5), Real(3.0 / 40.0));
}

void SymmetricTetrahedronQuadrature::initialize_order_4() {
    set_order(4);
    pts_.reserve(11);
    wts_.reserve(11);
    add_1_point(Real(-74.0 / 5625.0));
    add_4_symmetric_points(Real(0.78571428571428571428571428571429), Real(343.0 / 45000.0));
    add_6_symmetric_points(Real(0.39940357616679920500628988229592), Real(56.0 / 2250.0));
}

void SymmetricTetrahedronQuadrature::initialize_order_5() {
    // Keast rule 7 (degree 5): 15 points
    set_order(5);
    pts_.reserve(15);
    wts_.reserve(15);
    add_1_point(Real(0.0302836780970891856));
    add_4_symmetric_points(Real(0.0), Real(0.00602678571428571597));               // face centers
    add_4_symmetric_points(Real(0.727272727272727273), Real(0.0116452490860289742)); // 8/11
    add_6_symmetric_points(Real(0.0665501535736642813), Real(0.0109491415613864534));
}

void SymmetricTetrahedronQuadrature::initialize_order_6() {
    // Keast rule 8 (degree 6): 24 points
    set_order(6);
    pts_.reserve(24);
    wts_.reserve(24);
    add_4_symmetric_points(Real(0.356191386222544953), Real(0.00665379170969464506));
    add_4_symmetric_points(Real(0.877978124396165982), Real(0.00167953517588677620));
    add_4_symmetric_points(Real(0.0329863295731730594), Real(0.00922619692394239843));
    add_12_symmetric_points(Real(0.0636610018750175299), Real(0.269672331458315867),
                            Real(0.00803571428571428248));
}

void SymmetricTetrahedronQuadrature::initialize_order_7() {
    // Keast rule 9 (degree 7): 31 points
    // Note: contains a negative weight.
    set_order(7);
    pts_.reserve(31);
    wts_.reserve(31);
    add_1_point(Real(0.0182642234661087939));
    add_4_symmetric_points(Real(0.765360423009044044), Real(0.0105999415244141609));
    add_4_symmetric_points(Real(0.634470350008286765), Real(-0.0625177401143299494));
    add_4_symmetric_points(Real(0.00238250666073834549), Real(0.00489142526307353653));
    add_6_symmetric_points(Real(0.5), Real(0.000970017636684296702)); // edge midpoints
    add_12_symmetric_points(Real(0.1), Real(0.2), Real(0.0275573192239850917));
}

void SymmetricTetrahedronQuadrature::initialize_order_8() {
    // Keast rule 10 (degree 8): 45 points
    // Note: contains a negative centroid weight.
    set_order(8);
    pts_.reserve(45);
    wts_.reserve(45);
    add_1_point(Real(-0.0393270066412926145));
    add_4_symmetric_points(Real(0.617587190300082967), Real(0.00408131605934270525));
    add_4_symmetric_points(Real(0.903763508822103123), Real(0.000658086773304341943));
    add_6_symmetric_points(Real(0.0497770956432810185), Real(0.00438425882512284693));
    add_6_symmetric_points(Real(0.183730447398549945), Real(0.0138300638425098166));
    add_12_symmetric_points(Real(0.231901089397150906), Real(0.0229177878448171174),
                            Real(0.00424043742468372453));
    add_12_symmetric_points(Real(0.0379700484718286102), Real(0.730313427807538396),
                            Real(0.00223873973961420164));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp

