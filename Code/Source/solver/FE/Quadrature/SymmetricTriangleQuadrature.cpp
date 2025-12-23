/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SymmetricTriangleQuadrature.h"

namespace svmp {
namespace FE {
namespace quadrature {

SymmetricTriangleQuadrature::SymmetricTriangleQuadrature(int requested_order)
    : QuadratureRule(svmp::CellFamily::Triangle, 2) {

    if (requested_order < 1) {
        throw FEException("SymmetricTriangleQuadrature: order must be >= 1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (requested_order > max_order()) {
        throw FEException("SymmetricTriangleQuadrature: order > 20 not supported; use TriangleQuadrature",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Select appropriate rule based on requested order
    switch (requested_order) {
        case 1:  initialize_order_1();  break;
        case 2:  initialize_order_2();  break;
        case 3:  initialize_order_3();  break;
        case 4:  initialize_order_4();  break;
        case 5:  initialize_order_5();  break;
        case 6:  initialize_order_6();  break;
        case 7:  initialize_order_7();  break;
        case 8:  initialize_order_8();  break;
        case 9:  initialize_order_9();  break;
        case 10: initialize_order_10(); break;
        case 11: initialize_order_11(); break;
        case 12: initialize_order_12(); break;
        case 13: initialize_order_13(); break;
        case 14: initialize_order_14(); break;
        case 15: initialize_order_15(); break;
        case 16: initialize_order_16(); break;
        case 17: initialize_order_17(); break;
        case 18: initialize_order_18(); break;
        case 19: initialize_order_19(); break;
        case 20: initialize_order_20(); break;
        default: initialize_order_20(); break;
    }

    set_data(std::move(pts_), std::move(wts_));
}

void SymmetricTriangleQuadrature::add_centroid_point(Real weight) {
    // Centroid at (1/3, 1/3) in reference coordinates
    pts_.push_back(QuadPoint{Real(1.0/3.0), Real(1.0/3.0), Real(0)});
    wts_.push_back(weight);
}

void SymmetricTriangleQuadrature::add_3_symmetric_points(Real a, Real weight) {
    // Three points with barycentric coordinates (a, a, 1-2a) and permutations
    // Convert to reference coordinates (xi, eta) where xi + eta <= 1
    const Real b = a;
    const Real c = Real(1) - Real(2) * a;

    // Point 1: barycentric (a, b, c) -> reference (b, c)
    pts_.push_back(QuadPoint{b, c, Real(0)});
    wts_.push_back(weight);

    // Point 2: barycentric (c, a, b) -> reference (a, b)
    pts_.push_back(QuadPoint{a, b, Real(0)});
    wts_.push_back(weight);

    // Point 3: barycentric (b, c, a) -> reference (c, a)
    pts_.push_back(QuadPoint{c, a, Real(0)});
    wts_.push_back(weight);
}

void SymmetricTriangleQuadrature::add_6_symmetric_points(Real a, Real b, Real weight) {
    // Six points with barycentric coordinates (a, b, c) and all permutations
    // where c = 1 - a - b
    const Real c = Real(1) - a - b;

    // All 6 permutations of (a, b, c)
    // Reference coords (xi, eta) correspond to barycentric coords (lambda_2, lambda_3)
    // where lambda_1 = 1 - xi - eta

    pts_.push_back(QuadPoint{b, c, Real(0)});
    wts_.push_back(weight);

    pts_.push_back(QuadPoint{c, b, Real(0)});
    wts_.push_back(weight);

    pts_.push_back(QuadPoint{a, c, Real(0)});
    wts_.push_back(weight);

    pts_.push_back(QuadPoint{c, a, Real(0)});
    wts_.push_back(weight);

    pts_.push_back(QuadPoint{a, b, Real(0)});
    wts_.push_back(weight);

    pts_.push_back(QuadPoint{b, a, Real(0)});
    wts_.push_back(weight);
}

// Dunavant rules (from "High Degree Efficient Symmetrical Gaussian
// Quadrature Rules for the Triangle", D.A. Dunavant, 1985)
// Weights are for reference triangle with area 0.5

void SymmetricTriangleQuadrature::initialize_order_1() {
    set_order(1);
    pts_.reserve(1);
    wts_.reserve(1);
    add_centroid_point(Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_2() {
    set_order(2);
    pts_.reserve(3);
    wts_.reserve(3);
    add_3_symmetric_points(Real(1.0/6.0), Real(1.0/6.0));
}

void SymmetricTriangleQuadrature::initialize_order_3() {
    set_order(3);
    pts_.reserve(4);
    wts_.reserve(4);
    add_centroid_point(Real(-9.0/32.0));
    add_3_symmetric_points(Real(0.2), Real(25.0/96.0));
}

void SymmetricTriangleQuadrature::initialize_order_4() {
    set_order(4);
    pts_.reserve(6);
    wts_.reserve(6);
    add_3_symmetric_points(Real(0.44594849091596488631832925388305),
                           Real(0.22338158967801146569500700843312) * Real(0.5));
    add_3_symmetric_points(Real(0.09157621350977074345957146340220),
                           Real(0.10995174365532186763832632490021) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_5() {
    set_order(5);
    pts_.reserve(7);
    wts_.reserve(7);
    add_centroid_point(Real(0.225) * Real(0.5));
    add_3_symmetric_points(Real(0.47014206410511508977044120951345),
                           Real(0.13239415278850618073764938783315) * Real(0.5));
    add_3_symmetric_points(Real(0.10128650732345633880098736191512),
                           Real(0.12593918054482715259568394550018) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_6() {
    set_order(6);
    pts_.reserve(12);
    wts_.reserve(12);
    add_3_symmetric_points(Real(0.24928674517091042129163855310702),
                           Real(0.11678627572637936602528961138558) * Real(0.5));
    add_3_symmetric_points(Real(0.06308901449150222834033160287082),
                           Real(0.05084490637020681692093680910686) * Real(0.5));
    add_6_symmetric_points(Real(0.31035245103378440541660773395655),
                           Real(0.63650249912139864723014259441205),
                           Real(0.08285107561837357519355345642044) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_7() {
    set_order(7);
    pts_.reserve(13);
    wts_.reserve(13);
    add_centroid_point(Real(-0.14957004446767497031448264617551) * Real(0.5));
    add_3_symmetric_points(Real(0.26034596607904134570479766426679),
                           Real(0.17561525743321691348266882420661) * Real(0.5));
    add_3_symmetric_points(Real(0.06513010290221623036887891059948),
                           Real(0.05334723560883960230296261044945) * Real(0.5));
    add_6_symmetric_points(Real(0.31286549600487084236305913996091),
                           Real(0.63844418856981280478426570854075),
                           Real(0.07711376089026831199824523496780) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_8() {
    set_order(8);
    pts_.reserve(16);
    wts_.reserve(16);
    add_centroid_point(Real(0.14431560767778716825109111048906) * Real(0.5));
    add_3_symmetric_points(Real(0.17056930775176020662229350149146),
                           Real(0.10321737053471825028179155029212) * Real(0.5));
    add_3_symmetric_points(Real(0.05054722831703097545842355059660),
                           Real(0.03245849762319808031092592834178) * Real(0.5));
    add_3_symmetric_points(Real(0.45929258829272315602881551449417),
                           Real(0.09509163426728462479389610438858) * Real(0.5));
    add_6_symmetric_points(Real(0.26311282963463811342178578628464),
                           Real(0.72849239295540428124100037917606),
                           Real(0.02723031417443499426484469007390) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_9() {
    set_order(9);
    pts_.reserve(19);
    wts_.reserve(19);
    add_centroid_point(Real(0.09713579628279609890744676309485) * Real(0.5));
    add_3_symmetric_points(Real(0.48968251919873762778370692483619),
                           Real(0.03133470022713983234393199080984) * Real(0.5));
    add_3_symmetric_points(Real(0.43708959149293663726993036443535),
                           Real(0.07782754100477543338465495857972) * Real(0.5));
    add_3_symmetric_points(Real(0.18820353561903273024096128046733),
                           Real(0.07964773892720910288013526957424) * Real(0.5));
    add_3_symmetric_points(Real(0.04472951339445297061024247196780),
                           Real(0.02557767565869810438673914467637) * Real(0.5));
    add_6_symmetric_points(Real(0.22196298916076569567510252769319),
                           Real(0.74119859878449802069007987352342),
                           Real(0.04328353937728937728937728937729) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_10() {
    // Dunavant rule 10: 25 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    set_order(10);
    pts_.reserve(25);
    wts_.reserve(25);
    add_centroid_point(Real(0.090817990382754) * Real(0.5));
    add_3_symmetric_points(Real(0.485577633383657),
                           Real(0.036725957756467) * Real(0.5));
    add_3_symmetric_points(Real(0.109481575485037),
                           Real(0.045321059435528) * Real(0.5));
    add_6_symmetric_points(Real(0.141707219414880), Real(0.307939838764121),
                           Real(0.072757916845420) * Real(0.5));
    add_6_symmetric_points(Real(0.025003534762686), Real(0.246672560639903),
                           Real(0.028327242531057) * Real(0.5));
    add_6_symmetric_points(Real(0.009540815400299), Real(0.066803251012200),
                           Real(0.009421666963733) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_11() {
    // Dunavant rule 11: 27 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    // Note: This rule has points slightly outside the triangle (negative coord)
    set_order(11);
    pts_.reserve(27);
    wts_.reserve(27);
    add_3_symmetric_points(Real(0.534611048270758),
                           Real(0.000927006328961) * Real(0.5));
    add_3_symmetric_points(Real(0.398969302965855),
                           Real(0.077149534914813) * Real(0.5));
    add_3_symmetric_points(Real(0.203309900431282),
                           Real(0.059322977380774) * Real(0.5));
    add_3_symmetric_points(Real(0.119350912282581),
                           Real(0.036184540503418) * Real(0.5));
    add_3_symmetric_points(Real(0.032364948111276),
                           Real(0.013659731002678) * Real(0.5));
    add_6_symmetric_points(Real(0.050178138310495), Real(0.356620648261293),
                           Real(0.052337111962204) * Real(0.5));
    add_6_symmetric_points(Real(0.021022016536166), Real(0.171488980304042),
                           Real(0.020707659639141) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_12() {
    // Dunavant rule 12: 33 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    set_order(12);
    pts_.reserve(33);
    wts_.reserve(33);
    add_3_symmetric_points(Real(0.488217389773805),
                           Real(0.025731066440455) * Real(0.5));
    add_3_symmetric_points(Real(0.439724392294460),
                           Real(0.043692544538038) * Real(0.5));
    add_3_symmetric_points(Real(0.271210385012116),
                           Real(0.062858224217885) * Real(0.5));
    add_3_symmetric_points(Real(0.127576145541586),
                           Real(0.034796112930709) * Real(0.5));
    add_3_symmetric_points(Real(0.021317350453210),
                           Real(0.006166261051559) * Real(0.5));
    add_6_symmetric_points(Real(0.115343494534698), Real(0.275713269685514),
                           Real(0.040371557766381) * Real(0.5));
    add_6_symmetric_points(Real(0.022838332222257), Real(0.281325580989940),
                           Real(0.022356773202303) * Real(0.5));
    add_6_symmetric_points(Real(0.025734050548330), Real(0.116251915907597),
                           Real(0.017316231108659) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_13() {
    // Dunavant rule 13: 37 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    set_order(13);
    pts_.reserve(37);
    wts_.reserve(37);
    add_centroid_point(Real(0.052520923400802) * Real(0.5));
    add_3_symmetric_points(Real(0.495048184939705),
                           Real(0.011280145209330) * Real(0.5));
    add_3_symmetric_points(Real(0.468716635109574),
                           Real(0.031423518362454) * Real(0.5));
    add_3_symmetric_points(Real(0.414521336801277),
                           Real(0.047072502504194) * Real(0.5));
    add_3_symmetric_points(Real(0.229399572042831),
                           Real(0.047363586536355) * Real(0.5));
    add_3_symmetric_points(Real(0.114424495196330),
                           Real(0.031167529045794) * Real(0.5));
    add_3_symmetric_points(Real(0.024811391363459),
                           Real(0.007975771465074) * Real(0.5));
    add_6_symmetric_points(Real(0.094853828379579), Real(0.268794997058761),
                           Real(0.036848402728732) * Real(0.5));
    add_6_symmetric_points(Real(0.018100773278807), Real(0.291730066734288),
                           Real(0.017401463303822) * Real(0.5));
    add_6_symmetric_points(Real(0.022233076674090), Real(0.126357385491669),
                           Real(0.015521786839045) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_14() {
    // Dunavant rule 14: 42 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    set_order(14);
    pts_.reserve(42);
    wts_.reserve(42);
    add_3_symmetric_points(Real(0.488963910362179),
                           Real(0.021883581369429) * Real(0.5));
    add_3_symmetric_points(Real(0.417644719340454),
                           Real(0.032788353544125) * Real(0.5));
    add_3_symmetric_points(Real(0.273477528308839),
                           Real(0.051774104507292) * Real(0.5));
    add_3_symmetric_points(Real(0.177205532412543),
                           Real(0.042162588736993) * Real(0.5));
    add_3_symmetric_points(Real(0.061799883090873),
                           Real(0.014433699669777) * Real(0.5));
    add_3_symmetric_points(Real(0.019390961248701),
                           Real(0.004923403602400) * Real(0.5));
    add_6_symmetric_points(Real(0.057124757403648), Real(0.172266687821356),
                           Real(0.024665753212564) * Real(0.5));
    add_6_symmetric_points(Real(0.092916249356972), Real(0.336861459796345),
                           Real(0.038571510787061) * Real(0.5));
    add_6_symmetric_points(Real(0.014646950055654), Real(0.298372882136258),
                           Real(0.014436308113534) * Real(0.5));
    add_6_symmetric_points(Real(0.001268330932872), Real(0.118974497696957),
                           Real(0.005010228838501) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_15() {
    // Dunavant rule 15: 48 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    // Note: This rule has some points slightly outside the triangle
    set_order(15);
    pts_.reserve(48);
    wts_.reserve(48);
    add_3_symmetric_points(Real(0.506972916858243),
                           Real(0.001916875642849) * Real(0.5));
    add_3_symmetric_points(Real(0.431406354283023),
                           Real(0.044249027271145) * Real(0.5));
    add_3_symmetric_points(Real(0.277693644847144),
                           Real(0.051186548718852) * Real(0.5));
    add_3_symmetric_points(Real(0.126464891041254),
                           Real(0.023687735870688) * Real(0.5));
    add_3_symmetric_points(Real(0.070808385974686),
                           Real(0.013289775690021) * Real(0.5));
    add_3_symmetric_points(Real(0.018965170241073),
                           Real(0.004748916608192) * Real(0.5));
    add_6_symmetric_points(Real(0.133734161966621), Real(0.261311371140087),
                           Real(0.038550072599593) * Real(0.5));
    add_6_symmetric_points(Real(0.036366677396917), Real(0.388046767090269),
                           Real(0.027215814320624) * Real(0.5));
    add_6_symmetric_points(Real(-0.010174883126571), Real(0.285712220049916),
                           Real(0.002182077366797) * Real(0.5));
    add_6_symmetric_points(Real(0.036843869875878), Real(0.215599664072284),
                           Real(0.021505319847731) * Real(0.5));
    add_6_symmetric_points(Real(0.012459809331199), Real(0.103575616576386),
                           Real(0.007673942631049) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_16() {
    // Dunavant rule 16: 52 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    // Note: This rule has some points slightly outside the triangle
    set_order(16);
    pts_.reserve(52);
    wts_.reserve(52);
    add_centroid_point(Real(0.046875697427642) * Real(0.5));
    add_3_symmetric_points(Real(0.497380541948438),
                           Real(0.006405878578585) * Real(0.5));
    add_3_symmetric_points(Real(0.413469438549352),
                           Real(0.041710296739387) * Real(0.5));
    add_3_symmetric_points(Real(0.470458599066991),
                           Real(0.026891484250064) * Real(0.5));
    add_3_symmetric_points(Real(0.240553749969521),
                           Real(0.042132522761650) * Real(0.5));
    add_3_symmetric_points(Real(0.147965794222573),
                           Real(0.030000266842773) * Real(0.5));
    add_3_symmetric_points(Real(0.075465187657474),
                           Real(0.014200098925024) * Real(0.5));
    add_3_symmetric_points(Real(0.016596402623025),
                           Real(0.003582462351273) * Real(0.5));
    add_6_symmetric_points(Real(0.103575692245252), Real(0.296555596579887),
                           Real(0.032773147460627) * Real(0.5));
    add_6_symmetric_points(Real(0.020083411655416), Real(0.337723063403079),
                           Real(0.015298306248441) * Real(0.5));
    add_6_symmetric_points(Real(-0.004341002614139), Real(0.204748281642812),
                           Real(0.002386244192839) * Real(0.5));
    add_6_symmetric_points(Real(0.041941786468010), Real(0.189358492130623),
                           Real(0.019084792755899) * Real(0.5));
    add_6_symmetric_points(Real(0.014317320230681), Real(0.085283615682657),
                           Real(0.006850054546542) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_17() {
    // Dunavant rule 17: 61 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    set_order(17);
    pts_.reserve(61);
    wts_.reserve(61);
    add_centroid_point(Real(0.033437199290803) * Real(0.5));
    add_3_symmetric_points(Real(0.497170540556774),
                           Real(0.005093415440507) * Real(0.5));
    add_3_symmetric_points(Real(0.482176322624625),
                           Real(0.014670864527638) * Real(0.5));
    add_3_symmetric_points(Real(0.450239969020782),
                           Real(0.024350878353672) * Real(0.5));
    add_3_symmetric_points(Real(0.400266239377397),
                           Real(0.031107550868969) * Real(0.5));
    add_3_symmetric_points(Real(0.252141267970953),
                           Real(0.031257111218620) * Real(0.5));
    add_3_symmetric_points(Real(0.162047004658461),
                           Real(0.024815654339665) * Real(0.5));
    add_3_symmetric_points(Real(0.075875882260746),
                           Real(0.014056073070557) * Real(0.5));
    add_3_symmetric_points(Real(0.015654726967822),
                           Real(0.003194676173779) * Real(0.5));
    add_6_symmetric_points(Real(0.010186928826919), Real(0.334319867363658),
                           Real(0.008119655318993) * Real(0.5));
    add_6_symmetric_points(Real(0.135440871671036), Real(0.292221537796944),
                           Real(0.026805742283163) * Real(0.5));
    add_6_symmetric_points(Real(0.054423924290583), Real(0.319574885423190),
                           Real(0.018459993210822) * Real(0.5));
    add_6_symmetric_points(Real(0.012868560833637), Real(0.190704224192292),
                           Real(0.008476868534328) * Real(0.5));
    add_6_symmetric_points(Real(0.067165782413524), Real(0.180483211648746),
                           Real(0.018292796770025) * Real(0.5));
    add_6_symmetric_points(Real(0.014663182224828), Real(0.080711313679564),
                           Real(0.006665632004165) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_18() {
    // Dunavant rule 18: 70 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    // Note: This rule has negative weights and points outside the triangle
    set_order(18);
    pts_.reserve(70);
    wts_.reserve(70);
    add_centroid_point(Real(0.030809939937647) * Real(0.5));
    add_3_symmetric_points(Real(0.493344808630921),
                           Real(0.009072436679404) * Real(0.5));
    add_3_symmetric_points(Real(0.469210594241957),
                           Real(0.018761316939594) * Real(0.5));
    add_3_symmetric_points(Real(0.436281395887006),
                           Real(0.019441097985477) * Real(0.5));
    add_3_symmetric_points(Real(0.394846170673416),
                           Real(0.027753948610810) * Real(0.5));
    add_3_symmetric_points(Real(0.249794568803157),
                           Real(0.032256225351457) * Real(0.5));
    add_3_symmetric_points(Real(0.161432193743843),
                           Real(0.025074032616922) * Real(0.5));
    add_3_symmetric_points(Real(0.076598227485371),
                           Real(0.015271927971832) * Real(0.5));
    add_3_symmetric_points(Real(0.024252439353450),
                           Real(0.006793922022963) * Real(0.5));
    add_3_symmetric_points(Real(0.043146367216965),
                           Real(-0.002223098729920) * Real(0.5));  // Negative weight
    add_6_symmetric_points(Real(0.008430536202420), Real(0.358911494940944),
                           Real(0.006331914076406) * Real(0.5));
    add_6_symmetric_points(Real(0.131186551737188), Real(0.294402476751957),
                           Real(0.027257538049138) * Real(0.5));
    add_6_symmetric_points(Real(0.050203151565675), Real(0.325017801641814),
                           Real(0.017676785649465) * Real(0.5));
    add_6_symmetric_points(Real(0.066329263810916), Real(0.184737559666046),
                           Real(0.018379484638070) * Real(0.5));
    add_6_symmetric_points(Real(0.011996194566236), Real(0.218796800013321),
                           Real(0.008104732808192) * Real(0.5));
    add_6_symmetric_points(Real(0.014858100590125), Real(0.101179597136408),
                           Real(0.007634129070725) * Real(0.5));
    add_6_symmetric_points(Real(-0.035222015287949), Real(0.020874755282586),
                           Real(0.000046187660794) * Real(0.5));  // Point outside triangle
}

void SymmetricTriangleQuadrature::initialize_order_19() {
    // Dunavant rule 19: 73 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    set_order(19);
    pts_.reserve(73);
    wts_.reserve(73);
    add_centroid_point(Real(0.032906331388919) * Real(0.5));
    add_3_symmetric_points(Real(0.489609987073006),
                           Real(0.010330731891272) * Real(0.5));
    add_3_symmetric_points(Real(0.454536892697893),
                           Real(0.022387247263016) * Real(0.5));
    add_3_symmetric_points(Real(0.401416680649431),
                           Real(0.030266125869468) * Real(0.5));
    add_3_symmetric_points(Real(0.255551654403098),
                           Real(0.030490967802198) * Real(0.5));
    add_3_symmetric_points(Real(0.177077942152130),
                           Real(0.024159212741641) * Real(0.5));
    add_3_symmetric_points(Real(0.110061053227952),
                           Real(0.016050803586801) * Real(0.5));
    add_3_symmetric_points(Real(0.055528624251840),
                           Real(0.008084580261784) * Real(0.5));
    add_3_symmetric_points(Real(0.012621863777229),
                           Real(0.002079362027485) * Real(0.5));
    add_6_symmetric_points(Real(0.003611417848412), Real(0.395754787356943),
                           Real(0.003884876904981) * Real(0.5));
    add_6_symmetric_points(Real(0.134466754530780), Real(0.307929983880436),
                           Real(0.025574160612022) * Real(0.5));
    add_6_symmetric_points(Real(0.014446025776115), Real(0.264566948406520),
                           Real(0.008880903573338) * Real(0.5));
    add_6_symmetric_points(Real(0.046933578838178), Real(0.358539352205951),
                           Real(0.016124546761731) * Real(0.5));
    add_6_symmetric_points(Real(0.002861120350567), Real(0.157807405968595),
                           Real(0.002491941817491) * Real(0.5));
    add_6_symmetric_points(Real(0.223861424097916), Real(0.075050596975911),
                           Real(0.018242840118951) * Real(0.5));
    add_6_symmetric_points(Real(0.034647074816760), Real(0.142421601113383),
                           Real(0.010258563736199) * Real(0.5));
    add_6_symmetric_points(Real(0.010161119296278), Real(0.065494628082938),
                           Real(0.003799928855302) * Real(0.5));
}

void SymmetricTriangleQuadrature::initialize_order_20() {
    // Dunavant rule 20: 79 points
    // Reference: J. Burkardt's triangle_dunavant_rule implementation
    // Note: This rule has negative weights and points outside the triangle
    set_order(20);
    pts_.reserve(79);
    wts_.reserve(79);
    add_centroid_point(Real(0.033057055541624) * Real(0.5));
    add_3_symmetric_points(Real(0.500950464352200),
                           Real(0.000867019185663) * Real(0.5));  // Point outside triangle
    add_3_symmetric_points(Real(0.488212957934729),
                           Real(0.011660052716448) * Real(0.5));
    add_3_symmetric_points(Real(0.455136681950283),
                           Real(0.022876936356421) * Real(0.5));
    add_3_symmetric_points(Real(0.401996259318289),
                           Real(0.030448982673938) * Real(0.5));
    add_3_symmetric_points(Real(0.255892909759421),
                           Real(0.030624891725355) * Real(0.5));
    add_3_symmetric_points(Real(0.176488255995106),
                           Real(0.024368057676800) * Real(0.5));
    add_3_symmetric_points(Real(0.104170855336758),
                           Real(0.015997432032024) * Real(0.5));
    add_3_symmetric_points(Real(0.053068963840930),
                           Real(0.007698301815602) * Real(0.5));
    add_3_symmetric_points(Real(0.041618715196029),
                           Real(-0.000632060497488) * Real(0.5));  // Negative weight
    add_3_symmetric_points(Real(0.011581921406822),
                           Real(0.001751134301193) * Real(0.5));
    add_6_symmetric_points(Real(0.048741583664839), Real(0.344855770229001),
                           Real(0.016465839189576) * Real(0.5));
    add_6_symmetric_points(Real(0.006314115948605), Real(0.377843269594854),
                           Real(0.004839033540485) * Real(0.5));
    add_6_symmetric_points(Real(0.134316520547348), Real(0.306635479062357),
                           Real(0.025804906534650) * Real(0.5));
    add_6_symmetric_points(Real(0.013973893962392), Real(0.249419362774742),
                           Real(0.008471091054441) * Real(0.5));
    add_6_symmetric_points(Real(0.075549132909764), Real(0.212775724802802),
                           Real(0.018354914106280) * Real(0.5));
    add_6_symmetric_points(Real(-0.008368153208227), Real(0.146965436053239),
                           Real(0.000704404677908) * Real(0.5));  // Point outside triangle
    add_6_symmetric_points(Real(0.026686063258714), Real(0.137726978828923),
                           Real(0.010112684927462) * Real(0.5));
    add_6_symmetric_points(Real(0.010547719294141), Real(0.059696109149007),
                           Real(0.003573909385950) * Real(0.5));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
