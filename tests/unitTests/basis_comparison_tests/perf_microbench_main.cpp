// Standalone tight-loop binary for hardware-counter profiling.
//
// Usage: perf_microbench_one <ELEMENT> <OP> <ITERATIONS>
//   ELEMENT: TRI3, TRI6, TET4, TET10, HEX8, HEX27
//   OP:      values, gradients, hessians, all
//   ITERATIONS: positive integer (e.g., 5000000)
//
// Designed to be wrapped with `perf stat -e ...` so that the measured cycles,
// instructions, cache misses, and branch misses correspond to the inner
// evaluate loop with minimal program-startup contamination.

#include "FE/Basis/LagrangeBasis.h"
#include "FE/Core/Types.h"
#include "Math/Vector.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Defeat dead-code elimination.
volatile double g_sink = 0.0;

struct Case {
    const char* name;
    svmp::FE::ElementType type;
    int order;
    double xi[3];  // representative interior point
};

Case lookup_case(const std::string& name) {
    if (name == "TRI3")  return {"TRI3",  svmp::FE::ElementType::Triangle3, 1, {1.0/3, 1.0/3, 0.0}};
    if (name == "TRI6")  return {"TRI6",  svmp::FE::ElementType::Triangle6, 2, {1.0/3, 1.0/3, 0.0}};
    if (name == "TET4")  return {"TET4",  svmp::FE::ElementType::Tetra4,    1, {0.25, 0.25, 0.25}};
    if (name == "TET10") return {"TET10", svmp::FE::ElementType::Tetra10,   2, {0.25, 0.25, 0.25}};
    if (name == "HEX8")  return {"HEX8",  svmp::FE::ElementType::Hex8,      1, {0.0, 0.0, 0.0}};
    if (name == "HEX27") return {"HEX27", svmp::FE::ElementType::Hex27,     2, {0.0, 0.0, 0.0}};
    std::fprintf(stderr, "Unknown element: %s\n", name.c_str());
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s ELEMENT OP ITERATIONS\n", argv[0]);
        return 2;
    }
    Case cs = lookup_case(argv[1]);
    std::string op = argv[2];
    long iterations = std::atol(argv[3]);
    if (iterations <= 0) {
        std::fprintf(stderr, "ITERATIONS must be positive.\n");
        return 2;
    }

    auto basis = std::make_unique<svmp::FE::basis::LagrangeBasis>(cs.type, cs.order);
    svmp::FE::math::Vector<svmp::FE::Real, 3> pt;
    pt[0] = cs.xi[0]; pt[1] = cs.xi[1]; pt[2] = cs.xi[2];

    std::vector<svmp::FE::Real> values(basis->size());
    std::vector<svmp::FE::basis::Gradient> gradients(basis->size());
    std::vector<svmp::FE::basis::Hessian> hessians(basis->size());

    // Warm-up to populate caches.
    for (int i = 0; i < 1000; ++i) {
        basis->evaluate_values(pt, values);
        basis->evaluate_gradients(pt, gradients);
        basis->evaluate_hessians(pt, hessians);
        g_sink += values[0] + gradients[0][0] + hessians[0](0, 0);
    }

    // Hot loop — this is what perf stat measures.
    auto t0 = Clock::now();
    if (op == "values") {
        for (long i = 0; i < iterations; ++i) {
            basis->evaluate_values(pt, values);
            g_sink += values[0];
        }
    } else if (op == "gradients") {
        for (long i = 0; i < iterations; ++i) {
            basis->evaluate_gradients(pt, gradients);
            g_sink += gradients[0][0];
        }
    } else if (op == "hessians") {
        for (long i = 0; i < iterations; ++i) {
            basis->evaluate_hessians(pt, hessians);
            g_sink += hessians[0](0, 0);
        }
    } else if (op == "all") {
        for (long i = 0; i < iterations; ++i) {
            basis->evaluate_all(pt, values, gradients, hessians);
            g_sink += values[0] + gradients[0][0] + hessians[0](0, 0);
        }
    } else {
        std::fprintf(stderr, "Unknown op: %s\n", op.c_str());
        return 2;
    }
    auto t1 = Clock::now();

    double seconds = std::chrono::duration<double>(t1 - t0).count();
    double ns_per_call = seconds * 1e9 / static_cast<double>(iterations);
    std::printf("# %s %s iterations=%ld seconds=%.6f ns_per_call=%.3f sink=%.3e\n",
                cs.name, op.c_str(), iterations, seconds, ns_per_call,
                static_cast<double>(g_sink));
    return 0;
}
