#ifndef SVMP_FE_TESTS_UNIT_FORMS_PERF_TEST_HELPERS_H
#define SVMP_FE_TESTS_UNIT_FORMS_PERF_TEST_HELPERS_H

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

[[nodiscard]] inline bool perfTestsEnabled()
{
    const char* v = std::getenv("SVMP_FE_RUN_PERF_TESTS");
    return v != nullptr && std::string_view(v) == "1";
}

namespace detail {

using Clock = std::chrono::steady_clock;

template <class Fn>
[[nodiscard]] inline double timeSeconds(Fn&& fn)
{
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

[[nodiscard]] inline int getenvInt(std::string_view name, int default_value)
{
    const char* v = std::getenv(std::string(name).c_str());
    if (v == nullptr) return default_value;
    char* end = nullptr;
    const long out = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    if (out < std::numeric_limits<int>::min()) return default_value;
    if (out > std::numeric_limits<int>::max()) return default_value;
    return static_cast<int>(out);
}

[[nodiscard]] inline double getenvDouble(std::string_view name, double default_value)
{
    const char* v = std::getenv(std::string(name).c_str());
    if (v == nullptr) return default_value;
    char* end = nullptr;
    const double out = std::strtod(v, &end);
    if (end == v) return default_value;
    return out;
}

template <class Fn>
[[nodiscard]] inline double bestOfSeconds(int repeats, Fn&& fn)
{
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < repeats; ++r) {
        best = std::min(best, timeSeconds(fn));
    }
    return best;
}

} // namespace detail
} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TESTS_UNIT_FORMS_PERF_TEST_HELPERS_H
