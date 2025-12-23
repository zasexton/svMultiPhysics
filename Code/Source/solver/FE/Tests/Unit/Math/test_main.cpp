/**
 * @file test_main.cpp
 * @brief Main entry point for running all FE Math module unit tests
 */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>

// Custom test environment for setup/teardown
class MathTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        std::cout << "========================================\n";
        std::cout << "FE Math Module Unit Tests\n";
        std::cout << "========================================\n";

        // Check SIMD availability
        CheckSIMDSupport();

        start_time = std::chrono::high_resolution_clock::now();
    }

    void TearDown() override {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "========================================\n";
        std::cout << "Total test execution time: " << duration.count() << " ms\n";
        std::cout << "========================================\n";
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;

    void CheckSIMDSupport() {
        std::cout << "SIMD Support Detection:\n";

        #ifdef __AVX__
        std::cout << "  AVX: Supported\n";
        #else
        std::cout << "  AVX: Not supported\n";
        #endif

        #ifdef __AVX2__
        std::cout << "  AVX2: Supported\n";
        #else
        std::cout << "  AVX2: Not supported\n";
        #endif

        #ifdef __AVX512F__
        std::cout << "  AVX512: Supported\n";
        #else
        std::cout << "  AVX512: Not supported\n";
        #endif

        #ifdef __FMA__
        std::cout << "  FMA: Supported\n";
        #else
        std::cout << "  FMA: Not supported\n";
        #endif

        std::cout << "========================================\n";
    }
};

// Custom test listener for detailed output
class DetailedTestListener : public ::testing::TestEventListener {
public:
    DetailedTestListener(TestEventListener* default_listener)
        : default_listener_(default_listener) {}

    void OnTestStart(const ::testing::TestInfo& test_info) override {
        default_listener_->OnTestStart(test_info);
    }

    void OnTestPartResult(const ::testing::TestPartResult& result) override {
        default_listener_->OnTestPartResult(result);
    }

    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        default_listener_->OnTestEnd(test_info);

        // Print timing information for slow tests
        if (test_info.result()->elapsed_time() > 100) {  // More than 100ms
            std::cout << "  [SLOW] " << test_info.test_case_name() << "."
                      << test_info.name() << " took "
                      << test_info.result()->elapsed_time() << " ms\n";
        }
    }

    void OnTestCaseStart(const ::testing::TestCase& test_case) override {
        default_listener_->OnTestCaseStart(test_case);
    }

    void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsSetUpStart(unit_test);
    }

    void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsSetUpEnd(unit_test);
    }

    void OnTestCaseEnd(const ::testing::TestCase& test_case) override {
        default_listener_->OnTestCaseEnd(test_case);
    }

    void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsTearDownStart(unit_test);
    }

    void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsTearDownEnd(unit_test);
    }

    void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override {
        default_listener_->OnTestIterationStart(unit_test, iteration);
    }

    void OnTestProgramStart(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnTestProgramStart(unit_test);
    }

    void OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) override {
        default_listener_->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnTestProgramEnd(unit_test);
    }

private:
    TestEventListener* default_listener_;
};

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Add custom environment
    ::testing::AddGlobalTestEnvironment(new MathTestEnvironment);

    // Add custom listener for detailed output (optional)
    if (argc > 1 && std::string(argv[1]) == "--verbose") {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        auto* default_listener = listeners.default_result_printer();
        listeners.Release(default_listener);
        listeners.Append(new DetailedTestListener(default_listener));
    }

    // Parse additional command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            std::cout << "FE Math Module Unit Tests\n";
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --help          Show this help message\n";
            std::cout << "  --verbose       Enable verbose output\n";
            std::cout << "  --gtest_filter=PATTERN  Run only tests matching pattern\n";
            std::cout << "  --gtest_repeat=N        Repeat tests N times\n";
            std::cout << "  --gtest_shuffle         Shuffle test order\n";
            std::cout << "  --gtest_random_seed=N   Set random seed\n";
            return 0;
        }
    }

    // Run all tests
    return RUN_ALL_TESTS();
}