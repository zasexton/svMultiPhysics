/**
 * @file test_main.cpp
 * @brief Main entry point for running all FE Dofs module unit tests
 */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>

// Custom test environment for setup/teardown
class DofsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        std::cout << "========================================\n";
        std::cout << "FE Dofs Module Unit Tests\n";
        std::cout << "========================================\n";
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
};

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Add custom environment
    ::testing::AddGlobalTestEnvironment(new DofsTestEnvironment);

    // Parse additional command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            std::cout << "FE Dofs Module Unit Tests\n";
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --help          Show this help message\n";
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
