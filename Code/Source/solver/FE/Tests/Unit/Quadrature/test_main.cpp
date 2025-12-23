/**
 * @file test_main.cpp
 * @brief Google Test main entry point for Quadrature unit tests
 */

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
