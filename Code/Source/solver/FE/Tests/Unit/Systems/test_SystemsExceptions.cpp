/**
 * @file test_SystemsExceptions.cpp
 * @brief Unit tests for Systems exception types
 */

#include <gtest/gtest.h>

#include "Systems/SystemsExceptions.h"

#include <string>

using svmp::FE::FEException;
using svmp::FE::systems::InvalidStateException;
using svmp::FE::systems::SystemsException;

TEST(SystemsExceptions, SystemsException_InheritsFromFEException)
{
    try {
        throw SystemsException("msg");
    } catch (const FEException&) {
        SUCCEED();
    } catch (...) {
        FAIL();
    }
}

TEST(SystemsExceptions, SystemsException_ContainsMessage)
{
    try {
        throw SystemsException("hello systems");
    } catch (const FEException& e) {
        EXPECT_NE(std::string(e.what()).find("hello systems"), std::string::npos);
    }
}

TEST(SystemsExceptions, InvalidStateException_InheritsFromFEException)
{
    try {
        throw InvalidStateException("msg");
    } catch (const FEException&) {
        SUCCEED();
    } catch (...) {
        FAIL();
    }
}

TEST(SystemsExceptions, InvalidStateException_ContainsMessage)
{
    try {
        throw InvalidStateException("bad state");
    } catch (const FEException& e) {
        EXPECT_NE(std::string(e.what()).find("bad state"), std::string::npos);
    }
}

