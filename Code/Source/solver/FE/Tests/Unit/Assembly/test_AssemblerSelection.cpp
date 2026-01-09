/**
 * @file test_AssemblerSelection.cpp
 * @brief Unit tests for selection-aware assembler factory + decorator composition
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Assembly/AssemblerSelection.h"

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

TEST(AssemblerSelectionFactory, NamedStandard_NoDecorators)
{
    AssemblyOptions options{};
    FormCharacteristics form{};
    SystemCharacteristics sys{};

    auto assembler = createAssembler(options, "StandardAssembler", form, sys);
    ASSERT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->name(), "StandardAssembler");
}

TEST(AssemblerSelectionFactory, Auto_ConservativeSelectsStandard)
{
    AssemblyOptions options{};
    FormCharacteristics form{};
    SystemCharacteristics sys{};

    auto assembler = createAssembler(options, "Auto", form, sys);
    ASSERT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->name(), "StandardAssembler");
}

TEST(AssemblerSelectionFactory, Decorators_ComposeInDeterministicOrder)
{
    AssemblyOptions options{};
    options.schedule_elements = true;
    options.schedule_strategy = 0;  // Natural
    options.cache_element_data = true;
    options.use_batching = true;
    options.batch_size = 16;

    FormCharacteristics form{};
    SystemCharacteristics sys{};

    auto assembler = createAssembler(options, "StandardAssembler", form, sys);
    ASSERT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->name(), "Vectorized(Cached(Scheduled(StandardAssembler)))");
}

TEST(AssemblerSelectionFactory, DGIncompatibleSelection_Throws)
{
    AssemblyOptions options{};
    FormCharacteristics form{};
    form.has_interior_face_terms = true;
    SystemCharacteristics sys{};

    EXPECT_THROW((void)createAssembler(options, "WorkStreamAssembler", form, sys), FEException);
}

TEST(AssemblerSelectionFactory, SolutionIncompatibleSelection_Throws)
{
    AssemblyOptions options{};
    FormCharacteristics form{};
    // Only requires coefficients (no quadrature/basis), so this specifically exercises supportsSolution().
    form.required_data = RequiredData::SolutionCoefficients;
    SystemCharacteristics sys{};

    EXPECT_THROW((void)createAssembler(options, "WorkStreamAssembler", form, sys), FEException);
}

TEST(AssemblerSelectionFactory, MultiFieldOffsetIncompatibleSelection_Throws)
{
    AssemblyOptions options{};
    FormCharacteristics form{};
    SystemCharacteristics sys{};
    sys.num_fields = 2;

    EXPECT_THROW((void)createAssembler(options, "WorkStreamAssembler", form, sys), FEException);
}

TEST(AssemblerSelectionFactory, DeviceAssembler_DGRequirementAcceptedWithCpuFallback)
{
    AssemblyOptions options{};
    FormCharacteristics form{};
    form.has_interior_face_terms = true;
    SystemCharacteristics sys{};

    EXPECT_NO_THROW({
        auto assembler = createAssembler(options, "DeviceAssembler", form, sys);
        ASSERT_NE(assembler, nullptr);
        EXPECT_EQ(assembler->name(), "DeviceAssembler");
        EXPECT_TRUE(assembler->supportsDG());
    });
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
