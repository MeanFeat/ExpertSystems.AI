#include <gtest/gtest.h>
#include "es_test.h"

#define GTEST_RUN(suite, name, call) \
	TEST(suite, name) { \
		testResult rst = call; \
		EXPECT_TRUE(rst.passed) << "\n" << rst.message; \
	}

#include GENERATED_GTEST_TESTS
