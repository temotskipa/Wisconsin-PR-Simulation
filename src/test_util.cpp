#include "util.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <process.h>
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#else
#include <unistd.h>
#endif

void test_parse_float_env() {
    std::printf("Running test_parse_float_env...\n");

    // Test default value when env var is missing
    unsetenv("TEST_FLOAT");
    assert(wisconsin_pr::ParseFloatEnv("TEST_FLOAT", 1.23f) == 1.23f);

    // Test valid float
    setenv("TEST_FLOAT", "4.56", 1);
    assert(std::fabs(wisconsin_pr::ParseFloatEnv("TEST_FLOAT", 1.23f) - 4.56f) < 1e-5f);

    // Test another valid float (scientific notation)
    setenv("TEST_FLOAT", "1.23e2", 1);
    assert(std::fabs(wisconsin_pr::ParseFloatEnv("TEST_FLOAT", 0.0f) - 123.0f) < 1e-5f);

    // Test invalid float (non-numeric)
    setenv("TEST_FLOAT", "abc", 1);
    assert(wisconsin_pr::ParseFloatEnv("TEST_FLOAT", 7.89f) == 7.89f);

    // Test invalid float (trailing characters)
    setenv("TEST_FLOAT", "1.23abc", 1);
    assert(wisconsin_pr::ParseFloatEnv("TEST_FLOAT", 7.89f) == 7.89f);

    // Test empty string
    setenv("TEST_FLOAT", "", 1);
    assert(wisconsin_pr::ParseFloatEnv("TEST_FLOAT", 7.89f) == 7.89f);

    std::printf("test_parse_float_env passed!\n");
}

void test_parse_unsigned_env() {
    std::printf("Running test_parse_unsigned_env...\n");

    unsetenv("TEST_UINT");
    assert(wisconsin_pr::ParseUnsignedEnv("TEST_UINT", 42u) == 42u);

    setenv("TEST_UINT", "123", 1);
    assert(wisconsin_pr::ParseUnsignedEnv("TEST_UINT", 42u) == 123u);

    setenv("TEST_UINT", "abc", 1);
    assert(wisconsin_pr::ParseUnsignedEnv("TEST_UINT", 42u) == 42u);

    setenv("TEST_UINT", "4294967296", 1); // 2^32, too big for 32-bit uint
    assert(wisconsin_pr::ParseUnsignedEnv("TEST_UINT", 42u) == 42u);

    std::printf("test_parse_unsigned_env passed!\n");
}

void test_parse_divisor_method_env() {
    std::printf("Running test_parse_divisor_method_env...\n");

    unsetenv("TEST_METHOD");
    assert(wisconsin_pr::ParseDivisorMethodEnv("TEST_METHOD", wisconsin_pr::SAINTE_LAGUE) == wisconsin_pr::SAINTE_LAGUE);

    setenv("TEST_METHOD", "dhondt", 1);
    assert(wisconsin_pr::ParseDivisorMethodEnv("TEST_METHOD", wisconsin_pr::SAINTE_LAGUE) == wisconsin_pr::DHONDT);

    setenv("TEST_METHOD", "SAINTE_LAGUE", 1);
    assert(wisconsin_pr::ParseDivisorMethodEnv("TEST_METHOD", wisconsin_pr::DHONDT) == wisconsin_pr::SAINTE_LAGUE);

    setenv("TEST_METHOD", "invalid", 1);
    assert(wisconsin_pr::ParseDivisorMethodEnv("TEST_METHOD", wisconsin_pr::DHONDT) == wisconsin_pr::DHONDT);

    std::printf("test_parse_divisor_method_env passed!\n");
}

int main() {
    test_parse_float_env();
    test_parse_unsigned_env();
    test_parse_divisor_method_env();
    std::printf("All tests passed!\n");
    return 0;
}
