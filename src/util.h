#ifndef SRC_UTIL_H_
#define SRC_UTIL_H_

#include <string>

namespace wisconsin_pr {

enum DivisorMethod : unsigned int {
    SAINTE_LAGUE = 0u,
    DHONDT = 1u,
};

unsigned int ParseUnsignedEnv(const char* name, const unsigned int default_value);
float ParseFloatEnv(const char* name, const float default_value);
unsigned int ParseDivisorMethodEnv(const char* name, const unsigned int default_value);

}  // namespace wisconsin_pr

#endif  // SRC_UTIL_H_
