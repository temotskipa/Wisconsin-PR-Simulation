#include "util.h"
#include <algorithm>
#include <string>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace wisconsin_pr {

unsigned int ParseUnsignedEnv(const char* name, const unsigned int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed > std::numeric_limits<unsigned int>::max()) {
        std::printf("Ignoring invalid %s=%s, using %u\n", name, raw, default_value);
        return default_value;
    }
    return static_cast<unsigned int>(parsed);
}

float ParseFloatEnv(const char* name, const float default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    char* end = nullptr;
    const float parsed = std::strtof(raw, &end);
    if (end == raw || *end != '\0') {
        std::printf("Ignoring invalid %s=%s, using %.3f\n", name, raw, default_value);
        return default_value;
    }
    return parsed;
}

unsigned int ParseDivisorMethodEnv(const char* name, const unsigned int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    std::string method(raw);
    std::transform(method.begin(), method.end(), method.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (method == "sainte_lague" || method == "sainte-lague" || method == "saintelague") {
        return SAINTE_LAGUE;
    }
    if (method == "dhondt" || method == "d_hondt" || method == "d-hondt" || method == "d'hondt") {
        return DHONDT;
    }
    std::printf("Ignoring invalid %s=%s, using %s\n", name, raw,
        default_value == DHONDT ? "dhondt" : "sainte_lague");
    return default_value;
}

}  // namespace wisconsin_pr
