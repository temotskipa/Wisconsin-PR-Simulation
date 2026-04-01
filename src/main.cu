#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flamegpu/flamegpu.h"

namespace wisconsin_pr {

constexpr char kModelName[] = "Wisconsin PR Simulation";
constexpr unsigned int kDefaultSeats = 99;
constexpr float kDefaultThreshold = 0.05f;
constexpr unsigned int kDefaultRandomSeed = 42;
constexpr unsigned int kDefaultVoterCount = 5970000u;
constexpr unsigned int kDefaultActivistCount = 2500u;
constexpr float kDefaultMinorEntryShare = 0.020f;
constexpr unsigned int kDefaultCampaignSteps = 6u;
constexpr unsigned int kAbstain = 999u;
constexpr float kPi = 3.14159265358979323846f;

enum PartyId : unsigned int {
    DEM = 0u,
    REP = 1u,
    GREEN = 2u,
    LIBERTARIAN = 3u,
    CENTRIST = 4u,
    POPULIST = 5u,
    PARTY_COUNT = 6u,
};

enum DivisorMethod : unsigned int {
    SAINTE_LAGUE = 0u,
    DHONDT = 1u,
};

enum RegionId : unsigned int {
    MILWAUKEE_METRO = 0u,
    DANE_METRO = 1u,
    WOW_SUBURBS = 2u,
    SOUTHEAST_INDUSTRIAL = 3u,
    FOX_VALLEY = 4u,
    DRIFTLESS_WEST = 5u,
    NORTHWOODS = 6u,
    RURAL_HEARTLAND = 7u,
    REGION_COUNT = 8u,
};

constexpr unsigned int kRegionPartySlots = REGION_COUNT * PARTY_COUNT;

struct FamilyAggregate {
    unsigned int count = 0u;
    double ideology_sum = 0.0;
    double organizer_skill_sum = 0.0;
    double donor_access_sum = 0.0;
    double launch_tendency_sum = 0.0;
    double field_reach_sum = 0.0;
};

struct PartyState {
    unsigned int party_id = 0u;
    float ideology = 0.0f;
    float organization = 0.0f;
    float brand = 0.0f;
    float credibility = 0.0f;
    float projected_viability = 0.0f;
    float field_strength = 0.0f;
    float momentum = 0.0f;
    float fundraising = 0.0f;
    float media_reach = 0.0f;
};

struct PartyResult {
    unsigned int party_id = 0u;
    std::uint64_t votes = 0u;
    float share = 0.0f;
    unsigned int seats = 0u;
};

struct SimulationArtifactConfig {
    unsigned int total_voters;
    unsigned int total_valid;
    unsigned int abstentions;
    float turnout_share;
    float threshold;
    unsigned int seats;
    unsigned int divisor_method;
    unsigned int campaign_steps;
};

struct RegionProfile {
    const char* label;
    float population_share;
    float ideology_shift;
    float urbanity;
    float turnout_bonus;
    float college_share;
    float union_share;
    float religiosity;
    float green_affinity;
    float libertarian_affinity;
    float populist_affinity;
    float activist_density;
};

constexpr std::array<RegionProfile, REGION_COUNT> kRegionProfiles = {{
    {"Milwaukee Metro", 0.16f, -0.25f, 0.95f, -0.01f, 0.34f, 0.23f, 0.34f, 0.44f, 0.09f, 0.15f, 1.25f},
    {"Dane / Madison", 0.09f, -0.35f, 0.90f, 0.04f, 0.55f, 0.16f, 0.23f, 0.58f, 0.11f, 0.08f, 1.30f},
    {"WOW Suburbs", 0.08f, 0.33f, 0.58f, 0.05f, 0.44f, 0.12f, 0.60f, 0.11f, 0.20f, 0.15f, 0.85f},
    {"Southeast Industrial", 0.13f, 0.02f, 0.52f, 0.03f, 0.31f, 0.24f, 0.48f, 0.17f, 0.10f, 0.26f, 1.05f},
    {"Fox Valley", 0.14f, 0.12f, 0.44f, 0.03f, 0.29f, 0.18f, 0.56f, 0.14f, 0.17f, 0.23f, 0.95f},
    {"Driftless West", 0.12f, -0.10f, 0.32f, 0.02f, 0.28f, 0.19f, 0.46f, 0.20f, 0.12f, 0.19f, 0.95f},
    {"Northwoods", 0.10f, 0.18f, 0.18f, 0.05f, 0.22f, 0.18f, 0.54f, 0.09f, 0.13f, 0.30f, 0.80f},
    {"Rural Heartland", 0.18f, 0.09f, 0.24f, 0.03f, 0.23f, 0.17f, 0.55f, 0.12f, 0.14f, 0.27f, 0.75f},
}};

FLAMEGPU_HOST_DEVICE_FUNCTION float ClampFloat(const float value, const float lower, const float upper) {
    return value < lower ? lower : (value > upper ? upper : value);
}

std::string SanitizeEnvForLog(const char* raw) {
    if (!raw) return "";
    const size_t kMaxLogLen = 128;
    std::string sanitized;
    sanitized.reserve(kMaxLogLen + 4);
    for (size_t i = 0; raw[i] != '\0'; ++i) {
        if (i >= kMaxLogLen) {
            sanitized += "...";
            break;
        }
        const unsigned char c = static_cast<unsigned char>(raw[i]);
        sanitized += std::isprint(c) ? static_cast<char>(c) : '?';
    }
    return sanitized;
}

unsigned int ParseUnsignedEnv(const char* name, const unsigned int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed > std::numeric_limits<unsigned int>::max()) {
        std::printf("Ignoring invalid %s=%s, using %u\n", name, SanitizeEnvForLog(raw).c_str(), default_value);
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
        std::printf("Ignoring invalid %s=%s, using %.3f\n", name, SanitizeEnvForLog(raw).c_str(), default_value);
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
    std::printf("Ignoring invalid %s=%s, using %s\n", name, SanitizeEnvForLog(raw).c_str(),
        default_value == DHONDT ? "dhondt" : "sainte_lague");
    return default_value;
}

const char* PartyLabel(const unsigned int party_id) {
    switch (party_id) {
        case DEM: return "Democratic";
        case REP: return "Republican";
        case GREEN: return "Green";
        case LIBERTARIAN: return "Libertarian";
        case CENTRIST: return "Forward/Centrist";
        case POPULIST: return "Rural Populist";
        default: return "Unknown";
    }
}

const char* PartyColor(const unsigned int party_id) {
    switch (party_id) {
        case DEM: return "#2563eb";
        case REP: return "#dc2626";
        case GREEN: return "#15803d";
        case LIBERTARIAN: return "#ca8a04";
        case CENTRIST: return "#0891b2";
        case POPULIST: return "#b45309";
        default: return "#6b7280";
    }
}

unsigned int PartySeatOrder(const unsigned int party_id) {
    switch (party_id) {
        case GREEN: return 0u;
        case DEM: return 1u;
        case CENTRIST: return 2u;
        case POPULIST: return 3u;
        case REP: return 4u;
        case LIBERTARIAN: return 5u;
        default: return PARTY_COUNT;
    }
}

FLAMEGPU_HOST_DEVICE_FUNCTION bool IsMajorParty(const unsigned int party_id) {
    return party_id == DEM || party_id == REP;
}

const char* DivisorMethodLabel(const unsigned int method) {
    return method == DHONDT ? "dhondt" : "sainte_lague";
}

unsigned int ChoosePartyFamily(const float ideology, const float anti_establishment) {
    if (fabsf(ideology) < 0.12f) {
        if (anti_establishment > 0.74f) {
            return POPULIST;
        }
        return CENTRIST;
    }
    if (ideology < -0.62f && anti_establishment > 0.58f) {
        return GREEN;
    }
    if (ideology > 0.70f && anti_establishment > 0.68f) {
        return LIBERTARIAN;
    }
    if (anti_establishment > 0.80f) {
        return POPULIST;
    }
    return ideology < 0.0f ? DEM : REP;
}

double DivisorValue(const unsigned int seats_already_awarded, const unsigned int method) {
    if (method == DHONDT) {
        return static_cast<double>(seats_already_awarded + 1u);
    }
    return static_cast<double>(2u * seats_already_awarded + 1u);
}

std::unordered_map<unsigned int, unsigned int> AllocateDivisorSeats(
    const std::vector<std::pair<unsigned int, std::uint64_t>>& qualified_votes,
    const unsigned int total_seats,
    const unsigned int method) {
    std::unordered_map<unsigned int, unsigned int> seat_counts;
    seat_counts.reserve(qualified_votes.size());
    for (const auto& [party_id, _] : qualified_votes) {
        seat_counts.emplace(party_id, 0u);
    }

    for (unsigned int seat = 0u; seat < total_seats; ++seat) {
        unsigned int best_party = qualified_votes.front().first;
        double best_quotient = -1.0;
        for (const auto& [party_id, votes] : qualified_votes) {
            const unsigned int seats_awarded = seat_counts[party_id];
            const double quotient = static_cast<double>(votes) / DivisorValue(seats_awarded, method);
            if (quotient > best_quotient) {
                best_quotient = quotient;
                best_party = party_id;
            }
        }
        ++seat_counts[best_party];
    }

    return seat_counts;
}

FLAMEGPU_HOST_DEVICE_FUNCTION unsigned int FlattenRegionPartyIndex(const unsigned int region_id, const unsigned int party_id) {
    return region_id * PARTY_COUNT + party_id;
}

unsigned int ChooseRegion(const float draw, const bool activist_weighted) {
    float total_weight = 0.0f;
    for (const RegionProfile& region : kRegionProfiles) {
        total_weight += region.population_share * (activist_weighted ? region.activist_density : 1.0f);
    }

    float cumulative = 0.0f;
    for (unsigned int region_id = 0u; region_id < REGION_COUNT; ++region_id) {
        const float weight = kRegionProfiles[region_id].population_share
            * (activist_weighted ? kRegionProfiles[region_id].activist_density : 1.0f);
        cumulative += weight / total_weight;
        if (draw <= cumulative) {
            return region_id;
        }
    }
    return REGION_COUNT - 1u;
}

float AverageOrZero(const double sum, const unsigned int count) {
    return count > 0u ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
}

std::array<float, kRegionPartySlots> ZeroRegionPartyContact() {
    std::array<float, kRegionPartySlots> data{};
    data.fill(0.0f);
    return data;
}

template <typename ActivistRange>
std::array<FamilyAggregate, kRegionPartySlots> SummarizeRegionalFamilies(
    const ActivistRange& activists,
    std::array<unsigned int, REGION_COUNT>& region_totals) {
    region_totals.fill(0u);
    std::array<FamilyAggregate, kRegionPartySlots> families{};

    for (const auto& activist : activists) {
        const unsigned int region_id = activist.template getVariable<unsigned int>("home_region");
        const unsigned int family = activist.template getVariable<unsigned int>("preferred_family");
        if (region_id >= REGION_COUNT || family >= PARTY_COUNT) {
            continue;
        }

        ++region_totals[region_id];
        FamilyAggregate& aggregate = families[FlattenRegionPartyIndex(region_id, family)];
        ++aggregate.count;
        aggregate.ideology_sum += activist.template getVariable<float>("ideology");
        aggregate.organizer_skill_sum += activist.template getVariable<float>("organizer_skill");
        aggregate.donor_access_sum += activist.template getVariable<float>("donor_access");
        aggregate.launch_tendency_sum += activist.template getVariable<float>("launch_tendency");
        aggregate.field_reach_sum += activist.template getVariable<float>("field_reach");
    }

    return families;
}

std::array<FamilyAggregate, PARTY_COUNT> CollapseStatewideFamilies(
    const std::array<FamilyAggregate, kRegionPartySlots>& regional_families) {
    std::array<FamilyAggregate, PARTY_COUNT> statewide{};
    for (unsigned int region_id = 0u; region_id < REGION_COUNT; ++region_id) {
        for (unsigned int party_id = 0u; party_id < PARTY_COUNT; ++party_id) {
            const FamilyAggregate& regional = regional_families[FlattenRegionPartyIndex(region_id, party_id)];
            FamilyAggregate& aggregate = statewide[party_id];
            aggregate.count += regional.count;
            aggregate.ideology_sum += regional.ideology_sum;
            aggregate.organizer_skill_sum += regional.organizer_skill_sum;
            aggregate.donor_access_sum += regional.donor_access_sum;
            aggregate.launch_tendency_sum += regional.launch_tendency_sum;
            aggregate.field_reach_sum += regional.field_reach_sum;
        }
    }
    return statewide;
}

float ComputeRegionalBreadth(
    const std::array<FamilyAggregate, kRegionPartySlots>& regional_families,
    const std::array<unsigned int, REGION_COUNT>& region_totals,
    const unsigned int party_id) {
    unsigned int active_regions = 0u;
    for (unsigned int region_id = 0u; region_id < REGION_COUNT; ++region_id) {
        const unsigned int total = region_totals[region_id];
        if (total == 0u) {
            continue;
        }
        const FamilyAggregate& regional = regional_families[FlattenRegionPartyIndex(region_id, party_id)];
        const float local_share = static_cast<float>(regional.count) / static_cast<float>(total);
        if (regional.count >= 4u && local_share >= 0.05f) {
            ++active_regions;
        }
    }
    return static_cast<float>(active_regions) / static_cast<float>(REGION_COUNT);
}

template <typename PartyRange>
std::unordered_map<unsigned int, PartyState> SnapshotParties(const PartyRange& parties) {
    std::unordered_map<unsigned int, PartyState> party_states;
    party_states.reserve(PARTY_COUNT);

    for (const auto& party : parties) {
        const unsigned int party_id = party.template getVariable<unsigned int>("party_id");
        party_states.emplace(party_id, PartyState{
            party_id,
            party.template getVariable<float>("ideology"),
            party.template getVariable<float>("organization"),
            party.template getVariable<float>("brand"),
            party.template getVariable<float>("credibility"),
            party.template getVariable<float>("projected_viability"),
            party.template getVariable<float>("field_strength"),
            party.template getVariable<float>("momentum"),
            party.template getVariable<float>("fundraising"),
            party.template getVariable<float>("media_reach"),
        });
    }

    return party_states;
}

std::array<float, kRegionPartySlots> BuildRegionPartyContacts(
    const std::array<FamilyAggregate, kRegionPartySlots>& regional_families,
    const std::array<unsigned int, REGION_COUNT>& region_totals,
    const std::unordered_map<unsigned int, PartyState>& party_states) {
    std::array<float, kRegionPartySlots> contacts = ZeroRegionPartyContact();

    for (unsigned int region_id = 0u; region_id < REGION_COUNT; ++region_id) {
        const RegionProfile& region = kRegionProfiles[region_id];
        const float total_activists = static_cast<float>(std::max(1u, region_totals[region_id]));

        for (unsigned int party_id = 0u; party_id < PARTY_COUNT; ++party_id) {
            const FamilyAggregate& aggregate = regional_families[FlattenRegionPartyIndex(region_id, party_id)];
            const float family_share = static_cast<float>(aggregate.count) / total_activists;
            const float avg_skill = AverageOrZero(aggregate.organizer_skill_sum, aggregate.count);
            const float avg_donor = AverageOrZero(aggregate.donor_access_sum, aggregate.count);
            const float avg_launch = AverageOrZero(aggregate.launch_tendency_sum, aggregate.count);
            const float avg_reach = AverageOrZero(aggregate.field_reach_sum, aggregate.count);

            float contact = 0.0f;
            switch (party_id) {
                case DEM:
                    contact = ClampFloat(
                        0.30f + 0.32f * region.urbanity + 0.18f * region.union_share - 0.16f * std::max(region.ideology_shift, 0.0f)
                        + 2.0f * family_share + 0.10f * avg_skill,
                        0.0f,
                        1.0f);
                    break;
                case REP:
                    contact = ClampFloat(
                        0.30f + 0.24f * (1.0f - region.urbanity) + 0.18f * region.religiosity + 0.18f * std::max(region.ideology_shift, 0.0f)
                        + 2.0f * family_share + 0.10f * avg_skill,
                        0.0f,
                        1.0f);
                    break;
                case GREEN:
                    contact = ClampFloat(
                        0.01f + 2.4f * family_share + 0.16f * region.green_affinity + 0.10f * avg_skill + 0.05f * avg_donor,
                        0.0f,
                        1.0f);
                    break;
                case LIBERTARIAN:
                    contact = ClampFloat(
                        0.01f + 2.1f * family_share + 0.14f * region.libertarian_affinity + 0.08f * avg_skill + 0.06f * avg_donor,
                        0.0f,
                        1.0f);
                    break;
                case CENTRIST:
                    contact = ClampFloat(
                        0.01f + 2.8f * family_share + 0.12f * region.college_share + 0.10f * (1.0f - fabsf(region.ideology_shift))
                        + 0.07f * avg_launch,
                        0.0f,
                        1.0f);
                    break;
                case POPULIST:
                    contact = ClampFloat(
                        0.01f + 2.7f * family_share + 0.12f * region.populist_affinity + 0.10f * region.union_share
                        + 0.08f * avg_reach,
                        0.0f,
                        1.0f);
                    break;
                default:
                    break;
            }

            const auto state_it = party_states.find(party_id);
            if (state_it != party_states.end()) {
                const PartyState& state = state_it->second;
                const bool major_party = IsMajorParty(party_id);
                contact = ClampFloat(
                    contact * (major_party
                        ? (0.64f + 0.24f * state.momentum + 0.20f * state.field_strength
                            + 0.16f * state.projected_viability + 0.16f * state.fundraising + 0.14f * state.media_reach)
                        : (0.56f + 0.16f * state.momentum + 0.14f * state.field_strength
                            + 0.10f * state.projected_viability + 0.10f * state.fundraising + 0.08f * state.media_reach)),
                    0.0f,
                    1.0f);
            }

            contacts[FlattenRegionPartyIndex(region_id, party_id)] = contact;
        }
    }

    return contacts;
}

float WeightedPartyContact(const std::array<float, kRegionPartySlots>& contacts, const unsigned int party_id) {
    float weighted_average = 0.0f;
    for (unsigned int region_id = 0u; region_id < REGION_COUNT; ++region_id) {
        weighted_average += kRegionProfiles[region_id].population_share
            * contacts[FlattenRegionPartyIndex(region_id, party_id)];
    }
    return ClampFloat(weighted_average, 0.0f, 1.0f);
}

template <typename PartyRange>
std::vector<PartyResult> BuildPartyResults(
    flamegpu::HostAgentAPI& voters,
    const PartyRange& parties,
    const unsigned int total_valid,
    const float threshold,
    std::vector<std::pair<unsigned int, std::uint64_t>>* qualified_votes = nullptr) {
    std::vector<PartyResult> results;
    results.reserve(parties.size());

    if (qualified_votes) {
        qualified_votes->clear();
        qualified_votes->reserve(parties.size());
    }

    for (const auto& party : parties) {
        const unsigned int party_id = party.template getVariable<unsigned int>("party_id");
        const std::uint64_t votes = static_cast<std::uint64_t>(voters.count<unsigned int>("vote_choice", party_id));
        const float share = total_valid > 0u ? static_cast<float>(votes) / static_cast<float>(total_valid) : 0.0f;
        results.push_back(PartyResult{party_id, votes, share, 0u});
        if (qualified_votes && share >= threshold) {
            qualified_votes->emplace_back(party_id, votes);
        }
    }

    std::sort(results.begin(), results.end(), [](const PartyResult& lhs, const PartyResult& rhs) {
        if (lhs.votes != rhs.votes) {
            return lhs.votes > rhs.votes;
        }
        return lhs.party_id < rhs.party_id;
    });
    return results;
}

void PrintCampaignCheckpoint(
    const std::vector<PartyResult>& results,
    const unsigned int step,
    const unsigned int campaign_steps,
    const unsigned int total_valid) {
    std::printf("Campaign step %u/%u, valid votes=%u: ", step + 1u, campaign_steps, total_valid);
    const unsigned int printed = std::min<unsigned int>(4u, static_cast<unsigned int>(results.size()));
    for (unsigned int i = 0u; i < printed; ++i) {
        const PartyResult& result = results[i];
        std::printf("%s %.1f%%%s",
            PartyLabel(result.party_id),
            result.share * 100.0f,
            i + 1u == printed ? "\n" : " | ");
    }
}

std::filesystem::path ResolveReportDirectory() {
    const char* raw = std::getenv("WISCONSIN_PR_REPORT_DIR");
    if (raw && *raw) {
        return std::filesystem::path(raw);
    }
    return std::filesystem::path("reports");
}

std::vector<PartyResult> BuildParliamentOrdering(const std::vector<PartyResult>& results) {
    std::vector<PartyResult> ordered = results;
    std::sort(ordered.begin(), ordered.end(), [](const PartyResult& lhs, const PartyResult& rhs) {
        const unsigned int lhs_order = PartySeatOrder(lhs.party_id);
        const unsigned int rhs_order = PartySeatOrder(rhs.party_id);
        if (lhs_order != rhs_order) {
            return lhs_order < rhs_order;
        }
        return lhs.party_id < rhs.party_id;
    });
    return ordered;
}

std::string BuildParliamentSvg(
    const std::vector<PartyResult>& results,
    const SimulationArtifactConfig& config) {
    const float width = 1100.0f;
    const float height = 760.0f;
    const float center_x = 390.0f;
    const float center_y = 610.0f;
    const float inner_radius = 120.0f;
    const float ring_gap = 42.0f;
    const float seat_radius = 9.0f;
    const unsigned int rows = 5u;
    const unsigned int majority = config.seats / 2u + 1u;

    std::vector<float> radii(rows);
    float total_weight = 0.0f;
    for (unsigned int row = 0u; row < rows; ++row) {
        radii[row] = inner_radius + ring_gap * static_cast<float>(row);
        total_weight += radii[row];
    }

    std::vector<unsigned int> row_counts(rows, 1u);
    unsigned int assigned_seats = 0u;
    for (unsigned int row = 0u; row < rows; ++row) {
        row_counts[row] = std::max(1u, static_cast<unsigned int>(std::lround(
            static_cast<float>(config.seats) * radii[row] / total_weight)));
        assigned_seats += row_counts[row];
    }
    while (assigned_seats > config.seats) {
        const auto max_it = std::max_element(row_counts.begin(), row_counts.end());
        if (*max_it <= 1u) {
            break;
        }
        --(*max_it);
        --assigned_seats;
    }
    while (assigned_seats < config.seats) {
        const auto min_it = std::min_element(row_counts.begin(), row_counts.end());
        ++(*min_it);
        assigned_seats++;
    }

    std::vector<unsigned int> seat_parties;
    seat_parties.reserve(config.seats);
    for (const PartyResult& result : BuildParliamentOrdering(results)) {
        for (unsigned int seat = 0u; seat < result.seats; ++seat) {
            seat_parties.push_back(result.party_id);
        }
    }
    while (seat_parties.size() < config.seats) {
        seat_parties.push_back(kAbstain);
    }

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
        << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>\n";
    svg << "<text x=\"56\" y=\"58\" font-size=\"30\" font-family=\"Segoe UI, Arial, sans-serif\" fill=\"#0f172a\">Wisconsin PR Simulation</text>\n";
    svg << "<text x=\"56\" y=\"92\" font-size=\"15\" font-family=\"Segoe UI, Arial, sans-serif\" fill=\"#475569\">"
        << "Parliament chart, turnout " << std::fixed << std::setprecision(1) << config.turnout_share * 100.0f
        << "%, threshold " << config.threshold * 100.0f << "%, " << DivisorMethodLabel(config.divisor_method) << "</text>\n";
    svg << "<text x=\"56\" y=\"126\" font-size=\"14\" font-family=\"Segoe UI, Arial, sans-serif\" fill=\"#334155\">"
        << "Majority: " << majority << " seats</text>\n";

    unsigned int seat_index = 0u;
    for (unsigned int row = 0u; row < rows; ++row) {
        const unsigned int seats_in_row = row_counts[row];
        const float radius = radii[row];
        for (unsigned int seat = 0u; seat < seats_in_row && seat_index < seat_parties.size(); ++seat, ++seat_index) {
            const float angle = kPi
                - kPi * (static_cast<float>(seat) + 0.5f) / static_cast<float>(seats_in_row);
            const float x = center_x + radius * std::cos(angle);
            const float y = center_y - radius * std::sin(angle);
            svg << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"" << seat_radius
                << "\" fill=\"" << PartyColor(seat_parties[seat_index]) << "\" stroke=\"#ffffff\" stroke-width=\"1.6\"/>\n";
        }
    }

    svg << "<path d=\"M " << center_x - radii.back() - 12.0f << " " << center_y
        << " A " << radii.back() + 12.0f << " " << radii.back() + 12.0f
        << " 0 0 1 " << center_x + radii.back() + 12.0f << " " << center_y
        << "\" fill=\"none\" stroke=\"#cbd5e1\" stroke-width=\"2\" stroke-dasharray=\"5 6\"/>\n";

    float legend_y = 170.0f;
    for (const PartyResult& result : results) {
        svg << "<rect x=\"650\" y=\"" << legend_y - 12.0f << "\" width=\"16\" height=\"16\" rx=\"3\" fill=\""
            << PartyColor(result.party_id) << "\"/>\n";
        svg << "<text x=\"676\" y=\"" << legend_y + 1.0f
            << "\" font-size=\"15\" font-family=\"Segoe UI, Arial, sans-serif\" fill=\"#0f172a\">"
            << PartyLabel(result.party_id) << "  " << std::fixed << std::setprecision(1) << result.share * 100.0f
            << "%  " << result.seats << " seats</text>\n";
        legend_y += 30.0f;
    }

    svg << "</svg>\n";
    return svg.str();
}

void WriteResultsArtifacts(
    const std::vector<PartyResult>& results,
    const SimulationArtifactConfig& config) {
    const std::filesystem::path report_dir = ResolveReportDirectory();
    std::filesystem::create_directories(report_dir);

    const std::filesystem::path svg_path = report_dir / "wisconsin_pr_results.svg";
    const std::filesystem::path html_path = report_dir / "wisconsin_pr_results.html";

    const std::string svg = BuildParliamentSvg(results, config);
    {
        std::ofstream svg_file(svg_path, std::ios::out | std::ios::trunc);
        svg_file << svg;
    }

    std::ostringstream html;
    html << "<!doctype html><html><head><meta charset=\"utf-8\"><title>Wisconsin PR Simulation</title>"
         << "<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:24px;}"
         << ".wrap{display:grid;grid-template-columns:minmax(420px,1.2fr) minmax(300px,0.8fr);gap:24px;align-items:start;}"
         << ".card{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px 22px;box-shadow:0 12px 28px rgba(15,23,42,0.08);}"
         << "h1{margin:0 0 8px;font-size:30px;}table{width:100%;border-collapse:collapse;}th,td{padding:10px 0;border-bottom:1px solid #e2e8f0;text-align:left;}"
         << ".swatch{display:inline-block;width:12px;height:12px;border-radius:999px;margin-right:8px;vertical-align:middle;}"
         << ".meta{display:grid;grid-template-columns:repeat(2,minmax(140px,1fr));gap:12px 18px;margin:16px 0 6px;}"
         << ".meta div{background:#f8fafc;border-radius:14px;padding:10px 12px;}</style></head><body>";
    html << "<div class=\"wrap\"><div class=\"card\"><h1>Wisconsin PR Simulation</h1><p>"
         << "Campaign steps " << config.campaign_steps << ", threshold " << std::fixed << std::setprecision(1) << config.threshold * 100.0f
         << "%, divisor " << DivisorMethodLabel(config.divisor_method) << ".</p>" << svg << "</div>";
    html << "<div class=\"card\"><h2>Summary</h2><div class=\"meta\">"
         << "<div><strong>Total voters</strong><br>" << config.total_voters << "</div>"
         << "<div><strong>Valid votes</strong><br>" << config.total_valid << "</div>"
         << "<div><strong>Abstentions</strong><br>" << config.abstentions << "</div>"
         << "<div><strong>Turnout</strong><br>" << std::fixed << std::setprecision(2) << config.turnout_share * 100.0f << "%</div>"
         << "</div><table><thead><tr><th>Party</th><th>Votes</th><th>Share</th><th>Seats</th></tr></thead><tbody>";
    for (const PartyResult& result : results) {
        html << "<tr><td><span class=\"swatch\" style=\"background:" << PartyColor(result.party_id) << "\"></span>"
             << PartyLabel(result.party_id) << "</td><td>" << result.votes << "</td><td>"
             << std::fixed << std::setprecision(2) << result.share * 100.0f << "%</td><td>" << result.seats << "</td></tr>";
    }
    html << "</tbody></table></div></div></body></html>";

    {
        std::ofstream html_file(html_path, std::ios::out | std::ios::trunc);
        html_file << html.str();
    }

    const std::string svg_out = svg_path.string();
    const std::string html_out = html_path.string();
    std::printf("Report SVG: %s\n", svg_out.c_str());
    std::printf("Report HTML: %s\n", html_out.c_str());
}

FLAMEGPU_INIT_FUNCTION(SeedVoters) {
    flamegpu::HostAgentAPI voters = FLAMEGPU->agent("voter");
    const unsigned int voter_count = FLAMEGPU->environment.getProperty<unsigned int>("VOTER_COUNT");
    const unsigned int random_seed = FLAMEGPU->environment.getProperty<unsigned int>("RANDOM_SEED");

    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::normal_distribution<float> voter_noise(0.0f, 0.05f);
    std::normal_distribution<float> urban_noise(0.0f, 0.08f);
    std::normal_distribution<float> trait_noise(0.0f, 0.06f);

    for (unsigned int i = 0u; i < voter_count; ++i) {
        flamegpu::HostNewAgentAPI voter = voters.newAgent();
        const unsigned int region_id = ChooseRegion(unit(rng), false);
        const RegionProfile& region = kRegionProfiles[region_id];

        float ideology = 0.0f;
        const float cluster_draw = unit(rng);
        if (cluster_draw < 0.29f) {
            ideology = -0.50f + 0.14f * voter_noise(rng) / 0.05f;
        } else if (cluster_draw < 0.65f) {
            ideology = 0.17f * voter_noise(rng) / 0.05f;
        } else {
            ideology = 0.47f + 0.13f * voter_noise(rng) / 0.05f;
        }
        ideology = ClampFloat(ideology + 0.82f * region.ideology_shift, -1.0f, 1.0f);

        const float college_prob = ClampFloat(region.college_share + 0.08f * (1.0f - fabsf(ideology)), 0.02f, 0.95f);
        const float union_prob = ClampFloat(region.union_share + 0.06f * (ideology < 0.0f ? 1.0f : 0.0f), 0.01f, 0.90f);
        const unsigned int college = unit(rng) < college_prob ? 1u : 0u;
        const unsigned int union_member = unit(rng) < union_prob ? 1u : 0u;

        const float urbanity = ClampFloat(region.urbanity - 0.10f * ideology + urban_noise(rng), 0.0f, 1.0f);
        const float religiosity = ClampFloat(region.religiosity + trait_noise(rng), 0.0f, 1.0f);
        const float sophistication = ClampFloat(
            0.16f + 0.38f * fabsf(ideology) + 0.12f * static_cast<float>(college) + 0.15f * region.college_share + 0.12f * unit(rng),
            0.0f,
            1.0f);
        const float anti_est = ClampFloat(
            0.12f + 0.32f * fabsf(ideology) + 0.12f * region.populist_affinity + 0.08f * (1.0f - region.urbanity) + 0.10f * unit(rng),
            0.0f,
            1.0f);
        const float turnout = ClampFloat(
            0.22f + 0.34f * fabsf(ideology) + region.turnout_bonus + 0.07f * static_cast<float>(college)
            + 0.04f * static_cast<float>(union_member) + voter_noise(rng),
            0.0f,
            1.0f);
        const float green_affinity = ClampFloat(
            region.green_affinity + 0.14f * static_cast<float>(college) + 0.08f * urbanity
            - 0.10f * religiosity - 0.08f * std::max(ideology, 0.0f) + trait_noise(rng),
            0.0f,
            1.0f);
        const float libertarian_affinity = ClampFloat(
            region.libertarian_affinity + 0.14f * std::max(ideology, 0.0f) + 0.10f * sophistication
            - 0.08f * static_cast<float>(union_member) - 0.05f * urbanity + trait_noise(rng),
            0.0f,
            1.0f);
        const float populist_affinity = ClampFloat(
            region.populist_affinity + 0.18f * anti_est + 0.07f * static_cast<float>(union_member)
            + 0.05f * (1.0f - static_cast<float>(college)) + trait_noise(rng),
            0.0f,
            1.0f);
        const float dem_loyalty = ClampFloat(
            0.12f + 0.32f * (ideology < 0.0f ? 1.0f : 0.0f) + 0.18f * static_cast<float>(union_member)
            + 0.12f * urbanity + 0.08f * static_cast<float>(college) - 0.10f * anti_est + trait_noise(rng),
            0.0f,
            1.0f);
        const float rep_loyalty = ClampFloat(
            0.08f + 0.30f * (ideology > 0.0f ? 1.0f : 0.0f) + 0.16f * religiosity
            + 0.09f * (1.0f - urbanity) - 0.13f * anti_est - 0.04f * static_cast<float>(college) + trait_noise(rng),
            0.0f,
            1.0f);
        const float centrist_affinity = ClampFloat(
            0.10f + 0.15f * sophistication + 0.10f * static_cast<float>(college)
            + 0.15f * (1.0f - fabsf(ideology)) - 0.08f * anti_est + trait_noise(rng),
            0.0f,
            1.0f);
        const float minor_openness = ClampFloat(
            0.06f + 0.26f * anti_est + 0.08f * sophistication + 0.07f * (1.0f - 0.5f * (dem_loyalty + rep_loyalty))
            - 0.07f * religiosity + 0.05f * unit(rng),
            0.0f,
            1.0f);

        voter.setVariable<float>("ideology", ideology);
        voter.setVariable<float>("turnout", turnout);
        voter.setVariable<float>("sophistication", sophistication);
        voter.setVariable<float>("anti_est", anti_est);
        voter.setVariable<float>("urbanity", urbanity);
        voter.setVariable<unsigned int>("college", college);
        voter.setVariable<unsigned int>("union_member", union_member);
        voter.setVariable<float>("religiosity", religiosity);
        voter.setVariable<float>("green_affinity", green_affinity);
        voter.setVariable<float>("libertarian_affinity", libertarian_affinity);
        voter.setVariable<float>("populist_affinity", populist_affinity);
        voter.setVariable<float>("dem_loyalty", dem_loyalty);
        voter.setVariable<float>("rep_loyalty", rep_loyalty);
        voter.setVariable<float>("centrist_affinity", centrist_affinity);
        voter.setVariable<float>("minor_openness", minor_openness);
        voter.setVariable<unsigned int>("region_id", region_id);
        voter.setVariable<unsigned int>("vote_choice", kAbstain);
    }

    FLAMEGPU->environment.setProperty<float, kRegionPartySlots>("REGION_PARTY_CONTACT", ZeroRegionPartyContact());
}

FLAMEGPU_INIT_FUNCTION(SeedActivists) {
    flamegpu::HostAgentAPI activists = FLAMEGPU->agent("activist");
    const unsigned int activist_count = FLAMEGPU->environment.getProperty<unsigned int>("ACTIVIST_COUNT");
    const unsigned int random_seed = FLAMEGPU->environment.getProperty<unsigned int>("RANDOM_SEED");

    std::mt19937 rng(random_seed + 17u);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::normal_distribution<float> activist_noise(0.0f, 0.06f);

    for (unsigned int i = 0u; i < activist_count; ++i) {
        flamegpu::HostNewAgentAPI activist = activists.newAgent();
        const unsigned int region_id = ChooseRegion(unit(rng), true);
        const RegionProfile& region = kRegionProfiles[region_id];

        float ideology = 0.0f;
        const float cluster_draw = unit(rng);
        if (cluster_draw < 0.18f) {
            ideology = -0.70f + 0.12f * activist_noise(rng) / 0.06f;
        } else if (cluster_draw < 0.40f) {
            ideology = -0.25f + 0.15f * activist_noise(rng) / 0.06f;
        } else if (cluster_draw < 0.68f) {
            ideology = 0.14f * activist_noise(rng) / 0.06f;
        } else if (cluster_draw < 0.85f) {
            ideology = 0.30f + 0.14f * activist_noise(rng) / 0.06f;
        } else {
            ideology = 0.68f + 0.12f * activist_noise(rng) / 0.06f;
        }
        ideology = ClampFloat(ideology + 0.55f * region.ideology_shift, -1.0f, 1.0f);

        const float organizer_skill = ClampFloat(0.20f + 0.60f * unit(rng) + 0.12f * region.activist_density, 0.0f, 1.0f);
        const float donor_access = ClampFloat(0.10f + 0.55f * unit(rng) + 0.10f * region.college_share, 0.0f, 1.0f);
        const float anti_est = ClampFloat(
            0.18f + 0.55f * fabsf(ideology) + 0.08f * region.populist_affinity + 0.10f * unit(rng),
            0.0f,
            1.0f);

        unsigned int preferred_family = ChoosePartyFamily(ideology, anti_est);
        const float family_tilt = unit(rng);
        if (fabsf(ideology) < 0.15f && anti_est < 0.52f && donor_access > 0.34f && family_tilt < 0.55f) {
            preferred_family = CENTRIST;
        } else if (anti_est > 0.66f && region.populist_affinity > 0.20f && family_tilt < 0.42f) {
            preferred_family = POPULIST;
        } else if (ideology < -0.40f && region.green_affinity > 0.32f && family_tilt < 0.22f) {
            preferred_family = GREEN;
        } else if (ideology > 0.52f && region.libertarian_affinity > 0.16f && donor_access > 0.40f && family_tilt < 0.16f) {
            preferred_family = LIBERTARIAN;
        }

        const float launch_tendency = ClampFloat(
            0.08f + 0.45f * anti_est + 0.22f * organizer_skill + 0.17f * donor_access + 0.10f * region.activist_density,
            0.0f,
            1.0f);
        const float field_reach = ClampFloat(
            0.20f + 0.34f * organizer_skill + 0.18f * donor_access + 0.16f * launch_tendency + 0.12f * region.activist_density,
            0.0f,
            1.0f);

        activist.setVariable<float>("ideology", ideology);
        activist.setVariable<float>("organizer_skill", organizer_skill);
        activist.setVariable<float>("donor_access", donor_access);
        activist.setVariable<float>("anti_est", anti_est);
        activist.setVariable<unsigned int>("preferred_family", preferred_family);
        activist.setVariable<float>("launch_tendency", launch_tendency);
        activist.setVariable<float>("field_reach", field_reach);
        activist.setVariable<unsigned int>("home_region", region_id);
    }
}

FLAMEGPU_AGENT_FUNCTION(PartyBroadcast, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<unsigned int>("party_id", FLAMEGPU->getVariable<unsigned int>("party_id"));
    FLAMEGPU->message_out.setVariable<float>("ideology", FLAMEGPU->getVariable<float>("ideology"));
    FLAMEGPU->message_out.setVariable<float>("organization", FLAMEGPU->getVariable<float>("organization"));
    FLAMEGPU->message_out.setVariable<float>("brand", FLAMEGPU->getVariable<float>("brand"));
    FLAMEGPU->message_out.setVariable<float>("credibility", FLAMEGPU->getVariable<float>("credibility"));
    FLAMEGPU->message_out.setVariable<float>("projected_viability", FLAMEGPU->getVariable<float>("projected_viability"));
    FLAMEGPU->message_out.setVariable<float>("momentum", FLAMEGPU->getVariable<float>("momentum"));
    FLAMEGPU->message_out.setVariable<float>("fundraising", FLAMEGPU->getVariable<float>("fundraising"));
    FLAMEGPU->message_out.setVariable<float>("media_reach", FLAMEGPU->getVariable<float>("media_reach"));
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(VoterChoose, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const float ideology = FLAMEGPU->getVariable<float>("ideology");
    const float turnout = FLAMEGPU->getVariable<float>("turnout");
    const float sophistication = FLAMEGPU->getVariable<float>("sophistication");
    const float anti_est = FLAMEGPU->getVariable<float>("anti_est");
    const float urbanity = FLAMEGPU->getVariable<float>("urbanity");
    const float religiosity = FLAMEGPU->getVariable<float>("religiosity");
    const float green_affinity = FLAMEGPU->getVariable<float>("green_affinity");
    const float libertarian_affinity = FLAMEGPU->getVariable<float>("libertarian_affinity");
    const float populist_affinity = FLAMEGPU->getVariable<float>("populist_affinity");
    const float dem_loyalty = FLAMEGPU->getVariable<float>("dem_loyalty");
    const float rep_loyalty = FLAMEGPU->getVariable<float>("rep_loyalty");
    const float centrist_affinity = FLAMEGPU->getVariable<float>("centrist_affinity");
    const float minor_openness = FLAMEGPU->getVariable<float>("minor_openness");
    const float college = FLAMEGPU->getVariable<unsigned int>("college") ? 1.0f : 0.0f;
    const float union_member = FLAMEGPU->getVariable<unsigned int>("union_member") ? 1.0f : 0.0f;
    const unsigned int region_id = FLAMEGPU->getVariable<unsigned int>("region_id");
    const float threshold = FLAMEGPU->environment.getProperty<float>("THRESHOLD");
    const unsigned int campaign_steps = FLAMEGPU->environment.getProperty<unsigned int>("CAMPAIGN_STEPS");
    const unsigned int current_step = FLAMEGPU->getStepCounter();
    const float step_fraction = campaign_steps > 1u
        ? static_cast<float>(current_step) / static_cast<float>(campaign_steps - 1u)
        : 1.0f;

    float best_score = -1.0e9f;
    float best_contact = 0.0f;
    unsigned int sampled_party = kAbstain;
    float sampled_score = -1.0e9f;
    float sampled_contact = 0.0f;
    unsigned int candidate_count = 0u;
    unsigned int candidate_ids[PARTY_COUNT]{};
    float candidate_scores[PARTY_COUNT]{};
    float candidate_contacts[PARTY_COUNT]{};

    for (const auto& msg : FLAMEGPU->message_in) {
        const unsigned int party_id = msg.getVariable<unsigned int>("party_id");
        const float party_ideology = msg.getVariable<float>("ideology");
        const float organization = msg.getVariable<float>("organization");
        const float brand = msg.getVariable<float>("brand");
        const float credibility = msg.getVariable<float>("credibility");
        const float projected_viability = msg.getVariable<float>("projected_viability");
        const float momentum = msg.getVariable<float>("momentum");
        const float fundraising = msg.getVariable<float>("fundraising");
        const float media_reach = msg.getVariable<float>("media_reach");
        const float regional_contact = FLAMEGPU->environment.getProperty<float, kRegionPartySlots>(
            "REGION_PARTY_CONTACT",
            FlattenRegionPartyIndex(region_id, party_id));

        const float closeness = 1.0f - fabsf(ideology - party_ideology);
        float target_urbanity = 0.50f;
        switch (party_id) {
            case DEM:
            case GREEN:
                target_urbanity = 0.85f;
                break;
            case REP:
            case LIBERTARIAN:
                target_urbanity = 0.22f;
                break;
            case POPULIST:
                target_urbanity = 0.32f;
                break;
            default:
                break;
        }

        const float place_fit = 1.0f - fabsf(urbanity - target_urbanity);
        float demographic_fit = 0.0f;
        switch (party_id) {
            case DEM:
                demographic_fit = 0.20f * college + 0.18f * union_member + 0.10f * urbanity - 0.10f * religiosity;
                break;
            case REP:
                demographic_fit = 0.17f * religiosity + 0.12f * (1.0f - urbanity) + 0.07f * (ideology > 0.0f ? 1.0f : 0.0f) - 0.05f * union_member;
                break;
            case GREEN:
                demographic_fit = 0.28f * green_affinity + 0.12f * college + 0.06f * urbanity - 0.08f * religiosity;
                break;
            case LIBERTARIAN:
                demographic_fit = 0.28f * libertarian_affinity + 0.10f * sophistication + 0.08f * (1.0f - union_member) - 0.03f * urbanity;
                break;
            case CENTRIST:
                demographic_fit = 0.22f * centrist_affinity + 0.08f * college + 0.06f * sophistication + 0.04f * (1.0f - anti_est);
                break;
            case POPULIST:
                demographic_fit = 0.24f * populist_affinity + 0.10f * anti_est + 0.08f * union_member + 0.04f * (1.0f - college);
                break;
            default:
                break;
        }

        float habit_bonus = 0.0f;
        switch (party_id) {
            case DEM:
                habit_bonus = 0.50f * dem_loyalty;
                break;
            case REP:
                habit_bonus = 0.50f * rep_loyalty;
                break;
            case GREEN:
                habit_bonus = 0.20f * green_affinity * minor_openness;
                break;
            case LIBERTARIAN:
                habit_bonus = 0.20f * libertarian_affinity * minor_openness;
                break;
            case CENTRIST:
                habit_bonus = 0.32f * centrist_affinity;
                break;
            case POPULIST:
                habit_bonus = 0.22f * populist_affinity * minor_openness;
                break;
            default:
                break;
        }

        const float duopoly_fatigue = ClampFloat(
            0.18f + 0.55f * anti_est + 0.18f * minor_openness - 0.32f * 0.5f * (dem_loyalty + rep_loyalty),
            0.0f,
            1.0f);
        float expressive_bonus = 0.0f;
        switch (party_id) {
            case GREEN:
                expressive_bonus = 0.12f * green_affinity * minor_openness * (0.45f + 0.55f * duopoly_fatigue);
                break;
            case LIBERTARIAN:
                expressive_bonus = 0.12f * libertarian_affinity * minor_openness * (0.45f + 0.55f * duopoly_fatigue);
                break;
            case CENTRIST:
                expressive_bonus = 0.18f * centrist_affinity * (0.35f + 0.65f * duopoly_fatigue);
                break;
            case POPULIST:
                expressive_bonus = 0.16f * populist_affinity * minor_openness * (0.40f + 0.60f * duopoly_fatigue);
                break;
            default:
                break;
        }

        float viability_gap = threshold - projected_viability;
        if (viability_gap < 0.0f) {
            viability_gap = 0.0f;
        }
        const bool major_party = IsMajorParty(party_id);
        const float strategic_penalty = viability_gap
            * (0.16f + 0.54f * sophistication)
            * (0.35f + 0.65f * step_fraction);
        const float strategic_bonus = projected_viability >= threshold
            ? (major_party ? (0.06f + 0.10f * sophistication) : (0.02f + 0.05f * sophistication))
            : (-0.02f * sophistication * step_fraction);
        const float outsider_signal = major_party
            ? 0.0f
            : 0.04f * anti_est * minor_openness * (0.72f - 0.30f * organization + 0.18f * step_fraction);
        const float resource_signal = major_party
            ? (0.22f * fundraising + 0.20f * media_reach + 0.10f * organization)
            : (0.10f * fundraising + 0.09f * media_reach + 0.06f * organization);
        const float coordination_readiness = 0.45f * projected_viability + 0.20f * fundraising
            + 0.18f * media_reach + 0.17f * regional_contact;
        const float minor_risk_penalty = major_party
            ? 0.0f
            : ClampFloat(
                (0.14f + 0.30f * (1.0f - minor_openness) + 0.14f * sophistication + 0.08f * step_fraction)
                * std::max(0.0f, 0.24f + 0.85f * viability_gap - coordination_readiness),
                0.0f,
                0.95f);

        const float score =
            1.68f * closeness
            + 0.26f * organization
            + 0.30f * brand
            + 0.30f * credibility
            + 0.42f * regional_contact
            + 0.16f * momentum
            + 0.18f * place_fit
            + demographic_fit
            + habit_bonus
            + expressive_bonus
            + resource_signal
            + outsider_signal
            + strategic_bonus
            - strategic_penalty
            - minor_risk_penalty;

        if (score > best_score) {
            best_score = score;
            best_contact = regional_contact;
        }
        if (candidate_count < PARTY_COUNT) {
            candidate_ids[candidate_count] = party_id;
            candidate_scores[candidate_count] = score;
            candidate_contacts[candidate_count] = regional_contact;
            ++candidate_count;
        }
    }

    if (candidate_count > 0u) {
        const float decisiveness = ClampFloat(
            1.55f + 1.05f * sophistication + 0.55f * 0.5f * (dem_loyalty + rep_loyalty) - 0.75f * minor_openness,
            0.95f,
            2.60f);
        float total_weight = 0.0f;
        float candidate_weights[PARTY_COUNT]{};
        for (unsigned int i = 0u; i < candidate_count; ++i) {
            const float weight = expf(decisiveness * (candidate_scores[i] - best_score));
            candidate_weights[i] = weight;
            total_weight += weight;
        }

        float draw = FLAMEGPU->random.uniform<float>() * total_weight;
        float cumulative_weight = 0.0f;
        for (unsigned int i = 0u; i < candidate_count; ++i) {
            cumulative_weight += candidate_weights[i];
            if (draw <= cumulative_weight || i + 1u == candidate_count) {
                sampled_party = candidate_ids[i];
                sampled_score = candidate_scores[i];
                sampled_contact = candidate_contacts[i];
                break;
            }
        }
    }

    float turnout_prob =
        0.05f
        + 0.52f * turnout
        + 0.10f * sophistication
        + 0.06f * college
        + 0.06f * (sampled_party != kAbstain ? sampled_contact : best_contact)
        + 0.05f * step_fraction
        + 0.03f * ((sampled_party != kAbstain ? sampled_score : best_score) > 0.0f
            ? (sampled_party != kAbstain ? sampled_score : best_score)
            : 0.0f);
    turnout_prob = ClampFloat(turnout_prob, 0.0f, 1.0f);

    if (sampled_party != kAbstain && FLAMEGPU->random.uniform<float>() < turnout_prob) {
        FLAMEGPU->setVariable<unsigned int>("vote_choice", sampled_party);
    } else {
        FLAMEGPU->setVariable<unsigned int>("vote_choice", kAbstain);
    }
    return flamegpu::ALIVE;
}

FLAMEGPU_INIT_FUNCTION(FormPartiesFromActivists) {
    flamegpu::HostAgentAPI party_agents = FLAMEGPU->agent("party");
    flamegpu::DeviceAgentVector activists = FLAMEGPU->agent("activist").getPopulationData();
    std::array<unsigned int, REGION_COUNT> region_totals{};
    const auto regional_families = SummarizeRegionalFamilies(activists, region_totals);
    const unsigned int total_activists = static_cast<unsigned int>(activists.size());
    const float total_activists_f = static_cast<float>(std::max(1u, total_activists));
    const float minor_entry_share = ClampFloat(FLAMEGPU->environment.getProperty<float>("MINOR_ENTRY_SHARE"), 0.0f, 1.0f);

    const auto statewide_families = CollapseStatewideFamilies(regional_families);

    std::unordered_map<unsigned int, PartyState> party_states;
    party_states.reserve(PARTY_COUNT);

    const auto add_major_party = [&](const unsigned int party_id, const float anchor_ideology) {
        const FamilyAggregate& aggregate = statewide_families[party_id];
        const float share = static_cast<float>(aggregate.count) / total_activists_f;
        const float avg_ideology = aggregate.count > 0u ? AverageOrZero(aggregate.ideology_sum, aggregate.count) : anchor_ideology;
        const float avg_skill = AverageOrZero(aggregate.organizer_skill_sum, aggregate.count);
        const float avg_donor = AverageOrZero(aggregate.donor_access_sum, aggregate.count);
        const float avg_launch = AverageOrZero(aggregate.launch_tendency_sum, aggregate.count);
        const float avg_reach = AverageOrZero(aggregate.field_reach_sum, aggregate.count);
        const float regional_depth = ComputeRegionalBreadth(regional_families, region_totals, party_id);
        const float fundraising = ClampFloat(0.58f + 0.30f * share + 0.18f * avg_donor + 0.08f * regional_depth, 0.45f, 1.0f);
        const float media_reach = ClampFloat(0.48f + 0.26f * fundraising + 0.12f * avg_skill + 0.10f * avg_launch, 0.38f, 1.0f);

        party_states.emplace(party_id, PartyState{
            party_id,
            ClampFloat(0.70f * anchor_ideology + 0.30f * avg_ideology, -1.0f, 1.0f),
            ClampFloat(0.72f + 0.95f * share + 0.10f * avg_skill, 0.60f, 1.0f),
            ClampFloat(0.70f + 0.18f * avg_donor + 0.08f * share, 0.50f, 1.0f),
            ClampFloat(0.72f + 0.10f * avg_launch + 0.08f * avg_skill, 0.55f, 1.0f),
            ClampFloat(0.42f + 0.70f * share + 0.08f * regional_depth + 0.05f * avg_skill, 0.32f, 0.85f),
            ClampFloat(0.58f + 0.80f * share + 0.10f * avg_reach + 0.10f * regional_depth, 0.42f, 1.0f),
            ClampFloat(0.42f + 0.35f * share + 0.10f * avg_launch, 0.24f, 0.88f),
            fundraising,
            media_reach,
        });
    };

    add_major_party(DEM, -0.35f);
    add_major_party(REP, 0.35f);

    for (unsigned int family : {GREEN, LIBERTARIAN, CENTRIST, POPULIST}) {
        const FamilyAggregate& aggregate = statewide_families[family];
        const float share = static_cast<float>(aggregate.count) / total_activists_f;
        const float regional_depth = ComputeRegionalBreadth(regional_families, region_totals, family);
        const float entry_score = 0.55f * share + 0.18f * AverageOrZero(aggregate.organizer_skill_sum, aggregate.count)
            + 0.17f * AverageOrZero(aggregate.donor_access_sum, aggregate.count) + 0.10f * regional_depth;
        if (aggregate.count < 10u || (share < minor_entry_share && entry_score < 0.18f)) {
            continue;
        }

        const float avg_ideology = AverageOrZero(aggregate.ideology_sum, aggregate.count);
        const float avg_skill = AverageOrZero(aggregate.organizer_skill_sum, aggregate.count);
        const float avg_donor = AverageOrZero(aggregate.donor_access_sum, aggregate.count);
        const float avg_launch = AverageOrZero(aggregate.launch_tendency_sum, aggregate.count);
        const float avg_reach = AverageOrZero(aggregate.field_reach_sum, aggregate.count);

        float ideology = avg_ideology;
        switch (family) {
            case GREEN:
                ideology = ClampFloat(aggregate.count > 0u ? avg_ideology : -0.72f, -1.0f, -0.15f);
                break;
            case LIBERTARIAN:
                ideology = ClampFloat(aggregate.count > 0u ? avg_ideology : 0.72f, 0.15f, 1.0f);
                break;
            case CENTRIST:
                ideology = ClampFloat(0.35f * avg_ideology, -0.20f, 0.20f);
                break;
            case POPULIST:
                ideology = ClampFloat(avg_ideology == 0.0f ? 0.05f : avg_ideology, -0.35f, 0.35f);
                break;
            default:
                break;
        }

        const float base_organization = ClampFloat(0.06f + 2.40f * share + 0.16f * avg_skill + 0.12f * avg_donor + 0.10f * regional_depth, 0.0f, 0.90f);
        const float fundraising = ClampFloat(0.03f + 1.25f * share + 0.24f * avg_donor + 0.10f * regional_depth, 0.0f, 0.72f);
        const float media_reach = ClampFloat(0.02f + 0.42f * fundraising + 0.10f * avg_launch + 0.08f * regional_depth, 0.0f, 0.72f);
        party_states.emplace(family, PartyState{
            family,
            ideology,
            base_organization,
            ClampFloat(0.10f + 0.34f * base_organization + 0.10f * avg_donor + 0.08f * avg_launch, 0.0f, 0.80f),
            ClampFloat(0.08f + 0.28f * base_organization + 0.12f * avg_launch + 0.10f * fundraising, 0.0f, 0.85f),
            ClampFloat(0.02f + 0.80f * share + 0.08f * avg_skill + 0.08f * regional_depth + 0.10f * fundraising, 0.0f, 0.40f),
            ClampFloat(0.04f + 2.00f * share + 0.14f * avg_reach + 0.12f * regional_depth, 0.0f, 0.70f),
            ClampFloat(0.08f + 0.45f * share + 0.10f * avg_launch, 0.0f, 0.60f),
            fundraising,
            media_reach,
        });
    }

    auto contacts = BuildRegionPartyContacts(regional_families, region_totals, party_states);
    for (auto& [party_id, state] : party_states) {
        state.field_strength = ClampFloat(0.62f * state.field_strength + 0.38f * WeightedPartyContact(contacts, party_id), 0.0f, 1.0f);
        state.momentum = ClampFloat(0.72f * state.momentum + 0.28f * state.projected_viability, 0.0f, 1.0f);
    }
    contacts = BuildRegionPartyContacts(regional_families, region_totals, party_states);
    FLAMEGPU->environment.setProperty<float, kRegionPartySlots>("REGION_PARTY_CONTACT", contacts);

    for (unsigned int party_id : {DEM, REP, GREEN, LIBERTARIAN, CENTRIST, POPULIST}) {
        const auto state_it = party_states.find(party_id);
        if (state_it == party_states.end()) {
            continue;
        }
        const PartyState& state = state_it->second;
        flamegpu::HostNewAgentAPI party = party_agents.newAgent();
        party.setVariable<unsigned int>("party_id", state.party_id);
        party.setVariable<float>("ideology", state.ideology);
        party.setVariable<float>("organization", state.organization);
        party.setVariable<float>("brand", state.brand);
        party.setVariable<float>("credibility", state.credibility);
        party.setVariable<float>("projected_viability", state.projected_viability);
        party.setVariable<float>("field_strength", state.field_strength);
        party.setVariable<float>("momentum", state.momentum);
        party.setVariable<float>("fundraising", state.fundraising);
        party.setVariable<float>("media_reach", state.media_reach);
    }
}

FLAMEGPU_STEP_FUNCTION(AdvanceCampaignAndAllocateSeats) {
    flamegpu::HostAgentAPI voters = FLAMEGPU->agent("voter");
    flamegpu::HostAgentAPI activist_agent = FLAMEGPU->agent("activist");
    flamegpu::HostAgentAPI party_agent = FLAMEGPU->agent("party");

    const unsigned int step = FLAMEGPU->getStepCounter();
    const unsigned int campaign_steps = FLAMEGPU->getSimulationConfig().steps;
    const bool final_step = campaign_steps > 0u && step + 1u == campaign_steps;
    const unsigned int total_voters = voters.count();
    const unsigned int abstentions = voters.count<unsigned int>("vote_choice", kAbstain);
    const unsigned int total_valid = total_voters > abstentions ? total_voters - abstentions : 0u;
    const unsigned int seats = FLAMEGPU->environment.getProperty<unsigned int>("SEATS");
    const float threshold = FLAMEGPU->environment.getProperty<float>("THRESHOLD");
    const unsigned int divisor_method = FLAMEGPU->environment.getProperty<unsigned int>("DIVISOR_METHOD");

    flamegpu::DeviceAgentVector parties = party_agent.getPopulationData();
    std::vector<std::pair<unsigned int, std::uint64_t>> qualified_votes;
    auto results = BuildPartyResults(voters, parties, total_valid, threshold, final_step ? &qualified_votes : nullptr);

    if (!final_step) {
        PrintCampaignCheckpoint(results, step, campaign_steps, total_valid);

        flamegpu::DeviceAgentVector activists = activist_agent.getPopulationData();
        std::array<unsigned int, REGION_COUNT> region_totals{};
        const auto regional_families = SummarizeRegionalFamilies(activists, region_totals);
        const auto statewide_families = CollapseStatewideFamilies(regional_families);
        const float total_activists_f = static_cast<float>(std::max(1u, static_cast<unsigned int>(activists.size())));
        auto party_states = SnapshotParties(parties);
        const auto existing_contacts = FLAMEGPU->environment.getProperty<float, kRegionPartySlots>("REGION_PARTY_CONTACT");

        for (const PartyResult& result : results) {
            auto state_it = party_states.find(result.party_id);
            if (state_it == party_states.end()) {
                continue;
            }
            PartyState& state = state_it->second;
            const float contact = WeightedPartyContact(existing_contacts, state.party_id);
            const bool major_party = IsMajorParty(state.party_id);
            const FamilyAggregate& aggregate = statewide_families[state.party_id];
            const float activist_share = static_cast<float>(aggregate.count) / total_activists_f;
            const float avg_skill = AverageOrZero(aggregate.organizer_skill_sum, aggregate.count);
            const float avg_donor = AverageOrZero(aggregate.donor_access_sum, aggregate.count);
            const float avg_launch = AverageOrZero(aggregate.launch_tendency_sum, aggregate.count);
            const float avg_reach = AverageOrZero(aggregate.field_reach_sum, aggregate.count);
            const float regional_depth = ComputeRegionalBreadth(regional_families, region_totals, state.party_id);
            const float funding_target = major_party
                ? ClampFloat(0.60f + 0.18f * result.share + 0.18f * avg_donor + 0.10f * regional_depth + 0.08f * contact, 0.48f, 1.0f)
                : ClampFloat(0.02f + 0.82f * activist_share + 0.14f * avg_donor + 0.10f * regional_depth + 0.04f * result.share, 0.0f, 0.55f);
            const float media_target = major_party
                ? ClampFloat(0.48f + 0.28f * funding_target + 0.10f * avg_skill + 0.10f * result.share + 0.08f * state.brand, 0.38f, 1.0f)
                : ClampFloat(0.01f + 0.34f * funding_target + 0.08f * avg_launch + 0.06f * regional_depth + 0.04f * result.share, 0.0f, 0.52f);
            const float viability_target = major_party
                ? ClampFloat(0.38f + 0.60f * result.share + 0.10f * contact + 0.06f * media_target, 0.0f, 1.0f)
                : ClampFloat(0.02f + 0.46f * result.share + 0.06f * contact + 0.08f * funding_target + 0.06f * media_target + 0.04f * regional_depth, 0.0f, 0.42f);
            const float momentum_target = major_party
                ? ClampFloat(0.38f + 1.70f * (result.share - threshold) + 0.18f * contact + 0.12f * media_target, 0.0f, 1.0f)
                : ClampFloat(0.06f + 1.10f * (result.share - 0.55f * threshold) + 0.12f * contact + 0.08f * media_target, 0.0f, 0.55f);

            state.fundraising = ClampFloat(0.60f * state.fundraising + 0.40f * funding_target, 0.0f, 1.0f);
            state.media_reach = ClampFloat(0.58f * state.media_reach + 0.42f * media_target, 0.0f, 1.0f);
            state.projected_viability = ClampFloat((major_party ? 0.55f : 0.70f) * state.projected_viability
                + (major_party ? 0.45f : 0.30f) * viability_target, 0.0f, 1.0f);
            state.momentum = ClampFloat(0.62f * state.momentum + 0.38f * momentum_target, 0.0f, 1.0f);
            state.organization = ClampFloat(
                0.68f * state.organization + 0.14f * state.fundraising + 0.08f * avg_skill + 0.06f * contact,
                major_party ? 0.55f : 0.0f,
                1.0f);
            state.field_strength = ClampFloat(
                0.58f * state.field_strength + 0.20f * contact + 0.14f * state.organization + 0.08f * avg_reach + 0.06f * regional_depth,
                0.0f,
                1.0f);
            state.brand = ClampFloat(
                0.70f * state.brand + 0.14f * state.media_reach + 0.08f * state.organization + 0.06f * result.share,
                major_party ? 0.45f : 0.0f,
                1.0f);
            state.credibility = ClampFloat(
                0.70f * state.credibility + (result.share >= threshold ? 0.12f : -0.03f) + 0.08f * state.organization + 0.05f * state.fundraising,
                major_party ? 0.50f : 0.0f,
                1.0f);
        }

        const auto contacts = BuildRegionPartyContacts(regional_families, region_totals, party_states);
        FLAMEGPU->environment.setProperty<float, kRegionPartySlots>("REGION_PARTY_CONTACT", contacts);

        for (auto party : parties) {
            const unsigned int party_id = party.getVariable<unsigned int>("party_id");
            const auto state_it = party_states.find(party_id);
            if (state_it == party_states.end()) {
                continue;
            }
            const PartyState& state = state_it->second;
            party.setVariable<float>("ideology", state.ideology);
            party.setVariable<float>("organization", state.organization);
            party.setVariable<float>("brand", state.brand);
            party.setVariable<float>("credibility", state.credibility);
            party.setVariable<float>("projected_viability", state.projected_viability);
            party.setVariable<float>("field_strength", state.field_strength);
            party.setVariable<float>("momentum", state.momentum);
            party.setVariable<float>("fundraising", state.fundraising);
            party.setVariable<float>("media_reach", state.media_reach);
        }
        parties.syncChanges();
        return;
    }

    if (total_valid == 0u) {
        std::printf("No valid votes were cast.\n");
        return;
    }

    if (qualified_votes.empty() && !results.empty()) {
        qualified_votes.emplace_back(results.front().party_id, results.front().votes);
    }
    const auto seat_counts = AllocateDivisorSeats(qualified_votes, seats, divisor_method);
    for (PartyResult& result : results) {
        const auto seat_it = seat_counts.find(result.party_id);
        if (seat_it != seat_counts.end()) {
            result.seats = seat_it->second;
        }
    }

    const float turnout_share = total_voters > 0u ? static_cast<float>(total_valid) / static_cast<float>(total_voters) : 0.0f;

    std::printf("\n=== Wisconsin PR simulation ===\n");
    std::printf("Campaign steps: %u\n", campaign_steps);
    std::printf("Total voters: %u\n", total_voters);
    std::printf("Valid votes: %u\n", total_valid);
    std::printf("Abstentions: %u\n", abstentions);
    std::printf("Turnout: %.2f%%\n", turnout_share * 100.0f);
    std::printf("Threshold: %.1f%%\n", threshold * 100.0f);
    std::printf("Divisor method: %s\n", DivisorMethodLabel(divisor_method));
    std::printf("Seats: %u\n\n", seats);

    for (const PartyResult& result : results) {
        const char* status = result.share >= threshold ? "qualified" : "below threshold";
        std::printf("%-18s votes=%10llu share=%6.2f%% seats=%2u %s\n",
            PartyLabel(result.party_id),
            static_cast<unsigned long long>(result.votes),
            result.share * 100.0f,
            result.seats,
            status);
    }
    const SimulationArtifactConfig config{
        total_voters,
        total_valid,
        abstentions,
        turnout_share,
        threshold,
        seats,
        divisor_method,
        campaign_steps
    };
    WriteResultsArtifacts(results, config);
}

void BuildModel(
    flamegpu::ModelDescription& model,
    const unsigned int voter_count,
    const unsigned int activist_count,
    const unsigned int seats,
    const float threshold,
    const unsigned int random_seed,
    const float minor_entry_share,
    const unsigned int divisor_method,
    const unsigned int campaign_steps) {
    flamegpu::EnvironmentDescription env = model.Environment();
    env.newProperty<unsigned int>("VOTER_COUNT", voter_count);
    env.newProperty<unsigned int>("ACTIVIST_COUNT", activist_count);
    env.newProperty<unsigned int>("SEATS", seats);
    env.newProperty<float>("THRESHOLD", threshold);
    env.newProperty<unsigned int>("RANDOM_SEED", random_seed);
    env.newProperty<float>("MINOR_ENTRY_SHARE", minor_entry_share);
    env.newProperty<unsigned int>("DIVISOR_METHOD", divisor_method);
    env.newProperty<unsigned int>("CAMPAIGN_STEPS", campaign_steps);
    env.newProperty<float, kRegionPartySlots>("REGION_PARTY_CONTACT", ZeroRegionPartyContact());

    auto party_msg = model.newMessage<flamegpu::MessageBruteForce>("party_msg");
    party_msg.newVariable<unsigned int>("party_id");
    party_msg.newVariable<float>("ideology");
    party_msg.newVariable<float>("organization");
    party_msg.newVariable<float>("brand");
    party_msg.newVariable<float>("credibility");
    party_msg.newVariable<float>("projected_viability");
    party_msg.newVariable<float>("momentum");
    party_msg.newVariable<float>("fundraising");
    party_msg.newVariable<float>("media_reach");

    flamegpu::AgentDescription voter = model.newAgent("voter");
    voter.newVariable<float>("ideology");
    voter.newVariable<float>("turnout");
    voter.newVariable<float>("sophistication");
    voter.newVariable<float>("anti_est");
    voter.newVariable<float>("urbanity");
    voter.newVariable<unsigned int>("college");
    voter.newVariable<unsigned int>("union_member");
    voter.newVariable<float>("religiosity");
    voter.newVariable<float>("green_affinity");
    voter.newVariable<float>("libertarian_affinity");
    voter.newVariable<float>("populist_affinity");
    voter.newVariable<float>("dem_loyalty");
    voter.newVariable<float>("rep_loyalty");
    voter.newVariable<float>("centrist_affinity");
    voter.newVariable<float>("minor_openness");
    voter.newVariable<unsigned int>("region_id");
    voter.newVariable<unsigned int>("vote_choice", kAbstain);

    flamegpu::AgentDescription activist = model.newAgent("activist");
    activist.newVariable<float>("ideology");
    activist.newVariable<float>("organizer_skill");
    activist.newVariable<float>("donor_access");
    activist.newVariable<float>("anti_est");
    activist.newVariable<unsigned int>("preferred_family");
    activist.newVariable<float>("launch_tendency");
    activist.newVariable<float>("field_reach");
    activist.newVariable<unsigned int>("home_region");

    flamegpu::AgentDescription party = model.newAgent("party");
    party.newVariable<unsigned int>("party_id");
    party.newVariable<float>("ideology");
    party.newVariable<float>("organization");
    party.newVariable<float>("brand");
    party.newVariable<float>("credibility");
    party.newVariable<float>("projected_viability");
    party.newVariable<float>("field_strength");
    party.newVariable<float>("momentum");
    party.newVariable<float>("fundraising");
    party.newVariable<float>("media_reach");

    auto party_fn = party.newFunction("party_broadcast", PartyBroadcast);
    party_fn.setMessageOutput("party_msg");
    auto voter_fn = voter.newFunction("voter_choose", VoterChoose);
    voter_fn.setMessageInput("party_msg");

    model.newLayer("party_broadcast").addAgentFunction(party_fn);
    model.newLayer("voter_choice").addAgentFunction(voter_fn);

    model.addInitFunction(SeedVoters);
    model.addInitFunction(SeedActivists);
    model.addInitFunction(FormPartiesFromActivists);
    model.addStepFunction(AdvanceCampaignAndAllocateSeats);
}

int RunMain(int argc, const char** argv) {
    const unsigned int voter_count = ParseUnsignedEnv("WISCONSIN_PR_VOTERS", kDefaultVoterCount);
    const unsigned int activist_count = ParseUnsignedEnv("WISCONSIN_PR_ACTIVISTS", kDefaultActivistCount);
    const unsigned int seats = std::max(1u, ParseUnsignedEnv("WISCONSIN_PR_SEATS", kDefaultSeats));
    const unsigned int random_seed = ParseUnsignedEnv("WISCONSIN_PR_RANDOM_SEED", kDefaultRandomSeed);
    const float threshold = ClampFloat(ParseFloatEnv("WISCONSIN_PR_THRESHOLD", kDefaultThreshold), 0.0f, 1.0f);
    const float minor_entry_share = ClampFloat(ParseFloatEnv("WISCONSIN_PR_MINOR_ENTRY_SHARE", kDefaultMinorEntryShare), 0.0f, 1.0f);
    const unsigned int divisor_method = ParseDivisorMethodEnv("WISCONSIN_PR_DIVISOR_METHOD", SAINTE_LAGUE);
    const unsigned int campaign_steps = std::max(1u, ParseUnsignedEnv("WISCONSIN_PR_STEPS", kDefaultCampaignSteps));

    flamegpu::ModelDescription model(kModelName);
    BuildModel(
        model,
        voter_count,
        activist_count,
        seats,
        threshold,
        random_seed,
        minor_entry_share,
        divisor_method,
        campaign_steps);

    std::printf(
        "Launching %s with %u voters, %u activists, %u campaign steps, %u seats, %.1f%% threshold, %s, seed=%u\n",
        kModelName,
        voter_count,
        activist_count,
        campaign_steps,
        seats,
        threshold * 100.0f,
        DivisorMethodLabel(divisor_method),
        random_seed);

    flamegpu::CUDASimulation simulation(model);
    simulation.SimulationConfig().random_seed = random_seed;
    simulation.SimulationConfig().steps = campaign_steps;
    simulation.initialise(argc, argv);
    simulation.simulate();

    flamegpu::util::cleanup();
    return EXIT_SUCCESS;
}

}  // namespace wisconsin_pr

int main(int argc, const char** argv) {
    return wisconsin_pr::RunMain(argc, argv);
}
