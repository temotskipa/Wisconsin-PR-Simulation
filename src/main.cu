#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
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

// ===== CONSTANTS =====

constexpr char kModelName[] = "Wisconsin PR Simulation";
constexpr unsigned int MAX_PARTIES = 16u;
constexpr unsigned int kDefaultSeats = 99u;
constexpr unsigned int kDefaultRandomSeed = 42u;
constexpr unsigned int kDefaultVoterCount = 5970000u;
constexpr unsigned int kDefaultActivistCount = 2500u;
constexpr float kDefaultMinorEntryShare = 0.020f;
constexpr unsigned int kDefaultCampaignSteps = 6u;
constexpr unsigned int kDefaultOrganizingSteps = 3u;
constexpr unsigned int kAbstain = 999u;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kSpatialMin = 0.0f;
constexpr float kSpatialMax = 100.0f;
constexpr float kSpatialRadius = 0.4f;
constexpr unsigned int CLUSTER_GRID = 10u;
constexpr unsigned int MIN_CLUSTER_ACTIVISTS = 30u;

// ===== ENUMS (no PartyId — parties are dynamic) =====

enum DivisorMethod : unsigned int { SAINTE_LAGUE = 0u, DHONDT = 1u };

enum RegionId : unsigned int {
    MILWAUKEE_METRO = 0u, DANE_METRO = 1u, WOW_SUBURBS = 2u,
    SOUTHEAST_INDUSTRIAL = 3u, FOX_VALLEY = 4u, DRIFTLESS_WEST = 5u,
    NORTHWOODS = 6u, RURAL_HEARTLAND = 7u, REGION_COUNT = 8u,
};

constexpr unsigned int kRegionPartySlots = REGION_COUNT * MAX_PARTIES;

// ===== STRUCTS =====

struct RegionProfile {
    const char* label;
    float population_share;
    float econ_shift;       // regional economic ideology deviation
    float social_shift;     // regional social ideology deviation
    float urbanity;
    float turnout_bonus;
    float college_share;
    float union_share;
    float religiosity;
    float activist_density;
    float center_x;         // spatial centroid
    float center_y;
};

struct IdeologyCluster {
    unsigned int count = 0u;
    double econ_sum = 0.0;
    double social_sum = 0.0;
    double anti_est_sum = 0.0;
    double organizer_skill_sum = 0.0;
    double donor_access_sum = 0.0;
    double launch_tendency_sum = 0.0;
    double field_reach_sum = 0.0;
    std::array<unsigned int, REGION_COUNT> region_counts{};
};

struct PartyState {
    unsigned int party_id = 0u;
    float econ_ideology = 0.0f;
    float social_ideology = 0.0f;
    float urban_orientation = 0.5f;
    float anti_est_posture = 0.0f;
    float religiosity_align = 0.0f;
    float union_alignment = 0.0f;
    float college_alignment = 0.0f;
    float establishment_age = 0.0f;
    float organization = 0.0f;
    float brand = 0.0f;
    float credibility = 0.0f;
    float projected_viability = 0.0f;
    float field_strength = 0.0f;
    float momentum = 0.0f;
    float fundraising = 0.0f;
    float media_reach = 0.0f;
    unsigned int is_alive = 1u;
    unsigned int is_legacy = 0u;
    unsigned int consecutive_low_viability = 0u;
    std::string display_name;
};

struct PartyResult {
    unsigned int party_id = 0u;
    std::uint64_t votes = 0u;
    float share = 0.0f;
    unsigned int seats = 0u;
    std::string name;
    std::string color;
    float econ_ideology = 0.0f;
};

// ===== REGION DATA (Census ACS 2024, BLS 2025, Pew Research) =====

constexpr std::array<RegionProfile, REGION_COUNT> kRegionProfiles = {{
    // label              pop%    econ   social urban  turn   coll   union  relig  activ  cx    cy
    {"Milwaukee Metro",   0.156f, -0.25f, -0.15f, 0.99f, -0.04f, 0.33f, 0.08f, 0.34f, 1.25f, 85.0f, 30.0f},
    {"Dane / Madison",    0.097f, -0.35f, -0.40f, 0.85f,  0.05f, 0.55f, 0.07f, 0.23f, 1.30f, 55.0f, 35.0f},
    {"WOW Suburbs",       0.109f,  0.30f,  0.25f, 0.90f,  0.06f, 0.48f, 0.03f, 0.60f, 0.85f, 78.0f, 35.0f},
    {"SE Industrial",     0.063f, -0.05f,  0.05f, 0.75f,  0.00f, 0.26f, 0.12f, 0.48f, 1.05f, 85.0f, 20.0f},
    {"Fox Valley",        0.155f,  0.10f,  0.15f, 0.60f,  0.01f, 0.30f, 0.10f, 0.56f, 0.95f, 70.0f, 55.0f},
    {"Driftless West",    0.160f, -0.08f, -0.05f, 0.40f,  0.00f, 0.28f, 0.06f, 0.42f, 0.95f, 25.0f, 40.0f},
    {"Northwoods",        0.101f,  0.15f,  0.20f, 0.25f, -0.01f, 0.21f, 0.04f, 0.50f, 0.80f, 50.0f, 75.0f},
    {"Rural Heartland",   0.159f,  0.08f,  0.12f, 0.30f,  0.00f, 0.22f, 0.05f, 0.52f, 0.75f, 45.0f, 50.0f},
}};

// ===== UTILITY FUNCTIONS =====

FLAMEGPU_HOST_DEVICE_FUNCTION float ClampFloat(const float v, const float lo, const float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

unsigned int ParseUnsignedEnv(const char* name, const unsigned int def) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return def;
    char* end = nullptr;
    const auto parsed = static_cast<std::uint64_t>(std::strtoull(raw, &end, 10));
    if (end == raw || *end != '\0' || parsed > std::numeric_limits<unsigned int>::max()) {
        std::printf("Ignoring invalid %s=%s, using %u\n", name, raw, def);
        return def;
    }
    return static_cast<unsigned int>(parsed);
}

float ParseFloatEnv(const char* name, const float def) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return def;
    char* end = nullptr;
    const float parsed = std::strtof(raw, &end);
    if (end == raw || *end != '\0') {
        std::printf("Ignoring invalid %s=%s, using %.3f\n", name, raw, def);
        return def;
    }
    return parsed;
}

unsigned int ParseDivisorMethodEnv(const char* name, const unsigned int def) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return def;
    std::string m(raw);
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (m == "sainte_lague" || m == "sainte-lague" || m == "saintelague") return SAINTE_LAGUE;
    if (m == "dhondt" || m == "d_hondt" || m == "d-hondt" || m == "d'hondt") return DHONDT;
    std::printf("Ignoring invalid %s=%s, using %s\n", name, raw, def == DHONDT ? "dhondt" : "sainte_lague");
    return def;
}

const char* DivisorMethodLabel(const unsigned int m) { return m == DHONDT ? "dhondt" : "sainte_lague"; }

float AverageOrZero(const double sum, const unsigned int count) {
    return count > 0u ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
}

unsigned int ChooseRegion(const float draw, const bool activist_weighted) {
    float total = 0.0f;
    for (const auto& r : kRegionProfiles)
        total += r.population_share * (activist_weighted ? r.activist_density : 1.0f);
    float cum = 0.0f;
    for (unsigned int i = 0u; i < REGION_COUNT; ++i) {
        cum += kRegionProfiles[i].population_share
            * (activist_weighted ? kRegionProfiles[i].activist_density : 1.0f) / total;
        if (draw <= cum) return i;
    }
    return REGION_COUNT - 1u;
}

FLAMEGPU_HOST_DEVICE_FUNCTION unsigned int FlatRegionParty(const unsigned int region, const unsigned int party) {
    return region * MAX_PARTIES + party;
}

std::array<float, kRegionPartySlots> ZeroContacts() {
    std::array<float, kRegionPartySlots> d{};
    d.fill(0.0f);
    return d;
}

// ===== PROCEDURAL NAMING & COLORS =====

std::string GeneratePartyName(float econ, float social, float anti_est, float urban, bool is_legacy, unsigned int legacy_id) {
    if (is_legacy) {
        return legacy_id == 0u ? "Democratic" : "Republican";
    }
    std::string name;
    if (anti_est > 0.65f) name += "People's ";
    else if (anti_est > 0.45f) name += "New ";

    if (econ < -0.3f && social < -0.2f) name += "Progressive";
    else if (econ < -0.2f && social > 0.2f) name += "Labor";
    else if (econ > 0.25f && social > 0.3f) name += "Conservative";
    else if (econ > 0.25f && social < -0.2f) name += "Liberty";
    else if (fabsf(econ) < 0.2f && fabsf(social) < 0.2f) name += "Moderate";
    else if (social > 0.4f) name += "Traditionalist";
    else if (social < -0.4f) name += "Reform";
    else name += "Independent";

    if (urban > 0.7f) name += " Urban Coalition";
    else if (urban < 0.3f) name += " Rural Alliance";
    else name += " Party";
    return name;
}

std::string IdeologyToColor(float econ, float social, float anti_est, float est_age, bool is_legacy, unsigned int legacy_id) {
    if (is_legacy) return legacy_id == 0u ? "#2563eb" : "#dc2626";
    // Map ideology to HSL hue: left=240(blue), center=120(green), right=0/360(red)
    float hue = 240.0f - (econ + 1.0f) * 120.0f; // [-1,1] -> [360, 0]
    hue += social * 30.0f; // social axis shifts hue
    if (hue < 0.0f) hue += 360.0f;
    if (hue >= 360.0f) hue -= 360.0f;
    float sat = 55.0f + 25.0f * anti_est;
    float lit = 42.0f + 10.0f * est_age;
    std::ostringstream ss;
    ss << "hsl(" << static_cast<int>(hue) << "," << static_cast<int>(sat) << "%," << static_cast<int>(lit) << "%)";
    return ss.str();
}

// ===== SEAT ALLOCATION =====

double DivisorValue(const unsigned int seats, const unsigned int method) {
    return method == DHONDT ? static_cast<double>(seats + 1u) : static_cast<double>(2u * seats + 1u);
}

std::unordered_map<unsigned int, unsigned int> AllocateDivisorSeats(
    const std::vector<std::pair<unsigned int, std::uint64_t>>& qualified,
    const unsigned int total_seats, const unsigned int method) {
    std::unordered_map<unsigned int, unsigned int> sc;
    sc.reserve(qualified.size());
    for (const auto& [pid, _] : qualified) sc.emplace(pid, 0u);
    for (unsigned int s = 0u; s < total_seats; ++s) {
        unsigned int best = qualified.front().first;
        double best_q = -1.0;
        for (const auto& [pid, votes] : qualified) {
            double q = static_cast<double>(votes) / DivisorValue(sc[pid], method);
            if (q > best_q) { best_q = q; best = pid; }
        }
        ++sc[best];
    }
    return sc;
}

// ===== GENERIC CONTACT BUILDING =====

std::array<float, kRegionPartySlots> BuildContacts(
    const std::unordered_map<unsigned int, PartyState>& parties,
    const std::array<unsigned int, REGION_COUNT>& region_activist_totals,
    const std::unordered_map<unsigned int, IdeologyCluster>& party_clusters) {
    auto contacts = ZeroContacts();
    for (unsigned int rid = 0u; rid < REGION_COUNT; ++rid) {
        const auto& rp = kRegionProfiles[rid];
        const float total_act = static_cast<float>(std::max(1u, region_activist_totals[rid]));
        for (const auto& [pid, state] : parties) {
            if (!state.is_alive) continue;
            // Generic contact formula using continuous party attributes
            float act_share = 0.0f;
            float avg_skill = 0.0f;
            auto cit = party_clusters.find(pid);
            if (cit != party_clusters.end()) {
                const auto& cl = cit->second;
                act_share = static_cast<float>(cl.region_counts[rid]) / total_act;
                avg_skill = AverageOrZero(cl.organizer_skill_sum, cl.count);
            }
            // Urban/rural fit
            float urban_fit = 1.0f - fabsf(rp.urbanity - state.urban_orientation);
            // Base contact from activist presence + regional fit
            float contact = ClampFloat(
                0.02f + 2.0f * act_share + 0.14f * urban_fit + 0.10f * avg_skill
                + 0.08f * state.organization + 0.06f * state.fundraising,
                0.0f, 1.0f);
            // Establishment bonus
            if (state.establishment_age > 0.5f) {
                contact = ClampFloat(contact + 0.20f * state.establishment_age + 0.12f * state.media_reach, 0.0f, 1.0f);
            }
            // Campaign modifiers
            contact = ClampFloat(
                contact * (0.60f + 0.20f * state.momentum + 0.12f * state.field_strength + 0.08f * state.projected_viability),
                0.0f, 1.0f);
            contacts[FlatRegionParty(rid, pid)] = contact;
        }
    }
    return contacts;
}

float WeightedContact(const std::array<float, kRegionPartySlots>& contacts, const unsigned int pid) {
    float w = 0.0f;
    for (unsigned int r = 0u; r < REGION_COUNT; ++r)
        w += kRegionProfiles[r].population_share * contacts[FlatRegionParty(r, pid)];
    return ClampFloat(w, 0.0f, 1.0f);
}

// ===== CLUSTERING (grid-based density scan) =====

struct GridClusterResult {
    float econ_centroid = 0.0f;
    float social_centroid = 0.0f;
    unsigned int total_activists = 0u;
    IdeologyCluster cluster;
};

std::vector<GridClusterResult> ClusterActivists(
    flamegpu::DeviceAgentVector activists,
    const unsigned int min_size) {
    // Grid cell counts in 2D ideology space [-1,1]x[-1,1]
    unsigned int cell_count[CLUSTER_GRID][CLUSTER_GRID] = {};
    double cell_econ[CLUSTER_GRID][CLUSTER_GRID] = {};
    double cell_social[CLUSTER_GRID][CLUSTER_GRID] = {};
    constexpr float gmin = -1.0f, gmax = 1.0f;
    constexpr float cell_sz = (gmax - gmin) / static_cast<float>(CLUSTER_GRID);

    for (const auto& a : activists) {
        float e = a.getVariable<float>("econ_ideology");
        float s = a.getVariable<float>("social_ideology");
        unsigned int gx = std::min(CLUSTER_GRID - 1u, static_cast<unsigned int>((e - gmin) / cell_sz));
        unsigned int gy = std::min(CLUSTER_GRID - 1u, static_cast<unsigned int>((s - gmin) / cell_sz));
        cell_count[gx][gy]++;
        cell_econ[gx][gy] += e;
        cell_social[gx][gy] += s;
    }

    // Density threshold: requires dense cores to prevent merging distinct factions
    unsigned int threshold = std::max(5u, min_size / 3u);

    // DFS flood-fill to find connected components of dense cells
    int labels[CLUSTER_GRID][CLUSTER_GRID];
    for (auto& row : labels) for (auto& v : row) v = -1;
    int next_label = 0;

    for (unsigned int x = 0u; x < CLUSTER_GRID; ++x) {
        for (unsigned int y = 0u; y < CLUSTER_GRID; ++y) {
            if (cell_count[x][y] < threshold || labels[x][y] >= 0) continue;
            // DFS
            std::vector<std::pair<int,int>> stack;
            stack.push_back({static_cast<int>(x), static_cast<int>(y)});
            labels[x][y] = next_label;
            while (!stack.empty()) {
                auto [cx, cy] = stack.back(); stack.pop_back();
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        int nx = cx + dx, ny = cy + dy;
                        if (nx >= 0 && nx < static_cast<int>(CLUSTER_GRID) &&
                            ny >= 0 && ny < static_cast<int>(CLUSTER_GRID) &&
                            labels[nx][ny] < 0 && cell_count[nx][ny] >= threshold) {
                            labels[nx][ny] = next_label;
                            stack.push_back({nx, ny});
                        }
                    }
                }
            }
            ++next_label;
        }
    }

    if (next_label == 0) return {};

    // Compute cluster centroids from grid
    std::vector<GridClusterResult> results(next_label);
    for (unsigned int x = 0u; x < CLUSTER_GRID; ++x) {
        for (unsigned int y = 0u; y < CLUSTER_GRID; ++y) {
            if (labels[x][y] < 0) continue;
            auto& r = results[labels[x][y]];
            r.total_activists += cell_count[x][y];
        }
    }

    // Now assign each activist to nearest cluster centroid and build full stats
    // First compute rough centroids from grid
    for (unsigned int x = 0u; x < CLUSTER_GRID; ++x) {
        for (unsigned int y = 0u; y < CLUSTER_GRID; ++y) {
            if (labels[x][y] < 0) continue;
            auto& r = results[labels[x][y]];
            r.econ_centroid += static_cast<float>(cell_econ[x][y]);
            r.social_centroid += static_cast<float>(cell_social[x][y]);
        }
    }
    for (auto& r : results) {
        if (r.total_activists > 0u) {
            r.econ_centroid /= static_cast<float>(r.total_activists);
            r.social_centroid /= static_cast<float>(r.total_activists);
        }
    }

    // Assign activists to nearest centroid and accumulate full cluster stats
    for (const auto& a : activists) {
        float e = a.getVariable<float>("econ_ideology");
        float s = a.getVariable<float>("social_ideology");
        float best_dist = 1e9f;
        int best_cl = -1;
        for (int i = 0; i < next_label; ++i) {
            float de = e - results[i].econ_centroid;
            float ds = s - results[i].social_centroid;
            float d = de * de + ds * ds;
            if (d < best_dist) { best_dist = d; best_cl = i; }
        }
        if (best_cl < 0) continue;
        auto& cl = results[best_cl].cluster;
        cl.count++;
        cl.econ_sum += e;
        cl.social_sum += s;
        cl.anti_est_sum += a.getVariable<float>("anti_est");
        cl.organizer_skill_sum += a.getVariable<float>("organizer_skill");
        cl.donor_access_sum += a.getVariable<float>("donor_access");
        cl.launch_tendency_sum += a.getVariable<float>("launch_tendency");
        cl.field_reach_sum += a.getVariable<float>("field_reach");
        unsigned int hr = a.getVariable<unsigned int>("home_region");
        if (hr < REGION_COUNT) cl.region_counts[hr]++;
    }

    // Update centroids from full assignment
    for (auto& r : results) {
        if (r.cluster.count > 0u) {
            r.econ_centroid = static_cast<float>(r.cluster.econ_sum / r.cluster.count);
            r.social_centroid = static_cast<float>(r.cluster.social_sum / r.cluster.count);
        }
    }

    // Filter by minimum size and regional breadth
    std::vector<GridClusterResult> filtered;
    for (auto& r : results) {
        if (r.cluster.count < min_size) continue;
        unsigned int active_regions = 0u;
        for (unsigned int i = 0u; i < REGION_COUNT; ++i)
            if (r.cluster.region_counts[i] >= 2u) ++active_regions;
        if (active_regions >= 2u) filtered.push_back(std::move(r));
    }
    return filtered;
}

// ===== SVG PARLIAMENT CHART =====

std::string BuildParliamentSvg(const std::vector<PartyResult>& results,
                                const unsigned int total_seats) {
    // Sort parties by economic ideology (left to right) for seat ordering
    std::vector<PartyResult> ordered = results;
    std::sort(ordered.begin(), ordered.end(),
        [](const PartyResult& a, const PartyResult& b) { return a.econ_ideology < b.econ_ideology; });

    // Filter to parties with seats
    std::vector<PartyResult> with_seats;
    for (const auto& pr : ordered) {
        if (pr.seats > 0u) with_seats.push_back(pr);
    }
    if (with_seats.empty()) {
        return R"(<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300"><text x="250" y="150" text-anchor="middle" fill="#64748b" font-family="Inter,sans-serif">No seats allocated</text></svg>)";
    }

    // Layout constants
    const float svg_w = 500.0f;
    const float svg_h = 310.0f;
    const float cx_center = svg_w / 2.0f;
    const float cy_base = 240.0f;    // Baseline of the hemicycle (bottom of arcs)
    const float r_inner = 60.0f;
    const float r_outer = 200.0f;
    const float angle_pad = 0.08f;   // Padding at edges (radians)
    const float angle_min = angle_pad;
    const float angle_max = kPi - angle_pad;
    const float dot_r = 3.5f;

    // Determine number of rows
    unsigned int num_rows;
    if (total_seats > 200u) num_rows = 9u;
    else if (total_seats > 120u) num_rows = 7u;
    else if (total_seats > 60u) num_rows = 6u;
    else if (total_seats > 30u) num_rows = 5u;
    else num_rows = 4u;

    // Compute radius and arc length for each row
    std::vector<float> row_radii(num_rows);
    float total_arc = 0.0f;
    for (unsigned int row = 0u; row < num_rows; ++row) {
        row_radii[row] = r_inner + (r_outer - r_inner) * static_cast<float>(row) / static_cast<float>(num_rows - 1u);
        total_arc += row_radii[row];  // Arc length ∝ radius (angle span is same)
    }

    // Distribute total seats across rows proportional to arc length
    unsigned int total_with_seats_count = 0u;
    for (const auto& pr : with_seats) total_with_seats_count += pr.seats;

    std::vector<unsigned int> row_seat_count(num_rows, 0u);
    unsigned int allocated = 0u;
    for (unsigned int row = 0u; row < num_rows; ++row) {
        float frac = row_radii[row] / total_arc;
        unsigned int n = static_cast<unsigned int>(std::round(frac * static_cast<float>(total_with_seats_count)));
        row_seat_count[row] = n;
        allocated += n;
    }
    // Fix rounding error
    while (allocated < total_with_seats_count) { row_seat_count[num_rows - 1u]++; allocated++; }
    while (allocated > total_with_seats_count) {
        for (unsigned int row = 0u; row < num_rows && allocated > total_with_seats_count; ++row) {
            if (row_seat_count[row] > 1u) { row_seat_count[row]--; allocated--; }
        }
    }

    std::ostringstream svg;
    svg << std::fixed << std::setprecision(3);
    svg << R"(<?xml version="1.0" encoding="UTF-8"?>)" "\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
        << static_cast<int>(svg_w) << " " << static_cast<int>(svg_h) << "\">\n"
        << "<rect width=\"" << static_cast<int>(svg_w) << "\" height=\""
        << static_cast<int>(svg_h) << "\" fill=\"#f8fafc\" rx=\"12\"/>\n";

    // Render seats row by row. Within each row, each party gets seats
    // proportional to its share of total, ensuring radial party boundaries.
    for (unsigned int row = 0u; row < num_rows; ++row) {
        unsigned int n = row_seat_count[row];
        if (n == 0u) continue;
        float r = row_radii[row];

        // Distribute this row's seats among parties proportionally
        struct RowParty { std::string color; std::string name; unsigned int count; };
        std::vector<RowParty> row_parties;
        unsigned int row_allocated = 0u;
        for (const auto& pr : with_seats) {
            float frac = static_cast<float>(pr.seats) / static_cast<float>(total_with_seats_count);
            unsigned int pn = static_cast<unsigned int>(std::round(frac * static_cast<float>(n)));
            row_parties.push_back({pr.color, pr.name, pn});
            row_allocated += pn;
        }
        // Fix rounding for this row
        while (row_allocated < n) {
            // Add to the largest party
            unsigned int best = 0u;
            for (unsigned int pi = 1u; pi < row_parties.size(); ++pi)
                if (with_seats[pi].seats > with_seats[best].seats) best = pi;
            row_parties[best].count++;
            row_allocated++;
        }
        while (row_allocated > n) {
            for (unsigned int pi = 0u; pi < row_parties.size() && row_allocated > n; ++pi) {
                if (row_parties[pi].count > 0u) { row_parties[pi].count--; row_allocated--; }
            }
        }

        // Render each party's seats contiguously within this row
        unsigned int seat_pos = 0u;
        for (const auto& rp : row_parties) {
            for (unsigned int s = 0u; s < rp.count; ++s) {
                float angle;
                if (n == 1u) {
                    angle = (angle_min + angle_max) / 2.0f;
                } else {
                    angle = angle_min + (angle_max - angle_min) * static_cast<float>(seat_pos) / static_cast<float>(n - 1u);
                }
                float px = cx_center - r * cosf(angle);
                float py = cy_base - r * sinf(angle);
                svg << "<circle cx=\"" << px << "\" cy=\"" << py
                    << "\" r=\"" << dot_r << "\" fill=\"" << rp.color
                    << "\"><title>" << rp.name << "</title></circle>\n";
                ++seat_pos;
            }
        }
    }

    // Title
    svg << "<text x=\"" << cx_center << "\" y=\"24\" text-anchor=\"middle\" "
        << "font-family=\"Inter,system-ui,sans-serif\" font-size=\"14\" font-weight=\"700\" fill=\"#1e293b\">"
        << "Wisconsin Legislature (" << total_with_seats_count << " seats)</text>\n";

    // Legend – centered below the hemicycle
    float legend_y = cy_base + 16.0f;
    // Measure total legend width first
    float item_spacing = 16.0f;  // gap between items
    float total_legend_w = 0.0f;
    for (const auto& pr : with_seats) {
        // swatch + gap + text at ~7px/char for font-size 10
        float text_w = static_cast<float>(pr.name.size() + 5u) * 7.0f;
        total_legend_w += 10.0f + 5.0f + text_w + item_spacing;
    }
    total_legend_w -= item_spacing; // no trailing gap
    float legend_x = cx_center - total_legend_w / 2.0f;
    for (const auto& pr : with_seats) {
        svg << "<rect x=\"" << legend_x << "\" y=\"" << legend_y
            << "\" width=\"10\" height=\"10\" rx=\"2\" fill=\"" << pr.color << "\"/>\n";
        legend_x += 15.0f;
        svg << "<text x=\"" << legend_x << "\" y=\"" << (legend_y + 9.0f)
            << "\" font-family=\"Inter,system-ui,sans-serif\" font-size=\"10\" fill=\"#475569\">"
            << pr.name << " (" << pr.seats << ")</text>\n";
        float text_w = static_cast<float>(pr.name.size() + 5u) * 7.0f;
        legend_x += text_w + item_spacing;
    }

    svg << "</svg>\n";
    return svg.str();
}

// ===== HTML REPORT =====

std::string BuildHtmlReport(const std::vector<PartyResult>& results,
                             const unsigned int total_seats,
                             const std::uint64_t total_votes,
                             const std::uint64_t total_voters,
                             const unsigned int divisor,
                             const float threshold,
                             const std::string& svg_file) {
    float turnout_pct = total_voters > 0u ? 100.0f * static_cast<float>(total_votes) / static_cast<float>(total_voters) : 0.0f;

    std::ostringstream h;
    h << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
      << "<title>Wisconsin PR Simulation Results</title>\n"
      << "<style>body{font-family:Inter,sans-serif;background:#0f172a;color:#e2e8f0;max-width:900px;margin:0 auto;padding:2rem}"
      << "h1{color:#38bdf8;text-align:center}table{width:100%;border-collapse:collapse;margin:1rem 0}"
      << "th,td{padding:.5rem .75rem;text-align:left;border-bottom:1px solid #334155}"
      << "th{color:#94a3b8}tr:hover{background:#1e293b}"
      << ".swatch{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:6px;vertical-align:middle}"
      << ".card{background:#1e293b;border-radius:10px;padding:1.5rem;margin:1rem 0}"
      << ".stat{color:#94a3b8;font-size:.85rem}.val{color:#f8fafc;font-size:1.3rem;font-weight:700}"
      << ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}"
      << "</style></head><body>\n"
      << "<h1>Wisconsin PR Simulation</h1>\n";

    h << "<div class=\"grid\">\n"
      << "<div class=\"card\"><div class=\"stat\">Total Seats</div><div class=\"val\">" << total_seats << "</div></div>\n"
      << "<div class=\"card\"><div class=\"stat\">Turnout</div><div class=\"val\">" << std::fixed << std::setprecision(1) << turnout_pct << "%</div></div>\n"
      << "<div class=\"card\"><div class=\"stat\">Method</div><div class=\"val\">" << DivisorMethodLabel(divisor) << "</div></div>\n"
      << "</div>\n";

    h << "<div class=\"card\">\n<h2>Parliament</h2>\n"
      << "<img src=\"" << svg_file << "\" alt=\"Parliament\" style=\"width:100%\">\n"
      << "</div>\n";

    h << "<div class=\"card\">\n<h2>Results</h2>\n"
      << "<table><tr><th>Party</th><th>Votes</th><th>Share</th><th>Seats</th></tr>\n";
    for (const auto& pr : results) {
        h << "<tr><td><span class=\"swatch\" style=\"background:" << pr.color << "\"></span>"
          << pr.name << "</td><td>" << pr.votes << "</td><td>"
          << std::fixed << std::setprecision(1) << (pr.share * 100.0f) << "%</td><td>"
          << pr.seats << "</td></tr>\n";
    }
    h << "</table></div>\n";

    unsigned int majority_target = total_seats / 2 + 1;
    std::string governance_text;
    bool single_party_majority = false;
    for (const auto& pr : results) {
        if (pr.seats >= majority_target) {
            governance_text = "<b>" + pr.name + "</b> has an absolute majority (" + std::to_string(pr.seats) + " seats) and can govern alone.";
            single_party_majority = true;
            break;
        }
    }

    if (!single_party_majority) {
        std::vector<PartyResult> with_seats;
        for (const auto& pr : results) if (pr.seats > 0) with_seats.push_back(pr);
        
        if (with_seats.size() >= 2) {
            governance_text = "No single party holds a majority (" + std::to_string(majority_target) + " seats required). A coalition government is required.<br><br>";
            governance_text += "<b>Simplest Potential Coalition (Largest Parties):</b><br><br>";
            unsigned int seats_accum = with_seats[0].seats;
            governance_text += "<span class=\"swatch\" style=\"background:" + with_seats[0].color + "\"></span>" + with_seats[0].name;
            for (size_t i = 1; i < with_seats.size(); ++i) {
                seats_accum += with_seats[i].seats;
                governance_text += " + <span class=\"swatch\" style=\"background:" + with_seats[i].color + "\"></span>" + with_seats[i].name;
                if (seats_accum >= majority_target) {
                    governance_text += " <b>(" + std::to_string(seats_accum) + " seats total)</b>";
                    break;
                }
            }
        } else {
            governance_text = "No party holds a majority.";
        }
    }

    h << "<div class=\"card\">\n<h2>Governance & Coalition</h2>\n"
      << "<p style=\"line-height:1.5\">" << governance_text << "</p>\n"
      << "</div>\n";

    h << "</body></html>\n";
    return h.str();
}

void WriteResultsArtifacts(const std::vector<PartyResult>& results,
                            const unsigned int total_seats,
                            const std::uint64_t total_votes,
                            const std::uint64_t total_voters,
                            const unsigned int divisor,
                            const float threshold) {
    std::filesystem::create_directories("reports");
    std::string svg = BuildParliamentSvg(results, total_seats);
    {
        std::ofstream f("reports/wisconsin_pr_results.svg");
        f << svg;
    }
    std::string html = BuildHtmlReport(results, total_seats, total_votes, total_voters,
                                        divisor, threshold, "wisconsin_pr_results.svg");
    {
        std::ofstream f("reports/wisconsin_pr_results.html");
        f << html;
    }
    std::printf("Reports written to reports/\n");
}

// ===== INIT FUNCTIONS =====

// Voter ideology cluster definitions (MLSP 2026 + ACS 2024 derived)
struct VoterCluster {
    float econ_mu, social_mu, conv_mu, share;
    float econ_sigma, social_sigma;
};

constexpr std::array<VoterCluster, 6> kVoterClusters = {{
    // econ_mu social_mu conv_mu share  econ_σ  social_σ
    { -0.45f,  -0.50f,   0.55f,  0.18f, 0.20f,  0.22f },  // Urban progressive
    { -0.30f,   0.15f,   0.45f,  0.16f, 0.25f,  0.20f },  // Labor/union
    {  0.10f,  -0.05f,   0.30f,  0.22f, 0.28f,  0.28f },  // Suburban moderate
    {  0.35f,   0.45f,   0.50f,  0.24f, 0.22f,  0.20f },  // Rural traditional
    {  0.40f,  -0.30f,   0.65f,  0.10f, 0.18f,  0.18f },  // Liberty-oriented
    { -0.05f,   0.20f,   0.25f,  0.10f, 0.30f,  0.30f },  // Disaffected/low-info
}};

FLAMEGPU_INIT_FUNCTION(SeedVoters) {
    const unsigned int voter_count = FLAMEGPU->environment.getProperty<unsigned int>("VOTER_COUNT");
    const unsigned int seed = FLAMEGPU->environment.getProperty<unsigned int>("RANDOM_SEED");
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    flamegpu::HostAgentAPI voters = FLAMEGPU->agent("voter");

    for (unsigned int i = 0u; i < voter_count; ++i) {
        flamegpu::HostNewAgentAPI v = voters.newAgent();

        // 1. Choose cluster
        float draw = u01(rng);
        unsigned int cl_idx = 0u;
        float cum = 0.0f;
        for (unsigned int c = 0u; c < kVoterClusters.size(); ++c) {
            cum += kVoterClusters[c].share;
            if (draw <= cum) { cl_idx = c; break; }
        }
        const auto& cl = kVoterClusters[cl_idx];

        // 2. Choose region
        unsigned int region_id = ChooseRegion(u01(rng), false);
        const auto& rp = kRegionProfiles[region_id];

        // 3. 2D ideology with cluster + regional shift + noise
        float econ = ClampFloat(cl.econ_mu + rp.econ_shift * 0.3f + norm(rng) * cl.econ_sigma, -1.0f, 1.0f);
        float social = ClampFloat(cl.social_mu + rp.social_shift * 0.3f + norm(rng) * cl.social_sigma, -1.0f, 1.0f);

        // 4. Conviction: stronger at extremes, weaker at center
        float dist = sqrtf(econ * econ + social * social);
        float sophistication = ClampFloat(0.3f + 0.4f * rp.college_share + norm(rng) * 0.15f, 0.0f, 1.0f);
        float conviction = ClampFloat(
            0.15f + 0.45f * dist + 0.20f * sophistication + norm(rng) * 0.10f,
            0.05f, 0.95f);

        // 5. Other demographics from census data
        float turnout_base = ClampFloat(0.55f + rp.turnout_bonus + 0.12f * sophistication + norm(rng) * 0.12f, 0.15f, 0.99f);
        unsigned int college = (u01(rng) < rp.college_share) ? 1u : 0u;
        unsigned int union_member = (u01(rng) < rp.union_share) ? 1u : 0u;
        float religiosity = ClampFloat(rp.religiosity + norm(rng) * 0.18f, 0.0f, 1.0f);
        float anti_est = ClampFloat(cl.conv_mu < 0.35f ? 0.35f + norm(rng) * 0.20f : 0.25f + norm(rng) * 0.18f, 0.0f, 1.0f);
        float minor_openness = ClampFloat(0.15f + 0.30f * anti_est + norm(rng) * 0.12f, 0.0f, 1.0f);

        // 6. Spatial position from region centroid + jitter
        float jitter_scale = 3.0f * (1.0f + 0.5f * (1.0f - rp.urbanity));
        float pos_x = ClampFloat(rp.center_x + norm(rng) * jitter_scale, kSpatialMin, kSpatialMax);
        float pos_y = ClampFloat(rp.center_y + norm(rng) * jitter_scale, kSpatialMin, kSpatialMax);

        v.setVariable<float>("econ_ideology", econ);
        v.setVariable<float>("social_ideology", social);
        v.setVariable<float>("conviction", conviction);
        v.setVariable<float>("turnout", turnout_base);
        v.setVariable<float>("sophistication", sophistication);
        v.setVariable<float>("anti_est", anti_est);
        v.setVariable<float>("urbanity", rp.urbanity);
        v.setVariable<unsigned int>("college", college);
        v.setVariable<unsigned int>("union_member", union_member);
        v.setVariable<float>("religiosity", religiosity);
        v.setVariable<float>("minor_openness", minor_openness);
        v.setVariable<unsigned int>("region_id", region_id);
        v.setVariable<unsigned int>("vote_choice", kAbstain);
        v.setVariable<unsigned int>("previous_vote", kAbstain);
        v.setVariable<float>("party_loyalty", 0.0f);
        v.setVariable<unsigned int>("vote_streak", 0u);
        v.setVariable<float>("perceived_local_support", 0.0f);
        v.setVariable<float>("perceived_local_turnout", 0.5f);
        v.setVariable<float>("pos_x", pos_x);
        v.setVariable<float>("pos_y", pos_y);
    }
    std::printf("Seeded %u voters across %u regions\n", voter_count, static_cast<unsigned int>(REGION_COUNT));
}

FLAMEGPU_INIT_FUNCTION(SeedActivists) {
    const unsigned int count = FLAMEGPU->environment.getProperty<unsigned int>("ACTIVIST_COUNT");
    const unsigned int seed = FLAMEGPU->environment.getProperty<unsigned int>("RANDOM_SEED");
    std::mt19937 rng(seed + 7777u);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    flamegpu::HostAgentAPI activists = FLAMEGPU->agent("activist");

    for (unsigned int i = 0u; i < count; ++i) {
        flamegpu::HostNewAgentAPI a = activists.newAgent();
        unsigned int region_id = ChooseRegion(u01(rng), true);
        const auto& rp = kRegionProfiles[region_id];

        float econ = ClampFloat(rp.econ_shift * 1.5f + norm(rng) * 0.35f, -1.0f, 1.0f);
        float social = ClampFloat(rp.social_shift * 1.5f + norm(rng) * 0.35f, -1.0f, 1.0f);
        float anti_est = ClampFloat(0.35f + norm(rng) * 0.22f, 0.0f, 1.0f);

        a.setVariable<float>("econ_ideology", econ);
        a.setVariable<float>("social_ideology", social);
        a.setVariable<float>("anti_est", anti_est);
        a.setVariable<unsigned int>("home_region", region_id);
        a.setVariable<float>("organizer_skill", ClampFloat(0.35f + norm(rng) * 0.20f, 0.05f, 1.0f));
        a.setVariable<float>("donor_access", ClampFloat(0.2f + 0.25f * rp.college_share + norm(rng) * 0.15f, 0.0f, 1.0f));
        a.setVariable<float>("launch_tendency", ClampFloat(0.3f + norm(rng) * 0.18f, 0.0f, 1.0f));
        a.setVariable<float>("field_reach", ClampFloat(0.3f + norm(rng) * 0.20f, 0.05f, 1.0f));
        a.setVariable<float>("effort", 0.5f);
        a.setVariable<unsigned int>("affiliated_party", kAbstain);
    }
    std::printf("Seeded %u activists\n", count);
}

FLAMEGPU_INIT_FUNCTION(SeedLegacyParties) {
    flamegpu::HostAgentAPI parties = FLAMEGPU->agent("party");

    // Democratic Party — legacy seed
    {
        auto p = parties.newAgent();
        p.setVariable<unsigned int>("party_id", 0u);
        p.setVariable<float>("econ_ideology", -0.28f);
        p.setVariable<float>("social_ideology", -0.18f);
        p.setVariable<float>("urban_orientation", 0.68f);
        p.setVariable<float>("anti_est_posture", 0.15f);
        p.setVariable<float>("religiosity_align", -0.20f);
        p.setVariable<float>("union_alignment", 0.45f);
        p.setVariable<float>("college_alignment", 0.35f);
        p.setVariable<float>("establishment_age", 0.92f);
        p.setVariable<float>("organization", 0.80f);
        p.setVariable<float>("brand", 0.75f);
        p.setVariable<float>("credibility", 0.70f);
        p.setVariable<float>("projected_viability", 0.48f);
        p.setVariable<float>("field_strength", 0.42f);
        p.setVariable<float>("momentum", 0.50f);
        p.setVariable<float>("fundraising", 0.70f);
        p.setVariable<float>("media_reach", 0.75f);
        p.setVariable<unsigned int>("is_alive", 1u);
        p.setVariable<unsigned int>("is_legacy", 1u);
        p.setVariable<unsigned int>("legacy_id", 0u);
        p.setVariable<unsigned int>("consecutive_low_viability", 0u);
    }
    // Republican Party — legacy seed
    {
        auto p = parties.newAgent();
        p.setVariable<unsigned int>("party_id", 1u);
        p.setVariable<float>("econ_ideology", 0.32f);
        p.setVariable<float>("social_ideology", 0.35f);
        p.setVariable<float>("urban_orientation", 0.35f);
        p.setVariable<float>("anti_est_posture", 0.18f);
        p.setVariable<float>("religiosity_align", 0.40f);
        p.setVariable<float>("union_alignment", -0.15f);
        p.setVariable<float>("college_alignment", -0.10f);
        p.setVariable<float>("establishment_age", 0.92f);
        p.setVariable<float>("organization", 0.78f);
        p.setVariable<float>("brand", 0.72f);
        p.setVariable<float>("credibility", 0.65f);
        p.setVariable<float>("projected_viability", 0.50f);
        p.setVariable<float>("field_strength", 0.40f);
        p.setVariable<float>("momentum", 0.52f);
        p.setVariable<float>("fundraising", 0.68f);
        p.setVariable<float>("media_reach", 0.72f);
        p.setVariable<unsigned int>("is_alive", 1u);
        p.setVariable<unsigned int>("is_legacy", 1u);
        p.setVariable<unsigned int>("legacy_id", 1u);
        p.setVariable<unsigned int>("consecutive_low_viability", 0u);
    }
    std::printf("Seeded 2 legacy parties (Democratic, Republican)\n");
}

FLAMEGPU_INIT_FUNCTION(DiscoverEmergentParties) {
    flamegpu::DeviceAgentVector activists = FLAMEGPU->agent("activist").getPopulationData();
    auto clusters = ClusterActivists(activists, MIN_CLUSTER_ACTIVISTS);

    unsigned int next_id = 2u;  // 0=DEM, 1=REP
    flamegpu::HostAgentAPI parties = FLAMEGPU->agent("party");
    flamegpu::DeviceAgentVector existing_parties = parties.getPopulationData();

    // Check each cluster against existing legacy parties
    for (auto& cr : clusters) {
        if (next_id >= MAX_PARTIES) break;
        float ce = cr.econ_centroid;
        float cs = cr.social_centroid;

        // Skip if too close to a legacy party
        bool too_close = false;
        for (const auto& ep : existing_parties) {
            float de = ce - ep.getVariable<float>("econ_ideology");
            float ds = cs - ep.getVariable<float>("social_ideology");
            if (sqrtf(de * de + ds * ds) < 0.20f) { too_close = true; break; }
        }
        if (too_close) continue;

        // Birth emergent party from cluster
        const auto& cl = cr.cluster;
        float anti_est = AverageOrZero(cl.anti_est_sum, cl.count);
        float avg_skill = AverageOrZero(cl.organizer_skill_sum, cl.count);
        float avg_donor = AverageOrZero(cl.donor_access_sum, cl.count);
        float avg_reach = AverageOrZero(cl.field_reach_sum, cl.count);

        // Compute urban orientation from regional presence
        float urban_sum = 0.0f, urban_wt = 0.0f;
        for (unsigned int r = 0u; r < REGION_COUNT; ++r) {
            float w = static_cast<float>(cl.region_counts[r]);
            urban_sum += w * kRegionProfiles[r].urbanity;
            urban_wt += w;
        }
        float urban_orient = urban_wt > 0.0f ? urban_sum / urban_wt : 0.5f;

        // Religiosity/union/college alignment from regional correlation
        float relig_sum = 0.0f, union_sum = 0.0f, college_sum = 0.0f;
        for (unsigned int r = 0u; r < REGION_COUNT; ++r) {
            float w = static_cast<float>(cl.region_counts[r]) / std::max(1.0f, static_cast<float>(cl.count));
            relig_sum += w * (kRegionProfiles[r].religiosity - 0.45f) * 2.0f;
            union_sum += w * (kRegionProfiles[r].union_share - 0.06f) * 8.0f;
            college_sum += w * (kRegionProfiles[r].college_share - 0.32f) * 3.0f;
        }

        auto p = parties.newAgent();
        p.setVariable<unsigned int>("party_id", next_id);
        p.setVariable<float>("econ_ideology", ce);
        p.setVariable<float>("social_ideology", cs);
        p.setVariable<float>("urban_orientation", ClampFloat(urban_orient, 0.0f, 1.0f));
        p.setVariable<float>("anti_est_posture", ClampFloat(anti_est, 0.0f, 1.0f));
        p.setVariable<float>("religiosity_align", ClampFloat(relig_sum, -1.0f, 1.0f));
        p.setVariable<float>("union_alignment", ClampFloat(union_sum, -1.0f, 1.0f));
        p.setVariable<float>("college_alignment", ClampFloat(college_sum, -1.0f, 1.0f));
        p.setVariable<float>("establishment_age", 0.0f);
        p.setVariable<float>("organization", ClampFloat(avg_skill * 0.6f, 0.05f, 0.50f));
        p.setVariable<float>("brand", ClampFloat(0.05f + avg_reach * 0.15f, 0.05f, 0.30f));
        p.setVariable<float>("credibility", ClampFloat(0.10f + avg_skill * 0.25f, 0.05f, 0.40f));
        p.setVariable<float>("projected_viability", ClampFloat(0.02f + static_cast<float>(cl.count) / 5000.0f, 0.02f, 0.15f));
        p.setVariable<float>("field_strength", ClampFloat(avg_reach * 0.5f, 0.02f, 0.30f));
        p.setVariable<float>("momentum", 0.10f);
        p.setVariable<float>("fundraising", ClampFloat(avg_donor * 0.4f, 0.02f, 0.25f));
        p.setVariable<float>("media_reach", ClampFloat(0.03f + avg_donor * 0.15f, 0.02f, 0.20f));
        p.setVariable<unsigned int>("is_alive", 1u);
        p.setVariable<unsigned int>("is_legacy", 0u);
        p.setVariable<unsigned int>("legacy_id", next_id);
        p.setVariable<unsigned int>("consecutive_low_viability", 0u);

        std::string name = GeneratePartyName(ce, cs, anti_est, urban_orient, false, 0u);
        std::printf("  Emerged: %s (econ=%.2f, social=%.2f, activists=%u)\n",
                    name.c_str(), ce, cs, cl.count);
        ++next_id;
    }
    std::printf("Total parties: %u\n", next_id);
}

// ===== AGENT FUNCTION CONDITIONS =====

FLAMEGPU_AGENT_FUNCTION_CONDITION(PartyAliveCondition) {
    return FLAMEGPU->getVariable<unsigned int>("is_alive") == 1u;
}

FLAMEGPU_AGENT_FUNCTION_CONDITION(CampaignPhaseCondition) {
    return FLAMEGPU->environment.getProperty<unsigned int>("PHASE") >= 1u;
}

// ===== AGENT FUNCTIONS =====

FLAMEGPU_AGENT_FUNCTION(PartyBroadcast, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<unsigned int>("party_id", FLAMEGPU->getVariable<unsigned int>("party_id"));
    FLAMEGPU->message_out.setVariable<float>("econ_ideology", FLAMEGPU->getVariable<float>("econ_ideology"));
    FLAMEGPU->message_out.setVariable<float>("social_ideology", FLAMEGPU->getVariable<float>("social_ideology"));
    FLAMEGPU->message_out.setVariable<float>("urban_orientation", FLAMEGPU->getVariable<float>("urban_orientation"));
    FLAMEGPU->message_out.setVariable<float>("anti_est_posture", FLAMEGPU->getVariable<float>("anti_est_posture"));
    FLAMEGPU->message_out.setVariable<float>("religiosity_align", FLAMEGPU->getVariable<float>("religiosity_align"));
    FLAMEGPU->message_out.setVariable<float>("union_alignment", FLAMEGPU->getVariable<float>("union_alignment"));
    FLAMEGPU->message_out.setVariable<float>("college_alignment", FLAMEGPU->getVariable<float>("college_alignment"));
    FLAMEGPU->message_out.setVariable<float>("establishment_age", FLAMEGPU->getVariable<float>("establishment_age"));
    FLAMEGPU->message_out.setVariable<float>("organization", FLAMEGPU->getVariable<float>("organization"));
    FLAMEGPU->message_out.setVariable<float>("brand", FLAMEGPU->getVariable<float>("brand"));
    FLAMEGPU->message_out.setVariable<float>("credibility", FLAMEGPU->getVariable<float>("credibility"));
    FLAMEGPU->message_out.setVariable<float>("projected_viability", FLAMEGPU->getVariable<float>("projected_viability"));
    FLAMEGPU->message_out.setVariable<float>("momentum", FLAMEGPU->getVariable<float>("momentum"));
    FLAMEGPU->message_out.setVariable<float>("media_reach", FLAMEGPU->getVariable<float>("media_reach"));
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(ActivistCanvass, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    // Individual activist chooses party closest to own beliefs
    const float my_econ = FLAMEGPU->getVariable<float>("econ_ideology");
    const float my_social = FLAMEGPU->getVariable<float>("social_ideology");
    float best_affinity = -1.0f;
    unsigned int best_party = kAbstain;

    for (const auto& msg : FLAMEGPU->message_in) {
        const float de = my_econ - msg.getVariable<float>("econ_ideology");
        const float ds = my_social - msg.getVariable<float>("social_ideology");
        const float affinity = 1.0f - sqrtf(de * de + ds * ds);
        if (affinity > best_affinity) {
            best_affinity = affinity;
            best_party = msg.getVariable<unsigned int>("party_id");
        }
    }
    FLAMEGPU->setVariable<unsigned int>("affiliated_party", best_party);

    // Effort based on individual motivation
    const float base_reach = FLAMEGPU->getVariable<float>("field_reach");
    const float effort = base_reach * (0.6f + 0.4f * fmaxf(0.0f, best_affinity))
        * (0.7f + 0.3f * FLAMEGPU->getVariable<float>("launch_tendency"));
    FLAMEGPU->setVariable<float>("effort", ClampFloat(effort, 0.0f, 1.0f));
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(VoterPostSignal, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("pos_x"),
        FLAMEGPU->getVariable<float>("pos_y"));
    FLAMEGPU->message_out.setVariable<unsigned int>("previous_vote",
        FLAMEGPU->getVariable<unsigned int>("vote_choice"));
    FLAMEGPU->message_out.setVariable<float>("turnout_signal",
        FLAMEGPU->getVariable<float>("turnout"));
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(VoterObserveNeighbors, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const float my_x = FLAMEGPU->getVariable<float>("pos_x");
    const float my_y = FLAMEGPU->getVariable<float>("pos_y");
    const unsigned int my_prev = FLAMEGPU->getVariable<unsigned int>("previous_vote");

    unsigned int neighbor_count = 0u;
    unsigned int same_choice_count = 0u;
    float turnout_sum = 0.0f;

    for (const auto& msg : FLAMEGPU->message_in(my_x, my_y)) {
        ++neighbor_count;
        if (msg.getVariable<unsigned int>("previous_vote") == my_prev && my_prev != kAbstain) {
            ++same_choice_count;
        }
        turnout_sum += msg.getVariable<float>("turnout_signal");
    }

    if (neighbor_count > 0u) {
        FLAMEGPU->setVariable<float>("perceived_local_support",
            static_cast<float>(same_choice_count) / static_cast<float>(neighbor_count));
        FLAMEGPU->setVariable<float>("perceived_local_turnout",
            turnout_sum / static_cast<float>(neighbor_count));
    }
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(VoterChoose, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    // --- Individual voter decision ---
    const float voter_econ = FLAMEGPU->getVariable<float>("econ_ideology");
    const float voter_social = FLAMEGPU->getVariable<float>("social_ideology");
    const float conviction = FLAMEGPU->getVariable<float>("conviction");
    const float turnout_prob = FLAMEGPU->getVariable<float>("turnout");
    const float sophistication = FLAMEGPU->getVariable<float>("sophistication");
    const float anti_est = FLAMEGPU->getVariable<float>("anti_est");
    const float urbanity = FLAMEGPU->getVariable<float>("urbanity");
    const float college_f = static_cast<float>(FLAMEGPU->getVariable<unsigned int>("college"));
    const float union_f = static_cast<float>(FLAMEGPU->getVariable<unsigned int>("union_member"));
    const float religiosity = FLAMEGPU->getVariable<float>("religiosity");
    const float minor_openness = FLAMEGPU->getVariable<float>("minor_openness");
    const unsigned int region_id = FLAMEGPU->getVariable<unsigned int>("region_id");
    const unsigned int prev_vote = FLAMEGPU->getVariable<unsigned int>("vote_choice");
    const float party_loyalty = FLAMEGPU->getVariable<float>("party_loyalty");
    const unsigned int vote_streak = FLAMEGPU->getVariable<unsigned int>("vote_streak");
    const float local_support = FLAMEGPU->getVariable<float>("perceived_local_support");
    const float local_turnout = FLAMEGPU->getVariable<float>("perceived_local_turnout");

    // Social influence on turnout
    float adjusted_turnout = turnout_prob * (0.85f + 0.15f * local_turnout);

    // Turnout threshold
    const float threshold = FLAMEGPU->environment.getProperty<float>("THRESHOLD");
    const float draw_turnout = FLAMEGPU->random.uniform<float>();
    if (draw_turnout > adjusted_turnout) {
        FLAMEGPU->setVariable<unsigned int>("previous_vote", prev_vote);
        FLAMEGPU->setVariable<unsigned int>("vote_choice", kAbstain);
        FLAMEGPU->setVariable<float>("party_loyalty", ClampFloat(party_loyalty * 0.9f, 0.0f, 1.0f));
        return flamegpu::ALIVE;
    }

    // Conviction-modulated weights
    const float ideology_weight = 1.10f + 0.80f * conviction;
    const float signal_weight = 0.40f - 0.20f * conviction;

    // Evaluate every party via broadcast messages
    unsigned int candidate_ids[MAX_PARTIES]{};
    float candidate_scores[MAX_PARTIES]{};
    unsigned int candidate_count = 0u;

    for (const auto& msg : FLAMEGPU->message_in) {
        if (candidate_count >= MAX_PARTIES) break;
        const unsigned int pid = msg.getVariable<unsigned int>("party_id");
        const float p_econ = msg.getVariable<float>("econ_ideology");
        const float p_social = msg.getVariable<float>("social_ideology");
        const float p_urban = msg.getVariable<float>("urban_orientation");
        const float p_anti_est = msg.getVariable<float>("anti_est_posture");
        const float p_relig = msg.getVariable<float>("religiosity_align");
        const float p_union = msg.getVariable<float>("union_alignment");
        const float p_college = msg.getVariable<float>("college_alignment");
        const float p_est_age = msg.getVariable<float>("establishment_age");
        const float org = msg.getVariable<float>("organization");
        const float brand = msg.getVariable<float>("brand");
        const float cred = msg.getVariable<float>("credibility");
        const float viab = msg.getVariable<float>("projected_viability");
        const float mom = msg.getVariable<float>("momentum");
        const float media = msg.getVariable<float>("media_reach");

        // 2D ideology distance
        const float de = voter_econ - p_econ;
        const float ds = voter_social - p_social;
        const float closeness = fmaxf(0.0f, 1.0f - 0.7f * sqrtf(de * de + ds * ds));

        // Place fit
        const float place_fit = 1.0f - fabsf(urbanity - p_urban);

        // Demographic alignment (dot product)
        const float demographic_fit =
            0.16f * p_college * college_f
            + 0.14f * p_union * union_f
            + 0.13f * p_relig * religiosity
            + 0.11f * place_fit;

        // Individual loyalty
        const float streak_f = static_cast<float>(vote_streak > 10u ? 10u : vote_streak) / 10.0f;
        const float loyalty_bonus = (prev_vote == pid && prev_vote != kAbstain)
            ? party_loyalty * (0.30f + 0.15f * streak_f) : 0.0f;

        // Anti-establishment resonance
        const float outsider_bonus = 0.08f * anti_est * p_anti_est * minor_openness;

        // Establishment trust/distrust
        const float est_factor = anti_est > 0.5f
            ? -0.06f * p_est_age
            :  0.04f * p_est_age;

        // Social conformity (individual perception)
        const float conformity = (1.0f - anti_est) * (1.0f - conviction) * 0.08f;
        const float social_bonus = (prev_vote == pid && prev_vote != kAbstain)
            ? conformity * local_support : 0.0f;

        // Regional contact from environment
        const float regional_contact = FLAMEGPU->environment.getProperty<float, kRegionPartySlots>(
            "REGION_PARTY_CONTACT", FlatRegionParty(region_id, pid));

        // Strategic voting (minor party penalty)
        const float strategic_penalty = (viab < threshold && sophistication > 0.5f)
            ? 0.25f * sophistication * (1.0f - minor_openness) : 0.0f;

        // Final score — conviction modulates which factors dominate
        const float score =
            ideology_weight * closeness
            + signal_weight * org
            + signal_weight * brand
            + signal_weight * media
            + 0.28f * cred
            + 0.38f * regional_contact
            + 0.14f * mom
            + demographic_fit
            + loyalty_bonus
            + outsider_bonus
            + est_factor
            + social_bonus
            - strategic_penalty;

        candidate_ids[candidate_count] = pid;
        candidate_scores[candidate_count] = score;
        ++candidate_count;
    }

    // Choose best party
    unsigned int best_id = kAbstain;
    float best_score = -1e9f;
    for (unsigned int c = 0u; c < candidate_count; ++c) {
        if (candidate_scores[c] > best_score) {
            best_score = candidate_scores[c];
            best_id = candidate_ids[c];
        }
    }

    // Add slight randomness for tie-breaking
    if (candidate_count >= 2u) {
        float weights[MAX_PARTIES]{};
        float max_w = -1e9f;
        for (unsigned int c = 0u; c < candidate_count; ++c) {
            weights[c] = candidate_scores[c];
            if (weights[c] > max_w) max_w = weights[c];
        }
        float sum_w = 0.0f;
        for (unsigned int c = 0u; c < candidate_count; ++c) {
            weights[c] = expf(5.0f * (weights[c] - max_w));
            sum_w += weights[c];
        }
        float roll = FLAMEGPU->random.uniform<float>() * sum_w;
        float cum = 0.0f;
        for (unsigned int c = 0u; c < candidate_count; ++c) {
            cum += weights[c];
            if (roll <= cum) { best_id = candidate_ids[c]; break; }
        }
    }

    // Update memory
    FLAMEGPU->setVariable<unsigned int>("previous_vote", prev_vote);
    FLAMEGPU->setVariable<unsigned int>("vote_choice", best_id);
    if (best_id == prev_vote && prev_vote != kAbstain) {
        FLAMEGPU->setVariable<unsigned int>("vote_streak", vote_streak + 1u);
        FLAMEGPU->setVariable<float>("party_loyalty", ClampFloat(party_loyalty + 0.05f, 0.0f, 1.0f));
    } else {
        FLAMEGPU->setVariable<unsigned int>("vote_streak", 0u);
        FLAMEGPU->setVariable<float>("party_loyalty", ClampFloat(party_loyalty * 0.5f, 0.0f, 1.0f));
    }
    return flamegpu::ALIVE;
}

// ===== HOST STEP FUNCTIONS =====

FLAMEGPU_STEP_FUNCTION(AdvanceCampaignAndAllocateSeats) {
    const unsigned int step = FLAMEGPU->getStepCounter();
    const unsigned int org_steps = FLAMEGPU->environment.getProperty<unsigned int>("ORGANIZING_STEPS");
    const unsigned int total_steps = org_steps + FLAMEGPU->environment.getProperty<unsigned int>("CAMPAIGN_STEPS");
    const bool is_organizing = step < org_steps;
    const bool final_step = (step + 1u == total_steps);

    // Phase transition
    if (step == org_steps) {
        FLAMEGPU->environment.setProperty<unsigned int>("PHASE", 1u);
        std::printf("\n=== CAMPAIGN PHASE BEGINS (step %u) ===\n\n", step);
    }

    // Aggregate activist data for contacts
    flamegpu::DeviceAgentVector activists = FLAMEGPU->agent("activist").getPopulationData();
    flamegpu::DeviceAgentVector party_pop = FLAMEGPU->agent("party").getPopulationData();

    std::unordered_map<unsigned int, PartyState> party_states;
    std::unordered_map<unsigned int, IdeologyCluster> party_clusters;
    std::array<unsigned int, REGION_COUNT> region_act_totals{};

    for (const auto& p : party_pop) {
        unsigned int pid = p.getVariable<unsigned int>("party_id");
        PartyState ps;
        ps.party_id = pid;
        ps.econ_ideology = p.getVariable<float>("econ_ideology");
        ps.social_ideology = p.getVariable<float>("social_ideology");
        ps.urban_orientation = p.getVariable<float>("urban_orientation");
        ps.anti_est_posture = p.getVariable<float>("anti_est_posture");
        ps.organization = p.getVariable<float>("organization");
        ps.brand = p.getVariable<float>("brand");
        ps.credibility = p.getVariable<float>("credibility");
        ps.projected_viability = p.getVariable<float>("projected_viability");
        ps.field_strength = p.getVariable<float>("field_strength");
        ps.momentum = p.getVariable<float>("momentum");
        ps.fundraising = p.getVariable<float>("fundraising");
        ps.media_reach = p.getVariable<float>("media_reach");
        ps.establishment_age = p.getVariable<float>("establishment_age");
        ps.is_alive = p.getVariable<unsigned int>("is_alive");
        ps.is_legacy = p.getVariable<unsigned int>("is_legacy");
        party_states[pid] = ps;
        party_clusters[pid] = IdeologyCluster{};
    }

    for (const auto& a : activists) {
        unsigned int hr = a.getVariable<unsigned int>("home_region");
        unsigned int ap = a.getVariable<unsigned int>("affiliated_party");
        if (hr < REGION_COUNT) region_act_totals[hr]++;
        if (party_clusters.count(ap)) {
            auto& cl = party_clusters[ap];
            cl.count++;
            cl.region_counts[hr]++;
            cl.organizer_skill_sum += a.getVariable<float>("organizer_skill");
            cl.field_reach_sum += a.getVariable<float>("effort");
        }
    }

    auto contacts = BuildContacts(party_states, region_act_totals, party_clusters);
    FLAMEGPU->environment.setProperty<float, kRegionPartySlots>("REGION_PARTY_CONTACT", contacts);

    if (is_organizing) {
        std::printf("[Organizing step %u] Parties active: %zu\n", step, party_states.size());
        return;
    }

    // Campaign phase: count votes and update party stats
    flamegpu::DeviceAgentVector voters = FLAMEGPU->agent("voter").getPopulationData();
    const unsigned int voter_count = static_cast<unsigned int>(voters.size());

    std::unordered_map<unsigned int, std::uint64_t> vote_counts;
    std::uint64_t total_votes = 0u;
    for (const auto& v : voters) {
        unsigned int vc = v.getVariable<unsigned int>("vote_choice");
        if (vc != kAbstain) {
            vote_counts[vc]++;
            ++total_votes;
        }
    }

    // Update party viability
    for (auto& [pid, ps] : party_states) {
        if (!ps.is_alive) continue;
        float share = total_votes > 0u ? static_cast<float>(vote_counts[pid]) / static_cast<float>(total_votes) : 0.0f;
        float new_viab = ClampFloat(share * 2.0f, 0.0f, 1.0f);
        // Find the party agent and update
        for (unsigned int pi = 0u; pi < party_pop.size(); ++pi) {
            auto p = party_pop[pi];
            if (p.getVariable<unsigned int>("party_id") == pid) {
                float old_viab = p.getVariable<float>("projected_viability");
                p.setVariable<float>("projected_viability", 0.4f * old_viab + 0.6f * new_viab);
                p.setVariable<float>("momentum", ClampFloat(new_viab - old_viab + 0.5f, 0.0f, 1.0f));
                float wc = WeightedContact(contacts, pid);
                p.setVariable<float>("field_strength", ClampFloat(wc, 0.0f, 1.0f));
                break;
            }
        }
    }

    float turnout_pct = static_cast<float>(total_votes) * 100.0f / std::max(1u, voter_count);
    std::printf("[Campaign step %u] Turnout: %.1f%% | Votes cast: %" PRIu64 "\n",
                step, turnout_pct, total_votes);
    for (const auto& [pid, count] : vote_counts) {
        float share = static_cast<float>(count) * 100.0f / std::max(static_cast<std::uint64_t>(1), total_votes);
        std::printf("  Party %u: %" PRIu64 " votes (%.1f%%)\n", pid, count, share);
    }

    // Final step: allocate seats and write reports
    if (final_step) {
        const unsigned int total_seats = FLAMEGPU->environment.getProperty<unsigned int>("TOTAL_SEATS");
        const float thresh = FLAMEGPU->environment.getProperty<float>("THRESHOLD");
        const unsigned int div_method = FLAMEGPU->environment.getProperty<unsigned int>("DIVISOR_METHOD");

        std::vector<std::pair<unsigned int, std::uint64_t>> qualified;
        for (const auto& [pid, count] : vote_counts) {
            float share = static_cast<float>(count) / std::max(static_cast<std::uint64_t>(1), total_votes);
            if (share >= thresh) qualified.push_back({pid, count});
        }

        auto seat_map = AllocateDivisorSeats(qualified, total_seats, div_method);

        std::printf("\n=== FINAL RESULTS ===\n");
        std::vector<PartyResult> results;
        for (const auto& [pid, count] : vote_counts) {
            PartyResult pr;
            pr.party_id = pid;
            pr.votes = count;
            pr.share = static_cast<float>(count) / std::max(static_cast<std::uint64_t>(1), total_votes);
            pr.seats = seat_map.count(pid) ? seat_map[pid] : 0u;

            // Find party attributes for naming
            auto ps_it = party_states.find(pid);
            if (ps_it != party_states.end()) {
                const auto& ps = ps_it->second;
                pr.name = GeneratePartyName(ps.econ_ideology, ps.social_ideology,
                    ps.anti_est_posture, ps.urban_orientation,
                    ps.is_legacy != 0u, ps.is_legacy != 0u ? (pid == 0u ? 0u : 1u) : 0u);
                pr.color = IdeologyToColor(ps.econ_ideology, ps.social_ideology,
                    ps.anti_est_posture, ps.establishment_age,
                    ps.is_legacy != 0u, ps.is_legacy != 0u ? (pid == 0u ? 0u : 1u) : 0u);
                pr.econ_ideology = ps.econ_ideology;
            } else {
                pr.name = "Unknown";
                pr.color = "#666666";
            }
            results.push_back(pr);
            std::printf("  %s: %" PRIu64 " votes (%.1f%%) → %u seats\n",
                        pr.name.c_str(), pr.votes, pr.share * 100.0f, pr.seats);
        }

        WriteResultsArtifacts(results, total_seats, total_votes,
                               static_cast<std::uint64_t>(voter_count), div_method, thresh);
    }
}

FLAMEGPU_STEP_FUNCTION(PartyLifecycle) {
    const unsigned int step = FLAMEGPU->getStepCounter();
    const unsigned int org_steps = FLAMEGPU->environment.getProperty<unsigned int>("ORGANIZING_STEPS");
    const unsigned int total_steps = org_steps + FLAMEGPU->environment.getProperty<unsigned int>("CAMPAIGN_STEPS");
    if (step + 1u == total_steps) return;  // Don't kill parties on final step

    flamegpu::DeviceAgentVector parties = FLAMEGPU->agent("party").getPopulationData();
    for (unsigned int pi = 0u; pi < parties.size(); ++pi) {
        auto p = parties[pi];
        if (p.getVariable<unsigned int>("is_alive") == 0u) continue;
        float viab = p.getVariable<float>("projected_viability");
        unsigned int low_count = p.getVariable<unsigned int>("consecutive_low_viability");
        if (viab < 0.01f) {
            ++low_count;
            p.setVariable<unsigned int>("consecutive_low_viability", low_count);
            if (low_count >= 2u) {
                p.setVariable<unsigned int>("is_alive", 0u);
                std::printf("  Party %u died (viability collapsed)\n",
                            p.getVariable<unsigned int>("party_id"));
            }
        } else {
            p.setVariable<unsigned int>("consecutive_low_viability", 0u);
        }
    }
}

// ===== MODEL BUILDER =====

void BuildModel(flamegpu::ModelDescription& model) {
    auto env = model.Environment();

    // Environment properties
    env.newProperty<unsigned int>("VOTER_COUNT", ParseUnsignedEnv("WISCONSIN_PR_VOTERS", kDefaultVoterCount));
    env.newProperty<unsigned int>("ACTIVIST_COUNT", ParseUnsignedEnv("WISCONSIN_PR_ACTIVISTS", kDefaultActivistCount));
    
    unsigned int total_seats = ParseUnsignedEnv("WISCONSIN_PR_SEATS", kDefaultSeats);
    env.newProperty<unsigned int>("TOTAL_SEATS", total_seats);
    
    float natural_threshold = 1.0f / static_cast<float>(total_seats);
    env.newProperty<float>("THRESHOLD", ParseFloatEnv("WISCONSIN_PR_THRESHOLD", natural_threshold));
    env.newProperty<unsigned int>("RANDOM_SEED", ParseUnsignedEnv("WISCONSIN_PR_SEED", kDefaultRandomSeed));
    env.newProperty<unsigned int>("DIVISOR_METHOD", ParseDivisorMethodEnv("WISCONSIN_PR_DIVISOR", SAINTE_LAGUE));
    env.newProperty<unsigned int>("CAMPAIGN_STEPS", ParseUnsignedEnv("WISCONSIN_PR_CAMPAIGN_STEPS", kDefaultCampaignSteps));
    env.newProperty<unsigned int>("ORGANIZING_STEPS", ParseUnsignedEnv("WISCONSIN_PR_ORGANIZING_STEPS", kDefaultOrganizingSteps));
    env.newProperty<unsigned int>("PHASE", 0u);
    env.newProperty<float, kRegionPartySlots>("REGION_PARTY_CONTACT", ZeroContacts());

    // ---- Messages ----
    auto party_msg = model.newMessage<flamegpu::MessageBruteForce>("party_msg");
    party_msg.newVariable<unsigned int>("party_id");
    party_msg.newVariable<float>("econ_ideology");
    party_msg.newVariable<float>("social_ideology");
    party_msg.newVariable<float>("urban_orientation");
    party_msg.newVariable<float>("anti_est_posture");
    party_msg.newVariable<float>("religiosity_align");
    party_msg.newVariable<float>("union_alignment");
    party_msg.newVariable<float>("college_alignment");
    party_msg.newVariable<float>("establishment_age");
    party_msg.newVariable<float>("organization");
    party_msg.newVariable<float>("brand");
    party_msg.newVariable<float>("credibility");
    party_msg.newVariable<float>("projected_viability");
    party_msg.newVariable<float>("momentum");
    party_msg.newVariable<float>("media_reach");

    auto spatial_msg = model.newMessage<flamegpu::MessageSpatial2D>("spatial_voter_msg");
    spatial_msg.setMin(kSpatialMin, kSpatialMin);
    spatial_msg.setMax(kSpatialMax, kSpatialMax);
    spatial_msg.setRadius(kSpatialRadius);
    spatial_msg.newVariable<unsigned int>("previous_vote");
    spatial_msg.newVariable<float>("turnout_signal");

    // ---- Party agent ----
    auto party_agent = model.newAgent("party");
    party_agent.newVariable<unsigned int>("party_id");
    party_agent.newVariable<float>("econ_ideology");
    party_agent.newVariable<float>("social_ideology");
    party_agent.newVariable<float>("urban_orientation");
    party_agent.newVariable<float>("anti_est_posture");
    party_agent.newVariable<float>("religiosity_align");
    party_agent.newVariable<float>("union_alignment");
    party_agent.newVariable<float>("college_alignment");
    party_agent.newVariable<float>("establishment_age");
    party_agent.newVariable<float>("organization");
    party_agent.newVariable<float>("brand");
    party_agent.newVariable<float>("credibility");
    party_agent.newVariable<float>("projected_viability");
    party_agent.newVariable<float>("field_strength");
    party_agent.newVariable<float>("momentum");
    party_agent.newVariable<float>("fundraising");
    party_agent.newVariable<float>("media_reach");
    party_agent.newVariable<unsigned int>("is_alive");
    party_agent.newVariable<unsigned int>("is_legacy");
    party_agent.newVariable<unsigned int>("legacy_id");
    party_agent.newVariable<unsigned int>("consecutive_low_viability");
    auto fn_broadcast = party_agent.newFunction("PartyBroadcast", PartyBroadcast);
    fn_broadcast.setMessageOutput("party_msg");
    fn_broadcast.setFunctionCondition(PartyAliveCondition);

    // ---- Activist agent ----
    auto activist_agent = model.newAgent("activist");
    activist_agent.newVariable<float>("econ_ideology");
    activist_agent.newVariable<float>("social_ideology");
    activist_agent.newVariable<float>("anti_est");
    activist_agent.newVariable<unsigned int>("home_region");
    activist_agent.newVariable<float>("organizer_skill");
    activist_agent.newVariable<float>("donor_access");
    activist_agent.newVariable<float>("launch_tendency");
    activist_agent.newVariable<float>("field_reach");
    activist_agent.newVariable<float>("effort");
    activist_agent.newVariable<unsigned int>("affiliated_party");
    auto fn_canvass = activist_agent.newFunction("ActivistCanvass", ActivistCanvass);
    fn_canvass.setMessageInput("party_msg");

    // ---- Voter agent ----
    auto voter_agent = model.newAgent("voter");
    voter_agent.newVariable<float>("econ_ideology");
    voter_agent.newVariable<float>("social_ideology");
    voter_agent.newVariable<float>("conviction");
    voter_agent.newVariable<float>("turnout");
    voter_agent.newVariable<float>("sophistication");
    voter_agent.newVariable<float>("anti_est");
    voter_agent.newVariable<float>("urbanity");
    voter_agent.newVariable<unsigned int>("college");
    voter_agent.newVariable<unsigned int>("union_member");
    voter_agent.newVariable<float>("religiosity");
    voter_agent.newVariable<float>("minor_openness");
    voter_agent.newVariable<unsigned int>("region_id");
    voter_agent.newVariable<unsigned int>("vote_choice");
    voter_agent.newVariable<unsigned int>("previous_vote");
    voter_agent.newVariable<float>("party_loyalty");
    voter_agent.newVariable<unsigned int>("vote_streak");
    voter_agent.newVariable<float>("perceived_local_support");
    voter_agent.newVariable<float>("perceived_local_turnout");
    voter_agent.newVariable<float>("pos_x");
    voter_agent.newVariable<float>("pos_y");

    auto fn_post = voter_agent.newFunction("VoterPostSignal", VoterPostSignal);
    fn_post.setMessageOutput("spatial_voter_msg");
    fn_post.setFunctionCondition(CampaignPhaseCondition);

    auto fn_observe = voter_agent.newFunction("VoterObserveNeighbors", VoterObserveNeighbors);
    fn_observe.setMessageInput("spatial_voter_msg");
    fn_observe.setFunctionCondition(CampaignPhaseCondition);

    auto fn_choose = voter_agent.newFunction("VoterChoose", VoterChoose);
    fn_choose.setMessageInput("party_msg");
    fn_choose.setFunctionCondition(CampaignPhaseCondition);

    // ---- Layers ----
    model.newLayer("L1_PartyBroadcast").addAgentFunction(fn_broadcast);
    model.newLayer("L2_ActivistCanvass").addAgentFunction(fn_canvass);
    model.newLayer("L3_VoterPostSignal").addAgentFunction(fn_post);
    model.newLayer("L4_VoterObserveNeighbors").addAgentFunction(fn_observe);
    model.newLayer("L5_VoterChoose").addAgentFunction(fn_choose);

    // ---- Step functions ----
    model.addStepFunction(AdvanceCampaignAndAllocateSeats);
    model.addStepFunction(PartyLifecycle);

    // ---- Init functions ----
    model.addInitFunction(SeedVoters);
    model.addInitFunction(SeedActivists);
    model.addInitFunction(SeedLegacyParties);
    model.addInitFunction(DiscoverEmergentParties);
}

// ===== ENTRY POINT =====

void RunMain(int argc, const char** argv) {
    flamegpu::ModelDescription model(kModelName);
    BuildModel(model);
    flamegpu::CUDASimulation simulation(model);
    simulation.SimulationConfig().steps =
        ParseUnsignedEnv("WISCONSIN_PR_ORGANIZING_STEPS", kDefaultOrganizingSteps)
        + ParseUnsignedEnv("WISCONSIN_PR_CAMPAIGN_STEPS", kDefaultCampaignSteps);
    simulation.SimulationConfig().random_seed =
        ParseUnsignedEnv("WISCONSIN_PR_SEED", kDefaultRandomSeed);
    simulation.applyConfig();
    simulation.simulate();
}

}  // namespace wisconsin_pr

int main(int argc, const char** argv) {
    wisconsin_pr::RunMain(argc, argv);
    return 0;
}

