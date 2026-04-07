# Wisconsin PR Simulation

FLAME-GPU 2 simulation of a multi-step statewide Wisconsin proportional-representation campaign built on the framework of **Methodological Individualism**. All macroeconomic outcomes (election results, party emergence, coalitions) emerge entirely from the independent logic and actions of individuals, without top-down macro-level enforcement.

It features regional heterogeneity and three agent populations:

- `voter`: synthetic Wisconsin electorate with a 2D ideology (`econ_ideology`, `social_ideology`), conviction depth, macro-region, turnout propensity, sophistication, urbanity, education, union membership, and religiosity. Modeled on 2024 census and polling profiles.
- `activist`: ideological organizers positioned in the 2D ideology space. Their density and clustering directly determine which political parties dynamically emerge and enter the ballot.
- `party`: dynamically generated ballot options (procedurally named with procedural colors derived from HSL ideology mappings) that broadcast continuous attributes like organization, brand, credibility, momentum, and field strength.

## Model Flow

1. Host-side seeding builds a regionalized Wisconsin electorate and activist corps across eight macro-regions. Legacy parties (Democratic, Republican) are seeded with high establishment age.
2. A density-based scan over the 2D ideology grid identifies clusters of activists. If enough activists share similar economic and social views, a new dynamic minor party emerges on the ballot.
3. **Organizing Phase**: Newly formed parties gather initial momentum, fundraising, and regional field strength based on the skill and reach of their clustered activists.
4. **Campaign Phase**: `party` agents broadcast statewide campaign signals. `voter` agents observe spatial signals (word-of-mouth), regional field contacts, and party platform distance in 2D space.
5. Voters score parties continuously based on ideology fit, urban/rural resonance, momentum, and campaign contact strength, then make a sharp probabilistic choice or abstain.
6. On the final step, the host tallies votes and allocates `99` seats using Sainte-Laguë or D'Hondt.

## Implementation Notes

- **Dynamic Emergence**: No hardcoded `PartyId` enums exist. All party interaction is generic math scaling across 2D continuous space.
- **Seat Allocation**: Tallying happens via FLAME-GPU reductions on the GPU.
- **Report Generation**: Emits an HTML summary containing the election outcome and a **Governance & Coalition** analysis predicting which parties can form a governing majority.
- **Visuals**: Emits a professional, Wikipedia-style SVG parliament hemicycle chart using a proportional row-distribution algorithm for radial party boundaries.

## Defaults

- Voters: `5,970,000`
- Activists: `2,500`
- Seats: `99`
- Threshold: `Natural Threshold (1 / Seats)`
- Divisor method: `sainte_lague`
- Campaign steps: `6`
- Organizing steps: `3`
- Random seed: `42`

## Build

Requirements are the standard FLAME-GPU 2 native C++ requirements on Windows:

- CMake `>= 3.25.2`
- Visual Studio 2026 with C++ and CMake support installed
- CUDA toolkit compatible with your Visual Studio host compiler (e.g., CUDA 13.2)
- NVIDIA GPU supported by the selected CUDA version

Open `Developer PowerShell for VS 2026`, change to the repository root, and run:

```powershell
cmake -S . -B out/build/ninja-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe"
cmake --build out/build/ninja-debug --target wisconsin_pr_simulation
```

Visual Studio's integrated CMake flow can also generate `out/build/x64-Debug`, which has been verified with `Build All`.

## Run

From PowerShell, run the executable directly:

```powershell
.\out\build\x64-Debug\bin\Debug\wisconsin_pr_simulation.exe
```

For a smaller debug run, set the session environment overrides first:

```powershell
$env:WISCONSIN_PR_VOTERS = "50000"
$env:WISCONSIN_PR_ACTIVISTS = "3000"
$env:WISCONSIN_PR_CAMPAIGN_STEPS = "6"
$env:WISCONSIN_PR_ORGANIZING_STEPS = "3"
.\out\build\x64-Debug\bin\Debug\wisconsin_pr_simulation.exe
```

To switch the seat-allocation method to D'Hondt:

```powershell
$env:WISCONSIN_PR_DIVISOR_METHOD = "dhondt"
.\out\build\x64-Debug\bin\Debug\wisconsin_pr_simulation.exe
```

## Environment Overrides

These environment variables override the built-in defaults:

- `WISCONSIN_PR_VOTERS`
- `WISCONSIN_PR_ACTIVISTS`
- `WISCONSIN_PR_SEATS`
- `WISCONSIN_PR_RANDOM_SEED`
- `WISCONSIN_PR_THRESHOLD`
- `WISCONSIN_PR_DIVISOR_METHOD`
- `WISCONSIN_PR_CAMPAIGN_STEPS`
- `WISCONSIN_PR_ORGANIZING_STEPS`
- `WISCONSIN_PR_REPORT_DIR`

Supported divisor-method values:

- `sainte_lague`
- `dhondt`

## Output

The executable outputs to stdout:

- Organizing phase active party counts
- Campaign phase turnout and vote shares
- Final seat allocation

It writes two report artifacts to `WISCONSIN_PR_REPORT_DIR` (default: `reports/`):

- `wisconsin_pr_results.html` (Data table and Coalition possibilities)
- `wisconsin_pr_results.svg` (Professional hemicycle visualization)

## Current Scope

This simulation implements a robust agent-based macro-region framework showing dynamic party emergence in a PR system. Future additions could include:

- County-level or district-level mapping
- Real individual-voter microdata injection
- Dynamic candidate events and scandal modeling
- Temporal multi-election simulation retaining party state across cycles
