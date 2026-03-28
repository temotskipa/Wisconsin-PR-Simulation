# Wisconsin PR Simulation

FLAME-GPU 2 simulation of a multi-step statewide Wisconsin proportional-representation campaign with regional heterogeneity and three agent populations:

- `voter`: synthetic Wisconsin electorate with macro-region, ideology, turnout propensity, sophistication, anti-establishment sentiment, urbanity, education, union membership, religiosity, and party-family affinities.
- `activist`: ideological organizers with home regions, launch tendency, and field reach whose clustering determines which minor parties enter the ballot.
- `party`: ballot options that broadcast ideology, organization, brand, credibility, projected viability, campaign momentum, fundraising, and media reach.

## Model Flow

1. Host-side seeding builds a regionalized Wisconsin electorate and activist corps across eight macro-regions.
2. An init host function groups activists into party families, applies endogenous entry rules, and creates the ballot parties.
3. `party` agents broadcast statewide campaign signals through a brute-force message list each step.
4. `voter` agents score parties from individual ideology, loyalties, affinities, regional contact, strategic viability, and campaign quality signals, then make a sharp probabilistic choice or abstain.
5. A host step function updates party fundraising, media reach, momentum, viability, and regional field strength after each campaign round.
6. On the final step, the model tallies votes and allocates `99` seats with Sainte-Lague by default.

## Implementation Notes

- The model uses one canonical electoral threshold everywhere. The default is `5%`.
- Vote tallying is implemented with FLAME-GPU reductions instead of downloading millions of voters back to the CPU.
- Regional campaign contact strength is stored in an environment array and refreshed by host functions between rounds.
- Party entry is endogenous: activist clustering, organizer skill, donor access, and regional breadth determine which minor parties actually reach the ballot.
- Final results are exported as both a browser-friendly HTML summary and an SVG semicircular parliament chart.

## Defaults

- Voters: `5,970,000`
- Activists: `2,500`
- Seats: `99`
- Threshold: `5%`
- Divisor method: `sainte_lague`
- Simulation steps: `6`
- Random seed: `42`

## Build

Requirements are the standard FLAME-GPU 2 native C++ requirements on Windows:

- CMake `>= 3.25.2`
- Visual Studio 2026 with C++ and CMake support installed
- CUDA toolkit compatible with your Visual Studio host compiler
- NVIDIA GPU supported by the selected CUDA version
- Git, unless you provide a local `FLAMEGPU_ROOT`

Open `Developer PowerShell for VS 2026`, change to the repository root, and run:

```powershell
cmake -S . -B out/build/ninja-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe"
cmake --build out/build/ninja-debug --target flamegpu
cmake --build out/build/ninja-debug --target wisconsin_pr_simulation
```

Visual Studio's integrated CMake flow can also generate `out/build/x64-Debug`, and the IDE build path has also been verified with `Build All`.

The verified executable is emitted under:

```text
out/build/ninja-debug/bin/Debug/wisconsin_pr_simulation.exe
```

## Run

From PowerShell, run the executable directly:

```powershell
.\out\build\ninja-debug\bin\Debug\wisconsin_pr_simulation.exe
```

For a smaller debug run, set the session environment overrides first:

```powershell
$env:WISCONSIN_PR_VOTERS = "250000"
$env:WISCONSIN_PR_ACTIVISTS = "1200"
$env:WISCONSIN_PR_STEPS = "6"
.\out\build\ninja-debug\bin\Debug\wisconsin_pr_simulation.exe
```

To switch the seat-allocation method to D'Hondt:

```powershell
$env:WISCONSIN_PR_DIVISOR_METHOD = "dhondt"
.\out\build\ninja-debug\bin\Debug\wisconsin_pr_simulation.exe
```

## Environment Overrides

These environment variables override the built-in defaults:

- `WISCONSIN_PR_VOTERS`
- `WISCONSIN_PR_ACTIVISTS`
- `WISCONSIN_PR_SEATS`
- `WISCONSIN_PR_RANDOM_SEED`
- `WISCONSIN_PR_THRESHOLD`
- `WISCONSIN_PR_MINOR_ENTRY_SHARE`
- `WISCONSIN_PR_DIVISOR_METHOD`
- `WISCONSIN_PR_STEPS`
- `WISCONSIN_PR_REPORT_DIR`

Supported divisor-method values:

- `sainte_lague`
- `dhondt`

## Output

The executable prints:

- campaign checkpoint lines for each pre-final step
- electorate size
- turnout and abstentions
- threshold and divisor method
- per-party vote totals
- per-party vote shares
- seat allocation

It also writes these report artifacts to `WISCONSIN_PR_REPORT_DIR` or `reports/` by default:

- `wisconsin_pr_results.html`
- `wisconsin_pr_results.svg`

## Current Scope

This is still a synthetic statewide prototype, not a calibrated Wisconsin electoral forecast. It now includes macro-regional structure, endogenous party entry, simplified fundraising/media feedback, and campaign time dynamics, but it does not yet include:

- county-level or precinct-level geography
- district maps or mixed-member systems
- real demographic microdata or voter files
- candidate-specific events, scandals, or ad buys
- coalition formation after seat allocation

It is, however, a scalable FLAME-GPU base for pushing toward those additions.
