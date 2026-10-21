# RADI web model

RADI (Reactive–Advective–Diffusive–Irrigative) is a one-dimensional model of early diagenesis in marine
sediments. This folder is a static web page that solves a RADI sediment column **directly to steady
state in the browser**: porewater and solid profiles, reaction rates, benthic fluxes and column budgets,
for ten ready-made stations or any site the user describes, with optional user-defined species and
reactions.

The engine is a Python port (numpy + scipy) of the Julia RADI steady-state engine, run with
[Pyodide](https://pyodide.org) inside a Web Worker. There is no server-side code and no build step.

Version **2026-10-21**.

## Files

| File | What it is |
|---|---|
| `index.html` | The page: station presets, inputs, the species and reaction builder, results (profiles, reaction rates, fluxes and budgets, solver report, downloads) and the model description ("The model"). Plots use Plotly 2.27.0. |
| `worker.js` | Web Worker: loads Pyodide 0.27.0 with numpy and scipy, fetches the two Python files, runs `radi_core.run()` and posts progress and results back to the page. |
| `radi_core.py` | The engine: site set-up, reaction network, porewater pH, transport, steady-state solver, outputs, user species and reactions. |
| `carbconst.py` | Seawater density and the carbonate-system constants (CO2SYS set, pressure-corrected), porewater pH and saturation states. |
| `presets.json` | The ten stations: configuration, converged steady state (`seed`) and reference fluxes from the Julia engine. |
| `tests/test_presets.py` | Regression test, run in ordinary Python (CPython). |

## Running it locally

```
cd site
python3 -m http.server 8000
```

then open <http://localhost:8000>. The page must be served over HTTP, not opened as a file, because the
worker fetches the Python files. It needs internet access for Pyodide (cdn.jsdelivr.net), Plotly
(cdn.plot.ly) and the Inter font (Google Fonts).

The first visit downloads about 22 MB, almost all of it Pyodide with numpy and scipy, and the engine is
ready after 6–10 s; the browser caches these files for later visits.

## Publishing

Copy the files of this folder, unchanged, to any static host (GitHub Pages, an institutional web
server). Nothing needs compiling.

When `radi_core.py`, `carbconst.py` or `presets.json` change, bump `APP_VERSION` in `index.html` and
`VERSION` in `radi_core.py` to the same new value. The page and the worker add `?v=<version>` to their
requests, so returning visitors then fetch the new files instead of cached copies. Inputs that visitors
saved in their browser are restored only when the version matches, so a version bump also resets them.

## Using the engine without the page

The engine runs unchanged in ordinary Python with numpy and scipy (tested with Python 3.13, numpy 2.5,
scipy 1.18, and in Pyodide 0.27.0, which runs Python 3.12):

```python
import json, radi_core

station = json.load(open("presets.json"))["stations"][0]          # W2
out = json.loads(radi_core.run({"config": station["config"], "seed": station["seed"]}))
print(out["solver"]["converged"], out["fluxes"]["TA"])           # True 0.43070946...
```

Request keys:

- `config`: the site, in the keys of `presets.json` (`WATER_DEPTH`, `TEMPERATURE`, `POC_RAIN`, ...);
  missing keys take the values of `radi_core.DEFAULTS`.
- `extensions` (optional): user species and reactions, see below.
- `seed` (optional): a converged state to start from, as rows of species × columns of layers (mol m⁻³);
  the 18 built-in rows are enough, user species are then started from scratch. Without a seed the solver
  starts from scratch.
- `tol` (optional, default 0.02): the largest accepted |d(state)/dt| of the built-in species, in
  mol m⁻³ yr⁻¹.

The result holds `species`, `profiles` (solutes in µmol kg⁻¹, solids in mol m⁻³ of solid and dry
weight %, pH on the free scale, Ω, depth in cm), `rates`, `fluxes` (mol m⁻² yr⁻¹, positive out of the
sediment), `budget` (calcite, aragonite and organic-carbon budgets, organic carbon by pathway, closure
errors), `summary`, `solver` (convergence report) and `state` (the final state, to seed the next run).

## User species and reactions

On top of the 18 built-in species, users can add any solute or solid. A **solute** needs its
free-solution diffusion coefficient D = D0 + DT·T (m² yr⁻¹, T in °C) and its bottom-water
concentration (µmol kg⁻¹); a **solid** needs its rain to the seafloor (mol m⁻² yr⁻¹) and, if it is not a
trace, its molar mass (g mol⁻¹), which adds to the sediment accumulation. Reactions take one of five
rate laws, with stoichiometric coefficients for any built-in or user species:

| Type | Rate R | Rate per m³ of |
|---|---|---|
| `first_order` | k [A] | the phase of A |
| `mass_action` | k [A][B] (or [A][B][C]) | the phase of A |
| `monod` | k [A] · Π [L]/(K + [L]) · Π K/(K + [I]) | the phase of A |
| `zero_order` | k · Π Monod and inhibition terms (production only) | the phase given |
| `per_Corg` | k × organic-carbon degradation rate of a pathway (all, aerobic, denitrification, Mn, Fe, sulfate, methanogenesis) | solid |

The species tendency is `stoich × R`, converted between porewater and solid volumes where needed.
User species are accepted as converged when their column budget closes to 10⁻⁶ of their throughput.
The page offers four worked examples (conservative tracer, ²¹⁰Pb excess, barium released with organic
carbon, leaching of a solid additive). The JSON format of `extensions` is documented in the
`_Extensions` class of `radi_core.py`.

## Validation

`python3 tests/test_presets.py` (CPython with numpy and scipy) checks that:

- each of the ten stations, solved from its stored state, reproduces the Julia reference fluxes of
  alkalinity, DIC and O₂ to better than 10⁻⁸ relative, in 0.03–0.1 s;
- W2 and Monterey Bay solved from scratch converge to the same fluxes (within 10⁻⁸) in about 1.5 and
  3 s;
- the builder examples (decaying tracer, solid-to-solute leaching) close their column budgets to about
  10⁻¹².

During development the engine was also checked against the Julia engine term by term: constants to
10⁻¹³, the residual of the equations to round-off, identical Newton iteration counts at all ten
stations, and benthic fluxes within about 0.1 % in thirty perturbed cases (organic-carbon rain × 2,
bottom-water DIC + 100 µmol kg⁻¹, calcite rain × 0.5).

In the browser (headless Chromium, Pyodide 0.27.0): engine ready in 6–8 s; a station preset solves in
0.05–0.2 s; an edit, warm-started from the last solution, in 1–3 s; a fresh start or a new grid in 2–3 s.
No page errors in an end-to-end run over all stations, the builder examples, the comparison of kept
runs, dark mode and a phone-width layout.

## How the presets were made

Each station's inputs come from the RADI.jl station configuration files; the short descriptions and
literature tags shown on the page come from the same files. Each steady state was computed with the
Julia engine (RADI.jl, conservative flux-form transport of September 2026) under the same calcite law,
converged to max |d(state)/dt| ≤ 10⁻⁷ mol m⁻³ yr⁻¹, rounded to ten significant digits and stored as
`seed`; `reference` holds the Julia fluxes. When a station's inputs are unchanged, the page shows its
result next to these reference values.

To add a station, append an object with `id`, `name`, `short` (the label on its card), `description`,
`config` (all keys of `radi_core.DEFAULTS`), and optionally `seed` (18 × N_LAYERS, mol m⁻³) and
`reference` (`FTA`, `FDIC`, `FO2`, in mol m⁻² yr⁻¹). Without a seed, the station is solved from scratch.

## Calcite law

Calcite dissolution follows the two-regime law of Naviaux et al. (2019) in smoothed form: near-equilibrium
exponent 2 instead of 0.11, the two regimes blended by a logistic function of width 0.02 centred on
Ω = 0.8275 (where the two published branches meet), and both rate constants multiplied by 25. The
published law, with its (1 − Ω)^0.11 branch, cannot be solved directly to steady state where calcite
persists near saturation (tested at all ten stations). Precipitation follows Zuddas and Mucci (1998),
aragonite dissolution Dong et al. (2019).

## Limits

- Steady state only.
- Aragonite keeps its published near-equilibrium exponent (0.13), so a large aragonite rain to a
  sediment where aragonite survives near saturation may not converge.
- The two Iberian Margin stations carry the water depth of their original configuration file (4380 m),
  a placeholder; set the true depth before using them.
- No iron-sulfide chemistry, adsorption or silica cycle in the built-in network; users can add what
  they need as their own species and reactions.

## Citing

Please cite:

- Sulpis, O., Humphreys, M. P., Wilhelmus, M. M., Carroll, D., Berelson, W. M., Menemenlis, D.,
  Middelburg, J. J., and Adkins, J. F. (2022). RADIv1: a non-steady-state early diagenetic model for
  ocean sediments in Julia and MATLAB/GNU Octave. *Geoscientific Model Development* 15, 2105–2131.
  <https://doi.org/10.5194/gmd-15-2105-2022>
- van der Zant, H. F., Sulpis, O., Middelburg, J. J., Humphreys, M. P., Savelli, R., Carroll, D.,
  Menemenlis, D., Sušelj, K., and Le Fouest, V. (2026). RADIv2: an adaptable and versatile diagenetic
  model for coastal and open-ocean sediments. *Geoscientific Model Development* 19, 1965–1989.
  <https://doi.org/10.5194/gmd-19-1965-2026>

and give the version of this web model (2026-10-21). The full reference list is under "The model" on
the page.

## Version history

- **2026-10-21**: rebuilt. New engine (Python port of the RADI.jl steady-state solver with
  conservative flux-form transport, validated against Julia), smoothed two-regime calcite law, user
  species and reactions, ten station presets with converged states, new interface (comparison of
  kept runs, depth zoom, reaction-rate and budget views, CSV and JSON downloads, dark mode, phone
  layout). Replaces the earlier web version.
