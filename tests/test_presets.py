"""Self-contained regression test of the RADI web engine (CPython; needs numpy and scipy).

1. Every station preset, started from its stored steady state, must stay there and reproduce the
   reference benthic fluxes of the Julia RADI.jl engine (stored in presets.json) to 1e-4.
2. Two stations solved from scratch (scheduled march + Newton) must land on the same fluxes.
3. The species builder must close the budgets of user species: a decaying solid tracer and a solid
   additive leaching into the porewater.
usage: python3 tests/test_presets.py
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import radi_core as rc

P = json.load(open(os.path.join(ROOT, "presets.json")))
fails = 0


def check(name, ok, detail):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + f"{name:44s} {detail}", flush=True)


def flux_errors(o, ref):
    got = {"FTA": o["fluxes"]["TA"], "FDIC": o["fluxes"]["DIC"], "FO2": -o["fluxes"]["O2"]}
    return max(abs(got[k] - ref[k]) / abs(ref[k]) for k in got)


for st in P["stations"]:
    site = rc.Site(st["config"])
    t = time.time()
    u, info = rc.solve(site, np.array(st["seed"]).T.reshape(-1))
    o = rc.outputs(site, u, info)
    e = flux_errors(o, st["reference"])
    check(f"preset {st['id']}", info["converged"] and e < 1e-4, f"max flux error {e:.1e}, {time.time() - t:.2f} s")

for sid in ("W2", "MontereyBay"):
    st = next(s for s in P["stations"] if s["id"] == sid)
    site = rc.Site(st["config"])
    t = time.time()
    u, info = rc.solve(site, None)
    o = rc.outputs(site, u, info)
    e = flux_errors(o, st["reference"])
    check(f"from scratch {sid}", info["converged"] and e < 2e-3, f"max flux error {e:.1e}, {time.time() - t:.2f} s")

st = next(s for s in P["stations"] if s["id"] == "W2")
lam = np.log(2) / 22.3
ext = {"species": [{"id": "Pb", "phase": "solid", "rain": 1.0},
                   {"id": "A", "phase": "solid", "rain": 1e-3}, {"id": "B", "phase": "solute", "D0": 0.015, "DT": 0.0007, "bw": 0.0}],
       "reactions": [{"name": "decay", "type": "first_order", "k": lam, "reactant": "Pb", "stoich": {"Pb": -1}},
                     {"name": "leach", "type": "first_order", "k": 0.05, "reactant": "A", "stoich": {"A": -1, "B": 1}},
                     {"name": "B decay", "type": "first_order", "k": 2.0, "reactant": "B", "stoich": {"B": -1}}]}
site = rc.Site(st["config"], ext)
full = rc.cold_start(site).reshape(site.Nz, site.nvar).T
full[:18] = np.array(st["seed"])
u, info = rc.solve(site, full.T.reshape(-1))
o = rc.outputs(site, u, info)
U = u.reshape(site.Nz, site.nvar).T
vs, vd = site.phiS * site.dx, site.phi * site.dx
iPb, iA, iB = site.nvar - 3, site.nvar - 2, site.nvar - 1
c_pb = (1.0 - lam * np.sum(U[iPb] * vs) - site.w[-1] * site.phiS[-1] * U[iPb, -1]) / 1.0
leach = 0.05 * np.sum(U[iA] * vs)
c_a = (1e-3 - leach - site.w[-1] * site.phiS[-1] * U[iA, -1]) / 1e-3
c_b = (leach - 2.0 * np.sum(U[iB] * vd) - o["fluxes"]["B"] - site.phi[-1] * site.u_bur[-1] * U[iB, -1]) / leach
check("builder budgets (tracer decay, leaching)", info["converged"] and max(abs(c_pb), abs(c_a), abs(c_b)) < 1e-4,
      f"closures {c_pb:.1e}, {c_a:.1e}, {c_b:.1e}")
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
