"""
radi_core.py -- the RADI steady-state engine of the RADI web model (Python 3 + numpy + scipy;
runs unchanged in Pyodide).

RADI = Reactive-Advective-Diffusive-Irrigative early-diagenesis model
(Sulpis et al., 2022, Geosci. Model Dev. 15, 2105-2131; van der Zant et al., 2026, Geosci. Model Dev. 19,
1965-1989).

This file is a line-by-line port of the Julia steady-state engine of RADI.jl (RADI_model.jl with the
conservative flux-form transport of September 2026), driven the way the RADI.jl station set-up
(RADI_setup.jl and its station configuration files) drives it, with the smoothed two-regime calcite dissolution law
(Naviaux et al. 2019 with near-equilibrium exponent 2, a logistic blend of width 0.02 at Omega = 0.8275
and rate constants x 25). It reproduces the Julia residual to round-off (see tests/test_presets.py).

On top of the fixed 18-species RADI core, users may add their own solutes and solids and their own
reactions (`extensions`, see `_Extensions`).

Units inside the engine: depth m, time yr, solutes mol m-3 of porewater, solids mol m-3 of solid,
fluxes mol m-2 yr-1. Inputs and outputs use friendly units (umol/kg, cm, mol m-2 yr-1).
"""
import math
import json
import time
import numpy as np
from scipy.linalg import solve_banded

import carbconst

VERSION = "2026-10-21"
SPY = 60 * 60 * 24 * 365.25          # seconds per year

# ---------------------------------------------------------------------------------------------------
# 1. Core species (state order is fixed: it is the Julia engine's)
# ---------------------------------------------------------------------------------------------------
CORE = [
    # id,          label,                                  phase
    ("O2",        "Dissolved oxygen",                       "solute"),
    ("DIC",       "Dissolved inorganic carbon",             "solute"),
    ("NO3",       "Nitrate",                                "solute"),
    ("SO4",       "Sulfate",                                "solute"),
    ("PO4",       "Phosphate",                              "solute"),
    ("NH4",       "Ammonium",                               "solute"),
    ("H2S",       "Hydrogen sulfide (total)",               "solute"),
    ("FeII",      "Dissolved iron(II)",                     "solute"),
    ("MnII",      "Dissolved manganese(II)",                "solute"),
    ("CH4",       "Methane",                                "solute"),
    ("TA",        "Total alkalinity",                       "solute"),
    ("Ca",        "Dissolved calcium",                      "solute"),
    ("POC_fast",  "Fast-degrading particulate organic C",   "solid"),
    ("POC_slow",  "Slow-degrading particulate organic C",   "solid"),
    ("FeOH3",     "Iron(III) hydroxide Fe(OH)3",            "solid"),
    ("MnO2",      "Manganese oxide MnO2",                   "solid"),
    ("calcite",   "Calcite CaCO3",                          "solid"),
    ("aragonite", "Aragonite CaCO3",                        "solid"),
]
NCORE = len(CORE)
IDX = {s[0]: i for i, s in enumerate(CORE)}

# Free-solution diffusion coefficients D = a + b*T (m2/yr, T in degC); Params.jl
D_COEF = {
    "O2": (0.031558, 0.001428), "DIC": (0.015179, 0.000795), "NO3": (0.030863, 0.001153),
    "SO4": (0.015779, 0.000712), "PO4": (0.009783, 0.000513), "NH4": (0.030926, 0.001225),
    "H2S": (0.028938, 0.001314), "FeII": (0.010761, 0.000466), "MnII": (0.009625, 0.000481),
    "CH4": (0.041628576888000035, 0.00076369), "TA": (0.015179, 0.000795), "Ca": (0.011771, 0.000529),
}

# Molar masses (g/mol) used in the solid mass flux that sets the burial velocity (calculate_constants)
M_MnO2, M_FeOH3, M_CaCO3, M_CLAY = 86.9368, 106.867, 100.0869, 360.31

# The calcite law (smoothed two-regime Naviaux) and the other fixed reactive constants (React.jl)
LAW = dict(RATELAW="snaviaux", N1EXP=2.0, SBLEND=0.02, SCROSS=0.8275, KCAL_MULT=25.0, KCAL1_MULT=1.0)
KM_dO2, KMi_dO2 = 0.003, 0.01
KM_dtNO3, KMi_dtNO3 = 0.01, 0.005      # RADI_model.jl sets React.KM_dtNO3 = 0.01
KM_pMnO2, KMi_pMnO2 = 42.4, 42.4
KM_pFeOH3, KMi_pFeOH3 = 265.0, 265.0
KM_dtSO4, KMi_dtSO4 = 1.6, 1.6
k_MnII, k_FeII, k_NH3, k_H2S, k_CH4_O2, k_CH4_SO4 = 1e6, 1e6, 1e4, 3e5, 1e7, 1e4

HMIN, HMAX = 1e-12, 1.0                # clamp of [H+] in mol m-3 (RADI_model.jl)
RTOL_USER = 1e-6                       # user species: residual relative to their own gross rates

# The station presets' configuration keys (RADI.jl station configuration files), with defaults
DEFAULTS = dict(
    RUN_NAME="custom", WATER_DEPTH=4310.0, TEMPERATURE=1.4, SALINITY=34.69, CURRENT_SPEED=0.02,
    DBL_THICKNESS=1000.0,
    O2_BW=159.7, DIC_BW=2324.0, TALK_BW=2426.0, NO3_BW=36.93, PO4_BW=2.39, NH4_BW=1.0, H2S_BW=0.0,
    SILICATE_BW=120.0, CH4_BW=0.0, FeII_BW=0.0005, MnII_BW=0.0005, CA_BW="from_salinity",
    POC_RAIN=0.203423, CALCITE_RAIN=0.22, ARAGONITE_RAIN=0.0, CLAY_RAIN=2.0 / 360.31,
    MnO2_RAIN=0.0005, FeOH3_RAIN=0.0005,
    POC_FAST_FRACTION=0.7, POC_SLOW_FRACTION=0.27, K_FAST=18.0, K_SLOW=0.006,
    SURFACE_POROSITY=0.85, DEEP_POROSITY=0.74, POROSITY_DECAY=33.0, PERMEABILITY=6.6e-12,
    IRRIGATION_DEPTH=0.05, BIOTURBATION_DEPTH=0.08,
    N_LAYERS=51, SED_DEPTH=0.8, DZ_TOP=1.0e-3, DZ_BOTTOM=5.0e-2,
)
RHO_SOLID = 2.65e6                     # g m-3 (rho_p_baseline)


# ---------------------------------------------------------------------------------------------------
# 2. Site constants (calculate_constants + RADI_setup.jl)
# ---------------------------------------------------------------------------------------------------
def grid_geometric(depth, dz_top, dz_bot, Nz, r_max=1.10):
    """Params.prepdepth_geometric: geometric spacing from dz_top to ~dz_bot, scaled to the depth."""
    r = (dz_bot / dz_top) ** (1.0 / (Nz - 1))
    r = max(1.0, min(r, r_max))
    dx_raw = dz_top * r ** np.arange(Nz, dtype=float)
    s = depth / np.sum(dx_raw)
    dx = s * dx_raw
    ze = np.concatenate(([0.0], np.cumsum(dx)))
    zc = 0.5 * (ze[:-1] + ze[1:])
    return zc, dx, ze, r


def fv_sigma(Pe):
    """Fiadeiro-Veronis weighting sigma(Pe) = coth(Pe) - 1/Pe (small-Pe series below 1e-4)."""
    Pe = np.asarray(Pe, dtype=float)
    out = Pe / 3.0
    big = np.abs(Pe) >= 1e-4
    pb = Pe[big]
    out[big] = 1.0 / np.tanh(pb) - 1.0 / pb
    return out


def _dbl_species(D_myr, T, U):
    """DBL thickness (m) from the near-bed current (Boudreau & Jorgensen 2001), RADI_model.jl DBL()."""
    u_star = 0.00136 - 2.19598542e-5 * T + 2.35862843e-2 * U
    nu = 1.747567451381780806e-6 - 3.23886387e-8 * T
    D = D_myr / (60 * 60 * 24 * 365.25)
    Sc = nu / D
    beta = 0.0417 * u_star * Sc ** (-2 / 3)
    return D / beta


def _ksed(D_myr, T, U, permeability):
    """Current/permeability-enhanced apparent diffusivity (McGinnis et al. 2014), RADI_model.jl Ksed()."""
    u_star = 0.00136 - 2.19598542e-5 * T + 2.35862843e-2 * U
    alpha = 1.0 if permeability >= 1e-12 else 0.0
    D = D_myr / (60 * 60 * 24 * 365.25)
    return D * math.exp(alpha * 0.4 * (u_star * 1000)) * 60 * 60 * 24 * 365.25


def _num(cfg, key):
    v = cfg.get(key, DEFAULTS.get(key))
    return float(v)


class Site:
    """Everything that is constant during a solve: grid, porosity, transport coefficients, chemistry
    constants, boundary conditions, the species list (core + user) and the linear transport operator."""

    def __init__(self, config, extensions=None):
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in (config or {}).items() if v is not None})
        self.cfg = cfg
        T = _num(cfg, "TEMPERATURE"); S = _num(cfg, "SALINITY"); P = _num(cfg, "WATER_DEPTH")
        U = _num(cfg, "CURRENT_SPEED")
        self.T, self.S, self.P, self.U = T, S, P, U

        # ---- bottom water (RADI_setup.jl: umol/kg -> mol/kg) ----
        bw = {}
        for sp, key in (("O2", "O2_BW"), ("DIC", "DIC_BW"), ("TA", "TALK_BW"), ("NO3", "NO3_BW"),
                        ("PO4", "PO4_BW"), ("NH4", "NH4_BW"), ("H2S", "H2S_BW"), ("CH4", "CH4_BW"),
                        ("FeII", "FeII_BW"), ("MnII", "MnII_BW")):
            bw[sp] = _num(cfg, key) * 1e-6
        ca = cfg.get("CA_BW", "from_salinity")
        bw["Ca"] = (0.02128 / 40.087 * S / 1.80655) if (ca is None or ca == "from_salinity") else float(ca) * 1e-6
        bw["SO4"] = 29264.2e-6 * S / 35
        dSi_w = _num(cfg, "SILICATE_BW") * 1e-6
        self.bw_kg = bw                       # mol/kg

        # ---- grid and porosity ----
        Nz = int(round(_num(cfg, "N_LAYERS")))
        zc, dx, ze, r = grid_geometric(_num(cfg, "SED_DEPTH"), _num(cfg, "DZ_TOP"), _num(cfg, "DZ_BOTTOM"), Nz)
        phi0, phiInf, beta = _num(cfg, "SURFACE_POROSITY"), _num(cfg, "DEEP_POROSITY"), _num(cfg, "POROSITY_DECAY")
        phi = (phi0 - phiInf) * np.exp(-beta * zc) + phiInf
        phiS = 1.0 - phi
        self.Nz, self.zc, self.dx, self.ze, self.r = Nz, zc, dx, ze, r
        self.phi, self.phiS, self.phiS_phi = phi, phiS, phiS / phi
        self.tort2 = 1.0 - 2.0 * np.log(phi)

        # ---- seawater ----
        rho = carbconst.seawater_density(S, T, P)
        self.rho = rho

        # ---- organic matter and solid fluxes ----
        RC, RN, RP = 1.0, 16.0 / 106.0, 1.0 / 106.0
        Mpom = RC * 30.031 + RN * 17.031 + RP * 97.994
        self.RC, self.RN, self.RP, self.Mpom = RC, RN, RP, Mpom
        Fpom = (_num(cfg, "POC_RAIN") / RC) * Mpom          # g m-2 yr-1 (RADI_setup.jl Fpom_setup)
        Fpom_mol = Fpom / Mpom
        Fpoc = Fpom_mol * RC
        ffast, fslow = _num(cfg, "POC_FAST_FRACTION"), _num(cfg, "POC_SLOW_FRACTION")
        self.Fpoc = Fpoc
        self.Ffoc, self.Fsoc = Fpoc * ffast, Fpoc * fslow
        self.Froc = Fpoc * (1.0 - ffast - fslow)
        F_cal, F_ara = _num(cfg, "CALCITE_RAIN"), _num(cfg, "ARAGONITE_RAIN")
        F_MnO2, F_FeOH3, F_clay = _num(cfg, "MnO2_RAIN"), _num(cfg, "FeOH3_RAIN"), _num(cfg, "CLAY_RAIN")
        self.rain = {"POC_fast": self.Ffoc, "POC_slow": self.Fsoc, "FeOH3": F_FeOH3, "MnO2": F_MnO2,
                     "calcite": F_cal, "aragonite": F_ara}
        Fp = Fpom + F_MnO2 * M_MnO2 + F_FeOH3 * M_FeOH3 + (F_cal + F_ara) * M_CaCO3 + F_clay * M_CLAY

        # ---- user species (they add their solid mass flux to the burial velocity) ----
        self.ext = _Extensions(extensions, self)
        Fp = Fp + self.ext.extra_solid_mass_flux()
        self.Fp = Fp

        # ---- bioturbation, burial, irrigation ----
        O2w_m3 = bw["O2"] * rho
        lam_b = _num(cfg, "BIOTURBATION_DEPTH")
        D_bio_0 = 0.0232e-4 * (1e2 * Fpoc) ** 0.85
        self.D_bio = D_bio_0 * np.exp(-(zc / lam_b) ** 2) * O2w_m3 / (O2w_m3 + 0.02)
        x0 = Fp / (RHO_SOLID * phiS[0])
        xinf = x0 * phiS[0] / phiS[-1]
        self.u_bur = xinf * phi[-1] / phi
        self.w = xinf * phiS[-1] / phiS
        alpha_0 = (11.0 * (math.atan((1e2 * Fpoc * 5.0 - 400.0) / 400.0) / math.pi + 0.5) - 0.9 +
                   20.0 * (O2w_m3 / (O2w_m3 + 0.01)) * math.exp(-O2w_m3 / 0.01) * 1e2 * Fpoc / (1e2 * Fpoc + 30.0))
        self.alpha = alpha_0 * np.exp(-(zc / _num(cfg, "IRRIGATION_DEPTH")) ** 2)
        self.alpha_0 = alpha_0

        # ---- degradation constants ----
        self.kfast = np.full(Nz, _num(cfg, "K_FAST"))
        self.kslow = np.full(Nz, _num(cfg, "K_SLOW"))

        # ---- carbonate chemistry (CO2SYS, free scale) -> mol m-3 units ----
        c = carbconst.carbonate_constants(S, T, P)
        TK = T + 273.15
        K1 = c["K1"] * rho
        K2 = c["K2"] * rho
        K1 = K1 * 10.0 ** (-4876.77 / TK + 111.2321 - 16.65226 * math.log(TK))   # Lueker -> Sulpis et al. 2020
        K2 = K2 * 10.0 ** (-3754.45 / TK + 85.3926 - 12.77784 * math.log(TK))
        self.K = dict(K1=K1, K2=K2, Kw=c["KW"] * rho ** 2, KB=c["KB"] * rho, KF=c["KF"] * rho,
                      KSO4=c["KSO4"] * rho, KP1=c["KP1"] * rho, KP2=c["KP2"] * rho, KP3=c["KP3"] * rho,
                      KSi=c["KSi"] * rho, KNH3=c["KNH4"] * rho, KH2S=c["KH2S"] * rho,
                      TB=c["TB"] * (1e-6 * rho), TF=c["TF"] * (1e-6 * rho),
                      KCa=c["KCa"] * rho ** 2, KAr=c["KAr"] * rho ** 2)
        self.dSi_inp = dSi_w * rho
        # bottom-water pH and saturation (initial [H+] guess and display)
        try:
            pH0, omCa, omAr = carbconst.ph_and_omega(1e6 * bw["TA"], 1e6 * bw["DIC"], S, T, P,
                                                     1e6 * dSi_w, 1e6 * bw["PO4"], 1e6 * bw["NH4"], 1e6 * bw["H2S"])
        except Exception:          # the rare CO2SYS non-convergence corner: fall back to our own solve
            pH0 = None; omCa = omAr = float("nan")
        self.omegaCa_bw, self.omegaAr_bw = omCa, omAr
        if pH0 is None:
            H = _solve_H(np.array([bw["TA"] * rho]), np.array([bw["DIC"] * rho]), np.array([bw["NH4"] * rho]),
                         np.array([bw["PO4"] * rho]), np.array([bw["SO4"] * rho]), np.array([bw["H2S"] * rho]),
                         self, np.array([1e-8 * rho]), 80)[0]
        else:
            H = 10.0 ** (-pH0) * rho
        self.H_init = float(H)

        # ---- species list (core + user), transport coefficients ----
        self.species = [dict(id=s[0], label=s[1], phase=s[2], core=True) for s in CORE] + self.ext.species
        self.nvar = len(self.species)
        self.solid = np.array([s["phase"] == "solid" for s in self.species])
        dbl_cfg = cfg.get("DBL_THICKNESS", 1000.0)
        self.dbl_auto = isinstance(dbl_cfg, str) and dbl_cfg.strip().lower() == "auto"
        perm = _num(cfg, "PERMEABILITY")
        self.Dfree, self.Ds, self.dbl, self.Cw = {}, {}, {}, {}
        for s in self.species:
            if s["phase"] != "solute":
                continue
            sid = s["id"]
            if s["core"]:
                a, b = D_COEF[sid]
                D = a + b * T
                Cw = bw[sid] * rho
            else:
                D = s["D0"] + s["DT"] * T
                Cw = s["bw"] * 1e-6 * rho
            self.Dfree[sid] = D
            self.Ds[sid] = _ksed(D, T, U, perm) / self.tort2
            self.dbl[sid] = _dbl_species(D, T, U) if self.dbl_auto else float(dbl_cfg) * 1e-6
            self.Cw[sid] = Cw
        self.ext.finalize(self)
        self._build_transport()

    # ---- linear transport operator: dC/dt = Tdiag*C + Tup*C(k+1) + Tlow*C(k-1) + Tconst ----
    def _build_transport(self):
        Nz, nv, dx = self.Nz, self.nvar, self.dx
        n = Nz * nv
        Tdiag = np.zeros(n); Tup = np.zeros(n); Tlow = np.zeros(n); Tconst = np.zeros(n)
        dzc = 0.5 * (dx[:-1] + dx[1:])
        for i, s in enumerate(self.species):
            if s["phase"] == "solute":
                Ds = self.Ds[s["id"]]
                por, vel = self.phi, self.u_bur
            else:
                Ds = self.D_bio
                por, vel = self.phiS, self.w
            pDf = 0.5 * (por[:-1] * Ds[:-1] + por[1:] * Ds[1:])
            Df = 0.5 * (Ds[:-1] + Ds[1:])
            Pe = 0.5 * vel[:-1] * dzc / np.maximum(Df, 1e-30)
            sg = fv_sigma(Pe)
            # face flux J_{k+1/2} = a_k C_k + b_k C_{k+1}
            a = pDf / dzc + por[:-1] * vel[:-1] * 0.5 * (1.0 + sg)
            b = -pDf / dzc + por[:-1] * vel[:-1] * 0.5 * (1.0 - sg)
            vol = por * dx
            diag = np.zeros(Nz); low = np.zeros(Nz); up = np.zeros(Nz); const = np.zeros(Nz)
            # Jdn
            diag[:-1] -= a
            up[:-1] -= b
            diag[-1] -= por[-1] * vel[-1]
            # Jup
            low[1:] += a
            diag[1:] += b
            if s["phase"] == "solute":
                sid = s["id"]
                Cw = self.Cw[sid]
                G = 1.0 / (self.dbl[sid] / self.Dfree[sid] + 0.5 * dx[0] / (por[0] * Ds[0]))
                diag[0] -= G
                const[0] += G * Cw + por[0] * vel[0] * Cw
                diag /= vol; low /= vol; up /= vol; const /= vol
                diag -= self.alpha
                const += self.alpha * Cw
            else:
                const[0] += self.rain_of(s["id"])
                diag /= vol; low /= vol; up /= vol; const /= vol
            Tdiag[i::nv] = diag
            Tup[i::nv] = up
            Tlow[i::nv] = low
            Tconst[i::nv] = const
        self.Tdiag, self.Tup, self.Tlow, self.Tconst = Tdiag, Tup, Tlow, Tconst

    def rain_of(self, sid):
        if sid in self.rain:
            return self.rain[sid]
        return self.ext.rain.get(sid, 0.0)

    def transport(self, uv):
        nv = self.nvar
        y = self.Tdiag * uv + self.Tconst
        y[:-nv] += self.Tup[:-nv] * uv[nv:]
        y[nv:] += self.Tlow[nv:] * uv[:-nv]
        return y

    def transport_abs(self, uv):
        """Sum of the absolute values of the transport terms (a scale for relative convergence)."""
        nv = self.nvar
        y = np.abs(self.Tdiag * uv) + np.abs(self.Tconst)
        y[:-nv] += np.abs(self.Tup[:-nv] * uv[nv:])
        y[nv:] += np.abs(self.Tlow[nv:] * uv[:-nv])
        return y


# ---------------------------------------------------------------------------------------------------
# 3. Porewater pH: Newton on total alkalinity, fixed number of iterations (H_solve_det)
# ---------------------------------------------------------------------------------------------------
def _solve_H(TA_m3, tCO2_m3, tNH4_m3, tPO4_m3, tSO4_m3, tH2S_m3, site, H0_m3, iters):
    rho = site.rho; K = site.K
    K1, K2, Kw = K["K1"] / rho, K["K2"] / rho, K["Kw"] / (rho * rho)
    KB, KF, KSO4 = K["KB"] / rho, K["KF"] / rho, K["KSO4"] / rho
    KP1, KP2, KP3 = K["KP1"] / rho, K["KP2"] / rho, K["KP3"] / rho
    KSi, KNH3, KH2S = K["KSi"] / rho, K["KNH3"] / rho, K["KH2S"] / rho
    TB, TF = K["TB"] / rho, K["TF"] / rho
    TA = TA_m3 / rho; tCO2 = tCO2_m3 / rho; tNH4 = tNH4_m3 / rho
    tPO4 = tPO4_m3 / rho; tSO4 = tSO4_m3 / rho; tH2S = tH2S_m3 / rho
    dSi = site.dSi_inp / rho
    H = np.clip(H0_m3 / rho, 1e-12, 1.0)
    with np.errstate(all="ignore"):
        for _ in range(iters):
            alk_bor = TB * KB / (KB + H)
            H2 = H * H; H3 = H2 * H
            alk_nc = (tNH4 * KNH3 / (KNH3 + H) + alk_bor
                      + (-TF / (1.0 + KF / H))
                      + tPO4 * (KP1 * KP2 * H + 2.0 * KP1 * KP2 * KP3 - H3) / (H3 + KP1 * H2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
                      + dSi * KSi / (KSi + H)
                      + (-tSO4 / (1.0 + KSO4 / H))
                      + tH2S * KH2S / (KH2S + H)
                      + (Kw / H - H))
            den = H2 + K1 * H + K1 * K2
            alk_c = tCO2 * K1 * (H + 2.0 * K2) / den
            res = TA - (alk_c + alk_nc)
            dAlk = -(tCO2 * K1 * (H2 + K1 * K2 + 4.0 * H * K2) / (den * den) + alk_bor / (KB + H) + Kw / H2 + 1.0)
            ok = np.isfinite(dAlk) & (np.abs(dAlk) > 1e-18)
            delta = np.where(ok, res / np.where(ok, dAlk, 1.0), 0.0)
            ok2 = ok & np.isfinite(delta)
            H = np.where(ok2, np.clip(H + delta, 1e-12, 1.0), H)
    return H * rho


def solve_pH(site, U, H0, iters):
    """[H+] (mol m-3) for each column of the state block U (nvar x M), from the guess H0 (M)."""
    nzc = lambda x: np.maximum(x, 0.0)
    H = _solve_H(nzc(U[IDX["TA"]]), nzc(U[IDX["DIC"]]), nzc(U[IDX["NH4"]]), nzc(U[IDX["PO4"]]),
                 nzc(U[IDX["SO4"]]), nzc(U[IDX["H2S"]]), site, np.clip(H0, HMIN, HMAX), iters)
    return np.clip(H, HMIN, HMAX)


# ---------------------------------------------------------------------------------------------------
# 4. Reactions (React.jl + rates_wD!), vectorised over columns
# ---------------------------------------------------------------------------------------------------
NDIAG = 7   # Rdeg_dO2, Rdeg_dtNO3, Rdeg_dtSO4, Rdeg_pFeOH3, Rdeg_pMnO2, Rdeg_dCH4, Rdeg_total (solid basis)
DIAG_NAMES = ["aerobic_respiration", "denitrification", "sulfate_reduction", "iron_reduction",
              "manganese_reduction", "methanogenesis", "total_Corg_degradation"]


def _calcite_rate(Om, pcal, law):
    """Calcite dissolution (per m3 solid per yr); smoothed two-regime Naviaux et al. (2019)."""
    om = 1.0 - Om
    with np.errstate(all="ignore"):
        w = 1.0 / (1.0 + np.exp(-(Om - law["SCROSS"]) / law["SBLEND"]))
        Rnear = 0.00632 * law["KCAL1_MULT"] * law["KCAL_MULT"] * np.power(np.maximum(om, 0.0), law["N1EXP"])
        Rfar = 20.0 * law["KCAL_MULT"] * np.power(np.maximum(om, 0.0), 4.7)
        R = w * Rnear + (1.0 - w) * Rfar
    return np.where(Om < 1.0, pcal * R, 0.0)


def core_rates(site, U, H, kfast, kslow, p2d, law=LAW):
    """Reaction rates of the 18 core species (rows) for each column of U (core rows only, any values:
    negatives are clamped to zero as in the Julia engine). Returns (R[18,M], diag[7,M], Omega_ca[M])."""
    C = np.maximum(U[:NCORE], 0.0)
    dO2, dtCO2, dtNO3, dtSO4, dtPO4, dtNH4, dtH2S, dFeII, dMnII, dCH4, dalk, dCa = C[:12]
    pfoc, psoc, pFeOH3, pMnO2, pcal, parag = C[12:18]
    K1, K2, KCa, KAr = site.K["K1"], site.K["K2"], site.K["KCa"], site.K["KAr"]
    RC, RN, RP = site.RC, site.RN, site.RP
    dCO3 = (dtCO2 * K1 * K2) / (K1 * K2 + K1 * H + H * H)
    pfoc_k = pfoc * kfast
    psoc_k = psoc * kslow
    # degradation factors (Boudreau 1996 / Couture et al. 2010)
    Mi_dO2 = KMi_dO2 / (KMi_dO2 + dO2)
    Mi_dtNO3 = KMi_dtNO3 / (KMi_dtNO3 + dtNO3)
    Mi_pMnO2 = KMi_pMnO2 / (KMi_pMnO2 + pMnO2)
    Mi_pFeOH3 = KMi_pFeOH3 / (KMi_pFeOH3 + pFeOH3)
    Mi_dtSO4 = KMi_dtSO4 / (KMi_dtSO4 + dtSO4)
    fdO2 = dO2 / (KM_dO2 + dO2)
    fdtNO3 = Mi_dO2 * dtNO3 / (KM_dtNO3 + dtNO3)
    fpMnO2 = Mi_dO2 * Mi_dtNO3 * pMnO2 / (KM_pMnO2 + pMnO2)
    fpFeOH3 = Mi_dO2 * Mi_dtNO3 * Mi_pMnO2 * pFeOH3 / (KM_pFeOH3 + pFeOH3)
    fdtSO4 = Mi_dO2 * Mi_dtNO3 * Mi_pMnO2 * Mi_pFeOH3 * dtSO4 / (KM_dtSO4 + dtSO4)
    fdCH4 = Mi_dO2 * Mi_dtNO3 * Mi_pMnO2 * Mi_pFeOH3 * Mi_dtSO4
    fox = fdO2 + fdtNO3 + fpMnO2 + fpFeOH3 + fdtSO4 + fdCH4
    Rdeg_dO2 = fdO2 * pfoc_k + fdO2 * psoc_k
    Rdeg_dtNO3 = fdtNO3 * pfoc_k + fdtNO3 * psoc_k
    Rdeg_pMnO2 = fpMnO2 * pfoc_k + fpMnO2 * psoc_k
    Rdeg_pFeOH3 = fpFeOH3 * pfoc_k + fpFeOH3 * psoc_k
    Rdeg_dtSO4 = fdtSO4 * pfoc_k + fdtSO4 * psoc_k
    Rdeg_dCH4 = fdCH4 * pfoc_k + fdCH4 * psoc_k
    Rfast_total = fox * pfoc_k
    Rslow_total = fox * psoc_k
    Rdeg_total = Rfast_total + Rslow_total
    # second-order re-oxidation (RADIv1 constants)
    R_NH3 = k_NH3 * dtNH4 * dO2
    R_H2S = k_H2S * dtH2S * dO2
    R_FeII = k_FeII * dFeII * dO2
    R_MnII = k_MnII * dMnII * dO2
    R_CH4_O2 = k_CH4_O2 * dCH4 * dO2
    R_CH4_SO4 = k_CH4_SO4 * dCH4 * dtSO4
    # CaCO3
    OmCa = dCa * dCO3 / KCa
    OmAr = dCa * dCO3 / KAr
    Rdiss_cal = _calcite_rate(OmCa, pcal, law)
    with np.errstate(all="ignore"):
        x = 1.0 - OmAr
        Rdiss_ar = np.where((OmAr > 0.835) & (OmAr <= 1.0), parag * 0.0038 * np.power(np.maximum(x, 0.0), 0.13),
                            np.where(OmAr <= 0.835, parag * 0.042 * np.power(np.maximum(x, 0.0), 1.46), 0.0))
        Rprec_cal = np.where(OmCa > 1.0, 0.4075 * np.power(np.maximum(OmCa - 1.0, 0.0), 1.76), 0.0)
    Rdiss = Rdiss_cal + Rdiss_ar - Rprec_cal
    d2p = 1.0 / p2d
    R = np.empty((NCORE, C.shape[1]))
    R[0] = p2d * -Rdeg_dO2 - (R_FeII / 4.0 + R_MnII / 2.0 + 2.0 * R_H2S + 2.0 * R_NH3) - 2.0 * R_CH4_O2
    R[1] = p2d * (RC * (Rdeg_total - 0.5 * Rdeg_dCH4) + Rdiss) + R_CH4_O2 + R_CH4_SO4
    R[2] = p2d * RC * (-0.8 * Rdeg_dtNO3) + R_NH3
    R[3] = p2d * RC * (-0.5 * Rdeg_dtSO4) + R_H2S - R_CH4_SO4
    R[4] = p2d * RP * Rdeg_total
    R[5] = p2d * RN * Rdeg_total - R_NH3
    R[6] = p2d * RC * (0.5 * Rdeg_dtSO4) - R_H2S + R_CH4_SO4
    R[7] = p2d * RC * (4.0 * Rdeg_pFeOH3) - R_FeII
    R[8] = p2d * RC * (2.0 * Rdeg_pMnO2) - R_MnII
    R[9] = p2d * RC * (0.5 * Rdeg_dCH4) - R_CH4_O2 - R_CH4_SO4
    R[10] = (p2d * ((RN - RP) * (Rdeg_dO2 + Rdeg_dCH4) + (RN - RP + 0.8 * RC) * Rdeg_dtNO3 +
                    (RN - RP + 4.0 * RC) * Rdeg_pMnO2 + (RN - RP + 8.0 * RC) * Rdeg_pFeOH3 +
                    (RN - RP + 1.0 * RC) * Rdeg_dtSO4 + 2.0 * Rdiss) +
             2.0 * R_CH4_SO4 - 2.0 * (R_MnII + R_FeII + R_NH3 + R_H2S))
    R[11] = p2d * Rdiss
    R[12] = -Rfast_total
    R[13] = -Rslow_total
    R[14] = RC * (-4.0 * Rdeg_pFeOH3) + d2p * R_FeII
    R[15] = RC * (-2.0 * Rdeg_pMnO2) + d2p * R_MnII
    R[16] = -Rdiss_cal + Rprec_cal
    R[17] = -Rdiss_ar
    D = np.vstack([Rdeg_dO2, Rdeg_dtNO3, Rdeg_dtSO4, Rdeg_pFeOH3, Rdeg_pMnO2, Rdeg_dCH4, Rdeg_total])
    extra = dict(OmegaCa=OmCa, OmegaAr=OmAr, CO3=dCO3, Rdiss_cal=Rdiss_cal, Rprec_cal=Rprec_cal, Rdiss_ar=Rdiss_ar)
    return R, D, extra


def all_rates(site, U, H, cellidx):
    """Rates of every species (core + user) for the columns of U; cellidx maps columns to depth cells."""
    p2d = site.phiS_phi[cellidx]
    Rc, D, extra = core_rates(site, U, H, site.kfast[cellidx], site.kslow[cellidx], p2d)
    if site.nvar == NCORE and not site.ext.reactions:
        return Rc, D, extra
    R = np.zeros((site.nvar, U.shape[1]))
    R[:NCORE] = Rc
    site.ext.add_rates(R, U, D, p2d, cellidx)
    return R, D, extra


# ---------------------------------------------------------------------------------------------------
# 5. User species and reactions (the "builder")
# ---------------------------------------------------------------------------------------------------
class _Extensions:
    """User-defined species and reactions on top of the RADI core.

    extensions = {
      "species": [
         {"id": "Ba", "label": "Dissolved barium", "phase": "solute",
          "D0": 0.0213, "DT": 0.0009,        # free-solution diffusion coefficient D = D0 + DT*T, m2/yr
          "bw": 0.1},                         # bottom-water concentration, umol/kg
         {"id": "Pb210", "label": "210Pb excess", "phase": "solid",
          "rain": 1e-6,                       # rain at the seafloor, mol m-2 yr-1
          "molar_mass": 0.0}                  # g/mol; adds rain*molar_mass to the solid mass flux (0 = trace)
      ],
      "reactions": [
         {"name": "decay", "type": "first_order", "k": 0.0311, "reactant": "Pb210",
          "stoich": {"Pb210": -1}},
         {"name": "Ba release with organic C", "type": "per_Corg", "k": 0.001, "pathway": "total",
          "stoich": {"Ba": 1}},
         {"name": "X oxidation", "type": "mass_action", "k": 1e3, "reactants": ["X", "O2"],
          "stoich": {"X": -1, "O2": -2, "TA": -2}},
         {"name": "Y uptake", "type": "monod", "k": 5.0, "reactant": "Y",
          "monod": [{"species": "O2", "K": 0.003}], "inhibition": [{"species": "H2S", "K": 0.01}],
          "stoich": {"Y": -1}}
      ]}

    Rate laws (R in mol per m3 of the reference phase per yr; concentrations are clamped at zero):
      first_order : R = k [A]                                   reference phase = phase of A
      mass_action : R = k [A][B]  (or [A] alone, or [A][B][C])  reference phase = phase of A
      monod       : R = k [A] * prod [L]/(K_L + [L]) * prod K_I/(K_I + [I])
      zero_order  : R = k * prod Monod/inhibition terms          reference phase given by "phase"
                    (production only: negative stoichiometry is refused)
      per_Corg    : R = k * (organic C degradation rate of the chosen pathway, mol C per m3 solid per yr)
                    reference phase = solid; pathway in total, aerobic, denitrification, manganese,
                    iron, sulfate, methanogenesis
    The tendency of species s is stoich[s] * R, converted between phases with phiS/phi (solid -> solute)
    or phi/phiS (solute -> solid). Concentrations: solutes mol m-3 porewater, solids mol m-3 solid.
    """
    PATHWAYS = {"aerobic": 0, "denitrification": 1, "sulfate": 2, "iron": 3, "manganese": 4,
                "methanogenesis": 5, "total": 6}

    def __init__(self, spec, site):
        spec = spec or {}
        self.species, self.rain, self.reactions = [], {}, []
        self._mass_flux = 0.0
        taken = set(IDX)
        for s in spec.get("species", []) or []:
            sid = str(s.get("id", "")).strip()
            if not sid or not sid.replace("_", "").isalnum() or not (sid[0].isalpha()):
                raise ValueError(f"species id '{sid}': use letters, digits and _ only, starting with a letter")
            if sid in taken:
                raise ValueError(f"species id '{sid}' is already used")
            taken.add(sid)
            phase = s.get("phase", "solute")
            if phase not in ("solute", "solid"):
                raise ValueError(f"species '{sid}': phase must be 'solute' or 'solid'")
            d = dict(id=sid, label=str(s.get("label", sid)), phase=phase, core=False,
                     color=s.get("color"))
            if phase == "solute":
                D0 = float(s.get("D0", 0.0)); DT = float(s.get("DT", 0.0))
                if not D0 + DT * site.T > 0:
                    raise ValueError(f"solute '{sid}': the diffusion coefficient D0 + DT*T must be > 0 (m2/yr)")
                d.update(D0=D0, DT=DT, bw=float(s.get("bw", 0.0)))
                if d["bw"] < 0:
                    raise ValueError(f"solute '{sid}': bottom-water concentration must be >= 0")
            else:
                rain = float(s.get("rain", 0.0)); mm = float(s.get("molar_mass", 0.0) or 0.0)
                if rain < 0 or mm < 0:
                    raise ValueError(f"solid '{sid}': rain and molar mass must be >= 0")
                d.update(rain=rain, molar_mass=mm)
                self.rain[sid] = rain
                self._mass_flux += rain * mm
            self.species.append(d)
        self._spec_reactions = spec.get("reactions", []) or []

    def extra_solid_mass_flux(self):
        return self._mass_flux

    def finalize(self, site):
        index = {s["id"]: i for i, s in enumerate(site.species)}
        phase = {s["id"]: s["phase"] for s in site.species}
        self.index = index
        for r in self._spec_reactions:
            name = str(r.get("name", "reaction"))
            typ = r.get("type", "first_order")
            k = float(r.get("k", 0.0))
            if k < 0:
                raise ValueError(f"reaction '{name}': k must be >= 0")
            stoich = {str(s): float(v) for s, v in (r.get("stoich") or {}).items() if float(v) != 0.0}
            if not stoich:
                raise ValueError(f"reaction '{name}': give at least one stoichiometric coefficient")
            for s in stoich:
                if s not in index:
                    raise ValueError(f"reaction '{name}': unknown species '{s}'")
            def terms(key):
                out = []
                for t in r.get(key, []) or []:
                    s = str(t["species"])
                    if s not in index:
                        raise ValueError(f"reaction '{name}': unknown species '{s}' in {key}")
                    K = float(t["K"])
                    if not K > 0:
                        raise ValueError(f"reaction '{name}': half-saturation constants must be > 0")
                    out.append((index[s], K))
                return out
            monod, inhib = terms("monod"), terms("inhibition")
            if typ in ("first_order", "monod"):
                a = str(r.get("reactant") or (r.get("reactants") or [None])[0])
                if a not in index:
                    raise ValueError(f"reaction '{name}': reactant '{a}' unknown")
                reac, ref = [index[a]], phase[a]
            elif typ == "mass_action":
                rs = [str(x) for x in (r.get("reactants") or [])]
                if not 1 <= len(rs) <= 3 or any(x not in index for x in rs):
                    raise ValueError(f"reaction '{name}': mass_action needs 1-3 known reactants")
                reac, ref = [index[x] for x in rs], phase[rs[0]]
            elif typ == "zero_order":
                reac, ref = [], r.get("phase", "solute")
                if any(v < 0 for v in stoich.values()):
                    raise ValueError(f"reaction '{name}': a zero-order reaction can only produce")
            elif typ == "per_Corg":
                pw = r.get("pathway", "total")
                if pw not in self.PATHWAYS:
                    raise ValueError(f"reaction '{name}': pathway must be one of {list(self.PATHWAYS)}")
                reac, ref = [], "solid"
            else:
                raise ValueError(f"reaction '{name}': unknown type '{typ}'")
            if typ != "per_Corg" and typ != "zero_order" and not monod and typ == "monod":
                pass
            sto_idx = [(index[s], v, phase[s]) for s, v in stoich.items()]
            self.reactions.append(dict(name=name, type=typ, k=k, reac=reac, ref=ref, monod=monod,
                                       inhib=inhib, stoich=sto_idx, pathway=self.PATHWAYS.get(r.get("pathway", "total"), 6)))

    def add_rates(self, R, U, D, p2d, cellidx):
        if not self.reactions:
            return
        C = np.maximum(U, 0.0)
        for rx in self.reactions:
            if rx["type"] == "per_Corg":
                rate = rx["k"] * D[rx["pathway"]]
            else:
                rate = np.full(U.shape[1], rx["k"])
                for i in rx["reac"]:
                    rate = rate * C[i]
                for i, K in rx["monod"]:
                    rate = rate * C[i] / (K + C[i])
                for i, K in rx["inhib"]:
                    rate = rate * K / (K + C[i])
            for i, v, ph in rx["stoich"]:
                if ph == rx["ref"]:
                    R[i] += v * rate
                elif rx["ref"] == "solid":       # solid basis -> solute: x phiS/phi
                    R[i] += v * rate * p2d
                else:                            # solute basis -> solid: x phi/phiS
                    R[i] += v * rate / p2d


# ---------------------------------------------------------------------------------------------------
# 6. Residual and Jacobian
# ---------------------------------------------------------------------------------------------------
class Model:
    """Residual F(u) = d(state)/dt and its Jacobian, with the porewater [H+] cache of the Julia engine."""

    def __init__(self, site):
        self.site = site
        Nz, nv = site.Nz, site.nvar
        self.n = Nz * nv
        self.H = np.full(Nz, site.H_init)
        self.cells = np.arange(Nz)
        # FD-Jacobian batch: column b*Nz + k is cell k, perturbation b (b = 0 base, 1..nv species b-1)
        self.bcells = np.tile(self.cells, nv + 1)
        # banded storage indices of the block-diagonal reaction Jacobian (A[i,j] -> ab[nv + i - j, j])
        a, b = np.meshgrid(np.arange(nv), np.arange(nv), indexing="ij")
        self.blk_rows = (nv + a - b)[None, :, :].repeat(Nz, axis=0)
        self.blk_cols = (np.arange(Nz)[:, None, None] * nv + b[None, :, :])
        self.n_res = 0
        self.n_jac = 0
        self.user_rows = np.arange(NCORE, nv) if nv > NCORE else None
        self.user_scale = None
        if self.user_rows is not None:
            sp = site.species[NCORE:]
            self.user_vol = np.stack([(site.phiS if s["phase"] == "solid" else site.phi) * site.dx for s in sp], axis=1)
            self.user_rain = np.array([site.rain_of(s["id"]) if s["phase"] == "solid" else 0.0 for s in sp])

    def refresh_H(self, U, iters=60):
        self.H = solve_pH(self.site, U, self.H, iters)
        return self.H

    def residual(self, uv, pH_iters=60):
        """F(u): refresh [H+] from the cache (fixed Newton iterations), then rates + transport."""
        site = self.site
        U = uv.reshape(site.Nz, site.nvar).T
        self.refresh_H(U, pH_iters)
        R, _, _ = all_rates(site, U, self.H, self.cells)
        self.n_res += 1
        tr = site.transport(uv)
        rv = R.T.reshape(-1)
        if self.user_rows is not None:
            ur = self.user_rows
            # throughput of every user species (mol m-2 yr-1): column integral of |reaction| plus its rain
            Rm = np.abs(rv.reshape(site.Nz, site.nvar)[:, ur])
            self.user_Q = np.sum(Rm * self.user_vol, axis=0) + self.user_rain
            # fallback scale when nothing is produced or consumed: gross transport terms
            S = (site.transport_abs(uv) + np.abs(rv)).reshape(site.Nz, site.nvar)[:, ur]
            self.user_scale = np.max(S, axis=0)
        return tr + rv

    def norm(self, F, tol):
        """Convergence norm. Core species: max |F| (mol m-3 yr-1), exactly the Julia criterion. User species:
        their column-integrated imbalance relative to their throughput (reaction + rain), so that trace
        species converge too: imbalance <= RTOL_USER x throughput maps onto the same tolerance."""
        site = self.site
        if self.user_rows is None:
            return float(np.max(np.abs(F)))
        Fm = F.reshape(site.Nz, site.nvar)
        r_core = float(np.max(np.abs(Fm[:, :NCORE])))
        Fu = np.abs(Fm[:, self.user_rows])
        E = np.sum(Fu * self.user_vol, axis=0)
        Q = self.user_Q
        rel_budget = np.where(Q > 0, E / np.maximum(RTOL_USER * Q, 1e-300), 0.0)
        rel_local = np.where(Q > 0, 0.0, np.max(Fu, axis=0) / np.maximum(RTOL_USER * self.user_scale, 1e-300))
        rel_local = np.where(np.max(Fu, axis=0) > 0, rel_local, 0.0)
        r_user = float(np.max(np.maximum(rel_budget, rel_local))) * tol
        return max(r_core, r_user)


    def reaction_jacobian(self, uv, iters=40):
        """Block-diagonal reaction Jacobian (Nz, nv, nv) by forward differences, h = 1e-6 max(|u|, 1e-6),
        with a fresh pH solve for every perturbed state (add_react_jac! / react_cell!)."""
        site = self.site
        Nz, nv = site.Nz, site.nvar
        U = uv.reshape(Nz, nv).T                       # (nv, Nz)
        h = 1e-6 * np.maximum(np.abs(U), 1e-6)         # (nv, Nz)
        X = np.repeat(U[:, None, :], nv + 1, axis=1)   # (nv, nv+1, Nz)
        for j in range(nv):
            X[j, j + 1, :] = U[j] + h[j]
        Xf = X.reshape(nv, (nv + 1) * Nz)
        H = solve_pH(site, Xf, np.tile(self.H, nv + 1), iters)
        R, _, _ = all_rates(site, Xf, H, self.bcells)
        R = R.reshape(nv, nv + 1, Nz)
        # actual perturbation (rounding): X[j, j+1] - U[j] differs from h by ulp; Julia divides by h
        Jr = (R[:, 1:, :] - R[:, :1, :]) / h[None, :, :]    # (nv, nv, Nz): [row a, col b, cell]
        self.n_jac += 1
        return np.transpose(Jr, (2, 0, 1))                   # (Nz, nv, nv)

    def banded(self, Jr, dt=None):
        """Banded storage (2nv+1, n) of A = (1/dt) I - J, or of J itself when dt is None."""
        site = self.site
        nv, n = site.nvar, self.n
        ab = np.zeros((2 * nv + 1, n))
        ab[self.blk_rows, self.blk_cols] = Jr
        ab[nv, :] += site.Tdiag
        ab[0, nv:] += site.Tup[:-nv]
        ab[2 * nv, :-nv] += site.Tlow[nv:]
        if dt is None:
            return ab
        ab = -ab
        ab[nv, :] += 1.0 / dt
        return ab


# ---------------------------------------------------------------------------------------------------
# 7. Solvers: pseudo-transient continuation (solve_steady_state) and the scheduled march (ptc_sched.jl)
# ---------------------------------------------------------------------------------------------------
SCHED_DEFAULT = [(10.0, 30), (100.0, 60), (1e3, 60), (1e4, 40), (1e5, 20)]


class _Progress:
    def __init__(self, cb, every=0.25):
        self.cb, self.every, self.t0, self.last = cb, every, time.time(), -1.0

    def __call__(self, force=False, **info):
        if self.cb is None:
            return
        t = time.time() - self.t0
        if force or t - self.last >= self.every:
            self.last = t
            info["elapsed"] = t
            self.cb(info)


def ptc(model, u0, tol=0.02, max_iter=1500, dt0=1e-3, dtmax=1e12, dtmin=1e-11, pH_iters=50,
        stall_limit=60, max_restart=6, solid_step=0.3, progress=None, phase="newton"):
    """Pseudo-transient continuation; a port of solve_steady_state (same step control)."""
    site = model.site
    nv = site.nvar
    solid = np.tile(site.solid, site.Nz)
    uv = np.array(u0, dtype=float).reshape(-1).copy()
    model.refresh_H(uv.reshape(site.Nz, nv).T, 80)
    F = model.residual(uv, pH_iters)
    r = model.norm(F, tol)
    best = uv.copy(); rbest = r
    dt = dt0; stall = 0; restarts = 0; nrej = 0; it_done = 0; why = "max_iter"
    for it in range(1, max_iter + 1):
        if rbest < tol:
            why = "tol"; break
        it_done = it
        Jr = model.reaction_jacobian(uv)
        ab = model.banded(Jr, dt)
        try:
            delta = solve_banded((nv, nv), ab, F, overwrite_ab=True, check_finite=False)
        except Exception:
            delta = np.full_like(F, np.nan)
        if not np.all(np.isfinite(delta)):
            dt = max(dt * 0.25, dtmin); stall += 1; nrej += 1
        else:
            lim = solid_step * np.maximum(np.abs(uv[solid]), 1.0)
            ad = np.abs(delta[solid])
            sc = 1.0
            over = ad > lim
            if np.any(over):
                sc = min(1.0, float(np.min(lim[over] / ad[over])))
            ut = uv + sc * delta
            ut[solid] = np.maximum(ut[solid], 0.0)
            Ft = model.residual(ut, pH_iters)
            rn = model.norm(Ft, tol)
            if math.isfinite(rn) and rn < 1.5 * r:
                uv, F, r = ut, Ft, rn
                dt = min(dt * 2.5, dtmax)
                if rn < rbest * (1.0 - 1e-6):
                    best = uv.copy(); rbest = rn; stall = 0
                else:
                    stall += 1
            else:
                dt = max(dt * 0.25, dtmin); stall += 1; nrej += 1
        if stall >= stall_limit:
            restarts += 1
            if restarts > max_restart:
                why = "restarts"; break
            uv = best.copy()
            F = model.residual(uv, pH_iters); r = rbest
            dt = dt0; stall = 0
        if progress:
            progress(phase=phase, iteration=it, residual=rbest, dt=dt)
    if rbest < tol:
        why = "tol"
    return best, rbest, dict(iterations=it_done, rejected=nrej, restarts=restarts, stop=why)


def march(model, u0, sched=SCHED_DEFAULT, tol=0.0, pH_iters=50, progress=None):
    """Linearised implicit-Euler march on a dt schedule, one Newton step per time step (ptc_sched.jl)."""
    site = model.site
    nv = site.nvar
    solid = np.tile(site.solid, site.Nz)
    uv = np.array(u0, dtype=float).reshape(-1).copy()
    model.refresh_H(uv.reshape(site.Nz, nv).T, 80)
    total = sum(ns for _, ns in sched); done = 0
    for dt, nsteps in sched:
        F = model.residual(uv, pH_iters)
        if model.norm(F, max(tol, 1e-300)) < tol:
            break
        for s in range(nsteps):
            F = model.residual(uv, pH_iters)
            Jr = model.reaction_jacobian(uv)
            ab = model.banded(Jr, dt)
            delta = solve_banded((nv, nv), ab, F, overwrite_ab=True, check_finite=False)
            if not np.all(np.isfinite(delta)):
                break
            uv = uv + delta
            uv[solid] = np.maximum(uv[solid], 0.0)
            done += 1
            if progress:
                progress(phase="march", step=done, of=total, dt=dt)
    return uv


def cold_start(site):
    """build_cold_start: every solute at its bottom-water value, solids at modest contents."""
    Nz, nv = site.Nz, site.nvar
    U = np.zeros((nv, Nz))
    rho = site.rho
    for i, s in enumerate(site.species[:12]):
        U[i] = site.bw_kg[s["id"]] * rho
    U[12] = 1.0e2; U[13] = 3.0e2; U[16] = 4.5e3
    for i, s in enumerate(site.species):
        if not s["core"] and s["phase"] == "solute":
            U[i] = site.Cw[s["id"]]
    return U.T.reshape(-1)


def solve(site, seed=None, tol=0.02, max_ptc=250, progress=None):
    """Steady state of the site. With a seed: PTC first (capped), then the scheduled march + polish from the
    seed if PTC fails. Without a seed: march + polish from the cold start. Returns (u, info)."""
    model = Model(site)
    prog = _Progress(progress)
    t0 = time.time()
    attempts = []
    u_best, r_best = None, float("inf")
    if seed is not None:
        u0 = np.asarray(seed, dtype=float).reshape(-1)
        u, r, inf = ptc(model, u0, tol=tol, max_iter=max_ptc, progress=prog, phase="newton")
        attempts.append(dict(method="newton", residual=r, **inf))
        u_best, r_best = u, r
    else:
        u0 = cold_start(site)
    if r_best >= tol:
        um = march(model, u0, tol=0.2 * tol, progress=prog)
        u, r, inf = ptc(model, um, tol=tol, max_iter=300, progress=prog, phase="polish")
        attempts.append(dict(method="march+polish", residual=r, **inf))
        if r < r_best:
            u_best, r_best = u, r
    if r_best < tol:
        # tighten: a few near-Newton steps from the converged state (closes the budgets of trace species)
        u, r, inf = ptc(model, u_best, tol=tol * 1e-3, max_iter=25, dt0=1e4, stall_limit=8, max_restart=0,
                        progress=prog, phase="tighten")
        F = model.residual(u, 50)
        r_t = model.norm(F, tol)
        attempts.append(dict(method="tighten", residual=r_t, **inf))
        if r_t <= r_best:
            u_best, r_best = u, r_t
    prog(force=True, phase="done", residual=r_best)
    info = dict(converged=bool(r_best < tol), residual=r_best, tol=tol, seconds=time.time() - t0,
                attempts=attempts, residual_evaluations=model.n_res, jacobians=model.n_jac)
    return u_best, info


# ---------------------------------------------------------------------------------------------------
# 8. Outputs: profiles, rates, benthic fluxes, column budgets
# ---------------------------------------------------------------------------------------------------
def outputs(site, uv, info=None):
    model = Model(site)
    Nz, nv, rho = site.Nz, site.nvar, site.rho
    U = uv.reshape(Nz, nv).T
    model.refresh_H(U, 100)
    H = model.H
    R, D, extra = all_rates(site, U, H, model.cells)
    F = site.transport(uv) + R.T.reshape(-1)
    dx, phi, phiS = site.dx, site.phi, site.phiS
    prof = {"depth_cm": (site.zc * 100).tolist(), "porosity": phi.tolist(),
            "pH_free": (-np.log10(H / rho)).tolist(), "OmegaCa": extra["OmegaCa"].tolist(),
            "OmegaAr": extra["OmegaAr"].tolist(), "CO3_umol_kg": (extra["CO3"] / rho * 1e6).tolist()}
    for i, s in enumerate(site.species):
        if s["phase"] == "solute":
            prof[s["id"]] = (U[i] / rho * 1e6).tolist()          # umol/kg
        else:
            prof[s["id"]] = U[i].tolist()                         # mol m-3 solid
    prof["calcite_wt_pct"] = (np.maximum(U[16], 0) * M_CaCO3 / RHO_SOLID * 100).tolist()
    prof["aragonite_wt_pct"] = (np.maximum(U[17], 0) * M_CaCO3 / RHO_SOLID * 100).tolist()
    prof["POC_wt_pct"] = ((np.maximum(U[12], 0) + np.maximum(U[13], 0)) * 12.011 / RHO_SOLID * 100).tolist()
    rates = {"depth_cm": prof["depth_cm"]}
    for j, name in enumerate(DIAG_NAMES):
        rates[name] = D[j].tolist()                              # mol C m-3 solid yr-1
    rates["calcite_dissolution"] = extra["Rdiss_cal"].tolist()   # mol m-3 solid yr-1
    rates["calcite_precipitation"] = extra["Rprec_cal"].tolist()
    rates["aragonite_dissolution"] = extra["Rdiss_ar"].tolist()
    for i, s in enumerate(site.species):
        rates["net_" + s["id"]] = R[i].tolist()
    # benthic fluxes: positive = out of the sediment (efflux); O2 also given as uptake
    flux = {}
    for i, s in enumerate(site.species):
        if s["phase"] != "solute":
            continue
        sid = s["id"]
        Cw = site.Cw[sid]
        G = 1.0 / (site.dbl[sid] / site.Dfree[sid] + 0.5 * dx[0] / (phi[0] * site.Ds[sid][0]))
        Jin = G * (Cw - U[i, 0]) + phi[0] * site.u_bur[0] * Cw
        Jin += float(np.sum(site.alpha * (Cw - U[i]) * phi * dx))
        flux[sid] = -Jin
    vol_s, vol_d = phiS * dx, phi * dx
    budget = {
        "calcite_rain": site.rain["calcite"],
        "calcite_dissolution": float(np.sum(extra["Rdiss_cal"] * vol_s)),
        "calcite_precipitation": float(np.sum(extra["Rprec_cal"] * vol_s)),
        "calcite_burial": float(site.w[-1] * phiS[-1] * max(U[16, -1], 0.0)),
        "aragonite_rain": site.rain["aragonite"],
        "aragonite_dissolution": float(np.sum(extra["Rdiss_ar"] * vol_s)),
        "POC_rain": site.Fpoc,
        "POC_degradation": float(np.sum(D[6] * vol_s)),
        "POC_burial": float(site.w[-1] * phiS[-1] * (max(U[12, -1], 0.0) + max(U[13, -1], 0.0))) + site.Froc,
        "TA_column_source": float(np.sum(R[10] * vol_d)),
        "O2_column_consumption": float(-np.sum(R[0] * vol_d)),
    }
    for j, name in enumerate(DIAG_NAMES[:6]):
        budget["Corg_" + name] = float(np.sum(D[j] * vol_s))
    # closures (relative): TA = SWI influx + column source - burial out; calcite = rain - net dissolution - burial
    TA_in = -flux["TA"]
    budget["TA_closure"] = (TA_in + budget["TA_column_source"] - phi[-1] * site.u_bur[-1] * U[10, -1]) / max(abs(budget["TA_column_source"]), 1e-12)
    cal_in = site.rain["calcite"]
    budget["calcite_closure"] = ((cal_in - (budget["calcite_dissolution"] - budget["calcite_precipitation"]) - budget["calcite_burial"])
                                 / max(cal_in, 1e-12)) if cal_in > 0 else 0.0
    opd = None
    for k in range(Nz):
        if U[0, k] / rho * 1e6 < 1.0:
            opd = site.zc[k] * 100; break
    summary = dict(OPD_cm=opd, rho_sw=rho, omegaCa_bw=site.omegaCa_bw, omegaAr_bw=site.omegaAr_bw,
                   DBL_um={k: v * 1e6 for k, v in site.dbl.items()}, burial_velocity_cm_kyr=site.w[-1] * 1e5,
                   bioturbation_D0_cm2_yr=site.D_bio[0] * 1e4, irrigation_alpha0_per_yr=site.alpha_0,
                   max_abs_residual=float(np.max(np.abs(F))))
    return dict(version=VERSION, species=[{k: v for k, v in s.items() if k in ("id", "label", "phase", "core")}
                                          for s in site.species],
                profiles=prof, rates=rates, fluxes=flux, budget=budget, summary=summary, solver=info or {})


# ---------------------------------------------------------------------------------------------------
# 9. Entry point used by the web worker
# ---------------------------------------------------------------------------------------------------
def run(request_json, progress=None):
    """request = {"config": {...}, "extensions": {...}, "seed": [[...]] (nvar x Nz, optional),
                  "tol": 0.02}; returns a JSON string with outputs and the final state (to seed the next run)."""
    req = json.loads(request_json) if isinstance(request_json, str) else request_json
    site = Site(req.get("config", {}), req.get("extensions"))
    seed = req.get("seed")
    u0 = None
    if seed is not None:
        S = np.asarray(seed, dtype=float)
        if S.shape == (site.nvar, site.Nz):
            u0 = S.T.reshape(-1)
        elif S.shape[1] == site.Nz and S.shape[0] < site.nvar:      # core seed + new user species
            full = cold_start(site).reshape(site.Nz, site.nvar).T
            full[:S.shape[0]] = S
            u0 = full.T.reshape(-1)
    u, info = solve(site, u0, tol=float(req.get("tol", 0.02)), progress=progress)
    out = outputs(site, u, info)
    out["state"] = u.reshape(site.Nz, site.nvar).T.tolist()
    return json.dumps(out)
