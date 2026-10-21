"""
carbconst -- exact Python/numpy port of the carbonate-system numbers RADI.jl
takes from CO2System.jl (CO2SYS v2.0 Julia port) and gsw_rho.jl.

Pure Python + numpy (``math`` from the standard library for scalars); runs
unchanged in Pyodide.  No scipy.

What is reproduced
------------------
``RADI_model.jl`` (function ``calculate_constants``) does::

    rho_sw = gsw_rho(S, T, P)
    co2s   = CO2System.CO2SYS(TA, DIC, 1, 2, S, T, T, P, P,
                              Si, PO4, NH4, H2S, 3, 10, 1)[1]

with *scalar* arguments, i.e.

* input pair TA (PAR1TYPE = 1) and DIC (PAR2TYPE = 2), all in umol/kg-SW;
* input and output conditions identical (TEMPIN = TEMPOUT = T,
  PRESIN = PRESOUT = P; the model passes water depth in m as P in dbar);
* pHSCALEIN = 3 -> every K is reported on the **free** pH scale;
* K1K2CONSTANTS = 10 -> Lueker et al. (2000);
* KSO4CONSTANTS = 1 -> KSO4 of Dickson (1990) and total borate of
  Uppstrom (1974).

Only these option branches are implemented here.

Column map of the RADI build of CO2System.jl (it appends KNH4 and KH2S to the
"K input" and "K output" blocks, so these numbers differ from the header comment
at the top of CO2System.jl, which still lists the original CO2SYS v2.0 layout):

=====  ==============  ===================================================
col    CO2SYS name     returned here as
=====  ==============  ===================================================
30     OmegaCAout      ``ph_and_omega(...)[1]``     (output conditions)
31     OmegaARout      ``ph_and_omega(...)[2]``     (output conditions)
35     pHinFREE        ``ph_and_omega(...)[0]``     (input conditions)
54     K1input         ``carbonate_constants(...)["K1"]``
55     K2input         ``["K2"]``
58     KWinput         ``["KW"]``
59     KBinput         ``["KB"]``
60     KFinput         ``["KF"]``
61     KSinput         ``["KSO4"]``
62-64  KP1/2/3input    ``["KP1"]``, ``["KP2"]``, ``["KP3"]``
65     KSiinput        ``["KSi"]``
66     KNH4input       ``["KNH4"]``  (RADI calls it KNH3)
67     KH2Sinput       ``["KH2S"]``
83     TB              ``["TB"]``    (umol/kg-SW, i.e. TB*1e6)
84     TF              ``["TF"]``    (umol/kg-SW, i.e. TF*1e6)
86     KCainput        ``["KCa"]``
87     KArinput        ``["KAr"]``
=====  ==============  ===================================================

Because T and P are the same at "input" and "output", CO2SYS evaluates the
same code with the same numbers twice, so input and output columns are
bit-identical (3/18, 15/30, 16/31, 35/39, 53-67/68-82, 86/88, 87/89; checked in
``test_carbconst.py``).

Units
-----
* K1, K2, KB, KP1, KP2, KP3, KSi, KNH4, KH2S: mol/kg-SW, **free** pH scale,
  at in-situ T and P.
* KW: (mol/kg-SW)^2, free scale.
* KF, KSO4: mol/kg-SW, free scale (inherently free), pressure-corrected.
* TB, TF: **umol/kg-SW** (as in columns 83/84; RADI multiplies by 1e-6*rho).
* KCa, KAr: stoichiometric solubility products, (mol/kg-SW)^2, P-corrected.
* pH: free scale (column 35).

Exactness
---------
Every expression mirrors the Julia source operation by operation, so
rounding happens the same way: Julia's operator precedence, left-to-right
sums, and ``x^2``/``x^3`` written as an integer literal (Julia lowers these to
``x*x`` / ``x*x*x`` via ``Base.literal_pow``, so they are written that way
here, not as ``x**2``). The only operations that can still differ in the last
bit are exp/log/pow, because Julia uses its own pure-Julia implementations and
CPython/numpy use the C library (~1 ulp at most).  Measured against Julia on
467 points (test_carbconst.py): 96-100 % of values bit-identical per quantity; max relative
error 1.6e-15 for the constants (5.3e-14 for KNH4 under Pyodide, where musl's
pow differs and the Clegg & Whitfield polynomial cancels), 0 for density, TB
and TF, 6.2e-16 for pH, 1.1e-14 for Omega.

The pH is **not** a fully converged root.  CO2SYS v2 solves the alkalinity
equation by Newton's method from pH = 8 with an *approximate* slope (it omits
the phosphate, silicate, NH4, H2S, HSO4 and HF terms) and stops as soon as
|delta pH| <= 1e-4.  The returned value is up to ~5e-5 pH (median 1e-6) away
from the exact root of its own equation, i.e. up to ~1e-4 relative in Omega.
``ph_and_omega`` reproduces the same iteration exactly (same start, residual,
slope, step halving and stopping rule), which is the only way to match
columns 35/30/31 to round-off.  In a thin corner of input space the iteration
never converges (see ``ph_and_omega``; Julia hangs, this port raises).

Note for the web port: RADI_model.jl does not use K1/K2 from CO2SYS as is.
It takes columns 54/55 (Lueker 2000), multiplies by rho, and then rescales to
Sulpis et al. (2020) with a temperature-only factor (K1K2CONSTANTS = 16 is
dead code in this CO2System.jl: ``F=(WhichKs==16)`` compares a vector with an
integer, so it is always false and K1/K2 stay NaN)::

    _TKsu = T + 273.15
    K1 = K1 * 10.0^(-4876.77/_TKsu + 111.2321 - 16.65226*log(_TKsu))
    K2 = K2 * 10.0^(-3754.45/_TKsu + 85.3926  - 12.77784*log(_TKsu))

``carbonate_constants`` returns the raw CO2SYS columns, so that step (and the
rho scaling) still has to be applied by the caller, as in the Julia model.

References for every constant are given at the point of use below.
"""

import math

import numpy as np

__all__ = ["seawater_density", "carbonate_constants", "ph_and_omega"]

# ---------------------------------------------------------------------------
# CO2SYS settings used by RADI (only these branches are ported)
# ---------------------------------------------------------------------------
_R_GAS = 83.1451          # ml bar-1 K-1 mol-1, DOEv2 (CO2SYS "RGasConstant")
_LN10 = math.log(10)      # Julia: log(10)  (bit-identical, 2.302585092994046)
_LOG01 = math.log(0.1)    # Julia: log(0.1) (bit-identical)
_PH_GUESS = 8.0           # CO2SYS first guess
_PH_TOL = 0.0001          # CO2SYS tolerance: stop when |deltapH| <= 1e-4
# Guard only -- the Julia original has no limit and simply never returns when
# the iteration falls into a 2-cycle.  Converging cases need <= ~50 iterations
# over the whole test domain; the cap is large so that slowly converging cases
# close to the cycle boundary still finish exactly as in Julia.
_MAX_ITER = 100000
_NONCONV_MSG = (
    "CO2SYS pH iteration does not converge for {what}: Newton's method with "
    "CO2SYS's approximate slope (which omits the H2S, Si, NH4, PO4, HSO4 and HF "
    "terms) falls into a 2-cycle when those terms exceed half of dTA/dpH "
    "(e.g. TA~500, DIC~400, H2S~100 umol/kg at low S and high P). The Julia "
    "original loops forever on such input.")


# ---------------------------------------------------------------------------
# Scalar / array back-ends.
# Scalars use the ``math`` module (C libm, like Julia's exp/log/pow these are
# correctly rounded in almost all cases); arrays use numpy ufuncs.
# ---------------------------------------------------------------------------
class _ScalarOps:
    exp = staticmethod(math.exp)
    log = staticmethod(math.log)
    sqrt = staticmethod(math.sqrt)

    @staticmethod
    def pow(x, y):
        # Julia ``x^y`` with a non-literal / non-integer Float64 exponent
        return x ** y


class _ArrayOps:
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    pow = staticmethod(np.power)


def _prepare(*args):
    """Pick the back-end.  All 0-d -> Python floats; else broadcast float arrays."""
    if all(np.ndim(a) == 0 for a in args):
        return _ScalarOps, True, tuple(float(a) for a in args)
    arrs = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
    return _ArrayOps, False, tuple(np.array(a, dtype=float) for a in arrs)


# ---------------------------------------------------------------------------
# Seawater density  (gsw_rho.jl)
# ---------------------------------------------------------------------------
def seawater_density(S, T, P):
    """In-situ seawater density (kg m-3), identical to Julia ``gsw_rho(S, T, P)``.

    75-term polynomial for specific volume of Roquet et al. (2015), as in the
    TEOS-10 GSW toolbox v3.05 function ``gsw_rho(SA, CT, p)``:

    * IOC, SCOR and IAPSO (2010): The international thermodynamic equation of
      seawater - 2010.  IOC Manuals and Guides No. 56 (appendix A.20, K).
    * Roquet, F., Madec, G., McDougall, T.J., Barker, P.M. (2015): Accurate
      polynomial expressions for the density and specific volume of seawater
      using the TEOS-10 standard.  Ocean Modelling 90, 29-43.
    * McDougall, T.J. et al. (2003): J. Atmos. Ocean. Tech. 20, 730-741.

    Parameters follow how RADI calls it: ``S`` is passed as Absolute Salinity
    SA [g/kg] (RADI gives practical salinity), ``T`` as Conservative
    Temperature CT [degC] (RADI gives in-situ temperature) and ``P`` as sea
    pressure [dbar] (RADI gives water depth in m).  No conversion is done here,
    so the result equals the model's ``rho_sw`` exactly.

    Scalars return a float; array-likes broadcast and return an ndarray.
    """
    M, scalar, (SA, CT, p) = _prepare(S, T, P)

    sfac = 0.0248826675584615            # sfac = 1/(40*(35.16504/35))
    offset = 5.971840214030754e-1        # offset = deltaS*sfac, deltaS = 24

    x2 = sfac * SA
    xs = M.sqrt(x2 + offset)
    ys = CT * 0.025
    z = p * 1e-4

    v000 = 1.0769995862e-3
    v001 = -6.0799143809e-5
    v002 = 9.9856169219e-6
    v003 = -1.1309361437e-6
    v004 = 1.0531153080e-7
    v005 = -1.2647261286e-8
    v006 = 1.9613503930e-9
    v010 = -1.5649734675e-5
    v011 = 1.8505765429e-5
    v012 = -1.1736386731e-6
    v013 = -3.6527006553e-7
    v014 = 3.1454099902e-7
    v020 = 2.7762106484e-5
    v021 = -1.1716606853e-5
    v022 = 2.1305028740e-6
    v023 = 2.8695905159e-7
    v030 = -1.6521159259e-5
    v031 = 7.9279656173e-6
    v032 = -4.6132540037e-7
    v040 = 6.9111322702e-6
    v041 = -3.4102187482e-6
    v042 = -6.3352916514e-8
    v050 = -8.0539615540e-7
    v051 = 5.0736766814e-7
    v060 = 2.0543094268e-7
    v100 = -3.1038981976e-4
    v101 = 2.4262468747e-5
    v102 = -5.8484432984e-7
    v103 = 3.6310188515e-7
    v104 = -1.1147125423e-7
    v110 = 3.5009599764e-5
    v111 = -9.5677088156e-6
    v112 = -5.5699154557e-6
    v113 = -2.7295696237e-7
    v120 = -3.7435842344e-5
    v121 = -2.3678308361e-7
    v122 = 3.9137387080e-7
    v130 = 2.4141479483e-5
    v131 = -3.4558773655e-6
    v132 = 7.7618888092e-9
    v140 = -8.7595873154e-6
    v141 = 1.2956717783e-6
    v150 = -3.3052758900e-7
    v200 = 6.6928067038e-4
    v201 = -3.4792460974e-5
    v202 = -4.8122251597e-6
    v203 = 1.6746303780e-8
    v210 = -4.3592678561e-5
    v211 = 1.1100834765e-5
    v212 = 5.4620748834e-6
    v220 = 3.5907822760e-5
    v221 = 2.9283346295e-6
    v222 = -6.5731104067e-7
    v230 = -1.4353633048e-5
    v231 = 3.1655306078e-7
    v240 = 4.3703680598e-6
    v300 = -8.5047933937e-4
    v301 = 3.7470777305e-5
    v302 = 4.9263106998e-6
    v310 = 3.4532461828e-5
    v311 = -9.8447117844e-6
    v312 = -1.3544185627e-6
    v320 = -1.8698584187e-5
    v321 = -4.8826139200e-7
    v330 = 2.2863324556e-6
    v400 = 5.8086069943e-4
    v401 = -1.7322218612e-5
    v402 = -1.7811974727e-6
    v410 = -1.1959409788e-5
    v411 = 2.5909225260e-6
    v420 = 3.8595339244e-6
    v500 = -2.1092370507e-4
    v501 = 3.0927427253e-6
    v510 = 1.3864594581e-6
    v600 = 3.1932457305e-5

    # Same nesting (hence same rounding) as gsw_rho.jl lines 150-162.
    v = (v000 + xs*(v100 + xs*(v200 + xs*(v300 + xs*(v400 + xs*(v500
        + v600*xs))))) + ys*(v010 + xs*(v110 + xs*(v210 + xs*(v310 + xs*(v410
        + v510*xs)))) + ys*(v020 + xs*(v120 + xs*(v220 + xs*(v320 + v420*xs)))
        + ys*(v030 + xs*(v130 + xs*(v230 + v330*xs)) + ys*(v040 + xs*(v140
        + v240*xs) + ys*(v050 + v150*xs + v060*ys))))) + z*(v001 + xs*(v101
        + xs*(v201 + xs*(v301 + xs*(v401 + v501*xs)))) + ys*(v011 + xs*(v111
        + xs*(v211 + xs*(v311 + v411*xs))) + ys*(v021 + xs*(v121 + xs*(v221
        + v321*xs)) + ys*(v031 + xs*(v131 + v231*xs) + ys*(v041 + v141*xs
        + v051*ys)))) + z*(v002 + xs*(v102 + xs*(v202 + xs*(v302 + v402*xs)))
        + ys*(v012 + xs*(v112 + xs*(v212 + v312*xs)) + ys*(v022 + xs*(v122
        + v222*xs) + ys*(v032 + v132*xs + v042*ys))) + z*(v003 + xs*(v103
        + v203*xs) + ys*(v013 + v113*xs + v023*ys) + z*(v004 + v104*xs + v014*ys
        + z*(v005 + v006*z))))))

    return 1 / v


# ---------------------------------------------------------------------------
# CO2SYS "Constants(TempC, Pdbar)" + "CaSolubility" for
# pHScale = 3, WhichKs = 10, WhoseKSO4 = 1
# ---------------------------------------------------------------------------
def _pfac(deltaV, Kappa, Pbar, RT):
    """ln(K_P/K_0) = (-dV + 0.5*Kappa*P)*P/(R*T)  (Millero 1995, eqs. 90-92).

    deltaV in cm3/mol, Kappa in cm3/mol/bar, P in bar, R in ml bar/K/mol.
    Same operation order as every ``ln*fac`` line in CO2System.jl.
    """
    return (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT


def _constants(Sal, TempC, Pdbar, M):
    """All constants at (S, T, P), mirroring CO2System.jl ``Constants`` and the
    constant part of ``CaSolubility`` statement by statement.

    Returns a dict: K's on the free scale (mol/kg-SW), KW in (mol/kg-SW)^2,
    KF/KS free scale, TB/TF/TS/Ca in **mol/kg-SW**, KCa/KAr in (mol/kg-SW)^2.
    """
    sqrSal = M.sqrt(Sal)
    TempK = TempC + 273.15
    RT = _R_GAS * TempK
    logTempK = M.log(TempK)
    Pbar = Pdbar / 10

    # ---- Total concentrations (mol/kg-SW) ---------------------------------
    # Total borate: Uppstrom, L., Deep-Sea Research 21:161-162, 1974
    #   (KSO4CONSTANTS = 1): 0.0004157*S/35
    TB = 0.0004157 * Sal / 35
    # Total fluoride: Riley, J. P., Deep-Sea Research 12:219-220, 1965
    TF = (0.000067 / 18.998) * Sal / 1.80655
    # Total sulfate: Morris, A. W. and Riley, J. P., Deep-Sea Res. 13:699-705, 1966
    TS = (0.14 / 96.062) * Sal / 1.80655

    # Ionic strength: DOE handbook (1994), ch. 5 p. 13/22, eq. 7.2.4
    IonS = 19.924*Sal / (1000 - 1.005*Sal)
    sqrtIonS = M.sqrt(IonS)

    # ---- KS (HSO4- = H+ + SO4--), free scale -------------------------------
    # Dickson, A. G., J. Chemical Thermodynamics 22:113-127, 1990 (eqs 22-23,
    # Table 4; typo on p. 121: e9 should be e8).  Given in mol/kg-H2O,
    # converted to mol/kg-SW with (1 - 0.001005*S).
    lnKS = (-4276.1/TempK + 141.328 - 23.093*logTempK
            + (-13856/TempK + 324.57 - 47.986*logTempK)*sqrtIonS
            + (35474/TempK - 771.54 + 114.723*logTempK)*IonS
            + (-2698/TempK)*sqrtIonS*IonS + (1776/TempK)*(IonS*IonS))
    KS = M.exp(lnKS) * (1 - 0.001005*Sal)

    # ---- KF (HF = H+ + F-), free scale -------------------------------------
    # Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979.
    # mol/kg-H2O -> mol/kg-SW.  NB: Julia writes IonS^0.5 (pow, not sqrt).
    lnKF = 1590.2/TempK - 12.641 + 1.525*M.pow(IonS, 0.5)
    KF = M.exp(lnKF) * (1 - 0.001005*Sal)

    # pH-scale conversion factors at 1 atm (NOT pressure corrected):
    SWStoTOT = (1 + TS/KS) / (1 + TS/KS + TF/KF)

    # ---- KB (boric acid), total -> SWS -------------------------------------
    # Dickson, A. G., Deep-Sea Research 37:755-766, 1990.
    lnKBtop = (-8966.9 - 2890.53*sqrSal - 77.942*Sal + 1.728*sqrSal*Sal
               - 0.0996*(Sal*Sal))
    lnKB = (lnKBtop/TempK + 148.0248 + 137.1942*sqrSal + 1.62142*Sal
            + (-24.4344 - 25.085*sqrSal - 0.2474*Sal)*logTempK
            + 0.053105*sqrSal*TempK)
    KB = M.exp(lnKB) / SWStoTOT

    # ---- KW (water), SWS, (mol/kg-SW)^2 -----------------------------------
    # Millero, F. J., Geochim. Cosmochim. Acta 59:661-677, 1995
    # (his check value of 1.6 umol/kg-SW should be 6.2).
    lnKW = (148.9802 - 13847.26/TempK - 23.6521*logTempK
            + (-5.977 + 118.67/TempK + 1.0495*logTempK)*sqrSal - 0.01615*Sal)
    KW = M.exp(lnKW)

    # ---- KP1, KP2, KP3 (phosphoric acid), KSi (silicic acid), SWS ----------
    # Yao, W. and Millero, F. J., Aquatic Geochemistry 1:53-88, 1995.
    # KP's are SWS in mol/kg-SW; KSi given SWS in molal units -> mol/kg-SW.
    lnKP1 = (-4576.752/TempK + 115.54 - 18.453*logTempK
             + (-106.736/TempK + 0.69171)*sqrSal
             + (-0.65643/TempK - 0.01844)*Sal)
    KP1 = M.exp(lnKP1)
    lnKP2 = (-8814.715/TempK + 172.1033 - 27.927*logTempK
             + (-160.34/TempK + 1.3566)*sqrSal
             + (0.37335/TempK - 0.05778)*Sal)
    KP2 = M.exp(lnKP2)
    lnKP3 = (-3070.75/TempK - 18.126 + (17.27039/TempK + 2.81197)*sqrSal
             + (-44.99486/TempK - 0.09984)*Sal)
    KP3 = M.exp(lnKP3)
    lnKSi = (-8904.2/TempK + 117.4 - 19.334*logTempK
             + (-458.79/TempK + 3.5913)*sqrtIonS
             + (188.74/TempK - 1.5998)*IonS
             + (-12.1652/TempK + 0.07871)*(IonS*IonS))
    KSi = M.exp(lnKSi) * (1 - 0.001005*Sal)

    # ---- KNH4 (NH4+ = NH3 + H+), total, mol/kg-H2O -> mol/kg-SW -> SWS -----
    # Clegg, S. L. and Whitfield, M., Geochim. Cosmochim. Acta 59:2403-2421,
    # 1995 (added to CO2SYS by J. D. Sharp).
    TK05 = M.pow(TempK, 0.5)
    PKNH4 = (9.244605 - 2729.33*(1/298.15 - 1/TempK)
             + (0.04203362 - 11.24742/TempK)*M.pow(Sal, 0.25)
             + (-13.6416 + 1.176949*TK05
                - 0.02860785*TempK + 545.4834/TempK)*M.pow(Sal, 0.5)
             + (-0.1462507 + 0.0090226468*TK05
                - 0.0001471361*TempK + 10.5425/TempK)*M.pow(Sal, 1.5)
             + (0.004669309 - 0.0001691742*TK05
                - 0.5677934/TempK)*(Sal*Sal)
             + (-2.354039E-05 + 0.009698623/TempK)*M.pow(Sal, 2.5))
    KNH4 = M.pow(10.0, -PKNH4)            # total scale, mol/kg-H2O
    KNH4 = KNH4 * (1 - 0.001005*Sal)      # mol/kg-SW
    KNH4 = KNH4 / SWStoTOT                # SWS scale

    # ---- KH2S (H2S = H+ + HS-), total -> SWS --------------------------------
    # Millero, F. J., Plese, T. and Fernandez, M., Limnol. Oceanogr.
    # 33:269-274, 1988 (added to CO2SYS by J. D. Sharp).
    KH2S = M.exp(225.838 - 13275.3/TempK - 34.6435*M.log(TempK)
                 + 0.3449*M.pow(Sal, 0.5) - 0.0274*Sal) / SWStoTOT

    # ---- K1, K2 (carbonic acid), K1K2CONSTANTS = 10 ------------------------
    # Lueker, T. J., Dickson, A. G. and Keeling, C. D., Marine Chemistry
    # 70:105-119, 2000 (Mehrbach et al. 1973 data refit on the total scale,
    # mol/kg-SW), converted total -> SWS.
    pK1 = (3633.86/TempK - 61.2172 + 9.6777*M.log(TempK) - 0.011555*Sal
           + 0.0001152*(Sal*Sal))
    K1 = M.pow(10.0, -pK1) / SWStoTOT
    pK2 = (471.78/TempK + 25.929 - 3.16967*M.log(TempK) - 0.01781*Sal
           + 0.0001122*(Sal*Sal))
    K2 = M.pow(10.0, -pK2) / SWStoTOT

    # ---- Pressure corrections ----------------------------------------------
    # deltaV in cm3/mol, Kappa in cm3/mol/bar; no salinity dependence (S~35).
    TC2 = TempC*TempC
    # K1: Millero (1995) (= Millero 1979, 1992; data of Culberson & Pytkowicz 1968)
    lnK1fac = _pfac(-25.5 + 0.1271*TempC,
                    (-3.08 + 0.0877*TempC) / 1000, Pbar, RT)
    # K2: Millero (1995), same sources
    lnK2fac = _pfac(-15.82 - 0.0219*TempC,
                    (1.13 - 0.1475*TempC) / 1000, Pbar, RT)
    # KB: Millero (1979), data of Culberson & Pytkowicz (1968)
    lnKBfac = _pfac(-29.48 + 0.1622*TempC - 0.002608*TC2,
                    -2.84/1000, Pbar, RT)
    # KW: Millero (1983) and his programs CO2ROY(T).BAS
    lnKWfac = _pfac(-20.02 + 0.1119*TempC - 0.001409*TC2,
                    (-5.13 + 0.0794*TempC) / 1000, Pbar, RT)
    # KF: Millero (1995) (= Millero 1983), free scale
    lnKFfac = _pfac(-9.78 - 0.009*TempC - 0.000942*TC2,
                    (-3.91 + 0.054*TempC) / 1000, Pbar, RT)
    # KS: Millero (1995) (= Millero 1983), free scale
    lnKSfac = _pfac(-18.03 + 0.0466*TempC + 0.000316*TC2,
                    (-4.53 + 0.09*TempC) / 1000, Pbar, RT)
    # KP1, KP2, KP3: Millero (1995) (= Millero 1983)
    lnKP1fac = _pfac(-14.51 + 0.1211*TempC - 0.000321*TC2,
                     (-2.67 + 0.0427*TempC) / 1000, Pbar, RT)
    lnKP2fac = _pfac(-23.12 + 0.1758*TempC - 0.002647*TC2,
                     (-5.15 + 0.09*TempC) / 1000, Pbar, RT)
    lnKP3fac = _pfac(-26.57 + 0.202*TempC - 0.003042*TC2,
                     (-4.08 + 0.0714*TempC) / 1000, Pbar, RT)
    # KSi: no published values; CO2SYS uses those of boric acid
    lnKSifac = _pfac(-29.48 + 0.1622*TempC - 0.002608*TC2,
                     -2.84 / 1000, Pbar, RT)
    # KNH4, KH2S: Millero (1995) coefficients, added to CO2SYS by J. D. Sharp
    lnKNH4fac = _pfac(-26.43 + 0.0889*TempC - 0.000905*TC2,
                      (-5.03 + 0.0814*TempC) / 1000, Pbar, RT)
    lnKH2Sfac = _pfac(-11.07 - 0.009*TempC - 0.000942*TC2,
                      (-2.89 + 0.054*TempC) / 1000, Pbar, RT)

    K1 = K1 * M.exp(lnK1fac)
    K2 = K2 * M.exp(lnK2fac)
    KW = KW * M.exp(lnKWfac)
    KB = KB * M.exp(lnKBfac)
    KF = KF * M.exp(lnKFfac)
    KS = KS * M.exp(lnKSfac)
    KP1 = KP1 * M.exp(lnKP1fac)
    KP2 = KP2 * M.exp(lnKP2fac)
    KP3 = KP3 * M.exp(lnKP3fac)
    KSi = KSi * M.exp(lnKSifac)
    KNH4 = KNH4 * M.exp(lnKNH4fac)
    KH2S = KH2S * M.exp(lnKH2Sfac)

    # ---- SWS -> free scale (pHScale = 3), with P-corrected KS and KF -------
    SWStoTOT = (1 + TS/KS) / (1 + TS/KS + TF/KF)
    FREEtoTOT = 1 + TS/KS
    pHfactor = SWStoTOT / FREEtoTOT
    K1 = K1 * pHfactor
    K2 = K2 * pHfactor
    KW = KW * pHfactor
    KB = KB * pHfactor
    KP1 = KP1 * pHfactor
    KP2 = KP2 * pHfactor
    KP3 = KP3 * pHfactor
    KSi = KSi * pHfactor
    KNH4 = KNH4 * pHfactor
    KH2S = KH2S * pHfactor

    # ---- CaSolubility: [Ca], KCa, KAr --------------------------------------
    # [Ca]: Riley, J. P. and Tongudai, M., Chemical Geology 2:263-269, 1967
    Ca = 0.02128/40.087 * (Sal/1.80655)
    # Calcite: Mucci, A., Amer. J. of Science 283:781-799, 1983
    logKCa = -171.9065 - 0.077993*TempK + 2839.319/TempK
    logKCa = logKCa + 71.595*logTempK/_LN10
    logKCa = logKCa + (-0.77712 + 0.0028426*TempK + 178.34/TempK)*sqrSal
    logKCa = logKCa - 0.07711*Sal + 0.0041249*sqrSal*Sal
    KCa = M.pow(10.0, logKCa)
    # Aragonite: Mucci (1983)
    logKAr = -171.945 - 0.077993*TempK + 2903.293/TempK
    logKAr = logKAr + 71.595*logTempK/_LN10
    logKAr = logKAr + (-0.068393 + 0.0017276*TempK + 88.135/TempK)*sqrSal
    logKAr = logKAr - 0.10018*Sal + 0.0059415*sqrSal*Sal
    KAr = M.pow(10.0, logKAr)
    # Pressure, calcite: Ingle, Marine Chemistry 3:301-319, 1975
    # (= Millero 1979; Millero 1995 has typos)
    deltaVKCa = -48.76 + 0.5304*TempC
    KappaKCa = (-11.76 + 0.3692*TempC) / 1000
    KCa = KCa * M.exp(_pfac(deltaVKCa, KappaKCa, Pbar, RT))
    # Pressure, aragonite: Millero, GCA 43:1651-1661, 1979 (dV_Ar = dV_Ca + 2.8)
    deltaVKAr = deltaVKCa + 2.8
    KappaKAr = KappaKCa
    KAr = KAr * M.exp(_pfac(deltaVKAr, KappaKAr, Pbar, RT))

    return dict(K1=K1, K2=K2, KW=KW, KB=KB, KF=KF, KS=KS, KP1=KP1, KP2=KP2,
                KP3=KP3, KSi=KSi, KNH4=KNH4, KH2S=KH2S, TB=TB, TF=TF, TS=TS,
                Ca=Ca, KCa=KCa, KAr=KAr, FREEtoTOT=FREEtoTOT)


def carbonate_constants(S, T, P):
    """Equilibrium constants exactly as RADI reads them from CO2SYS.

    Equivalent to columns 54, 55, 58-67, 83, 84, 86, 87 of
    ``CO2System.CO2SYS(TA, DIC, 1, 2, S, T, T, P, P, Si, PO4, NH4, H2S, 3, 10, 1)``
    (none of them depends on TA, DIC or the nutrients).

    Parameters
    ----------
    S : practical salinity
    T : temperature, degC
    P : pressure, dbar (RADI passes water depth in m)

    Returns
    -------
    dict with keys (all at in-situ T and P):

    ``K1``, ``K2``  carbonic acid, Lueker et al. (2000), mol/kg-SW, free scale
    ``KW``          water, Millero (1995), (mol/kg-SW)^2, free scale
    ``KB``          boric acid, Dickson (1990b), mol/kg-SW, free scale
    ``KF``          HF, Dickson & Riley (1979), mol/kg-SW, free scale
    ``KSO4``        HSO4-, Dickson (1990a), mol/kg-SW, free scale (CO2SYS "KS")
    ``KP1..KP3``    phosphoric acid, Yao & Millero (1995), mol/kg-SW, free
    ``KSi``         silicic acid, Yao & Millero (1995), mol/kg-SW, free
    ``KNH4``        ammonium, Clegg & Whitfield (1995), mol/kg-SW, free
    ``KH2S``        hydrogen sulfide, Millero et al. (1988), mol/kg-SW, free
    ``TB``          total borate, Uppstrom (1974), **umol/kg-SW** (col. 83)
    ``TF``          total fluoride, Riley (1965), **umol/kg-SW** (col. 84)
    ``KCa``, ``KAr`` calcite / aragonite solubility products, Mucci (1983),
                    pressure-corrected after Ingle (1975) / Millero (1979),
                    (mol/kg-SW)^2

    Pressure corrections are those of CO2SYS (Millero 1979/1983/1995); the
    K's are converted to the SWS scale, pressure-corrected, then multiplied
    by SWStoTOT/FREEtoTOT evaluated with the pressure-corrected KS and KF.

    Scalars return floats; array-likes broadcast and return ndarrays.
    """
    M, scalar, (S_, T_, P_) = _prepare(S, T, P)
    c = _constants(S_, T_, P_, M)
    return {
        "K1": c["K1"], "K2": c["K2"], "KW": c["KW"], "KB": c["KB"],
        "KF": c["KF"], "KSO4": c["KS"],
        "KP1": c["KP1"], "KP2": c["KP2"], "KP3": c["KP3"], "KSi": c["KSi"],
        "KNH4": c["KNH4"], "KH2S": c["KH2S"],
        "TB": c["TB"] * 1e6, "TF": c["TF"] * 1e6,     # CO2SYS: TVEC*1e6
        "KCa": c["KCa"], "KAr": c["KAr"],
    }


# ---------------------------------------------------------------------------
# pH from TA and DIC: CO2SYS "CalculatepHfromTATC" (RADI build)
# ---------------------------------------------------------------------------
def _residual_slope(pH, TA, TC, TP, TSi, TNH4, TH2S, c, M):
    """Alkalinity residual and approximate dTA/dpH used by CO2SYS.

    Line-for-line copy of the loop body of ``CalculatepHfromTATC`` in the RADI
    build of CO2System.jl (Lewis & Wallace 1998 algorithm, v04.01, with the
    NH4 and H2S alkalinity terms added by RADI). All concentrations in mol/kg.

    Quirks kept on purpose (they are what CO2SYS v2 computes):
    * ``Hfree = H/FREEtoTOT`` assumes H is on the total scale, but with
      pHScale = 3 the K's (hence H) are already free, so Hfree, HSO4 and HF
      use [H]free/(1 + TS/KS) instead of [H]free.
    * The slope ignores the PAlk, SiAlk, NH4, H2S, HSO4 and HF terms (as in
      the original), so the iteration converges only linearly.
    """
    K1 = c["K1"]; K2 = c["K2"]; KW = c["KW"]; KB = c["KB"]
    KP1 = c["KP1"]; KP2 = c["KP2"]; KP3 = c["KP3"]; KSi = c["KSi"]
    KNH4 = c["KNH4"]; KH2S = c["KH2S"]; TB = c["TB"]
    TS = c["TS"]; KS = c["KS"]; TF = c["TF"]; KF = c["KF"]

    H = M.pow(10.0, -pH)
    Denom = H*H + K1*H + K1*K2
    CAlk = TC*K1*(H + 2*K2) / Denom
    BAlk = TB*KB / (KB + H)
    OH = KW / H
    PhosTop = KP1*KP2*H + 2*KP1*KP2*KP3 - H*H*H
    PhosBot = H*H*H + KP1*(H*H) + KP1*KP2*H + KP1*KP2*KP3
    PAlk = TP*PhosTop / PhosBot
    SiAlk = TSi*KSi / (KSi + H)
    AmmAlk = TNH4*KNH4 / (KNH4 + H)
    HSAlk = TH2S*KH2S / (KH2S + H)
    FREEtoTOT = 1 + TS/KS            # pH scale conversion factor
    Hfree = H / FREEtoTOT            # "for H on the total scale"
    HSO4 = TS / (1 + KS/Hfree)       # KS is on the free scale
    HF = TF / (1 + KF/Hfree)         # KF is on the free scale
    Residual = (TA - CAlk - BAlk - OH - PAlk - SiAlk - AmmAlk - HSAlk
                + Hfree + HSO4 + HF)
    Slope = _LN10*(TC*K1*H*(H*H + K1*K2 + 4*H*K2)/Denom/Denom
                   + BAlk*H/(KB + H) + OH + H)
    return Residual, Slope


def _solve_ph_scalar(TA, TC, TP, TSi, TNH4, TH2S, c):
    """CO2SYS Newton iteration for one sample (what RADI's scalar call does)."""
    pH = _PH_GUESS
    deltapH = _PH_TOL + 1
    n = 0
    while abs(deltapH) > _PH_TOL:
        n += 1
        if n > _MAX_ITER:
            raise RuntimeError(_NONCONV_MSG.format(
                what=f"TA={TA*1e6:g}, DIC={TC*1e6:g} umol/kg"))
        Residual, Slope = _residual_slope(pH, TA, TC, TP, TSi, TNH4, TH2S,
                                          c, _ScalarOps)
        deltapH = Residual / Slope       # Newton's method
        while abs(deltapH) > 1:          # keep the jump from being too big
            deltapH = deltapH / 2
        pH = pH + deltapH
    return pH


def _solve_ph_array(TA, TC, TP, TSi, TNH4, TH2S, c):
    """Same iteration, vectorised, with a *per-element* stopping rule.

    Each element is iterated exactly as if CO2SYS had been called with that
    sample alone (RADI's usage).  Julia's CO2SYS called with vectors keeps
    iterating *all* samples until *every* sample has converged, which gives
    slightly different (more converged) numbers -- not reproduced here.
    """
    pH = np.full(TA.shape, _PH_GUESS)
    deltapH = np.full(TA.shape, _PH_TOL + 1)
    active = np.abs(deltapH) > _PH_TOL
    n = 0
    while active.any():
        n += 1
        if n > _MAX_ITER:
            raise RuntimeError(_NONCONV_MSG.format(
                what=f"{int(active.sum())} element(s), flat indices "
                     f"{np.nonzero(active)[0][:10].tolist()}"))
        i = np.nonzero(active)[0]
        ci = {k: v[i] for k, v in c.items()}
        Residual, Slope = _residual_slope(pH[i], TA[i], TC[i], TP[i], TSi[i],
                                          TNH4[i], TH2S[i], ci, _ArrayOps)
        d = Residual / Slope
        big = np.abs(d) > 1
        while big.any():
            d[big] = d[big] / 2
            big = np.abs(d) > 1
        pH[i] = pH[i] + d
        deltapH[i] = d
        active = np.abs(deltapH) > _PH_TOL
    return pH


def ph_and_omega(TA, DIC, S, T, P, Si=0, PO4=0, NH4=0, H2S=0):
    """Free-scale pH and calcite/aragonite saturation states, as CO2SYS gives them.

    Equivalent to columns 35 (pHinFREE), 30 (OmegaCAout) and 31 (OmegaARout)
    of ``CO2System.CO2SYS(TA, DIC, 1, 2, S, T, T, P, P, Si, PO4, NH4, H2S,
    3, 10, 1)`` called with scalars, as RADI does.

    Parameters
    ----------
    TA, DIC : total alkalinity and dissolved inorganic carbon, umol/kg-SW
    S, T, P : practical salinity, temperature (degC), pressure (dbar)
    Si, PO4, NH4, H2S : total silicate, phosphate, ammonium, sulfide,
        umol/kg-SW (NH4 and H2S enter only the pH solver, as in the Julia code)

    Returns
    -------
    (pH_free, OmegaCa, OmegaAr)

    Method (exactly CO2SYS v2 / Lewis & Wallace 1998):

    1. TA, DIC and nutrients are converted to mol/kg by dividing by 1e6.
    2. pH (on the scale of the K's, here free) is found by Newton's method
       from pH = 8 on the alkalinity equation TA = HCO3 + 2CO3 + B(OH)4 + OH
       + PAlk + SiAlk + NH3 + HS - Hfree - HSO4 - HF, stepping until
       |delta pH| <= 1e-4 (steps larger than 1 are halved).
    3. Column 35 is that pH passed through FindpHOnAllScales:
       pHtot = pH - (-ln(FREEtoTOT)/ln(0.1)); pHfree = pHtot - ln(FREEtoTOT)/ln(0.1)
       (identical to pH up to one rounding).
    4. Omega = [CO3--][Ca++]/Ksp with [CO3--] = DIC*K1*K2/(K1*H + H^2 + K1*K2),
       [Ca++] from Riley & Tongudai (1967), Ksp from Mucci (1983) with the
       pressure corrections of Ingle (1975) / Millero (1979).

    Scalars return floats; array-likes broadcast and return ndarrays (each
    element is solved independently, like a scalar CO2SYS call).

    Raises
    ------
    RuntimeError
        if the CO2SYS iteration does not converge.  This happens only in a thin
        corner of input space where the terms missing from CO2SYS's slope
        (mainly H2S, then Si, PO4, NH4) exceed half of dTA/dpH, e.g.
        TA=500, DIC=400, Si=200, PO4=10, H2S=100 umol/kg at S=5, T=-2, P=11000:
        the iterate then alternates forever between two pH values and the
        Julia original never returns.
    """
    M, scalar, (TA_, DIC_, S_, T_, P_, Si_, PO4_, NH4_, H2S_) = _prepare(
        TA, DIC, S, T, P, Si, PO4, NH4, H2S)
    shape = None if scalar else TA_.shape
    if not scalar:
        TA_, DIC_, S_, T_, P_, Si_, PO4_, NH4_, H2S_ = (
            a.ravel() for a in (TA_, DIC_, S_, T_, P_, Si_, PO4_, NH4_, H2S_))

    c = _constants(S_, T_, P_, M)

    # umol/kg -> mol/kg exactly as CO2SYS does (division, not *1e-6)
    TAm = TA_ / 1e6
    TCm = DIC_ / 1e6
    TP = PO4_ / 1e6
    TSi = Si_ / 1e6
    TNH4 = NH4_ / 1e6
    TH2S = H2S_ / 1e6
    TAm = TAm - 0.0      # TAc - PengCorrection (0 for K1K2CONSTANTS = 10)

    if scalar:
        pH = _solve_ph_scalar(TAm, TCm, TP, TSi, TNH4, TH2S, c)
    else:
        pH = _solve_ph_array(TAm, TCm, TP, TSi, TNH4, TH2S, c)

    # Column 35: FindpHOnAllScales(PHic) with pHScale = 3
    FREEtoTOT = 1 + c["TS"]/c["KS"]
    factor = -M.log(FREEtoTOT) / _LOG01
    pHtot = pH - factor
    pHfree = pHtot - M.log(FREEtoTOT) / _LOG01

    # Columns 30/31: CaSolubility(Sal, TempCo, Pdbaro, TCc, PHoc).  PHoc is
    # re-solved at "output" conditions; with TEMPOUT = TEMPIN and
    # PRESOUT = PRESIN it is bit-identical to the input pH.
    K1 = c["K1"]; K2 = c["K2"]
    H = M.pow(10.0, -pH)
    CO3 = TCm*K1*K2 / (K1*H + H*H + K1*K2)
    OmegaCa = CO3*c["Ca"] / c["KCa"]
    OmegaAr = CO3*c["Ca"] / c["KAr"]

    if not scalar:
        pHfree, OmegaCa, OmegaAr = (np.reshape(a, shape)
                                    for a in (pHfree, OmegaCa, OmegaAr))
    return pHfree, OmegaCa, OmegaAr
