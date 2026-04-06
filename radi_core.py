"""
RADI (Reactive-Advective-Diffusive-Irrigative) diagenetic sediment model.

A generic, data-driven 1D model of biogeochemical processes in marine sediments.
Solves coupled PDEs for dissolved (solute) and particulate (solid) species in a
vertical sediment column.

Species and reactions are user-defined; the core handles transport physics,
carbonate chemistry, and time integration via scipy.integrate.solve_ivp (BDF).

Author: O. Sulpis (CEREGE/ERC Deep-C)
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, csc_matrix
from scipy.optimize import brentq
import warnings


# ============================================================================
# DATACLASSES: Species, Reaction, Environment
# ============================================================================

@dataclass
class Species:
    """Definition of a chemical species (solute or solid)."""
    name: str
    long_name: str
    species_type: str  # "solute" or "solid"

    # Diffusion (for solutes; ignored for solids)
    D0: float  # free-solution diffusivity at 0°C [m²/yr]
    D_T_coeff: float  # temperature coefficient: D(T) = D0 + D_T_coeff * T [m²/yr/°C]

    # Boundary conditions
    bc_top_type: str  # "concentration" or "flux"
    bc_top_value: float  # concentration [mol/m³] or flux [mol/m²/yr]
    bc_bot_type: str  # "gradient" (zero-gradient)
    bc_bot_value: float  # typically 0.0

    # Initial condition
    initial_value: float  # uniform initial concentration [mol/m³]

    # Display
    unit: str  # e.g. "mol/m³"
    color: str  # hex color for plotting

    def __post_init__(self):
        """Validate species definition."""
        if self.species_type not in ["solute", "solid"]:
            raise ValueError(f"Invalid species_type: {self.species_type}")
        if self.bc_top_type not in ["concentration", "flux"]:
            raise ValueError(f"Invalid bc_top_type: {self.bc_top_type}")


@dataclass
class Reaction:
    """Definition of a chemical reaction."""
    name: str
    rate_type: str  # "monod", "mass_action", "saturation", "inhibited_monod", "custom"
    rate_constant: float  # k [units depend on rate_type]
    reactants: List[Tuple[str, float]] = field(default_factory=list)  # [(species, coeff), ...]
    products: List[Tuple[str, float]] = field(default_factory=list)   # [(species, coeff), ...]

    # Monod kinetics
    monod_species: Optional[str] = None
    half_saturation: float = 1.0  # KM [mol/m³]

    # Inhibition
    inhibitors: List[Tuple[str, float]] = field(default_factory=list)  # [(species, KI), ...]

    # Saturation-state reactions (CaCO3)
    saturation_params: Dict = field(default_factory=dict)


@dataclass
class Environment:
    """Physical and chemical environment parameters."""
    T: float  # temperature [°C]
    S: float  # salinity [PSU]
    P: float  # pressure [dbar]
    depth: float  # water depth [m]
    U: float  # bottom current speed [m/s]

    # Porosity
    phi0: float  # surface porosity
    phi_inf: float  # deep porosity
    beta: float  # porosity decay constant [1/m]

    # Bioturbation & irrigation
    D_bio_0: float  # surface bioturbation [m²/yr]
    lambda_b: float  # bioturbation decay depth [m]
    lambda_i: float  # irrigation decay depth [m]
    alpha0: float  # surface irrigation rate [1/yr]

    # Grid
    depth_sed: float  # sediment column depth [m]
    Nz: int  # number of depth cells
    dz_top: float  # top cell thickness [m]
    dz_bot: float  # bottom cell thickness [m]

    # Time
    tspan: Tuple[float, float]  # (t_start, t_end) [years]
    mode: str  # "steady_state" or "transient"

    # Redfield ratios
    RC: float = 1.0  # C:C
    RN: float = 0.16  # N:C
    RP: float = 0.01  # P:C

    # DBL thickness override (if set, use directly instead of computing from current speed)
    dbl_thickness_override: Optional[float] = None  # [m]

    # Output time step control for transient simulations
    t_eval_points: Optional[int] = None  # number of output time points

    # Carbonate system flag — set to False to skip CO3/Omega computation
    # (huge speed-up when no CaCO3 species are in the model)
    solve_carbonate: bool = True


# ============================================================================
# DIFFUSIVITY LIBRARY (Boudreau 1997)
# ============================================================================

DIFFUSIVITY_LIBRARY = {
    "O2": {"D0": 0.031558, "D_T_coeff": 0.001428},
    "HCO3": {"D0": 0.015179, "D_T_coeff": 0.000795},
    "CO3": {"D0": 0.012380, "D_T_coeff": 0.000600},
    "NO3": {"D0": 0.030863, "D_T_coeff": 0.001153},
    "SO4": {"D0": 0.015779, "D_T_coeff": 0.000712},
    "PO4": {"D0": 0.009783, "D_T_coeff": 0.000513},
    "NH4": {"D0": 0.030926, "D_T_coeff": 0.001225},
    "H2S": {"D0": 0.028938, "D_T_coeff": 0.001314},
    "Fe2": {"D0": 0.010761, "D_T_coeff": 0.000466},
    "Mn2": {"D0": 0.009625, "D_T_coeff": 0.000481},
    "CH4": {"D0": 0.041629, "D_T_coeff": 0.000764},
    "Ca": {"D0": 0.011771, "D_T_coeff": 0.000529},
    "DIC": {"D0": 0.015179, "D_T_coeff": 0.000795},  # proxy for HCO3
}


# ============================================================================
# UTILITY FUNCTIONS: Grid, Density, Carbonate Chemistry
# ============================================================================

def generate_grid(depth_sed: float, Nz: int, dz_top: float, dz_bot: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate geometrically stretched depth grid.

    Parameters
    ----------
    depth_sed : float
        Total sediment column depth [m]
    Nz : int
        Number of cells
    dz_top : float
        Top cell thickness [m]
    dz_bot : float
        Bottom cell thickness [m]

    Returns
    -------
    z_centers : ndarray of shape (Nz,)
        Depth at cell centers [m]
    z_edges : ndarray of shape (Nz+1,)
        Depth at cell edges [m]
    dz : ndarray of shape (Nz,)
        Cell thicknesses [m]
    """
    if Nz < 2:
        raise ValueError("Nz must be >= 2")

    # Growth rate for geometric progression
    r = (dz_bot / dz_top) ** (1.0 / (Nz - 1))
    r = min(r, 1.10)  # cap to avoid excessive stretching

    # Generate raw spacings
    dz_raw = dz_top * r ** np.arange(Nz, dtype=float)

    # Scale to exact depth
    dz = dz_raw * (depth_sed / np.sum(dz_raw))

    # Cell edges (bottom = depth_sed)
    z_edges = np.concatenate([[0.0], np.cumsum(dz)])

    # Cell centers
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return z_centers, z_edges, dz


def sw_density(T: float, S: float, P: float = 0.0) -> float:
    """
    Simplified seawater equation of state.

    Parameters
    ----------
    T : float
        Temperature [°C]
    S : float
        Salinity [PSU]
    P : float
        Pressure [dbar], default 0

    Returns
    -------
    rho : float
        Density [kg/m³]
    """
    # UNESCO (1983) / Millero & Poisson (1981) formulation
    rho = (999.842594
           + 6.793952e-2 * T
           - 9.095290e-3 * T**2
           + 1.001685e-4 * T**3
           - 1.120083e-6 * T**4
           + 6.536332e-9 * T**5)

    rho += S * (0.824493
                - 4.0899e-3 * T
                + 7.6438e-5 * T**2
                - 8.2467e-7 * T**3
                + 5.3875e-9 * T**4)

    rho += S**1.5 * (-5.72466e-3
                     + 1.0227e-4 * T
                     - 1.6546e-6 * T**2)

    rho += 4.8314e-4 * S**2

    # Pressure correction (simplified)
    if P > 0:
        rho += P * 4.5e-3

    return rho


def dbl_thickness(D_w: float, U: float) -> float:
    """
    Diffusive boundary layer thickness.

    Boudreau & Jorgensen (2001) parameterization:
    dbl = D_w / (0.00138 * U^0.567)

    Parameters
    ----------
    D_w : float
        Molecular diffusivity at water temp [m²/yr]
    U : float
        Bottom current speed [m/s]

    Returns
    -------
    dbl : float
        DBL thickness [m]
    """
    if U < 1e-10:
        return 1.0e-3  # 1 mm default

    # Boudreau & Jorgensen (2001) parameterization
    # dbl = D_w / (0.00138 * U^0.567)
    # where D_w is in m²/s and U is in m/s, result in meters
    D_w_m2s = D_w / 3.1557e7  # convert m²/yr to m²/s
    dbl = D_w_m2s / (0.00138 * U**0.567)
    return dbl


class CarbonateSystem:
    """
    Simplified but functional carbonate chemistry solver.

    Uses Newton-Raphson to solve the alkalinity balance and compute
    pH, [CO3²⁻], and saturation states for calcite and aragonite.

    Equilibrium constants from Lueker et al. (2000), Millero (1995), Mucci (1983).
    """

    def __init__(self, T: float, S: float, P: float):
        """
        Initialize with T/S/P conditions.

        Parameters
        ----------
        T : float
            Temperature [°C]
        S : float
            Salinity [PSU]
        P : float
            Pressure [dbar]
        """
        self.T = T
        self.S = S
        self.P = P
        self.T_K = T + 273.15

        # Compute equilibrium constants (Lueker et al. 2000 formulation)
        self._compute_constants()

    def _compute_constants(self):
        """Compute K1, K2, Kb, Kw, Ksp_calcite, Ksp_aragonite."""
        T = self.T
        S = self.S
        P = self.P
        T_K = self.T_K

        # K1 (Lueker et al. 2000)
        lnK1 = (-60.2409 + 93.4517 * (100.0 / T_K) + 23.3585 * np.log(T_K / 100.0)
                + S * (0.02169 - 0.0002 * (T_K - 298.15)))
        self.K1 = np.exp(lnK1)

        # K2 (Lueker et al. 2000)
        lnK2 = (-60.7883 + 93.4517 * (100.0 / T_K) + 23.3585 * np.log(T_K / 100.0)
                + S * (0.02169 - 0.0002 * (T_K - 298.15)) - 34.0447)
        self.K2 = np.exp(lnK2)

        # Kb (Dickson 1990)
        lnKb = (-8966.90 - 2890.53 * np.sqrt(S) - 77.27 * S
                + 1.728e-4 * S**1.5 + (148.0 + 137.1925 * np.sqrt(S)
                + 1.62142 * S) / T_K
                - (0.47569 + 0.181967 * np.sqrt(S)) * np.log(T_K))
        self.Kb = np.exp(lnKb)

        # Kw (Millero 1995)
        lnKw = 148.96502 - 13847.26 / T_K - 23.6521 * np.log(T_K)
        self.Kw = np.exp(lnKw)

        # Ksp calcite (Mucci 1983 + pressure correction)
        lnKsp_cal_0 = (-171.9065 - 0.077993 * T_K + 2839.319 / T_K
                       + 71.595 * np.log10(T_K)
                       + S * (-0.77712 + 0.0028426 * T_K + 178.34 / T_K))
        self.Ksp_calcite_0 = 10.0 ** lnKsp_cal_0

        # Pressure correction (Millero 1995)
        dV_cal = -48.76 - 0.5304 * T  # molar volume change [cm³/mol]
        dK_cal = -11.76e-4 * T - 0.3692  # compressibility change [cm³·bar/mol]
        Pbar = P * 10.0  # convert dbar to bar
        ln_ratio_cal = -(dV_cal / 83.14472 / T_K) * Pbar + (0.5 * dK_cal / 83.14472 / T_K) * Pbar**2
        self.Ksp_calcite = self.Ksp_calcite_0 * np.exp(ln_ratio_cal)

        # Ksp aragonite (similar, slightly different parameters)
        lnKsp_ara_0 = (-171.945 - 0.077993 * T_K + 2903.293 / T_K
                       + 71.595 * np.log10(T_K)
                       + S * (-0.068393 + 0.0017276 * T_K + 88.135 / T_K))
        self.Ksp_aragonite_0 = 10.0 ** lnKsp_ara_0

        dV_ara = -46.18 - 0.528 * T
        dK_ara = -11.24e-4 * T - 0.3962
        ln_ratio_ara = -(dV_ara / 83.14472 / T_K) * Pbar + (0.5 * dK_ara / 83.14472 / T_K) * Pbar**2
        self.Ksp_aragonite = self.Ksp_aragonite_0 * np.exp(ln_ratio_ara)

    def solve_alkalinity(self, DIC, ALK, Ca=10.28e-3):
        """
        Solve the alkalinity balance for pH, [CO3²⁻], Omega_calcite, Omega_aragonite.

        Accepts both scalar and array inputs.

        Parameters
        ----------
        DIC : float or ndarray
            Dissolved Inorganic Carbon [mol/kg-SW]. Can be scalar or shape (Nz,)
        ALK : float or ndarray
            Alkalinity [mol/kg-SW]. Can be scalar or shape (Nz,)
        Ca : float or ndarray
            Calcium concentration [mol/kg-SW], default 10.28e-3 (seawater)
            Can be scalar or shape (Nz,)

        Returns
        -------
        pH : float or ndarray
            pH (total scale)
        CO3 : float or ndarray
            [CO3²⁻] [mol/kg-SW]
        Omega_calcite : float or ndarray
            Saturation state for calcite
        Omega_aragonite : float or ndarray
            Saturation state for aragonite
        """
        # Convert to numpy arrays, preserving scalar case
        DIC = np.asarray(DIC)
        ALK = np.asarray(ALK)
        Ca = np.asarray(Ca)
        is_scalar = DIC.ndim == 0

        DIC = np.atleast_1d(DIC)
        ALK = np.atleast_1d(ALK)
        Ca = np.atleast_1d(Ca)

        K1, K2 = self.K1, self.K2
        Kb, Kw = self.Kb, self.Kw

        # Initial guess: solve simplified system
        H = np.maximum(K1 * DIC / ALK, 1.0e-10)

        # Newton-Raphson iteration (vectorized)
        for iteration in range(50):
            # Carbonate speciation
            HCO3 = DIC * K1 / (H + K1)
            CO3_calc = DIC * K1 * K2 / (H * H + H * K1 + K1 * K2)

            # Borate
            B_total = 416.0e-6  # mol/kg-SW (constant for ocean)
            BOH4 = B_total * Kb / (H + Kb)

            # OH-
            OH = Kw / H

            # ALK equation
            f = HCO3 + 2.0 * CO3_calc + BOH4 + OH - H - ALK

            # Derivative w.r.t. H
            d_HCO3 = -DIC * K1 / (H + K1)**2
            d_CO3 = -DIC * K1 * K2 * (2.0 * H + K1) / (H * H + H * K1 + K1 * K2)**2
            d_BOH4 = -B_total * Kb / (H + Kb)**2
            d_OH = -Kw / H**2
            df = d_HCO3 + 2.0 * d_CO3 + d_BOH4 + d_OH - 1.0

            # Newton step (avoid division by zero)
            df_safe = np.where(np.abs(df) < 1e-14, 1.0, df)
            H_new = H - f / df_safe
            H_new = np.maximum(H_new, 1.0e-12)

            # Check convergence
            converged = np.abs(H_new - H) / np.maximum(H, 1e-14) < 1.0e-8
            if np.all(converged):
                H = H_new
                break
            H = H_new

        pH = -np.log10(H)

        # Final speciation
        HCO3 = DIC * K1 / (H + K1)
        CO3 = DIC * K1 * K2 / (H * H + H * K1 + K1 * K2)

        # Saturation states
        Omega_calcite = (Ca * CO3) / self.Ksp_calcite
        Omega_aragonite = (Ca * CO3) / self.Ksp_aragonite

        # Return scalars if input was scalar
        if is_scalar:
            return float(pH[0]), float(CO3[0]), float(Omega_calcite[0]), float(Omega_aragonite[0])
        else:
            return pH, CO3, Omega_calcite, Omega_aragonite


# ============================================================================
# RADI MODEL CLASS
# ============================================================================

class RADIModel:
    """
    The main RADI diagenetic model solver.

    This class handles:
    - Grid generation and depth-dependent parameters
    - Transport equations (solute diffusion, irrigation, advection; solid bioturbation, advection)
    - Generic reaction network
    - Time integration via scipy.integrate.solve_ivp with BDF method
    """

    def __init__(self, species_list: List[Species], reaction_list: List[Reaction],
                 environment: Environment):
        """
        Initialize the RADI model.

        Parameters
        ----------
        species_list : List[Species]
            List of chemical species definitions
        reaction_list : List[Reaction]
            List of reactions
        environment : Environment
            Physical/chemical environment
        """
        self.species_list = species_list
        self.species = {s.name: s for s in species_list}
        self.reaction_list = reaction_list
        self.env = environment

        # Build ordered lists for indexing
        self.species_names = [s.name for s in species_list]
        self.n_species = len(species_list)

        # Pre-compute name-to-index mapping for fast lookups
        self.species_idx = {name: i for i, name in enumerate(self.species_names)}

        # Classify solutes and solids
        self.solute_names = [s.name for s in species_list if s.species_type == "solute"]
        self.solid_names = [s.name for s in species_list if s.species_type == "solid"]
        self.n_solutes = len(self.solute_names)
        self.n_solids = len(self.solid_names)

        # Pre-compute solute index mapping
        self.solute_idx = {name: i for i, name in enumerate(self.solute_names)}

        # Grid and parameters
        self._setup_grid()
        self._setup_parameters()
        self._build_jacobian_sparsity()

        # Carbonate system (only if DIC present AND solve_carbonate is True)
        self.carb_sys = None
        if "DIC" in self.species and self.env.solve_carbonate:
            self.carb_sys = CarbonateSystem(environment.T, environment.S, environment.P)

    def _setup_grid(self):
        """Generate depth grid and precompute geometries."""
        self.z_centers, self.z_edges, self.dz = generate_grid(
            self.env.depth_sed, self.env.Nz,
            self.env.dz_top, self.env.dz_bot
        )
        self.Nz = self.env.Nz

    def _setup_parameters(self):
        """Precompute porosity, diffusivities, burial velocities, etc."""
        z = self.z_centers
        dz = self.dz

        # Porosity profile
        self.phi = (self.env.phi0 - self.env.phi_inf) * np.exp(-self.env.beta * z) + self.env.phi_inf
        self.phi_s = 1.0 - self.phi

        # Tortuosity squared (Boudreau)
        self.tort2 = 1.0 - 2.0 * np.log(self.phi)
        self.tort2 = np.maximum(self.tort2, 0.1)  # avoid singularities

        # Diffusivities (solutes only)
        self.D_eff = np.zeros((self.n_solutes, self.Nz))
        for i, sp_name in enumerate(self.solute_names):
            sp = self.species[sp_name]
            D_T = sp.D0 + sp.D_T_coeff * self.env.T
            self.D_eff[i, :] = D_T

        # Diffusivity gradients (for advection correction)
        self.DFF = np.zeros((self.n_solutes, self.Nz))
        for i in range(self.n_solutes):
            # Central differences for d(ln(D/tort2))/dz
            D_tort = self.D_eff[i, :] / self.tort2
            for k in range(1, self.Nz - 1):
                dD_tort = (D_tort[k + 1] - D_tort[k - 1]) / (2.0 * dz[k])
                self.DFF[i, k] = dD_tort / D_tort[k]
            # Boundary: one-sided
            self.DFF[i, 0] = (D_tort[1] - D_tort[0]) / dz[0] / D_tort[0]
            self.DFF[i, -1] = (D_tort[-1] - D_tort[-2]) / dz[-1] / D_tort[-1]

        # Bioturbation coefficient (solids)
        self.D_bio = self.env.D_bio_0 * np.exp(-(z / self.env.lambda_b)**2)

        # Irrigation coefficient (solutes)
        self.alpha = self.env.alpha0 * np.exp(-(z / self.env.lambda_i)**2)

        # Burial velocities (to be computed dynamically)
        self.w_inf = 1.0e-5  # default [m/yr], will update based on solid fluxes

    def _compute_burial_velocity(self, u: np.ndarray) -> float:
        """
        Estimate burial velocity from total solid flux.

        Parameters
        ----------
        u : ndarray of shape (n_species * Nz,)
            Current state vector

        Returns
        -------
        w_inf : float
            Burial velocity [m/yr]
        """
        if self.n_solids == 0:
            return 1.0e-5  # default if no solids

        # Extract solid concentrations
        total_solid_flux = 0.0
        for j, sp_name in enumerate(self.solid_names):
            sp = self.species[sp_name]
            # Boundary flux (top of sediment)
            if sp.bc_top_type == "flux":
                total_solid_flux += sp.bc_top_value

        # Convert flux to burial velocity
        # flux = rho_sed * phi_s_inf * w_inf
        rho_sed = 2650.0  # kg/m³
        phi_s_inf = 1.0 - self.env.phi_inf

        if total_solid_flux > 1.0e-10:
            w_inf = total_solid_flux / (rho_sed * phi_s_inf)
        else:
            w_inf = 1.0e-5

        return max(w_inf, 1.0e-6)

    def _build_jacobian_sparsity(self):
        """
        Build sparse Jacobian pattern.

        Each species at depth k couples to:
        - itself at k-1, k, k+1 (transport)
        - other species at k (reactions)
        """
        n_odes = self.n_species * self.Nz
        jac_sparsity = lil_matrix((n_odes, n_odes))

        for k in range(self.Nz):
            # Each species at depth k
            for i in range(self.n_species):
                row = i * self.Nz + k

                # Couples to all species at same depth k (reactions)
                for j in range(self.n_species):
                    col = j * self.Nz + k
                    jac_sparsity[row, col] = 1

                # Couples to own species at k-1, k+1 (transport)
                if k > 0:
                    jac_sparsity[row, row - 1] = 1
                if k < self.Nz - 1:
                    jac_sparsity[row, row + 1] = 1

        self.jac_sparsity = csc_matrix(jac_sparsity)

    def _get_carbonate_speciation(self, DIC: np.ndarray, ALK: np.ndarray) -> Dict:
        """
        Compute carbonate speciation and saturation states.

        Parameters
        ----------
        DIC : ndarray of shape (Nz,)
            [mol/m³]
        ALK : ndarray of shape (Nz,)
            [mol/m³]

        Returns
        -------
        dict with keys: pH, CO3, HCO3, Omega_calcite, Omega_aragonite (all shape Nz,)
        """
        if self.carb_sys is None:
            return {}

        # Convert from mol/m³ to mol/kg-SW
        rho_sw = sw_density(self.env.T, self.env.S, self.env.P)
        DIC_kg = DIC / rho_sw
        ALK_kg = ALK / rho_sw

        pH, CO3_kg, Omega_cal, Omega_ara = self.carb_sys.solve_alkalinity(DIC_kg, ALK_kg)

        return {
            "pH": pH,
            "CO3": CO3_kg * rho_sw,  # back to mol/m³
            "Omega_calcite": Omega_cal,
            "Omega_aragonite": Omega_ara,
        }

    def _compute_reaction_rates(self, u_matrix: np.ndarray) -> np.ndarray:
        """
        Compute reaction rate matrix for all species at all depths.

        Parameters
        ----------
        u_matrix : ndarray of shape (n_species, Nz)
            State matrix: u_matrix[i, k] = species i at depth k

        Returns
        -------
        rates_matrix : ndarray of shape (n_species, Nz)
            Net reaction rate for each species at each depth [mol/m³/yr]
        """
        rates_matrix = np.zeros((self.n_species, self.Nz))

        # Get carbonate speciation if available (vectorized)
        carb_info = {}
        if "DIC" in self.species:
            j_dic = self.species_idx["DIC"]
            j_alk = self.species_idx.get("ALK", None)
            if j_alk is not None:
                dic_profile = u_matrix[j_dic, :]
                alk_profile = u_matrix[j_alk, :]
                carb_info = self._get_carbonate_speciation(dic_profile, alk_profile)

        # Loop over reactions
        for rxn in self.reaction_list:
            # Build rate law based on rate_type (vectorized over all depths)
            if rxn.rate_type == "mass_action":
                # R = k * product(c[reactant_i]^stoich_i for all reactants)
                rate = np.full(self.Nz, rxn.rate_constant, dtype=np.float64)
                for reactant_name, stoich in rxn.reactants:
                    j_reactant = self.species_idx[reactant_name]
                    c_reactant = np.maximum(u_matrix[j_reactant, :], 1e-30)
                    rate = rate * (c_reactant ** float(stoich))

            elif rxn.rate_type == "monod":
                # R = k * c[substrate] * c[monod_species] / (KM + c[monod_species])
                # Must multiply by substrate (first reactant) to match RADI.jl
                j_mon = self.species_idx[rxn.monod_species]
                c_mon = u_matrix[j_mon, :]
                rate = float(rxn.rate_constant) * c_mon / (float(rxn.half_saturation) + c_mon)
                # Multiply by first reactant (substrate) concentration
                if rxn.reactants:
                    j_sub = self.species_idx[rxn.reactants[0][0]]
                    c_sub = np.maximum(u_matrix[j_sub, :], 0.0)
                    rate *= c_sub

            elif rxn.rate_type == "inhibited_monod":
                # R = k * c[substrate] * c[monod] / (KM + c[monod]) * prod(KI/(KI+c[inh]))
                # Must multiply by substrate (first reactant) to match RADI.jl
                j_mon = self.species_idx[rxn.monod_species]
                c_mon = u_matrix[j_mon, :]
                rate = float(rxn.rate_constant) * c_mon / (float(rxn.half_saturation) + c_mon)
                # Multiply by first reactant (substrate) concentration
                if rxn.reactants:
                    j_sub = self.species_idx[rxn.reactants[0][0]]
                    c_sub = np.maximum(u_matrix[j_sub, :], 0.0)
                    rate *= c_sub

                # Apply inhibition factors (vectorized)
                for inh_name, KI in rxn.inhibitors:
                    j_inh = self.species_idx[inh_name]
                    c_inh = u_matrix[j_inh, :]
                    rate *= float(KI) / (float(KI) + c_inh)

            elif rxn.rate_type == "saturation":
                # For CaCO3 dissolution (saturation-state dependent)
                if "Omega_calcite" in carb_info and rxn.name.lower() == "calcite_dissolution":
                    # Calcite dissolution (Naviaux et al. 2019)
                    Omega = carb_info["Omega_calcite"]
                    if "calcite" in self.species_idx:
                        j_cal = self.species_idx["calcite"]
                        calcite_conc = u_matrix[j_cal, :]
                        # Vectorized saturation rate (two-regime kinetics)
                        # Regime 1 (Ω > 0.8275): k_near = 0.00632, n = 0.11
                        # Regime 2 (Ω ≤ 0.8275): k_far = 20.0, n = 4.7
                        rate = np.where(
                            Omega <= 0.8275,
                            20.0 * calcite_conc * (1.0 - Omega)**4.7,
                            np.where(
                                Omega <= 1.0,
                                0.00632 * calcite_conc * (1.0 - Omega)**0.11,
                                0.0
                            )
                        )
                    else:
                        rate = np.zeros(self.Nz)
                elif "Omega_aragonite" in carb_info and rxn.name.lower() == "aragonite_dissolution":
                    # Aragonite dissolution (Dong et al. 2019)
                    Omega = carb_info["Omega_aragonite"]
                    if "aragonite" in self.species_idx:
                        j_ara = self.species_idx["aragonite"]
                        aragonite_conc = u_matrix[j_ara, :]
                        # Vectorized saturation rate (two-regime kinetics)
                        # Regime 1 (Ω > 0.835): k_near = 0.0157, n = 0.13
                        # Regime 2 (Ω ≤ 0.835): k_far = 7.76, n = 1.46
                        rate = np.where(
                            Omega <= 0.835,
                            7.76 * aragonite_conc * (1.0 - Omega)**1.46,
                            np.where(
                                Omega <= 1.0,
                                0.0157 * aragonite_conc * (1.0 - Omega)**0.13,
                                0.0
                            )
                        )
                    else:
                        rate = np.zeros(self.Nz)
                else:
                    rate = np.zeros(self.Nz)

            else:  # custom or unknown
                rate = np.zeros(self.Nz)

            # Apply stoichiometry to all reactants and products (vectorized)
            for species_name, stoich in rxn.reactants:
                j = self.species_idx[species_name]
                rates_matrix[j, :] -= stoich * rate

            for species_name, stoich in rxn.products:
                j = self.species_idx[species_name]
                rates_matrix[j, :] += stoich * rate

        return rates_matrix

    def rhs(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the ODE system (fully vectorized).

        Parameters
        ----------
        t : float
            Time [years]
        u : ndarray of shape (n_species * Nz,)
            State vector: u[i*Nz + k] = species i at depth k

        Returns
        -------
        dudt : ndarray of shape (n_species * Nz,)
            Time derivative
        """
        dudt = np.zeros_like(u)

        # Clamp concentrations to non-negative to avoid NaN from
        # fractional exponents and singular Jacobians
        u = np.maximum(u, 0.0)

        # Reshape state to (n_species, Nz)
        u_matrix = u.reshape((self.n_species, self.Nz))

        # Tiny self-decay regularization to prevent singular Jacobian
        # when transport vanishes (e.g. deep cells with zero bioturbation).
        # This ensures every diagonal entry of the Jacobian is nonzero.
        dudt -= 1.0e-6 * u

        # Update burial velocity based on current fluxes
        w_inf = self._compute_burial_velocity(u)

        # Porewater burial velocity (shape Nz)
        u_bur = w_inf * self.env.phi_inf / self.phi

        # Solid burial velocity (shape Nz)
        w_s = w_inf * (1.0 - self.env.phi_inf) / self.phi_s

        # DBL thickness (for top boundary)
        # Use override if provided, otherwise compute from current speed
        if self.env.dbl_thickness_override is not None:
            dbl = self.env.dbl_thickness_override
        elif self.n_solutes > 0:
            dbl = dbl_thickness(self.D_eff[0, 0], self.env.U)
        else:
            dbl = 1.0e-3

        # Precompute reaction rates once for all species and depths
        rates_matrix = self._compute_reaction_rates(u_matrix)

        # Precompute grid spacings for interior cells
        dz = self.dz
        dz_p = 0.5 * (dz[1:-1] + dz[2:])      # spacing from k to k+1 for k=1..Nz-2
        dz_m = 0.5 * (dz[:-2] + dz[1:-1])      # spacing from k-1 to k for k=1..Nz-2

        # Process solutes (vectorized over depth cells within each species loop)
        for i in range(self.n_solutes):
            sp_idx = None
            for idx, sp in enumerate(self.species_list):
                if sp.name == self.solute_names[i]:
                    sp_idx = idx
                    break
            if sp_idx is None:
                continue

            sp = self.species_list[sp_idx]
            c = u_matrix[sp_idx, :]  # shape (Nz,)

            # Get this solute's diffusivity and other transport params
            D_eff = self.D_eff[i, :]
            DFF = self.DFF[i, :]
            tort2 = self.tort2

            # Get reaction rates for this species
            rxn_rates = rates_matrix[sp_idx, :]

            # Interior cells (k = 1, ..., Nz-2) - fully vectorized
            k_int = np.arange(1, self.Nz - 1)

            # Extract profiles at needed points
            c_p = c[2:]           # c[k+1] for k=1..Nz-2
            c_0 = c[1:-1]         # c[k] for k=1..Nz-2
            c_m = c[:-2]          # c[k-1] for k=1..Nz-2

            # Diffusion (centered finite differences)
            D_tort_p = D_eff[2:] / tort2[2:]
            D_tort_0 = D_eff[1:-1] / tort2[1:-1]
            D_tort_m = D_eff[:-2] / tort2[:-2]

            diff_term = (D_tort_p * (c_p - c_0) / dz_p - D_tort_m * (c_0 - c_m) / dz_m) / dz[1:-1]

            # Advection (with diffusivity gradient correction)
            flux_adv_bottom = (u_bur[2:] - D_tort_p * DFF[2:]) * c_p
            flux_adv_top = (u_bur[1:-1] - D_tort_0 * DFF[1:-1]) * c_0
            adv_term = -(flux_adv_bottom - flux_adv_top) / dz[1:-1]

            # Irrigation
            c_w = sp.bc_top_value
            irr_term = self.alpha[1:-1] * (c_w - c_0)

            dudt[sp_idx * self.Nz + 1:(sp_idx + 1) * self.Nz - 1] = diff_term + adv_term + irr_term + rxn_rates[1:-1]

            # Top cell (k = 0)
            k = 0
            dz_12_p = 0.5 * (dz[k] + dz[k + 1])

            # SWI boundary (two-resistor model matching RADI.jl)
            # Julia: D_tort2 * (2*(c2-c1) + TR*(cw-c1)) / z_res²
            # where TR = 2*z_res*tort2/dbl
            D_w = D_eff[0]                      # molecular diffusivity [m²/yr]
            D_s = D_eff[0] / tort2[0]           # sediment diffusivity [m²/yr]

            c_w = sp.bc_top_value

            # SWI flux via two-resistor: G_dbl in series with G_cell
            G_dbl = D_w / dbl                       # DBL conductance
            G_cell = D_s / (0.5 * dz[0])            # sediment half-cell conductance
            G_series = 1.0 / (1.0 / G_dbl + 1.0 / G_cell)

            # Factor 2 accounts for the half-cell distance from SWI to cell center
            # No 1/phi — consistent with Julia code and interior formula
            diff_swi = 2.0 * G_series * (c_w - c[k]) / dz[k]

            # Diffusive flux from cell 1 into cell 0 (same sign convention as interior)
            diff_int = D_s * (c[1] - c[0]) / dz_12_p / dz[0]

            # Advection
            D_s_1 = D_eff[1] / tort2[1]
            flux_adv_bottom = (u_bur[1] - D_s_1 * DFF[1]) * c[1]
            adv_term = -flux_adv_bottom / dz[0]

            # Irrigation
            irr_term = self.alpha[0] * (c_w - c[0])

            dudt[sp_idx * self.Nz + 0] = diff_swi + diff_int + adv_term + irr_term + rxn_rates[0]

            # Bottom cell (k = Nz-1)
            k = self.Nz - 1
            dz_12_m = 0.5 * (dz[k - 1] + dz[k])

            # Zero-gradient bottom: no flux from below
            D_tort_km1 = D_eff[k - 1] / tort2[k - 1]
            D_tort_k = D_eff[k] / tort2[k]
            diff_term = -D_tort_k * (c[k] - c[k - 1]) / dz_12_m / dz[k]

            # Advection: zero-gradient bottom means outgoing flux ≈ incoming flux → net ≈ 0
            adv_term = 0.0

            # Irrigation
            irr_term = self.alpha[k] * (c_w - c[k])

            dudt[sp_idx * self.Nz + k] = diff_term + adv_term + irr_term + rxn_rates[k]

        # Process solids (vectorized over depth cells within each species loop)
        for i in range(self.n_solids):
            sp_idx = None
            for idx, sp in enumerate(self.species_list):
                if sp.name == self.solid_names[i]:
                    sp_idx = idx
                    break
            if sp_idx is None:
                continue

            sp = self.species_list[sp_idx]
            c = u_matrix[sp_idx, :]  # shape (Nz,)

            # Get reaction rates for this species
            rxn_rates = rates_matrix[sp_idx, :]

            D_bio = self.D_bio

            # Interior cells (k = 1, ..., Nz-2) - fully vectorized
            k_int = np.arange(1, self.Nz - 1)

            c_p = c[2:]
            c_0 = c[1:-1]
            c_m = c[:-2]

            # Bioturbation (centered)
            bioturb = (D_bio[2:] * (c_p - c_0) / dz_p - D_bio[1:-1] * (c_0 - c_m) / dz_m) / dz[1:-1]

            # Advection with Fiadeiro-Veronis upwind weighting
            Peh_m = w_s[1:-1] * dz_m / np.maximum(D_bio[1:-1], 1e-20)
            Peh_p = w_s[2:] * dz_p / np.maximum(D_bio[2:], 1e-20)

            # Coth approximation (vectorized)
            sigma_m = np.where(
                np.abs(Peh_m) < 0.01,
                0.0,
                1.0 / np.tanh(Peh_m) - 1.0 / Peh_m
            )
            sigma_p = np.where(
                np.abs(Peh_p) < 0.01,
                0.0,
                1.0 / np.tanh(Peh_p) - 1.0 / Peh_p
            )

            # Weighted advection
            flux_adv_bottom = w_s[2:] * (c_p * (1.0 - sigma_p) + c_0 * sigma_p)
            flux_adv_top = w_s[1:-1] * (c_0 * (1.0 - sigma_m) + c_m * sigma_m)
            adv_term = -(flux_adv_bottom - flux_adv_top) / dz[1:-1]

            dudt[sp_idx * self.Nz + 1:(sp_idx + 1) * self.Nz - 1] = bioturb + adv_term + rxn_rates[1:-1]

            # Top cell (k = 0): flux boundary
            k = 0
            dz_12_p = 0.5 * (dz[k] + dz[k + 1])

            # Top flux (prescribed)
            if sp.bc_top_type == "flux":
                flux_top = sp.bc_top_value / self.phi_s[k]  # convert to concentration rate
            else:
                flux_top = 0.0

            # Bioturbative flux from cell 1 into cell 0 (same sign convention as interior)
            bioturb = D_bio[k] * (c[k + 1] - c[k]) / dz_12_p / dz[k]

            # Advection
            Peh_p = w_s[k + 1] * dz_12_p / np.maximum(D_bio[k + 1], 1e-20)
            sigma_p = 0.0 if np.abs(Peh_p) < 0.01 else 1.0 / np.tanh(Peh_p) - 1.0 / Peh_p
            flux_adv_bottom = w_s[k + 1] * (c[k + 1] * (1.0 - sigma_p) + c[k] * sigma_p)
            adv_term = -flux_adv_bottom / dz[k]

            dudt[sp_idx * self.Nz + k] = flux_top + bioturb + adv_term + rxn_rates[k]

            # Bottom cell (k = Nz-1): zero-gradient
            k = self.Nz - 1
            dz_12_m = 0.5 * (dz[k - 1] + dz[k])

            # Bioturbation: zero-gradient from below
            bioturb = -D_bio[k] * (c[k] - c[k - 1]) / dz_12_m / dz[k]

            # Advection: zero-gradient means outgoing flux = w_s * c[k]
            Peh_m = w_s[k] * dz_12_m / np.maximum(D_bio[k], 1e-20)
            sigma_m = 0.0 if np.abs(Peh_m) < 0.01 else 1.0 / np.tanh(Peh_m) - 1.0 / Peh_m
            flux_adv_top = w_s[k] * (c[k] * (1.0 - sigma_m) + c[k - 1] * sigma_m)
            flux_adv_bottom = w_s[k] * c[k]  # zero-gradient: c_ghost = c[k]
            adv_term = -(flux_adv_bottom - flux_adv_top) / dz[k]

            dudt[sp_idx * self.Nz + k] = bioturb + adv_term + rxn_rates[k]

        return dudt

    def solve(self, callback: Optional[Callable] = None) -> Dict:
        """
        Solve the model using scipy's solve_ivp with BDF method.

        Parameters
        ----------
        callback : callable, optional
            Callback function(t, progress_fraction) for progress reporting

        Returns
        -------
        dict with keys:
            - z_centers: depth array [m]
            - z_edges: cell edge depths [m]
            - species_names: list of species names
            - species_data: dict {species_name: {t: [...], profile: [...], ...}}
            - solution: scipy sol object
            - success: bool
        """
        # Build initial condition (floor at tiny positive to avoid singular Jacobian)
        u0 = np.zeros(self.n_species * self.Nz)
        for i, sp in enumerate(self.species_list):
            val = max(sp.initial_value, 1.0e-20)
            u0[i * self.Nz:(i + 1) * self.Nz] = val

        # Solve
        t_start, t_end = self.env.tspan

        # Wrap rhs to track progress and periodically report it
        import time as _time
        _last_report = [_time.monotonic()]
        _max_t = [t_start]

        def rhs_with_progress(t, u):
            if t > _max_t[0]:
                _max_t[0] = t
            now = _time.monotonic()
            # Report at most every 0.15 s to avoid slowing the solver
            if now - _last_report[0] >= 0.15:
                _last_report[0] = now
                frac = min((_max_t[0] - t_start) / (t_end - t_start), 1.0)
                if callback is not None:
                    callback(_max_t[0], frac)
            return self.rhs(t, u)

        # Compute output time points if requested for transient mode
        t_eval = None
        if self.env.t_eval_points is not None and self.env.mode == "transient":
            t_eval = np.linspace(self.env.tspan[0], self.env.tspan[1], self.env.t_eval_points)

        # Try Radau first (most robust for stiff + near-singular),
        # fall back to BDF, then LSODA if both fail.
        methods_to_try = [
            ("Radau", dict(jac_sparsity=self.jac_sparsity)),
            ("BDF",   dict(jac_sparsity=self.jac_sparsity)),
            ("Radau", dict()),  # dense Jacobian fallback
            ("LSODA", dict()),
        ]

        sol = None
        attempt = 0
        n_methods = len(methods_to_try) + 1  # +1 for RK45 fallback
        for method, extra_opts in methods_to_try:
            attempt += 1
            # Reset progress tracker for each attempt
            _max_t[0] = t_start
            _last_report[0] = _time.monotonic()
            try:
                sol = solve_ivp(
                    rhs_with_progress,
                    self.env.tspan,
                    u0,
                    method=method,
                    rtol=1.0e-3,
                    atol=1.0e-6,
                    dense_output=True,
                    first_step=1.0e-4,
                    max_step=max(1000.0, (self.env.tspan[1] - self.env.tspan[0]) / 10.0),
                    t_eval=t_eval,
                    **extra_opts,
                )
                if sol.success:
                    break
            except (RuntimeError, ValueError):
                continue

        if sol is None:
            # Last resort: explicit method (not ideal for stiff, but won't crash)
            _max_t[0] = t_start
            sol = solve_ivp(
                rhs_with_progress, self.env.tspan, u0,
                method="RK45", rtol=1e-3, atol=1e-6,
                dense_output=True, t_eval=t_eval,
                max_step=10.0,
            )

        # Final progress report
        if callback is not None:
            callback(t_end, 1.0)

        # Clamp solution to non-negative (ODE solver can overshoot to small negatives)
        if sol is not None and sol.y is not None:
            sol.y = np.maximum(sol.y, 0.0)

        return self._format_results(sol)

    def _format_results(self, sol) -> Dict:
        """Format scipy solution into user-friendly dict."""
        results = {
            "z_centers": self.z_centers,
            "z_edges": self.z_edges,
            "dz": self.dz,
            "species_names": self.species_names,
            "species_data": {},
            "solution": sol,
            "success": sol.status == 0,
            "message": sol.message,
        }

        # Extract final profiles
        u_final = sol.y[:, -1]
        for i, sp_name in enumerate(self.species_names):
            profile = u_final[i * self.Nz:(i + 1) * self.Nz]
            species_data = {
                "final_profile": profile,
                "t_initial": sol.t[0],
                "t_final": sol.t[-1],
            }
            
            # Include all time point profiles if t_eval was used (transient with t_eval_points)
            if self.env.t_eval_points is not None and self.env.mode == "transient":
                # sol.y has shape (n_species * Nz, n_time_points)
                all_profiles = []
                for t_idx in range(sol.y.shape[1]):
                    profile_t = sol.y[i * self.Nz:(i + 1) * self.Nz, t_idx]
                    all_profiles.append(profile_t)
                species_data["time_series_profiles"] = all_profiles
                species_data["time_points"] = sol.t
            
            results["species_data"][sp_name] = species_data

        return results

    def estimate_runtime(self) -> float:
        """
        Estimate wall-clock runtime for this configuration.

        Returns
        -------
        seconds : float
            Rough estimate [seconds]
        """
        n_odes = self.n_species * self.Nz
        t_range = self.env.tspan[1] - self.env.tspan[0]

        # Empirical: ~0.001 sec per ODE per 1000 model-years
        estimate = n_odes * 0.001 * (t_range / 1000.0)

        return max(1.0, estimate)


# ============================================================================
# BENCHMARK: W2 Station Configuration
# ============================================================================

def create_w2_benchmark() -> RADIModel:
    """
    Create a pre-configured RADI model for the W2 station.

    This represents a typical deep ocean station with:
    - T=1.4°C, S=34.69, P=4310 dbar
    - 18 species (O2, DIC, ALK, NO3, SO4, NH4, H2S, Fe2, Mn2, CH4, Ca, etc.)
    - Full redox cascade (O2 -> NO3 -> Mn -> Fe -> SO4)
    - CaCO3 (calcite, aragonite) dissolution
    - Organic matter degradation pathways

    Returns
    -------
    model : RADIModel
    """

    # Species definitions
    species_list = [
        # Solutes
        Species(name="O2", long_name="Dissolved Oxygen", species_type="solute",
                D0=0.031558, D_T_coeff=0.001428,
                bc_top_type="concentration", bc_top_value=250.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=50.0, unit="μmol/kg", color="#0066FF"),

        Species(name="DIC", long_name="Dissolved Inorganic Carbon", species_type="solute",
                D0=0.015179, D_T_coeff=0.000795,
                bc_top_type="concentration", bc_top_value=2300.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=2350.0, unit="μmol/kg", color="#FF6600"),

        Species(name="ALK", long_name="Alkalinity", species_type="solute",
                D0=0.015179, D_T_coeff=0.000795,
                bc_top_type="concentration", bc_top_value=2400.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=2410.0, unit="μmol/kg", color="#FF0066"),

        Species(name="NO3", long_name="Nitrate", species_type="solute",
                D0=0.030863, D_T_coeff=0.001153,
                bc_top_type="concentration", bc_top_value=10.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=5.0, unit="μmol/kg", color="#00FF66"),

        Species(name="SO4", long_name="Sulfate", species_type="solute",
                D0=0.015779, D_T_coeff=0.000712,
                bc_top_type="concentration", bc_top_value=28000.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=27500.0, unit="μmol/kg", color="#FFFF00"),

        Species(name="NH4", long_name="Ammonium", species_type="solute",
                D0=0.030926, D_T_coeff=0.001225,
                bc_top_type="concentration", bc_top_value=0.1,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=1.0, unit="μmol/kg", color="#FF00FF"),

        Species(name="H2S", long_name="Hydrogen Sulfide", species_type="solute",
                D0=0.028938, D_T_coeff=0.001314,
                bc_top_type="concentration", bc_top_value=0.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=0.0, unit="μmol/kg", color="#00FFFF"),

        Species(name="Fe2", long_name="Dissolved Iron(II)", species_type="solute",
                D0=0.010761, D_T_coeff=0.000466,
                bc_top_type="concentration", bc_top_value=0.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=0.0, unit="μmol/kg", color="#FF6666"),

        Species(name="Mn2", long_name="Dissolved Manganese(II)", species_type="solute",
                D0=0.009625, D_T_coeff=0.000481,
                bc_top_type="concentration", bc_top_value=0.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=0.0, unit="μmol/kg", color="#FF9900"),

        Species(name="CH4", long_name="Methane", species_type="solute",
                D0=0.041629, D_T_coeff=0.000764,
                bc_top_type="concentration", bc_top_value=0.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=0.0, unit="μmol/kg", color="#9900FF"),

        Species(name="Ca", long_name="Calcium", species_type="solute",
                D0=0.011771, D_T_coeff=0.000529,
                bc_top_type="concentration", bc_top_value=10280.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=10250.0, unit="μmol/kg", color="#CCCCCC"),

        # Solids
        Species(name="OM", long_name="Organic Matter", species_type="solid",
                D0=0.0, D_T_coeff=0.0,
                bc_top_type="flux", bc_top_value=10.0,  # 10 mol/m²/yr
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=0.5, unit="wt%", color="#664422"),

        Species(name="calcite", long_name="Calcite", species_type="solid",
                D0=0.0, D_T_coeff=0.0,
                bc_top_type="flux", bc_top_value=2.0,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=0.1, unit="wt%", color="#FFFFFF"),

        Species(name="aragonite", long_name="Aragonite", species_type="solid",
                D0=0.0, D_T_coeff=0.0,
                bc_top_type="flux", bc_top_value=0.5,
                bc_bot_type="gradient", bc_bot_value=0.0,
                initial_value=0.02, unit="wt%", color="#EEEEEE"),
    ]

    # Reactions
    reaction_list = [
        # Aerobic respiration (OM + O2 -> DIC + NH4)
        Reaction(name="aerobic_respiration", rate_type="inhibited_monod",
                 rate_constant=10.0,  # [1/yr]
                 reactants=[("OM", 1.0), ("O2", 1.0)],
                 products=[("DIC", 1.0), ("NH4", 0.16)],
                 monod_species="OM", half_saturation=1.0,
                 inhibitors=[("O2", 100.0)]),

        # Denitrification (OM + NO3 -> DIC + N2, inhibited by O2)
        Reaction(name="denitrification", rate_type="inhibited_monod",
                 rate_constant=8.0,
                 reactants=[("OM", 1.0), ("NO3", 0.8)],
                 products=[("DIC", 1.0), ("NH4", 0.16)],
                 monod_species="OM", half_saturation=1.0,
                 inhibitors=[("O2", 50.0)]),

        # Mn reduction
        Reaction(name="mn_reduction", rate_type="inhibited_monod",
                 rate_constant=5.0,
                 reactants=[("OM", 1.0)],
                 products=[("DIC", 1.0), ("Mn2", 1.0), ("NH4", 0.16)],
                 monod_species="OM", half_saturation=1.0,
                 inhibitors=[("O2", 30.0), ("NO3", 20.0)]),

        # Fe reduction
        Reaction(name="fe_reduction", rate_type="inhibited_monod",
                 rate_constant=5.0,
                 reactants=[("OM", 1.0)],
                 products=[("DIC", 1.0), ("Fe2", 1.0), ("NH4", 0.16)],
                 monod_species="OM", half_saturation=1.0,
                 inhibitors=[("O2", 20.0), ("NO3", 15.0), ("Mn2", 10.0)]),

        # Sulfate reduction
        Reaction(name="sulfate_reduction", rate_type="inhibited_monod",
                 rate_constant=3.0,
                 reactants=[("OM", 1.0), ("SO4", 0.5)],
                 products=[("DIC", 1.0), ("H2S", 0.5), ("NH4", 0.16)],
                 monod_species="OM", half_saturation=1.0,
                 inhibitors=[("O2", 10.0), ("NO3", 8.0), ("Mn2", 5.0), ("Fe2", 5.0)]),

        # Methanogenesis
        Reaction(name="methanogenesis", rate_type="inhibited_monod",
                 rate_constant=1.0,
                 reactants=[("OM", 1.0)],
                 products=[("DIC", 0.5), ("CH4", 0.5), ("NH4", 0.16)],
                 monod_species="OM", half_saturation=1.0,
                 inhibitors=[("O2", 5.0), ("NO3", 4.0), ("Mn2", 2.0),
                             ("Fe2", 2.0), ("SO4", 2.0)]),

        # Calcite dissolution (Omega-dependent)
        Reaction(name="calcite_dissolution", rate_type="saturation",
                 rate_constant=1.0,  # placeholder
                 reactants=[("calcite", 1.0)],
                 products=[("Ca", 1.0), ("DIC", 1.0)]),

        # Aragonite dissolution
        Reaction(name="aragonite_dissolution", rate_type="saturation",
                 rate_constant=1.0,
                 reactants=[("aragonite", 1.0)],
                 products=[("Ca", 1.0), ("DIC", 1.0)]),
    ]

    # Environment
    env = Environment(
        T=1.4, S=34.69, P=4310.0, depth=4400.0, U=0.01,
        phi0=0.8, phi_inf=0.65, beta=2.0,
        D_bio_0=1.0e-3, lambda_b=0.05,
        lambda_i=0.02, alpha0=10.0,
        depth_sed=0.5, Nz=50,
        dz_top=1.0e-4, dz_bot=1.0e-2,
        tspan=(0.0, 1000.0),
        mode="transient",
    )

    return RADIModel(species_list, reaction_list, env)


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    print("Creating W2 benchmark model...")
    model = create_w2_benchmark()

    print(f"Model setup: {model.n_species} species, {model.Nz} depth cells")
    print(f"Solutes: {model.solute_names}")
    print(f"Solids: {model.solid_names}")
    print(f"Reactions: {len(model.reaction_list)}")
    print(f"Estimated runtime: {model.estimate_runtime():.1f} seconds")

    print("\nSolving...")
    results = model.solve()

    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Time range: {results['solution'].t[0]:.1f} to {results['solution'].t[-1]:.1f} years")

    # Print final profiles for a few key species
    for sp_name in ["O2", "DIC", "NO3", "SO4", "H2S", "CH4"]:
        if sp_name in results["species_data"]:
            profile = results["species_data"][sp_name]["final_profile"]
            print(f"{sp_name}: min={profile.min():.2e}, max={profile.max():.2e}")
