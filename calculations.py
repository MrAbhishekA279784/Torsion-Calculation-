"""ShaftLab - All torsion calculations in one file.

Linear-elastic (Saint-Venant) torsion theory for circular shafts:

    Polar moment of inertia   J = pi*D^4/32          (solid)
                              J = pi*(Do^4-Di^4)/32  (hollow)
    Max shear stress          tau_max = T*R_o / J    (linear with radius)
    Angle of twist            theta   = T*L / (J*G)
    Factor of safety          FOS     = tau_allow / tau_max
    Torsional stiffness       kt      = G*J / L

Units:
    The engine works internally in SI (m, N, m^4, Pa, rad) and stores
    human-friendly display units behind the dataclass fields
    (mm, N.m, mm^4, MPa, MPa, deg, GPa).
"""

import math
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Material database (copied from the removed core/materials.py)
# ---------------------------------------------------------------------------
MATERIALS = {
    "Structural Steel (ASTM A36)": {
        "G": 79.3, "tau_allow": 145.0, "rho": 7850.0,
        "desc": "Standard mild carbon steel for machine shafts.",
    },
    "Alloy Steel (AISI 4140)": {
        "G": 80.0, "tau_allow": 210.0, "rho": 7850.0,
        "desc": "High-strength chrome-moly steel for driveshafts.",
    },
    "Stainless Steel (AISI 304)": {
        "G": 77.2, "tau_allow": 125.0, "rho": 8000.0,
        "desc": "Corrosion-resistant austenitic stainless steel.",
    },
    "Aluminium Alloy (6061-T6)": {
        "G": 26.0, "tau_allow": 95.0, "rho": 2700.0,
        "desc": "Lightweight aircraft-grade aluminium alloy.",
    },
    "Titanium Alloy (Ti-6Al-4V)": {
        "G": 42.0, "tau_allow": 310.0, "rho": 4430.0,
        "desc": "High-performance aerospace alloy.",
    },
    "Copper Alloy (C11000)": {
        "G": 44.0, "tau_allow": 70.0, "rho": 8940.0,
        "desc": "Highly conductive copper for rotor shafts.",
    },
    "Brass (C36000)": {
        "G": 39.0, "tau_allow": 85.0, "rho": 8500.0,
        "desc": "Free-cutting copper-zinc alloy.",
    },
    "Cast Iron (Grey Class 40)": {
        "G": 40.0, "tau_allow": 60.0, "rho": 7200.0,
        "desc": "Vibration-damping grey cast iron.",
    },
}

MATERIAL_NAMES = list(MATERIALS.keys()) + ["Custom Material"]

CUSTOM_MATERIAL = {"G": 80.0, "tau_allow": 150.0, "rho": 7850.0}


def get_material(name: str) -> dict:
    """Return the property dictionary for a material name."""
    return MATERIALS.get(name, CUSTOM_MATERIAL)


# ---------------------------------------------------------------------------
# Input / result data classes
# ---------------------------------------------------------------------------
@dataclass
class TorsionInputs:
    """User inputs. Display units, always the latest values from the UI."""
    shaft_type: str = "Solid"            # "Solid" | "Hollow"
    outer_diameter_mm: float = 50.0      # Do in mm
    inner_diameter_mm: float = 0.0       # Di in mm (0 for solid)
    length_mm: float = 1000.0            # L in mm
    applied_torque_nm: float = 2000.0    # T in N*m
    shear_modulus_gpa: float = 80.0      # G in GPa
    allowable_shear_stress_mpa: float = 200.0   # tau_allow in MPa
    max_twist_deg: float = 3.0           # theta_allow in degrees
    density_kg_m3: float = 7850.0        # rho in kg/m^3 (from material)
    material: str = "Alloy Steel (AISI 4140)"


@dataclass
class TorsionResults:
    """Complete torsion analysis results (display units)."""
    radius_mm: float                     # R = Do/2
    inner_radius_mm: float               # Ri (0 for solid)
    polar_moment_mm4: float              # J in mm^4
    max_shear_stress_mpa: float          # tau_max at R
    min_shear_stress_mpa: float          # tau at Ri (0 for solid)
    angle_of_twist_rad: float
    angle_of_twist_deg: float
    factor_of_safety: float
    torsional_stiffness_nm_rad: float    # kt = G*J/L
    polar_section_modulus_mm3: float     # Zp = J/R
    strain_energy_j: float               # U = 0.5*T*theta
    volume_mm3: float
    mass_kg: float
    is_stress_safe: bool
    is_twist_safe: bool
    is_safe: bool
    status: str                          # "SAFE" | "MARGINAL" | "UNSAFE"
    status_detail: str
    stress_margin_pct: float             # (tau_allow - tau_max)/tau_allow * 100
    twist_margin_pct: float              # (theta_allow - theta)/theta_allow * 100


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_inputs(inp: TorsionInputs) -> tuple:
    """Validate inputs with physically meaningful constraints.

    Returns (is_valid, message).
    """
    if inp.outer_diameter_mm <= 0:
        return False, "Outer diameter must be greater than zero."
    if inp.length_mm <= 0:
        return False, "Shaft length must be greater than zero."
    if inp.applied_torque_nm < 0:
        return False, "Applied torque cannot be negative."
    if inp.shear_modulus_gpa <= 0:
        return False, "Shear modulus G must be greater than zero."
    if inp.allowable_shear_stress_mpa <= 0:
        return False, "Allowable shear stress must be greater than zero."
    if inp.max_twist_deg <= 0:
        return False, "Maximum permissible twist must be greater than zero."

    if inp.shaft_type == "Hollow":
        if inp.inner_diameter_mm <= 0:
            return False, "For a hollow shaft the inner diameter must be greater than zero."
        if inp.inner_diameter_mm >= inp.outer_diameter_mm:
            return False, (
                f"Inner diameter ({inp.inner_diameter_mm:.1f} mm) must be smaller "
                f"than outer diameter ({inp.outer_diameter_mm:.1f} mm)."
            )
        wall = (inp.outer_diameter_mm - inp.inner_diameter_mm) / 2.0
        if wall < 1.0:
            return False, (
                f"Wall thickness ({wall:.2f} mm) is below the practical limit of 1 mm. "
                "Reduce the inner diameter or increase the outer diameter."
            )
    return True, "Valid inputs"


# ---------------------------------------------------------------------------
# Core torsion engine
# ---------------------------------------------------------------------------
def _polar_moment_m4(inp: TorsionInputs) -> float:
    """Return polar moment of inertia J in m^4 (SI)."""
    do = inp.outer_diameter_mm / 1000.0        # mm -> m
    di = (inp.inner_diameter_mm / 1000.0) if inp.shaft_type == "Hollow" else 0.0
    if inp.shaft_type == "Hollow":
        return math.pi * (do ** 4 - di ** 4) / 32.0
    return math.pi * do ** 4 / 32.0


def calculate_torsion(inp: TorsionInputs) -> TorsionResults:
    """Solve elastic torsion for the *latest* inputs. No caching anywhere.

    All intermediate math in SI units; results converted to display units.
    """
    t_nm = inp.applied_torque_nm               # N*m  (SI torque)
    g_pa = inp.shear_modulus_gpa * 1e9         # GPa -> Pa
    length_m = inp.length_mm / 1000.0          # mm -> m
    tau_allow_mpa = inp.allowable_shear_stress_mpa    # MPa (display)

    r_outer = inp.outer_diameter_mm / 2.0      # mm
    r_inner = (inp.inner_diameter_mm / 2.0) if inp.shaft_type == "Hollow" else 0.0

    j_m4 = _polar_moment_m4(inp)               # SI
    j_mm4 = j_m4 * 1e12                        # m^4 -> mm^4

    # tau_max = T*R/J   (Pa). For T=0 the stress is zero.
    r_outer_m = r_outer / 1000.0
    tau_max_mpa = (t_nm * r_outer_m / j_m4 / 1e6) if t_nm > 0 else 0.0   # Pa -> MPa
    tau_min_mpa = (
        (t_nm * (r_inner / 1000.0) / j_m4 / 1e6)
        if (t_nm > 0 and r_inner > 0) else 0.0
    )  # Pa -> MPa

    # theta = T*L/(J*G)  (rad)
    theta_rad = (t_nm * length_m / (j_m4 * g_pa)) if t_nm > 0 else 0.0
    theta_deg = math.degrees(theta_rad)

    fos = (tau_allow_mpa / tau_max_mpa) if tau_max_mpa > 0 else float("inf")
    kt = (g_pa * j_m4 / length_m) if length_m > 0 else 0.0  # N*m/rad
    zp_mm3 = j_mm4 / r_outer                               # mm^3
    strain_energy = 0.5 * t_nm * theta_rad                 # J
    volume_mm3 = math.pi / 4.0 * (
        inp.outer_diameter_mm ** 2
        - (inp.inner_diameter_mm ** 2 if inp.shaft_type == "Hollow" else 0.0)
    ) * inp.length_mm
    mass_kg = (volume_mm3 * 1e-9) * inp.density_kg_m3

    eps = 1e-9
    is_stress_safe = tau_max_mpa <= tau_allow_mpa * (1 + eps)
    is_twist_safe = theta_deg <= inp.max_twist_deg + eps
    is_safe = is_stress_safe and is_twist_safe
    marginal = is_safe and (not math.isinf(fos) and fos < 1.5)
    status = "UNSAFE" if not is_safe else ("MARGINAL" if marginal else "SAFE")

    if not is_stress_safe:
        status_detail = (
            f"Shear stress exceeds the allowable limit by "
            f"{(tau_max_mpa / tau_allow_mpa - 1) * 100:.1f}% - increase the diameter "
            "or use a stronger material."
        )
    elif not is_twist_safe:
        status_detail = (
            f"Twist exceeds the permissible {inp.max_twist_deg:.2f} deg limit "
            "by "
            f"{(theta_deg / inp.max_twist_deg - 1) * 100:.1f}% - stiffen the shaft "
            "(larger J, shorter L)."
        )
    else:
        status_detail = (
            f"tau_max = {tau_max_mpa:.1f} MPa and theta = {theta_deg:.2f} deg are "
            "within the permissible limits."
        )

    stress_margin_pct = (1 - tau_max_mpa / tau_allow_mpa) * 100.0
    twist_margin_pct = (1 - theta_deg / inp.max_twist_deg) * 100.0

    return TorsionResults(
        radius_mm=r_outer,
        inner_radius_mm=r_inner,
        polar_moment_mm4=j_mm4,
        max_shear_stress_mpa=tau_max_mpa,
        min_shear_stress_mpa=tau_min_mpa,
        angle_of_twist_rad=theta_rad,
        angle_of_twist_deg=theta_deg,
        factor_of_safety=fos,
        torsional_stiffness_nm_rad=kt,
        polar_section_modulus_mm3=zp_mm3,
        strain_energy_j=strain_energy,
        volume_mm3=volume_mm3,
        mass_kg=mass_kg,
        is_stress_safe=is_stress_safe,
        is_twist_safe=is_twist_safe,
        is_safe=is_safe,
        status=status,
        status_detail=status_detail,
        stress_margin_pct=stress_margin_pct,
        twist_margin_pct=twist_margin_pct,
    )


# ---------------------------------------------------------------------------
# Graph data generators (used by the dynamic charts)
# ---------------------------------------------------------------------------
def shear_stress_profile(inp: TorsionInputs, results: TorsionResults, n_points: int = 120):
    """Linear elastic shear stress distribution tau(r) = T*r/J.

    Solid shaft: r from 0 (center, tau=0) to R (surface, tau_max).
    Hollow shaft: r from Ri to R (no material in the bore).

    Returns (r_mm np.array, tau_mpa np.array).
    """
    r_start = results.inner_radius_mm if inp.shaft_type == "Hollow" else 0.0
    r_mm = np.linspace(r_start, results.radius_mm, n_points)
    if results.max_shear_stress_mpa <= 0:
        tau_mpa = np.zeros_like(r_mm)
    else:
        # Linear distribution tau(r) = T*r/J. For hollow shafts there is one key
        # identity: the engine computes tau_min = T*Ri/J and tau_max = T*Ro/J, so
        # tau(r) = (r/Ro)*tau_max is exact for BOTH shaft types (no interp needed).
        tau_mpa = (r_mm / results.radius_mm) * results.max_shear_stress_mpa
    return r_mm, tau_mpa


def torque_twist_curve(inp: TorsionInputs, results: TorsionResults, n_points: int = 120):
    """Torque vs angle of twist: theta = T*L/(G*J) -> T = kt*theta (linear).

    Plots T up to ~1.6x the larger of the current operating point and
    the allowable-twist line so relative changes stay visible.

    Returns (theta_deg np.array, torque_nm np.array, kt N*m/rad).
    """
    kt = results.torsional_stiffness_nm_rad
    theta_allow_rad = math.radians(inp.max_twist_deg)
    theta_current = results.angle_of_twist_rad
    theta_max = max(1.6 * theta_allow_rad, 2.0 * theta_current, 1e-6)
    theta_linspace = np.linspace(0.0, theta_max, n_points)
    if kt > 0:
        torque = kt * theta_linspace
    else:
        torque = np.zeros_like(theta_linspace)
    return np.degrees(theta_linspace), torque, kt
