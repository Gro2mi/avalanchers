"""Analytical Coulomb slider on the energy-line parabola.

For increasing horizontal x, B'=p, J=sqrt(1+p^2), ds=J dx,
kappa=B''/J^3, and N/m=g/J+kappa*v^2. With w=v^2/(2g):

    dw/dx + 2*mu*B''/(1+p^2)*w = -p-mu.

The integrating factor is exp(2*mu*(atan(p)-atan(p_start))).
The solution below evaluates the resulting definite integral by Gauss-Legendre
quadrature, not by stepping the slider equations. Runout is the first zero of w
after acceleration. On the flat continuation, dw/dx=-mu.

The reference is exact for a point slider in contact with this concave bed,
without internal pressure, turbulent drag, or entrainment. Comparison with a
finite particle release is a regression approximation, not an exact COM law.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


@dataclass(frozen=True)
class ParabolicBed:
    vertex_x: float
    bottom_z: float
    second_derivative: float
    max_x: float

    def __post_init__(self):
        if not np.isfinite([self.vertex_x, self.bottom_z, self.second_derivative, self.max_x]).all():
            raise ValueError("Parabolic bed parameters must be finite.")
        if self.second_derivative <= 0 or self.max_x < self.vertex_x:
            raise ValueError("A concave parabola and a domain reaching its vertex are required.")

    def elevation(self, x):
        offset = np.minimum(np.asarray(x, dtype=float) - self.vertex_x, 0.0)
        return self.bottom_z + 0.5 * self.second_derivative * offset**2

    def slope(self, x):
        return self.second_derivative * np.minimum(np.asarray(x, dtype=float) - self.vertex_x, 0.0)

    def surface_distance(self, start_x, end_x):
        """Arc length on the parabola plus any flat continuation."""
        def primitive(x):
            p = self.second_derivative * (min(x, self.vertex_x) - self.vertex_x)
            return (p * np.sqrt(1.0 + p*p) + np.arcsinh(p)) / (2.0 * self.second_derivative)

        return float(primitive(end_x) - primitive(start_x)
                     + max(end_x - self.vertex_x, 0.0) - max(start_x - self.vertex_x, 0.0))


def fit_energy_line_bed(dem, tolerance=1e-3):
    """Recover and verify the upstream parabola and horizontal continuation."""
    header = dem["header"]
    raster = np.asarray(dem["rasterData"], dtype=float)
    if not np.isfinite(raster).all():
        raise ValueError("Curvature verification requires a finite DEM.")
    profile = raster[raster.shape[0] // 2]
    if np.max(np.abs(raster - profile)) > tolerance:
        raise ValueError("Curvature verification requires a bed independent of y.")
    x = header["xmin"] + (np.arange(len(profile)) + 0.5) * header["cellsize"]
    upstream = profile > profile.min() + tolerance
    if upstream.sum() < 3:
        raise ValueError("At least three upstream parabola samples are required.")
    a, b, c = np.polyfit(x[upstream] - x[0], profile[upstream], 2)
    if a <= 0:
        raise ValueError("Curvature verification requires a concave parabola.")
    bed = ParabolicBed(float(x[0] - b / (2*a)), float(c - b*b / (4*a)),
                       float(2*a), float(x[-1]))
    residual = float(np.max(np.abs(bed.elevation(x) - profile)))
    if residual > tolerance:
        raise ValueError(f"DEM is not a parabola with flat continuation: residual {residual:g} m.")
    return bed, residual


def kinetic_height(bed, start_x, x, mu, curvature=True, initial_height=0.0):
    """Integrating-factor solution; its continuation can be negative after stopping."""
    if not np.isfinite([start_x, mu, initial_height]).all() or mu < 0 or initial_height < 0:
        raise ValueError("Initial position, nonnegative friction and kinetic height must be finite.")
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or np.any(x < start_x):
        raise ValueError("Reference positions must be finite and downstream of the start.")
    end = np.minimum(x, bed.vertex_x)
    begin = min(start_x, bed.vertex_x)
    width = end - begin
    nodes, weights = leggauss(64)
    points = begin + 0.5 * width[..., None] * (nodes + 1.0)
    p = bed.slope(points)
    p_start = bed.slope(start_x)
    phase = 2.0 * mu * (np.arctan(p) - np.arctan(p_start)) if curvature else np.zeros_like(p)
    integral = 0.5 * width * np.sum(weights * np.exp(phase) * (-p-mu), axis=-1)
    end_phase = 2.0 * mu * (np.arctan(bed.slope(end)) - np.arctan(p_start)) if curvature else 0.0
    w = (initial_height + integral) * np.exp(-end_phase)
    flat_distance = np.maximum(x-bed.vertex_x, 0.0) - max(start_x-bed.vertex_x, 0.0)
    return w - mu * flat_distance


def analytical_runout(bed, start_x, mu, curvature=True):
    """First nontrivial stopping point for release from rest, within the DEM."""
    if not np.isfinite([mu, start_x]).all() or mu <= 0 or bed.slope(start_x) >= -mu:
        raise ValueError("Release from rest requires mu > 0 and an initial slope steeper than mu.")
    peak_x = bed.vertex_x - mu / bed.second_derivative
    w_bottom = float(kinetic_height(bed, start_x, bed.vertex_x, mu, curvature))
    if w_bottom > 0:
        stop_x = bed.vertex_x + w_bottom / mu
    else:
        stop_x = brentq(lambda x: float(kinetic_height(bed, start_x, x, mu, curvature)),
                        peak_x, bed.vertex_x, xtol=1e-9)
    if stop_x > bed.max_x:
        raise ValueError("Analytical runout exceeds the available DEM.")
    return {
        "stop_x": float(stop_x),
        "horizontal_runout": float(stop_x-start_x),
        "surface_runout": bed.surface_distance(start_x, stop_x),
        "stop_z": float(bed.elevation(stop_x)),
    }


def compare_profile(bed, profile, mu, curvature=True):
    if len(profile["x"]) < 2 or not all(np.isfinite(profile[k]).all() for k in ("x", "z", "u2")):
        raise ValueError("At least two finite trajectory samples are required.")
    start_x = float(profile["x"][0])
    reference = analytical_runout(bed, start_x, mu, curvature)
    reference_x = np.clip(profile["x"], start_x, reference["stop_x"])
    w = kinetic_height(bed, start_x, reference_x, mu, curvature)
    energy = bed.elevation(reference_x) + np.maximum(w, 0.0)
    simulated_energy = profile["z"] + profile["u2"] / (2.0 * 9.81)
    simulated_runout = float(profile["x"][-1]-start_x)
    return {
        **reference,
        "simulated_runout": simulated_runout,
        "runout_error": simulated_runout-reference["horizontal_runout"],
        "energy_rmse": float(np.sqrt(np.mean((simulated_energy-energy)**2))),
        "reference_energy": energy,
        "simulated_energy": simulated_energy,
    }
