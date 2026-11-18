"""Coulomb point slider with detachment and landing on a profile z = B(x).

Reference of the double-bump test. The slider moves along +x, starts at rest
and obeys the same rules as the particles of the 3D model.

Sliding: with B' = dB/dx, J = sqrt(1 + B'^2), ds = J dx, the normal
curvature kappa = B''/J^3 and the bed-normal load N/m = g/J + kappa v^2,
Coulomb friction mu N gives for the velocity height w = v^2 / (2g)

    dw/dx + 2 mu B'' / (1 + B'^2) w = -B' - mu .

Because B''/(1 + B'^2) = d(atan B')/dx, the integrating factor is
exp(2 mu phi) with phi(x) = atan B'(x) - atan B'(x_0) on any bed, so

    w(x) = exp(-2 mu phi(x)) [ w_0 + int_{x_0}^{x} exp(2 mu phi) (-B' - mu) dx' ] .

The integral is evaluated by quadrature on a fine grid (the curvature test
uses the same solution on a parabola). The slider stops where w = 0. The
travel time is t = int J / sqrt(2 g w) dx, with the integrable singularity at
a start from rest taken exactly over the first interval.

Take-off: where the normal load turns negative, N/m < 0, i.e.

    1 + 2 w B'' / (1 + B'^2) < 0 .

This is the limit dt -> 0 of the model's test, which detaches a particle
where one free-flight step from the bed would end above it: the bed then
falls away faster than gravity can pull the particle down.

the slider leaves the bed with the velocity v (1, B') / J and flies. Without
air drag the flight is the parabola x = x_d + v_x t,
z = z_d + v_z t - g t^2 / 2; with the quadratic drag of the model,
du/dt = -g e_z - g |u| u / v_t^2 (terminal velocity v_t), it is integrated
numerically.

Landing: at the first point below the bed. The velocity component into the
bed u_n = max(-u . n, 0), with n = (-B', 1) / J, is removed and Coulomb
friction removes mu u_n of the tangential speed u_t = u . (1, B') / J:

    v = max(u_t - mu u_n, 0) .

Then the slider slides on. The reference is exact for a point slider; the
simulated flow has a finite extent and a pressure gradient, so its center of
mass is only expected to follow it approximately.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G = 9.81


@dataclass
class SliderPath:
    """Path of the point slider: samples of time, x and z, whether the
    slider flies, and the take-off / landing / stop events."""
    t: np.ndarray
    x: np.ndarray
    z: np.ndarray
    flying: np.ndarray
    events: list = field(default_factory=list)

    def position(self, t):
        """(x, z) of the slider at time t (last position after the stop)."""
        return float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.z))

    def event_x(self, kind, index=0):
        """x of the index-th event of a kind ('takeoff', 'landing', 'stop')."""
        matches = [e for e in self.events if e["kind"] == kind]
        return matches[index]["x"] if len(matches) > index else float("nan")


def _slide(bed, slope, curvature, mu, x_start, w_start, t_start, x_max, dx):
    """Slides from x_start with the velocity height w_start. Returns the
    sample arrays and how the slide ended: ('takeoff' | 'stop' | 'end', x, w)."""
    x = np.arange(x_start, x_max, dx)
    if len(x) < 2:
        return (np.array([x_start]), np.array([w_start]), np.array([t_start])), ("end", x_start, w_start)
    p, q = slope(x), curvature(x)
    phi = np.arctan(p) - np.arctan(p[0])
    integrand = np.exp(2.0 * mu * phi) * (-p - mu)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dx)])
    w = np.exp(-2.0 * mu * phi) * (w_start + integral)
    normal = 1.0 + 2.0 * w * q / (1.0 + p * p)

    # first sample (after the start) where the slider stops or takes off
    stop = np.nonzero(w[1:] <= 0.0)[0]
    detach = np.nonzero(normal[1:] < 0.0)[0]
    k_stop = stop[0] + 1 if len(stop) else len(x)
    k_detach = detach[0] + 1 if len(detach) else len(x)
    end = min(k_stop, k_detach)
    if end == len(x):
        kind, k = "end", len(x) - 1
    elif k_detach < k_stop:
        kind, k = "takeoff", k_detach
    else:
        kind, k = "stop", k_stop
    if kind == "end":
        x_end, w_end = x[k], w[k]
    else:
        # linear interpolation of the root within the last interval
        values = w if kind == "stop" else normal
        frac = values[k - 1] / (values[k - 1] - values[k])
        x_end = x[k - 1] + frac * dx
        w_end = 0.0 if kind == "stop" else w[k - 1] + frac * (w[k] - w[k - 1])
    xs = np.append(x[:k], x_end)
    ws = np.maximum(np.append(w[:k], w_end), 0.0)

    # travel time: midpoint rule, the first interval exact for a start from rest
    J = np.sqrt(1.0 + slope(xs) ** 2)
    h = np.diff(xs)
    w_mid = 0.5 * (ws[1:] + ws[:-1])
    dt = np.where(w_mid > 0.0, 0.5 * (J[1:] + J[:-1]) * h / np.sqrt(2.0 * G * np.maximum(w_mid, 1e-300)), 0.0)
    if w_start == 0.0 and len(h) and ws[1] > 0.0:
        dt[0] = 2.0 * J[0] * h[0] / np.sqrt(2.0 * G * ws[1])
    t = t_start + np.concatenate([[0.0], np.cumsum(dt)])
    return (xs, ws, t), (kind, x_end, w_end)


def _fly(bed, slope, x_d, w_d, t_d, x_max, terminal_velocity=0.0, dt_scan=0.005,
         t_max=60.0):
    """Flight from the take-off point, a parabola without drag and numerically
    integrated with the quadratic drag of terminal velocity terminal_velocity.
    Returns the sample arrays, the landing point and the velocity (u_x, u_z)
    at landing."""
    v = np.sqrt(2.0 * G * w_d)
    p = float(slope(x_d))
    J = np.sqrt(1.0 + p * p)
    vx, vz = v / J, v * p / J
    z_d = float(bed(x_d))

    if terminal_velocity > 0.0:
        k = G / terminal_velocity**2

        def rhs(_, y):
            speed = np.hypot(y[2], y[3])
            return [y[2], y[3], -k * speed * y[2], -G - k * speed * y[3]]

        solution = solve_ivp(rhs, (0.0, t_max), [x_d, z_d, vx, vz], dense_output=True,
                             rtol=1e-10, atol=1e-10)
        state = solution.sol
    else:
        def state(tau):
            tau = np.asarray(tau, dtype=float)
            return np.array([x_d + vx * tau, z_d + vz * tau - 0.5 * G * tau * tau,
                             vx + 0.0 * tau, vz - G * tau])

    def gap(tau):
        x, z = state(tau)[:2]
        return float(z - bed(x))

    taus = np.arange(dt_scan, t_max, dt_scan)
    track = state(taus)
    gaps = track[1] - bed(track[0])
    # first crossing below the bed after the slider has risen above it
    above = np.nonzero(gaps > 0.0)[0]
    start = above[0] if len(above) else 0
    below = np.nonzero(gaps[start:] <= 0.0)[0]
    if not len(below) or track[0, start + below[0]] > x_max:
        raise ValueError("The point slider does not land on the domain.")
    k_land = start + below[0]
    tau_land = brentq(gap, taus[k_land - 1], taus[k_land]) if k_land > 0 else taus[0]
    tau = np.concatenate([[0.0], taus[taus < tau_land], [tau_land]])
    x, z, ux, uz = state(tau)
    return (x, z, t_d + tau), (float(x[-1]), float(ux[-1]), float(uz[-1]))


def point_slider(bed, slope, curvature, mu, x_start, x_max, dx=0.02, terminal_velocity=0.0):
    """Path of a Coulomb point slider released at rest at x_start on the bed
    z = bed(x) with the derivatives slope and curvature (vectorized callables)
    and the terminal velocity of the air drag in flight (0 = none) of the
    model."""
    ts, xs, zs, flying, events = [], [], [], [], []
    x, w, t = float(x_start), 0.0, 0.0
    for _ in range(20):  # at most this many flights
        (sx, sw, st), (kind, x_end, w_end) = _slide(bed, slope, curvature, mu, x, w, t,
                                                    x_max, dx)
        ts.append(st), xs.append(sx), zs.append(bed(sx)), flying.append(np.zeros(len(sx), bool))
        t = st[-1]
        if kind != "takeoff":
            events.append({"kind": kind, "x": float(x_end), "t": float(t),
                           "speed": float(np.sqrt(2.0 * G * w_end))})
            break
        events.append({"kind": "takeoff", "x": float(x_end), "t": float(t),
                       "speed": float(np.sqrt(2.0 * G * w_end))})
        (fx, fz, ft), (x_land, ux, uz) = _fly(bed, slope, x_end, w_end, t, x_max,
                                              terminal_velocity)
        ts.append(ft), xs.append(fx), zs.append(fz), flying.append(np.ones(len(fx), bool))
        t = ft[-1]
        p = float(slope(x_land))
        J = np.sqrt(1.0 + p * p)
        u_n = max(-(-p * ux + uz) / J, 0.0)
        u_t = (ux + p * uz) / J
        v = max(u_t - mu * u_n, 0.0)
        events.append({"kind": "landing", "x": float(x_land), "t": float(t),
                       "speed": float(v), "impact_speed": float(u_n)})
        x, w = x_land, v * v / (2.0 * G)
    return SliderPath(np.concatenate(ts), np.concatenate(xs), np.concatenate(zs),
                      np.concatenate(flying), events)
