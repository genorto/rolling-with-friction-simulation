from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from scenarios import g

slip_tolerance = 1e-9


@dataclass
class Solution:
    t: np.ndarray
    s: np.ndarray
    v: np.ndarray
    phi: np.ndarray
    omega: np.ndarray
    friction_work: np.ndarray


def rolling_equations(scenario):
    a = (
        g
        * np.sin(scenario.alpha)
        / (1.0 + scenario.inertia / (scenario.mass * scenario.radius**2))
    )

    def equations(t, state):
        _, v, _, omega, _ = state
        return [v, a, omega, a / scenario.radius, 0.0]

    return equations


def slipping_equations(scenario, direction):
    normal = scenario.mass * g * np.cos(scenario.alpha)
    friction = -direction * scenario.mu * normal

    def equations(t, state):
        _, v, _, omega, _ = state
        dv_dt = g * np.sin(scenario.alpha) + friction / scenario.mass
        domega_dt = -friction * scenario.radius / scenario.inertia
        dwork_dt = scenario.mu * normal * direction * (v - omega * scenario.radius)
        return [v, dv_dt, omega, domega_dt, dwork_dt]

    return equations


def choose_mode(scenario, slip_velocity):
    if abs(slip_velocity) > slip_tolerance:
        return np.sign(slip_velocity)
    if scenario.can_roll:
        return None
    return 1.0


def simulate(scenario, fps=60):
    t_max = scenario.duration

    def reached_end_of_slope(t, state):
        return state[0] - scenario.length

    reached_end_of_slope.terminal = True  # type: ignore[attr-defined]

    def returned_to_top(t, state):
        return state[0]

    returned_to_top.terminal = True  # type: ignore[attr-defined]
    returned_to_top.direction = -1  # type: ignore[attr-defined]

    def slip_stopped(t, state):
        return state[1] - state[3] * scenario.radius

    slip_stopped.terminal = True  # type: ignore[attr-defined]

    t_grid = np.arange(0.0, t_max, 1.0 / fps)
    state = [0.0, scenario.v0, 0.0, scenario.omega0, 0.0]
    t_start = 0.0
    segments = []

    while True:
        direction = choose_mode(scenario, state[1] - state[3] * scenario.radius)
        if direction is None:
            equations = rolling_equations(scenario)
            events = [reached_end_of_slope, returned_to_top]
        else:
            equations = slipping_equations(scenario, direction)
            slip_stopped.direction = -direction  # type: ignore[attr-defined]
            events = [reached_end_of_slope, returned_to_top, slip_stopped]

        if segments:
            t_eval = t_grid[t_grid > t_start]
        else:
            t_eval = t_grid
        solution = solve_ivp(
            equations,
            (t_start, t_max),
            state,
            t_eval=t_eval,
            events=events,
            rtol=1e-9,
            atol=1e-12,
        )
        segments.append(solution.y)

        if direction is None or solution.t_events[2].size == 0:
            break

        t_start = solution.t_events[2][0]
        state = list(solution.y_events[2][0])
        state[1] = state[3] * scenario.radius

    y = np.concatenate(segments, axis=1)
    return Solution(
        t=t_grid[: y.shape[1]],
        s=y[0],
        v=y[1],
        phi=y[2],
        omega=y[3],
        friction_work=y[4],
    )


def energies(scenario, solution):
    kinetic_translational = 0.5 * scenario.mass * solution.v**2
    kinetic_rotational = 0.5 * scenario.inertia * solution.omega**2
    potential = (
        scenario.mass * g * (scenario.length - solution.s) * np.sin(scenario.alpha)
    )
    total = kinetic_translational + kinetic_rotational + potential
    return kinetic_translational, kinetic_rotational, potential, total


def angular_momentum(scenario, solution):
    momentum = scenario.mass * solution.v * scenario.radius + (
        scenario.inertia * solution.omega
    )
    torque = scenario.mass * g * scenario.radius * np.sin(scenario.alpha)
    expected = momentum[0] + torque * (solution.t - solution.t[0])
    return momentum, expected
