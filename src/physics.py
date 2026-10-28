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
        if start_at_top:
            if stationary_at_top:
                return 1.0
            if t == t_start:
                return initial_velocity
            return state[0] / (t - t_start)
        return state[0]

    returned_to_top.terminal = True  # type: ignore[attr-defined]
    returned_to_top.direction = -1  # type: ignore[attr-defined]

    def slip_stopped(t, state):
        return state[1] - state[3] * scenario.radius

    slip_stopped.terminal = True  # type: ignore[attr-defined]

    t_grid = np.arange(0.0, t_max, 1.0 / fps)
    t_grid = np.append(t_grid[t_grid < t_max], t_max)
    state = [0.0, scenario.v0, 0.0, scenario.omega0, 0.0]
    t_start = 0.0
    segments = []
    times = []

    while True:
        direction = choose_mode(scenario, state[1] - state[3] * scenario.radius)
        if direction is None:
            equations = rolling_equations(scenario)
            events = [reached_end_of_slope, returned_to_top]
        else:
            equations = slipping_equations(scenario, direction)
            slip_stopped.direction = -direction  # type: ignore[attr-defined]
            events = [reached_end_of_slope, returned_to_top, slip_stopped]

        start_at_top = state[0] == 0.0
        initial_velocity = state[1]
        stationary_at_top = (
            start_at_top
            and initial_velocity == 0.0
            and equations(t_start, state)[1] == 0.0
        )

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
        if not solution.success:
            raise RuntimeError(f"Ошибка численного интегрирования: {solution.message}")
        if np.size(solution.y):
            segments.append(solution.y)
            times.append(solution.t)

        if direction is None or solution.t_events[2].size == 0:
            break

        next_start = solution.t_events[2][0]
        if next_start <= t_start:
            raise RuntimeError("Смена режима произошла без продвижения времени")
        if next_start >= t_max:
            break
        t_start = next_start
        state = list(solution.y_events[2][0])
        state[1] = state[3] * scenario.radius

    y = np.concatenate(segments, axis=1)
    t = np.concatenate(times)
    for event_times, event_states in zip(solution.t_events, solution.y_events):
        if event_times.size:
            if event_times[0] > t[-1]:
                t = np.append(t, event_times[0])
                y = np.column_stack((y, event_states[0]))
            else:
                y[:, -1] = event_states[0]
            break
    return Solution(
        t=t,
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
