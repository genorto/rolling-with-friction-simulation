import numpy as np
from scipy.integrate import solve_ivp

g = 9.81
radius = 0.3
alpha_degrees = 5
alpha = np.radians(alpha_degrees)
length = 10.0


def ball_equations(t, state):
    _, v, _, omega = state

    ds_dt = v
    dv_dt = (5.0 / 7.0) * g * np.sin(alpha)
    dphi_dt = omega
    domega_dt = dv_dt / radius  # без проскальзывания: omega = v / R

    return [ds_dt, dv_dt, dphi_dt, domega_dt]


def reached_end_of_slope(t, state):
    s = state[0]
    return s - length


reached_end_of_slope.terminal = True  # type: ignore[attr-defined]


def simulate(fps=60):
    initial_state = [0.0, 0.0, 0.0, 0.0]
    t_max = 10.0
    t_eval = np.arange(0.0, t_max, 1.0 / fps)

    solution = solve_ivp(
        ball_equations,
        (0.0, t_max),
        initial_state,
        t_eval=t_eval,
        events=reached_end_of_slope,
    )

    t = solution.t
    s = solution.y[0]
    phi = solution.y[2]

    return t, s, phi
