import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from physics import angular_momentum, energies, slip_tolerance


def slope_top(scenario):
    return 0.0, scenario.length * np.sin(scenario.alpha)


def ball_center(scenario, s):
    x_top, y_top = slope_top(scenario)
    alpha, radius = scenario.alpha, scenario.radius
    x = x_top + s * np.cos(alpha) + radius * np.sin(alpha)
    y = y_top - s * np.sin(alpha) + radius * np.cos(alpha)
    return x, y


def draw_slope(ax, scenario):
    x_top, y_top = slope_top(scenario)
    x_bottom = scenario.length * np.cos(scenario.alpha)

    ax.plot(
        [x_top, x_bottom],
        [y_top, 0.0],
        color="darkgray",
        lw=4,
    )

    radius = scenario.radius
    center_x, center_y = ball_center(
        scenario,
        np.array([0.0, scenario.length]),
    )

    padding = 0.05 * max(scenario.length, 2 * radius, 1.0)

    ax.set_xlim(
        min(x_top, x_bottom, center_x.min() - radius) - padding,
        max(x_top, x_bottom, center_x.max() + radius) + padding,
    )
    ax.set_ylim(
        min(0.0, y_top, center_y.min() - radius) - padding,
        max(0.0, y_top, center_y.max() + radius) + padding,
    )

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        f"{scenario.title}\n(склон {scenario.alpha_degrees}°, "
        f"R = {scenario.radius} м, μ = {scenario.mu})"
    )


def setup_energy_plot(ax, t, energy_curves):
    ax.set_xlim(t[0], t[-1])
    upper = max(curve.max() for _, curve, _ in energy_curves) or 1.0
    ax.set_ylim(-0.05 * upper, 1.45 * upper)
    ax.set_xlabel("t, с")
    ax.set_ylabel("E, Дж")
    ax.set_title("Баланс энергии")
    ax.grid(True, linestyle="--", alpha=0.5)

    lines = [
        ax.plot([], [], label=label, **style)[0] for label, _, style in energy_curves
    ]
    ax.legend(loc="upper center", ncol=3, fontsize=7)
    return lines


def setup_momentum_plot(ax, t, momentum, expected):
    ax.set_xlim(t[0], t[-1])
    lower = min(0.0, momentum.min(), expected.min())
    upper = max(0.0, momentum.max(), expected.max())
    padding = 0.1 * (upper - lower) or 1.0
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlabel("t, с")
    ax.set_ylabel("L, кг·м²/с")
    ax.set_title("Теорема об изменении момента импульса")
    ax.grid(True, linestyle="--", alpha=0.5)

    (simulated,) = ax.plot([], [], color="crimson", lw=2, label="L = mvR + Iω")
    (theory,) = ax.plot(
        [], [], color="black", lw=1, linestyle="--", label="L(0) + ∫M dt"
    )

    ends_high = expected[-1] > 0.5 * (lower + upper)
    ax.legend(loc="lower right" if ends_high else "upper right", fontsize=8)
    return simulated, theory


def animate(scenario, solution):
    t, phi, radius = solution.t, solution.phi, scenario.radius
    x, y = ball_center(scenario, solution.s)
    slipping = np.abs(solution.v - solution.omega * radius) > 1e3 * slip_tolerance

    kinetic_translational, kinetic_rotational, potential, total = energies(
        scenario, solution
    )
    momentum, expected = angular_momentum(scenario, solution)

    fig = plt.figure(figsize=(9, 6))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.8, 1])
    ax = fig.add_subplot(grid[0, :])
    ax_energy = fig.add_subplot(grid[1, 0])
    ax_momentum = fig.add_subplot(grid[1, 1])

    draw_slope(ax, scenario)

    ball = plt.Circle((x[0], y[0]), radius, color="crimson", zorder=3)
    ax.add_patch(ball)

    (marker,) = ax.plot([], [], color="white", lw=2, zorder=4)
    mode_label = ax.text(0.99, 0.93, "", transform=ax.transAxes, ha="right")

    energy_curves = [
        ("Поступательная", kinetic_translational, {"color": "tab:blue"}),
        ("Вращательная", kinetic_rotational, {"color": "tab:orange"}),
        ("Потенциальная", potential, {"color": "tab:green"}),
        ("Потери на трение Q", solution.friction_work, {"color": "tab:purple"}),
        ("Полная E", total, {"color": "black", "lw": 2}),
        (
            "E + Q",
            total + solution.friction_work,
            {"color": "gray", "lw": 1, "linestyle": "--"},
        ),
    ]
    energy_lines = setup_energy_plot(ax_energy, t, energy_curves)
    momentum_line, expected_line = setup_momentum_plot(
        ax_momentum, t, momentum, expected
    )

    fig.tight_layout()

    def update(frame):
        cx, cy = x[frame], y[frame]
        ball.center = (cx, cy)

        angle = phi[frame]
        marker.set_data(
            [cx, cx + radius * np.sin(angle)],
            [cy, cy + radius * np.cos(angle)],
        )
        mode_label.set_text("проскальзывание" if slipping[frame] else "качение")

        visible_t = t[: frame + 1]
        for line, (_, curve, _) in zip(energy_lines, energy_curves):
            line.set_data(visible_t, curve[: frame + 1])
        momentum_line.set_data(visible_t, momentum[: frame + 1])
        expected_line.set_data(visible_t, expected[: frame + 1])

        return ball, marker, mode_label, *energy_lines, momentum_line, expected_line

    dt = t[1] - t[0] if len(t) > 1 else 1.0 / 60
    return FuncAnimation(fig, update, frames=len(t), interval=1000 * dt, blit=True)
