import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from physics import alpha, alpha_degrees, length, radius

x_top = 0.0
y_top = length * np.sin(alpha)


def ball_center(s):
    x = x_top + s * np.cos(alpha) + radius * np.sin(alpha)
    y = y_top - s * np.sin(alpha) + radius * np.cos(alpha)
    return x, y


def draw_slope(ax):
    x_bottom = length * np.cos(alpha)
    ax.plot([x_top, x_bottom], [y_top, 0.0], color="darkgray", lw=4)

    ax.set_xlim(-0.5, x_bottom + 1)
    ax.set_ylim(-0.5, y_top + 1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Моделирование качения шарика (склон {alpha_degrees}°, R = {radius} м)"
    )


def animate(t, s, phi):
    x, y = ball_center(s)

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_slope(ax)

    ball = plt.Circle((x[0], y[0]), radius, color="crimson", zorder=3)
    ax.add_patch(ball)

    (marker,) = ax.plot([], [], color="white", lw=2, zorder=4)

    def update(frame):
        cx, cy = x[frame], y[frame]
        ball.center = (cx, cy)

        angle = phi[frame]
        marker.set_data(
            [cx, cx + radius * np.sin(angle)],
            [cy, cy + radius * np.cos(angle)],
        )

        return ball, marker

    dt = t[1] - t[0]
    return FuncAnimation(fig, update, frames=len(t), interval=1000 * dt, blit=True)
