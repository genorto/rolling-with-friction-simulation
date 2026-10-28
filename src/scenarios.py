from dataclasses import dataclass

import numpy as np

g = 9.81
speed_of_light = 299_792_458.0
max_speed = 0.01 * speed_of_light
max_mu = 2.0
max_duration = 120.0
min_visible_time = 0.2


@dataclass(frozen=True)
class Scenario:
    title: str
    alpha_degrees: float
    mu: float
    radius: float = 0.3
    mass: float = 1.0
    length: float = 10.0
    v0: float = 0.0
    omega0: float = 0.0
    duration: float = 10.0

    def __post_init__(self):
        errors = validate(self)
        if errors:
            raise ValueError("\n".join(errors))

    @property
    def alpha(self):
        return np.radians(self.alpha_degrees)

    @property
    def inertia(self):
        return 0.4 * self.mass * self.radius**2

    @property
    def can_roll(self):
        ratio = self.mass * self.radius**2 / self.inertia
        return np.tan(self.alpha) <= self.mu * (1.0 + ratio)


def validate(scenario):
    numbers = {
        "угол": scenario.alpha_degrees,
        "μ": scenario.mu,
        "R": scenario.radius,
        "m": scenario.mass,
        "длина склона": scenario.length,
        "v0": scenario.v0,
        "ω0": scenario.omega0,
        "длительность": scenario.duration,
    }
    infinite = [name for name, value in numbers.items() if not np.isfinite(value)]
    if infinite:
        return [f"параметры должны быть конечными числами: {', '.join(infinite)}"]

    errors = []
    if not 0 <= scenario.alpha_degrees < 90:
        errors.append(
            "угол склона должен быть в диапазоне [0°, 90°): "
            "на вертикальной стене шар не касается склона"
        )
    if not 0 <= scenario.mu <= max_mu:
        errors.append(f"коэффициент трения μ должен быть в диапазоне [0, {max_mu}]")
    if scenario.radius <= 0:
        errors.append("радиус R должен быть положительным")
    if scenario.mass <= 0:
        errors.append("масса m должна быть положительной")
    if scenario.length <= 0:
        errors.append("длина склона должна быть положительной")
    elif scenario.radius >= scenario.length:
        errors.append("радиус шара должен быть меньше длины склона")
    if scenario.v0 < 0:
        errors.append("начальная скорость v0 не может быть отрицательной")
    if not 0 < scenario.duration <= max_duration:
        errors.append(f"длительность должна быть в диапазоне (0, {max_duration:g}] с")
    if errors:
        return errors

    if (
        scenario.v0 == 0
        and scenario.omega0 < 0
        and np.tan(scenario.alpha) < scenario.mu
    ):
        errors.append(
            "при v0 = 0 и закрутке назад (ω0 < 0) трение тянет шар вверх сильнее, "
            f"чем сила тяжести вниз (μ = {scenario.mu} > tan α = "
            f"{np.tan(scenario.alpha):.3g}): шар сразу уедет за верх склона; "
            "задайте v0 > 0, увеличьте угол или уменьшите μ"
        )

    inertia = scenario.inertia
    energy = (
        0.5 * scenario.mass * scenario.v0**2
        + 0.5 * inertia * scenario.omega0**2
        + scenario.mass * g * scenario.length * np.sin(scenario.alpha)
    )
    speed_bound = np.sqrt(2 * energy / scenario.mass) + scenario.radius * np.sqrt(
        2 * energy / inertia
    )
    if speed_bound > max_speed:
        errors.append(
            f"точки шара могут разогнаться до {speed_bound:.3g} м/с, "
            f"а модель нерелятивистская: допустимо не более {max_speed:.3g} м/с "
            "(1% скорости света)"
        )

    center_speed_bound = np.sqrt(2 * energy / scenario.mass)
    if scenario.length / center_speed_bound < min_visible_time:
        errors.append(
            f"шар может пройти склон быстрее чем за {min_visible_time} с — "
            "уменьшите v0 или увеличьте длину"
        )
    return errors


SCENARIOS = {
    "rolling": Scenario(
        title="Качение без проскальзывания",
        alpha_degrees=5,
        mu=0.3,
    ),
    "slipping": Scenario(
        title="Качение с проскальзыванием",
        alpha_degrees=30,
        mu=0.1,
    ),
    "slip_to_roll": Scenario(
        title="Толчок без вращения: проскальзывание переходит в качение",
        alpha_degrees=5,
        mu=0.1,
        v0=4.0,
    ),
    "custom": Scenario(
        title="Пользовательский сценарий",
        alpha_degrees=10,
        mu=0.2,
    ),
}
