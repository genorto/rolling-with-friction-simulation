import argparse
import dataclasses

from scenarios import SCENARIOS

PARAMETERS = {
    "--angle": ("alpha_degrees", "угол склона, градусы"),
    "--mu": ("mu", "коэффициент трения"),
    "--radius": ("radius", "радиус шара, м"),
    "--mass": ("mass", "масса шара, кг"),
    "--length": ("length", "длина склона, м"),
    "--v0": ("v0", "начальная скорость центра вдоль склона, м/с"),
    "--omega0": ("omega0", "начальная угловая скорость, рад/с (< 0 — закрутка назад)"),
    "--duration": ("duration", "максимальное время симуляции, с"),
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Качение шара по наклонной плоскости",
        epilog=(
            "Параметры можно задать для любого сценария, например: "
            "custom --angle 20 --mu 0.05 --v0 2 или slipping --mu 0.15"
        ),
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="rolling",
        choices=SCENARIOS,
        help="сценарий моделирования (по умолчанию: %(default)s)",
    )
    for flag, (field, description) in PARAMETERS.items():
        parser.add_argument(flag, dest=field, type=float, help=description)
    return parser


def build_scenario(parser, args):
    base = SCENARIOS[args.scenario]
    overrides = {
        field: getattr(args, field)
        for field, _ in PARAMETERS.values()
        if getattr(args, field) is not None
    }
    if overrides and args.scenario != "custom":
        overrides["title"] = f"{base.title} (изменённые параметры)"

    try:
        return dataclasses.replace(base, **overrides)
    except ValueError as error:
        parser.error(f"недопустимые параметры:\n{error}")


if __name__ == "__main__":
    parser = build_parser()
    scenario = build_scenario(parser, parser.parse_args())

    import matplotlib.pyplot as plt #при первом запуске c --help долго грузится матплотлиб
    from physics import simulate
    from visualization import animate

    solution = simulate(scenario)

    min_visible_time = 0.2
    visible_time = solution.t[-1] - solution.t[0]
    if visible_time < min_visible_time:
        parser.error(
            f"анимация длится меньше {min_visible_time} с — "
            "измените параметры расчёта"
        )
    animation = animate(scenario, solution)
    plt.show()
