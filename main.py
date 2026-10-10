"""Точка входа: запускает эксперименты и сохраняет графики в папку ``images/``.

Примеры::

    python main.py                 # все эксперименты
    python main.py fire            # один пожар: анимация, снимки, динамика
    python main.py sweep --runs 50 # перколяционный эксперимент
    python main.py ds              # модель Драсселя–Швабля
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from forest_fire import Neighborhood
from forest_fire import experiments as exp
from forest_fire import visualization as vis

# Теоретические пороги перколяции по узлам квадратной решётки.
PERCOLATION_THRESHOLDS = {"von Neumann": 0.5927, "Moore": 0.4073}
NEIGHBORHOODS = {"von Neumann": Neighborhood.VON_NEUMANN, "Moore": Neighborhood.MOORE}


def run_fire(out: Path, seed: int) -> None:
    """Один пожар в лесу плотности 0.6 для двух типов окрестности."""
    results = {
        name: exp.single_fire(density=0.6, neighborhood=nb, seed=seed)
        for name, nb in NEIGHBORHOODS.items()
    }
    for name, res in results.items():
        steps = len(res.frames) - 1
        print(f"  {name:<12} пожар погас через {steps} шагов")

    vis.plot_snapshots(results, out / "fire_snapshots.png")
    vis.plot_history(results, out / "fire_history.png")
    # Для анимации берём лес вблизи порога перколяции фон Неймана: фронт
    # огня получается рваным и «фрактальным», это красиво и показательно.
    gif_run = exp.single_fire(density=0.62, frame_every=2, seed=seed)
    vis.save_gif(gif_run.frames, out / "fire.gif", fps=25)


def run_sweep(out: Path, seed: int, runs: int) -> None:
    """Зависимость распространения огня от плотности леса."""
    densities = np.round(np.arange(0.25, 0.851, 0.02), 2)
    results = {
        name: exp.percolation_sweep(densities, n_runs=runs, neighborhood=nb, seed=seed)
        for name, nb in NEIGHBORHOODS.items()
    }
    for name, res in results.items():
        # Эмпирический порог — плотность, при которой вероятность пересечения
        # впервые превышает 1/2.
        idx = int(np.argmax(res.crossing_probability >= 0.5))
        print(f"  {name:<12} p_c ≈ {res.densities[idx]:.2f} "
              f"(теория {PERCOLATION_THRESHOLDS[name]:.3f})")
    vis.plot_sweep(results, PERCOLATION_THRESHOLDS, out / "percolation.png")


def run_drossel_schwabl(out: Path, seed: int) -> None:
    """Самоорганизованная критичность: лес растёт и горит бесконечно."""
    shape = (200, 200)
    res = exp.drossel_schwabl(shape=shape, steps=1500, frame_every=2, frames_from=1000, seed=seed)
    vis.plot_drossel_schwabl(res, shape[0] * shape[1], out / "drossel_schwabl.png")
    vis.save_gif(res.frames, out / "drossel_schwabl.gif", fps=25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forest-fire cellular automaton")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="all",
        choices=["all", "fire", "sweep", "ds"],
        help="какой эксперимент запустить (по умолчанию все)",
    )
    parser.add_argument("--seed", type=int, default=1097, help="seed генератора случайных чисел")
    parser.add_argument("--runs", type=int, default=30, help="прогонов на точку в sweep")
    parser.add_argument("--out", type=Path, default=Path("images"), help="папка для картинок")
    args = parser.parse_args()

    tasks = {
        "fire": ("Один пожар", lambda: run_fire(args.out, args.seed)),
        "sweep": ("Перколяция", lambda: run_sweep(args.out, args.seed, args.runs)),
        "ds": ("Драссель–Швабль", lambda: run_drossel_schwabl(args.out, args.seed)),
    }
    selected = tasks if args.experiment == "all" else {args.experiment: tasks[args.experiment]}
    for title, task in selected.values():
        started = time.perf_counter()
        print(f"{title}...")
        task()
        print(f"  готово за {time.perf_counter() - started:.1f} с")
    print(f"Картинки сохранены в {args.out.resolve()}")


if __name__ == "__main__":
    main()
