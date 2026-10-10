"""Численные эксперименты с моделью лесного пожара.

Функции этого модуля только считают и возвращают данные — без графиков.
Визуализация вынесена в :mod:`forest_fire.visualization`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import CellState, ForestFire, Neighborhood


@dataclass
class RunResult:
    """Результат одного прогона автомата."""

    # Кадры поля (если запись кадров включена), каждый — копия матрицы.
    frames: list[np.ndarray] = field(default_factory=list)
    # Число клеток каждого типа на каждом шаге, ключ — состояние клетки.
    history: dict[CellState, list[int]] = field(
        default_factory=lambda: {state: [] for state in CellState}
    )

    def record(self, model: ForestFire, keep_frame: bool) -> None:
        """Добавить текущее состояние модели в результат."""
        for state, count in model.counts().items():
            self.history[state].append(count)
        if keep_frame:
            self.frames.append(model.grid.copy())


def single_fire(
    shape: tuple[int, int] = (200, 200),
    density: float = 0.6,
    neighborhood: Neighborhood = Neighborhood.VON_NEUMANN,
    n_fires: int | None = None,
    max_steps: int = 2000,
    frame_every: int = 1,
    seed: int | None = 1097,
) -> RunResult:
    """Один пожар в случайном лесу без роста деревьев и молний.

    Если ``n_fires`` не задано, поджигается весь левый край леса (фронт огня),
    иначе — ``n_fires`` случайных деревьев. Моделирование идёт до тех пор,
    пока огонь не погаснет (или до ``max_steps``).
    """
    rng = np.random.default_rng(seed)
    grid = ForestFire.random_forest(shape, density, rng)
    if n_fires is None:
        ForestFire.ignite_left_edge(grid)
    else:
        ForestFire.ignite_random(grid, n_fires, rng)
    model = ForestFire(grid, neighborhood, rng=rng)

    result = RunResult()
    result.record(model, keep_frame=True)
    while model.is_burning and model.time < max_steps:
        model.step()
        result.record(model, keep_frame=model.time % frame_every == 0)
    # Последний кадр сохраняем всегда, чтобы анимация заканчивалась итогом.
    if frame_every > 1 and model.time % frame_every:
        result.frames.append(model.grid.copy())
    return result


@dataclass
class SweepResult:
    """Результат перебора плотности леса."""

    densities: np.ndarray
    # Средняя доля сгоревших деревьев от их начального числа.
    burned_fraction: np.ndarray
    # Доля прогонов, в которых огонь дошёл от левого края до правого.
    crossing_probability: np.ndarray


def percolation_sweep(
    densities: np.ndarray,
    size: int = 100,
    n_runs: int = 30,
    neighborhood: Neighborhood = Neighborhood.VON_NEUMANN,
    seed: int | None = 1097,
) -> SweepResult:
    """Перколяционный эксперимент: как распространение огня зависит от плотности.

    В каждом прогоне поджигается весь левый столбец леса. Огонь горит, пока не
    погаснет, после чего считаем:

    * долю сгоревших деревьев;
    * дошёл ли огонь до правого края (перколяция).

    Для бесконечной решётки вероятность пересечения скачком меняется с 0 на 1
    при критической плотности: ~0.593 для окрестности фон Неймана и ~0.407 для
    окрестности Мура (порог перколяции по узлам квадратной решётки).
    """
    rng = np.random.default_rng(seed)
    burned = np.zeros(len(densities))
    crossed = np.zeros(len(densities))

    for k, density in enumerate(densities):
        for _ in range(n_runs):
            grid = ForestFire.random_forest((size, size), density, rng)
            ForestFire.ignite_left_edge(grid)
            n_trees = np.count_nonzero(grid != CellState.EMPTY)
            model = ForestFire(grid, neighborhood, rng=rng)

            reached_right = False
            while model.is_burning:
                model.step()
                # Проверяем правый столбец на каждом шаге: горящие клетки
                # сгорают за один шаг, и в конце их уже не увидеть.
                reached_right |= bool(np.any(model.grid[:, -1] == CellState.BURNING))

            # Деревья, которых не осталось, — сгорели.
            remaining = np.count_nonzero(model.grid == CellState.TREE)
            burned[k] += (n_trees - remaining) / max(n_trees, 1)
            crossed[k] += reached_right

    return SweepResult(
        densities=np.asarray(densities),
        burned_fraction=burned / n_runs,
        crossing_probability=crossed / n_runs,
    )


def drossel_schwabl(
    shape: tuple[int, int] = (200, 200),
    p_growth: float = 0.02,
    p_lightning: float = 2e-5,
    initial_density: float = 0.5,
    steps: int = 1500,
    frame_every: int = 2,
    frames_from: int = 0,
    neighborhood: Neighborhood = Neighborhood.VON_NEUMANN,
    seed: int | None = 1097,
) -> RunResult:
    """Модель Драсселя–Швабля: лес непрерывно растёт и выгорает.

    Кадры записываются начиная с шага ``frames_from`` — так в анимацию не
    попадает переходный процесс от случайного начального леса.
    """
    rng = np.random.default_rng(seed)
    grid = ForestFire.random_forest(shape, initial_density, rng)
    model = ForestFire(grid, neighborhood, p_growth, p_lightning, rng)

    result = RunResult()
    result.record(model, keep_frame=frames_from == 0)
    for _ in range(steps):
        model.step()
        keep = model.time >= frames_from and model.time % frame_every == 0
        result.record(model, keep_frame=keep)
    return result
