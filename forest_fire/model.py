"""Клеточный автомат лесного пожара.

Модель — двумерная решётка, каждая клетка которой находится в одном из трёх
состояний: пусто, дерево, горящее дерево. За один шаг все клетки обновляются
одновременно (синхронно) по правилам:

1. Горящее дерево сгорает дотла — клетка становится пустой.
2. Дерево загорается, если в его окрестности есть хотя бы одно горящее дерево.
3. Дерево без горящих соседей загорается само с вероятностью ``p_lightning``
   (удар молнии).
4. На пустой клетке с вероятностью ``p_growth`` вырастает новое дерево.

При ``p_growth = p_lightning = 0`` получается классическая задача о перколяции:
один пожар проходит по лесу и гаснет. При ненулевых вероятностях — модель
Драсселя–Швабля (Drossel & Schwabl, 1992), в которой лес растёт и горит
бесконечно, а система сама приходит в критическое состояние.
"""

from __future__ import annotations

from enum import Enum, IntEnum

import numpy as np


class CellState(IntEnum):
    """Состояние клетки. Значения — это коды, которые хранятся в матрице."""

    EMPTY = 0
    TREE = 1
    BURNING = 2


class Neighborhood(Enum):
    """Тип окрестности клетки.

    Значение — список смещений ``(di, dj)`` до соседей.
    """

    # Окрестность фон Неймана («+»): 4 соседа по сторонам.
    VON_NEUMANN = ((-1, 0), (1, 0), (0, -1), (0, 1))
    # Окрестность Мура («x» + «+»): 8 соседей, включая диагональных.
    MOORE = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )


def has_burning_neighbor(grid: np.ndarray, neighborhood: Neighborhood) -> np.ndarray:
    """Вернуть булеву маску клеток, у которых есть горящий сосед.

    Векторизованная реализация: вместо цикла по клеткам сдвигаем всю матрицу
    «горящих» клеток на каждое смещение окрестности и объединяем результаты
    через логическое ИЛИ. Граница не периодическая — за краем поля «пусто»,
    поэтому матрица дополняется нулевой рамкой шириной в одну клетку.
    """
    burning = np.pad(grid == CellState.BURNING, 1, constant_values=False)
    h, w = grid.shape
    result = np.zeros((h, w), dtype=bool)
    for di, dj in neighborhood.value:
        # Срез из дополненной матрицы, сдвинутый на (di, dj) относительно центра.
        result |= burning[1 + di : 1 + di + h, 1 + dj : 1 + dj + w]
    return result


class ForestFire:
    """Клеточный автомат лесного пожара.

    Параметры
    ---------
    grid:
        Начальное состояние поля (матрица кодов :class:`CellState`).
        Копируется, исходный массив не изменяется.
    neighborhood:
        Тип окрестности, по которой распространяется огонь.
    p_growth:
        Вероятность того, что на пустой клетке за шаг вырастет дерево.
    p_lightning:
        Вероятность самовозгорания дерева за шаг (удар молнии).
    rng:
        Генератор случайных чисел. Передайте генератор с фиксированным
        ``seed``, чтобы результат был воспроизводимым.
    """

    def __init__(
        self,
        grid: np.ndarray,
        neighborhood: Neighborhood = Neighborhood.VON_NEUMANN,
        p_growth: float = 0.0,
        p_lightning: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not (0.0 <= p_growth <= 1.0 and 0.0 <= p_lightning <= 1.0):
            raise ValueError("Вероятности должны лежать в отрезке [0, 1]")
        self.grid = np.array(grid, dtype=np.int8, copy=True)
        self.neighborhood = neighborhood
        self.p_growth = p_growth
        self.p_lightning = p_lightning
        self.rng = rng if rng is not None else np.random.default_rng()
        # Номер текущего шага (модельное время).
        self.time = 0

    # ------------------------------------------------------------------
    # Начальные состояния
    # ------------------------------------------------------------------
    @staticmethod
    def random_forest(
        shape: tuple[int, int],
        density: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Случайный лес: каждая клетка — дерево с вероятностью ``density``."""
        if not 0.0 <= density <= 1.0:
            raise ValueError("Плотность леса должна лежать в отрезке [0, 1]")
        trees = rng.random(shape) < density
        return np.where(trees, CellState.TREE, CellState.EMPTY).astype(np.int8)

    @staticmethod
    def ignite_random(grid: np.ndarray, n_fires: int, rng: np.random.Generator) -> np.ndarray:
        """Поджечь ``n_fires`` случайных деревьев (на месте) и вернуть поле."""
        trees = np.flatnonzero(grid == CellState.TREE)
        if trees.size:
            chosen = rng.choice(trees, size=min(n_fires, trees.size), replace=False)
            grid.flat[chosen] = CellState.BURNING
        return grid

    @staticmethod
    def ignite_left_edge(grid: np.ndarray) -> np.ndarray:
        """Поджечь все деревья в левом столбце (на месте) — «фронт пожара»."""
        column = grid[:, 0]
        column[column == CellState.TREE] = CellState.BURNING
        return grid

    # ------------------------------------------------------------------
    # Динамика
    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        """Сделать один синхронный шаг автомата и вернуть новое поле."""
        g = self.grid
        is_empty = g == CellState.EMPTY
        is_tree = g == CellState.TREE
        is_burning = g == CellState.BURNING

        # Правило 2: огонь переходит на деревья с горящими соседями.
        catches_fire = is_tree & has_burning_neighbor(g, self.neighborhood)
        # Правило 3: удар молнии. Случайные числа генерируем, только если
        # вероятность ненулевая, — так одиночный пожар считается быстрее.
        if self.p_lightning > 0:
            catches_fire |= is_tree & (self.rng.random(g.shape) < self.p_lightning)
        # Правило 4: рост новых деревьев на пустых клетках.
        grows = np.zeros_like(is_empty)
        if self.p_growth > 0:
            grows = is_empty & (self.rng.random(g.shape) < self.p_growth)

        new = g.copy()
        new[is_burning] = CellState.EMPTY   # Правило 1: сгоревшее дерево исчезает.
        new[catches_fire] = CellState.BURNING
        new[grows] = CellState.TREE

        self.grid = new
        self.time += 1
        return new

    @property
    def is_burning(self) -> bool:
        """Есть ли на поле хотя бы одно горящее дерево."""
        return bool(np.any(self.grid == CellState.BURNING))

    def counts(self) -> dict[CellState, int]:
        """Число клеток в каждом состоянии."""
        values = np.bincount(self.grid.ravel(), minlength=len(CellState))
        return {state: int(values[state]) for state in CellState}
