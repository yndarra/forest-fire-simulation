"""Графики и анимации для модели лесного пожара."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

from .experiments import RunResult, SweepResult
from .model import CellState

# Цвета состояний: индекс в списке совпадает с кодом CellState.
STATE_COLORS = {
    CellState.EMPTY: "#2b2420",    # выжженная земля
    CellState.TREE: "#3a9a3f",     # дерево
    CellState.BURNING: "#ff9a1f",  # огонь
}
STATE_LABELS = {
    CellState.EMPTY: "Empty",
    CellState.TREE: "Tree",
    CellState.BURNING: "Burning",
}
CMAP = ListedColormap([STATE_COLORS[s] for s in CellState])

# Палитра для быстрого рендера кадров без matplotlib: код клетки -> RGB.
_PALETTE = np.array(
    [tuple(int(STATE_COLORS[s][i : i + 2], 16) for i in (1, 3, 5)) for s in CellState],
    dtype=np.uint8,
)


def draw_grid(ax: plt.Axes, grid: np.ndarray, title: str = "") -> None:
    """Нарисовать поле автомата на осях ``ax``.

    ``vmin``/``vmax`` заданы явно: иначе matplotlib нормирует цвета по
    фактическому диапазону значений, и на кадре без огня деревья
    окрасились бы «горящим» цветом.
    """
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=len(CellState) - 1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def legend_handles() -> list[Patch]:
    """Элементы легенды с цветами состояний."""
    return [Patch(color=STATE_COLORS[s], label=STATE_LABELS[s]) for s in CellState]


def grid_to_image(grid: np.ndarray, scale: int = 2) -> Image.Image:
    """Перевести поле в картинку PIL, увеличив каждую клетку до ``scale`` пикселей."""
    rgb = _PALETTE[grid]
    image = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        image = image.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    return image


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 20, scale: int = 2) -> None:
    """Сохранить кадры в GIF.

    Рендерим напрямую через PIL — это на порядки быстрее, чем рисовать каждый
    кадр matplotlib'ом и склеивать PNG-файлы с диска.
    """
    images = [grid_to_image(f, scale) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def plot_snapshots(results: dict[str, RunResult], path: Path) -> None:
    """Начальное и конечное состояния для нескольких прогонов (по строке на прогон)."""
    fig, axes = plt.subplots(len(results), 2, figsize=(8, 4 * len(results)), squeeze=False)
    for row, (name, res) in enumerate(results.items()):
        draw_grid(axes[row, 0], res.frames[0], f"{name}: t = 0")
        last_step = len(res.history[CellState.TREE]) - 1
        draw_grid(axes[row, 1], res.frames[-1], f"{name}: t = {last_step}")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def plot_history(results: dict[str, RunResult], path: Path) -> None:
    """Число деревьев и горящих клеток во времени для нескольких прогонов."""
    fig, (ax_trees, ax_fire) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, res in results.items():
        ax_trees.plot(res.history[CellState.TREE], label=name)
        ax_fire.plot(res.history[CellState.BURNING], label=name)
    ax_trees.set(title="Trees", xlabel="Step", ylabel="Cells")
    ax_fire.set(title="Burning trees", xlabel="Step", ylabel="Cells")
    for ax in (ax_trees, ax_fire):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_sweep(results: dict[str, SweepResult], thresholds: dict[str, float], path: Path) -> None:
    """Доля сгоревшего леса и вероятность перколяции в зависимости от плотности."""
    fig, (ax_burn, ax_cross) = plt.subplots(1, 2, figsize=(12, 4.5))
    for (name, res), color in zip(results.items(), ("tab:green", "tab:orange"), strict=False):
        ax_burn.plot(res.densities, res.burned_fraction, "o-", ms=3, color=color, label=name)
        ax_cross.plot(res.densities, res.crossing_probability, "o-", ms=3, color=color, label=name)
        for ax in (ax_burn, ax_cross):
            ax.axvline(thresholds[name], color=color, ls="--", alpha=0.6)
    ax_burn.set(title="Burned fraction of the forest", xlabel="Forest density", ylabel="Fraction")
    ax_cross.set(
        title="Probability that fire crosses the forest",
        xlabel="Forest density",
        ylabel="Probability",
    )
    for ax in (ax_burn, ax_cross):
        ax.grid(alpha=0.3)
        ax.legend(title="dashed: theoretical $p_c$")
    fig.tight_layout()
    _save(fig, path)


def plot_drossel_schwabl(result: RunResult, n_cells: int, path: Path) -> None:
    """Плотность деревьев и число горящих клеток в модели Драсселя–Швабля."""
    trees = np.array(result.history[CellState.TREE]) / n_cells
    fire = np.array(result.history[CellState.BURNING]) / n_cells
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(trees, color=STATE_COLORS[CellState.TREE], label="Tree density")
    ax.set(xlabel="Step", ylabel="Tree density", title="Drossel–Schwabl forest-fire model")
    ax.grid(alpha=0.3)
    # Горящих клеток на порядок меньше, поэтому у них своя ось справа.
    ax_fire = ax.twinx()
    ax_fire.plot(fire, color=STATE_COLORS[CellState.BURNING], label="Burning density")
    ax_fire.set_ylabel("Burning density")
    lines = ax.get_lines() + ax_fire.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    """Сохранить фигуру и закрыть её, чтобы не копить открытые окна."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
