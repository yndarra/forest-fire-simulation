import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from matplotlib.colors import ListedColormap
import random
from typing import List, Tuple, Callable, Dict

from PIL import Image
import os
import glob

rs = np.random.RandomState(seed=1097)

class NeighborhoodType(Enum):
    CROSS = "+"
    NEUMAN = "x"

class CellState(Enum):
    EMPTY = 0
    FIRING = 1
    TREE = 2

colors = ["#49423D", "green", "orange"]
cmap_forest = ListedColormap(colors)

class CellColor(Enum):
    EMPTY = colors[0]
    FIRING = colors[1]
    TREE = colors[2]

def plot_grid(ca, ax=None, **kw):
    # ca - клеточный автомат (матрица)
    if ax is None:
        # Не был передан уже созданный график
        fig, ax = plt.subplots()
    # Визуализируем КА, используя нашу цветовую карту
    ax.matshow(ca, cmap=cmap_forest)
    # В ключевых аргументах kw можно передать заголовок графика,
    # а можно не передавать. В последнем случае по умолчанию
    # в title будет лежать пустая строка.
    # В данной работе в title имеет смысл записать тип окрестности.
    title = kw.get("title", "")
    # Наводим красоту
    ax.set(
        xlabel="Length",
        ylabel="Width",
        title=title,
        aspect="equal"
    )
    # Возвращаем построенный график
    return fig, ax

def create_ca(w: int, h: int, fill_value: int = 0):
    # Возвращаем numpy array вместо списка
    return np.full((h, w), fill_value, dtype=int)

def init_state(ca: np.ndarray, eta: float, f: int):
    h, w = ca.shape
    # Вычисляем общее количество клеток с деревьями
    total_trees = int(eta * w * h)
    # Создаем список всех возможных позиций
    all_positions = [(i, j) for i in range(h) for j in range(w)]
    # Случайно выбираем позиции для деревьев
    tree_positions = random.sample(all_positions, total_trees)
    # Размещаем деревья (значение 1 - дерево)
    for i, j in tree_positions:
        ca[i, j] = 1
    # Случайно выбираем f деревьев для поджигания
    if total_trees > 0 and f > 0:
        burning_trees = random.sample(tree_positions, min(f, total_trees))
        for i, j in burning_trees:
            ca[i, j] = 2  # 2 - горящее дерево
    return ca

def update(ca: np.ndarray, nt: NeighborhoodType) -> np.ndarray:
    h, w = ca.shape
    new_ca = np.copy(ca)

    # Выбираем функцию для получения окрестности
    if nt == NeighborhoodType.CROSS:
        neighborhood_func = get_cross_neighborhood
    else:
        neighborhood_func = get_neuman_neighborhood

    # Обновляем каждую клетку
    for i in range(h):
        for j in range(w):
            neighbor_indexes = neighborhood_func((i, j), (h, w))
            update_cell(ca, new_ca, (i, j), neighbor_indexes)

    return new_ca

def update_cell(ca: np.ndarray,
                new_ca: np.ndarray,
                cell: Tuple[int, int],
                neighbor_indexes: List[Tuple[int, int]]):
    i, j = cell
    current_state = ca[i, j]

    # Правила для модели лесного пожара:
    # 0 - пусто
    # 1 - дерево
    # 2 - горящее дерево
    # 3 - пепел

    if current_state == 0:  # Пустая клетка
        new_ca[i, j] = 0

    elif current_state == 1:  # Дерево
        # Проверяем, есть ли горящие соседи
        burning_neighbors = any(ca[ni, nj] == 2 for ni, nj in neighbor_indexes)
        if burning_neighbors:
            new_ca[i, j] = 2  # Дерево загорается
        else:
            new_ca[i, j] = 1  # Дерево остается деревом

    elif current_state == 2:  # Горящее дерево
        new_ca[i, j] = 3  # Превращается в пепел

    elif current_state == 3:  # Пепел
        new_ca[i, j] = 0  # Пепел исчезает

def get_cross_neighborhood(cell: Tuple[int, int],
                           ca_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    i, j = cell
    h, w = ca_shape
    neighbors = []

    # Верхний сосед
    if i > 0:
        neighbors.append((i - 1, j))
    # Нижний сосед
    if i < h - 1:
        neighbors.append((i + 1, j))
    # Левый сосед
    if j > 0:
        neighbors.append((i, j - 1))
    # Правый сосед
    if j < w - 1:
        neighbors.append((i, j + 1))

    return neighbors

def get_neuman_neighborhood(cell: Tuple[int, int],
                            ca_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    i, j = cell
    h, w = ca_shape
    neighbors = []

    # Все возможные направления (включая диагонали)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # Верхний ряд
        (0, -1), (0, 1),  # Средний ряд (без центра)
        (1, -1), (1, 0), (1, 1)  # Нижний ряд
    ]

    for di, dj in directions:
        ni, nj = i + di, j + dj
        # Проверяем, что индексы в пределах границ
        if 0 <= ni < h and 0 <= nj < w:
            neighbors.append((ni, nj))

    return neighbors

class Statistics:
    """Класс для сбора статистики по моделированию."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.tree_counts: List[int] = []
        self.burning_counts: List[int] = []
        self.ash_counts: List[int] = []
        self.empty_counts: List[int] = []

    def add_state(self, ca: np.ndarray):
        """Добавить состояние КА и посчитать статистику."""
        self.states.append(np.copy(ca))

        # Считаем количество клеток каждого типа
        tree_count = np.sum(ca == 1)
        burning_count = np.sum(ca == 2)
        ash_count = np.sum(ca == 3)
        empty_count = np.sum(ca == 0)

        self.tree_counts.append(tree_count)
        self.burning_counts.append(burning_count)
        self.ash_counts.append(ash_count)
        self.empty_counts.append(empty_count)

    def plot_statistics(self):
        """Построить графики статистики."""
        time_steps = range(len(self.tree_counts))

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(time_steps, self.tree_counts, 'g-', label='Деревья')
        plt.xlabel('Время')
        plt.ylabel('Количество')
        plt.title('Деревья')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(time_steps, self.burning_counts, 'r-', label='Горящие')
        plt.xlabel('Время')
        plt.ylabel('Количество')
        plt.title('Горящие деревья')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(time_steps, self.ash_counts, 'k-', label='Пепел')
        plt.xlabel('Время')
        plt.ylabel('Количество')
        plt.title('Пепел')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(time_steps, self.empty_counts, 'b-', label='Пустые')
        plt.xlabel('Время')
        plt.ylabel('Количество')
        plt.title('Пустые клетки')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_final_stats(self) -> Dict[str, int]:
        """Получить финальную статистику."""
        return {
            'total_steps': len(self.states) - 1,
            'final_trees': self.tree_counts[-1],
            'final_burning': self.burning_counts[-1],
            'final_ash': self.ash_counts[-1],
            'final_empty': self.empty_counts[-1],
            'max_burning': max(self.burning_counts) if self.burning_counts else 0,
            'total_burned': self.ash_counts[-1] + self.empty_counts[-1] - self.empty_counts[0]
        }

def simulate(ca: np.ndarray,
             nt: NeighborhoodType,
             time: int) -> Statistics:
    st = Statistics()
    st.add_state(ca)  # Добавляем начальное состояние

    current_ca = np.copy(ca)

    # В цикле по времени от 0 до time
    for t in range(time):
        # Обновляем состояние КА
        new_ca = update(current_ca, nt)

        # Добавляем новое состояние в статистику
        st.add_state(new_ca)

        # Проверяем, закончился ли пожар
        if np.sum(new_ca == 2) == 0:  # Нет горящих деревьев
            print(f"Пожар закончился на шаге {t + 1}")
            break

        current_ca = new_ca

    return st


def save_frame(ca: np.ndarray, frame_number: int, output_dir: str = "frames"):
    """Сохранить текущее состояние КА как изображение для анимации."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(ca, cmap=cmap_forest)
    ax.set_title(f"Time step: {frame_number}")
    ax.axis('off')

    plt.savefig(f"{output_dir}/frame_{frame_number:04d}.png",
                bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)


def create_gif_animation(frame_pattern: str = "frames/frame_*.png",
                         output_gif: str = "forest_fire_animation.gif",
                         duration: int = 100):
    """Создать GIF анимацию из сохраненных кадров."""
    frames = []
    frame_files = sorted(glob.glob(frame_pattern))

    for frame_file in frame_files:
        frame = Image.open(frame_file)
        frames.append(frame)

    if frames:
        frames[0].save(
            output_gif,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=0
        )
        print(f"GIF анимация сохранена как {output_gif}")


def simulate(ca: np.ndarray,
             nt: NeighborhoodType,
             time: int,
             save_frames: bool = False) -> Statistics:
    st = Statistics()
    st.add_state(ca)  # Добавляем начальное состояние

    current_ca = np.copy(ca)

    # Сохраняем начальный кадр
    if save_frames:
        save_frame(current_ca, 0)

    # В цикле по времени от 0 до time
    for t in range(time):
        # Обновляем состояние КА
        new_ca = update(current_ca, nt)

        # Добавляем новое состояние в статистику
        st.add_state(new_ca)

        # Сохраняем кадр для анимации
        if save_frames:
            save_frame(new_ca, t + 1)

        # Проверяем, закончился ли пожар
        if np.sum(new_ca == 2) == 0:  # Нет горящих деревьев
            print(f"Пожар закончился на шаге {t + 1}")
            break

        current_ca = new_ca

    # Создаем GIF анимацию если сохранялись кадры
    if save_frames:
        create_gif_animation(duration=50)  # 50ms между кадрами

    return st

def cleanup_frames(output_dir: str = "frames"):
    """Удалить временные файлы кадров."""
    if os.path.exists(output_dir):
        for file in glob.glob(f"{output_dir}/*.png"):
            os.remove(file)
        os.rmdir(output_dir)
        print("Временные файлы кадров удалены")


# "Затравка" для генератора псевдослучайных чисел
seed = 1097
# Размеры клеточного поля
w, h = 200, 200
# Доля клеток с деревьями
eta = 0.8
# Число изначально горящих деревьев
f = 1
# Вероятность вырастания нового дерева
p_g = 0.02
# Вероятность случайного воспламенения дерева
p_f = 2e-5
# Модельное время (число обновлений автомата)
sim_time = 500

# Создаём "пустой" КА
ca = create_ca(w, h)
# Задаём начальное состояние КА
init_state(ca, eta, f)
# Также стоит запомнить исходное состояние
initial_ca = np.copy(ca)
# Симулируем.
# В данном случае рассматриваем "+"-окрестность
st = simulate(ca, NeighborhoodType.CROSS, time=350, save_frames=True)
# Обрабатываем результаты, строим графики и пр.

# Для визуализации результатов можно добавить:
print("Начальное состояние:")
fig, ax = plot_grid(initial_ca, title="Начальное состояние")
plt.show()

print("Финальное состояние:")
fig, ax = plot_grid(ca, title="Финальное состояние")
plt.show()

st.plot_statistics()
st.plot_statistics()