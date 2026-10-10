"""Тесты правил клеточного автомата."""

import numpy as np
import pytest

from forest_fire import CellState, ForestFire, Neighborhood
from forest_fire.model import has_burning_neighbor

E, T, B = CellState.EMPTY, CellState.TREE, CellState.BURNING


def make(grid, neighborhood=Neighborhood.VON_NEUMANN, **kwargs):
    return ForestFire(np.array(grid), neighborhood, rng=np.random.default_rng(0), **kwargs)


def test_burning_tree_becomes_empty():
    model = make([[B]])
    model.step()
    assert model.grid[0, 0] == E


def test_fire_spreads_to_side_neighbors_only_in_von_neumann():
    grid = [
        [T, T, T],
        [T, B, T],
        [T, T, T],
    ]
    model = make(grid)
    model.step()
    expected = np.array([
        [T, B, T],
        [B, E, B],
        [T, B, T],
    ])
    np.testing.assert_array_equal(model.grid, expected)


def test_fire_spreads_diagonally_in_moore():
    grid = [
        [T, T, T],
        [T, B, T],
        [T, T, T],
    ]
    model = make(grid, Neighborhood.MOORE)
    model.step()
    expected = np.full((3, 3), B)
    expected[1, 1] = E
    np.testing.assert_array_equal(model.grid, expected)


def test_boundary_is_not_periodic():
    # Огонь у левого края не должен «перепрыгнуть» на правый край.
    grid = [[B, E, E, T]]
    assert not has_burning_neighbor(np.array(grid), Neighborhood.MOORE)[0, 3]


def test_fire_dies_out_without_growth():
    rng = np.random.default_rng(1)
    grid = ForestFire.random_forest((50, 50), 0.7, rng)
    ForestFire.ignite_random(grid, 3, rng)
    model = ForestFire(grid, rng=rng)
    while model.is_burning:
        model.step()
    assert model.counts()[B] == 0
    assert model.time < 50 * 50


def test_growth_probability_one_fills_empty_cells():
    model = make(np.full((4, 4), E), p_growth=1.0)
    model.step()
    assert np.all(model.grid == T)


def test_lightning_probability_one_ignites_all_trees():
    model = make(np.full((4, 4), T), p_lightning=1.0)
    model.step()
    assert np.all(model.grid == B)


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        make([[T]], p_growth=1.5)


def test_random_forest_density():
    grid = ForestFire.random_forest((300, 300), 0.4, np.random.default_rng(2))
    assert abs(np.mean(grid == T) - 0.4) < 0.01
