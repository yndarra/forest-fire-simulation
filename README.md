# Forest-Fire Cellular Automaton

[![tests](https://github.com/yndarra/forest-fire-simulation/actions/workflows/tests.yml/badge.svg)](https://github.com/yndarra/forest-fire-simulation/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**English** | [Русский](README.ru.md)

Simulation of a spreading forest fire with a two-dimensional cellular automaton.
The project covers two classic settings: a single fire passing through a random
forest, which is a **site-percolation** problem, and the **Drossel–Schwabl
model**, where trees keep growing and burning and the system organizes itself
into a critical state.

<p align="center">
  <img src="images/fire.gif" width="360" alt="Fire front moving through a forest of density 0.62">
  &nbsp;
  <img src="images/drossel_schwabl.gif" width="360" alt="Drossel–Schwabl model in the steady state">
</p>
<p align="center"><sub>Left: a single fire front, density 0.62, von Neumann neighborhood. Right: Drossel–Schwabl steady state.</sub></p>

> Coursework for the *Computer Modelling* university course.

## Model

Each cell of a rectangular grid is **empty**, a **tree**, or **burning**. All cells
are updated simultaneously:

| Current state | Condition | Next state |
|---|---|---|
| Burning | always | Empty |
| Tree | at least one burning neighbor | Burning |
| Tree | otherwise, with probability `p_lightning` | Burning |
| Empty | with probability `p_growth` | Tree |

Two neighborhoods are supported:

* **von Neumann** (`+`): 4 side neighbors;
* **Moore** (`x` + `+`): 8 neighbors, including diagonals.

The grid border is not periodic: there is no forest beyond the edge.

The update is fully vectorized with NumPy. Neighbors are found by shifting the
"burning" mask, with no Python loops over cells, so a 200 × 200 grid takes
well under a millisecond per step.

## Experiments

### 1. A single fire

The forest has density 0.6, and the whole left edge is ignited. With the Moore
neighborhood fire also jumps diagonally, so the same forest burns down almost
completely. With von Neumann, 0.6 is right at the critical density, and large
islands survive.

![Initial and final states](images/fire_snapshots.png)
![Number of trees and burning cells over time](images/fire_history.png)

### 2. Percolation threshold

For each density from 0.25 to 0.85 the program runs 30 independent forests of
100 × 100 cells and measures two things: the fraction of trees that burned, and
how often the fire reached the opposite edge. Both curves jump sharply near the
theoretical **site-percolation thresholds** of the square lattice (dashed lines).

![Percolation sweep](images/percolation.png)

| Neighborhood | Measured p<sub>c</sub> (crossing probability ≥ 0.5) | Theory |
|---|---|---|
| von Neumann | ≈ 0.61 | 0.5927 |
| Moore | ≈ 0.41 | 0.4073 |

The measured von Neumann threshold is slightly higher than the theoretical one,
as expected for a finite 100 × 100 grid.

### 3. Drossel–Schwabl model

Here `p_growth = 0.02` and `p_lightning = 2·10⁻⁵`. After a transient with damped
oscillations, the tree density settles around **≈ 0.44** and keeps fluctuating:
fires of every size burn out patches of forest that slowly grow back. This is a
standard example of **self-organized criticality**.

![Tree density over time](images/drossel_schwabl.png)

## Getting started

```bash
git clone https://github.com/yndarra/forest-fire-simulation.git
cd forest-fire-simulation
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python main.py                # run every experiment, images go to images/
python main.py fire           # one experiment: fire | sweep | ds
python main.py sweep --runs 50 --seed 42
```

Running everything takes about 15 seconds. All results are reproducible because
the random seed is fixed (`--seed`, default 1097).

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Project structure

```
forest_fire/
├── model.py          # cell states, neighborhoods, ForestFire automaton
├── experiments.py    # single fire, percolation sweep, Drossel–Schwabl run
└── visualization.py  # plots and fast GIF rendering with Pillow
tests/                # unit tests for the update rules
main.py               # command-line entry point
```

## References

* B. Drossel, F. Schwabl. *Self-organized critical forest-fire model.* Phys. Rev. Lett. 69, 1629 (1992).
* D. Stauffer, A. Aharony. *Introduction to Percolation Theory.* Taylor & Francis, 1994.
* [Forest-fire model](https://en.wikipedia.org/wiki/Forest-fire_model) on Wikipedia.

## License

[MIT](LICENSE)
