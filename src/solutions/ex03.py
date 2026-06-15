from src.utils.typing import Vector, VecNumMap, Vec2NumMap

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.finite_elements.elliptic_pde_linear import LinearFEMEllipticPDE
from src.utils.common import triangle_area
from src.mesh_tools.mesh_tools import Triangulation, read_msh

'''
ex03 (c):
The solution is no longer unique, because kappa_zero = 0 (meaning if u is
a solution, then u(x) + c is also a solution). This leads to the matrix A
being (almost) singular and the program prints an error when trying to solve
the matrix vector equation using np.solve(A, b):

```
/venv/lib/python3.14/site-packages/matplotlib/tri/_triangulation.py:181: RuntimeWarning: invalid value encountered in cast
  triangles = np.asarray(triangles, dtype=np.int32)
```
'''

def approx_L2_error(pde: LinearFEMEllipticPDE, solution: np.ndarray, numerical_solution: np.ndarray) -> float:
    '''
    Calculate the approximation for the error in the L2 norm by using quadrature formula
    '''
    # approximate the error using a quadrature formula
    approx_error = 0
    for k in range(pde.m):
        tri_idx = pde.triang._tri_idx[k]
        points = pde.triang._points[tri_idx]
        residuals_squared = (solution[tri_idx] - numerical_solution[tri_idx])**2
        approx_error += triangle_area(points[0], points[1], points[2]) / 3 * np.sum(residuals_squared)
    return np.sqrt(approx_error)

def _test_problem_kappa(vec: Vector) -> np.ndarray:
    '''
    Helper function to describe kappa
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    a = np.exp(2*x*x + y*y)
    b = np.exp(x*x)
    return np.array([
        [a, b],
        [b, a]
    ])

def _test_problem_kappa_zero(vec: Vector) -> float:
    return 0

'''
ex03 (b)
'''
def _b_test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side of the poisson PDE
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    return 6 - 12*x*y*np.exp(-x*x - y*y)

def _b_test_problem_g_dir(vec: Vector) -> np.ndarray | float:
    vec = np.asarray(vec)
    x, y = vec[..., 0], vec[..., 1]
    on_gamma_1 = np.isclose(x*x + y*y, 1)
    return np.where(
        on_gamma_1,
        np.exp(-x*x - 1),
        np.exp(-x*x - 0.25)
    )

def _b_test_problem_g_neu(vec1: Vector, vec2: Vector) -> float:
    vec1, vec2 = np.asarray(vec1), np.asarray(vec2)
    x1, y1, x2, y2 = vec1[0], vec1[1], vec2[0], vec2[1]
    vec1_on_gamma_1 = np.isclose(x1*x1 + y1*y1, 1)
    vec2_on_gamma_2 = np.isclose(x2*x2 + y2*y2, 1)
    if vec1_on_gamma_1 != vec2_on_gamma_2:
        raise ValueError('Two points of an edge cannot lie on two different boundaries!')
    midpoint = 0.5 * (vec1 + vec2)
    x, y = midpoint[0], midpoint[1]
    return (-2*x*x - 2 - 6*x*y*np.exp(-1)) * vec1_on_gamma_1 + (4*x*x + 1 + 12*x*y*np.exp(-0.25)) * (1 - vec1_on_gamma_1)

'''
ex03 (d)
'''
def _d_test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side of the poisson PDE
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    return x + y

def _d_test_problem_g_dir(vec: Vector) -> np.ndarray | float:
    vec = np.asarray(vec)
    x, y = vec[..., 0], vec[..., 1]
    return -x*y

def _d_test_problem_g_neu(vec1: Vector, vec2: Vector) -> float:
    vec1, vec2 = np.asarray(vec1), np.asarray(vec2)
    x1, y1, x2, y2 = vec1[0], vec1[1], vec2[0], vec2[1]
    vec1_on_gamma_1 = np.isclose(x1*x1 + y1*y1, 1)
    vec2_on_gamma_2 = np.isclose(x2*x2 + y2*y2, 1)
    if vec1_on_gamma_1 != vec2_on_gamma_2:
        raise ValueError('Two points of an edge cannot lie on two different boundaries!')
    midpoint = 0.5 * (vec1 + vec2)
    x, y = midpoint[0], midpoint[1]
    return -4 * vec1_on_gamma_1 + 4 * (1-vec1_on_gamma_1)

def _test_problem_u(vec: Vector) -> np.ndarray:
    '''
    Theoretical solution for the test problem
    '''
    vec = np.asarray(vec)
    x = vec[..., 0]
    y = vec[..., 1]
    return np.exp(-2*x*x - y*y)

def plot_finite_elements_problem(
        f: VecNumMap,
        g_dir: VecNumMap,
        g_neu: Vec2NumMap,
        actual_ax: Axes3D | None,
        numerical_axes: list[Axes3D],
        plot_titles: list[str],
        triangs: list[Triangulation]
):
    '''
    solution: ex03 (b, d)
    '''
    solution = None
    if actual_ax is not None:
        solution_triang = triangs[0]

        # create exact solution plot
        solution = _test_problem_u(solution_triang._points)
        x, y = solution_triang._points[:, 0], solution_triang._points[:, 1]
        actual_ax.plot_trisurf(x, y, solution_triang._tri_idx, solution, cmap='viridis')
        actual_ax.set_title("Actual solution")

    for numerical_ax, title, triang in zip(numerical_axes, plot_titles, triangs):
        pde = LinearFEMEllipticPDE(
            f,
            triang,
            _test_problem_kappa,
            _test_problem_kappa_zero,
            g_dir,
            g_neu,
        )

        # compute numerical solution
        print(f'Computing solution for plot {title}...')
        numerical_solution = pde.solve()

        x, y = triang._points[:, 0], triang._points[:, 1]
        numerical_ax.plot_trisurf(x, y, pde.triang._tri_idx, numerical_solution, cmap='viridis')
        numerical_ax.set_title(title)

        if solution is not None:
            # approximate the error using a quadrature formula
            print(f"Approximate error for {title}: {approx_L2_error(pde, solution, numerical_solution)}")

def show_plots():
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    sol = fig.add_subplot(gs[0, 0], projection='3d')
    tri_A = fig.add_subplot(gs[0, 1], projection='3d')
    tri_B = fig.add_subplot(gs[1, 0], projection='3d')
    tri_C = fig.add_subplot(gs[1, 1], projection='3d')

    detail = 'fine'
    boundary_combinations = [('A', 'DIR', 'DIR'), ('B', 'DIR', 'NEU'), ('C', 'NEU', 'DIR')] # (name, outer, inner)

    plot_finite_elements_problem(
        _b_test_problem_f,
        _b_test_problem_g_dir,
        _b_test_problem_g_neu,
        sol,
        [tri_A, tri_B, tri_C],
        [f'({c[0]}) out {c[1]}, inn {c[2]}' for c in boundary_combinations],
        [read_msh(f'assets/meshes/circular_ring_{detail}_out_{c[1]}_inn_{c[2]}') for c in boundary_combinations]
    )

    plt.tight_layout()
    plt.savefig('figures/ex03b.pdf')
    plt.show()

    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 3)

    tri_A = fig.add_subplot(gs[0, 0], projection='3d')
    tri_B = fig.add_subplot(gs[0, 1], projection='3d')
    tri_C = fig.add_subplot(gs[0, 2], projection='3d')

    plot_finite_elements_problem(
        _d_test_problem_f,
        _d_test_problem_g_dir,
        _d_test_problem_g_neu,
        None,
        [tri_A, tri_B, tri_C],
        [f'({c[0]}) out {c[1]}, inn {c[2]}' for c in boundary_combinations],
        [read_msh(f'assets/meshes/circular_ring_{detail}_out_{c[1]}_inn_{c[2]}') for c in boundary_combinations]
    )

    plt.tight_layout()
    plt.savefig('figures/ex03d.pdf')
    plt.show()

show_plots()
