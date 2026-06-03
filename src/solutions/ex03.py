from src.utils.typing import Vector

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from src.finite_elements.poisson_edges import LinearFEMPoissonPDEVariational
from src.utils.common import time_function, triangle_area, draw_convergence_plot
from src.mesh_tools.mesh_tools import Triangulation, read_msh

def _test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side of the poisson PDE
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    return 6 - 12*x*y*np.exp(-x*x - y*y)

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

def _test_problem_g_dir(vec: Vector) -> float:
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    on_gamma_1 = x*x + y*y == 1
    return np.exp(-x*x - 1) * on_gamma_1 + np.exp(-x*x - 0.25) * (1 - on_gamma_1)

def _test_problem_g_neu(vec1: Vector, vec2: Vector) -> float:
    vec1, vec2 = np.asarray(vec1), np.asarray(vec2)
    x1, y1, x2, y2 = vec1[0], vec1[1], vec2[0], vec2[1]
    vec1_on_gamma_1 = x1*x1 + y1*y1 == 1
    vec2_on_gamma_2 = x2*x2 + y2*y2 == 1
    if vec1_on_gamma_1 != vec2_on_gamma_2:
        raise ValueError('Two points of an edge cannot lie on two different boundaries!')
    midpoint = 0.5 * (vec1 + vec2)
    x, y = midpoint[0], midpoint[1]
    return (-2*x*x - 2 - 6*x*y*np.exp(-1)) * vec1_on_gamma_1 + (4*x*x + 1 + 12*x*y*np.exp(-0.25)) * (1 - vec1_on_gamma_1)

def _test_problem_u(vec: Vector) -> np.ndarray:
    '''
    Theoretical solution for the test problem
    '''
    vec = np.asarray(vec)
    x = vec[..., 0]
    y = vec[..., 1]
    return np.exp(-2*x*x - y*y)

def plot_finite_elements_problem(actual_ax: Axes3D, numerical_ax: Axes3D, triang: Triangulation):
    '''
    solution: ex02 (b)
    '''
    pde = LinearFEMPoissonPDEVariational(
        _test_problem_f,
        triang,
        _test_problem_kappa,
        _test_problem_kappa_zero,
        _test_problem_g_dir,
        _test_problem_g_neu,
    )

    numerical_solution = pde.solve()

    # compute real solution
    x, y = triang._points[:, 0], triang._points[:, 1]
    solution = _test_problem_u(triang._points)

    # create the plots
    actual_ax.plot_trisurf(x, y, solution, cmap='viridis')
    actual_ax.set_title("Actual solution")

    numerical_ax.plot_trisurf(x, y, numerical_solution, cmap='viridis')
    numerical_ax.set_title(f"Numerical solution (N = {pde.N}, m = {pde.m})")

def show_plots():
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    top_left = fig.add_subplot(gs[0, 0], projection='3d')
    top_right = fig.add_subplot(gs[0, 1], projection='3d')
    lower_row = fig.add_subplot(gs[1, :])

    filename = 'assets/meshes/circular_ring_coarse_out_NEU_inn_NEU.msh'
    tri = read_msh(filename)

    plot_finite_elements_problem(top_left, top_right, tri)

    plt.tight_layout()
    plt.savefig('figures/ex03.pdf')
    plt.show()

show_plots()
