from src.utils.typing import Vector, VecNumMap, Vec2NumMap

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.finite_elements.elliptic_pde_linear import LinearFEMEllipticPDE
from src.finite_elements.elliptic_pde_quad import QuadraticFEMEllipticPDE
from src.utils.common import triangle_area
from src.mesh_tools.mesh_tools import Triangulation, TriangulationQuad, read_msh

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

def _test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side of the poisson PDE
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    return 6 - 12*x*y*np.exp(-x*x - y*y)

def _test_problem_g_dir(vec: Vector) -> np.ndarray | float:
    vec = np.asarray(vec)
    x, y = vec[..., 0], vec[..., 1]
    on_gamma_1 = np.isclose(x*x + y*y, 1)
    return np.where(
        on_gamma_1,
        np.exp(-x*x - 1),
        np.exp(-x*x - 0.25)
    )

def _test_problem_u(vec: Vector) -> np.ndarray:
    '''
    Theoretical solution for the test problem
    '''
    vec = np.asarray(vec)
    x = vec[..., 0]
    y = vec[..., 1]
    return np.exp(-2*x*x - y*y)

def plot_finite_elements_problem(
        actual_ax: Axes3D,
        linear_ax: Axes3D,
        quad_ax: Axes3D,
        triang: Triangulation
):
    '''
    solution: ex03 (b, d)
    '''
    solution = None

    # create exact solution plot
    solution = _test_problem_u(triang._points)
    x, y = triang._points[:, 0], triang._points[:, 1]
    actual_ax.plot_trisurf(x, y, triang._tri_idx, solution, cmap='viridis')
    actual_ax.set_title("Actual solution")

    # create linear FEM solution plot
    linear_solver = LinearFEMEllipticPDE(
        _test_problem_f,
        triang,
        _test_problem_kappa,
        _test_problem_kappa_zero,
        _test_problem_g_dir,
    )
    linear_solution = linear_solver.solve()
    linear_ax.plot_trisurf(x, y, triang._tri_idx, linear_solution, cmap='viridis')
    linear_ax.set_title("Linear FEM")

    # create quadratic FEM solution plot
    quad_triang = TriangulationQuad.from_triangulation(triang)
    quadratic_solver = QuadraticFEMEllipticPDE(
        _test_problem_f,
        quad_triang,
        _test_problem_kappa,
        _test_problem_kappa_zero,
        _test_problem_g_dir,
    )
    quadratic_solution = quadratic_solver.solve()
    quad_ax.plot_trisurf(quad_triang._points[:, 0], quad_triang._points[:, 1], quad_triang._tri_idx[:, :3], quadratic_solution, cmap='viridis')
    quad_ax.set_title("Quadratic FEM")

def show_plots():
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    sol = fig.add_subplot(gs[0, 0], projection='3d')
    exact_plot = fig.add_subplot(gs[0, 1], projection='3d')
    linear_plot = fig.add_subplot(gs[1, 0], projection='3d')
    quadratic_plot = fig.add_subplot(gs[1, 1], projection='3d')

    plot_finite_elements_problem(
        exact_plot,
        linear_plot,
        quadratic_plot,
        triang=read_msh('assets/meshes/circular_ring_00.msh')
    )

    plt.tight_layout()
    plt.savefig('figures/ex04d.pdf')
    plt.show()

show_plots()
