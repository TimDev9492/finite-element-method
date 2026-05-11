from utils.typing import Vector

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from finite_elements.poisson import LinearFEMPoissonPDE
from utils.common import time_function
from mesh_tools.mesh_tools import Triangulation, read_msh

def _test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side off poisson PDE
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    return 15*x*y*(1 - 0.25*x*x - y*y) - 0.5*x*x*x*y - 8*x*y*y*y

def _test_problem_u(vec: Vector) -> np.ndarray | float:
    vec = np.asarray(vec)
    x = vec[..., 0]
    y = vec[..., 1]
    return x*y*(1 - 0.25*x*x - y*y)**2

def plot_finite_elements_problem(actual_ax: Axes3D, numerical_ax: Axes3D, triang: Triangulation):
    '''
    solution: ex02 (b)
    '''
    pde = LinearFEMPoissonPDE(
        _test_problem_f,
        triang
    )

    numerical_solution = pde.solve()

    # compute real solution
    x, y = triang._points[:, 0], triang._points[:, 1]
    z = _test_problem_u(triang._points)

    # create the plots
    actual_ax.plot_trisurf(x, y, z, cmap='viridis')
    actual_ax.set_title("Actual solution")

    numerical_ax.plot_trisurf(x, y, numerical_solution, cmap='viridis')
    numerical_ax.set_title(f"Numerical solution ({pde.m} triangles)")

# @time_function
# def convergence_plot(ax: Axes, N_s: list[int] = [10, 14, 20, 28, 40, 56, 80]):
#     '''
#     solution: ex01 (c)
#     '''
#     h_s = []
#     max_errors = []
#     for N in N_s:
#         pde = FiniteDifferencesPoisson(
#             _test_problem_f,
#             _test_problem_g,
#             N
#         )

#         actual = evaluate_on_grid(pde.x_s, pde.y_s, _test_problem_solution)
#         numerical = pde.solve()

#         h_s.append(pde.h)
#         max_errors.append(np.max(np.abs(actual - numerical)))

#     # sort h_s first
#     idx = np.argsort(h_s)
#     h_s = np.array(h_s)[idx]
#     max_errors = np.array(max_errors)[idx]

#     # create loglog plot
#     ax.loglog(h_s, max_errors, marker='o', linestyle='-', label='Finite difference method')

#     # pick an anchor point (e.g. first point)
#     h0 = h_s[0]
#     e0 = max_errors[0]

#     h_ref = np.array([h_s.min(), h_s.max()])

#     # slope 1 reference line: e = C * h
#     C1 = e0 / h0
#     ax.loglog(h_ref, C1 * h_ref, '--', label='O(h)')

#     # slope 2 reference line: e = C * h^2
#     C2 = e0 / (h0**2)
#     ax.loglog(h_ref, C2 * h_ref**2, '--', label='O(h^2)')

#     # slope 3 reference line: e = C * h^3
#     C2 = e0 / (h0**3)
#     ax.loglog(h_ref, C2 * h_ref**3, '--', label='O(h^3)')

#     ax.set_xlabel('h')
#     ax.set_ylabel('Max error')
#     ax.set_title('Convergence plot')
#     ax.grid(True, which='both', linestyle='--')
#     ax.legend()

def show_plots():
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2)

    top_left = fig.add_subplot(gs[0, 0], projection='3d')
    top_right = fig.add_subplot(gs[0, 1], projection='3d')
    lower_row = fig.add_subplot(gs[1, :])

    filename = 'assets/meshes/ellipse05.msh'
    tri = read_msh(filename)

    plot_finite_elements_problem(top_left, top_right, tri)
    # convergence_plot(lower_row, N_s=[10, 14, 20, 28, 40, 56, 80])

    plt.tight_layout()
    plt.show()

show_plots()
