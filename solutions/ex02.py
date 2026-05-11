from utils.typing import Vector

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from finite_elements.poisson import LinearFEMPoissonPDE
from utils.common import time_function, triangle_area
from mesh_tools.mesh_tools import Triangulation, read_msh

def _test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side off poisson PDE
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    return 15*x*y*(1 - 0.25*x*x - y*y) - 0.5*x*x*x*y - 8*x*y*y*y

def _test_problem_u(vec: Vector) -> np.ndarray:
    vec = np.asarray(vec)
    x = vec[..., 0]
    y = vec[..., 1]
    return x*y*(1 - 0.25*x*x - y*y)**2

def approx_L2_error(pde: LinearFEMPoissonPDE, solution: np.ndarray, numerical_solution: np.ndarray) -> float:
    # approximate the error using a quadrature formula
    approx_error = 0
    for k in range(pde.m):
        tri_idx = pde.triang._tri_idx[k]
        points = pde.triang._points[tri_idx]
        residuals_squared = (solution[tri_idx] - numerical_solution[tri_idx])**2
        approx_error += triangle_area(points[0], points[1], points[2]) / 3 * np.sum(residuals_squared)
    return np.sqrt(approx_error)

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
    solution = _test_problem_u(triang._points)

    # create the plots
    actual_ax.plot_trisurf(x, y, solution, cmap='viridis')
    actual_ax.set_title("Actual solution")

    numerical_ax.plot_trisurf(x, y, numerical_solution, cmap='viridis')
    numerical_ax.set_title(f"Numerical solution (N = {pde.N}, m = {pde.m})")

    # approximate the error using a quadrature formula
    print(f"Approximate error (N={pde.N}): {approx_L2_error(pde, solution, numerical_solution)}")

@time_function
def convergence_plot(ax: Axes, filename_template: str, mesh_numbers: range):
    '''
    solution: ex02 (c)
    '''
    L2_errors = []
    hmax_s = []
    for mesh_no in mesh_numbers:
        filename = filename_template.format(mesh_no)
        print(f'Computing solution for {filename}...')
        tri = read_msh(filename)
        pde = LinearFEMPoissonPDE(
            _test_problem_f,
            tri
        )
        numerical_solution = pde.solve()

        # compute real solution
        solution = _test_problem_u(tri._points)

        L2_errors.append(approx_L2_error(pde, solution, numerical_solution))
        hmax_s.append(tri.get_hmax())

    # sort hmax_s first
    idx = np.argsort(hmax_s)
    hmax_s = np.array(hmax_s)[idx]
    L2_errors = np.array(L2_errors)[idx]

    # create loglog plot
    ax.loglog(hmax_s, L2_errors, marker='o', linestyle='-', label='Linear FEM')

    # pick an anchor point (e.g. first point)
    h0 = hmax_s[0]
    e0 = L2_errors[0]

    h_ref = np.array([hmax_s.min(), hmax_s.max()])

    # slope 1 reference line: e = C * h
    C1 = e0 / h0
    ax.loglog(h_ref, C1 * h_ref, '--', label='O(h)')

    # slope 2 reference line: e = C * h^2
    C2 = e0 / (h0**2)
    ax.loglog(h_ref, C2 * h_ref**2, '--', label='O(h^2)')

    # slope 3 reference line: e = C * h^3
    C2 = e0 / (h0**3)
    ax.loglog(h_ref, C2 * h_ref**3, '--', label='O(h^3)')

    ax.set_xlabel('h')
    ax.set_ylabel('Max error')
    ax.set_title('Convergence plot')
    ax.grid(True, which='both', linestyle='--')
    ax.legend()

def show_plots():
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2)

    top_left = fig.add_subplot(gs[0, 0], projection='3d')
    top_right = fig.add_subplot(gs[0, 1], projection='3d')
    lower_row = fig.add_subplot(gs[1, :])

    filename = 'assets/meshes/ellipse05.msh'
    tri = read_msh(filename)

    plot_finite_elements_problem(top_left, top_right, tri)
    convergence_plot(lower_row, filename_template='assets/meshes/ellipse{:02d}.msh', mesh_numbers=range(0, 10+1, 1))

    plt.tight_layout()
    plt.show()

show_plots()
