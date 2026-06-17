from src.utils.typing import Vector

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from src.finite_elements.elliptic_pde_linear import LinearFEMEllipticPDE
from src.finite_elements.elliptic_pde_quad import QuadraticFEMEllipticPDE
from src.utils.common import triangle_area
from src.mesh_tools.mesh_tools import Triangulation, TriangulationQuad, read_msh
from src.utils.common import draw_convergence_plot, time_function

'''
ex04 (g):
You could simply treat the midpoints of the boundary edges as edges on the boundary
themselves, however the evaluated g_dir on those points would be inaccurate and
cause larger errors in the computed solution. The error plot changes to an order
of convergence similar to linear FEM (around 2).
'''

def approx_L2_error_linear(pde: LinearFEMEllipticPDE, solution: np.ndarray, numerical_solution: np.ndarray) -> float:
    '''
    Calculate the approximation for the error in the L2 norm for linear FEM by using quadrature formula
    '''
    # approximate the error using a quadrature formula
    approx_error = 0
    for k in range(pde.m):
        tri_idx = pde.triang._tri_idx[k]
        points = pde.triang._points[tri_idx]
        residuals_squared = (solution[tri_idx] - numerical_solution[tri_idx])**2
        approx_error += triangle_area(points[0], points[1], points[2]) / 3 * np.sum(residuals_squared)
    return np.sqrt(approx_error)

def approx_L2_error_quad(pde: QuadraticFEMEllipticPDE, solution: np.ndarray, numerical_solution: np.ndarray) -> float:
    '''
    Calculate the approximation for the error in the L2 norm for quadratic FEM using enhanced quadrature formula
    '''
    approx_error = 0
    for k in range(pde.m):
        tri_idx = pde.triang._tri_idx[k]
        vertex_idx = tri_idx[:3]
        midpoint_idx = tri_idx[3:]
        vertices = pde.triang._points[vertex_idx]
        residuals_squared = (solution[midpoint_idx] - numerical_solution[midpoint_idx])**2
        approx_error += triangle_area(vertices[0], vertices[1], vertices[2]) / 3 * np.sum(residuals_squared)
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
    # vec = np.asarray(vec)
    # x, y = vec[..., 0], vec[..., 1]
    # radius = x*x + y*y
    # on_gamma_1 = radius > 0.75
    # # on_gamma_1 = np.isclose(x*x + y*y, 1)
    # return np.where(
    #     on_gamma_1,
    #     np.exp(-x*x - 1),
    #     np.exp(-x*x - 0.25)
    # )
    return _test_problem_u(vec)

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
    ex04 (d)
    '''
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
    print(f"Approximate error for linear FEM: {approx_L2_error_linear(linear_solver, solution, linear_solution):.3e}")

    # create quadratic FEM solution plot
    quad_triang = TriangulationQuad.from_triangulation(triang)
    quadratic_solver = QuadraticFEMEllipticPDE(
        _test_problem_f,
        quad_triang,
        _test_problem_kappa,
        _test_problem_g_dir,
        use_sparse=True,
    )
    solution = _test_problem_u(quad_triang._points)
    quadratic_solution = quadratic_solver.solve()
    quad_ax.plot_trisurf(quad_triang._points[:, 0], quad_triang._points[:, 1], quad_triang._tri_idx[:, :3], quadratic_solution, cmap='viridis')
    quad_ax.set_title("Quadratic FEM")
    print(f"Approximate error for quadratic FEM: {approx_L2_error_quad(quadratic_solver, solution, quadratic_solution):.3e}")

def compute_conv_work_prec(paths: list[str], conv_ax: Axes, work_prec_ax: Axes):
    '''
    ex04 (e,f)
    '''
    linear_errors = []
    quad_errors = []
    linear_runtimes = []
    quad_runtimes = []
    hmax = []

    header = [('h', 8), ('linear err', 3), ('slope', 5), ('quadr err', 3), ('slope', 0)]
    header_str = ''
    for (desc, padding) in header:
        header_str += desc + ' ' * padding
    print()
    print()
    print(header_str)
    for path in paths:
        triang = read_msh(path)
        quad_triang = TriangulationQuad.from_triangulation(triang)

        # print(f'Computing linear FEM for mesh {path} ...')
        linear_solver = LinearFEMEllipticPDE(
            _test_problem_f,
            triang,
            _test_problem_kappa,
            _test_problem_kappa_zero,
            _test_problem_g_dir
        )
        start = perf_counter_ns()
        linear_solution = linear_solver.solve()
        linear_runtimes.append((perf_counter_ns() - start) / 1e9)
        linear_errors.append(approx_L2_error_linear(
            pde=linear_solver,
            solution=_test_problem_u(triang._points),
            numerical_solution=linear_solution,
            ))
        hmax.append(triang.get_hmax())
        
        # print(f'Computing quadratic FEM for mesh {path} ...')
        quad_solver = QuadraticFEMEllipticPDE(
            _test_problem_f,
            quad_triang,
            _test_problem_kappa,
            _test_problem_g_dir,
            use_sparse=True,
        )
        start = perf_counter_ns()
        quad_solution = quad_solver.solve()
        quad_runtimes.append((perf_counter_ns() - start) / 1e9)
        quad_errors.append(approx_L2_error_quad(
            pde=quad_solver,
            solution=_test_problem_u(quad_triang._points),
            numerical_solution=quad_solution,
            ))
    

        linear_slope = quadratic_slope = None
        if len(hmax) > 1:
            linear_slope = (np.log(linear_errors[-1]) - np.log(linear_errors[-2])) / (np.log(hmax[-1]) - np.log(hmax[-2]))
            quadratic_slope = (np.log(quad_errors[-1]) - np.log(quad_errors[-2])) / (np.log(hmax[-1]) - np.log(hmax[-2]))
        values = [
            f'{hmax[-1]:.3f}',
            f'{linear_errors[-1]:.3e}',
            f'{linear_slope:.3f}' if linear_slope is not None else 'N/A',
            f'{quad_errors[-1]:.3e}',
            f'{quadratic_slope:.3f}' if quadratic_slope is not None else 'N/A'
        ]
        value_str = ''
        for (header_str, padding), value in zip(header, values):
            space_left = padding + len(header_str) - len(value)
            value_str += value + ' ' * space_left
        print(value_str)

    # draw convergence plot
    draw_convergence_plot(
        conv_ax,
        hmax,
        linear_errors,
        display_orders=[2, 3],
        title='Convergence plot',
        xlabel='hmax',
        ylabel='L2 error',
        data_label='Linear FEM',
    )
    draw_convergence_plot(
        conv_ax,
        hmax,
        quad_errors,
        display_orders=[3, 4],
        title='Convergence plot',
        xlabel='hmax',
        ylabel='L2 error',
        data_label='Quadratic FEM',
        color='red',
    )

    # draw work-precision plot
    draw_convergence_plot(
        work_prec_ax,
        linear_runtimes,
        linear_errors,
        display_orders=[],
        title='Work-Precision',
        xlabel='Runtime (s)',
        ylabel='L2 error',
        data_label='Linear FEM',
    )
    draw_convergence_plot(
        work_prec_ax,
        quad_runtimes,
        quad_errors,
        display_orders=[],
        title='Work-Precision',
        xlabel='Runtime (s)',
        ylabel='L2 error',
        data_label='Quadratic FEM',
        color='red',
    )

@time_function
def show_plots():
    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 3)

    exact_plot = fig.add_subplot(gs[0, 0], projection='3d')
    linear_plot = fig.add_subplot(gs[0, 1], projection='3d')
    quadratic_plot = fig.add_subplot(gs[0, 2], projection='3d')

    plot_finite_elements_problem(
        exact_plot,
        linear_plot,
        quadratic_plot,
        triang=read_msh('assets/meshes/circular_ring_02.msh')
    )

    plt.tight_layout()
    plt.savefig('figures/ex04d.pdf')
    # plt.show()

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1)

    conv_plot = fig.add_subplot(gs[0, 0])
    work_prec_plot = fig.add_subplot(gs[1, 0])

    compute_conv_work_prec(
        # implementation allows for loading all meshes, runtime on my machine:
        #  - meshes 00 to 06: ~36s  (half a minute)
        #  - meshes 00 to 08: ~195s (3+ minutes)
        [f'assets/meshes/circular_ring_{no:02d}.msh' for no in range(6+1)],
        conv_plot,
        work_prec_plot,
    )

    plt.tight_layout()
    plt.savefig('figures/ex04ef.pdf')
    # plt.show()

show_plots()
