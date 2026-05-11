from src.utils.typing import Vector

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from src.finite_differences.poisson import FiniteDifferencesPoisson
from src.utils.common import evaluate_on_grid, time_function, draw_convergence_plot

'''
Notes on problem (d):

The most basic method to solve the matrix equation to approximate
the pde solution uses LU decomposition, which has the same runtime as the
cholesky decomposition which is used in this implementation. Note that this
is only possible because the discrete matrix -A_h is symmetric and positive definite.
Both decompositions have a runtime of O(n^3). The other operations that are part of
the FiniteDifferencesPoisson.solve() method have negligible runtime.
This would mean that increasing N by a factor of 10 from N=80 to N=800
would increase the runtime by a factor of 10^3 = 1000. Using this approach
would take quite a lot of time.

Note that this code uses sparse matrices by default and solves the linear system
using scipy's spsolve method which runs significantly faster, so N=800 is quite
doable with this code. Dense matrices can be enable by using FiniteDifferencesPoisson(..., use_dense=True).

I also included an option to use the cg-method (conjugate gradient)
to solve the system. See FiniteDifferencesPoisson.solve(use_cg=True)
'''

def _test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side off poisson PDE
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]
    return np.sin(x*y*np.pi) * np.pi**2 * (x**2 + y**2)

def _test_problem_g(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe the value of the desired solution on the boundary
    '''
    vec = np.asarray(vec)
    x, y = vec[0], vec[1]

    out = np.zeros_like(x, dtype=float)

    mask_x = (x == 1)
    mask_y = (y == 1)

    out[mask_x] = np.sin(y[mask_x] * np.pi)
    out[mask_y] = np.sin(x[mask_y] * np.pi)

    return out

def _test_problem_solution(x, y):
    '''
    Helper function for this test problem
    '''
    return np.sin(x * y * np.pi)

def plot_finite_differences_problem(actual_ax: Axes3D, numerical_ax: Axes3D, N: int = 20):
    '''
    solution: ex01 (b)
    '''
    pde = FiniteDifferencesPoisson(
        _test_problem_f,
        _test_problem_g,
        N
    )

    numerical_solution = pde.solve()

    # compute real solution
    x, y = np.meshgrid(pde.x_s, pde.y_s)
    actual_solution = evaluate_on_grid(pde.x_s, pde.y_s, _test_problem_solution)

    max_diff = np.max(np.abs(actual_solution - numerical_solution))
    print(f"Max error on grid (N={pde.N}): {max_diff}")

    # create the plots
    actual_ax.plot_surface(x, y, actual_solution)
    actual_ax.set_title("Actual solution")

    numerical_ax.plot_surface(x, y, numerical_solution)
    numerical_ax.set_title(f"Numerical solution (N={pde.N})")

@time_function
def convergence_plot(ax: Axes, N_s: list[int] = [10, 14, 20, 28, 40, 56, 80]):
    '''
    solution: ex01 (c)
    '''
    h_s = []
    max_errors = []
    for N in N_s:
        pde = FiniteDifferencesPoisson(
            _test_problem_f,
            _test_problem_g,
            N
        )

        actual = evaluate_on_grid(pde.x_s, pde.y_s, _test_problem_solution)
        numerical = pde.solve()

        h_s.append(pde.h)
        max_errors.append(np.max(np.abs(actual - numerical)))

    draw_convergence_plot(ax, h_s, max_errors, display_orders=[1, 2, 3], title='Convergence plot', xlabel='h', ylabel='Max error', data_label='Finite difference method')

def show_plots():
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    top_left = fig.add_subplot(gs[0, 0], projection='3d')
    top_right = fig.add_subplot(gs[0, 1], projection='3d')
    lower_row = fig.add_subplot(gs[1, :])

    plot_finite_differences_problem(top_left, top_right, N=20)
    convergence_plot(lower_row, N_s=[10, 14, 20, 28, 40, 56, 80])

    plt.tight_layout()
    plt.savefig('figures/ex01.pdf')
    plt.show()

show_plots()
