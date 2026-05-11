from typing import Tuple

import time
import numpy as np
from scipy.sparse import diags, spmatrix
from functools import wraps
from matplotlib.axes import Axes

def tridiag(upper_diag: np.ndarray, main_diag: np.ndarray, lower_diag: np.ndarray, dense: bool = True) -> np.ndarray | spmatrix:
    M = diags(
        diagonals=[upper_diag, main_diag, lower_diag],
        offsets=[1, 0, -1],
    )
    return M.todense() if dense else M

def idx_2d_1d(x: int, y: int, row_size: int) -> int:
    return y * row_size + x

def idx_1d_2d(i: int, row_size: int) -> Tuple[int, int]:
    return (i // row_size, i % row_size)

def evaluate_on_grid(x_s: np.ndarray, y_s: np.ndarray, height_map) -> np.ndarray:
    x, y = np.meshgrid(x_s, y_s)
    return height_map(x, y)

def time_function(func):
    """
    Decorator that times a function call and prints elapsed time,
    while returning the original function output unchanged.

    Usage:
    
        @time_function
        def function_to_time(x):
            return compute_something(x)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper

def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

def draw_convergence_plot(ax: Axes, x: np.ndarray | list[float], y: np.ndarray | list[float], display_orders=[1, 2, 3], title='Convergence plot', xlabel='x', ylabel='y', data_label='Data'):
    # sort x first
    idx = np.argsort(x)
    x = np.array(x)[idx]
    y = np.array(y)[idx]

    # create loglog plot
    ax.loglog(x, y, marker='x', linestyle='-', label=data_label)

    # pick an anchor point (e.g. first point)
    x0 = x[0]
    y0 = y[0]

    x_ref = np.array([x.min(), x.max()])

    for order in display_orders:
        C = y0 / (x0**order)
        ax.loglog(x_ref, C * x_ref**(order), '--', label=f'O({xlabel}^{order})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
