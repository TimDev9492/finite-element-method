from typing import Tuple, Callable

import time
import numpy as np
from scipy.sparse import diags, spmatrix
from functools import wraps

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
