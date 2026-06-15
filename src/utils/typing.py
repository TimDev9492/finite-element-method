from typing import Callable

import numpy as np
from scipy.sparse import spmatrix

Vector = np.ndarray
FullMatrix = np.ndarray
SparseMatrix = spmatrix
Matrix = FullMatrix | SparseMatrix
VecVecMap = Callable[[Vector], Vector]
VecNumMap = Callable[[Vector], float | np.ndarray]
Vec2NumMap = Callable[[Vector, Vector], float | np.ndarray]
NumVecMap = Callable[[float], Vector]
VecMatrixMap = Callable[[Vector], FullMatrix]
