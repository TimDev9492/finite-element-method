from typing import Callable

import numpy as np
from scipy.sparse import spmatrix

Vector = np.ndarray
Matrix = np.ndarray | spmatrix
VecVecMap = Callable[[Vector], Vector]
VecNumMap = Callable[[Vector], float | np.ndarray]
NumVecMap = Callable[[float], Vector]
VecMatrixMap = Callable[[Vector], Matrix]
