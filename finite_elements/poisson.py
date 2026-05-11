from typing import Tuple
from utils.typing import VecNumMap, Vector

from utils.common import triangle_area
from pdes.poisson_pde import PoissonPDE
from mesh_tools.mesh_tools import Triangulation

import numpy as np

class LinearFEMPoissonPDE(PoissonPDE):
    def __init__(self, f: VecNumMap, triang: Triangulation):
        super().__init__(f, lambda v: 0)
        self.triang = triang
        self.N = len(triang._points)
        self.m = len(triang._tri_idx)
        
    def _compute_a_k(self, k: int, i: int, j: int) -> float:
        point_idx = self.triang._tri_idx[k]
        local_index_i = np.where(point_idx == i)[0]
        local_index_j = np.where(point_idx == j)[0]
        if len(local_index_i) == 0 or len(local_index_j) == 0:
            return 0
        local_index_i = local_index_i[0]
        local_index_j = local_index_j[0]
        points = self.triang._points[self.triang._tri_idx[k]]
        x = points[:, 0]
        y = points[:, 1]
        det_point_matrix = x[1]*y[2] + x[0]*y[1] + x[2]*y[0] - x[1]*y[0] - x[2]*y[1] - x[0]*y[2]
        # compute alphas, betas and gammas by inverting the matrix with column
        # of three ones and the points right next to it, using cramer's rule
        # alphas = [
        #     +1 / det_point_matrix * (x[1]*y[2] - x[2]*y[1]),
        #     -1 / det_point_matrix * (x[0]*y[2] - x[2]*y[0]),
        #     +1 / det_point_matrix * (x[0]*y[1] - x[1]*y[0]),
        # ]
        betas = [
            -1 / det_point_matrix * (y[2] - y[1]),
            +1 / det_point_matrix * (y[2] - y[0]),
            -1 / det_point_matrix * (y[1] - y[0]),
        ]
        gammas = [
            +1 / det_point_matrix * (x[2] - x[1]),
            -1 / det_point_matrix * (x[2] - x[0]),
            +1 / det_point_matrix * (x[1] - x[0]),
        ]

        return triangle_area(points[0], points[1], points[2]) * (
            betas[local_index_j] * betas[local_index_j] + gammas[local_index_i] * gammas[local_index_j])

    def _compute_A_b(self) -> Tuple[np.ndarray, np.ndarray]:
        b = np.zeros((self.N, 1))
        A = np.zeros((self.N, self.N))
        for k in range(self.m):
            global_point_idx = self.triang._tri_idx[k]
            points = self.triang._points[global_point_idx]
            center = (points[0] + points[1] + points[2]) / 3
            for i in range(3):
                b_i_k = triangle_area(points[0], points[1], points[2]) / 3 * self.f(center)
                b[global_point_idx[i]] += b_i_k
                for j in range(3):
                    a_ij_k = self._compute_a_k(k, global_point_idx[i], global_point_idx[j])
                    A[global_point_idx[i], global_point_idx[j]] += a_ij_k
        return (A, b)
    
    def solve(self) -> Vector:
        A, b = self._compute_A_b()
        v = np.linalg.solve(A, b)
        return v.reshape(v.shape[0],)
