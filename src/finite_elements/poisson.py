from typing import Tuple
from src.utils.typing import VecNumMap, Vector

from src.utils.common import triangle_area
from src.pdes.poisson_pde import PoissonPDE
from src.mesh_tools.mesh_tools import Triangulation

import numpy as np

class LinearFEMPoissonPDE(PoissonPDE):
    '''
    This class holds all values and parameters needed to describe
    and solve the poisson PDE using the linear finite element method
    '''
    def __init__(self, f: VecNumMap, triang: Triangulation):
        super().__init__(f, lambda v: 0)
        self.triang = triang
        self.N = len(triang._points)
        self.m = len(triang._tri_idx)

    def _compute_A_b(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute matrix A and vector b to solve Ax=b
        '''
        b = np.zeros((self.N, 1))
        A = np.zeros((self.N, self.N))
        for k in range(self.m):
            global_point_idx = self.triang._tri_idx[k]
            points = self.triang._points[global_point_idx]
            center = (points[0] + points[1] + points[2]) / 3
            tri_area = triangle_area(points[0], points[1], points[2])
            for i in range(3):
                # compute b_k
                b_i_k = tri_area / 3 * self.f(center)
                b[global_point_idx[i]] += b_i_k
                for j in range(3):
                    # compute a_k
                    x = points[:, 0]
                    y = points[:, 1]
                    # compute alphas, betas and gammas by inverting the matrix
                    # [
                    #   1  x[0]  y[0]
                    #   1  x[1]  y[1]
                    #   1  x[2]  y[2]
                    # ]
                    # using cramer's rule
                    det_point_matrix = x[1]*y[2] + x[0]*y[1] + x[2]*y[0] - x[1]*y[0] - x[2]*y[1] - x[0]*y[2]
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
                    a_ij_k = tri_area * (betas[i] * betas[j] + gammas[i] * gammas[j])
                    A[global_point_idx[i], global_point_idx[j]] += a_ij_k
        return (A, b)
    
    def solve(self) -> Vector:
        '''
        Compute the solution using np.linalg.solve

        Note: Can be optimized by using sparse matrix A and cg-method for example
        '''
        A, b = self._compute_A_b()

        # enforce dirichlet boundary conditions
        # by setting columns and rows of boundary
        # nodes to zero
        mask = self.triang.outer_point_mask()
        boundary_idx = np.where(mask)[0]
        A[boundary_idx, :] = 0
        A[:, boundary_idx] = 0
        A[boundary_idx, boundary_idx] = 1
        b[boundary_idx] = self.g(self.triang._points[boundary_idx])

        v = np.linalg.solve(A, b)

        return v.reshape(v.shape[0],)
