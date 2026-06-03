from typing import Tuple, Callable
from src.utils.typing import VecNumMap, Vector, VecMatrixMap

from src.utils.common import triangle_area
from src.finite_elements.poisson import LinearFEMPoissonPDE
from src.mesh_tools.mesh_tools import Triangulation

import numpy as np

# TODO: rename
class LinearFEMPoissonPDEVariational(LinearFEMPoissonPDE):
    '''
    This class holds all values and parameters needed to describe
    and solve the poisson PDE using the linear finite element method

    IMPORTANT: g_neu takes in two points and should return the expression
    evaluated at the midpoint!
    '''
    def __init__(
            self,
            f: VecNumMap,
            triang: Triangulation,
            kappa: VecMatrixMap = lambda v: np.ones(3),
            kappa_zero: VecNumMap = lambda v: 0,
            g_dir: VecNumMap = lambda v: 0,
            g_neu: Callable[[Vector, Vector], float] = lambda u, v: 0,
    ):
        super().__init__(f, triang)
        self.kappa = kappa
        self.kappa_zero = kappa_zero
        self.g_dir = g_dir
        self.g_neu = g_neu

    def _compute_A_b(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute matrix A and vector b to solve Ax=b
        '''
        b = np.zeros((self.N, 1))
        A = np.zeros((self.N, self.N))
        T_area = np.empty(self.m)

        P_dirichlet = np.unique(self.triang._edges_dir)
        P_neumann = np.unique(self.triang._edges_neu)

        for k in range(self.m):
            global_point_idx = self.triang._tri_idx[k]
            points = self.triang._points[global_point_idx]
            center = (points[0] + points[1] + points[2]) / 3
            T_area[k] = triangle_area(points[0], points[1], points[2])
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
            for i in range(3):
                a_i = []
                for j in range(3):
                    a_ij_k = T_area[k] * (np.inner(np.array([betas[i], gammas[i]]), self.kappa(center) @ np.array([betas[j], gammas[j]]))
                        + self.kappa_zero(center) / 9) # (kappa * phi_i * phi_j)(center) = kappa(center) * 1/3 * 1/3
                    a_i.append(a_ij_k)
                    A[global_point_idx[i], global_point_idx[j]] += a_ij_k

                # compute b_k
                b_i_k = T_area[k] / 3 * self.f(center)
                # if no triangle vertex is in dirichlet points
                if (tf_mask_dirichlet := np.isin(global_point_idx, P_dirichlet)).sum() == 0:
                    for j in range(len(tf_mask_dirichlet)):
                        if tf_mask_dirichlet[i]:
                            b_i_k -= self.g_dir(points[j])*a_i[j]
                
                tf_mask_neumann = np.isin(global_point_idx, P_neumann)
                # check if at least two vertices neumann edge adjacent
                if tf_mask_neumann.sum() >= 2:
                    for (m, n) in [(0,1), (1,2), (0,2)]:
                        # check if edge nodes are both neumann
                        if tf_mask_neumann[m] and tf_mask_neumann[n]:
                            # check if edge is neumann
                            if np.any(
                                ((self.triang._edges_neu[:, 0] == m) & (self.triang._edges_neu[:, 1] == n)) |
                                ((self.triang._edges_neu[:, 0] == n) & (self.triang._edges_neu[:, 1] == m))
                            ):
                                b_i_k += 0.5*np.linalg.norm(points[m] - points[n])*self.g_neu(points[m], points[n])

                b[global_point_idx[i]] += b_i_k

        return (A, b)
    
    def solve(self) -> Vector:
        '''
        Compute the solution using np.linalg.solve

        Note: Can be optimized by using sparse matrix A and cg-method for example
        '''
        A, b = self._compute_A_b()

        # enforce dirichlet boundary conditions
        # by setting columns and rows of dirichlet
        # nodes to zero
        boundary_idx = np.unique(self.triang._edges_dir)
        A[boundary_idx, :] = 0
        A[:, boundary_idx] = 0
        A[boundary_idx, boundary_idx] = 1
        b[boundary_idx] = 0

        v = np.linalg.solve(A, b)

        # return vector as 1D-array, not as column vector
        return v.reshape(v.shape[0],)
