from typing import Tuple, Callable
from src.utils.typing import VecNumMap, Vec2NumMap, Vector, VecMatrixMap

from src.utils.common import triangle_area
from src.mesh_tools.mesh_tools import Triangulation

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import cg

class LinearFEMEllipticPDE:
    '''
    This class holds all values and parameters needed to describe
    and solve general elliptic PDEs using the linear finite element method

    IMPORTANT: g_neu takes in two points and should return the expression
    evaluated at the midpoint!
    '''
    def __init__(
            self,
            f: VecNumMap,
            triang: Triangulation,
            kappa: VecMatrixMap = lambda v: np.eye(2),
            kappa_zero: VecNumMap = lambda v: 0,
            g_dir: VecNumMap = lambda v: 0,
            g_neu: Vec2NumMap = lambda u, v: 0,
            use_sparse: bool = False,
    ):
        self.f = f
        self.triang = triang
        self.N = len(triang._points)
        self.m = len(triang._tri_idx)

        self.kappa = kappa
        self.kappa_zero = kappa_zero
        self.g_dir = g_dir
        self.g_neu = g_neu

        self.use_sparse = use_sparse

    def _compute_A_b(self) -> Tuple[sparse.csr_array | np.ndarray, np.ndarray]:
        '''
        Compute matrix A and vector b to solve Ax=b
        '''
        b = np.zeros(self.N)
        A = sparse.lil_matrix((self.N, self.N)) if self.use_sparse else np.zeros((self.N, self.N))
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
                if (tf_mask_dirichlet := np.isin(global_point_idx, P_dirichlet)).sum() >= 1:
                    for j in range(len(tf_mask_dirichlet)):
                        if tf_mask_dirichlet[j]:
                            b_i_k -= self.g_dir(points[j])*a_i[j]
                
                tf_mask_neumann = np.isin(global_point_idx, P_neumann)
                # check if at least two vertices neumann edge adjacent
                if tf_mask_neumann.sum() >= 2:
                    for (m, n) in [(0,1), (1,2), (0,2)]:
                        if not (i == m or i == n):
                            continue
                        gm, gn = global_point_idx[m], global_point_idx[n]
                        # check if edge nodes are both neumann
                        if tf_mask_neumann[m] and tf_mask_neumann[n]:
                            # check if edge is neumann
                            if np.any(
                                ((self.triang._edges_neu[:, 0] == gm) & (self.triang._edges_neu[:, 1] == gn)) |
                                ((self.triang._edges_neu[:, 0] == gn) & (self.triang._edges_neu[:, 1] == gm))
                            ):
                                b_i_k += 0.5*np.linalg.norm(points[m] - points[n])*self.g_neu(points[m], points[n])

                b[global_point_idx[i]] += b_i_k
        
        if isinstance(A, sparse.lil_matrix):
            A = A.tocsr()
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

        v = None
        if not self.use_sparse:
            if isinstance(A, np.ndarray):
                v = np.linalg.solve(A, b)
            else:
                raise RuntimeError('Unexpected program state. Exiting...')
        else:
            v, info = cg(A, b, rtol=1e-10)
            if info != 0:
                raise RuntimeError(f"CG failed with info={info}")

        # add dirichlet values on the boundary back in
        v[boundary_idx] = np.array([self.g_dir(self.triang._points[i]) for i in boundary_idx])

        return v
