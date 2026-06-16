from typing import Tuple, Callable
from src.utils.typing import VecNumMap, Vec2NumMap, Vector, VecMatrixMap, Matrix

from src.utils.common import triangle_area
from src.mesh_tools.mesh_tools import TriangulationQuad

import numpy as np

class QuadraticFEMEllipticPDE:
    '''
    This class holds all values and parameters needed to describe
    and solve general elliptic PDEs using the quadratic finite element method
    using dirichlet boundary conditions
    '''
    def __init__(
            self,
            f: VecNumMap,
            triang: TriangulationQuad,
            kappa: VecMatrixMap = lambda v: np.eye(2),
            g_dir: VecNumMap = lambda v: 0,
    ):
        self.f = f
        self.triang = triang
        self.N = len(triang._points)
        self.m = len(triang._tri_idx)

        self.kappa = kappa
        self.g_dir = g_dir

        # store points for quadrature fromula
        self.xi_eta = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3],
        ])
        # z_1=(0,0), z_2=(1,0), z_3=(0,1), z_4=(0.5,0), z_5=(0.5,0.5), z_6=(0,0.5)
        self.Z = np.array([
            [1,    0,    0,    0,    0,    0],
            [1,    1,    0,    1,    0,    0],
            [1,    0,    1,    0,    0,    1],
            [1,  0.5,    0, 0.25,    0,    0],
            [1,  0.5,  0.5, 0.25, 0.25, 0.25],
            [1,    0,  0.5,    0,    0, 0.25],
        ])

    def _solve_right_matrix(self, A: np.ndarray) -> np.ndarray:
        # compute the matrix C defined by the equation:
        #   A*B = C where A and C are (3,6) and B=Z^-1
        # note that this is the same as
        #   A   = C*Z
        #   A.T = Z.T*C.T
        # which is the same as solving three linear systems of equations
        # to get the three rows of C
        return np.vstack([
            np.linalg.solve(self.Z.T, A[0, :]),
            np.linalg.solve(self.Z.T, A[1, :]),
            np.linalg.solve(self.Z.T, A[2, :])
        ])

    def _compute_A_b(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute matrix A and vector b to solve Ax=b
        '''
        b = np.zeros(self.N)
        A = np.zeros((self.N, self.N))

        # calculate phi_hat d_r phi and d_s phi
        phi = self._solve_right_matrix(np.array([
            [1, self.xi_eta[0, 0], self.xi_eta[0, 1], self.xi_eta[0, 0]**2, self.xi_eta[0, 0]*self.xi_eta[0, 1], self.xi_eta[0, 1]**2],
            [1, self.xi_eta[1, 0], self.xi_eta[1, 1], self.xi_eta[1, 0]**2, self.xi_eta[1, 0]*self.xi_eta[1, 1], self.xi_eta[1, 1]**2],
            [1, self.xi_eta[2, 0], self.xi_eta[2, 1], self.xi_eta[2, 0]**2, self.xi_eta[2, 0]*self.xi_eta[2, 1], self.xi_eta[2, 1]**2],
        ]))
        dr_phi = self._solve_right_matrix(np.array([
            [0, 1, 0, 2*self.xi_eta[0, 0], self.xi_eta[0, 1], 0],
            [0, 1, 0, 2*self.xi_eta[1, 0], self.xi_eta[1, 1], 0],
            [0, 1, 0, 2*self.xi_eta[2, 0], self.xi_eta[2, 1], 0],
        ]))
        ds_phi = self._solve_right_matrix(np.array([
            [0, 0, 1, 0, self.xi_eta[0, 0], 2*self.xi_eta[0, 1]],
            [0, 0, 1, 0, self.xi_eta[1, 0], 2*self.xi_eta[1, 1]],
            [0, 0, 1, 0, self.xi_eta[2, 0], 2*self.xi_eta[2, 1]],
        ]))

        P_dirichlet = np.unique(self.triang._edges_dir)

        for k in range(self.m):
            global_point_idx = self.triang._tri_idx[k]
            points = self.triang._points[global_point_idx]
            
            x = points[:, 0]
            y = points[:, 1]

            Phi_k_of_xi_eta = np.array([
                points[0] + (points[1] - points[0])*self.xi_eta[0, 0] + (points[2] - points[0])*self.xi_eta[0, 1],
                points[0] + (points[1] - points[0])*self.xi_eta[1, 0] + (points[2] - points[0])*self.xi_eta[1, 1],
                points[0] + (points[1] - points[0])*self.xi_eta[2, 0] + (points[2] - points[0])*self.xi_eta[2, 1],
            ])

            det_Jk = (x[1] - x[0])*(y[2] - y[0]) - (y[1] - y[0])*(x[2] - x[0])
            Jk_inv_T = 1 / det_Jk * np.array([
                [y[2] - y[0], y[0] - y[1]],
                [x[0] - x[2], x[1] - x[0]],
            ])

            a_k = np.zeros((6, 6))
            for i in range(6):
                b_i_k = 0
                tf_mask_dirichlet = np.isin(global_point_idx, P_dirichlet)
                for j in range(6):
                    # compute contribution to assembly matrix A
                    a_ij_k = abs(det_Jk) / 6 * (
                        np.dot(
                            Jk_inv_T @ np.array([dr_phi[0, i], ds_phi[0, i]]),
                            self.kappa(Phi_k_of_xi_eta[0]) @ Jk_inv_T @ np.array([dr_phi[0, j], ds_phi[0, j]])
                            ) +
                        np.dot(
                            Jk_inv_T @ np.array([dr_phi[1, i], ds_phi[1, i]]),
                            self.kappa(Phi_k_of_xi_eta[1]) @ Jk_inv_T @ np.array([dr_phi[1, j], ds_phi[1, j]])
                            ) +
                        np.dot(
                            Jk_inv_T @ np.array([dr_phi[2, i], ds_phi[2, i]]),
                            self.kappa(Phi_k_of_xi_eta[2]) @ Jk_inv_T @ np.array([dr_phi[2, j], ds_phi[2, j]])
                            )
                    )
                    a_k[i, j] = a_k[i, j] = a_ij_k
                    A[global_point_idx[i], global_point_idx[j]] += a_ij_k

                    # subtract correction term for dirichlet boundary condition
                    # (swapped indices compared to script!)
                    # computes a^k(u_star, phi^k_i) with loop over j
                    if tf_mask_dirichlet[j]:
                        b_i_k -= self.g_dir(points[j]) * a_k[j, i]

                # compute b_k
                b_i_k += abs(det_Jk) / 6 * (
                                self.f(Phi_k_of_xi_eta[0]) * phi[0, i] + 
                                self.f(Phi_k_of_xi_eta[1]) * phi[1, i] +
                                self.f(Phi_k_of_xi_eta[2]) * phi[2, i]
                            )

                # if tf_mask_dirichlet.sum() >= 1:
                #     b_i_k -= np.dot(self.g_dir(points) * a_k[:, i], tf_mask_dirichlet)

                b[global_point_idx[i]] += b_i_k

        return (A, b)
    
    def solve(self) -> Vector:
        '''
        Compute the solution using np.linalg.solve

        Note: Can be optimized by using sparse matrix A and cg-method for example
        '''
        A, b = self._compute_A_b()

        # enforce dirichlet boundary conditions
        # by removing cols and rows from matrix,
        # and re-inserting them afterwards
        boundary_idx = np.unique(self.triang._edges_dir)
        interior_mask = np.ones(self.N, dtype=bool)
        interior_mask[boundary_idx] = False
        interior_idx = np.where(interior_mask)[0]

        boundary_values = np.array([self.g_dir(self.triang._points[i]) for i in boundary_idx])

        A_ii = A[np.ix_(interior_idx, interior_idx)]
        A_ib = A[np.ix_(interior_idx, boundary_idx)]

        # remove boundary nodes
        # b_reduced = b[interior_idx] - A_ib @ boundary_values
        b_reduced = b[interior_idx]

        # solve reduced system
        u_interior = np.linalg.solve(A_ii, b_reduced)

        u = np.empty(self.N)
        u[interior_idx] = u_interior

        # add dirichlet values on the boundary back in
        u[boundary_idx] = boundary_values

        return u
