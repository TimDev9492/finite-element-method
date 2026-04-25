import utils.common as utils
from utils.typing import Vector, Matrix, VecNumMap

import numpy as np
import scipy.sparse as sparse

class PoissonPDE():
    '''
    This class holds all parameters needed to describe a poisson PDE of the form:
        -\\Delta u(x,y) = f(x,y)    for (x,y) \\in \\Omega = (0,1)^2
        u(x,y) = g(x,y)             for (x,y) \\in \\delta \\Omega
    '''
    def __init__(self, f: VecNumMap, g: VecNumMap, N: int, a: float = 0, b: float = 1, use_sparse: bool = True):
        '''
        Set parameters for poisson PDE on the domain \\Omega = (a,b)^2
        
        :param f: right hand side of the PDE
        :type f: VecNumMap
        :param g: desired solution on the boundary
        :type g: VecNumMap
        :param N: number of intervals to split the axes
        :type N: int
        :param use_sparse: whether to use sparse matrices when solving the PDE,
        dramatically speeds up computation time and reduces memory usage
        :type use_sparse: bool
        '''
        self.f = f
        self.g = g
        self.N = N
        self.h = (b - a)/N
        self.a = a
        self.b = b
        self.x_s = np.linspace(self.a, self.b, self.N+1)
        self.y_s = np.linspace(self.a, self.b, self.N+1)
        self.use_dense = not use_sparse

    def _get_grid_point(self, n: int, m: int) -> Vector:
        '''
        Returns the grid point (x_n, y_m) as a vector
        '''
        return np.array([self.x_s[n], self.y_s[m]])
    
    def solve(self, use_cg: bool = False):
        '''
        Solve the PDE using the provided parameters
        
        :param use_cg: Whether to use the conjugate gradient to solve the
        underlying matrix equation
        :type use_cg: bool
        '''
        A_h = self._discrete_A()
        b_h = self._discrete_b()

        if self.use_dense:
            # solve using cholesky decomposition (-A_h is spd)
            L = np.linalg.cholesky(-1 * A_h)
            v_interm = np.linalg.solve(L, b_h)
            v_h = np.linalg.solve(L.T, v_interm)
        elif use_cg:
            # solve using conjugate gradient method
            v_h, _ = sparse.linalg.cg(-1 * A_h, b_h)
        else:
            # solve using spsolve for sparse matrices
            v_h = sparse.linalg.spsolve(-1 * A_h, b_h)

        # populate solution matrix
        solution_grid = np.empty((self.N+1, self.N+1))
        L = (self.N-1)**2
        # sopulate values inside of the domain
        for i in range(L):
            x, y = utils.idx_1d_2d(i, self.N-1)
            solution_grid[x+1, y+1] = v_h[i]
        # populate the boundary by evaluating g on those points
        for x in range(self.N+1):
            solution_grid[x, 0] = self.g(self._get_grid_point(x, 0))
            solution_grid[x, self.N] = self.g(self._get_grid_point(x, self.N))
        for y in range(1, self.N):
            solution_grid[0, y] = self.g(self._get_grid_point(0, y))
            solution_grid[self.N, y] = self.g(self._get_grid_point(self.N, y))

        return solution_grid

    def _discrete_A(self) -> Matrix:
        '''
        Constructs the matrix describing the approximation
        of the \\Delta u = f using the 5-point-star

        @see _discrete_b
        '''
        M_h = 1/self.h**2 * utils.tridiag(np.ones(self.N-2), -2 * np.ones(self.N-1), np.ones(self.N-2), dense=self.use_dense)
        I = np.eye(self.N-1)

        if self.use_dense:
            return np.kron(I, M_h) + np.kron(M_h, I)
        return sparse.kron(I, M_h) + sparse.kron(M_h, I)

    def _discrete_b(self) -> Vector:
        '''
        Constructs the right hand vector b_h for -A_h v_h = b_h

        @see _discrete_A
        '''
        L = (self.N - 1)**2
        b = np.zeros(L)

        for m in range(1, self.N):
            for n in range(1, self.N):
                grid_point = self._get_grid_point(n, m)
                f = self.f(grid_point)
                b_value = f
                x_n = grid_point[0]
                y_m = grid_point[1]
                left_border = n-1 == 0
                right_border = n+1 == self.N
                lower_border = m-1 == 0
                upper_border = m+1 == self.N
                if right_border:
                    b_value += self.g(self._get_grid_point(self.N, m)) / self.h**2
                if upper_border:
                    b_value += self.g(self._get_grid_point(n, self.N)) / self.h**2
                if left_border:
                    b_value += self.g(self._get_grid_point(0, m)) / self.h**2
                if lower_border:
                    b_value += self.g(self._get_grid_point(n, 0)) / self.h**2

                b[utils.idx_2d_1d(n-1, m-1, self.N-1)] = b_value

        return b

    