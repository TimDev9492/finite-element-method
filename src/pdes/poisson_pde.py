from src.utils.typing import VecNumMap

class PoissonPDE():
    '''
    This class holds all parameters needed to describe a poisson PDE of the form:
        -\\Delta u(x,y) = f(x,y)    for (x,y) \\in \\Omega = (0,1)^2
        u(x,y) = g(x,y)             for (x,y) \\in \\delta \\Omega
    '''
    def __init__(self, f: VecNumMap, g: VecNumMap):
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
