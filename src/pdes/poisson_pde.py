from src.utils.typing import VecNumMap

class PoissonPDE():
    '''
    This class holds all parameters needed to describe a poisson PDE of the form:
        -\\Delta u(x,y) = f(x,y)    for (x,y) \\in \\Omega
        u(x,y) = g(x,y)             for (x,y) \\in \\delta \\Omega
    '''
    def __init__(self, f: VecNumMap, g: VecNumMap):
        '''
        Set parameters for poisson PDE
        
        :param f: right hand side of the PDE
        :type f: VecNumMap
        :param g: desired solution on the boundary
        :type g: VecNumMap
        '''
        self.f = f
        self.g = g
