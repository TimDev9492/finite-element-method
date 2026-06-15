import numpy as np
from src.mesh_tools.mesh_tools import read_msh, TriangulationQuad

T = read_msh('assets/meshes/circular_ring_00.msh')
QT = TriangulationQuad.from_triangulation(T)

T.plot()
QT.plot()
