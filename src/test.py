import numpy as np
from mesh_tools.mesh_tools import read_msh

T = read_msh('assets/meshes/circular_ring_coarse_out_DIR_inn_DIR.msh')
T.plot()
