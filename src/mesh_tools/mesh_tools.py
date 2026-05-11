from dataclasses import dataclass
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import meshio

@dataclass
class Triangulation:
    def __init__(self, points, triangles, edges_dir = [], edges_neu = []):
        self._points = np.array(points)
        self._tri_idx = np.array(triangles, dtype=int)
        self._edges_dir = np.array(edges_dir, dtype=int)
        self._edges_neu = np.array(edges_neu, dtype=int)
        self._outer_point_mask = self._compute_outer_point_mask()

    def plot(self, plot = None):
        display = plot is None
        if not display:
            plot = plt
        
        x = self._points[:, 0]
        y = self._points[:, 1]
        plt.triplot(x, y, self._tri_idx)
        plt.scatter(x[self._outer_point_mask], y[self._outer_point_mask], color='red')

        if display:
            plt.show()

    def get_hmax(self):
        '''
        Simply loop over all edges and return the longest one
        '''
        longest_sqrd = -1
        for polygon in self._tri_idx:
            for i in range(len(polygon)):
                next_i = (i + 1) % len(polygon)
                point_a = self._points[polygon[i]]
                point_b = self._points[polygon[next_i]]
                delta = point_b - point_a
                dist_sqrd = delta[0]*delta[0] + delta[1]*delta[1]
                if longest_sqrd == -1 or dist_sqrd > longest_sqrd:
                    longest_sqrd = dist_sqrd
        return np.sqrt(longest_sqrd)
    
    def _compute_outer_point_mask(self):
        hull = ConvexHull(self._points)
        mask = np.zeros(len(self._points), dtype=bool)
        mask[hull.vertices] = True
        return mask

    def outer_point_mask(self):
        return self._outer_point_mask


def read_msh(filename):
    """
    Lese msh-File ein, lese alle notwendigen Daten aus und erstelle daraus eine
    Instanz der selbst definierten Klasse 'Triangulation'
    
    Parameters
    ----------
    filename : str
        filename der msh-Datei, in der die Triangulierung gespeichert ist.

    Returns
    -------
    Triangulierung
        Triangulierung als Instanz der selbst definierten Klasse 
        'Triangulierung'
    """
    
    # Endung .msh ergänzen, falls noch nicht der Fall
    if not filename.endswith('.msh'):
        filename += '.msh'
        
    mesh = meshio.read(filename)
    
    # relevante Daten auslesen:
    
    # 1.) Punkteliste mit Koordinaten aller Punkte
    points_3d = mesh.points
    points = points_3d[:, :2]
    
    # 2.) Dreiecksliste
    triangles = mesh.cells_dict['triangle']
    
    # 3.) Dirichlet-Rand-Kanten
    if 'DirichletRand' in mesh.field_data: # Checke, ob entsprechender Key existiert
        dir_tag = mesh.field_data['DirichletRand'][0]
        edge_dir_YN = ( mesh.cell_data_dict['gmsh:physical']['line'] == dir_tag )
        edges_dir = mesh.cells_dict['line'][edge_dir_YN]
    else:
        edges_dir = np.zeros([0, 2], dtype=int) 
    
    # 4.) Neumann-Rand-Kanten
    if 'NeumannRand' in mesh.field_data: # Checke, ob entsprechender Key existiert
        neu_tag = mesh.field_data['NeumannRand'][0]
        edge_neu_YN = ( mesh.cell_data_dict['gmsh:physical']['line'] == neu_tag )
        edges_neu = mesh.cells_dict['line'][edge_neu_YN]
    else:
        edges_neu = np.zeros([0, 2], dtype=int)
    
    return Triangulation(points, triangles, edges_dir, edges_neu)

