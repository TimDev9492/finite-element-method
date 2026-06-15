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
        if plot is None:
            plot = plt
        
        x = self._points[:, 0]
        y = self._points[:, 1]

        # draw mesh
        plot.triplot(x, y, self._tri_idx[:, :3], color='lightgray')
        plot.scatter(x, y, color='gray', s=5)

        # dirichlet edges
        for e in self._edges_dir:
            i, j = e
            plot.plot([x[i], x[j]], [y[i], y[j]], color='red', linewidth=2)

        # neumann edges
        for e in self._edges_neu:
            i, j = e
            plot.plot([x[i], x[j]], [y[i], y[j]], color='green', linewidth=2)

        plot.scatter(x[self._outer_point_mask], y[self._outer_point_mask], color='black', zorder=3)

        if display:
            plot.show()

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

@dataclass
class TriangulationQuad(Triangulation):
    def __init__(self, points, triangles, edges_dir = [], edges_neu = []):
        super().__init__(points, triangles, edges_dir, edges_neu)
    
    def plot(self, plot = None):
        display = plot is None
        if not display:
            plot = plt

        # plot underlying triangulation
        super().plot(plot)

        if display:
            plt.show()

    @staticmethod
    def _split_edges_halfway(edges, midpoint_lookup):
        no_edges = len(edges)
        new_edges = np.zeros((2 * no_edges, 2), dtype=int)
        for i in range(no_edges):
            # split edge in two
            edge_dir = edges[i]
            start, end = edge_dir[0], edge_dir[1]
            middle = midpoint_lookup[edge_dir[0], edge_dir[1]]
            new_edges[2*i, 0] = start
            new_edges[2*i, 1] = middle
            new_edges[2*i + 1, 0] = middle
            new_edges[2*i + 1, 1] = end
        return new_edges

    @staticmethod
    def from_triangulation(triang: Triangulation) -> TriangulationQuad:
        # compute number of midpoints by counting every triangle edge
        #  -> every inner edge gets counted twice
        # and then double count the edges (neumann/dirichlet)
        # and divide by 2
        no_vertex_points = triang._points.shape[0]
        no_triangles = triang._tri_idx.shape[0]
        no_edges_dir = triang._edges_dir.shape[0]
        no_edges_neu = triang._edges_neu.shape[0]
        no_midpoints = (3 * no_triangles + no_edges_dir + no_edges_neu) // 2

        points = np.zeros((no_vertex_points + no_midpoints, 2))
        points[:no_vertex_points, :] = triang._points
        midpoint_fill_idx = no_vertex_points

        triangles = np.zeros((no_triangles, 6), dtype=int)
        triangles[:, :3] = triang._tri_idx

        # prepare matrix that holds index of midpoint of edge [P_i,P_j] at index (i,j) or -1
        K = np.full((no_vertex_points, no_vertex_points), -1, dtype=int)

        # compute edge midpoints
        for polygon_idx in range(no_triangles):
            polygon = triang._tri_idx[polygon_idx]
            for k in range(len(polygon)):
                next_k = (k + 1) % len(polygon)
                i = polygon[k]
                j = polygon[next_k]
                midpoint_idx = K[i,j]
                if midpoint_idx == -1:
                    # compute midpoint and store the midpoint in the points array
                    midpoint = 0.5 * (triang._points[i] + triang._points[j])
                    points[midpoint_fill_idx] = midpoint
                    K[i,j] = K[j,i] = midpoint_fill_idx

                    # add midpoint to triangles array
                    triangles[polygon_idx, k+3] = midpoint_fill_idx

                    # increment fill index in points matrix
                    midpoint_fill_idx += 1
                else:
                    triangles[polygon_idx, k+3] = midpoint_idx

        # compute neu dirichlet and neumann edges
        edges_dir = TriangulationQuad._split_edges_halfway(triang._edges_dir, K)
        edges_neu = TriangulationQuad._split_edges_halfway(triang._edges_neu, K)

        return TriangulationQuad(points, triangles, edges_dir, edges_neu)


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

