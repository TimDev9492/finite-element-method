from src.utils.typing import Vector, VecNumMap, Vec2NumMap

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from src.finite_elements.elliptic_pde_linear import LinearFEMEllipticPDE
from src.utils.common import triangle_area
from src.mesh_tools.mesh_tools import Triangulation, read_msh
from src.mesh_tools.frankfurt_skyline import write_skyline

SKY_HEIGHT = 30

def _test_problem_kappa(vec: Vector) -> np.ndarray:
    '''
    Helper function to describe kappa
    '''
    return np.eye(2)

def _test_problem_kappa_zero(vec: Vector) -> float:
    return 0

def _test_problem_f(vec: Vector) -> np.ndarray | float:
    '''
    Helper function to describe right side of the poisson PDE
    '''
    return 0

def _test_problem_g_dir(vec: Vector) -> np.ndarray | float:
    vec = np.asarray(vec)
    x, y = vec[..., 0], vec[..., 1]
    on_gamma_1 = np.isclose(y, SKY_HEIGHT)
    return np.where(on_gamma_1, 1, 0)

def _test_problem_g_neu(vec1: Vector, vec2: Vector) -> float:
    return 0

def plot_finite_elements_problem(
    f: VecNumMap,
    g_dir: VecNumMap,
    g_neu: Vec2NumMap,
    u_plot: Axes3D,
    E_plot: Axes,
    triang: Triangulation
):
    pde = LinearFEMEllipticPDE(
        f,
        triang,
        _test_problem_kappa,
        _test_problem_kappa_zero,
        g_dir,
        g_neu,
    )

    # compute numerical solution
    numerical_solution = pde.solve()

    x, y = triang._points[:, 0], triang._points[:, 1]
    u_plot.plot_trisurf(x, y, pde.triang._tri_idx, numerical_solution, cmap='viridis')
    u_plot.set_title('$u_h$ plot')

    # plot electric field
    M = len(triang._tri_idx)
    centroids = np.empty((M,2))
    E = np.empty((M,2))
    absE = np.empty(M)
    for m in range(M):
        tri_idx = triang._tri_idx[m]
        # get triangle nodes
        nodes = triang._points[tri_idx]
        centroids[m] = centroid = np.sum(nodes, axis=0) / 3
        u_nodes = numerical_solution[tri_idx]
        # interpolate values linearly on triangle, u = a + bx + cy
        # thus, grad(u) = (b, c)
        one_xy_matrix = np.hstack([np.ones((3,1)), nodes])
        abc = np.linalg.solve(one_xy_matrix, u_nodes)
        E[m] = abc[1:]
        absE[m] = np.linalg.norm(E[m])

    '''
    ex05 (c)
    '''
    m_max = np.argmax(absE)
    m_max_points = triang._points[triang._tri_idx[m_max]]
    print('|E| is largest on triangle with vertices:')
    print(f'  P1({m_max_points[0][0]:.2f}, {m_max_points[0][1]:.2f})')
    print(f'  P2({m_max_points[1][0]:.2f}, {m_max_points[1][1]:.2f})')
    print(f'  P3({m_max_points[2][0]:.2f}, {m_max_points[2][1]:.2f})')
    print(f'with a value of |E_m|={absE[m_max]:.3e}')

    '''
    ex06 (d)
    '''
    E_plot.set_aspect('equal', adjustable='box')
    E_plot.tripcolor(x, y, triang._tri_idx, facecolors=absE)
    E_plot.quiver(centroids[:, 0], centroids[:, 1], E[:, 0], E[:, 1], angles='xy', scale_units='xy', scale=0.8, color='red', width=0.001)
    E_plot.set_title('$|E|$ plot')


def show_plots():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    skyline_mesh_filename = 'assets/meshes/skyline.msh'
    write_skyline(width=40, height=SKY_HEIGHT, h_min=0.1, h_max=2, output_file=skyline_mesh_filename)
    triang = read_msh(skyline_mesh_filename)
    triang.plot(plt)
    plt.savefig('figures/ex05a.pdf')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)

    u_plot = fig.add_subplot(gs[0, 0], projection='3d')
    E_plot = fig.add_subplot(gs[0, 1])

    plot_finite_elements_problem(
        _test_problem_f,
        _test_problem_g_dir,
        _test_problem_g_neu,
        u_plot,
        E_plot,
        triang,
    )

    plt.tight_layout()
    plt.savefig('figures/ex05d.pdf')
    plt.show()

show_plots()
