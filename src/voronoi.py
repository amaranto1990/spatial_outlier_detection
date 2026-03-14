import numpy as np
import networkx as nx
from scipy.spatial import Voronoi, distance

def voronoi_neighbors(coords, values):
    '''
    Calculate Voronoi neighbors and distances.
    
    Parameters
    ----------
    coords : array of float
        Longitude-Latitude coordinates.
    values : array of float
        Values of the attribute to analyze.
        
    Returns
    -------
    dist_points : array of float
        Distances between geographic points.
    dist_values : array of float
        Distances between values of neighboring observations.
    Vnn : list of list of int
        List of lists of indices of the neighbors of each observation.
    k : array of int
        Number of neighbors per observation.
    '''
    size = len(coords)
    # Voronoi diagram
    DV = Voronoi(coords, qhull_options="QJ")
    edges = DV.ridge_points
    
    nodes = sorted(set(node for edge in edges for node in edge))
    
    # Graph to find neighbors
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Adjacency matrix
    mx_veins = nx.adjacency_matrix(G).toarray()
    mx_veins = mx_veins.astype(float)
    
    # Neighbors list
    Vnn = []
    for i in range(size):
        Vnn.append(list(np.where(mx_veins[i,:]==1)[0]))
        
    k = mx_veins.sum(0)
    
    dist_points = distance.cdist(coords, coords, 'euclidean')
    
    y_i = []
    for i in range(len(edges)):
        y_i.append(values[edges[i,0]])
        
    y_j = []
    for j in range(len(edges)):
        y_j.append(values[edges[j,1]])
        
    y_i_y_j = np.array(list(zip(y_i,y_j)))
    dist_yij = np.linalg.norm(y_i_y_j, axis=1)
    
    y_i = []
    for i in range(len(edges)):
        y_i.append(edges[i,0])
    y_j = []
    for j in range(len(edges)):
        y_j.append(edges[j,1])
        
    edges_yij = tuple(zip(y_i,y_j))
    dict_values = dict(list(zip(edges_yij, dist_yij)))
    
    dist_values = np.array([[np.nan]*size]*size)
    for i in range(len(list(dict_values.keys()))):
        dist_values[list(dict_values.keys())[i]] = list(dict_values.values())[i]
        dist_values[list(dict_values.keys())[i][1], list(dict_values.keys())[i][0]] = list(dict_values.values())[i]
        
    return dist_points, dist_values, Vnn, k
