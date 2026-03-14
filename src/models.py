import numpy as np
import pandas as pd
from scipy.stats import gamma

def GBSO(S, GBS, Y):
    '''
    Graph-Based Spatial Outlier Detection
    
    Parameters
    ----------
    S : Array of float
        Spatial attribute 2D: Longitude-Latitude.
    GBS : list of list of int
        List of lists of indices of the neighbors.
    Y : Array of float
        Attribute values.
        
    Returns
    -------
    detectOutlier : Series
        Outlier scores.
    '''
    AvgDist_Set = []
    for i in range(len(GBS)):
        AvgDist_Set.append(Y[i] - np.mean(Y[GBS[i]]))
        
    mu = np.mean(AvgDist_Set)
    sigma = np.std(AvgDist_Set)
    
    detectOutlier = abs((AvgDist_Set - mu)/sigma)
    return pd.Series(detectOutlier)

def SWOD(X, Y, k):
    '''
    Spatial Weighted Outlier Detection
    
    Parameters
    ----------
    X : Array of float
        Spatial position attribute (distance points).
    Y : Array of float
        Attribute values.
    k : list of list of int
        List of lists of indices of the neighbors.
        
    Returns
    -------
    OF : Series
        Outlier scores.
    '''
    size = len(X)
    weight = [0] * size
    for i in range(size):
        weight[i] = list(X[i,k[i]]/(sum(X[i,k[i]])))
        
    NbrAvg = [0] * size
    for i in range(size):
        NbrAvg[i] = sum(Y[k[i]]*weight[i])
        
    diff = [0] * size
    for i in range(size):
        diff[i] = Y[i] - NbrAvg[i]
        
    mu = np.mean(diff)
    sigma = np.std(diff)
    
    OF = [0] * size
    for i in range(size):
        OF[i] = abs((diff[i]-mu)/sigma)
        
    return pd.Series(OF)

def spatial_LOF(dist, Vnn, k):
    '''
    Local Outlier Factor for Spatial Data
    
    Parameters
    ----------
    dist : array of float
        Distances between values.
    Vnn : list of list of int
        List of lists of indices of the neighbors.
    k : array of int
        Number of neighbors per observation.
        
    Returns
    -------
    LOF : Series
        Outlier scores.
    '''
    size = len(Vnn)
    order = np.argsort(dist)
    
    idx_Vnn = []
    for i in range(size):
        idx_Vnn.append(list(order[i, :len(Vnn[i])]))
        
    radius = [0] * size
    for i in range(size):
        radius[i] = dist[i,idx_Vnn[i][-1]]
    radius = np.array(radius)
    
    LRD = []
    for i in range(size):
        LRD.append(np.mean(np.maximum(dist[i, idx_Vnn[i]], radius[idx_Vnn[i]])))
        
    rho = 1. / np.array(LRD)
    
    LOF = [0] * size
    for i in range(size):
        LOF[i] = np.sum(rho[idx_Vnn[i]])/ np.array(rho[i])
        
    LOF *= 1./k
    return pd.Series(LOF)

def gamma_standardization(Score):
    '''
    Gamma Cumulative Density Function Standardization
    
    Parameters
    ----------
    Score : array
        Outlier scores.
        
    Returns
    -------
    standard_score_gamma : Series
        Standardized scores.
    '''
    mu = np.mean(Score)
    sigma = np.std(Score)
    
    shape, scale = (mu**2)/(sigma**2), sigma/(mu**2)
    
    cdf_gamma = gamma.cdf(Score, a=shape, scale=1/scale)
    gamma_mu_s = gamma.cdf(mu, a=shape, scale=1/scale)
    
    standard_score_gamma = pd.Series(np.maximum(0, (cdf_gamma - gamma_mu_s )/(1 - gamma_mu_s)))
    return standard_score_gamma
