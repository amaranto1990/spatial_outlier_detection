import matplotlib.pyplot as plt
import numpy as np

def plot_outliers(S, score, title, threshold=None, top_m=None):
    '''
    Visualize outliers on a scatter plot.
    
    Parameters
    ----------
    S : Array of float
        Spatial attribute 2D: Longitude-Latitude.
    score : Array of float
        Outlier scores.
    title : str
        Plot title.
    threshold : float, optional
        Threshold for considering a point an outlier.
    top_m : int, optional
        Number of top outliers to consider.
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(S[:, 0], S[:, 1], c='green', s=10, edgecolors='None', alpha=0.5, label='Normal')
    
    if top_m is not None:
        idx = (-score).argsort()[:top_m]
    elif threshold is not None:
        idx = np.where(score >= threshold)[0]
    else:
        idx = []
        
    plt.scatter(S[idx, 0], S[idx, 1], c='red', s=10, edgecolors='None', alpha=0.5, label='Outlier')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    return plt.gcf()

def plot_gamma_standardization(score, title):
    '''
    Plot histogram of gamma standardized scores.
    '''
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 20)
    plt.hist(np.round(score, 4), bins=bins, histtype='stepfilled', color='cyan', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6)
    return plt.gcf()

def plot_gamma_outliers(S, normalized_score, title):
    '''
    Visualize outliers based on gamma standardized scores.
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(S[:, 0], S[:, 1], c='green', s=10, edgecolors='None', alpha=0.5, label='Normal')
    
    idx = np.where(np.round(normalized_score) == 1)[0]
    
    plt.scatter(S[idx, 0], S[idx, 1], c='red', s=10, edgecolors='None', alpha=0.5, label='Outlier')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    return plt.gcf()
