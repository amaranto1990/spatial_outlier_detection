import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.voronoi import voronoi_neighbors
from src.models import GBSO, SWOD, spatial_LOF, gamma_standardization
from src.visualization import plot_outliers, plot_gamma_outliers

def run_example():
    print("Generating synthetic data for demonstration...")
    # Synthetic coordinates
    x = [25.72548, 25.79358, 25.79681, 25.859773, 25.873889, 25.881927, 25.9315, 26.09083]
    y = [-80.31393, -80.19573, -80.39718, -80.341431, -80.372222, -80.253823, -80.20615, -80.14226]
    coords = np.array(list(zip(x, y)))
    
    # Synthetic values with one clear outlier
    values = np.array([0.0368, 0.0841, 0.027, 0.0, 2.86, 0.0, 0.0344, 0.0499])
    
    print("1. Calculating Voronoi neighbors and distances...")
    dist_points, dist_values, Vnn, k = voronoi_neighbors(coords, values)
    
    print("2. Running GBSO (Graph-Based Spatial Outlier Detection)...")
    gbso_scores = GBSO(S=coords, GBS=Vnn, Y=values)
    
    print("3. Running SWOD (Spatial Weighted Outlier Detection)...")
    swod_scores = SWOD(X=dist_points, Y=values, k=Vnn)
    
    print("4. Running Spatial LOF (Local Outlier Factor)...")
    lof_scores = spatial_LOF(dist=dist_values, Vnn=Vnn, k=k)
    
    print("\nResults:")
    results_df = pd.DataFrame({
        'Value': values,
        'GBSO_Score': gbso_scores,
        'SWOD_Score': swod_scores,
        'LOF_Score': lof_scores
    })
    print(results_df)
    
    print("\n5. Applying Gamma Standardization...")
    gbso_std = gamma_standardization(gbso_scores)
    
    # Plotting is disabled in this headless example, but functions are available
    # plot_outliers(coords, gbso_scores, 'GBSO Outliers', threshold=2)
    # plt.savefig('gbso_outliers.png')
    
    print("\nExecution completed successfully!")

if __name__ == "__main__":
    run_example()
