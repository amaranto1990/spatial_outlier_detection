# Spatial Outlier Detection Methods Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Read this in other languages: [English](README.md), [Español](README.es.md).*

This repository contains the implementation of several spatial outlier detection methodologies, primarily based on the concept of neighborhood determined by the **Voronoi diagram**. This project is the result of a Master's Thesis focused on comparing spatial outlier detection algorithms using geographic data.

## 📌 Overview

To treat the outlier detection problem, there are numerous methodologies that give rise to a group of different approaches. In this work, the search for outliers is applied to data with geographic information (geolocated data). 

The **Voronoi diagram** is incorporated to determine the concept of neighborhood as an alternative to distance-based methods. Given the geolocation of the data, the definition of the environment or neighborhood of a point is calculated from the position coordinates.

This repository implements three main spatial outlier detection models:
1. **GBSO (Graph-Based Spatial Outlier Detection)**
2. **SWOD (Spatial Weighted Outlier Detection)**
3. **Spatial LOF (Local Outlier Factor adapted for spatial data)**
4. **GBSO_ST (Spatio-Temporal GBSO)** - *New extension for time-series geospatial data*

## 🚀 Features

- **Voronoi Neighborhood Generation**: Automatically calculates neighbors and distances using Voronoi diagrams instead of simple Euclidean distance thresholds.
- **Multiple Detection Algorithms**: Includes three static approaches and one spatio-temporal extension to score outliers.
- **Spatio-Temporal Analysis**: `GBSO_ST` allows monitoring sensor networks over time, detecting both structural anomalies and sudden local events using a sliding temporal window over the spatial neighbourhood.
- **Gamma Standardization**: Standardizes the outlier scores using the Gamma Cumulative Density Function to allow for unified criteria and comparison.
- **Visualization Tools**: Built-in functions to visualize outliers on a map or scatter plot.

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/spatial_outlier_detection.git
   cd spatial_outlier_detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

Here is a basic example of how to use the models with synthetic data:

```python
import numpy as np
from src.voronoi import voronoi_neighbors
from src.models import GBSO, SWOD, spatial_LOF

# 1. Prepare your data
# coords: array of [longitude, latitude]
# values: array of the attribute to analyze
coords = np.array([[25.72, -80.31], [25.79, -80.19], [25.79, -80.39], [25.85, -80.34]])
values = np.array([0.0368, 0.0841, 0.027, 2.86]) # 2.86 is an outlier

# 2. Calculate Voronoi neighbors
dist_points, dist_values, Vnn, k = voronoi_neighbors(coords, values)

# 3. Run detection models
gbso_scores = GBSO(S=coords, GBS=Vnn, Y=values)
swod_scores = SWOD(X=dist_points, Y=values, k=Vnn)
lof_scores = spatial_LOF(dist=dist_values, Vnn=Vnn, k=k)

print(gbso_scores)
```

For a complete runnable example, check `example.py`.

## 🧠 Models Description

### Graph-Based Spatial Outlier Detection (GBSO)
Based on Shekhar et al. (2003), it estimates the sample standard deviation statistic by spatial neighborhood, as the difference between the value of the non-spatial attribute of the observation and the average of the neighboring observations.

### Spatial Weighted Outlier Detection (SWOD)
Based on Kou et al. (2006), it weights the distance between observations within each spatial neighborhood to capture the influence of each of them in the measurement of the weighted outlier function.

### Spatial Local Outlier Factor (Spatial LOF)
An adaptation of the classic LOF method (Breunig et al., 2000) that uses the $k$ nearest neighbors defined by the Voronoi diagram instead of simple Euclidean distance, calculating a local reachability density.

### Spatio-Temporal GBSO (GBSO_ST)
An original extension of GBSO for time-series geospatial data (e.g., sensor networks). For each observation at time step $t$, the score measures the deviation of its value from the spatio-temporal distribution of its Voronoi neighbours within a sliding temporal window $[t-k, t+k]$. This detects anomalies that are simultaneously unusual with respect to their immediate geographic environment and the recent temporal behaviour of that environment.

## 📚 References

- Schubert, E., Zimek, A., & Kriegel, H. P. (2014). Local outlier detection reconsidered: a generalized view on locality with applications to spatial, video, and network outlier detection.
- Shekhar, S., Lu, C. T., & Zhang, P. (2001). Detecting graph-based spatial outliers: algorithms and applications.
- Kou, Y., Lu, C. T., & Chen, D. (2006). Spatial weighted outlier detection.
- Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
