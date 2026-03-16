# Análisis de Métodos de Detección de Outliers Geoespaciales

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Leer en otros idiomas: [English](README.md), [Español](README.es.md).*

Este repositorio contiene la implementación de diversas metodologías de detección de outliers espaciales, basadas principalmente en el concepto de vecindario determinado por el **diagrama de Voronoi**. Este proyecto es el resultado de un Trabajo de Fin de Máster centrado en la comparación de algoritmos de detección de valores atípicos utilizando datos geográficos.

## 📌 Descripción General

Para abordar el problema de la detección de outliers, existen numerosas metodologías que dan lugar a diferentes enfoques. En este trabajo, la búsqueda de valores atípicos se aplica a datos con información geográfica (datos geolocalizados).

Se incorpora el **diagrama de Voronoi** para determinar el concepto de vecindario como alternativa a los métodos basados en distancias euclidianas. Dada la geolocalización de los datos, la definición del entorno o vecindario de un punto se calcula a partir de sus coordenadas de posición.

Este repositorio implementa tres modelos principales de detección de outliers espaciales:
1. **GBSO (Graph-Based Spatial Outlier Detection)**
2. **SWOD (Spatial Weighted Outlier Detection)**
3. **Spatial LOF (Local Outlier Factor adaptado a datos espaciales)**
4. **GBSO_ST (Spatio-Temporal GBSO)** - *Nueva extensión para series temporales de datos geoespaciales*

## 🚀 Características

- **Generación de Vecindarios con Voronoi**: Calcula automáticamente los vecinos y las distancias utilizando diagramas de Voronoi en lugar de simples umbrales de distancia euclidiana.
- **Múltiples Algoritmos de Detección**: Incluye tres enfoques estáticos y una extensión espacio-temporal para puntuar la "outlaridad".
- **Análisis Espacio-Temporal**: `GBSO_ST` permite monitorizar redes de sensores a lo largo del tiempo, detectando tanto anomalías estructurales como eventos locales repentinos usando una ventana temporal deslizante sobre el vecindario espacial.
- **Estandarización Gamma**: Estandariza las puntuaciones de los outliers utilizando la Función de Densidad Acumulativa Gamma para unificar criterios y permitir la comparación entre modelos.
- **Herramientas de Visualización**: Funciones integradas para visualizar los outliers en un mapa o diagrama de dispersión.

## 🛠️ Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tuusuario/spatial_outlier_detection.git
   cd spatial_outlier_detection
   ```

2. Instala las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Uso

A continuación se muestra un ejemplo básico de cómo usar los modelos con datos sintéticos:

```python
import numpy as np
from src.voronoi import voronoi_neighbors
from src.models import GBSO, SWOD, spatial_LOF

# 1. Prepara tus datos
# coords: array de [longitud, latitud]
# values: array del atributo a analizar
coords = np.array([[25.72, -80.31], [25.79, -80.19], [25.79, -80.39], [25.85, -80.34]])
values = np.array([0.0368, 0.0841, 0.027, 2.86]) # 2.86 es un outlier

# 2. Calcula los vecinos de Voronoi
dist_points, dist_values, Vnn, k = voronoi_neighbors(coords, values)

# 3. Ejecuta los modelos de detección
gbso_scores = GBSO(S=coords, GBS=Vnn, Y=values)
swod_scores = SWOD(X=dist_points, Y=values, k=Vnn)
lof_scores = spatial_LOF(dist=dist_values, Vnn=Vnn, k=k)

print(gbso_scores)
```

Para un ejemplo ejecutable completo, revisa el archivo `example.py`.

## 🧠 Descripción de los Modelos

### Graph-Based Spatial Outlier Detection (GBSO)
Basado en Shekhar et al. (2003), estima el estadístico de desviación estándar muestral por vecindario espacial, como la diferencia entre el valor del atributo no espacial de la observación y el promedio de las observaciones vecinas.

### Spatial Weighted Outlier Detection (SWOD)
Basado en Kou et al. (2006), pondera la distancia entre las observaciones dentro de cada vecindario espacial para capturar la influencia de cada uno de ellos en la medición de la función de outlaridad ponderada.

### Spatial Local Outlier Factor (Spatial LOF)
Una adaptación del método clásico LOF (Breunig et al., 2000) que utiliza los $k$ vecinos más cercanos definidos por el diagrama de Voronoi en lugar de la distancia euclidiana simple, calculando una densidad de accesibilidad local.

### Spatio-Temporal GBSO (GBSO_ST)
Una extensión original de GBSO para series temporales de datos geoespaciales (por ejemplo, redes de sensores). Para cada observación en el instante $t$, la puntuación mide la desviación de su valor respecto a la distribución espacio-temporal de sus vecinos de Voronoi dentro de una ventana temporal deslizante $[t-k, t+k]$. Esto detecta anomalías que son simultáneamente inusuales respecto a su entorno geográfico inmediato y al comportamiento temporal reciente de ese entorno.

## 📚 Referencias

- Schubert, E., Zimek, A., & Kriegel, H. P. (2014). Local outlier detection reconsidered: a generalized view on locality with applications to spatial, video, and network outlier detection.
- Shekhar, S., Lu, C. T., & Zhang, P. (2001). Detecting graph-based spatial outliers: algorithms and applications.
- Kou, Y., Lu, C. T., & Chen, D. (2006). Spatial weighted outlier detection.
- Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
