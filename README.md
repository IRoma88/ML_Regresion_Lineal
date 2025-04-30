# Predicción del Precio Medio de Viviendas en California

Este proyecto utiliza el conjunto de datos `california_housing_train.csv`, que viene incluido por defecto en Google Colab, para construir modelos de regresión que predicen el precio medio de una vivienda (`median_house_value`). Se exploran distintos enfoques de regresión: simple, polinómica (grado 2 y 3) y múltiple.

## 📁 Archivos

- `california_housing_train.csv`: dataset utilizado (ubicado en `sample_data/` de Google Colab).
- `notebook.ipynb`: notebook que contiene todo el análisis, gráficos y modelos.
- `README.md`: este archivo.

## 🧰 Herramientas utilizadas

- Python
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

## 📈 Modelos construidos

Se comparan varios tipos de regresión:

| Modelo         | Ecuación (aproximada)                                                                                                     | Score (R²) |
|----------------|----------------------------------------------------------------------------------------------------------------------------|------------|
| Simple         | precio = 43980.63 + 42054.07 · median_income                                                                               | 0.479      |
| Polinomial (2) | precio = 18920.20 + 53649.40 · median_income - 1066.65 · median_income²                                                   | 0.483      |
| Polinomial (3) | precio = 75138.94 + 15095.09 · median_income + 6050.31 · median_income² - 353.91 · median_income³                         | 0.491      |
| Múltiple       | precio = -3620600.89 + ... + 40507.07 · median_income (usa todas las variables predictoras excepto la variable objetivo) | 0.641      |

## ▶️ Cómo usarlo

1. Abre [Google Colab](https://colab.research.google.com/)
2. Carga el archivo del notebook (`.ipynb`)
3. Ejecuta las celdas paso a paso

> No necesitas descargar ningún dataset, ya que `california_housing_train.csv` viene incluido en la carpeta `sample_data` de Google Colab.

## 📌 Objetivo

Evaluar qué tipo de modelo de regresión es más adecuado para predecir el precio medio de una vivienda en California.

## ✅ Resultado

El modelo de regresión múltiple es el que mejor se ajusta, con un score de R² = 0.641, mostrando una mejora significativa respecto a los modelos simples y polinómicos.

