# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Vamos a trabajar con los csv que viene por defecto en google colab, en concreto con los de california_housing.

Empezaremos creando el dataframe correspondiente a 'california_housing_train.csv' llamando al df casas_train.
"""

casas_train = pd.read_csv('sample_data/california_housing_train.csv')

#@title Comprueba la creación del dataframe
def check():
  if str(casas_train.tail())=="       longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \\\n16995    -124.26     40.58                52.0       2217.0           394.0   \n16996    -124.27     40.69                36.0       2349.0           528.0   \n16997    -124.30     41.84                17.0       2677.0           531.0   \n16998    -124.30     41.80                19.0       2672.0           552.0   \n16999    -124.35     40.54                52.0       1820.0           300.0   \n\n       population  households  median_income  median_house_value  \n16995       907.0       369.0         2.3571            111400.0  \n16996      1194.0       465.0         2.5179             79000.0  \n16997      1244.0       456.0         3.0313            103600.0  \n16998      1298.0       478.0         1.9797             85800.0  \n16999       806.0       270.0         3.0147             94600.0  ":
    return 'Creación correcta'
  else:
    return 'Incorrecto, comprueba el datafame'
check()

"""Comprueba los 5 primeros elementos del dataframe."""

casas_train.head()

"""#Objetivo

Nuestro objetivo es obtener un modelo capaz de predecir el precio medio de la casa, para ello tendremos que comprobar diferentes modelos de regresiones lineales, polinómicas y múltiples.

#Vamos a empezar graficando cada columna frente a median_house_value para ver las tendencias
"""

# Columnas a comparar con median_house_value
columnas = casas_train.columns.tolist()
columnas.remove('median_house_value')  # Excluir la variable dependiente

# Crear gráficos por separado
for columna in columnas:
    #fig, axes= plt.subplots(9, figsize=(10,150))
    plt.figure(figsize=(8, 10))
    plt.scatter(casas_train[columna], casas_train['median_house_value'], alpha=0.5)
    plt.xlabel(columna)
    plt.ylabel('Median House Value')
    plt.title(f'{columna} vs Median House Value')
    plt.show()

"""Una vez visualizadas las gráficas, ¿qué variable parece que se ajuste mejor a una regresión lineal simple?"""

#@title Respuesta

'''Según las gráficas, la variable median_income es la que parece que mejor puede describir un modelo de regresión lineal simple'''

"""#Ejercicio 1

Vamos a crear el modelo de regresión lineal simple house_median_value = a + b · median_income.
"""

from sklearn.linear_model import LinearRegression

x = casas_train[['median_income']]
y = casas_train[['median_house_value']]

modelo = LinearRegression()
modelo.fit(x,y)

"""Almacena en la solucion_1 la lista que contenga el valor del intercept, del coeficiente y del score del modelo.

Ejemplo de solucion:

    solucion_1=[nombre_modelo.intercept_,nombre_modelo_coef_,model.score(median_income,house_value)]
"""

solucion_1=modelo.intercept_,modelo.coef_,modelo.score(x,y)

solucion_1

"""#Regresión lineal polinómica

La cantidad de datos que es capaz de predecir el modelo con regresión lineal simple es muy bajo, vamos a probar a aplicarle grado 2 y grado 3 al polinomio. Es decir que la fórmula sea

y = a + b·x + c·x<sup>2</sup>


y = a + b·x + c·x<sup>2</sup> + d·x<sup>3</sup>

##ejercicio 2

En la variable solucion_2 almacena la lista con el valor del intercept,  de los coeficientes y del score del modelo polinomial de grado 2 y en la variable solucion_2b la del modelo polinomial de grado 3.
"""

#Crea el modelo polinomial de grado 2
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)
x_poly2 = poly_features.fit_transform(x)

poly_model2 = LinearRegression()
poly_model2.fit(x_poly2, y)

#Crea el modelo polinomial de grado 3
poly_features = PolynomialFeatures(degree=3)
x_poly3 = poly_features.fit_transform(x)

poly_model3 = LinearRegression()
poly_model3.fit(x_poly3, y)

solucion_2=poly_model2.intercept_,poly_model2.coef_,poly_model2.score(x_poly2,y)

print(solucion_2)

solucion_2b=poly_model3.intercept_,poly_model3.coef_,poly_model3.score(x_poly3,y)

print(solucion_2b)

"""#Regresión lineal múltiple

Dada la escasa mejoría en la cantidad de datos que pueden explicar los modelos polinomiales frente al simple los vamos a descartar y vamos a probar con una regresión múltiple, usando todas las variables (excepto la que queremos predecir) como predictoras.


"""

prueba3=casas_train[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']]
y=casas_train[['median_house_value']]
model3 = LinearRegression()
model3.fit(prueba3, y)
model3.score(prueba3, y)

"""Realiza el modelo y almacena en la variable solucion_3 la lista con el valor del intercept, los coeficientes y el score del modelo."""

solucion_3=[model3.intercept_,model3.coef_,model3.score(prueba3,y)]

solucion_3


"""#TIP

Para que tu trabajo no sea en vano, es recomendable crear un dataframe que contenga un resumen de los modelos realizados.
"""

df=pd.DataFrame({
    'modelo':['simple','grado 2','grado 3', 'multivariable'],
    'ecuacion':['precio = 43980.6282 + 42054.0749 · median_income','precio = 18920.2047 + 53649.40 · median_income - 1066.6515 · median_income^2',' precio = 75138.9431 +15095.0888· median_income + 6050.3056 · median_income^2 - 353.9104 · median_income^3','precio = - 3620600.8930 - 43139.6373 · longitude - 42925.6731 · latitude + 1150.6950 · housing_median_age - 8.3782 · total_rooms + 117.6485 · total_bedrooms + 43794.5527 - 38.4888 · population + 45.4360 · households + 40507.0684 · median_income'],
    'score':[0.479,0.483,0.491,0.641]
})
df
