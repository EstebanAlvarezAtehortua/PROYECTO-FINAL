import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import OLSInfluence

# Carga del archivo CSV
df = pd.read_csv(r"d:\Desktop\prueba unam\Historicos succion y descarga Un Herveo.csv", index_col=0)

# Selección de las variables
df = df[['PIT-3401', 'PIT-3403']]
df.columns = ['PIT-3401', 'PIT-3403']
df['PIT-3403'] = pd.to_numeric(df['PIT-3403'], errors= 'coerce')
df.dropna(subset=['PIT-3403'], inplace=True)

# Subconjunto de datos
df = df.iloc[-1000:]

# Modelo de regresión lineal
model = LinearRegression()

# Ajuste del modelo
model.fit(df[['PIT-3401']], df['PIT-3403'])

# Obtención de la pendiente y la intersección
pendiente = model.coef_[0]
interseccion = model.intercept_

# Predicción de valores de PIT-3403
predicciones = model.predict(df[['PIT-3401']])

# Visualización
plt.scatter(df['PIT-3401'], df['PIT-3403'])
plt.plot(df['PIT-3401'], predicciones, color='red')
plt.xlabel('PIT-3401')
plt.ylabel('PIT-3403')
plt.annotate(f"y = {pendiente:.2f}x + {interseccion:.2f}", xy=(0.6, 0.8), fontsize=12)
plt.show()

# Evaluación del modelo
r2 = model.score(df[['PIT-3401']], df['PIT-3403'])
error_estandar = np.std(np.abs(df['PIT-3403'] - predicciones))

print(f"R^2: {r2:.2f}")
print(f"Error estándar: {error_estandar:.2f}")

# Análisis de residuos
plt.scatter(df['PIT-3401'], df['PIT-3403'] - predicciones)
plt.xlabel('PIT-3401')
plt.ylabel('Residuos')
plt.show()

# Prueba de normalidad
shapiro(df['PIT-3403'] - predicciones)

# Análisis de influencia
infl = OLSInfluence(model).summary()
print(infl.table_summary())

# Regresión polinomial de segundo grado
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['PIT-3401']])

# Ajuste del modelo
model.fit(X_poly, df['PIT-3403'])

# Predicciones
predicciones_poly = model.predict(X_poly)



# Importa el módulo subplots
import matplotlib.pyplot as plt

# Crea una figura y un conjunto de subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Muestra la gráfica de dispersión en el primer subplot
axes[0, 0].scatter(df['PIT-3401'], df['PIT-3403'])
axes[0, 0].plot(df['PIT-3401'], predicciones, color='red')
axes[0, 0].set_xlabel('PIT-3401')
axes[0, 0].set_ylabel('PIT-3403')
axes[0, 0].annotate(f"y = {pendiente:.2f}x + {interseccion:.2f}", xy=(0.6, 0.8), fontsize=12)

# Muestra la gráfica de residuos en el segundo subplot
axes[0, 1].scatter(df['PIT-3401'], df['PIT-3403'] - predicciones)
axes[0, 1].set_xlabel('PIT-3401')
axes[0, 1].set_ylabel('Residuos')

# Muestra la gráfica de la regresión polinomial en el tercer subplot
axes[1, 0].plot(df['PIT-3401'], predicciones_poly, color='green')
axes[1, 0].set_xlabel('PIT-3401')
axes[1, 0].set_ylabel('PIT-3403 (Regresión polinomial)')

# Ajusta el espacio entre subplots
plt.tight_layout()

# Muestra la figura
plt.show()
