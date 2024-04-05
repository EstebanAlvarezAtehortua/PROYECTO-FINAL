#Puntos experimentales
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt

#ingreso
xi = [1,   2,   3,  4,  5,   6, 7]
yi = [0.5, 2.5, 2., 4., 3.5, 6, 5.5]

datos = pd.read_csv('d:\Downloads\prueba1000.csv', header=0)

datos=datos.to_numpy().transpose().tolist()

xi=datos[1]
yi=datos[2]

#Procedimiento
xi = np.array(xi)
yi = np.array(yi)
n = len(xi)

#Sumatorias y medias
xm  = np.mean(xi)
ym  = np.mean(yi)
sx  = np.sum(xi)
sy  = np.sum(yi)
sxy = np.sum(xi*yi)
sx2 = np.sum(xi**2)
sy2 = np.sum(yi**2)


#coeficientes a0 y a1
a1 = (n*sxy-sx*sy)/(n*sx2-sx**2)
a0 = ym - a1*xm

# polinomio grado 1
x = sym.Symbol('x')
f = a0 + a1*x

fx = sym.lambdify(x,f)
fi = fx(xi)

# coeficiente de correlación
numerador = n*sxy - sx*sy
raiz1 = np.sqrt(n*sx2-sx**2)
raiz2 = np.sqrt(n*sy2-sy**2)
r = numerador/(raiz1*raiz2)

# coeficiente de determinacion
r2 = r**2
r2_porcentaje = np.around(r2*100,2)

# SALIDA
# print('ymedia =',ym)
print(' f = ',f)
print('coef_correlación   r  = ', r)
print('coef_determinación r2 = ', r2)
print(str(r2_porcentaje)+'% de los datos')
print('     está descrito en el modelo lineal')

# grafica
plt.plot(xi,yi,'o',label='(xi,yi)')
# plt.stem(xi,yi,bottom=ym,linefmt ='--')
plt.plot(xi,fi, color='orange',  label=f)

# lineas de error
for i in range(0,n,1):
    y0 = np.min([yi[i],fi[i]])
    y1 = np.max([yi[i],fi[i]])
    plt.vlines(xi[i],y0,y1, color='red',
               linestyle ='dotted')
plt.legend()
plt.xlabel('xi')
plt.title('minimos cuadrados')
plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ruta_imagen = 'UNIDAD.jpeg'
ruta_imagen1 = 'UNIDAD 2.jpeg'

imagen = mpimg.imread(ruta_imagen)
imagen1 = mpimg.imread(ruta_imagen1)

# Crear una figura más grande
plt.figure(figsize=(17, 15))

plt.subplot(1, 2, 1)
plt.imshow(imagen)
plt.axis('off')  # Ocultar ejes

plt.subplot(1, 2, 2)
plt.imshow(imagen1)
plt.axis('off')  # Ocultar ejes

plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Datos experimentales
xi = np.array([1, 2, 3, 4, 5, 6, 7])
yi = np.array([0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5])

# Ajuste de un polinomio cuadrático (grado 2)
coefficients = np.polyfit(xi, yi, 2)
a2, a1, a0 = coefficients

# Polinomio cuadrático
def quadratic_polynomial(x):
    return a2 * x**2 + a1 * x + a0

# Valores predichos
fi_quadratic = quadratic_polynomial(xi)

# Gráfica
plt.plot(xi, yi, 'o', label='Datos experimentales')
plt.plot(xi, fi_quadratic, color='orange', label=f'Polinomio cuadrático: f(x) = {a2:.6f}x^2 + {a1:.6f}x + {a0:.6f}')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresión Polinómica (Grado 2)')
plt.grid(True)
plt.show()



