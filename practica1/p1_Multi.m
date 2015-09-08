##
## Regresión lineal sobre varias variable, sobre los puntos dados en fichero.
##	Para ello se emplea el método de descenso de gradiente, 
##	para ajustar la recta de la función de coste.
##
## Después se compara con el método del vector normal
##


# Primero cargamos los puntos de entrenamiento

load ex1data2.txt;
datos = ex1data2(:,:);

# y los normalizamos

x = datos(:, 1:2);
y = datos(:, 3);
m = length(x); #m será el número de ejemplos de entrenamiento
[x_norm, mu, sigma] = normalizaAtributo(x);

#Luego añade una columna de 1s a la izquierda

x_ampliado = [ones(m, 1) x_norm];

# Generamos un theta de cero
# Y vamos aproximando el valor de t al mínimo coste, iterativamente

t = zeros(3, 1); 

# Calculamos el valor de coste J inicial, que se debe minimizar.
# Es similar que con una variable
J = (1/2*m) * sum(((x_ampliado * t) - y).^2);

#aplicamos el método de descenso de gradiente iterativamente

a = 0.01; # este será el valor alpha
valores_J = zeros(1500, 1); #aquí almacenaremos las J parciales

for i = 1:1500

	t = t .- a * (1/m) * (((x_ampliado*t) - y)' * x_ampliado)';
	valores_J(i) = (1/2*m) * sum(((x_ampliado * t) - y).^2);

endfor

# Pintamos el avance de la funcion de coste

figure(1);clf;

plot(1:numel(valores_J), valores_J);
title( "Alpha = 0.01","fontsize", 20);
xlabel('Numero de repeticiones', "fontsize", 14);
ylabel('Coste de J', "fontsize", 14);
set(gca, "xlim", [0 1500], "ylim", [0 max(valores_J)]);

#print("gradienteMulti_001.png", "-dpng")



#Ahora calculamos el valor optimo mediante el metodo de la ec. normal

J_gradiente = (1/2*m) * sum(((x_ampliado * t) - y).^2)

x_normal = [ones(m, 1) x];
t_normal = pinv(x_normal'*x_normal)*x_normal'*y;

J_normal = (1/2*m) * sum(((x_normal * t_normal) - y).^2)


#Ahora comparamos ambos resultados haciendo alguna prediccion
# de casa de 1.650 pies cuadrados y 3 habitaciones

ejem=[1650, 3];
precio_grad = [1, (ejem-mu)./sigma] * t
precio_norm = [1, ejem] * t_normal


