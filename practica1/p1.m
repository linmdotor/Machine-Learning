##
## Regresión lineal sobre una variable, sobre los puntos dados en fichero.
##	Para ello se emplea el método de descenso de gradiente, 
##	para ajustar la recta de la función de coste.
##


# Primero cargamos los puntos de entrenamiento

load ex1data1.txt;

x = ex1data1.'(1,:);
y = ex1data1.'(2,:);
m = length(x); #m será el número de puntos

# Generamos un theta0 y theta1 aleatorios
# Y vamos aproximando los valores de t0 y t1 al mínimo coste, iterativamente

t0 = 0;
t1 = 0; 

# Calculamos el valor de coste J inicial, que se debe minimizar.

fun = @(x) t0 + t1*x;
J = (1/2*m) * sum((fun(x) - y).^2)

#aplicamos el método de descenso de gradiente iterativamente

a = 0.01; # este será el valor alpha

for i = 1:1500

	fun = @(x) t0 + t1*x;
	temp0 = t0 - a * (1/m) * sum(fun(x) - y);
	temp1 = t1 - a * (1/m) * sum((fun(x) - y) .* x);

	t0 = temp0;
	t1 = temp1;

endfor

# Pintamos los puntos de entrenamiento

figure(1);clf;

plot(x,y,"rx");
xlabel("Poblacion", "fontsize", 14);
ylabel("Beneficio", "fontsize", 14);

# Pintamos la función de regresión

fun = @(x) t0 + t1*x;
hold on;
fplot(fun, [5,25]);
hold off;

J = (1/2*m) * sum((fun(x) - y).^2)

set(gca, "xlim", [5 25], "ylim", [-5 25]);

# Abre otra ventana para pintar la función de coste

t0_vals = linspace(-10, 10, 100);
t1_vals = linspace(-1, 4, 100);

# inicializa J como una matriz de ceros
J_vals = zeros(length(t0_vals), length(t1_vals));

# Rellena los valores de J
for i = 1:length(t0_vals)
    for j = 1:length(t1_vals)    
	  J_vals(i,j) = (1/2*m) * sum((fun(t0_vals(i)) - t1_vals(j)).^2);
    endfor
endfor


figure(2);clf;
surface(t0_vals, t1_vals, J_vals);
#contour(t0_vals, t1_vals, J_vals, logspace(-2, 3, 20));
