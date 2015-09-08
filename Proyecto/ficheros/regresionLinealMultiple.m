##
## Realiza el proceso de la regresión lineal con múltiples varibles
##


function [t, t_normal, mu, sigma] = regresionLinealMultiple(X, y, m, num_atr)

	# Normalizamos los atributos para acelerar la convergencia
	[X_norm, mu, sigma] = normalizaAtributo(X);
	# Añade la columna de 1s a la izquierda
	X_ampliado = [ones(m, 1) X_norm];

	# Generamos un theta de cero
	# Y vamos aproximando el valor de t al mínimo coste, iterativamente

	t = zeros(num_atr, 1); 

	# Calculamos el valor de coste J inicial, que se debe minimizar.
	J = (1/2*m) * sum(((X_ampliado * t) - y).^2);

	# Aplicamos el método de descenso de gradiente iterativamente
	a = 0.01; # alpha
	num_iter = 2500;
	valores_J = zeros(num_iter, 1); # J parciales

	for i = 1:num_iter

		t = t .- a * (1/m) * (((X_ampliado*t) - y)' * X_ampliado)';
		valores_J(i) = (1/2*m) * sum(((X_ampliado * t) - y).^2);

	endfor

	# Pintamos el avance de la funcion de coste
	figure(1);clf;

	plot(1:numel(valores_J), valores_J);
	title( "Alpha = 0.01","fontsize", 20);
	xlabel('Numero de repeticiones', "fontsize", 14);
	ylabel('Coste de J', "fontsize", 14);
	set(gca, "xlim", [0 num_iter], "ylim", [0 max(valores_J)]);

	#Ahora calculamos el valor optimo mediante el metodo de la ec. normal
	J_gradiente = (1/2*m) * sum(((X_ampliado * t) - y).^2);

	x_normal = [ones(m, 1) X];
	t_normal = pinv(x_normal'*x_normal)*x_normal'*y;

	J_normal = (1/2*m) * sum(((x_normal * t_normal) - y).^2);

	#Y elegiremos un número de iteraciones que lo aproxime lo suficiente

endfunction