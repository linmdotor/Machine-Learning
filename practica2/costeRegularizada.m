##
## Devuelve el valor de la funcion de Coste.
##	y un vector con los valores del gradiente de la misma funcion.
## Para la versión regularizada de la regresión logistica
##
## Uso:
##	J, grad = costeRegularizada(theta, x, y, landa)
##

function [J, grad] = costeRegularizada(theta, x, y, landa)

	m = length(y); # numero de ejemplos de entrenamiento

	# Inicializamos el coste J y el vector de gradiente

	J = 0;
	grad = zeros(size(theta));

	# Aplicamos la funcion ya existente
	[J, grad] = coste(theta, x, y);

	# Añadimos la parte de la regularizacion a la funcion original
	J = J + (landa / (2*m)) * norm(theta([2:end])) ^ 2;
	grad = grad + ((landa / m) * norm(theta([2:end])));

endfunction
