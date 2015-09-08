##
## Devuelve el valor de la funcion de Coste.
##	y un vector con los valores del gradiente de la misma funcion.
##
## Uso:
##	J, grad = coste(theta, x, y)
##

function [J, grad] = coste(theta, x, y	)

	m = length(y); # numero de ejemplos de entrenamiento

	# Inicializamos el coste J y el vector de gradiente

	J = 0;
	grad = zeros(size(theta));

	# Aplicamos las fórmulas

	h = sigmoide(x * theta);

	J = (1 / m) * sum((-y .* log(h)) - ((1 - y) .* log(1 - (h))));

	grad = (1 / m) .* x' * (h - y);

endfunction
