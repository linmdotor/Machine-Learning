##
## Devuelve el coste y gradiente de la
##	regresión logística regularizada.
##
## Uso:
##	J, grad = lrCostFunction(theta, X, y, lambda)
##

function [J, grad] = lrCostFunction(theta, X, y, lambda)

	m = length(y); # numero de ejemplos de entrenamiento

	# Inicializamos el coste J y el vector de gradiente
	J = 0;
	grad = zeros(size(theta));

	# Aplicamos las fórmulas (similar a la funcion de coste original)
	h = sigmoide(X * theta);
	J = (1 / m) * sum((-y .* log(h)) - ((1 - y) .* log(1 - (h))));
	grad = (1 / m) .* X' * (h - y);

	#Regulariza J, sumando el termino regularizado (evitando el elemento theta0)al J original 
	J = J + (lambda / (2 * m)) * norm(theta([2:end])) ^ 2;
	grad = grad + ((lambda / m) * norm(theta([2:end])));

	grad = grad(:);

endfunction
