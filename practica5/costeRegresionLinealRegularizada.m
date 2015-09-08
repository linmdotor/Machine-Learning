##
## Calcula el coste y el gradiente de la regresión lineal regularizada.
##
## Uso:
##	J, grad = costeRegresionLinealRegularizada(X, y, theta, landa)
##

function [J, grad] = costeRegresionLinealRegularizada(X, y, theta, landa)

	m = length(y); # numero de ejemplos de entrenamiento

	# Inicializamos el coste J y el vector de gradiente

	J = 0;
	grad = zeros(size(theta));


# Término del coste
coste = (1 / (2 * m)) * sum(((X * theta) - y) .^ 2);

# Término de regularización
reg = (landa / (2 * m)) * sum(theta(2:end) .^ 2);

# Coste regularizado.
J = coste + reg;

#Calculo del gradiente (separando j>=1)
grad = (1 / m) .* (X' * ((X * theta) - y));
grad(2:end) = grad(2:end) + ((landa / m) * theta(2:end));
grad = grad(:);


endfunction