##
## Calcula el valor de la funcion sigmoide.
##	Se puede aplicar indistintamente a número, vector o matriz.
##
## Uso:
##	g = sigmoide(z)
##

function g = sigmoide(z)

	g = zeros(size(z));
	g = 1 ./ (1 + exp(-z));

endfunction
