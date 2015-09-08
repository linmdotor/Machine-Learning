##
## Genera, a partir de un conjunto de atributos, nuevas columnas
##	que son el resultado de elevar a la "p" potencia sucesivamente.
##
## Uso:
##	[X_final] = elevarPotenciaAtributos(X, p)
##

function [X_final] = elevarPotenciaAtributos(X, p)

	X_final = zeros(numel(X), p);

	for i = 1:p
		X_final(:, i) = X .^ i;
	end

endfunction