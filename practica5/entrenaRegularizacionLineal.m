##
## Busca el valor theta que minimiza el coste (empleando fmincg).
## Para ello empleamos la función fmincg que se nos ofrece
##
## Uso:
##	[theta] = entrenaRegularizacionLineal(X, y, landa)
##

function [theta] = entrenaRegularizacionLineal(X, y, landa)

	initial_theta = zeros(size(X, 2), 1); 
	costFunction = @(t) costeRegresionLinealRegularizada(X, y, t, landa);
	options = optimset('MaxIter', 150, 'GradObj', 'on');
	theta = fmincg(costFunction, initial_theta, options);

endfunction