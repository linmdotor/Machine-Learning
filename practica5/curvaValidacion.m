##
## Devuleve los errores de una curva de aprendizaje para seleccionar.
## un landa determinado entre varios valores
##
## Uso:
##	[landa_vec, error_train, error_val] = curvaValidacion(X, y, Xval, yval)
##

function [landa_vec, error_train, error_val] = curvaValidacion(X, y, Xval, yval)

	#Partimos de los landa que nos dice el enunciado
	landa_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

	m = size(X, 1);

	error_train = zeros(length(landa_vec), 1);
	error_val   = zeros(length(landa_vec), 1);


	for i = 1:length(landa_vec)
		# Calcula lo thetas para este valor de landa.
		[thetas] = entrenaRegularizacionLineal(X, y, landa_vec(i));

		# Calcula los errores
		error_train(i) = costeRegresionLinealRegularizada(X, y, thetas, 0);
		error_val(i) = costeRegresionLinealRegularizada(Xval, yval, thetas, 0);
	end

endfunction