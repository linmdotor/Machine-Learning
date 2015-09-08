##
## Devuleve los errores de una curva de aprendizaje.
## Para ello empleamos las funciones "costeRegresionLinealRegularizada" y "entrenaRegularizacionLineal"
##
## Uso:
##	[error_train, error_val] = curvaAprendizaje(X, y, Xval, yval, landa)
##

function [error_train, error_val] = curvaAprendizaje(X, y, Xval, yval, landa)

	m = size(X, 1);

	error_train = zeros(m, 1);
	error_val   = zeros(m, 1);


	for i = 1:m
		trainX = X(1:i, :);
		trainY = y(1:i, :);

		[theta] = entrenaRegularizacionLineal(trainX, trainY, landa);

		error_train(i) = costeRegresionLinealRegularizada(trainX, trainY, theta, 0);
		error_val(i) = costeRegresionLinealRegularizada(Xval, yval, theta, 0);
	end

endfunction