##
## Selecciona los mejores parámetros C y sigma	
##
## Uso:
##	[C, sigma] = seleccionarParametros(X, y, Xval, yval)
##

function [C, sigma] = seleccionarParametros(X, y, Xval, yval)

	C = 1;
	sigma = 0.3;

	Cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
	sigmas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

	#Realizamos 2 bucles for para probar todas las combinaciones
	# Se generarán 64 modelos distintos (8^2)
	minError = intmax;
	prueba = 0;

	for i = 1:length(Cs)
		for j = 1:length(sigmas)
			prueba = prueba +1
			model = svmTrain(X, y, Cs(i), @(x1, x2) kernelGaussiano(x1, x2, sigmas(j)));
			pred = svmPredict(model, Xval);
			predError = mean(double(pred ~= yval));

			if (predError <= minError)
				minError = predError;
				C = Cs(i);
				sigma = sigmas(j);
			end;
		end;
	end;

endfunction
