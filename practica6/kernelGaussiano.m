##
## Calcula el kernel gaussiano para poder entrenar una SVM	
##
## Uso:
##	sim = kernelGaussiano(x1, x2, sigma)
##

function sim = kernelGaussiano(x1, x2, sigma)

	x1 = x1(:); x2 = x2(:);
	sim = 0;
	
	sim = exp(-(sum((x1 - x2) .^ 2)) / (2 * sigma .^ 2));

endfunction
