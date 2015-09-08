##
## Calcula el valor de la derivada de la funcion sigmoide.
##	Se puede aplicar indistintamente a número, vector o matriz.
## 	Se empleará para el calculo del gradiente en costeRN
##
## Uso:
##	g = sigmoideDerivada(z)
##

function g = sigmoideDerivada(z)

	sig_aux = sigmoide(z);
	g = sig_aux .* (1 - sig_aux);

endfunction
