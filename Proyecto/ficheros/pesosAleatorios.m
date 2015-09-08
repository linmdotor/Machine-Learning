##
## Inicializa una matriz de pesos de una capa
##		con los valores aleatorios en el rango [L_in, L_out]
##
## Uso:
##	W = pesosAleatorios(L_in, L_out)
##

function W = pesosAleatorios(L_in, L_out)

	epsilon = 0.12;
	W = rand(L_out, 1 + L_in) * 2 * epsilon - epsilon;

	epsilon = .12;
	W = rand(L_out, 1+L_in) * 2 * epsilon - epsilon;

endfunction
