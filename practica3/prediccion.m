##
## Predice la etiqueta de la entrada, mediante una red neuronal ya entrenada.
##
## Uso:
##	p = prediccion(Theta1, Theta2, X)

function p = prediccion(Theta1, Theta2, X)

	m = size(X, 1);
	num_labels = size(Theta2, 1);

	p = zeros(size(X, 1), 1);


	# Theta1 25 x 401
	# Theta2 10 x 26
	# p 5000 x 1

	# Capa de Entrada
	a1 = [ones(m, 1), X]; # [5000, 401]
	z2 = Theta1 * a1'; # [25, 5000]

	# Capa oculta
	a2 = sigmoide(z2);  # [25, 5000]
	a2 = [ones(1, size(a2, 2)); a2]; # [26, 5000]

	# Capa de salida
	z3 = Theta2 * a2; # [10, 5000]
	a3 = sigmoide(z3); # [10, 5000]
 
	# resulta, p, tiene las dimensiones [5000, 1]
	[val, p] = max(a3', [], 2);

endfunction
