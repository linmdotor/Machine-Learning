##
## Realiza el proceso de la regresión logística
##


function theta = regresionLogistica(X_clasi, y_clasi, landa)

	%POR EL MOMENTO NO VAMOS A EXTENDER EL NUMERO DE ATRIBUTOS
	% Extendemos el numero de aributos hasta la 6a potencia
	%X_clasi = mapFeature(X_clasi, 6);

	# Calculamos la funcion de coste regularizada y su gradiente
	theta_inicial = zeros(size(X_clasi, 2), 1);
	[cost, grad] = costeRegularizada(theta_inicial, X_clasi, y_clasi, landa);

	# Calculamos el valor optimo
	opciones = optimset('GradObj', 'on', 'MaxIter', 400);
	[theta, J] = fminunc(@(t)(costeRegularizada(t, X_clasi, y_clasi, landa)), theta_inicial, opciones);

endfunction