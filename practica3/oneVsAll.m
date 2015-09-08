##
## Entrena varios clasificadores por regresión logística
##	Devuelve el resultado en la matriz all_theta, donde
##	la fila i-esima corresponde al clasificador de la etiqueta i-esima.
##
## Uso:
##	all_theta = oneVsAll(X, y, num_etiquetas, lambda)
##

function [all_theta] = oneVsAll(X, y, num_etiquetas, lambda)

	#Inicializacion de variables
	m = size(X, 1);
	n = size(X, 2);
	all_theta = zeros(num_etiquetas, n + 1);

	#Añade una fila de unos a la matriz X
	X = [ones(m, 1) X];

	#Emplearemos fmincg porque es más eficiente con muchos parametros
	initial_theta = zeros(n + 1, 1);
     
	options = optimset('GradObj', 'on', 'MaxIter', 50);
	 
	for i = 1:num_etiquetas
		c = i * ones(size(y));
		[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options); #calcula un theta por cad ejemplo
		all_theta(i,:) = theta; #vamos agreando cada theta a los resultados
	endfor


endfunction
