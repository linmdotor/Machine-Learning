##
## Predice la probabilidad de que la imagen pertenezca a una determinada clase.
##
## Uso:
##	p = prediccionOneVsAll(all_theta, X)
##

function p = prediccionOneVsAll(all_theta, X)

	m = size(X, 1);
	num_labels = size(all_theta, 1);

	p = zeros(size(X, 1), 1);

	X = [ones(m, 1) X];

	A = X * all_theta';
	[m, p] = max(A, [], 2);

endfunction
