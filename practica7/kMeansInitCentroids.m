##
## Inicializa los centroides con K elementos aleatorios
##
## Uso:
##	centroids = kMeansInitCentroids(X, K)
##

function centroids = kMeansInitCentroids(X, K)

	centroids = zeros(K, size(X, 2));

	randidx = randperm(size(X, 1));
	centroids = X(randidx(1:K), :);

endfunction

