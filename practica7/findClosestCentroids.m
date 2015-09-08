##
## Calcula los centroides de cada ejemplo
##
## Uso:
##	idx = findClosestCentroids(X, centroids)
##

function idx = findClosestCentroids(X, centroids)

	K = size(centroids, 1);
	idx = zeros(size(X,1), 1);


	#Por cada ejemplo, encuentra el centroide mas cercano y almacena su indice en idx
	for i = 1:length(X)
		deltas = zeros(K,1);
		x = X(i,:);
		for j = 1:K
			k = centroids(j,:);
			delta = x - k;
			deltas(j) = delta * delta';
		end
		[y, idx(i)] = min(deltas);
	end

endfunction
