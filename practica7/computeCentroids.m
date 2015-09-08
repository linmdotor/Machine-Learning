##
## Calcula la nueva posición de los centroides a partir de las medias
##
## Uso:
##	centroids = computeCentroids(X, idx, K)
##

function centroids = computeCentroids(X, idx, K)

	[m n] = size(X);
	centroids = zeros(K, n);

	#Simplemente calcula la media de los puntos
	for k = 1:K
		centroids(k,:) = mean(X(find(idx == k),:));
	end
	
endfunction