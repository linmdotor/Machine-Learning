##
## Normaliza un atributo para aplicar el descenso de gradiente.
##	Haciendo el cociente entre su diferencia con la media, 
##	y la desviación estándar del atributo.
##
## Devuelve un "mu" con la media de cada atributo y
##  y un sigma con al desviacion estandar
##


function [X_norm, mu, sigma] = normalizaAtributo(X)

	#Inicializamos mu y sigma
	X_norm = X;
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));


	#Calcula la media y desviacion
	mu = mean(X);
	sigma = std(X);

	t = ones(length(X), 1);
	#Normaliza los atributos 
	X_norm = (X .- (t * mu)) ./ (t * sigma);

endfunction