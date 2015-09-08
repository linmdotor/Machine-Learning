##
## Clustering.
##


# Primero cargamos el dataset de ejemplo
load('ex7data2.mat'); # Almacena los datos leídos en X, y

# Inicializamos los centroides
initial_centroids = [3 3; 6 2; 8 5];
max_iters = 10;

#Hacemos la prueba sobre los datos de entrenamiento
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);



#Ahora aplicamos k-means para la compresión de imágenes

#Cargamos la imagen
A = double(imread('bird_small.png'));
A = A / 255; #Normaliza los valores
figure(2);
imagesc(A);

#Transformamos la matriz a 2 dimensiones
img_size = size(A);



#Creamos una matriz Nx3 para los 3 colores (rojo, verde, azul) y N para todos los pixels
X = reshape(A, img_size(1) * img_size(2), 3);

#Ejecutamos el k-means
K = 15; 
max_iters = 10;

#Inicializamos los centroides aleatoriamente y ejecutamos
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

#Encontramos los centroides más representativos
idx = findClosestCentroids(X, centroids);

#Ajustamos cada pixel al centroide más cercano
X_recovered = centroids(idx,:);

#Recuperamos la imagen
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

#Dibujamos el resultado 
figure(3);
imagesc(X_recovered)
