##
## Regresión logística multiclase para imágenes.
##


# Primero cargamos las imagenes
load('ex3data1.mat'); # Almacena los datos leídos en X, y

m = size(X, 1);

# Selecciona aleatoriamente 100 ejemplos y los pinta
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

#Empleamos el clasificador por regresion logistica
lambda = 0.1;
num_etiquetas = 10; #Emplearemos las etiquetas de 1 a 10
[all_theta] = oneVsAll(X, y, num_etiquetas, lambda);

#Y empleamos la funcion de prediccion para ver el prcentaje de acierto
pred = prediccionOneVsAll(all_theta, X);
fprintf('\nCorreccion de los ejemplos de entrenamiento: %f\n', mean(double(pred == y)) * 100);