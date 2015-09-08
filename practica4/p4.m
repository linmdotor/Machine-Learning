##
## Redes neuronales para imágenes.
##


# Primero cargamos las imagenes
load('ex4data1.mat'); # Almacena los datos leídos en X, y

m = size(X, 1);

# Selecciona aleatoriamente 100 ejemplos y los pinta
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

# Y Cargamos los pesos en Theta1 y Theta2
load('ex4weights.mat');
params_rn = [Theta1(:) ; Theta2(:)];

# Calculamos el coste 
num_etiquetas = 10; #Emplearemos las etiquetas de 1 a 10
num_entradas = 400;
num_ocultas = 25;
lambda = 0;
J = costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, lambda)
lambda = 1;
J = costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, lambda)

# Comprobamos que el gradiente es correcto
checkNNGradients;


# Inicializamos de forma aleatoria los pesos
initial_Theta1 = pesosAleatorios(num_entradas, num_ocultas);
initial_Theta2 = pesosAleatorios(num_ocultas, num_etiquetas);

initial_rn_params = [initial_Theta1(:) ; initial_Theta2(:)];

# Entrenamos la red neuronal
options = optimset('MaxIter', 50);
lambda = 1;
funcionCoste = @(p) costeRN(p, num_entradas, num_ocultas, num_etiquetas, X, y, lambda);

[params_rn, cost] = fmincg(funcionCoste, initial_rn_params, options);

Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));


# Visualizamos la red neuronal
displayData(Theta1(:, 2:end));

# Comprobamos la precisión
pred = prediccion(Theta1, Theta2, X);
fprintf('\nPrecision: %f\n', mean(double(pred == y)) * 100);