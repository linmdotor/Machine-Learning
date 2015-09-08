#########################################################
##
## Proyecto final de AA.
## Basado en el dataset "Wine Quality Data Set"
## 		extraído de: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
##
#########################################################

clear ; close all; clc;

##########################################################
#
# Primero lo que vamos a hacer es aplicar distintas técnicas por separado
# E intentar adivinar la "nota" que obtendrían los distintos vinos.
#
##########################################################

num_atr = 12;

# Primero cargamos el dataset de vino blanco
load winequalitywhite.csv;
data_Xw = winequalitywhite(:, 1:num_atr-1);
data_yw = winequalitywhite(:, num_atr);
data_mw = length(data_Xw);
#Y separamos aleatoriamente 250 ejemplos para hacer el test
rand_indices = randperm(data_mw);
Xw = data_Xw(rand_indices(251:data_mw), :);
yw = data_yw(rand_indices(251:data_mw), :);
Xwtest = data_Xw(rand_indices(1:250), :);
ywtest = data_yw(rand_indices(1:250), :);
mw = length(Xw);
mwtest = length(Xwtest);

# Y luego el de vino tinto
load winequalityred.csv;
data_Xr = winequalityred(:, 1:num_atr-1);
data_yr = winequalityred(:, num_atr);
data_mr = length(data_Xr);
#Y separamos aleatoriamente 250 ejemplos para hacer el test
rand_indices = randperm(data_mr);
Xr = data_Xr(rand_indices(251:data_mr), :);
yr = data_yr(rand_indices(251:data_mr), :);
Xrtest = data_Xr(rand_indices(1:250), :);
yrtest = data_yr(rand_indices(1:250), :);
mr = length(Xr);
mrtest = length(Xrtest);

# Y creamos los datos para la distinción entre vino blanco/tinto (clasificación)
# Cogiendo los 250 de blanco y los 250 de tinto para test
X_clasi = [Xw;Xr];
y_clasi = [ones(length(Xw),1);zeros(length(Xr),1)];
m_clasi = length(X_clasi);
X_clasitest = [Xwtest;Xrtest];
y_clasitest = [ones(length(Xwtest),1);zeros(length(Xrtest),1)];
m_clasitest = length(X_clasitest);


#############################################################
#
# T1. Regresión lineal con varios atributos mediante descenso de gradiente (regresión)
#
############################################################
fprintf('\n-------- REGRESION LINEAL --------\n');
fprintf('\n----------------------------------\n');

#Ejecutamos la regresión lineal con varias variables
# y comparamos ambos resultados tomando como predicción algún ejemplo de los elegidos

fprintf('\n--- VINO BLANCO ---\n');

[t, t_normal, mu, sigma] = regresionLinealMultiple(Xw, yw, mw, num_atr);

ejem = Xwtest;
precio_grad = [ones(mwtest,1), (ejem-mu)./sigma] * t;
precio_norm = [ones(mwtest,1), ejem] * t_normal;
precio_real = ywtest;
probabilidad_acierto_grad = mean(round(precio_grad) == precio_real) * 100
probabilidad_acierto_norm = mean(round(precio_norm) == precio_real) * 100

fprintf('\n--- VINO TINTO ---\n');

[t, t_normal, mu, sigma] = regresionLinealMultiple(Xr, yr, mr, num_atr);

ejem = Xrtest;
precio_grad = [ones(mrtest,1), (ejem-mu)./sigma] * t;
precio_norm = [ones(mrtest,1), ejem] * t_normal;
precio_real = yrtest;
probabilidad_acierto_grad = mean(round(precio_grad) == precio_real) * 100
probabilidad_acierto_norm = mean(round(precio_norm) == precio_real) * 100

fprintf('\n--- CLASIFICACION ---\n');

[t, t_normal, mu, sigma] = regresionLinealMultiple(X_clasi, y_clasi, m_clasi, num_atr);

ejem = X_clasitest;
clasi_grad = [ones(m_clasitest,1), (ejem-mu)./sigma] * t;
clasi_norm = [ones(m_clasitest,1), ejem] * t_normal;
clasi_real = y_clasitest;
probabilidad_acierto_grad = mean(round(clasi_grad) == clasi_real) * 100
probabilidad_acierto_norm = mean(round(clasi_norm) == clasi_real) * 100


#A pesar de todo esto podemos ver que los datos aplicados sobre los MISMOS EJEMPLOS
# de entrenamiento no son fiables, pues al ser lineal, es difícil que se ajuste a la realidad.

fprintf('\nPulsa una tecla para continuar.\n');
#pause;


#############################################################
#
# T2. Regresión logística (clasificación)
#
############################################################
fprintf('\n------ REGRESION LOGISTICA -------\n');
fprintf('\n----------------------------------\n');

#Ejecutamos la regresión logística para:
 % Adivinar el tipo de vino (2 clases) blanco-> 1, tinto->0

theta = regresionLogistica(X_clasi, y_clasi, 0.1);

# Evaluamos el porcentaje de aciertos de la funcion
pred = sigmoide(X_clasitest * theta) >= 0.5;
probabilidad_acierto = mean(pred == y_clasitest) * 100

fprintf('\nPulsa una tecla para continuar.\n');
#pause;

#############################################################
#
# T3. Regresión logística multiclase (clasificación)
#
############################################################
fprintf('\n- REGRESION LOGISTICA MULTICLASE -\n');
fprintf('\n----------------------------------\n');

fprintf('\n--- VINO BLANCO ---\n');

#Empleamos el clasificador por regresion logistica
num_etiquetas = 11; #Emplearemos las etiquetas de 0 a 10
[all_theta] = oneVsAll(Xw, yw, num_etiquetas, 0.1);

#Y empleamos la funcion de prediccion para ver el porcentaje de acierto
pred = prediccionOneVsAll(all_theta, Xwtest);
fprintf('\nCorreccion de los ejemplos de entrenamiento: %f\n', mean(double(pred == ywtest)) * 100);

fprintf('\n--- VINO TINTO ---\n');

#Empleamos el clasificador por regresion logistica
num_etiquetas = 11; #Emplearemos las etiquetas de 0 a 10
[all_theta] = oneVsAll(Xr, yr, num_etiquetas, 0.1);

#Y empleamos la funcion de prediccion para ver el porcentaje de acierto
pred = prediccionOneVsAll(all_theta, Xrtest);
fprintf('\nCorreccion de los ejemplos de entrenamiento: %f\n', mean(double(pred == yrtest)) * 100);

fprintf('\n--- CLASIFICACION ---\n');

#Empleamos el clasificador por regresion logistica
num_etiquetas = 2; #Emplearemos las etiquetas de 1 y 2
[all_theta] = oneVsAll(X_clasi, y_clasi, num_etiquetas, 0.1);

#Y empleamos la funcion de prediccion para ver el porcentaje de acierto
pred = prediccionOneVsAll(all_theta, X_clasitest);
fprintf('\nCorreccion de los ejemplos de entrenamiento: %f\n', mean(double(pred == y_clasitest)) * 100);

fprintf('\nPulsa una tecla para continuar.\n');
#pause;


#############################################################
#
# T3. Redes neuronales
#
############################################################
fprintf('\n-------- REDES NEURONALES --------\n');
fprintf('\n----------------------------------\n');

fprintf('\n--- VINO BLANCO ---\n');

num_etiquetas = 11; #Emplearemos las etiquetas de 0 a 10
num_entradas = size(Xw,2);
num_ocultas = 25;

# Inicializamos de forma aleatoria los pesos
initial_Theta1 = pesosAleatorios(num_entradas, num_ocultas);
initial_Theta2 = pesosAleatorios(num_ocultas, num_etiquetas);

initial_rn_params = [initial_Theta1(:) ; initial_Theta2(:)];

# Entrenamos la red neuronal
options = optimset('MaxIter', 50);
lambda = 1;
funcionCoste = @(p) costeRN(p, num_entradas, num_ocultas, num_etiquetas, Xw, yw, lambda);

[params_rn, cost] = fmincg(funcionCoste, initial_rn_params, options);

Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));

# Comprobamos la precisión
pred = prediccion(Theta1, Theta2, Xwtest);
fprintf('\nPrecision: %f\n', mean(double(pred == ywtest)) * 100);


fprintf('\n--- VINO TINTO ---\n');

num_etiquetas = 11; #Emplearemos las etiquetas de 0 a 10
num_entradas = size(Xr,2);
num_ocultas = 25;

# Inicializamos de forma aleatoria los pesos
initial_Theta1 = pesosAleatorios(num_entradas, num_ocultas);
initial_Theta2 = pesosAleatorios(num_ocultas, num_etiquetas);

initial_rn_params = [initial_Theta1(:) ; initial_Theta2(:)];

# Entrenamos la red neuronal
options = optimset('MaxIter', 50);
lambda = 1;
funcionCoste = @(p) costeRN(p, num_entradas, num_ocultas, num_etiquetas, Xr, yr, lambda);

[params_rn, cost] = fmincg(funcionCoste, initial_rn_params, options);

Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));

# Comprobamos la precisión
pred = prediccion(Theta1, Theta2, Xrtest);
fprintf('\nPrecision: %f\n', mean(double(pred == yrtest)) * 100);


fprintf('\n--- CLASIFICACION ---\n');

#Tenemos que sumar 1 a y_clasi y y_clasitest porque las redes neuronales no aceptan
	#indices de 0
y_clasi = y_clasi.+1;
y_clasitest = y_clasitest.+1;

num_etiquetas = 2; #Emplearemos las etiquetas 1 y 2
num_entradas = size(X_clasi,2);
num_ocultas = 25;

# Inicializamos de forma aleatoria los pesos
initial_Theta1 = pesosAleatorios(num_entradas, num_ocultas);
initial_Theta2 = pesosAleatorios(num_ocultas, num_etiquetas);

initial_rn_params = [initial_Theta1(:) ; initial_Theta2(:)];

# Entrenamos la red neuronal
options = optimset('MaxIter', 50);
lambda = 1;
funcionCoste = @(p) costeRN(p, num_entradas, num_ocultas, num_etiquetas, X_clasi, y_clasi, lambda);

[params_rn, cost] = fmincg(funcionCoste, initial_rn_params, options);

Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));

# Comprobamos la precisión
pred = prediccion(Theta1, Theta2, X_clasitest);
fprintf('\nPrecision: %f\n', mean(double(pred == y_clasitest)) * 100);