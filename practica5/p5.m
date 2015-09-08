##
## Regresión lineal regularizada: sesgo y varianza.
##


# Primero cargamos los datos de la presa
load('ex5data1.mat'); # Almacena los datos leídos en X, y, Xval, yval, Xtest, ytest

m = size(X, 1);

#Mostramos los datos de ejemplo
plot(X, y, 'rx', 'MarkerSize', 5, 'LineWidth', 2);

#Calculamos el coste y gradiente
theta = [1 ; 1];
landa=1;
[J, grad] = costeRegresionLinealRegularizada([ones(m, 1) X], y, theta, landa);

#Ahora buscamos el valor theta que minimiza el coste (empleando fmincg)
#Como son pocos parámetros, emepleamos landa=0

landa=0;
[theta] = entrenaRegularizacionLineal([ones(m, 1) X], y, landa);

#Y pintamos la recta resultante, que se debería acercar a los puntos
hold on;
plot(X, [ones(m, 1) X]*theta, '-', 'LineWidth', 2);
hold off;




#Emplearemos curvas de aprendizaje repitiendo el entrenamiento con distintos subconjuntos
[error_train, error_val] = curvaAprendizaje([ones(m, 1) X], y, [ones(size(Xval, 1), 1) Xval], yval, landa);

#Y pintamos la grafica
figure(2);
plot(1:m, error_train, 1:m, error_val);



#Aplicamos la regresión polinomial para conseguir un mayor ajuste

#Empezamos generando "nuevos" datos de entrenamiento
p = 8;
X_polinomial = elevarPotenciaAtributos(X, p);

#Los normalizamos para que el rango no sea muy grande
[X_polinomial, mu, sigma] = featureNormalize(X_polinomial);
X_polinomial = [ones(m, 1), X_polinomial];

% Normalizamos los datos Xval y Xtest con mu y sigma
Xtest_polinomial = elevarPotenciaAtributos(Xtest, p);
Xtest_polinomial = bsxfun(@minus, Xtest_polinomial, mu);
Xtest_polinomial = bsxfun(@rdivide, Xtest_polinomial, sigma);
Xtest_polinomial = [ones(size(Xtest_polinomial, 1), 1), Xtest_polinomial];

Xval_polinomial = elevarPotenciaAtributos(Xval, p);
Xval_polinomial = bsxfun(@minus, Xval_polinomial, mu);
Xval_polinomial = bsxfun(@rdivide, Xval_polinomial, sigma);
Xval_polinomial = [ones(size(Xval_polinomial, 1), 1), Xval_polinomial];



landa = 0;
[theta] = entrenaRegularizacionLineal(X_polinomial, y, landa);

#Pintamos la nueva curva
figure(3);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);


#Repetimos para distintos subconjuntos (curvas de aprendizaje)
[error_train, error_val] = curvaAprendizaje(X_polinomial, y, Xval_polinomial, yval, landa);

#Y pintamos la grafica
figure(4);
plot(1:m, error_train, 1:m, error_val);



#Ahora intentamos comprobar cuál es el mejor valor de landa
#Repitiendo para distintos valores de landa, el error sobre un conjunto de validación

[landa_vec, error_train, error_val] = ...
    curvaValidacion(X_polinomial, y, Xval_polinomial, yval);

figure(5);
plot(landa_vec, error_train, landa_vec, error_val);

fprintf('landa\t\tError Entren.\tError Validacion\n');
for i = 1:length(landa_vec)
	fprintf(' %f\t%f\t%f\n', ...
            landa_vec(i), error_train(i), error_val(i));
end
