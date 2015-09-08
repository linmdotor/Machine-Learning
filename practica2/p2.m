##
## Regresión logística.
##


# Primero cargamos los puntos de entrenamiento
load ex2data1.txt;
datos = ex2data1(:,:);

x = datos(:, 1:2);
y = datos(:, 3);
m = length(x); #m será el número de alumnos

# Pintamos los puntos de entrenamiento (x - adminitos, o - no admitidos)
figure(1);clf;
hold on;

# Obtiene un vector con los indices de ejemplos positivos o negativos
positivos = find(y == 1);
negativos = find(y == 0);

# Dibuja los ejemplos
plot(x(positivos, 1), x(positivos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(x(negativos, 1), x(negativos, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

xlabel('Nota del primer examen')
ylabel('Nota del segundo examen')

hold off;

# Calculamos la función de coste y su gradiente
[m, n] = size(x);
x = [ones(m, 1) x];
theta_inicial = zeros(n + 1, 1);
[cost, grad] = coste(theta_inicial, x, y);


# Calculamos el valor optimo
opciones = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(coste(t, x, y)), theta_inicial, opciones);

# Pintamos mediante la funcion plotDecisionBoundary
plotDecisionBoundary(theta, x, y);


# Evaluamos el porcentaje de aciertos de la funcion
pred = sigmoide(x * theta) >= 0.5;
probabilidad_acierto = mean(pred == y)







#
# Regresión logistica regularizada
#

# Primero cargamos los puntos de entrenamiento
load ex2data2.txt;
datos = ex2data2(:,:);

x = datos(:, 1:2);
y = datos(:, 3);

# Pintamos los puntos de entrenamiento (x - adminitos, o - no admitidos)
figure(3);clf;
hold on;

# Obtiene un vector con los indices de ejemplos positivos o negativos
positivos = find(y == 1);
negativos = find(y == 0);

# Dibuja los ejemplos
plot(x(positivos, 1), x(positivos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(x(negativos, 1), x(negativos, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

xlabel('Test 1')
ylabel('Test 2')

hold off;


# Extendemos el numero de aributos hasta la sexta potencia
x = mapFeature(x(:,1), x(:,2));

# Calculamos la funcion de coste regularizada y su gradiente
theta_inicial = zeros(size(x, 2), 1);
landa = 1;
[cost, grad] = costeRegularizada(theta_inicial, x, y, landa);


# Calculamos el valor optimo
opciones = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J] = fminunc(@(t)(costeRegularizada(t, x, y, landa)), theta_inicial, opciones);

# Pintamos mediante la funcion plotDecisionBoundary
plotDecisionBoundary(theta, x, y);

# Evaluamos el porcentaje de aciertos de la funcion
pred = sigmoide(x * theta) >= 0.5;
probabilidad_acierto = mean(pred == y)