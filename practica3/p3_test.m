##
## Ejercicio con redes neuronales
##

#Ahora cargamos el resultad ode haber entrendo la red neuronal en Theta1 y Theta2
load('ex3weights.mat');
#Theta1 es de dimension 25x401
#Theta2 es de dimension 10x26

pred = prediccion(Theta1, Theta2, X);
fprintf('\nCorreccion de los ejemplos de entrenamiento: %f\n', mean(double(pred == y)) * 100);

#Pintamos los ejemplos, junto a lo que está prediciendo

#Elige un ejemplo aleatorio
rp = randperm(m);

#Y cogemos por ejemplo el primero
displayData(X(rp(1), :));

pred = prediccion(Theta1, Theta2, X(rp(1),:));
fprintf('\nPrediccion: %d\n', mod(pred, 10));
