##
## Detección de spam.
##


# Primero procesamos los correos para generar los datos de entrenamiento y validación
%file_contents = readFile('easy_ham/0024.txt');
%email  = processEmail(file_contents);

#Luego convertimos el texto a un vector de atributos (1s y 0s)
%atributos = vectorAtributosEmail(email);


#Ahora lo que debemos hacer es leer todos los entrenamientos
#Y crear los dataset X e y (1 ejemplo por fila)
#Puede tardar varios minutos en generar los ejemplos

#TOTAL easy_ham 2551  #no spam
#TOTAL hard_ham 250  #no spam
#TOTAL spam 500  #spam

#Deberíamos leer TODOS los ejemplos pero
#solo vamos a utilizar unos pocos ejemplos (200)
#Utilizremos 150 para entrenamiento y 50 para validación
fprintf('\nCargando DataSet...\n')
#X = crearDataSet("easy_ham", 100);
#X = [X; crearDataSet("hard_ham", 100)];
#X = [X; crearDataSet("spam", 200)];
#y = zeros(100,1); #no spam
#y = [y; zeros(100,1)];
#y = [y; ones(200,1)]; #spam


#Cargamos 150 y 50 aleatoriamente

m = size(X, 1);
rand_indices = randperm(m);
sel_X = X(rand_indices(1:150), :);
sel_y = y(rand_indices(1:150), :);
Xtest = X(rand_indices(151:200), :);
ytest = y(rand_indices(151:200), :);

#Llamamos a SVM con kernel LINEAL
fprintf('\nEntrenando...\n')
C = 0.1
model = svmTrain(sel_X, sel_y, C, @linearKernel);
p = svmPredict(model, sel_X);
fprintf('Precision de entrenamiento: %f\n', mean(double(p == sel_y)) * 100);

#Ahora comprobamos la precisión
fprintf('\nRealizando test de precision ...\n')
p = svmPredict(model, Xtest);
fprintf('Precision: %f\n', mean(double(p == ytest)) * 100);
























#Probaremos primero con el kernel LINEAL
#C = 0.1;
#model = svmTrain(X, y, C, @linearKernel);
#p = svmPredict(model, X);
#fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);