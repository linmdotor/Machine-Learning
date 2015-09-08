##
## Support Vector Machines.
##


# Primero cargamos los dataset y los pintamos
load('ex6data1.mat'); # Almacena los datos leídos en X, y
plotData(X, y);


#Llamamos a SVM con kernel LINEAL con distintos valores de C
figure(2);
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

figure(3);
C = 100;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);



#Ahora empleamos el kernel GAUSIANO para otros dataset
x1 = [1 2 1];
x2 = [0 4 -1];
sigma = 2;
sim = kernelGaussiano(x1, x2, sigma);

# Ahora cargamos los nuevos dataset y los pintamos
load('ex6data2.mat'); # Almacena los datos leídos en X, y
plotData(X, y);

#Llamamos a SVM con kernel GAUSSIANO con distintos valores de C y sigma dados
figure(4);
C = 1;
sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) kernelGaussiano(x1, x2, sigma)); 
visualizeBoundary(X, y, model);




# Ahora cargamos los nuevos dataset y los pintamos
load('ex6data3.mat'); # Almacena los datos leídos en X, y
plotData(X, y);

#Probamos con distintos parámetros de SVM
[C, sigma] = seleccionarParametros(X, y, Xval, yval);

figure(5);
model= svmTrain(X, y, C, @(x1, x2) kernelGaussiano(x1, x2, sigma));
visualizeBoundary(X, y, model);