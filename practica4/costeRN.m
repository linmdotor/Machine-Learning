##
## Calcula el coste y el gradiente de una red neuronal de dos capas.
## 	Implementada de forma que funciona con culauqier número de etiquetas (>= 3)
##
## Uso:
##	J, grad = costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, lambda)
##

function [J, grad] = costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, lambda)

	Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas +1)), num_ocultas, (num_entradas + 1));
	Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas +1));

	m = size(X, 1);

	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));

	# Tomamos k como numero de etiquetas
	k = num_etiquetas;
	Y = eye(k)(y, :);

	# Calculamos por partes la formula del coste
	a1 = [ones(m, 1), X];
	a2 = sigmoide(Theta1 * a1');
	a2 = [ones(1, size(a2, 2)); a2];
	a3 = sigmoide(Theta2 * a2);

	# Aplicamos la fórmula de coste de una red neuronal sin regulaización
	J = (1 / m) * sum(sum((-Y .* log(a3)') - ((1 - Y) .* log(1 - a3)'), 2));

	# Y añadimos el término de regularización
	Theta1Reg = Theta1(:,2:end);
	Theta2Reg = Theta2(:,2:end);
	J = J + (lambda / (2*m)) * (sum(Theta1Reg(:) .^ 2) + sum(Theta2Reg(:) .^ 2));
	

	# Implementamos la retropropagación al sistema
	# Mediante un bucle que procese los ej. de entranamiento
	
	Delta1 = 0;
	Delta2 = 0;

	for t = 1:m
		# 1. Pasada hacia adelante
		a1 = [1; X(t, :)'];
		z2 = Theta1 * a1;
		a2 = [1; sigmoide(z2)];
		z3 = Theta2 * a2;
		a3 = sigmoide(z3);

		# 2. Capa de salida
		d3 = a3 - Y(t, :)';

		# 3. Capa oculta
		d2 = (Theta2Reg' * d3) .* sigmoideDerivada(z2);

		# 4. Acumula el gradiente
		Delta2 = Delta2 + (d3 * a2');
		Delta1 = Delta1 + (d2 * a1');
	endfor

	# Calculamos el gradiente
	Theta1_grad = (1/m) * Delta1;
	Theta2_grad = (1/m) * Delta2;


	# Ahora aplicamos la regularización con la función de coste y gradiente
	Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1Reg);
	Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2Reg);

	grad = [Theta1_grad(:) ; Theta2_grad(:)];

endfunction
