##
## Integral de fun entre a y b por el método de Monte Carlo.
##	Para ello genera num_puntos aleatoriamente.
##
## Uso:
##	I = mcint_iterative(fun, a, b, num_puntos)
##

function I = mcint_iterative(fun, a, b, num_puntos)

	# Primero calculamos 50 puntos de la función y su máximo M

	fun_x = [a:(b-a)/50:b];
	fun_y = fun(fun_x);

	M = max(fun_y);

	# Ahora hallamos los puntos aleatorios, entre x(a y b), y(0 y M)
	# Con dichos puntos, vemos cuántos están debajo de la función
	
	puntos_debajo = 0;

	for i = 1:num_puntos
		x = rand()*(b-a)+a;
		y = rand()*M;

		puntos_debajo = puntos_debajo + (y <= fun(x)); 
	endfor

	# Calculamos la Integral

	I = (puntos_debajo/num_puntos)*(b-a)*M;

endfunction