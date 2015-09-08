##
## Integral de fun entre a y b por el método de Monte Carlo.
##	Para ello genera num_puntos aleatoriamente.
##
## Uso:
##	I = mcint(fun, a, b, num_puntos)
##

function I = mcint(fun, a, b, num_puntos)

	# Primero calculamos 50 puntos de la función y su máximo M

	fun_x = [a:(b-a)/50:b];
	fun_y = fun(fun_x);

	M = max(fun_y);

	# Ahora hallamos los puntos aleatorios, entre x(a y b), y(0 y M)

	x = rand(1,num_puntos)*(b-a)+a;
	y = rand(1, num_puntos)*M;

	#Con dichos puntos, vemos cuántos están debajo de la función

	resu = sum(y <= fun(x));

	#Calculamos la Integral

	I = (resu/num_puntos)*(b-a)*M;

	# Pintamos la gráfica y los puntos
	plot(fun_x, fun_y, "linewidth", 3, x,y, "rx")
	set(gca, "xlim", [a b])
	set(gca, "ylim", [0 M])

endfunction