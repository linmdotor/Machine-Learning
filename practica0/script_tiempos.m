
time_mcint = 0;
time_mcint_iterative = 0;
puntos = 0;

num_puntos = 100;
for i = 1:10
	
	puntos = [puntos, num_puntos];
	num_puntos;

	tic()
	for repeticiones = 1:(200/i);
		integral_montecarlo = mcint(@(x) (x.^2 + 2*x) ,0,10,num_puntos);
	endfor
	media = toc()/repeticiones;
	time_mcint = [time_mcint, media];

	tic()
	for repeticiones = 1:(200/i);
		integral_montecarlo_it = mcint_iterative(@(x) (x.^2 + 2*x) ,0,10,num_puntos);
	endfor
	media = toc()/repeticiones;
	time_mcint_iterative = [time_mcint_iterative, media];

	num_puntos = num_puntos * 2;
endfor


plot(puntos, time_mcint, "bo-", "markersize", 4, "linewidth", 2, puntos, time_mcint_iterative, "g+-", "markersize", 4, "linewidth", 2)
title( "Comparativa de tiempos","fontsize", 20)
text(30000, 1.2, "iterative", "fontsize", 12)
text(20000, 0.1, "with vectors", "fontsize", 12)

xlabel("number of points", "fontsize", 14);
ylabel("time (secs)", "fontsize", 14);

print("montecarlo.png", "-dpng")