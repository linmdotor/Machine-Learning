integral_montecarlo = mcint(@(x) (x.^2 + 2*x) ,0,10,1000)
integral_montecarlo_it = mcint_iterative(@(x) (x.^2 + 2*x) ,0,10,1000)
integral_quad = quad( @(x) (x.^2 + 2*x), 0, 10)

#label = ["mcint -> ", num2str(integral_montecarlo), "     ;     quad -> ", num2str(integral_quad)];

#xlabel(label, "fontsize", 18);
#ylabel("1000 points", "fontsize", 18);

#print("montecarlo.png", "-dpng")