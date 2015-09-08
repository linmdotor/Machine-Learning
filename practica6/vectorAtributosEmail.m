##
## Convierte el texto de un email en un vector de atributos	
##
## Uso:
##	x = vectorAtributosEmail(email)
##

function x = vectorAtributosEmail(email)

	#Almacenamos el diccionario en un Array y calculamos el
	#Número de palabras en el diccionario
	vocabList = getVocabList();
	n = length(vocabList);

	x = zeros(n, 1);

	#Almacenamos el vocabulario en una estructura
	for i = 1:n
		vocabulario.( vocabList{i} ) = i;
	end

	#Comprobamos si cada palabra está en el diccionario 
	#Y guardamos en el vector de "diccionario" un 1 si está
	while ~isempty(email)
		[str, email] = strtok(email, [' ']);
		% procesa str
		if(isfield(vocabulario, str))
			x(vocabulario.(str)) = isfield(vocabulario, str);
		end
	end

endfunction
