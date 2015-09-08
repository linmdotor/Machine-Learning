##
## Crea un dataset a partir de unos determinados mails de un directorio
##
## Uso:
##	X = crearDataSet(directorio, num_mails)
##

function X = crearDataSet(directorio, num_mails)

	file_name = sprintf("%s/%04d.txt", directorio, 1);
	file_contents = readFile(file_name);
	email  = processEmail(file_contents);
	X = vectorAtributosEmail(email)';
	for i = 2:num_mails
		file_name = sprintf("%s/%04d.txt", directorio, i);
		file_contents = readFile(file_name);
		email  = processEmail(file_contents);
		X = [X; vectorAtributosEmail(email)'];
	end

endfunction
