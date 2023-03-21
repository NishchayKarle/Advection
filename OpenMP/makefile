my_advection_program: advection.c
	gcc-11 -std=c11 -Wall -Werror -pedantic -fopenmp -O3 -flto -march=native -mtune=native -o my_advection_program advection.c -lm

my_advection_program_parallel: advection.c
	gcc-11 -std=c11 -Wall -Werror -pedantic -fopenmp -DOPENMP -O3 -flto -march=native -mtune=native -o my_advection_program_parallel advection.c -lm

write_to_file: advection.c 
	gcc-11 -std=c11 -Wall -Werror -pedantic -fopenmp -DWRITETOFILE -O3 -flto -march=native -mtune=native -o write_to_file advection.c -lm

write_to_file_parallel: advection.c 
	gcc-11 -std=c11 -Wall -Werror -pedantic -fopenmp -DOPENMP -DWRITETOFILE -O3 -flto -march=native -mtune=native -o write_to_file_parallel advection.c -lm

# valgrind: advection.c
# 	gcc-11 -std=c11 -Wall -Werror -pedantic -g -fopenmp -DOPENMP -o my_advection_program_parallel_valgrind advection.c -lm

all: my_advection_program my_advection_program_parallel write_to_file write_to_file_parallel
	clear

clean: my_advection_program my_advection_program_parallel write_to_file write_to_file_parallel
	rm -rf my_advection_program my_advection_program_parallel write_to_file write_to_file_parallel