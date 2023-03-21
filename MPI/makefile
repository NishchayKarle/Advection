lax: lax.c
	mpicc -fopenmp -o lax lax.c

first: first_order.c
	mpicc -fopenmp -o first first.c

second: sec_order.c
	mpicc -fopenmp -o second sec_order.c

clean: lax first sec_order
	rm -rf lax first sec_order