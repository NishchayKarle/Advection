### Compiling serial code
> make my_advection_program <br>

### Compiling parallel code
> make my_advection_program_parallel <br>

### Executing 
> ./my_advection_program 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 lax -1 -1

> ./my_advection_program 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 first_order_upwind -1 -1

> ./my_advection_program 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 second_order_upwind -1 -1

> ./my_advection_program_parallel 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 lax 8 8

> ./my_advection_program_parallel 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 first_order_upwind 8 8

> ./my_advection_program_parallel 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 second_order_upwind 8 8

### Clean
> make clean

## GIFS
### Lax
![LAX.GIF](lax.gif)

### First Order Upwind
![FST.GIF](first_order_upwind.gif)

### Second Order Upwind
![SND.GIF](second_order_upwind.gif)

## GRAPHS
### Lax N = 200
![200_N_lax_static.png](Graphs/200N_lax_static.png)

### Lax N = 3200
![3200_N_lax_static.png](Graphs/3200N_lax_static.png)

### First Order Upwind N = 200
![200_N_first_order_static.png](Graphs/200N_first_order_static.png)

### First Order Upwind N = 3200
![3200_N_first_order_static.png](Graphs/3200N_first_order_static.png)

### Second Order Upwind N = 200
![200_N_second_order_static.png](Graphs/200N_second_order_static.png)

### Second Order Upwind N = 3200 
![3200_N_second_order_static.png](Graphs/3200N_second_order_static.png)

![Weak Scaling](Graphs/weak_scale.png)


* All tests were run on a system which has 64 threads avaialble to the user
* compiler used gcc9
* grind rate: 3,636,363,636.3636363636 cells/sec
* Files 01_serial.txt and 01_parallel.txt were generated at timestamp NT=1000 using serial and parallel execution and have the same output. (./write_to_file_parallel 200 2000 1.0 1.0e3 5.0e-7 2.85e-7 lax -1 -1, ./write_to_file_parallel 200 2000 1.0 1.0e3 5.0e-7 2.85e-7 lax 8 8)
