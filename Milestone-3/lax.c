#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#define TAG 1111
#define MASTER 0

int num_tasks,
	task_id;
MPI_Status status;

double **create_cont_2d_array(int size, int offset)
{
	double *arr = malloc(sizeof(double) * size);
	double **arr2 = malloc(sizeof(double *) * (size / offset));

	for (int i = 0; i < size / offset; i++)
	{
		arr2[i] = (arr + i * offset);
	}

	return arr2;
}

void set_2d_offset(double **arr, int size, int offset)
{

	for (int i = 0; i < size / offset; i++)
	{
		arr[i] = (arr[0] + i * offset);
	}
}

double *create_cont_1d_array(int size)
{
	double *arr = malloc(sizeof(double) * size);
	return arr;
}

void destroy_1d(double *arr)
{
	free(arr);
}

void destroy_2d(double **arr)
{
	free(arr[0]);
	free(arr);
}

int mod(int a, int b)
{
	int r = a % b;
	return r < 0 ? r + b : r;
}

double get_gaussian_value(int i, int j, double dx, double dy, double x_0,
						  double y_0, double s_x, double s_y)
{
	return exp(-(pow(i * dx - x_0, 2) / (2 * pow(s_x, 2)) +
				 pow(j * dx - y_0, 2) / (2 * pow(s_y, 2))));
}

double lax_val(double in_j, double ip_j, double i_jn, double i_jp,
			   double u, double v, double dt, double dx)
{
	return ((in_j + ip_j + i_jn + i_jp) / 4) - (((u * (in_j - ip_j)) + (v * (i_jn - i_jp))) * (dt / (2 * dx)));
}

// UPDATE INNER MATRIX
void update_inner(int cols, double arr2[cols][cols], double arr[cols][cols],
				  double u, double v, double dt, double dx)
{
#pragma omp parallel for num_threads(16)
	for (int i = 1; i < cols - 1; i++)
		for (int j = 1; j < cols - 1; j++)
			arr2[i][j] = lax_val(arr[i + 1][j], arr[i - 1][j],
								 arr[i][j + 1], arr[i][j - 1], u, v, dt, dx);
}

// UPDATE BOUNDARIES EXCEPT CORNERS
void update_boundaries(int cols, double arr2[cols][cols], double arr[cols][cols], double arr_up[cols],
					   double arr_down[cols], double arr_left[cols], double arr_right[cols],
					   double u, double v, double dt, double dx)
{
#pragma omp parallel for num_threads(16)
	for (int i = 1; i < cols - 1; i++)
	{
		// TOP
		arr2[0][i] = lax_val(arr[1][i], arr_up[i], arr[0][i + 1], arr[0][i - 1], u, v, dt, dx);

		// BOTTOM
		arr2[cols - 1][i] = lax_val(arr_down[i], arr[cols - 2][i], arr[cols - 1][i + 1],
									arr[cols - 1][i - 1], u, v, dt, dx);

		// LEFT
		arr2[i][0] = lax_val(arr[i + 1][0], arr[i - 1][0], arr[i][1], arr_left[i], u, v, dt, dx);

		// RIGHT
		arr2[i][cols - 1] = lax_val(arr[i + 1][cols - 1], arr[i - 1][cols - 1],
									arr_right[i], arr[i][cols - 2], u, v, dt, dx);
	}
}

void update_corners(int cols, double arr2[cols][cols], double arr[cols][cols], double arr_up[cols],
					double arr_down[cols], double arr_left[cols], double arr_right[cols],
					double u, double v, double dt, double dx)
{
	arr2[0][0] = lax_val(cols == 1 ? arr_down[0] : arr[1][0], arr_up[0],
						 cols == 1 ? arr_right[0] : arr[0][1], arr_left[0], u, v, dt, dx);

	arr2[cols - 1][0] = lax_val(arr_down[0], cols == 1 ? arr_up[0] : arr[cols - 2][0],
								cols == 1 ? arr_right[0] : arr[cols - 1][1],
								arr_left[cols - 1], u, v, dt, dx);

	arr2[0][cols - 1] = lax_val(cols == 1 ? arr_down[0] : arr[1][cols - 1], arr_up[cols - 1],
								arr_right[0], cols == 1 ? arr_left[0] : arr[0][cols - 2],
								u, v, dt, dx);

	arr2[cols - 1][cols - 1] = lax_val(arr_down[cols - 1],
									   cols == 1 ? arr_up[cols - 1] : arr[cols - 2][cols - 1],
									   arr_right[cols - 1],
									   cols == 1 ? arr_left[cols - 1] : arr[cols - 1][cols - 2],
									   u, v, dt, dx);
}

void lax(int N, int NT, double L, double T, double u, double v, double dx, double dt)
{
	int CHUNK_SIZE = sqrt(N * N / num_tasks);
	int N_CHUNK_SIZE = N / CHUNK_SIZE;

	// double **arr = create_cont_2d_array(CHUNK_SIZE * CHUNK_SIZE, CHUNK_SIZE);
	// double **arr2 = create_cont_2d_array(CHUNK_SIZE * CHUNK_SIZE, CHUNK_SIZE);
	double arr[CHUNK_SIZE][CHUNK_SIZE];
	double arr2[CHUNK_SIZE][CHUNK_SIZE];

	double x_0 = L / 2, y_0 = L / 2;
	double s_x = L / 4, s_y = L / 4;

	int x = task_id / (N_CHUNK_SIZE),
		y = task_id % (N_CHUNK_SIZE);

#pragma omp parallel for num_threads(16)
	for (int i = 0; i < CHUNK_SIZE; i++)
		for (int j = 0; j < CHUNK_SIZE; j++)
			arr2[i][j] = get_gaussian_value(i + x * CHUNK_SIZE, j + y * CHUNK_SIZE, dx, dx, x_0, y_0, s_x, s_y);

	int left = x * (N_CHUNK_SIZE) + mod(y - 1, N_CHUNK_SIZE),
		right = x * (N_CHUNK_SIZE) + mod(y + 1, N_CHUNK_SIZE),
		up = mod(x - 1, N_CHUNK_SIZE) * (N_CHUNK_SIZE) + y,
		down = mod(x + 1, N_CHUNK_SIZE) * (N_CHUNK_SIZE) + y;

	// double *arr_left = create_cont_1d_array(CHUNK_SIZE),
	// 	   *arr_right = create_cont_1d_array(CHUNK_SIZE),
	// 	   *arr_up = create_cont_1d_array(CHUNK_SIZE),
	// 	   *arr_down = create_cont_1d_array(CHUNK_SIZE);

	double arr_left[CHUNK_SIZE],
		arr_right[CHUNK_SIZE],
		arr_up[CHUNK_SIZE],
		arr_down[CHUNK_SIZE];

	// FOR LOOP FOR NT GOES HERE. INIT ARR2 BEFORE THIS AND COPY ARR2 INTO ARR1
	for (int n = 1; n <= NT; n++)
	{
#pragma omp parallel for num_threads(16)
		for (int i = 0; i < CHUNK_SIZE; i++)
			for (int j = 0; j < CHUNK_SIZE; j++)
				arr[i][j] = arr2[i][j];

		// UPDATE INNER MATRIX
		update_inner(CHUNK_SIZE, arr2, arr, u, v, dt, dx);

		if ((x + y) % 2 == 0)
		{
#pragma omp parallel for num_threads(16)
			for (int i = 0; i < CHUNK_SIZE; i++)
			{
				arr_left[i] = arr[i][0];
				arr_right[i] = arr[i][CHUNK_SIZE - 1];
				arr_up[i] = arr[0][i];
				arr_down[i] = arr[CHUNK_SIZE - 1][i];
			}

			// SEND TO ALL 4 NEIGHBORS
			MPI_Send(arr_up, CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD);
			MPI_Send(arr_down, CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD);
			MPI_Send(arr_left, CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD);
			MPI_Send(arr_right, CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD);

			// RECEIVE FROM ALL 4 NEIGHBORS
			MPI_Recv(arr_down, CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(arr_up, CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(arr_right, CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(arr_left, CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD, &status);

			// UPDATE BOUNDARIES EXCEPT CORNERS
			update_boundaries(CHUNK_SIZE, arr2, arr, arr_up, arr_down, arr_left, arr_right, u, v, dt, dx);

			// UPDATE CORNERS
			update_corners(CHUNK_SIZE, arr2, arr, arr_up, arr_down, arr_left, arr_right, u, v, dt, dx);
		}

		else
		{
			// RECEIVE FROM ALL 4 NEIGHBORS
			MPI_Recv(arr_down, CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(arr_up, CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(arr_right, CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(arr_left, CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD, &status);

			// UPDATE BOUNDARIES EXCEPT CORNERS
			update_boundaries(CHUNK_SIZE, arr2, arr, arr_up, arr_down, arr_left, arr_right, u, v, dt, dx);

			// UPDATE CORNERS
			update_corners(CHUNK_SIZE, arr2, arr, arr_up, arr_down, arr_left, arr_right, u, v, dt, dx);

#pragma omp parallel for num_threads(16)
			for (int i = 0; i < CHUNK_SIZE; i++)
			{
				arr_left[i] = arr[i][0];
				arr_right[i] = arr[i][CHUNK_SIZE - 1];
				arr_up[i] = arr[0][i];
				arr_down[i] = arr[CHUNK_SIZE - 1][i];
			}

			// SEND TO ALL 4 NEIGHBORS
			MPI_Send(arr_up, CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD);
			MPI_Send(arr_down, CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD);
			MPI_Send(arr_left, CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD);
			MPI_Send(arr_right, CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD);
		}
	}

	if (task_id == MASTER)
	{
		double **final_arr = (double **)malloc(sizeof(double *) * N);
		for (int i = 0; i < N; i++)
			final_arr[i] = (double *)malloc(sizeof(double) * N);

		for (int i = 0; i < CHUNK_SIZE; i++)
			for (int j = 0; j < CHUNK_SIZE; j++)
				final_arr[i][j] = arr2[i][j];

		for (int i = 0; i < N_CHUNK_SIZE; i++)
			for (int j = 0; j < N_CHUNK_SIZE; j++)
			{
				if (i + j == 0)
					continue;
				MPI_Recv(arr, CHUNK_SIZE * CHUNK_SIZE, MPI_DOUBLE, (i * N_CHUNK_SIZE + j),
						 TAG, MPI_COMM_WORLD, &status);
				// set_2d_offset(arr, CHUNK_SIZE * CHUNK_SIZE, CHUNK_SIZE);
				for (int k = 0; k < CHUNK_SIZE; k++)
					for (int l = 0; l < CHUNK_SIZE; l++)
						final_arr[k + i * CHUNK_SIZE][l + j * CHUNK_SIZE] = arr[k][l];
			}

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				printf("%e ", final_arr[i][j]);
			printf("\n");
		}
	}

	else
		MPI_Send(arr2, CHUNK_SIZE * CHUNK_SIZE, MPI_DOUBLE, MASTER, TAG, MPI_COMM_WORLD);

	// destroy_1d(arr_left);
	// destroy_1d(arr_right);
	// destroy_1d(arr_up);
	// destroy_1d(arr_down);

	// destroy_2d(arr);
	// destroy_2d(arr2);

	return;
}

int main(int argc, char **argv)
{
	if (argc == 7)
	{
		// double start = omp_get_wtime();
		int N, NT;
		double L, T, u, v;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
		MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
		if (task_id == 0)
		{
			N = atoi(argv[1]);
			NT = atoi(argv[2]);
			L = atof(argv[3]);
			T = atof(argv[4]);
			u = atof(argv[5]);
			v = atof(argv[6]);
			MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&NT, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&u, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&NT, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&u, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}

		double dx = L / N, dt = T / NT;
		assert(dt <= dx / pow(2 * (u * u + v * v), 0.5));

		lax(N, NT, L, T, u, v, dx, dt);

		// double end = omp_get_wtime();
		MPI_Finalize();
		return EXIT_SUCCESS;
	}

	else
	{
		printf("ERROR: Too few/many arguments\n");
		return EXIT_FAILURE;
	}
	return 0;
}