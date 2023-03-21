#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define TAG 1111
#define MASTER 0

int num_tasks,
    task_id;
MPI_Status status;

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

double **init_cont_2d_array(int size, int offset)
{
    double *arr = malloc(sizeof(double) * size);
    double **arr2 = malloc(sizeof(double *) * (size / offset));

    for (int i = 0; i < size / offset; i++)
    {
        arr2[i] = (arr + i * offset);
    }

    return arr2;
}

void destroy_2d(double **arr)
{
    free(arr[0]);
    free(arr);
}

double get_gaussian_value(int i, int j, double dx, double dy, double x_0,
                          double y_0, double s_x, double s_y)
{
    return exp(-(pow(i * dx - x_0, 2) / (2 * pow(s_x, 2)) +
                 pow(j * dx - y_0, 2) / (2 * pow(s_y, 2))));
}

double sec_order_val(double i_j, double in_j, double inn_j, double ip_j, double ipp_j,
                     double i_jn, double i_jnn, double i_jp, double i_jpp, double u, double v,
                     double dx, double dy, double dt)
{
    if (u >= 0 && v >= 0)
        return (
            (
                (-v * ((3 * i_j - 4 * i_jp + i_jpp) / (2 * dy))) +
                (-u * ((3 * i_j - 4 * ip_j + ipp_j) / (2 * dx)))) *
                dt +
            i_j);

    else if (u < 0 && v < 0)
        return (
            (
                (-v * ((-3 * i_j + 4 * i_jn - i_jnn) / (2 * dy))) +
                (-u * ((-3 * i_j + 4 * in_j - inn_j) / (2 * dx)))) *
                dt +
            i_j);

    else if (u < 0)
        return (
            (
                (-v * ((3 * i_j - 4 * i_jp + i_jpp) / (2 * dy))) +
                (-u * ((-3 * i_j + 4 * in_j - inn_j) / (2 * dx)))) *
                dt +
            i_j);
    else
        return (
            (
                (-v * ((-3 * i_j + 4 * i_jn - i_jnn) / (2 * dy))) +
                (-u * ((3 * i_j - 4 * ip_j + ipp_j) / (2 * dx)))) *
                dt +
            i_j);
}

void pad_matrix(int cols, double arr[cols + 4][cols + 4], double arr_up[2][cols],
                double arr_down[2][cols], double arr_left[cols][2], double arr_right[cols][2])
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < cols; j++)
        {
            arr[i][j + 2] = arr_up[i][j];
            arr[cols + 2 + i][j + 2] = arr_down[i][j];
        }

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < cols; i++)
        {
            arr[i + 2][j] = arr_left[i][j];
            arr[i + 2][cols + 2 + j] = arr_right[i][j];
        }
}

void update_matrix(int cols, double arr2[cols + 4][cols + 4], double arr[cols + 4][cols + 4],
                   double u, double v, double dx, double dt)
{

    for (int i = 2; i < cols + 2; i++)
        for (int j = 2; j < cols + 2; j++)
            arr2[i][j] = sec_order_val(arr[i][j], arr[i + 1][j], arr[i + 2][j], arr[i - 1][j],
                                       arr[i - 2][j], arr[i][j + 1], arr[i][j + 2], arr[i][j - 1],
                                       arr[i][j - 2], u, v, dx, dx, dt);
}

void second_order_upwind(int N, int NT, double L, double T, double u, double v, double dx, double dt)
{
    int CHUNK_SIZE = sqrt(N * N / num_tasks);
    int N_CHUNK_SIZE = N / CHUNK_SIZE;

    // double **arr = init_cont_2d_array(((CHUNK_SIZE + 4) * (CHUNK_SIZE + 4)), CHUNK_SIZE + 4);
    // double **arr2 = init_cont_2d_array(((CHUNK_SIZE + 4) * (CHUNK_SIZE + 4)), CHUNK_SIZE + 4);
    double arr[CHUNK_SIZE + 4][CHUNK_SIZE + 4];
    double arr2[CHUNK_SIZE + 4][CHUNK_SIZE + 4];

    double x_0 = L / 2, y_0 = L / 2;
    double s_x = L / 4, s_y = L / 4;

    int x = task_id / (N_CHUNK_SIZE),
        y = task_id % (N_CHUNK_SIZE);

    for (int i = 0; i < CHUNK_SIZE + 4; i++)
        for (int j = 0; j < CHUNK_SIZE + 4; j++)
            arr2[i][j] = 0.0;

    for (int i = 0; i < CHUNK_SIZE; i++)
        for (int j = 0; j < CHUNK_SIZE; j++)
            arr2[i + 2][j + 2] = get_gaussian_value(i + x * CHUNK_SIZE, j + y * CHUNK_SIZE, dx, dx, x_0, y_0, s_x, s_y);

    int left = x * (N_CHUNK_SIZE) + mod(y - 1, N_CHUNK_SIZE),
        right = x * (N_CHUNK_SIZE) + mod(y + 1, N_CHUNK_SIZE),
        up = mod(x - 1, N_CHUNK_SIZE) * (N_CHUNK_SIZE) + y,
        down = mod(x + 1, N_CHUNK_SIZE) * (N_CHUNK_SIZE) + y;

    // double **arr_left = init_cont_2d_array(2 * CHUNK_SIZE, CHUNK_SIZE),
    //        **arr_right = init_cont_2d_array(2 * CHUNK_SIZE, CHUNK_SIZE),
    //        **arr_up = init_cont_2d_array(2 * CHUNK_SIZE, CHUNK_SIZE),
    //        **arr_down = init_cont_2d_array(2 * CHUNK_SIZE, CHUNK_SIZE);

    double arr_left[CHUNK_SIZE][2],
        arr_right[CHUNK_SIZE][2],
        arr_up[2][CHUNK_SIZE],
        arr_down[2][CHUNK_SIZE];

    // NT EXECUTION BEGINS
    for (int n = 1; n <= NT; n++)
    {
        for (int i = 0; i < CHUNK_SIZE + 4; i++)
            for (int j = 0; j < CHUNK_SIZE + 4; j++)
                arr[i][j] = arr2[i][j];

        if ((x + y) % 2 == 0)
        {

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < CHUNK_SIZE; j++)
                {
                    arr_up[i][j] = arr[2 + i][j + 2];
                    arr_down[i][j] = arr[CHUNK_SIZE + i][j + 2];
                }

            for (int j = 0; j < 2; j++)
                for (int i = 0; i < CHUNK_SIZE; i++)
                {
                    arr_left[i][j] = arr[2 + i][j + 2];
                    arr_right[i][j] = arr[i + 2][CHUNK_SIZE + j];
                }

            // SEND TO ALL 4 NEIGHBORS
            MPI_Send(arr_up, 2 * CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD);
            MPI_Send(arr_down, 2 * CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD);
            MPI_Send(arr_left, 2 * CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD);
            MPI_Send(arr_right, 2 * CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD);

            // RECEIVE FROM ALL 4 NEIGHBORS
            MPI_Recv(arr_down, 2 * CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(arr_up, 2 * CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(arr_right, 2 * CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(arr_left, 2 * CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD, &status);

            pad_matrix(CHUNK_SIZE, arr, arr_up, arr_down, arr_left, arr_right);
            update_matrix(CHUNK_SIZE, arr2, arr, u, v, dx, dt);
        }

        else
        {
            // RECEIVE FROM ALL 4 NEIGHBORS
            MPI_Recv(arr_down, 2 * CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(arr_up, 2 * CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(arr_right, 2 * CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(arr_left, 2 * CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD, &status);

            // PAD MATRIX
            pad_matrix(CHUNK_SIZE, arr, arr_up, arr_down, arr_left, arr_right);

            // UPDATE MATRIX
            update_matrix(CHUNK_SIZE, arr2, arr, u, v, dx, dt);

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < CHUNK_SIZE; j++)
                {
                    arr_up[i][j] = arr[2 + i][j + 2];
                    arr_down[i][j] = arr[CHUNK_SIZE + i][j + 2];
                }

            for (int j = 0; j < 2; j++)
                for (int i = 0; i < CHUNK_SIZE; i++)
                {
                    arr_left[i][j] = arr[2 + i][j + 2];
                    arr_right[i][j] = arr[i + 2][CHUNK_SIZE + j];
                }

            // SEND TO ALL 4 NEIGHBORS
            MPI_Send(arr_up, 2 * CHUNK_SIZE, MPI_DOUBLE, up, TAG, MPI_COMM_WORLD);
            MPI_Send(arr_down, 2 * CHUNK_SIZE, MPI_DOUBLE, down, TAG, MPI_COMM_WORLD);
            MPI_Send(arr_left, 2 * CHUNK_SIZE, MPI_DOUBLE, left, TAG, MPI_COMM_WORLD);
            MPI_Send(arr_right, 2 * CHUNK_SIZE, MPI_DOUBLE, right, TAG, MPI_COMM_WORLD);
        }

        // double **temp = arr;
        // arr = arr2;
        // arr2 = temp;
    }

    if (task_id == MASTER)
    {
        double **final_arr = (double **)malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++)
            final_arr[i] = (double *)malloc(sizeof(double) * N);

        for (int i = 0; i < CHUNK_SIZE; i++)
            for (int j = 0; j < CHUNK_SIZE; j++)
                final_arr[i][j] = arr2[i + 2][j + 2];

        for (int i = 0; i < N_CHUNK_SIZE; i++)
            for (int j = 0; j < N_CHUNK_SIZE; j++)
            {
                if (i + j == 0)
                    continue;
                MPI_Recv(arr, (CHUNK_SIZE + 4) * (CHUNK_SIZE + 4), MPI_DOUBLE, (i * N_CHUNK_SIZE + j),
                         TAG, MPI_COMM_WORLD, &status);

                for (int k = 0; k < CHUNK_SIZE; k++)
                    for (int l = 0; l < CHUNK_SIZE; l++)
                        final_arr[k + i * CHUNK_SIZE][l + j * CHUNK_SIZE] = arr[k + 2][l + 2];
            }

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
                printf("%e ", final_arr[i][j]);
            printf("\n");
        }
    }

    else
        MPI_Send(arr2, (CHUNK_SIZE + 4) * (CHUNK_SIZE + 4), MPI_DOUBLE, MASTER, TAG, MPI_COMM_WORLD);

    // destroy_2d(arr_left);
    // destroy_2d(arr_right);
    // destroy_2d(arr_up);
    // destroy_2d(arr_down);

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

        second_order_upwind(N, NT, L, T, u, v, dx, dt);

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