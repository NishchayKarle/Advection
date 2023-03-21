#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <omp.h>

int t1, t2;

int mod(int a, int b);

void file_write(FILE *fp, int N, double **arr);

void destroy(double **arr, int N);

void set_gaussian_init(int N, double **C_n, double dx, double dy, double x_0,
                       double y_0, double s_x, double s_y);

void lax(int N, int NT, double L, double T, double u, double v);

double get_c_n1_first_order_upwind(double **C_n, int N, int i, int j, double u, double v,
                                   double dx, double dy, double dt);

void first_order_upwind(int N, int NT, double L, double T, double u, double v);

double get_c_n1_second_order_upwind(double **C_n, int N, int i, int j, double u, double v,
                                    double dx, double dy, double dt);

void second_order_upwind(int N, int NT, double L, double T, double u, double v);

int main(int argc, char **argv)
{
    if (argc == 10)
    {
        int N = atoi(argv[1]),
            NT = atoi(argv[2]);

        double L = atof(argv[3]),
               T = atof(argv[4]),
               u = atof(argv[5]),
               v = atof(argv[6]);
        
        t1 = atoi(argv[8]);
        t2 = atoi(argv[9]);

        printf("N  - Matrix Dimension: %d\n", N);
        printf("NT - Number of timesteps: %d\n", NT);
        printf("L  - Physical Cartesian Domain Length: %.10lf\n", L);
        printf("T  - Total Physical Timespan: %.10lf\n", T);
        printf("u  - X velocity Scalar: %.10lf\n", u);
        printf("v  - Y velocity Scalar: %.10lf\n", v);
        printf("ESTIMATE FOR MEMORY USAGE: %lu bytes\n", sizeof(double) * N * N * 2);
        printf("THREAD count T1 = %d, T2 = %d, (use -1 -1 for serial run)\n", t1, t2);

        if (strcmp(argv[7], "lax") == 0)
            lax(N, NT, L, T, u, v);

        else if (strcmp(argv[7], "first_order_upwind") == 0)
            first_order_upwind(N, NT, L, T, u, v);

        else
            second_order_upwind(N, NT, L, T, u, v);

        return EXIT_SUCCESS;
    }

    else
    {
        printf("ERROR: Too few/many arguments\n");
        return EXIT_FAILURE;
    }
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

void destroy(double **arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        free(arr[i]);
        arr[i] = NULL;
    }
    free(arr);
    arr = NULL;
    return;
}

void file_write(FILE *fp, int N, double **arr)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fprintf(fp, "%e, ", arr[i][j]);
        }
        fprintf(fp, "\n");
    }
}

void set_gaussian_init(int N, double **arr, double dx, double dy, double x_0,
                       double y_0, double s_x, double s_y)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            arr[i][j] = exp(
                -(
                    pow(i * dx - x_0, 2) / (2 * pow(s_x, 2)) +
                    pow(j * dx - y_0, 2) / (2 * pow(s_y, 2))));
}

void lax(int N, int NT, double L, double T, double u, double v)
{
    double **C_n = (double **)malloc(sizeof(double *) * N);
    double **C_n1 = (double **)malloc(sizeof(double *) * N);

    for (int i = 0; i < N; i++)
    {
        C_n[i] = (double *)malloc(sizeof(double) * N);
        C_n1[i] = (double *)malloc(sizeof(double) * N);
    }

    double dx = L / N, dt = T / NT;

    assert(dt <= dx / pow(2 * (u * u + v * v), 0.5));

    double x_0 = L / 2, y_0 = L / 2;
    double s_x = L / 4, s_y = L / 4;

    set_gaussian_init(N, C_n, dx, dx, x_0, y_0, s_x, s_y);

    double start_time = omp_get_wtime();
    for (int n = 0; n < NT; n++)
    {
#ifdef OPENMP
#pragma omp parallel for default(none) shared(N, C_n, C_n1, u, v, dx, dt, t1, t2) num_threads(t1) schedule(static)
#endif
        for (int i = 0; i < N; i++)
        {
#ifdef OPENMP
#pragma omp parallel for default(none) shared(i, N, C_n, C_n1, u, v, dx, dt, t1, t2) num_threads(t2) schedule(static)
#endif
            for (int j = 0; j < N; j++)
            {
                double C_n_ip_j = C_n[mod(i - 1, N)][j],
                       C_n_in_j = C_n[mod(i + 1, N)][j],
                       C_n_i_jp = C_n[i][mod(j - 1, N)],
                       C_n_i_jn = C_n[i][mod(j + 1, N)];

                C_n1[i][j] =
                    ((C_n_ip_j + C_n_in_j + C_n_i_jp + C_n_i_jn) / 4) -
                    ((u * (C_n_in_j - C_n_ip_j) + v * (C_n_i_jn - C_n_i_jp)) * (dt / (2 * dx)));
            }
        }

// WRITING MATRIX TO FILE AT DIFFERENT TIMESTAMPS
#ifdef WRITETOFILE
        if (n % 1000 == 0)
        {
            char file_name[] = "Lax/xx.txt";
            int index = n / 1000;

            if (index / 10 == 0)
            {
                file_name[4] = '0';
                file_name[5] = '0' + index;
            }

            else
            {
                file_name[4] = '0' + index / 10;
                file_name[5] = '0' + index % 10;
            }

            FILE *fp = fopen(file_name, "w");
            if (ferror(fp))
            {
                printf("ERROR: Couldn't open file - %s\n", file_name);
            }
            file_write(fp, N, C_n);
            fclose(fp);
        }
#endif

        double **temp = C_n;
        C_n = C_n1;
        C_n1 = temp;
    }
    double end_time = omp_get_wtime();
    printf("TIME TAKEN %f\n", end_time - start_time);

    destroy(C_n, N);
    destroy(C_n1, N);
    return;
}

double get_c_n1_first_order_upwind(double **C_n, int N, int i, int j, double u, double v,
                                   double dx, double dy, double dt)
{
    if (u >= 0 && v >= 0)
        return (
            (
                (-v * ((C_n[i][j] - C_n[i][mod(j - 1, N)]) / dy)) +
                (-u * ((C_n[i][j] - C_n[mod(i - 1, N)][j]) / dx))) *
                dt +
            C_n[i][j]);

    else if (u < 0 && v < 0)
        return (
            (
                (-v * ((C_n[i][mod(j + 1, N)] - C_n[i][j]) / dy)) +
                (-u * ((C_n[mod(i + 1, N)][j] - C_n[i][j]) / dx))) *
                dt +
            C_n[i][j]);

    else if (u < 0)
        return (
            (
                (-v * ((C_n[i][j] - C_n[i][mod(j - 1, N)]) / dy)) +
                (-u * ((C_n[mod(i + 1, N)][j] - C_n[i][j]) / dx))) *
                dt +
            C_n[i][j]);
    else
        return (
            (
                (-v * ((C_n[i][mod(j + 1, N)] - C_n[i][j]) / dy)) +
                (-u * ((C_n[i][j] - C_n[mod(i - 1, N)][j]) / dx))) *
                dt +
            C_n[i][j]);
}

void first_order_upwind(int N, int NT, double L, double T, double u, double v)
{
    double **C_n = (double **)malloc(sizeof(double *) * N);
    double **C_n1 = (double **)malloc(sizeof(double *) * N);

    for (int i = 0; i < N; i++)
    {
        C_n[i] = (double *)malloc(sizeof(double) * N);
        C_n1[i] = (double *)malloc(sizeof(double) * N);
    }

    double dx = L / N, dt = T / NT;

    assert(dt <= dx / pow(2 * (u * u + v * v), 0.5));

    double x_0 = L / 2, y_0 = L / 2;
    double s_x = L / 4, s_y = L / 4;

    set_gaussian_init(N, C_n, dx, dx, x_0, y_0, s_x, s_y);

    double start_time = omp_get_wtime();
    for (int n = 0; n < NT; n++)
    {
#ifdef OPENMP
#pragma omp parallel for default(none) shared(N, C_n, C_n1, u, v, dx, dt, t1, t2) num_threads(t1) schedule(static)
#endif
        for (int i = 0; i < N; i++)
        {
#ifdef OPENMP
#pragma omp parallel for default(none) shared(i, N, C_n, C_n1, u, v, dx, dt, t1, t2) num_threads(t2) schedule(static)
#endif
            for (int j = 0; j < N; j++)
            {
                C_n1[i][j] = get_c_n1_first_order_upwind(C_n, N, i, j, u, v, dx, dx, dt);
            }
        }

        // WRITING MATRIX TO FILE AT DIFFERENT TIMESTAMPS
#ifdef WRITETOFILE
        if (n % 1000 == 0)
        {
            char file_name[] = "First_Order_Upwind/xx.txt";
            int index = n / 1000;

            if (index / 10 == 0)
            {
                file_name[19] = '0';
                file_name[20] = '0' + index;
            }

            else
            {
                file_name[19] = '0' + index / 10;
                file_name[20] = '0' + index % 10;
            }

            FILE *fp = fopen(file_name, "w");
            if (ferror(fp))
            {
                printf("ERROR: Couldn't open file - %s\n", file_name);
            }
            file_write(fp, N, C_n);
            fclose(fp);
        }
#endif
        double **temp = C_n;
        C_n = C_n1;
        C_n1 = temp;
    }
    double end_time = omp_get_wtime();
    printf("TIME TAKEN %f\n", end_time - start_time);

    destroy(C_n, N);
    destroy(C_n1, N);
    return;
}

double get_c_n1_second_order_upwind(double **C_n, int N, int i, int j, double u, double v,
                                    double dx, double dy, double dt)
{
    if (u >= 0 && v >= 0)
        return (
            (
                (-v * ((3 * C_n[i][j] - 4 * C_n[i][mod(j - 1, N)] + C_n[i][mod(j - 2, N)]) / (2 * dy))) +
                (-u * ((3 * C_n[i][j] - 4 * C_n[mod(i - 1, N)][j] + C_n[mod(i - 2, N)][j]) / (2 * dx)))) *
                dt +
            C_n[i][j]);

    else if (u < 0 && v < 0)
        return (
            (
                (-v * ((-3 * C_n[i][j] + 4 * C_n[i][mod(j + 1, N)] - C_n[i][mod(j + 2, N)]) / (2 * dy))) +
                (-u * ((-3 * C_n[i][j] + 4 * C_n[mod(i + 1, N)][j] - C_n[mod(i + 2, N)][j]) / (2 * dx)))) *
                dt +
            C_n[i][j]);

    else if (u < 0)
        return (
            (
                (-v * ((3 * C_n[i][j] - 4 * C_n[i][mod(j - 1, N)] + C_n[i][mod(j - 2, N)]) / (2 * dy))) +
                (-u * ((-3 * C_n[i][j] + 4 * C_n[mod(i + 1, N)][j] - C_n[mod(i + 2, N)][j]) / (2 * dx)))) *
                dt +
            C_n[i][j]);
    else
        return (
            (
                (-v * ((-3 * C_n[i][j] + 4 * C_n[i][mod(j + 1, N)] - C_n[i][mod(j + 2, N)]) / (2 * dy))) +
                (-u * ((3 * C_n[i][j] - 4 * C_n[mod(i - 1, N)][j] + C_n[mod(i - 2, N)][j]) / (2 * dx)))) *
                dt +
            C_n[i][j]);
}

void second_order_upwind(int N, int NT, double L, double T, double u, double v)
{
    double **C_n = (double **)malloc(sizeof(double *) * N);
    double **C_n1 = (double **)malloc(sizeof(double *) * N);

    for (int i = 0; i < N; i++)
    {
        C_n[i] = (double *)malloc(sizeof(double) * N);
        C_n1[i] = (double *)malloc(sizeof(double) * N);
    }

    double dx = L / N, dt = T / NT;

    assert(dt <= dx / pow(2 * (u * u + v * v), 0.5));

    double x_0 = L / 2, y_0 = L / 2;
    double s_x = L / 4, s_y = L / 4;

    set_gaussian_init(N, C_n, dx, dx, x_0, y_0, s_x, s_y);

    double start_time = omp_get_wtime();
    for (int n = 0; n < NT; n++)
    {
#ifdef OPENMP
#pragma omp parallel for default(none) shared(N, C_n, C_n1, u, v, dx, dt, t1, t2) num_threads(t1) schedule(static)
#endif
        for (int i = 0; i < N; i++)
        {
#ifdef OPENMP
#pragma omp parallel for default(none) shared(i, N, C_n, C_n1, u, v, dx, dt, t1, t2) num_threads(t2) schedule(static)
#endif
            for (int j = 0; j < N; j++)
            {
                C_n1[i][j] = get_c_n1_second_order_upwind(C_n, N, i, j, u, v, dx, dx, dt);
            }
        }

        // WRITING MATRIX TO FILE AT DIFFERENT TIMESTAMPS
#ifdef WRITETOFILE
        if (n % 1000 == 0)
        {
            char a[] = "Sec_Order_Upwind/xx.txt";
            int index = n / 1000;
            if (index / 10 == 0)
            {
                a[17] = '0';
                a[18] = '0' + index;
            }
            else
            {
                a[17] = '0' + index / 10;
                a[18] = '0' + index % 10;
            }
            FILE *fp = fopen(a, "w");
            file_write(fp, N, C_n);
            fclose(fp);
        }
#endif
        double **temp = C_n;
        C_n = C_n1;
        C_n1 = temp;
    }
    double end_time = omp_get_wtime();
    printf("TIME TAKEN %f\n", end_time - start_time);

    destroy(C_n, N);
    destroy(C_n1, N);
    return;
}
