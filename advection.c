#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>

int mod (int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

void destroy (double ** arr, int N) {
    for (int i = 0; i < N; i++) {
        free(arr[i]);
        arr[i] = NULL;
    }
    free(arr);
    arr = NULL;
    return;
}

void file_write(FILE * fp, int N, double ** arr) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(fp,"%e, ",arr[i][j]);
        }
        fprintf(fp,"\n");
    }
}

void advection (int N, int NT, double L, double T, double u, double v) {
    double ** C_n = (double **) malloc(sizeof(double *) * N);
    double ** C_n1 = (double **) malloc(sizeof(double*) * N);

    for (int i = 0; i < N; i++) {
        C_n[i] = (double *) malloc(sizeof(double) * N);
        C_n1[i] = (double *) malloc(sizeof(double) * N);
    }

    double dx = L/N, dt = T/NT;

    assert(dt <= dx/pow(2*(u*u + v*v), 0.5));

    double x_0 = L/2, y_0 = L/2;
    double s_x = L/4, s_y = L/4;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C_n[i][j] = exp(
                -(
                    pow(i * dx - x_0, 2)/(2*pow(s_x, 2)) +
                    pow(j * dx - y_0, 2)/(2*pow(s_y, 2))
                )
            );
    
    FILE * fp1 = fopen("timestamp_init.txt", "w"), 
         * fp2 = fopen("timestamp_mid.txt", "w"), 
         * fp3 = fopen("timestamp_end.txt", "w");
    if (ferror(fp1) || ferror(fp2) || ferror(fp3)) {
        printf("ERROR: Couldn't open file(s)\n");
    }

    for (int n = 0; n < NT; n++) {
        if (n == 0)
            file_write(fp1, N, C_n);

        if (n == NT/2)
            file_write(fp2, N, C_n);

        if (n == NT -1)
            file_write(fp3, N, C_n);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double C_n_ip_j = C_n[mod(i-1, N)][j], 
                       C_n_in_j = C_n[mod(i+1, N)][j], 
                       C_n_i_jp = C_n[i][mod(j-1, N)],
                       C_n_i_jn = C_n[i][mod(j+1, N)];
                
                C_n1[i][j] = 
                    (
                        (C_n_ip_j + C_n_in_j + C_n_i_jp + C_n_i_jn)/4 
                    )-
                    (
                        (u * (C_n_in_j - C_n_ip_j) + v * (C_n_i_jn - C_n_i_jp)) * (dt / (2 * dx))
                    );
            }

        //set C_n = C_n1
        double ** temp = C_n;
        C_n = C_n1;
        C_n1 = temp;
    }

    fclose(fp1);
    fclose(fp2);
    fclose(fp3);

    destroy(C_n, N);
    destroy(C_n1, N);
    return;
}

int main (int argc, char ** argv) {
    if (argc == 7) {
        int N = atoi(argv[1]),
            NT = atoi(argv[2]);
        
        double L = atof(argv[3]), 
               T = atof(argv[4]), 
               u = atof(argv[5]), 
               v = atof(argv[6]);
        printf("N - Matrix Dimension: %d\n", N);
        printf("NT - Number of timesteps: %d\n", NT);
        printf("L - Physical Cartesian Domain Length: %.10lf\n", L);
        printf("T - Total Physical Timespan: %.10lf\n", T);
        printf("u - X velocity Scalar: %.10lf\n", u);
        printf("v - Y velocity Scalar: %.10lf\n", v);
        printf("ESTIMATE FOR MEMORY USAGE: %lu bytes\n", sizeof(double) * N * N * 2);
        advection(N, NT, L, T, u, v);
        return EXIT_SUCCESS;
    }

    else {
        printf("ERROR: Too few/many arguments\n");
        return EXIT_FAILURE;
    }
}
