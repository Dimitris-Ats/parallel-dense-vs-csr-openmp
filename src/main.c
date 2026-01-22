#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include <omp.h>
#include <unistd.h>

int thread_count;

double my_elapsed;
struct timespec my_start, my_finish;

void generate_main_array(int **main_array, int n, long long num_zeros){

    clock_gettime(CLOCK_REALTIME, &my_start );

    long long num_nonzeros = (n * n) - num_zeros;
    int i, j;

    for(int k = 0; k < num_nonzeros; k++){
        i = rand() % n;
        j = rand() % n;
        while(main_array[i][j] != 0){
            
            if(j < n - 1){
                j++;
            }

            else if(i < n - 1){
                j = 0;
                i ++;
            }

            else{
                j = 0;
                i = 0;
            }

        }
        main_array[i][j] = rand() + 1;
        main_array[i][j]*= (rand() % 2 ? 1 : -1);
            
    }    

    clock_gettime( CLOCK_REALTIME, &my_finish );    //finish time of generating main array

    my_elapsed = (my_finish.tv_sec - my_start.tv_sec) + (my_finish.tv_nsec - my_start.tv_nsec) / 1e9;

    printf("Creation of main array time = %lf seconds\n", my_elapsed);

}

void generate_vector(int n, double *dian){
    for(int i = 0; i < n; i++){
        dian[i] = rand();
        dian[i]*= (rand() % 2 ? 1 : -1);
    }
}

void generate_csr_arrays(int n, int **main_array, int *v, int *row_index, int *col_index, int thread_count){

    
    int *row_nonz;

    row_nonz = calloc(n, sizeof(int));

    row_index[0] = 0;

    clock_gettime(CLOCK_REALTIME, &my_start );             //start  time of generating csr

    #pragma omp parallel for num_threads(thread_count)

    for(int i = 0; i < n; i++){
        int counter = 0;                        //how many zeros per row
        for(int j = 0; j < n; j++){
            if(main_array[i][j] != 0){
                counter ++;
            }
        }
        row_nonz[i] = counter;
    }

    for(int i = 0; i < n; i++){
        row_index[i+1] = row_index[i] + row_nonz[i];        //fill row_index array
    }



    #pragma omp parallel for num_threads(thread_count)

    for(int i = 0; i < n; i++){
        int pos = row_index[i];
        for(int j = 0; j < n; j++){
            if(main_array[i][j] != 0){
                v[pos] = main_array[i][j];
                col_index[pos] = j;
                pos++;
            }
        }
    }


    clock_gettime( CLOCK_REALTIME, &my_finish );    //finish time of generating csr

    my_elapsed = (my_finish.tv_sec - my_start.tv_sec) + (my_finish.tv_nsec - my_start.tv_nsec) / 1e9;

    printf("parallel creation of csr time = %lf seconds\n", my_elapsed);

    free(row_nonz);

}

double* dense_mult(int n, int **main_array, double *dian, int thread_count){

    double *result = calloc(n, sizeof(double));
    if (!result) {
        fprintf(stderr, "calloc failed\n"); 
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for num_threads(thread_count)

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){                         
            result[i] += main_array[i][j] * dian[j];
        }
    }
    return result;

}

double* csr_mult(int n, int *v, int * row_index, int * col_index, double * dian, int num_nonzeros, int thread_count){
    
    double *result = calloc(n, sizeof(double));
    if (!result) {
        fprintf(stderr, "calloc failed\n"); 
        exit(EXIT_FAILURE);
    }

    // No reduction needed: each thread updates a distinct row
    #pragma omp parallel for num_threads(thread_count) 
    for (int i = 0; i < n; i++) {
        for (int k = row_index[i]; k < row_index[i + 1]; k++) {
            result[i] += v[k] * dian[col_index[k]];
        }
    }

    return result;
}



int compare_results(int n, double *dian_a, double * dian_b){

    for(int i = 0; i < n; i++){
        if (fabs(dian_a[i] - dian_b[i]) > 1e-6){
            return 0;
        }
    }
    return 1;
}


int main(int argc, char *argv[]){

    int n, pct_zeros, mult_count;
    int **main_array, *v, *col_index, *row_index;
    double *dian;
    long long num_zeros, num_nonzeros;
    double *dian_a, *dian_b, *temp;


    if (argc != 5) {
        fprintf(stderr, "Usage: %s <n> <pct_zeros> <mult_count> <threads>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    n = strtol(argv[1], NULL, 10);                      //array dimensions

    pct_zeros = strtol(argv[2], NULL, 10);              //percentage of zeros
    
    mult_count = strtol(argv[3], NULL, 10);               //multiplication count

    thread_count = strtol(argv[4], NULL, 10);  

    num_zeros = (long long) llround((double)n * n * pct_zeros / 100.0);                //find number of zeros needed
    
    main_array = malloc(n * sizeof(int*));              

    for(int i = 0; i < n; i++){
        main_array[i] = (int*)calloc(n , sizeof(int));         //allocate memory and intitialize main_array
    }


    generate_main_array(main_array, n, num_zeros);                


    dian = calloc(n, sizeof(double));

    generate_vector(n, dian);

    num_nonzeros = (n * n) - num_zeros;

    v = malloc(num_nonzeros * sizeof(int));
    col_index = malloc(num_nonzeros * sizeof(int));
    row_index = malloc((n + 1) * sizeof(int));

    generate_csr_arrays(n, main_array, v, row_index, col_index, thread_count);


    clock_gettime( CLOCK_REALTIME, &my_start );     //start time of Dense mult

    dian_a = dense_mult(n, main_array, (double*)dian, thread_count);
    for(int i = 1; i < mult_count; i++){
        temp = dense_mult(n, main_array, dian_a, thread_count);
        free(dian_a);
        dian_a = temp;
    }

    clock_gettime( CLOCK_REALTIME, &my_finish );    //finish time of dense mult

    my_elapsed = (my_finish.tv_sec - my_start.tv_sec) + (my_finish.tv_nsec - my_start.tv_nsec) / 1e9;

    printf("Dense multiplication time = %lf seconds\n", my_elapsed);


    clock_gettime( CLOCK_REALTIME, &my_start );     //start time of CSR mult

    dian_b = csr_mult(n, v, row_index, col_index, (double*)dian, num_nonzeros, thread_count);
    for(int i = 1; i < mult_count; i++){
        temp = csr_mult(n, v, row_index, col_index, dian_b, num_nonzeros, thread_count);    //multiply csr mult_count times
        free(dian_b);
        dian_b = temp;
    }
    clock_gettime( CLOCK_REALTIME, &my_finish );    //finish time of CSR mult

    my_elapsed = (my_finish.tv_sec - my_start.tv_sec) + (my_finish.tv_nsec - my_start.tv_nsec) / 1e9;

    printf("CSR multiplication time = %lf seconds\n", my_elapsed);

    //compare results of both algorithms
    if(compare_results(n, dian_a, dian_b)){
        printf("Success: Both algorithms gave the same result!\n");
    }

    else{
        printf("Failure: Algorithms gave different results\n");
    }



    //free arrays
    free(dian);

    free(dian_a);
    free(dian_b);

    free(col_index);
    free(row_index);
    free(v);
    

    for(int i = 0; i < n; i++){
        free(main_array[i]);
    }
    free(main_array);

    return 0;
}
