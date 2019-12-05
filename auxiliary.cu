/*
 * Author: Jacob Perricone
 * A
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cublas_v2.h>
#include "fstream"
#include "headers.h"
#include <errno.h>


/* Define constants max threads per block in each dimension*/
#define MAX_THREADS_PER_BLOCK_X (1024)
#define MAX_THREADS_PER_BLOCK_Y (1024)
#define MAX_THREADS_PER_BLOCK_Z (64)
#define MAX_BLOCKS (65535)
#define MAX_BLOCKS (65535)
#define GLOBAL_MEM_SIZE (4232052736)
#define CONSTANT_MEM_SIZE (65536)
#define SHARED_MEM_SIZE (49152)
#define THREADS_PER_BLOCK_Y 1024/4
#define THREADS_PER_BLOCK_X NUM_DEFECTS
#define TILE_WIDTH 16


cublasHandle_t handle;





void runEM_CPU(float *TFDF, int const num_complaints,
               int const vocab_size, float *company_vec,
               int const num_companies, float *issue_vec,
               int const num_issues, float *product_vec,
               int const num_products, int const num_defects,
               float *defect_priors_cpu, float *defect_posteriors_cpu,
               float *company_posteriors_cpu, float *issue_posteriors_cpu,
               float *product_posteriors_cpu, float *word_posteriors_cpu,
               const float tol, const int maxIter, const char * log_file_name,
               const char *results_file_name){


    double delta_likelihood = INFINITY;
    double old_likelihood = -INFINITY;
    double new_likelihood = 0;
    float *x_i_posteriors_cpu, *max_elements, *z, *TFDF_SUM, *d_posterior_sum;
    double tmp;
    double max;
    double denominator;
    double numerator;
    double epsilon = 1.e-6;
    int iter = 0;
    float time_e, time_m, total_estep_time, total_mstep_time;
    clock_t begin, end;


    x_i_posteriors_cpu = (float *) malloc(sizeof(float) * num_complaints * num_defects);
    z = (float *) malloc(sizeof(float) * num_complaints * num_defects);
    max_elements = (float *) malloc(sizeof(float) * num_complaints);
    TFDF_SUM = (float *) malloc(sizeof(float)*num_complaints);
    d_posterior_sum = (float *) malloc(sizeof(float)*num_defects);
    tmp = 0;

    for (int j =0; j < num_complaints; j++){
        for (int i =0; i < vocab_size; i++){
            tmp += TFDF[INDX(j, i, num_complaints)];
        }
        TFDF_SUM[j] = tmp;
    }


    FILE *s ;
    s = fopen(log_file_name, "w");
    fprintf(s, "-----------CPU: Beggining Expectation Maximization Routine on %d complaints and %d words---------\n",
            num_complaints, vocab_size);


    while (delta_likelihood > (float) .00001 || iter < 10) {
        iter++;
        memset(x_i_posteriors_cpu, 0, sizeof(float) * num_complaints * num_defects);
        memset(z, 0, sizeof(float) * num_complaints * num_defects);



        denominator = 0.0;
        numerator = 0.0;
        new_likelihood = 0.0;
        begin = clock();

        for (int i = 0; i < num_complaints; i++) {
            for (int j = 0; j < num_defects; j++) {
                x_i_posteriors_cpu[INDX(j, i, num_defects)] += logf(
                        issue_posteriors_cpu[INDX(j, (int)issue_vec[INDX(i, 0, num_complaints)], num_defects)]);

                x_i_posteriors_cpu[INDX(j, i, num_defects)] += logf(
                        product_posteriors_cpu[INDX(j, (int) product_vec[INDX(i, 0, num_complaints)], num_defects)]);

                x_i_posteriors_cpu[INDX(j, i, num_defects)] += logf(
                        company_posteriors_cpu[INDX(j, (int) company_vec[INDX(i, 0, num_complaints)], num_defects)]);

                tmp = 0;
                for (int k = 0; k < vocab_size; k++) {
                    tmp += TFDF[INDX(i, k, num_complaints)] * logf(word_posteriors_cpu[INDX(j, i, num_defects)]);
                }
                x_i_posteriors_cpu[INDX(j, i, num_defects)] += tmp;
                z[INDX(j, i, num_defects)] =
                        logf(defect_priors_cpu[INDX(j, 0, num_defects)]) + x_i_posteriors_cpu[INDX(j, i, num_defects)];

            }

            max = -INFINITY;
            for (int k = 0; k < num_defects; k++) {
                if (z[INDX(k, i, num_defects)] > max)
                    max = z[INDX(k, i, num_defects)];
            }

            max_elements[i] = (float) max;
            new_likelihood += max_elements[i];

            tmp = 0;
            for (int k = 0; k < num_defects; k++) {
                numerator = 0;
                denominator = 0;
                if (z[INDX(k, i, num_defects)] - max_elements[i] < -11) {
                    defect_posteriors_cpu[INDX(k, i, num_defects)] = 0.0;
                } else {
                    numerator = expf(z[INDX(k, i, num_defects)] - max_elements[i]);
                    tmp += numerator;
                    denominator += numerator;
                    for (int l = 0; l < num_defects && l != k; l++){
                        if (z[INDX(l, i, num_defects)] - max_elements[i] > -11) {
                            denominator += expf(z[INDX(l, i, num_defects)] - max_elements[i]);
                        }
                    }
                    defect_posteriors_cpu[INDX(k, i, num_defects)] = numerator / denominator;
                }
            }
            new_likelihood += logf(tmp);


        }
        end = clock();
        time_e =(float)(end - begin) / CLOCKS_PER_SEC;
        total_estep_time += time_e;
        fprintf(s, "---------Total time For E_STEP on CPU is %f sec ---------\n", time_e);

        delta_likelihood = fabsf(old_likelihood - new_likelihood);
        fprintf(s,"(OLD LIKELIHOOD = %f, UPDATED LIKELIHOOD = %f , Change in Likelihood =%f)\n",
                old_likelihood,new_likelihood, delta_likelihood);

        printf("Change in Likelihood is %f:\n", (new_likelihood-old_likelihood));

        old_likelihood= new_likelihood;


        /* M STEP*/
        memset(d_posterior_sum, 0, sizeof(float)*num_defects);
        fprintf(s, "--------------DOING M-STEP WITH CPU---------------------- \n");
        begin = clock();
        for (int j=0; j < num_defects; j++){

            for(int i=0; i < vocab_size; i++){
                numerator = 0;
                denominator = 0;
                for (int k= 0; k < num_complaints; k++){

                    numerator += defect_posteriors_cpu[INDX(j, k, num_defects)]*TFDF[INDX(k, i, num_complaints)];
                    denominator += defect_posteriors_cpu[INDX(j, k, num_defects)]*TFDF_SUM[k];

                }
                word_posteriors_cpu[INDX(j,i, num_defects)] = (1 + numerator )/(vocab_size + denominator);

            }
            for(int i=0; i < num_companies; i++){
                numerator = 0;
                denominator = 0;
                for (int k= 0; k < num_complaints; k++){
                    if ((int)company_vec[k] == i) {
                        numerator += defect_posteriors_cpu[INDX(j, k, num_defects)];
                    }
                    denominator += defect_posteriors_cpu[INDX(j, k, num_defects)];


                }
                company_posteriors_cpu[INDX(j,i, num_defects)] = (1 + numerator)/ (num_companies + denominator);

            }

            for (int k = 0; k < num_complaints; k++){
                d_posterior_sum[j] += defect_posteriors_cpu[INDX(j, k, num_defects)];
            }
            for(int i=0; i < num_products; i++){
                numerator = 0;
                denominator = 0;
                for (int k= 0; k < num_complaints; k++){
                    if ((int)company_vec[k] == i) {
                        numerator += defect_posteriors_cpu[INDX(j, k, num_defects)];
                    }

                    denominator += defect_posteriors_cpu[INDX(j, k, num_defects)];

                }
                product_posteriors_cpu[INDX(j,i, num_defects)] = (1 + numerator)/ (num_products + denominator);

            }

            for(int i=0; i < num_issues; i++){
                numerator = 0;
                denominator = 0;
                for (int k= 0; k < num_complaints; k++){
                    if ((int)issue_vec[k] == i) {
                        numerator += defect_posteriors_cpu[INDX(j, k, num_defects)];
                    }
                    denominator += defect_posteriors_cpu[INDX(j, k, num_defects)];

                }
                product_posteriors_cpu[INDX(j,i, num_defects)] = (1 + numerator)/ (num_issues + denominator);

            }



            if (d_posterior_sum[j] < epsilon){
                printf("YOO %f \n", d_posterior_sum[j]);
                defect_priors_cpu[j] = epsilon;
            }else{
                defect_priors_cpu[j] =  d_posterior_sum[j]/num_complaints;
            }

        }
        end = clock();
        time_m =(float)(end - begin) / CLOCKS_PER_SEC;
        total_mstep_time += time_m;
        fprintf(s, "---------Total time For M_STEP on CPU is %f sec ---------\n", time_e);




    }

    fprintf(s, "Total time till convergece is %f sec | %d iterations \n", total_mstep_time + total_estep_time, iter);
    fprintf(s, "Average Time of eStep is %f sec: \n", total_estep_time/iter);
    fprintf(s, "Average Time of MStep is %f sec: \n", total_mstep_time/iter);

    fprintf(s, "Finally Likelihood %f\n", old_likelihood);
    fprintf(s, "Change in likelihood %f\n", delta_likelihood);

    fclose(s);
    FILE *f = fopen(results_file_name, "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }


    for (int i=0; i < num_companies; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"COMPANY, %d, DEFECT, %d, POSTERIOR, %f \n", i, j,company_posteriors_cpu[INDX(j, i, num_defects)] );
        }
    }

    for (int i=0; i < num_issues; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"ISSUE, %d, DEFECT,%d, POSTERIOR, %f \n", i, j,issue_posteriors_cpu[INDX(j, i, num_defects)] );
        }
    }

    for (int i=0; i < vocab_size; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"WORD, %d,  DEFECT %d, POSTERIOR, %f \n", i, j,word_posteriors_cpu[INDX(j, i, num_defects)] );
        }
    }

    for (int i=0; i < num_complaints; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"Complaint, %d, DEFECT, %d, POSTERIOR: %f \n", i, j,defect_posteriors_cpu[INDX(j, i, num_defects)] );
        }
    }

    for (int j=0; j < num_defects; j++){
        fprintf(f,"DEFECT, %d, , Prior : %f \n", j,defect_priors_cpu[j] );
    }

    fclose(f);

    free(x_i_posteriors_cpu);
    free(z);
    free(max_elements);
    free(TFDF_SUM);
    free(d_posterior_sum);


}


__global__ void eStep2(int const num_complaints, int const vocab_size,
                       int const num_defects, float *d_defect_priors,
                       float *d_defect_posteriors, float *d_company_posteriors,
                       float *d_issue_posteriors, float *d_product_posteriors,
                       float *d_word_posteriors, float *d_TFDF,
                       float *d_company_vec, float *d_issue_vec,
                       float *d_product_vec, float *d_likelihood
) {


    int d_row = threadIdx.x;
    int d_offset = blockIdx.x * blockDim.y;
    int d_index = threadIdx.y;
    int d_sample = d_offset + d_index;

    __shared__ double x_posterior[NUM_DEFECTS][THREADS_PER_BLOCK_Y];
    __shared__ double z[NUM_DEFECTS][THREADS_PER_BLOCK_Y];
    __shared__ double max_elem[THREADS_PER_BLOCK_Y];
    __shared__ float block_likelihood[THREADS_PER_BLOCK_Y];
    __shared__ float c_post[NUM_DEFECTS][NUM_COMPANIES_SMALL];
    __shared__ float i_post[NUM_DEFECTS][NUM_ISSUES_SMALL];
    __shared__ float p_post[NUM_DEFECTS][NUM_PRODUCTS_SMALL];
    __shared__ int p_vec[THREADS_PER_BLOCK_Y];
    __shared__ int c_vec[THREADS_PER_BLOCK_Y];
    __shared__ int i_vec[THREADS_PER_BLOCK_Y];



    if (d_sample < num_complaints) {
        /* COPY THIS SHIT TO SHARED_MEM */
        /* Keep track of p(x_i | d_j) */

        c_vec[d_index] = (int) d_company_vec[INDX(d_sample, 1, 1)];
        p_vec[d_index] = (int) d_product_vec[INDX(d_sample, 1, 1)];
        i_vec[d_index] = (int) d_issue_vec[INDX(d_sample, 1, 1)];

        __syncthreads();

        if (d_index < NUM_PRODUCTS_SMALL) {
            c_post[d_row][d_index] = d_company_posteriors[INDX(d_row, d_index, num_defects)];
            i_post[d_row][d_index] = d_issue_posteriors[INDX(d_row, d_index, num_defects)];
            p_post[d_row][d_index] = d_product_posteriors[INDX(d_row, d_index, num_defects)];
        }else if (d_index < NUM_COMPANIES_SMALL ) {
            i_post[d_row][d_index] = d_issue_posteriors[INDX(d_row, d_index, num_defects)];
            c_post[d_row][d_index] = d_company_posteriors[INDX(d_row, d_index, num_defects)];
        }else if (d_index < NUM_ISSUES_SMALL){
            i_post[d_row][d_index] = d_issue_posteriors[INDX(d_row, d_index, num_defects)];
        }


        __syncthreads();

        x_posterior[d_row][d_index] = logf(c_post[d_row][c_vec[d_index]])
                                      + logf(i_post[d_row][i_vec[d_index]]) + logf(p_post[d_row][p_vec[d_index]]);

        float sum = 0;
        for (int i = 0; i < vocab_size; i++) {
            sum += d_TFDF[INDX(d_sample, i, num_complaints)] * logf(d_word_posteriors[INDX(d_row, i, num_defects)]);
        }
        x_posterior[d_row][d_index] += sum;

        /* Apply smoothing operations */
        z[d_row][d_index] = logf(d_defect_priors[d_row]) + x_posterior[d_row][d_index];

        __syncthreads();


        double max = -INFINITY;

        for (int i = 0; i < num_defects; i++) {
            if (z[i][d_index] > max) {
                max = z[i][d_index];
            }
        }

        block_likelihood[d_index] = 0;
        __syncthreads();
        max_elem[d_index] = max;
       //printf( "DEFECT %d %d %f %f %f %f \n", d_row, d_sample, z[0][d_index], z[1][d_index], z[2][d_index],max_elem[d_index] );


        double denom = 0.0;


        if (z[d_row][d_index] - max_elem[d_index] > -11) {
            block_likelihood[d_index] += expf(z[d_row][d_index] - max_elem[d_index]);
            for (int i = 0; i < num_defects; i++) {
                if (z[i][d_index] - max_elem[d_index] > -11) {
                    denom += expf(z[i][d_index] - max_elem[d_index]);
                }
            }
            d_defect_posteriors[INDX(d_row, d_sample, num_defects)] =  expf(z[d_row][d_index] - max_elem[d_index]) / denom;

        } else {
            d_defect_posteriors[INDX(d_row, d_sample, num_defects)] = 0.0;
        }

        __syncthreads();

        if (threadIdx.y==0 && threadIdx.x == 0) {
            for (int i = 0; i < THREADS_PER_BLOCK_Y && (d_offset + i) < num_complaints; i++) {
                d_likelihood[blockIdx.x] += max_elem[i] + logf(block_likelihood[i]);
            }
        }

    }

}


/* Do it without Shared Mem For now */
__global__ void eStep(int const num_complaints,
                      int const vocab_size,
                      int const num_defects,
                      float *d_defect_priors,
                      float *d_defect_posteriors,
                      float *d_company_posteriors,
                      float *d_issue_posteriors,
                      float *d_product_posteriors,
                      float *d_word_posteriors,
                      float *d_TFDF,
                      float *d_company_vec,
                      float *d_issue_vec,
                      float *d_product_vec,
                      float *d_likelihood
) {


    int d_row = threadIdx.x;

    int d_offset = blockIdx.x * blockDim.y;
    int d_index = threadIdx.y;
    int d_sample = d_offset + d_index;



    __shared__ double x_posterior[NUM_DEFECTS][THREADS_PER_BLOCK_Y];
    __shared__ double z[NUM_DEFECTS][THREADS_PER_BLOCK_Y];
    __shared__ double max_elem[THREADS_PER_BLOCK_Y];
    __shared__ float block_likelihood[THREADS_PER_BLOCK_Y];

    if (d_sample < NUM_COMPLAINTS_SMALL) {
        /* COPY THIS TO SHARED_MEM */
        /* Keep track of p(x_i | d_j) */

        x_posterior[d_row][d_index] =
                logf(d_company_posteriors[INDX(d_row, (int) d_company_vec[INDX(d_sample, 1, 1)], num_defects)]);

        x_posterior[d_row][d_index] += logf(
                d_issue_posteriors[INDX(d_row, (int) d_issue_vec[INDX(d_sample, 1, 1)], num_defects)]);
        x_posterior[d_row][d_index] += logf(
                d_product_posteriors[INDX(d_row, (int) d_product_vec[INDX(d_sample, 1, 1)], num_defects)]);

        double sum = 0;


        for (int i = 0; i < vocab_size; i++) {

            sum += d_TFDF[INDX(d_sample, i, num_complaints)] * logf(d_word_posteriors[INDX(d_row, i, num_defects)]);
        }
        x_posterior[d_row][d_index] += sum;


        /* Apply smoothing operations */

        z[d_row][d_index] = logf(d_defect_priors[d_row]) + x_posterior[d_row][d_index];

        __syncthreads();

        double max = -INFINITY;
        for (int i = 0; i < num_defects; i++) {
            if (z[i][d_index] > max) {
                max = z[i][d_index];
            }
        }

        block_likelihood[d_index] = 0;
        __syncthreads();
        max_elem[d_index] = max;


        double denom = 0.0;


        if (z[d_row][d_index] - max_elem[d_index] > -11) {
            block_likelihood[d_index] += expf(z[d_row][d_index] - max_elem[d_index]);
            for (int i = 0; i < num_defects; i++) {
                if (z[i][d_index] - max_elem[d_index] > -11) {
                    denom += expf(z[i][d_index] - max_elem[d_index]);
                }
            }
            d_defect_posteriors[INDX(d_row, d_sample, num_defects)] =  expf(z[d_row][d_index] - max_elem[d_index]) / denom;

        } else {
            d_defect_posteriors[INDX(d_row, d_sample, num_defects)] = 0.0;
        }

        __syncthreads();

        if (threadIdx.y==0 && threadIdx.x == 0) {
            for (int i = 0; i < THREADS_PER_BLOCK_Y && (d_offset + i) < num_complaints; i++) {
                d_likelihood[blockIdx.x] += max_elem[i] + logf(block_likelihood[i]);
            }
        }


    }

}



/* Find the sum of the colums of a matrix, I am going to launch a block size equal to the number
 * of rows of matrix in, but this is not always feasible. Esnure that num_rows == blockDim.y. */
__global__ void reduce_columns(const int num_rows, const int num_columns, const float * in, float * out){



    extern __shared__ float sArray[];


    int globalIndex = blockIdx.y * blockDim.y + threadIdx.y;




/* zero out the smem array */
    sArray[threadIdx.y] = 0.0;

/* Stride over the array
 */
    float tmp = 0;

    for( int i = globalIndex; i < num_columns; i += blockDim.y)
    {
        tmp += in[INDX(blockIdx.x,i,num_rows)];
    } /* end for */
    sArray[threadIdx.y] = tmp;
    __syncthreads();

/* do the final reduction in SMEM */
    for( int i = blockDim.y/2; i > 0; i = i / 2 )
    {
        if( threadIdx.y < i )
        {
            sArray[threadIdx.y] += sArray[threadIdx.y + i];
//            sArray[threadIdx.x] += sArray[threadIdx.x + i];
        } /* end if */
        __syncthreads();
    } /* end for */

/* thread0 of each threadblock writes the result to global memory */
    if( threadIdx.y == 0 ){
//        printf("BLOCK X has value, %f", (float) sArray[0]);
        out[blockIdx.x] = sArray[0];
    }



    return;


}


/* Tile Matrix Multiplication on GPU
 *
 * Params:
 * a_rows = number of d_A
 * a_columns = number of columns of d_A
 * b_rows = number of rows in d_B
 * b_columns = number of columns in d_B
 * d_C = matrix to save into
 * c_rows = rows of matrix
 * */

__global__ void mat_mul(const int a_rows, const int a_columns,
                        const int b_rows, const int b_columns,
                        float * d_A, float * d_B, float *  d_C,
                        const int c_rows, const int c_columns) {

/* setup some constants for later use */

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int iby = blockIdx.y * TILE_WIDTH;
    const int ibx = blockIdx.x * TILE_WIDTH;
    const int row = iby + ty;
    const int col = ibx + tx;

/* shared memory arrays for A and B */

    __shared__ double as[TILE_WIDTH][TILE_WIDTH];
    __shared__ double bs[TILE_WIDTH][TILE_WIDTH];

/* space for C to be held in registers */
    float value = 0;
    int tmp_col;
    int tmp_row;


    for (int i = 0; i < ceil(a_columns/(float)TILE_WIDTH); i++) {
        tmp_col = i * TILE_WIDTH + tx;
        if (tmp_col < a_columns && row < a_rows) {
            as[ty][tx] = d_A[tmp_col * a_rows + row];
        } else {
            as[ty][tx] = 0.0;
        }
        tmp_row = i * TILE_WIDTH + ty;
        if (tmp_row < b_rows && col < b_columns) {
            bs[ty][tx] = d_B[col * b_rows + tmp_row];
        } else {
            bs[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++) {
            value += as[threadIdx.y][j] * bs[j][threadIdx.x];
        }
        __syncthreads();

    }

    if (row < c_rows && col < c_columns) {
        int row_map = blockIdx.y * blockDim.y + threadIdx.y;
        int col_map = blockIdx.x * blockDim.x + threadIdx.x;
        d_C[INDX(row_map, col_map, c_rows)] = value;
    }


}


/* Update the issues posteriors by doing matrix multiplication and division in one step
 * same params as mat_mul with the aditional float * division, which should be the number of rows as d_C
 * the  constants applies the operation
 * (numerator_constant + x_{i,j) / (denominator_constant + divisor_i)
 */

__global__ void update_entities(const int a_rows, const int a_columns,
                                const int b_rows, const int b_columns,
                                float * d_A, float * d_B,
                                float *  d_C, const int c_rows,
                                const int c_columns, float * divisor,
                                float numerator_constant, float denominator_constant)
{

/* setup some constants for later use */

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int iby = blockIdx.y * TILE_WIDTH;
    const int ibx = blockIdx.x * TILE_WIDTH;
    const int row = iby + ty;
    const int col = ibx + tx;

/* shared memory arrays for A and B */

    __shared__ double as[TILE_WIDTH][TILE_WIDTH];
    __shared__ double bs[TILE_WIDTH][TILE_WIDTH];

/* space for C to be held in registers */
    float value = 0;

    int tmp_col;
    int tmp_row;
    for (int i = 0; i < ceil(a_columns/(float)TILE_WIDTH); i++) {
        tmp_col = i * TILE_WIDTH + tx;
        if (tmp_col < a_columns && row < a_rows) {
            as[ty][tx] = d_A[tmp_col * a_rows + row];
        } else {
            as[ty][tx] = 0.0;
        }
        tmp_row = i * TILE_WIDTH + ty;
        if (tmp_row < b_rows && col < b_columns) {
            bs[ty][tx] = d_B[col * b_rows + tmp_row];
        } else {
            bs[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++) {
            value += as[threadIdx.y][j] * bs[j][threadIdx.x];
        }
        __syncthreads();

    }

    if (row < c_rows && col < c_columns) {
        int row_map = blockIdx.y * blockDim.y + threadIdx.y;
        int col_map = blockIdx.x * blockDim.x + threadIdx.x;
        d_C[INDX(row_map, col_map, c_rows)] = (numerator_constant + value) / (divisor[row_map] + denominator_constant);
    }
    /* c

     */

}



/* Sums across columns of a matrix using cublas matrix operations */
void cublas_column_reduce( float * d_A, const int num_rows, const int num_columns, float * d_y){


    const float alpha = 1.0;
    float beta = 0.0;

    /* Make a vector of ones  for multiplication */
    float *h_x, *d_x;
    h_x = (float *) malloc(sizeof(float) * num_columns);
    for (int i = 0; i < num_columns; i++)
        h_x[INDX(i,0,num_columns)] = (float) 1.0;

    /* Copy to device memory */

    checkCUDA(cudaMalloc((void **) &d_x, sizeof(float) * num_columns));
    checkCUDA(cudaMemcpy(d_x, h_x, sizeof(float) * num_columns, cudaMemcpyHostToDevice));



    /* Start Timers */

    checkCUBLAS(cublasCreate(&handle));
    checkCUBLAS(cublasSgemv(handle, CUBLAS_OP_N,
                            num_rows, num_columns,
                            &alpha,
                            d_A, num_rows,
                            d_x, 1.0,
                            &beta,
                            d_y, 1.0));

    /* End Timers */

    /*Free the ones vector*/
    checkCUDA(cudaFree(d_x));
    free(h_x);
    /* Host copy isn't needed */


}





/* Fills randomly generated data between 0-1 into the dynamic float array data, of size: (rows, cols)
*
* Params:
* float * data: array to be filled
* int rows: number of rows in array
* int cols: number of columns to be filled
* int ld: number of rows per column
*/
void RandomInit(float *data, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[INDX(i, j, ld)] = (static_cast <float> (rand()) / static_cast <float>  (RAND_MAX)) + 1.e-2;

        }

    }
}




/* See header file for description*/
void runEM(float *TFDF, int const num_complaints,
           int const vocab_size, float *company_vec,
           int const num_companies, float *issue_vec,
           int const num_issues, float *product_vec,
           int const num_products, int const num_defects,
           float *defect_priors, float *defect_posteriors,
           float *company_posteriors, float *issue_posteriors,
           float *product_posteriors, float *word_posteriors,
           const float tol, const int maxIter, const char * log_file_name,
           const char *results_file_name
) {


    /* checkCUBLAS( cublasCreate( &cublasHandle ) );*/

    printf("------------------------------------ \n");
    printf("COPYING HOST DATA TO DEVICE MEMORY\n");


    int thread_size;
    float elapsedTime;
    float old_likelihood,  delta_likelihood;

    float *d_defect_priors, *d_defect_posteriors,
            *d_company_posteriors, *d_product_posteriors,
            *d_issue_posteriors,*d_word_posteriors;


    /* ALLOCATE AND COPY FOR PRIORS AND POSTERIORS */
    checkCUDA(cudaMalloc((void **) &d_defect_priors, sizeof(float) * num_defects));
    checkCUDA(cudaMemcpy(d_defect_priors, defect_priors, sizeof(float) * num_defects,
                         cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_defect_posteriors, sizeof(float) * num_defects * num_complaints));
    checkCUDA(cudaMemcpy(d_defect_posteriors, defect_posteriors, sizeof(float) * num_defects * num_complaints,
                         cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_company_posteriors, sizeof(float) * num_defects * num_companies));
    checkCUDA(cudaMemcpy(d_company_posteriors, company_posteriors, sizeof(float) * num_defects * num_companies,
                         cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_product_posteriors, sizeof(float) * num_defects * num_products));
    checkCUDA(cudaMemcpy(d_product_posteriors, product_posteriors, sizeof(float) * num_defects * num_products,
                         cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_issue_posteriors, sizeof(float) * num_defects * num_issues));
    checkCUDA(cudaMemcpy(d_issue_posteriors, issue_posteriors, sizeof(float) * num_defects * num_issues,
                         cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_word_posteriors, sizeof(float) * num_defects * vocab_size));
    checkCUDA(cudaMemcpy(d_word_posteriors, word_posteriors, sizeof(float) * num_defects * vocab_size,
                         cudaMemcpyHostToDevice));



    /* ALLOCATE AND COPY FOR EMPIRICAL DATA (i.e the entity data of all complaints ) */
    float *d_TFDF, *d_company_vec,*d_product_vec,*d_issue_vec;

    checkCUDA(cudaMalloc((void **) &d_TFDF, sizeof(float) * num_complaints * vocab_size));
    checkCUDA(cudaMemcpy(d_TFDF, TFDF, sizeof(float) * num_complaints * vocab_size, cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_company_vec, sizeof(float) * num_complaints));
    checkCUDA(cudaMemcpy((void **) d_company_vec, company_vec, sizeof(float) * num_complaints, cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_product_vec, sizeof(float) * num_complaints));
    checkCUDA(cudaMemcpy(d_product_vec, product_vec, sizeof(float) * num_complaints, cudaMemcpyHostToDevice));


    checkCUDA(cudaMalloc((void **) &d_issue_vec, sizeof(float) * num_complaints));
    checkCUDA(cudaMemcpy((void **) d_issue_vec, issue_vec, sizeof(float) * num_complaints, cudaMemcpyHostToDevice));


    /* CREATE THE EXPANDED MATRICES */
    /* Calculate Host Values that will not change */
    float* h_expanded_company, *h_expanded_product, *h_expanded_issue, *h_ones_matrix, *h_ones_vector;

    checkCUDA(cudaMallocHost((void**)&h_expanded_company, sizeof(float) * num_companies*num_complaints));
    memset(h_expanded_company, 0, sizeof(float)*num_companies*num_complaints);

    checkCUDA(cudaMallocHost((void**)&h_expanded_product, sizeof(float) * num_products*num_complaints));
    memset(h_expanded_product, 0, sizeof(float)*num_products*num_complaints);

    checkCUDA(cudaMallocHost((void**)&h_expanded_issue, sizeof(float)*num_issues*num_complaints));
    memset(h_expanded_issue, 0, sizeof(float)*num_issues*num_complaints);

    checkCUDA(cudaMallocHost((void**)&h_ones_matrix, sizeof(float) * num_defects * vocab_size));
    memset(h_ones_matrix, 0, sizeof(float)*num_defects*vocab_size);

    checkCUDA(cudaMallocHost((void**)&h_ones_vector, sizeof(float) * num_defects ));
    memset(h_ones_vector, 0, sizeof(float)*num_defects);


    for (int i = 0; i < num_defects; i++) {
        for (int j = 0; j < vocab_size; j++) {
            h_ones_matrix[INDX(i, j, num_defects)] = (float) 1.0;
        }
        h_ones_vector[INDX(i, 0, num_defects)] = (float) 1.0;
    }

    for (int i = 0; i < num_complaints; i++){
        h_expanded_company[INDX(i, (int) company_vec[INDX(i,1,1)], num_complaints)] = 1;
        h_expanded_issue[INDX(i, (int) issue_vec[INDX(i,1,1)], num_complaints)] = 1;
        h_expanded_product[INDX(i, (int) product_vec[INDX(i,1,1)],num_complaints)] = 1;
    }

    /* Allocate and copy to device */
    float * d_expanded_company, * d_expanded_product, * d_expanded_issue;


    checkCUDA(cudaMalloc(&d_expanded_company, sizeof(float)*num_complaints*num_companies));
    checkCUDA(cudaMemcpy(d_expanded_company, h_expanded_company,
                         sizeof(float)*num_complaints*num_companies, cudaMemcpyHostToDevice));

    checkCUDA(cudaMalloc(&d_expanded_product, sizeof(float)*num_complaints*num_products));
    checkCUDA(cudaMemcpy(d_expanded_product, h_expanded_product,
                         sizeof(float)*num_complaints*num_products, cudaMemcpyHostToDevice));

    checkCUDA(cudaMalloc(&d_expanded_issue, sizeof(float)*num_complaints*num_issues));
    checkCUDA(cudaMemcpy(d_expanded_issue, h_expanded_issue,
                         sizeof(float)*num_complaints*num_issues, cudaMemcpyHostToDevice));


    /* FInd the sum across all words for each complaint */
    float *d_TFDF_SUM;
    checkCUDA(cudaMalloc((void **) &d_TFDF_SUM, sizeof(float) * num_complaints));

    /* create thread elements */
    dim3 blocks(num_complaints, 1, 1);
    thread_size = 256;
    dim3 threads(1, thread_size,1);

    cudaEvent_t start, stop;
    cudaEvent_t start_total, stop_total;
    cudaError_t err;
    FILE *s ;
    s = fopen(log_file_name, "w");

    checkCUDA(cudaEventCreate(&start));
    checkCUDA(cudaEventCreate(&stop));
    checkCUDA(cudaEventRecord(start, 0));



    reduce_columns<<<blocks, threads, sizeof(float)*threads.y>>>(num_complaints, vocab_size, d_TFDF, d_TFDF_SUM);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    checkCUDA(cudaEventRecord(stop, 0));
    checkCUDA(cudaEventSynchronize(stop));
    checkCUDA(cudaEventElapsedTime(&elapsedTime, start, stop));

    /* print GPU CUBLAS timing information */
    fprintf(s, "Total time GPU KERNAL for TFDF SUM is %f sec\n", elapsedTime / 1000.0f);
    checkCUDA(cudaEventDestroy(start));
    checkCUDA(cudaEventDestroy(stop));




/* Sanity check to make sure the sum reduce worked */
//    /* Find the sum_{words} for each complaint of TFDF  using CUBLAS */
//    float * d_cublas_sum;
//    checkCUDA(cudaMalloc(&d_cublas_sum, sizeof(float)*num_complaints));
//    cublas_column_reduce(d_TFDF, num_complaints, vocab_size, d_cublas_sum);




    old_likelihood = -INFINITY;
    delta_likelihood = 10000000;
    float *new_likelihood;
    float *d_likelihood;
    fprintf(stdout, "-----------Beggining Expectation Maximization Routine on %d complaints and %d words---------\n",
           num_complaints, vocab_size);



    checkCUDA(cudaEventCreate(&start_total));
    checkCUDA(cudaEventCreate(&stop_total));
    checkCUDA(cudaEventRecord(start_total, 0));

    int iter = 0;
    double total_estep_time = 0.0;
    double total_mstep_time = 0.0;


    while (delta_likelihood > (float) .00001 || iter < 10) {
        iter = iter + 1;

        threads.x = num_defects;
        threads.y = THREADS_PER_BLOCK_Y;
        blocks.x = ceil(num_complaints / threads.y) + 1;
        blocks.y = 1;
        new_likelihood = (float *)malloc(sizeof(float)*blocks.x);
        checkCUDA(cudaMalloc((void **) &d_likelihood, sizeof(float) * blocks.x));
        checkCUDA(cudaMemset(d_likelihood, 0, sizeof(float) * blocks.x));

        checkCUDA(cudaEventCreate(&start));
        checkCUDA(cudaEventCreate(&stop));
        checkCUDA(cudaEventRecord(start, 0));

        eStep <<< blocks, threads >>> (num_complaints,  vocab_size,
                num_defects, d_defect_priors,
                d_defect_posteriors, d_company_posteriors,
                d_issue_posteriors, d_product_posteriors,
                d_word_posteriors, d_TFDF,
                d_company_vec, d_issue_vec,
                d_product_vec, d_likelihood);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("YOO Error: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        checkCUDA(cudaEventRecord(stop));
        cudaEventSynchronize(stop);
        checkCUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
        total_estep_time += elapsedTime/ 1000.0f;




        fprintf(s, "---------Total time For E_STEP on GPU is %f sec ---------\n", elapsedTime / 1000.0f);

        checkCUDA(cudaEventDestroy(start));
        checkCUDA(cudaEventDestroy(stop));
        checkCUDA(cudaMemcpy(new_likelihood, d_likelihood, sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));

        float total_likelihood = 0.0;
        for (int i = 0; i < blocks.x; i++) {
            total_likelihood += new_likelihood[i];
        }
        delta_likelihood = float(fabsf(old_likelihood -total_likelihood ));


        fprintf(s,"(OLD LIKELIHOOD = %f, UPDATED LIKELIHOOD = %f , Change in Likelihood =%f)\n",
               old_likelihood,total_likelihood, (total_likelihood-old_likelihood));

//        printf("Change in Likelihood is %f:\n", delta_likelihood);
        old_likelihood = total_likelihood;



//
//        fprintf(s, "--------------DOING M-STEP WITH CUBLAS---------------------- \n");
//
//
        checkCUDA(cudaEventCreate(&start));
        checkCUDA(cudaEventCreate(&stop));
        checkCUDA(cudaEventRecord(start, 0));

        M_STEP_CUBLAS(num_complaints, vocab_size,
                      num_defects, d_defect_priors,
                      d_defect_posteriors, d_company_posteriors,
                      d_issue_posteriors, d_product_posteriors,
                      d_word_posteriors, d_TFDF,
                      d_expanded_company, d_expanded_issue,
                      d_expanded_product, d_TFDF_SUM,
                      num_companies, num_products,
                      num_issues
        );
        checkCUDA(cudaEventRecord(stop, 0));
        checkCUDA(cudaEventSynchronize(stop));
        float elapsedTime;
        checkCUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
        total_mstep_time += elapsedTime/ 1000.0f;
        fprintf(s, "Total time GPU M Step %f sec\n", elapsedTime / 1000.0f);
        checkCUDA(cudaEventDestroy(start));
        checkCUDA(cudaEventDestroy(stop));
//
//        printf("--------------DOING M-STEP KERNEL---------------------- \n");
//
//
//        checkCUDA(cudaEventCreate(&start));
//        checkCUDA(cudaEventCreate(&stop));
//        checkCUDA(cudaEventRecord(start, 0));
//        execute_MStep(num_complaints, vocab_size,
//                      num_defects, d_defect_priors,
//                      d_defect_posteriors, d_company_posteriors,
//                      d_issue_posteriors, d_product_posteriors,
//                      d_word_posteriors, d_TFDF,
//                      d_expanded_company, d_expanded_issue,
//                      d_expanded_product,
//                      d_TFDF_SUM, num_companies,
//                      num_products, num_issues,0
//        );
//
//        checkCUDA(cudaEventRecord(stop));
//        cudaEventSynchronize(stop);
//        checkCUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
//        total_mstep_time += elapsedTime/ 1000.0f;
//
//        checkCUDA(cudaEventDestroy(start));
//        checkCUDA(cudaEventDestroy(stop));
//        fprintf(s, "--------------Total time For M-STEP on GPU is %f sec\n--------------", elapsedTime / 1000.0f);

        checkCUDA(cudaFree(d_likelihood));
        

    }
    checkCUDA(cudaEventRecord(stop_total, 0));
    checkCUDA(cudaEventSynchronize(stop_total));

    checkCUDA(cudaEventElapsedTime(&elapsedTime, start_total, stop_total));
    fprintf(s, "Total time till convergece is %f sec | %d iterations \n", total_estep_time + total_mstep_time, iter);
    fprintf(s, "Average Time of eStep is %f sec: \n", total_estep_time/iter);
    fprintf(s, "Average Time of MStep is %f sec: \n", total_mstep_time/iter);

    fprintf(s, "Finally Likelihood %f\n", old_likelihood);
    fprintf(s, "Change in likelihood %f\n", delta_likelihood);

    checkCUDA(cudaMemcpy(defect_posteriors, d_defect_posteriors, sizeof(float)*num_defects*num_complaints, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(defect_priors, d_defect_priors, sizeof(float)*num_defects, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(word_posteriors, d_word_posteriors, sizeof(float)*num_defects*vocab_size, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(product_posteriors, d_product_posteriors, sizeof(float)*num_defects*num_products, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(issue_posteriors, d_issue_posteriors, sizeof(float)*num_defects*num_issues, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(company_posteriors, d_company_posteriors, sizeof(float)*num_defects*num_products, cudaMemcpyDeviceToHost));
    fclose(s);
    FILE *f = fopen(results_file_name, "w");
    fprintf(stdout, "---------DONE---------\n");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }


    for (int i=0; i < num_companies; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"COMPANY, %d, DEFECT, %d, POSTERIOR, %f \n", i, j,company_posteriors[INDX(j, i, num_defects)] );
        }
    }

    for (int i=0; i < num_issues; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"ISSUE, %d, DEFECT,%d, POSTERIOR, %f \n", i, j,issue_posteriors[INDX(j, i, num_defects)] );
        }
    }

    for (int i=0; i < vocab_size; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"WORD, %d,  DEFECT %d, POSTERIOR, %f \n", i, j,word_posteriors[INDX(j, i, num_defects)] );
        }
    }

    for (int i=0; i < num_complaints; i++){
        for (int j=0; j < num_defects; j++){
            fprintf(f,"Complaint, %d, DEFECT, %d, POSTERIOR: %f \n", i, j,defect_posteriors[INDX(j, i, num_defects)] );
        }
    }

    for (int j=0; j < num_defects; j++){
        fprintf(f,"DEFECT, %d, , Prior : %f \n", j,defect_priors[j] );
    }

    fclose(f);






    checkCUDA(cudaFree(d_defect_posteriors));
    checkCUDA(cudaFree(d_defect_priors));
    checkCUDA(cudaFree(d_word_posteriors));
    checkCUDA(cudaFree(d_product_posteriors));
    checkCUDA(cudaFree(d_issue_posteriors));
    checkCUDA(cudaFree(d_TFDF));
    checkCUDA(cudaFree(d_TFDF_SUM));
    checkCUDA(cudaFree(d_expanded_company));
    checkCUDA(cudaFree(d_expanded_issue));
    checkCUDA(cudaFree(d_expanded_product));
    checkCUDA(cudaFree(d_issue_vec));
    checkCUDA(cudaFree(d_company_vec));
    checkCUDA(cudaFree(d_product_vec));







}



__global__ void elementwise_division(int const num_rows, int const num_columns,
                                     float *d_denominator, float *d_numerator,
                                     float *output, float numerator_lambda,
                                     float denominator_lamda)
{

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_columns; i += blockDim.x * gridDim.x) {

            output[INDX(threadIdx.y, i, num_rows)] =
                    (float) ((numerator_lambda +  d_numerator[INDX(threadIdx.y, i, num_rows)]) /
                             (denominator_lamda + d_denominator[INDX(threadIdx.y,0, num_rows)]));

    }


}




void execute_MStep(int const num_complaints, int const vocab_size,
                   int const num_defects, float *d_defect_priors,
                   float *d_defect_posteriors, float *d_company_posteriors,
                   float *d_issue_posteriors, float *d_product_posteriors,
                   float *d_word_posteriors, float *d_TFDF,
                   float *d_expanded_company_vec, float *d_expanded_issue_vec,
                   float *d_expanded_product_vec,
                   float *d_TFDF_SUM, int const num_companies,
                   int const num_products, int const num_issues,
                   int check_kernal)

{
    cudaError_t err;
    double temp;
    float epsilon = 1e-8;
    float *d_denominator, *d_defect_posterior_sum;
    float *h_denominator,  * h_defect_posterior_sum,*h_defect_prior;
    dim3 threads, blocks;


    /* printf("----------UPDATING THE WORD POSTERIORS INSIDE MATRIX MULTIPLICATION---------------\n"); */
    h_denominator = (float *) malloc(sizeof(float) * num_defects);
    checkCUDA(cudaMalloc(&d_denominator, sizeof(float) * num_defects));

    threads.x = TILE_WIDTH;
    threads.y = TILE_WIDTH;
    blocks.x = ceil(1 + threads.x - 1) / threads.x;
    blocks.y = ceil((num_defects + threads.y - 1) / threads.y);

    mat_mul <<< blocks, threads >>> (num_defects, num_complaints,
                                    num_complaints, 1,
                                    d_defect_posteriors, d_TFDF_SUM,
                                    d_denominator, num_defects, 1);

    if (check_kernal) {
        temp = 0.0;
        float *h_denominator_cublas;
        float *d_denominator_cublas;
        h_denominator_cublas = (float *) malloc(sizeof(float) * num_defects);
        checkCUDA(cudaMalloc(&d_denominator_cublas, sizeof(float) * num_defects))

        cublas_mat_mul(num_defects, num_complaints, num_complaints, 1, d_defect_posteriors, d_TFDF_SUM,
                       d_denominator_cublas, num_defects,
                       1, 1.0, 0);


        checkCUDA(cudaMemcpy(h_denominator, d_denominator, sizeof(float) * num_defects, cudaMemcpyDeviceToHost));
        checkCUDA(cudaMemcpy(h_denominator_cublas, d_denominator_cublas, sizeof(float) * num_defects,
                             cudaMemcpyDeviceToHost));


        for (int i = 0; i < num_defects; i++) {

            temp += (h_denominator[INDX(i, 0, num_defects)] - h_denominator_cublas[INDX(i, 0, num_defects)])
                    * (h_denominator[INDX(i, 0, num_defects)] - h_denominator_cublas[INDX(i, 0, num_defects)]);

        }
        printf("error is %f\n", temp);
        if (temp > 10) printf("FAIL\n");
        else printf("PASSED ACCURACY TEST  FOR DENOMINATOR \n");

        free(h_denominator_cublas);
        checkCUDA(cudaFree(d_denominator_cublas));

    }


    blocks.x = ceil((vocab_size + threads.x - 1) / threads.x);
    blocks.y = ceil((num_defects + threads.y - 1) / threads.y);


    update_entities <<< blocks, threads >>> (num_defects, num_complaints,
                                            num_complaints, vocab_size,
                                            d_defect_posteriors, d_TFDF,
                                            d_word_posteriors, num_defects,
                                            vocab_size, d_denominator,
                                            1, vocab_size);



    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("YOO Error: %s\n", cudaGetErrorString(err));
        exit(0);
    }



    /* printf("----------------------SUMMING POSTERIOR FOR DEFECT-----------------------\n");*/
    blocks.x = num_defects;
    blocks.y =1;
    threads.x = 1;
    threads.y = 256;

    checkCUDA(cudaMalloc(&d_defect_posterior_sum, sizeof(float)*num_defects));


    reduce_columns<<<blocks, threads, sizeof(float)*threads.y>>>(num_defects, num_complaints,
                                                                d_defect_posteriors, d_defect_posterior_sum);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    /*   printf("-------------------------UPDATING ENTITIES---------------------\n"); */

    free(h_denominator);
    checkCUDA(cudaFree(d_denominator));

    /*Companies */
    threads.x = TILE_WIDTH;
    threads.y = TILE_WIDTH;
    blocks.x = ceil(num_companies + threads.x - 1) / threads.x;
    blocks.y = ceil((num_defects + threads.y - 1) / threads.y);
    update_entities <<< blocks, threads >>> (num_defects, num_complaints,
                                            num_complaints, num_companies,
                                            d_defect_posteriors, d_expanded_company_vec,
                                            d_company_posteriors, num_defects,
                                            num_companies, d_defect_posterior_sum,
                                            1, num_companies);
    /*Issues */
    blocks.x = ceil(num_issues + threads.x - 1) / threads.x;

    update_entities <<< blocks, threads >>> (num_defects, num_complaints,
                                            num_complaints, num_issues,
                                            d_defect_posteriors, d_expanded_issue_vec,
                                            d_company_posteriors, num_defects,
                                            num_issues, d_defect_posterior_sum,
                                            1, num_issues);
    /*Products */
    blocks.x = ceil(num_products + threads.x - 1) / threads.x;
    update_entities <<<blocks, threads >>> (num_defects, num_complaints,
                                            num_complaints, num_products,
                                            d_defect_posteriors, d_expanded_product_vec,
                                            d_company_posteriors, num_defects,
                                            num_products, d_defect_posterior_sum,
                                            1, num_products);

    /* printf("----------UPDATING Priors--------------\n");*/

    h_defect_posterior_sum = (float *) malloc(sizeof(float)*num_defects);
    h_defect_prior = (float *) malloc(sizeof(float)*num_defects);
    checkCUDA(cudaMemcpy(h_defect_posterior_sum, d_defect_posterior_sum,
                         sizeof(float)*num_defects,cudaMemcpyDeviceToHost ));
    for(int i=0; i < num_defects; i ++){
        if (h_defect_posterior_sum[i] < epsilon){
            printf("prior %d is too small: %f \n", i,h_defect_posterior_sum[i] );
            h_defect_prior[i] = epsilon;

        }else{
            h_defect_prior[i] = (float) h_defect_posterior_sum[i]/num_complaints;
        }
    }

    checkCUDA(cudaMemcpy(d_defect_priors, h_defect_prior, sizeof(float)*num_defects,cudaMemcpyHostToDevice));

    /*  printf("------------------------------------ \n");*/

    checkCUDA(cudaFree(d_defect_posterior_sum));
    free(h_defect_posterior_sum);
    free(h_defect_prior);



}

void cublas_mat_mul(const int a_rows, const int a_columns,
                    const int b_rows, const int b_columns,
                    float * d_A, float * d_B,
                    float *  d_C, const int c_rows,
                    const int c_columns, float alpha,
                    float beta)
{


    checkCUBLAS(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            a_rows, b_columns, b_rows,
            &alpha, d_A,
            a_rows, d_B, b_rows,
            &beta, d_C, a_rows
    ));




}


void M_STEP_CUBLAS(int const num_complaints, int const vocab_size,
                   int const num_defects, float *d_defect_priors,
                   float *d_defect_posteriors, float *d_company_posteriors,
                   float *d_issue_posteriors, float *d_product_posteriors,
                   float *d_word_posteriors, float *d_TFDF,
                   float *d_expanded_company_vec, float *d_expanded_issue_vec,
                   float *d_expanded_product_vec, float *d_TFDF_SUM, int const num_companies,
                   int const num_products, int const num_issues

) {
    cudaError_t err;
    checkCUBLAS(cublasCreate(&handle));
    float epsilon = 1e-6;
    float * h_defect_posterior_sum,*h_defect_prior, *d_numerator,
            *d_denominator;


    /* printf("----------UPDATING THE WORD POSTERIORS ---------------\n"); */


    /* Numerator */
    checkCUDA(cudaMalloc(&d_numerator, sizeof(float) * num_defects * vocab_size));

    cublas_mat_mul(num_defects, num_complaints,
                   num_complaints, vocab_size,
                   d_defect_posteriors, d_TFDF,
                   d_numerator, num_defects,
                   vocab_size, 1.0, 0.0);

    /* Denominator */

    checkCUDA(cudaMalloc(&d_denominator, sizeof(float) * num_defects))

    cublas_mat_mul(num_defects, num_complaints, num_complaints, 1, d_defect_posteriors, d_TFDF_SUM,
                   d_denominator, num_defects,
                   1, 1.0, 0);


    /* Elementwise divison */
    int threads_per_block = THREADS_PER_BLOCK_Y;
    dim3 threads( threads_per_block,num_defects, 1);
    dim3 blocks((num_complaints / threads_per_block) + 1, 1, 1);

    elementwise_division<<<threads, blocks >>>  (num_defects, vocab_size,
                                                d_denominator, d_numerator,
                                                d_word_posteriors, (float) 1.0,
                                                (float) vocab_size);


    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    checkCUDA(cudaFree(d_numerator));
    checkCUDA(cudaFree(d_denominator));

    /* printf("----------SUMMING POSTERIOR FOR DEFECT--------------\n");*/
    float * h_x;
    float *d_x;
    h_x = (float *) malloc(sizeof(float) * num_complaints);

    for (int i = 0; i < num_complaints; i++) {
        h_x[INDX(i, 0, num_complaints)] = (float) 1.0;
    }


    checkCUDA(cudaMalloc(&d_x, sizeof(float) * num_complaints));
    checkCUDA(cudaMemcpy(d_x, h_x, sizeof(float) * num_complaints, cudaMemcpyHostToDevice));

    float *d_defect_posterior_sum;
    checkCUDA(cudaMalloc(&d_defect_posterior_sum, sizeof(float) * num_defects));


    cublas_mat_mul(num_defects, num_complaints,
                   num_complaints, 1,
                   d_defect_posteriors, d_x,
                   d_defect_posterior_sum, num_defects,
                   1, 1.0, 0);


    /* -----------------Updating Entities---------------------------- */


    checkCUDA(cudaMalloc(&d_numerator, sizeof(float)*num_companies*num_defects));

    /* Companies */
    cublas_mat_mul(num_defects, num_complaints,
                   num_complaints, num_companies,
                   d_defect_posteriors, d_expanded_company_vec,
                   d_numerator, num_defects,
                   1, 1.0, 0);

    elementwise_division <<< threads, blocks >>>(num_defects, num_companies,
                                                d_defect_posterior_sum, d_numerator,
                                                d_company_posteriors, (float) 1.0,
                                                (float) num_companies);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));


    /* Products */
    checkCUDA(cudaFree(d_numerator));
    checkCUDA(cudaMalloc(&d_numerator, sizeof(float)*num_products*num_defects));

    cublas_mat_mul(num_defects, num_complaints,
                   num_complaints, num_products,
                   d_defect_posteriors, d_expanded_product_vec,
                   d_numerator, num_defects,
                   1, 1.0, 0);

    elementwise_division <<< threads, blocks >>> (num_defects, num_products,
                                                  d_defect_posterior_sum, d_numerator,
                                                  d_product_posteriors, (float) 1.0,
                                                  (float) num_products);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    checkCUDA(cudaFree(d_numerator));

    /* Issues */
    checkCUDA(cudaMalloc(&d_numerator, sizeof(float)*num_issues*num_defects));
    cublas_mat_mul(num_defects, num_complaints,
                   num_complaints, num_issues,
                   d_defect_posteriors, d_expanded_product_vec,
                   d_numerator, num_defects,
                   1, 1.0, 0);

    elementwise_division <<< threads, blocks >>> (num_defects, num_issues,
                                                 d_defect_posterior_sum, d_numerator,
                                                 d_issue_posteriors, (float) 1.0,
                                                 (float) num_issues);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));


    /* -----------------Updating Priors---------------------------- */
    h_defect_posterior_sum = (float *) malloc(sizeof(float)*num_defects);
    h_defect_prior = (float *) malloc(sizeof(float)*num_defects);

    checkCUDA(cudaMemcpy(h_defect_posterior_sum, d_defect_posterior_sum,
                         sizeof(float)*num_defects,cudaMemcpyDeviceToHost ));

    for(int i=0; i < num_defects; i ++){
        if (h_defect_posterior_sum[i] < epsilon){
            printf("prior %d is too small: %f \n", i,h_defect_posterior_sum[i] );
            h_defect_prior[i] = epsilon;

        }else{
            h_defect_prior[i] = (float) h_defect_posterior_sum[i] / num_complaints;
        }
    }
    checkCUDA(cudaMemcpy(d_defect_priors, h_defect_prior, sizeof(float)*num_defects,cudaMemcpyHostToDevice));


    checkCUDA(cudaFree(d_numerator));
    checkCUDA(cudaFree(d_defect_posterior_sum));
    free(h_defect_posterior_sum);
    free(h_defect_prior);

}


void readMatrixFromFile(char *fileName,
                        float *matrix,
                        int const rows,
                        int const cols,
                        int const ld) {
    FILE *ifp;

    ifp = fopen(fileName, "r");


    if (ifp == NULL) {
        fprintf(stderr, "Error opening file %s\n", fileName);
        exit(911);
    } /* end if */

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {


            if (!fscanf(ifp, " %f",
                        &matrix[INDX(row, col, ld)])) {
                printf("%d\n", INDX(row, col, ld));
                printf("error in element %d and %d\n", row, col);
                fprintf(stderr, "error reading training matrix file \n");
                perror("scanf:");
                exit(911);
            }
            /* end if */
        } /* end for row */
    } /* end for col */

    fclose(ifp);
    return;
}

