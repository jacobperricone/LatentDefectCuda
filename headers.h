//
// Created by jacob  Perricone on 2/12/17.
//
//


extern "C"
{
#include <cblas.h>
}


/* Define the tolerance and max iterations for convergence of EM algorithm */
#define TOLERANCE (.0001)
#define MAXITER (10000)

/* Number of defects (latent variables in the model*/
#define NUM_DEFECTS 3

/*Number of companies for a small and mid sized sample */
#define NUM_COMPANIES_SMALL (37)
#define NUM_COMPANIES_MID (39)

/* Number of issues in a small and mid sized sample */
#define NUM_ISSUES_SMALL (55)
#define NUM_ISSUES_MID (61)

/* Number of products in a small and mid sized sample */
#define NUM_PRODUCTS_SMALL (6)
#define NUM_PRODUCTS_MID (6)

/* number of complaints in a small and mid sized sample */
#define NUM_COMPLAINTS_MID (13489)
#define NUM_COMPLAINTS_SMALL (2697)

/* Size of the vocabulary for small and mid sized sample */
#define VOCAB_SIZE_MID (6474)
#define VOCAB_SIZE_SMALL (3045)

#ifdef DEBUG
#define checkCUDA(F)  if( (F) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
__FILE__,__LINE__); exit(-1);}

#define checkKERNEL()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
__FILE__,__LINE__-1); exit(-1);} \
if( (cudaDeviceSynchronize()) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
__FILE__,__LINE__); exit(-1);}

#define checkCUBLAS(F)  if( (F) != CUBLAS_STATUS_SUCCESS ) \
{printf("Error %d at %s:%d\n", F, \
__FILE__,__LINE__); exit(-1);}

#else

#define checkCUDA(F) (F)
#define checkKERNEL()
#define checkCUBLAS(F) (F)

#endif



typedef float floatType_t;

/* macro to convert 2d coords to 1d offset */

#define INDX(row,col,ld) (((col) * (ld)) + (row))


/* Helper function to read data from file name
*
* Params:
* char * filename : file to read from
* float * data: matrix into which the data is loaded
* const int rows: number of rows to read
* const int cols: number of cols to read
* const in ld: number of rows per column (data is stored column-major)
*/
void readMatrixFromFile( char * filename, float * data, const int rows, const int cols, const int ld);

/* Fills randomly generated data between 0-1 into the dynamic float array data, of size: (rows, cols)
*
* Params:
* float * data: array to be filled
* int rows: number of rows in array
* int cols: number of columns to be filled
* int ld: number of rows per column
*/
void RandomInit(float *data,int rows, int cols, int ld);

/* Description:
 * runEM is the wrapper function that takes as inputs the parameters of data of the model and runs the expectation-
 * maximization algorithm on the dataset until convergence indicated by tolerance or maxIter is reached. It takes
 * as input host pointers to the term-frequency inverse-document frequency matrix, the entity_vecs (i.e.  product_vec
 * is a num_complaints sized vector where each entry is what product the complaint corresponded to), and randomly
 * initialized defect posteriors and priors. These samples are assumed to be drawn from a multinomial distribution.
 * More information of the math behind the probabalistic defect model can be found in the addendum paper.
 *
 * Params:
 * TFDF (nrows=num_complaints, cols=vocab_size): TF-IDF matrix
 * num_complaints : Number of complaints in the data set
 * vocab_size: number of words in the complaint corpus
 * num_defects: number of defects
 * num_companies: number of companies in the set
 * num_products: number of products in the complaint set
 * num_issues: number of issues in the set
 * company_vec (rows= num_complaints, cols = 1)
 * product_vec (rows= num_complaints, cols = 1)
 * issue_vec (rows=num_complaints, cols = 1)
 * defect_priors (rows=num_defects, cols = 1);
 * defect_posteriors (rows=num_defects, cols = num_complaints)
 * defect_posteriors (rows=num_defects, cols=num_complaints):  the probability of defect j given complaint i
 * company_posteriors (rows= num_defects, cols=num_companies): probability of company j given defect i
 * issue_posteriors (rows=num_defects, cols=num_issues): probability of issue j given defect i
 * product_posteriors (rows=num_defects, cols=num_products): probability of product j given defect i
 * word_posteriors (rows=num_defects, cols=vocab_size): probability of word j given defect i
 * tol: tolerance for successive change in log likelihood
 * maxITer: maximum iterations of the algo
 * log_file_name: where to write log file
 * results_file_name: where to pipe results
 * */

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
);


/*
* Description:
* M_STEP_CUBLAS performs the maximization step using primarily cuBLAS matrix operations, returning
* updated pointers to device posteriors (i.e. *_posteriors) and device priors. All variables beginning
* with d_ are device pointers. The points containing _expanded_x_vec contain num_complaint rows and num_x columns
* and have a value one in coordinate (i,j) if complain i has j in its corresponding x vector and zero otherwise
* For example if d_product_vec has value 4 in its i'th entry, d_expanded_product vec has a one in its i'th row
* and fourth column and zero in all other columns.
*
* Params:
* num_complaints : Number of complaints in the data set
* vocab_size: number of words in the complaint corpus
* num_defects: number of defects
* num_companies: number of companies in the set
* num_products: number of products in the complaint set
* num_issues: number of issues in the set
* d_defect_priors (rows=num_defects, cols=1): a device pointer holding a prior of the defects
* d_defect_posteriors (rows=num_defects, cols=num_complaints):  the probability of defect j given complaint i
* d_company_posteriors (rows= num_defects, cols=num_companies): probability of company j given defect i
* d_issue_posteriors (rows=num_defects, cols=num_issues): probability of issue j given defect i
* d_product_posteriors (rows=num_defects, cols=num_products): probability of product j given defect i
* d_word_posteriors (rows=num_defects, cols=vocab_size): probability of word j given defect i
* d_TFDF (rows=num_complaints, cols=vocab_size): Term Frequency Inverse document frequency matrix for the complaint
* corpus, each coordinate (i,j) is the number of times word j appeared in complaint i
* d_expanded_company_vec(rows = num_complaints, cols = num_companies)
* d_expanded_issue_vec(rows= num_complaints, cols = num_issues)
* d_expanded_product_vec(rows=num_complaints, cols=num_product)
* d_TFDF_SUM(rows = num_complaints, cols = 1): Sum over columns of d_TFDF
*
*/
void M_STEP_CUBLAS(int const num_complaints, int const vocab_size,
       int const num_defects, float *d_defect_priors,
       float *d_defect_posteriors, float *d_company_posteriors,
       float *d_issue_posteriors, float *d_product_posteriors,
       float *d_word_posteriors, float *d_TFDF,
       float *d_expanded_company_vec, float *d_expanded_issue_vec,
       float *d_expanded_product_vec, float *d_TFDF_SUM,
       int const num_companies, int const num_products,
       int const num_issues

);



/* Description:
 * execute_Mstep is a w rapper function for the cuda implementation of the Maximization step.
 * The paramaters are the same as M_STEP_CUBLAS with the addition of check_kernal, which is a debugging tool
 * to ensure the matrix multiplications were accurate by comparing them
 * to the cUBLAS implementation
 * */

void execute_MStep(int const num_complaints, int const vocab_size,
                   int const num_defects, float *d_defect_priors,
                   float *d_defect_posteriors, float *d_company_posteriors,
                   float *d_issue_posteriors, float *d_product_posteriors,
                   float *d_word_posteriors, float *d_TFDF,
                   float *d_expanded_company_vec, float *d_expanded_issue_vec,
                   float *d_expanded_product_vec,
                   float *d_TFDF_SUM, int const num_companies,
                   int const num_products, int const num_issues,
                   int check_kernal);

/* Sums across the columns of the matrix by multiplying the matrix d_A (device pointer) by
 * a vector of ones containting the same number of columns as d_A. stores the result in d_y,
 * which should be allocated for */
void cublas_column_reduce(const float * d_A, const int num_rows, const int num_columns, const float * d_y);




/* Wrapper function for a cublas call to clean up code */
void cublas_mat_mul(const int a_rows, const int a_columns, const int b_rows, const int b_columns,  float * d_A, float * d_B, float *  d_C, const int c_rows,
   const int c_columns, float alpha, float beta);


/* Run CPU version of code */
void runEM_CPU(float *TFDF, int const num_complaints,
               int const vocab_size, float *company_vec,
               int const num_companies, float *issue_vec,
               int const num_issues, float *product_vec,
               int const num_products, int const num_defects,
               float *defect_priors_cpu, float *defect_posteriors_cpu,
               float *company_posteriors_cpu, float *issue_posteriors_cpu,
               float *product_posteriors_cpu, float *word_posteriors_cpu,
               const float tol, const int maxIter, const char * log_file_name,
               const char *results_file_name);