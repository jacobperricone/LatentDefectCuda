/* Author: Jacob Perricone
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "headers.h"



int main(int argc, char *argv[])
{

    /* get GPU device number and name */
    char const * log_file_name, *results_file_name;
    if (argc == 3){
        log_file_name =(char const *) malloc(strlen(argv[1]) + strlen("logs/") + 1);
        strcpy((char *) log_file_name, argv[1]);
        results_file_name =  (char const *)malloc(sizeof(char)*strlen(argv[2]) + 1 + strlen("results/"));
        strcpy((char *)results_file_name, argv[2]);

    }else{
        log_file_name = (char const *) malloc(strlen( "logs/Log_File.txt") + 1);
        log_file_name = "logs/Log_File.txt";
        results_file_name = (char const *) malloc(strlen("logs/Results_File.txt") + 1);
        results_file_name = "results/Results_File.txt";
    }
    int dev;
    cudaDeviceProp deviceProp;
    checkCUDA( cudaGetDevice( &dev ) );
    checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
    printf("Using GPU %d: %s\n", dev, deviceProp.name );

    /* declare file pointers  to load the data from */
    char tfdfFilename[]    = "tfdf_Small.txt";
    char companyFilename[]      = "Company_Small.txt";
    char productFilename[]          = "Product_Small.txt";
    char issueFilename[]        = "Issue_Small.txt";

    /* define constants */
    int const max_iter = MAXITER;
    int const num_complaints = NUM_COMPLAINTS_SMALL;
    int const vocab_size = VOCAB_SIZE_SMALL;
    int const num_companies = NUM_COMPANIES_SMALL;
    int const num_products = NUM_PRODUCTS_SMALL;
    int const num_issues = NUM_ISSUES_SMALL;
    int const num_defects = NUM_DEFECTS;
    const float tol = TOLERANCE;


    printf("Numbeer of complaints %d %d \n", num_complaints, vocab_size);

    /* declare arrays to read in the data files, these will be pinned to the host for efficient transfer to device*/
    float *TFDF, *company_vec, *product_vec, *issue_vec;
    /*decare randomly initialized posterirors and priors */
    float *defect_priors, *defect_posteriors, *company_posteriors, *issue_posteriors, *product_posteriors;
    float *defect_priors_cpu, *defect_posteriors_cpu, *company_posteriors_cpu, *issue_posteriors_cpu, *product_posteriors_cpu;
    float *word_posteriors, * word_posteriors_cpu;


    /* Allocate Memory for TFDF matrix */
    checkCUDA(cudaMallocHost((void**)&TFDF, sizeof(float) * num_complaints * vocab_size));

    if (TFDF == NULL) {
        fprintf(stderr, "Unable to allocate TFDF matrix\n");
        exit(0);
    }
    memset( TFDF, 0, sizeof(float)*num_complaints*vocab_size);
    /* Read TFDF matrix from file  */
    readMatrixFromFile( tfdfFilename,
                        TFDF,
                        num_complaints, vocab_size,num_complaints);

    printf("Finished reading TFDF Matrix \n");


    checkCUDA(cudaMallocHost((void**)&company_vec, sizeof(float) * num_complaints));
    if (company_vec == NULL) {
        fprintf(stderr, "Unable to allocate Company matrix\n");
        exit(0);
    }

    memset( company_vec, 0, sizeof(float)*num_complaints );
    /* Read company vector from file  */
    readMatrixFromFile( companyFilename, company_vec, num_complaints, 1,1);
    printf("Finished reading company vector");

    /* Allocate Memory for Product Vector */
    checkCUDA(cudaMallocHost((void**)&product_vec, sizeof(float) * num_complaints));

    if (product_vec == NULL) {
        fprintf(stderr, "Unable to allocate TFDF matrix");
        exit(0);
    }

    memset( product_vec, 0, sizeof(float)*num_complaints );
    /* Read product vector from file  */
    readMatrixFromFile( productFilename, product_vec, num_complaints, 1,1);
    printf("Finished reading product vector");

    /* Allocate Memory for Issue Vector */
    checkCUDA(cudaMallocHost((void**)&issue_vec, sizeof(float) * num_complaints));
    if (issue_vec == NULL) {
        fprintf(stderr, "Unable to allocate TFDF matrix");
        exit(0);
    }
    memset( issue_vec, 0, sizeof(float)*num_complaints );
    /* Read issue vector from file  */
    readMatrixFromFile( issueFilename, issue_vec, num_complaints, 1,1);
    printf("Finished reading issue vector\n");



    /* ALLOCATE MEMORY FOR POSTERIORS TO BE ESTIMATED */

    /* Allocat For Defect Priors  and initialize with random numbers*/
    printf("Initializing Defect Priors... \n");
    defect_priors = (float *) malloc(sizeof(float) * num_defects);
    defect_priors_cpu = (float *) malloc(sizeof(float) * num_defects);

    RandomInit(defect_priors, num_defects, 1, 1);
    memcpy(defect_priors_cpu, defect_priors, sizeof(float) * num_defects);


    printf("Sucessfully Initialized Defect Priors\n");

    /* Allocate  For Defect Posteriors  and initialize with random numbers*/
    printf("Initializing Defect Posteriors... \n");
    defect_posteriors = (float *) malloc(sizeof(float)*num_complaints*num_defects);
    defect_posteriors_cpu =(float *) malloc(sizeof(float)*num_complaints*num_defects);

    RandomInit(defect_posteriors, num_defects, num_complaints, num_defects);

    memcpy(defect_posteriors_cpu, defect_posteriors, sizeof(float) * num_defects*num_complaints);
    printf("Sucessfully Initialized Defect Posteriors\n");


    printf("Initializing Company Posteriors... \n");
    /* Allocate for Company Posteriors and initialize with random numbers*/
    company_posteriors = (float *) malloc(sizeof(float)*num_defects*num_companies);
    company_posteriors_cpu = (float *) malloc(sizeof(float)*num_defects*num_companies);

    RandomInit(company_posteriors, num_defects, num_companies, num_defects);
    memcpy(company_posteriors_cpu, company_posteriors, sizeof(float) * num_defects*num_companies);
    printf("Succesfully Initialized Company Posteriors... \n");


    printf("Initializing Issue Posteriors... \n");
    /* Allocate for Issue Posteriors and initialize with random numbers */
    issue_posteriors = (float *) malloc(sizeof(float)*num_defects*num_issues);
    issue_posteriors_cpu = (float *) malloc(sizeof(float)*num_defects*num_issues);

    RandomInit(issue_posteriors, num_defects, num_issues, num_defects);
    memcpy(issue_posteriors_cpu, issue_posteriors, sizeof(float) * num_defects*num_issues);
    printf("Succesfully Initialized Issue Posteriors... \n");

    printf("Initializing Product Posteriors... \n");
    /* Allocate for Product Posteriors */
    product_posteriors = (float *) malloc(sizeof(float)*num_defects*num_products);
    product_posteriors_cpu = (float *) malloc(sizeof(float)*num_defects*num_products);

    RandomInit(product_posteriors, num_defects, num_products, num_defects);
    memcpy(product_posteriors_cpu, product_posteriors, sizeof(float) * num_defects*num_products);
    printf("Succesfully Initialized Product Posteriors... \n");

    /* Allocate for Word Posteriors */
    word_posteriors = (float *) malloc(sizeof(float)*vocab_size*num_defects);
    word_posteriors_cpu = (float *) malloc(sizeof(float)*vocab_size*num_defects);
    /* initialize uniformly random numbers */
    for (int i=0; i < num_defects; i++){
        for(int j=0; j < vocab_size; j++){
            word_posteriors[INDX(i,j, num_defects)] = 1/(float)vocab_size;
        }

    }
    memcpy(word_posteriors_cpu, word_posteriors, sizeof(float) * num_defects*vocab_size);




//
//    runEM_CPU( TFDF, num_complaints,
//           vocab_size,  company_vec,
//           num_companies, issue_vec,
//           num_issues, product_vec,
//           num_products, num_defects,
//           defect_priors_cpu, defect_posteriors_cpu,
//           company_posteriors_cpu, issue_posteriors_cpu,
//           product_posteriors_cpu, word_posteriors_cpu,
//           tol, max_iter, (const char *) "LOG_CPU.txt", (const char *) "RESULTS_CPU.txt"
//    );


    runEM( TFDF, num_complaints,
           vocab_size,  company_vec,
           num_companies, issue_vec,
           num_issues, product_vec,
           num_products, num_defects,
           defect_priors, defect_posteriors,
           company_posteriors, issue_posteriors,
           product_posteriors, word_posteriors,
           tol, max_iter, log_file_name, results_file_name
    );






}