#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
using namespace std;

#define DEBUG 
#ifdef DEBUG
#define Pr(f,...) fprintf(stderr,f,##__VA_ARGS__),fflush(stdout)
#else
#define Pr(f,...) ;
#endif

#define CHECK_CUDA(func)                                                       \
{                                                                              \
	cudaError_t status = (func);                                               \
	if (status != cudaSuccess) {                                               \
		printf("CUDA API failed at line %d with error: %s (%d)\n",             \
				__LINE__, cudaGetErrorString(status), status);                  \
		return EXIT_FAILURE;                                                   \
	}                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
	cusparseStatus_t status = (func);                                          \
	if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
		printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
				__LINE__, cusparseGetErrorString(status), status);              \
		return EXIT_FAILURE;                                                   \
	}                                                                          \
}

#define CHECK_CUDARUNTIME(func, str)\
{\
    float mean = 0, mx = 0, mn = 1e9, s=0; \
    std::vector<float> v;\
    const int K = 10;\
    for (int i = 0;i < K;++i) {\
       cudaEvent_t start, stop;\
       cudaEventCreate(&start);\
       cudaEventCreate(&stop);\
       cudaEventRecord(start);\
       float ms = 0;\
       func\
       cudaDeviceSynchronize();\
       cudaEventRecord(stop);\
       cudaEventSynchronize(stop);\
       cudaEventElapsedTime(&ms, start, stop);\
       if (i>0){\
           mean += ms; \
           mx = max(mx, ms); \
           mn = min(mn, ms);\
           v.push_back(ms);\
       }\
    }\
    mean /= K-1;\
    for (int i=0;i<K-1;++i)s+=(v[i]-mean)*(v[i]-mean);\
    s=sqrt(s/(K-2))/mean*100.0;\
    printf(str",%f,%.2f%%,%.2f%%,%.2f%%\n", \
        mean, (mx-mean)/mean*100, (mean-mn)/mean*100, s); \
} \

double H_n(long long n) { 
    if (n == 0) return 0;
    double ans = 0;
    if (n < (int)1e7) {
        for (int i = 1;i <= n;++i) ans += 1.0/i;
            return ans;
    } else {
        return 0.57721566490153286060651209 + log(n); 
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 7) {
        printf("usage:spmv_csr n m sparsity\n");
        return 0; 
    }
    const int A_num_rows      = atoi(argv[1]);
    const int A_num_cols      = atoi(argv[2]);
    const double sparsity = atof(argv[3]); 
    const int verbose = argc > 4 ? atoi(argv[4]) : 0; 
    const int gemm_on = argc > 5 ? atoi(argv[5]) : 0; 
    const int spmm_on = argc > 6 ? atoi(argv[6]) : 0; 
    freopen("newResult.txt", "aw", stdout);

    int A_nnz = 0; 
    int generateTimes = 1.0 * A_num_rows * sparsity * A_num_cols; 
    Pr("Anticipated nnz = %d[ approx memory usage : %f GB ] \n", generateTimes, 1.0 * generateTimes * 5 * 4 / 1e9); 
    double factor = (-H_n(1ll * A_num_rows * A_num_cols - generateTimes) + H_n(1ll * A_num_rows * A_num_cols));
    generateTimes = 1.0 * A_num_rows * A_num_cols * factor;
    Pr("Anticipated factor = %f\n", factor); 
    Pr("Anticipated generateTimes = %d\n", generateTimes); 

    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_val(0.0, 1.0); 
    std::uniform_int_distribution<> dis_row(0, A_num_rows - 1);
    std::uniform_int_distribution<> dis_col(0, A_num_cols - 1);

    vector<int> colv(generateTimes), rowv(generateTimes), id(generateTimes);
    const float alpha = 1.0f, beta = 0.0f; 

    for (int i = 0; i < generateTimes; ++i) {
        colv[i] = dis_col(gen);
        rowv[i] = dis_row(gen);
        id[i] = i;
    }


     __gnu_parallel::sort(id.begin(), id.end(), 
        [&](int a, int b) { return rowv[a] == rowv[b] ? (colv[a] < colv[b]) : (rowv[a] < rowv[b]); });
    id.erase(unique(id.begin(), id.end(), [&](int a, int b) { return rowv[a] == rowv[b] && colv[a] == colv[b]; }), id.end()); 
    A_nnz = id.size(); 
    printf("A : (%d,%d):{nnz:%d,sparsity=%f%%}\n", 
        A_num_rows, A_num_cols, A_nnz, 100.0*A_nnz/A_num_cols/A_num_rows); 
    int *hA_columns = new int [A_nnz]; 
    int *hA_rows = new int [A_nnz]; 
    float *hA_values = new float [A_nnz]; 
    for (int i = 0; i < A_nnz; ++i) hA_values[i] = dis_val(gen); 
    for (int i = 0; i < A_nnz; ++i) hA_columns[i] = colv[id[i]]; 
    for (int i = 0; i < A_nnz; ++i) hA_rows[i] = rowv[id[i]];

    float     *hX = new float[A_num_cols]; 
    float     *hY = new float[A_num_rows]; 
    for (int i = 0;i < A_num_cols;++i) hX[i] = dis_val(gen); 
    for (int i = 0;i < A_num_rows;++i) hY[i] = 0; 

    if (verbose) {
        printf("Testing with sparsity (%d,%d,%f) \n", A_num_rows, A_num_cols, sparsity); 
        printf("hA_columns : "); for (int i = 0;i < A_nnz;++i) printf("%d ", hA_columns[i]); puts(""); 
        printf("hA_rows "); for (int i = 0; i < A_nnz; ++i) printf("%d ", hA_rows[i]); puts(""); 
        printf("hA_values : "); for (int i = 0;i < A_nnz;++i) printf("%f ", hA_values[i]); puts(""); 
        printf("hX_values : "); for (int i = 0;i < A_num_cols;++i) printf("%f ", hX[i]); puts(""); 
    }
	

    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns, *dA_rows, *dA_cscOffsets, *dA_cscRows;
    float *dA_values, *dX, *dY, *dA_cscValues;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_cscOffsets, (A_num_cols + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_cscRows, (A_nnz) * sizeof(int)) )
	CHECK_CUDA( cudaMalloc((void**) &dA_rows,    A_nnz * sizeof(int))         )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dA_cscValues,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
    	cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_rows, hA_rows, A_nnz * sizeof(int),
        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
    	cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
    	cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(float),
    	cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matCOO;
	cusparseSpMatDescr_t matCSR;
    cusparseSpMatDescr_t matCSC;
    cusparseDnVecDescr_t vecX, vecY;
	CHECK_CUSPARSE( cusparseCreate(&handle) )

	CHECK_CUDARUNTIME(
			CHECK_CUSPARSE( cusparseXcoo2csr(handle, dA_rows, A_nnz, A_num_rows, dA_csrOffsets, CUSPARSE_INDEX_BASE_ZERO)), "COO2CSR"
	)
    CHECK_CUDARUNTIME(
            CHECK_CUSPARSE( cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_num_rows, 
                    dA_rows, CUSPARSE_INDEX_BASE_ZERO)), "CSR2COO"
    )


    {
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(handle, A_num_rows, A_num_cols, A_nnz, dA_values, dA_csrOffsets, dA_columns, 
            dA_cscValues, dA_cscOffsets, dA_cscRows, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize))
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
        CHECK_CUDARUNTIME(
            CHECK_CUSPARSE( cusparseCsr2cscEx2(handle, A_num_rows, A_num_cols, A_nnz, dA_values, dA_csrOffsets, dA_columns, 
                dA_cscValues, dA_cscOffsets, dA_cscRows, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
                CUSPARSE_CSR2CSC_ALG1, dBuffer)) , "CSR2CSC_ALG1"
            )
    }
    {
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(handle, A_num_rows, A_num_cols, A_nnz, dA_values, dA_csrOffsets, dA_columns, 
            dA_cscValues, dA_cscOffsets, dA_cscRows, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &bufferSize))
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
        CHECK_CUDARUNTIME(
            CHECK_CUSPARSE( cusparseCsr2cscEx2(handle, A_num_rows, A_num_cols, A_nnz, dA_values, dA_csrOffsets, dA_columns, 
                dA_cscValues, dA_cscOffsets, dA_cscRows, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
                CUSPARSE_CSR2CSC_ALG2, dBuffer)) , "CSR2CSC_ALG2"
            )
        CHECK_CUDA( cudaFree(dBuffer) )
    }

   // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    {
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        CHECK_CUSPARSE( cusparseCreateCoo(&matCOO, A_num_rows, A_num_cols, A_nnz,
            dA_rows, dA_columns, dA_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F) )
    // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
           handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
           &alpha, matCOO, vecX, &beta, vecY, CUDA_R_32F,
           CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
        CHECK_CUDARUNTIME(
          CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
           &alpha, matCOO, vecX, &beta, vecY, CUDA_R_32F,
           CUSPARSE_MV_ALG_DEFAULT, dBuffer) ), "SpMV_COO"
          )
        CHECK_CUDA( cudaFree(dBuffer) )
        CHECK_CUDA( cudaDeviceSynchronize() )
    }
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
        cudaMemcpyDeviceToHost) )
    if (verbose) {
        printf("hY: "); 
        for (int i = 0; i < A_num_rows; i++) {
            printf("%f ", hY[i]); 
        }
        puts(""); 
    }
    {
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        CHECK_CUSPARSE( cusparseCreateCsr(&matCSR, A_num_rows, A_num_cols, A_nnz,
            dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        )
    // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matCSR, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
        CHECK_CUDARUNTIME(
            CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matCSR, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_MV_ALG_DEFAULT, dBuffer) ), "SpMV_CSR"
            )
        CHECK_CUDA( cudaFree(dBuffer) )
        CHECK_CUDA( cudaDeviceSynchronize() )
    }
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
        cudaMemcpyDeviceToHost) )
    if (verbose) {
        printf("hY: "); 
        for (int i = 0; i < A_num_rows; i++) {
            printf("%f ", hY[i]); 
        }
        puts(""); 
    }
    CHECK_CUDA( cudaDeviceSynchronize() )

    {
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        CHECK_CUSPARSE( cusparseCreateCsc(&matCSC, A_num_rows, A_num_cols, A_nnz,
            dA_cscOffsets, dA_cscRows, dA_cscValues,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F) )
    // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matCSC, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
        CHECK_CUDARUNTIME(
            CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matCSC, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_MV_ALG_DEFAULT, dBuffer) ), "SpMV_CSC"
            )
        CHECK_CUDA( cudaFree(dBuffer) )
        CHECK_CUDA( cudaDeviceSynchronize() )
    }
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
        cudaMemcpyDeviceToHost) )
    if (verbose) {
        printf("hY: "); 
        for (int i = 0; i < A_num_rows; i++) {
            printf("%f ", hY[i]); 
        }
        puts(""); 
    }

    //SpMM
    if (spmm_on) {
        int B_size = A_num_cols * A_num_cols; 
        int C_size = A_num_rows * A_num_cols; 
        float *hB = new float[A_num_cols * A_num_cols]; 
        float *hC = new float[A_num_rows * A_num_cols]; 
        for (int i = 0; i < B_size; ++i) {
            hB[i] = dis_val(gen); 
        }
        if (verbose) {
            for (int i = 0; i < B_size; ++i) {
                printf("hB[%d] = %f\n", i, hB[i]);
            }
        }
        for (int i = 0;i < C_size; ++i) {
            hC[i] = .0; 
        }
        int B_num_rows = A_num_cols, B_num_cols = A_num_cols; 
        float *dB, *dC; 
        CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
        CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )
        CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
         cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
           cudaMemcpyHostToDevice) )

        cusparseDnMatDescr_t matB, matC;
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;

        // Create dense matrix B
        CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, B_num_rows, dB,
            CUDA_R_32F, CUSPARSE_ORDER_ROW) )
        // Create dense matrix C
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, A_num_cols, A_num_rows, dC,
            CUDA_R_32F, CUSPARSE_ORDER_ROW) )
        {
            CHECK_CUSPARSE( cusparseSpMM_bufferSize(
             handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matCSR, matB, &beta, matC, CUDA_R_32F,
             CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
            CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
            CHECK_CUDARUNTIME(CHECK_CUSPARSE( cusparseSpMM(handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matCSR, matB, &beta, matC, CUDA_R_32F,
             CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) ), "CSR_SPMM")
            CHECK_CUDA(cudaFree(dBuffer))
        }
        {
            CHECK_CUSPARSE( cusparseSpMM_bufferSize(
               handle,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, matCOO, matB, &beta, matC, CUDA_R_32F,
               CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
            CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
            CHECK_CUDARUNTIME(CHECK_CUSPARSE( cusparseSpMM(handle,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, matCOO, matB, &beta, matC, CUDA_R_32F,
               CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) ), "COO_SPMM")
            CHECK_CUDA(cudaFree(dBuffer))
        }

        if (verbose) {
            printf("SPMM mat C:\n");
            CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
               cudaMemcpyDeviceToHost) )
            for (int i = 0; i < A_num_rows; i++) {
                for (int j = 0; j < B_num_cols; j++) {
                    printf("%f ", hC[i * B_num_cols + j]);
                }
                puts(""); 
            }
        }
        CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
        CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
        delete[] hB;
        delete[] hC;
    } 

    // cuSparse SpGEMM
    int *dC_csrOffsets, *dC_columns; float *dC_values;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    // allocate C offsets
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, A_num_cols, A_num_rows, A_nnz,
                                      dA_cscOffsets, dA_cscRows, dA_cscValues,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, A_num_rows, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )
    cudaDataType        computeType = CUDA_R_32F;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;

    if (gemm_on) {
        CHECK_CUSPARSE(
            cusparseSpGEMM_workEstimation(handle, opA, opB,
              &alpha, matA, matB, &beta, matC,
              computeType, CUSPARSE_SPGEMM_DEFAULT,
              spgemmDesc, &bufferSize1, NULL) )
        Pr("BufferSize1 is arppoximately %f MB\n", 1.0 * bufferSize1 / 1024 / 1024);
        CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
        CHECK_CUSPARSE(
            cusparseSpGEMM_workEstimation(handle, opA, opB,
              &alpha, matA, matB, &beta, matC,
              computeType, CUSPARSE_SPGEMM_DEFAULT,
              spgemmDesc, &bufferSize1, dBuffer1) )

        CHECK_CUSPARSE(
            cusparseSpGEMM_compute(handle, opA, opB,
             &alpha, matA, matB, &beta, matC,
             computeType, CUSPARSE_SPGEMM_DEFAULT,
             spgemmDesc, &bufferSize2, NULL) )
        Pr("BufferSize2 is arppoximately %f MB\n", 1.0 * bufferSize2 / 1024 / 1024);
        CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

        CHECK_CUDARUNTIME(CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
         &alpha, matA, matB, &beta, matC,
         computeType, CUSPARSE_SPGEMM_DEFAULT,
         spgemmDesc, &bufferSize2, dBuffer2) ), 
        "SpGEMM")
        int64_t C_num_rows1, C_num_cols1, C_nnz1;
        CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
           &C_nnz1) )
        CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
        CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )
        CHECK_CUSPARSE(
            cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

        CHECK_CUSPARSE(
            cusparseSpGEMM_copy(handle, opA, opB,
                &alpha, matA, matB, &beta, matC,
                computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

        if (verbose) {
            int   *hC_csrOffsets_tmp = new int[A_num_rows + 1];
            int   *hC_columns_tmp = new int[A_nnz];
            float *hC_values_tmp = new float[A_nnz];
            CHECK_CUDA( cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
               (A_num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost) )
            CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz1 * sizeof(int),
               cudaMemcpyDeviceToHost) )
            CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, C_nnz1 * sizeof(float),
               cudaMemcpyDeviceToHost) )
            printf("C_csrOffsets:");
            for (int i = 0; i < A_num_rows + 1; i++) {
                printf("%d ", hC_csrOffsets_tmp[i]); 
            }
            puts(""); 
            printf("C_columns:");
            for (int i = 0; i < C_nnz1; i++) {
                printf("%d ", hC_columns_tmp[i]); 
            }
            puts(""); 
            printf("C_values:");
            for (int i = 0; i < C_nnz1; i++) {
                printf("%f ", hC_values_tmp[i]); 
            }
            puts(""); 
            delete[] hC_csrOffsets_tmp;
            delete[] hC_columns_tmp;
            delete[] hC_values_tmp;
        }
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matCOO) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matCSR) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matCSC) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    CHECK_CUDA( cudaFree(dC_csrOffsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_cscOffsets) )
    CHECK_CUDA( cudaFree(dA_rows) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    delete[] hA_columns;
    delete[] hA_rows;
    delete[] hA_values;
    delete[] hX;
    delete[] hY;

    return EXIT_SUCCESS;
}
