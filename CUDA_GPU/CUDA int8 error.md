## CUDA int8 细节
-------------------
CUDA里的int8要求计算（还是Computation type == CUDA_R_32I的时候）时向量维度必须是4的倍数，em。。
否则报错： ** On entry to GEMM_EX  parameter number 9 had an illegal value
number9指向的是 a的参数类型
下面是封装的int8 tensor乘法实例
耗时6小时

```C
  if (dataTypeA == X_INT8 && dataTypeB == X_INT8 && dataTypeC == X_INT) {
        //ShowNTErrors("TO DO!");
        int alpha2 = (int)alpha;
        int beta2 = (int)beta;
		printf("here second x_int\n");
		/*
			CUDA requires that the dimension of two tensor( lda, ldb ) should be multiples of 4.
			details in https://devtalk.nvidia.com/default/topic/999101/about-cublasgemm-int8-support/
		*/
		if (mb % 4 || ma % 4) {
			ShowNTErrors("mb, ma( lda, ldb ) should be multiples of 4!");
			return;
		}

    if (transposedA == X_NOTRANS && transposedB == X_NOTRANS) 
			cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const int8_t*)b, CUDA_R_8I, mb, (const int8_t*)a, CUDA_R_8I, ma, &beta2, (int*)c, CUDA_R_32I, mc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
		else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const int8_t*)b, CUDA_R_8I, mb, (const int8_t*)a, CUDA_R_8I, ma, &beta2, (int*)c, CUDA_R_32I, mc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasGemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const int8_t*)b, CUDA_R_8I, mb, (const int8_t*)a, CUDA_R_8I, ma, &beta2, (int*)c, CUDA_R_32I, mc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasGemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const int8_t*)b, CUDA_R_8I, mb, (const int8_t*)a, CUDA_R_8I, ma, &beta2, (int*)c, CUDA_R_32I, mc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
    }
```
