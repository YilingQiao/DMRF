#ifndef CUDA_ERROR_HANDLE_CUH
#define CUDA_ERROR_HANDLE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

//Check the error code when recalling a CUDA function
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
#endif // ! 

#ifndef DEF_CHECKCUDAERRORS
#define DEF_CHECKCUDAERRORS
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#endif