#include "cuda_error_handle.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

//Check the error code when recalling a CUDA function
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		std::cerr << cudaGetErrorString((cudaError_t)(static_cast<unsigned int>(result))) << std::endl;
		cudaDeviceReset();
		exit(99);
	}
}