#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>
#include "cuda_error_handle.cuh"
#include "vec3.cuh"

//Initialize the random seed for each pixel
__global__ void init_pixel_random_seed(int max_x, int max_y, curandState *random_seed);

//Generate a vector in an unit sphere according to the pixel's random seed
__device__ vec3 random_in_unit_sphere(curandState *random_seed);

//Generate a vector in an unit disk according to the pixel's random seed
__device__ vec3 random_in_unit_disk(curandState *random_seed);

#endif // !RANDOM_CUH
