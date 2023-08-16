#include "random.cuh"
#include <curand_kernel.h>
#include "cuda_error_handle.cuh"
#include "vec3.cuh"

//Initialize the random seed for each pixel
__global__ void init_pixel_random_seed(int max_x, int max_y, curandState *random_seed) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_idx = i * max_y + j;
	curand_init(1999, pixel_idx, 0, &random_seed[pixel_idx]);
}

//Generate a vector in an unit sphere according to the pixel's random seed
__device__ vec3 random_in_unit_sphere(curandState *random_seed) {
	//vec3 p;
	/*do {
		p = 2.0*vec3(curand_uniform(random_seed), curand_uniform(random_seed), curand_uniform(random_seed)) - vec3(1, 1, 1);
	} while (p.squared_length() >= 1.0);*/
	float xx = curand_normal(random_seed);
	float yy = curand_normal(random_seed);
	float zz = curand_normal(random_seed);
	float fenmu = sqrt(xx*xx + yy * yy + zz * zz);
	return vec3(xx/fenmu,yy/fenmu,zz/fenmu);
}

//Generate a vector in an unit disk according to the pixel's random seed
__device__ vec3 random_in_unit_disk(curandState *random_seed) {
	/*vec3 p;
	do {
		p = 2.0*vec3(curand_uniform(random_seed), curand_uniform(random_seed), 0.0f) - vec3(1, 1, 0);
	} while (p.squared_length() >= 1.0);
	return p;*/
	float xx = curand_normal(random_seed);
	float yy = curand_normal(random_seed);
	float fenmu = sqrt(xx*xx + yy * yy);
	return vec3(xx / fenmu, yy / fenmu, 0.0f);
}
