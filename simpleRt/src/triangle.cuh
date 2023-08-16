#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "hittable.cuh"
#include "cuda_error_handle.cuh"

class triangle :public hittable {

public:

	vec3 v1, v2, v3;
	vec3 normal;
	vec3 vn1, vn2, vn3;
	//vec3 center;
	__host__ __device__ triangle() {};

	__host__ __device__ triangle(vec3 vertexone, vec3 vertextwo, vec3 vertexthree,
		//vec3 vn1, vec3 vn2, vec3 vn3,
		material *mat_ptr);

	__host__ __device__ bool bounding_box(float t0, float t1, aabb& box);

	//Reference: https://github.com/tylermorganwall/rayrender/
	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

	virtual hittable* copy_to_gpu() const {
		hittable *device, *tmp;
		tmp = new triangle();
		memcpy(tmp, this, sizeof(triangle));
		tmp->mat_ptr = this->mat_ptr->copy_to_gpu();

		checkCudaErrors(cudaMalloc((void**)&device, sizeof(triangle)));
		checkCudaErrors(cudaMemcpy(device, tmp, sizeof(triangle), cudaMemcpyHostToDevice));

		return device;
	}
	__device__ vec3 sample(curandState *local_rand_state) const;
};
#endif