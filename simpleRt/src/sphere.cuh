#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "hittable.cuh"
#include "cuda_error_handle.cuh"

class sphere : public hittable {
public:
	__host__ __device__ sphere() {}
	__host__ __device__ sphere(vec3 cen, float r, material *m);
	__host__ __device__ bool bounding_box(float t0, float t1, aabb& box);
	__device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ vec3 sample(curandState *local_rand_state) const;
	
	//vec3 center;
	float radius;
	//material *mat_ptr;

	virtual hittable* copy_to_gpu() const {
		sphere *device,*tmp;
		tmp = new sphere();
		memcpy(tmp, this, sizeof(sphere));
		tmp->mat_ptr = this->mat_ptr->copy_to_gpu();

		checkCudaErrors(cudaMalloc((void**)&device, sizeof(sphere)));
		checkCudaErrors(cudaMemcpy(device, tmp, sizeof(sphere), cudaMemcpyHostToDevice));

		return device;
	}
};

#endif