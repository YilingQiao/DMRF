#ifndef HITTABLELIST_CUH
#define HITTABLELIST_CUH

#include "hittable.cuh"
#include "aabb.cuh"
#include "cuda_error_handle.cuh"

//copy objs to gpu
hittable** copy_objs_to_gpu(hittable **obj_list_host, int list_size);

class hittable_list : public hittable {
public:
	virtual hittable* copy_to_gpu() const {
		hittable_list *tmp, *device;
		checkCudaErrors(cudaMalloc((void**)&device, sizeof(hittable_list)));
		tmp = new hittable_list();
		memcpy(tmp, this, sizeof(hittable_list));
		tmp->list = copy_objs_to_gpu(this->list, this->list_size);
		checkCudaErrors(cudaMemcpy(device, tmp, sizeof(hittable_list), cudaMemcpyHostToDevice));
		return device;
	}

	__host__ __device__ hittable_list() {
		type = type_hittable_list;
	}
	__host__ __device__ hittable_list(hittable **l, int n) { 
		type = type_hittable_list;
		list = l; 
		list_size = n; 
	}
	__device__ bool hit(
		const ray& r, float tmin, float tmax, hit_record& rec) const;
	__host__ __device__ bool bounding_box(float t0, float t1, aabb& box);
	hittable **list;
	int list_size;
	__device__ vec3 sample(curandState *local_rand_state) const;
};

#endif