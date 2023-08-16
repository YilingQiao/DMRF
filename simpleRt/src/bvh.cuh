#ifndef BVH_CUH
#define BVH_CUH

#include <stdio.h>
#include <iostream>
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "aabb.cuh"
#include "cuda_error_handle.cuh"

class bvh_node : public hittable {
public:
	bvh_node();

	bvh_node(hittable **l, int L, int R, float time0, float time1, int depth);

	virtual hittable* copy_to_gpu() const {
		bvh_node *device, *tmp;
		tmp = new bvh_node();
		
		tmp->mat_ptr = NULL;
		tmp->box = this->box;
		tmp->center = this->center;
		//memcpy(tmp, this, sizeof(this));
		//if(this->mat_ptr!=NULL)
		//	tmp->mat_ptr = this->mat_ptr->copy_to_gpu();
		if(this->obj_list!=NULL)
			tmp->obj_list = (hittable_list*)this->obj_list->copy_to_gpu();

		if (this->left != NULL)
			tmp->left = this->left->copy_to_gpu();
		if (this->right != NULL)
			tmp->right = this->right->copy_to_gpu();
		
		checkCudaErrors(cudaMalloc((void**)&device, sizeof(bvh_node)));
		checkCudaErrors(cudaMemcpy(device, tmp, sizeof(bvh_node), cudaMemcpyHostToDevice));

		return device;
	}

	__device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__host__ __device__ bool bounding_box(float t0, float t1, aabb& b);

	__device__ void refit();

 	__device__ vec3 sample(curandState *local_rand_state) const;
	__device__ bool test(aabb &rec);

	hittable *left;
	hittable *right;

	hittable_list *obj_list;

	aabb box;
};

__global__ void visit(bvh_node **root, int *result);

#endif // !BVH_H
