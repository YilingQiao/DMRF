#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ray.cuh"
#include "vec3.cuh"
#include "aabb.cuh"
#include "random.cuh"
#include "material.cuh"
#include <curand_kernel.h>

class material;

class hit_record {
public:
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

enum hittable_type {
	type_hittable_list,
	type_sphere,
	type_triangle,
	type_bvh_node
};

class hittable {
public:
	__device__ bool hit(
		const ray& r, float t_min, float t_max, hit_record& rec) const;
	// __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	
	__host__ __device__ bool bounding_box(float t0, float t1, aabb& box);

	virtual hittable* copy_to_gpu() const = 0;
  __device__ vec3 sample(curandState *local_rand_state) const;

	vec3 center;
	material *mat_ptr;
	hittable_type type;
};

#endif