#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "hittable.cuh"
#include "random.cuh"

class hit_record;

enum material_type {
	type_dielectric,
	type_lambertian,
	type_metal,
	type_lightsource,
};

class material {
public:
	__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
		ray& scattered, curandState *pixel_random_seed);

	virtual material* copy_to_gpu() const = 0;

	material_type type;
};

class lightsource : public material {
public:
	__host__ __device__ lightsource(const vec3& a);

	__device__ bool scatter(const ray& r_in, const hit_record& rec,
		vec3& attenuation, ray& scattered, curandState *pixel_random_seed);

	virtual material* copy_to_gpu() const {
		material *device;
		checkCudaErrors(cudaMalloc((void**)&device, sizeof(lightsource)));
		checkCudaErrors(cudaMemcpy(device, this, sizeof(lightsource), cudaMemcpyHostToDevice));
		return device;
	}
	vec3 albedo;
};

class lambertian : public material {
public:
	__host__ __device__ lambertian(const vec3& a);

	__device__ bool scatter(const ray& r_in, const hit_record& rec,
		vec3& attenuation, ray& scattered, curandState *pixel_random_seed);

	virtual material* copy_to_gpu() const {
		material *device;
		checkCudaErrors(cudaMalloc((void**)&device, sizeof(lambertian)));
		checkCudaErrors(cudaMemcpy(device, this, sizeof(lambertian), cudaMemcpyHostToDevice));
		return device;
	}
	vec3 albedo;
};

class metal : public material {
public:
	__host__ __device__ metal(const vec3& a, float f);

	__device__ bool scatter(const ray& r_in, const hit_record& rec,
		vec3& attenuation, ray& scattered, curandState *pixel_random_seed);

	virtual material* copy_to_gpu() const {
		material *device;
		checkCudaErrors(cudaMalloc((void**)&device, sizeof(metal)));
		checkCudaErrors(cudaMemcpy(device, this, sizeof(metal), cudaMemcpyHostToDevice));
		return device;
	}

	vec3 albedo;
	float fuzz;
};

class dielectric : public material {
public:
	__host__ __device__ dielectric(const vec3& a, float ri);

	__device__ bool scatter(const ray& r_in, const hit_record& rec,
		vec3& attenuation, ray& scattered, curandState *pixel_random_seed);

	virtual material* copy_to_gpu() const {
		material *device;
		checkCudaErrors(cudaMalloc((void**)&device, sizeof(dielectric)));
		checkCudaErrors(cudaMemcpy(device, this, sizeof(dielectric), cudaMemcpyHostToDevice));
		return device;
	}

	float ref_idx;
	vec3 albedo;
};

#endif // !MATERIAL_CUH
