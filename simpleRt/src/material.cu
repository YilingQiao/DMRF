#include "material.cuh"
#include "hittable.cuh"
#include "random.cuh"

__device__ bool material::scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
	ray& scattered, curandState *pixel_random_seed) {
	switch (type) {
	case(type_lambertian):
		return ((lambertian*)this)->scatter(r_in, rec, attenuation, scattered, pixel_random_seed);
		break;
	case(type_metal):
		return ((metal*)this)->scatter(r_in, rec, attenuation, scattered, pixel_random_seed);
		break;
	case(type_dielectric):
		return ((dielectric*)this)->scatter(r_in, rec, attenuation, scattered, pixel_random_seed);
		break;
	case(type_lightsource):
		return ((lightsource*)this)->scatter(r_in, rec, attenuation, scattered, pixel_random_seed);
		break;
	}
	return false;
}

__host__ __device__ lambertian::lambertian(const vec3& a) : albedo(a) {
	type = type_lambertian;
}

__host__ __device__ lightsource::lightsource(const vec3& a) : albedo(a) {
	type = type_lightsource;
}

__host__ __device__ metal::metal(const vec3& a, float f) : albedo(a) {
	type = type_metal;
	if (f < 1) fuzz = f; else fuzz = 1;
}

__host__ __device__ dielectric::dielectric(const vec3& a, float ri) : albedo(a), ref_idx(ri) {
	type = type_dielectric;
}

__device__ bool lambertian::scatter(const ray& r_in, const hit_record& rec,
	vec3& attenuation, ray& scattered, curandState *pixel_random_seed) {
	vec3 neww_direction;
	do {
		neww_direction = rec.normal + random_in_unit_sphere(pixel_random_seed);
	} while (neww_direction.length() < 0.0000001f);
	scattered = ray(rec.p, neww_direction);

	// vec3 target = rec.p + rec.normal + random_in_unit_sphere(pixel_random_seed);
	// scattered = ray(rec.p, target - rec.p);
	attenuation = albedo;
	return true;
}

__device__ bool metal::scatter(const ray& r_in, const hit_record& rec,
	vec3& attenuation, ray& scattered, curandState *pixel_random_seed) {

	// two-sided metal
	vec3 normal = rec.normal;
	if (dot(r_in.direction(), rec.normal) > 0)
		normal = - rec.normal;

	vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
	scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(pixel_random_seed));
	attenuation = albedo;
	return (dot(scattered.direction(), normal) > 0);
}

__device__ bool lightsource::scatter(const ray& r_in, const hit_record& rec,
	vec3& attenuation, ray& scattered, curandState *pixel_random_seed) {

	float fuzz = 0.;
	// two-sided metal
	vec3 normal = rec.normal;
	if (dot(r_in.direction(), rec.normal) > 0)
		normal = - rec.normal;

	vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
	scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(pixel_random_seed));
	attenuation = albedo;
	return (dot(scattered.direction(), normal) > 0);
}


__device__ bool dielectric::scatter(const ray& r_in, const hit_record& rec,
	vec3& attenuation, ray& scattered, curandState *pixel_random_seed) {
	vec3 outward_normal;
	vec3 reflected = reflect(r_in.direction(), rec.normal);
	float ni_over_nt;
	//attenuation = vec3(1.0, 1.0, 1.0);
	attenuation = albedo;
	vec3 refracted;

	float reflect_prob;
	float cosine;

	if (dot(r_in.direction(), rec.normal) > 0) {
		outward_normal = -rec.normal;
		ni_over_nt = ref_idx;
		cosine = ref_idx * dot(r_in.direction(), rec.normal)
			/ r_in.direction().length();
	}
	else {
		outward_normal = rec.normal;
		ni_over_nt = 1.0 / ref_idx;
		cosine = -dot(r_in.direction(), rec.normal)
			/ r_in.direction().length();
	}

	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
		reflect_prob = schlick(cosine, ref_idx);
	}
	else {
		reflect_prob = 1.0;
	}
	// if (reflect_prob > 0.01)
	// 	printf("!%f\n", reflect_prob);

	if (ref_idx < 1.0001f)
		reflect_prob = 0.;

	float rand = curand_uniform(pixel_random_seed);
	if (rand < reflect_prob) {
		scattered = ray(rec.p, reflected);
		// printf("-- reflected! %f %f\n", rand, reflect_prob);
	}
	else {
		scattered = ray(rec.p, refracted);
	}

	return true;
}