#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class ray
{
public:
	__device__ ray() {}
	__device__ ray(const vec3& a, const vec3& b, float ti = 0.f) { A = a; B = b; _time = ti; }
	__device__ vec3 origin() const { return A; }
	__device__ vec3 direction() const { return B; }
  	__device__ float time() const { return _time ; }
	__device__ vec3 point_at_parameter(float t) const { return A + t * B; }

	vec3 A;
	vec3 B;
	float _time;
};

#endif