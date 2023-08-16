#include "triangle.cuh"
#include "vec3.cuh"
#include "aabb.cuh"

__host__ __device__ triangle::triangle(vec3 vertexone, vec3 vertextwo, vec3 vertexthree,
	//vec3 vn1, vec3 vn2, vec3 vn3,
	material *mat_ptr) {
	type = type_triangle;

	v1 = vertexone;
	v2 = vertextwo;
	v3 = vertexthree;
	//this->vn1 = cross(v3 - v1,v2-v1);
	this->vn1 = cross(v2 - v1, v3 - v1);
	this->vn1 /= this->vn1.length();
	this->vn2 = this->vn1;
	this->vn3 = this->vn1;
	/*this->vn1 = vn1;
	this->vn2 = vn2;
	this->vn3 = vn3;*/
	center = (v1 + v2 + v3) / 3;
	normal = cross(v2 - v1, v3 - v1);
	this->mat_ptr = mat_ptr;

}

__host__ __device__ bool triangle::bounding_box(float t0, float t1, aabb& box){
	box._min[0] = ffmin(ffmin(v1[0], v2[0]), v3[0]);
	box._max[0] = ffmax(ffmax(v1[0], v2[0]), v3[0]);

	box._min[1] = ffmin(ffmin(v1[1], v2[1]), v3[1]);
	box._max[1] = ffmax(ffmax(v1[1], v2[1]), v3[1]);

	box._min[2] = ffmin(ffmin(v1[2], v2[2]), v3[2]);
	box._max[2] = ffmax(ffmax(v1[2], v2[2]), v3[2]);

	return true;
}

//Reference: https://github.com/tylermorganwall/rayrender/
__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const{
	vec3 edge1 = v2 - v1;

	vec3 edge2 = v3 - v1;
	vec3 pvec = cross(r.direction(), edge2);

	float det = dot(pvec, edge1);



	// no culling

	if (std::fabs(det) < 1E-9) {
		// printf("return false triangle\n");
		return(false);

	}

	float invdet = 1.0 / det;

	vec3 tvec = r.origin() - v1;

	float u = dot(pvec, tvec) * invdet;

	if (u < 0.0 || u > 1.0) {

		return(false);

	}



	vec3 qvec = cross(tvec, edge1);

	float v = dot(qvec, r.direction()) * invdet;

	if (v < 0 || u + v > 1.0) {

		return(false);

	}

	float t = dot(qvec, edge2) * invdet;



	if (t < t_min || t > t_max) {

		return(false);

	}

	float w = 1 - u - v;

	rec.t = t;

	rec.p = r.point_at_parameter(t);

	//rec.u = u;

	//rec.v = v;

	rec.normal = w * vn1 + u * vn2 + v * vn3;

	rec.mat_ptr = mat_ptr;

	return(true);

}

__device__ vec3 triangle::sample(curandState *local_rand_state) const {
  vec3 p;
  // float a = 0.f;
  // float b = 0.f;
  float a = curand_uniform(local_rand_state);
  float b = curand_uniform(local_rand_state);
  p = v1 + a * (v2 - v1) + b * (v3 - v1);
  return p;
}
