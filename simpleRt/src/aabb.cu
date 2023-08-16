#include "aabb.cuh"
#include "vec3.cuh"
#include "ray.cuh"

__host__ __device__ aabb::aabb() {}
	
__host__ __device__ aabb::aabb(const vec3& a, const vec3& b) { 
	_min = a; _max = b; 
}
	
__host__ __device__ vec3 aabb::min() const {
	return _min; 
}

__host__ __device__ vec3 aabb::max() const { 
	return _max; 
}

	
// __device__ bool aabb::hit(const ray& r, float tmin, float tmax) {
// 	for (int a = 0; a < 3; a++) {
// 		float invD = 1.0f / r.direction()[a];
// 		float t0 = (min()[a] - r.origin()[a]) * invD;
// 		float t1 = (max()[a] - r.origin()[a]) * invD;
// 			if (invD < 0.0f) {
// 			float tmp = t0;
// 			t0 = t1;
// 			t1 = tmp;
// 		}
// 		//std::swap(t0, t1);
// 		tmin = t0 > tmin ? t0 : tmin;
// 		tmax = t1 < tmax ? t1 : tmax;
// 		if (tmax <= tmin)
// 			return false;
// 	}
// 	return true;
// }

//compute the bounding box of two boxes on the fly
__host__ __device__ aabb surrounding_box(aabb box0, aabb box1) {
	vec3 small(ffmin(box0.min().x(), box1.min().x()),
		ffmin(box0.min().y(), box1.min().y()),
		ffmin(box0.min().z(), box1.min().z()));
	vec3 big(ffmax(box0.max().x(), box1.max().x()),
		ffmax(box0.max().y(), box1.max().y()),
		ffmax(box0.max().z(), box1.max().z()));
	return aabb(small, big);
}