#ifndef AABB_CUH
#define AABB_CUH

#include "vec3.cuh"
#include "ray.cuh"

class aabb {
public:
	__host__ __device__ aabb();
	__host__ __device__ aabb(const vec3& a, const vec3& b);

	__host__ __device__ vec3 min() const;
	__host__ __device__ vec3 max() const;

	// __device__ bool hit(const ray& r, float tmin, float tmax);

	__device__ inline bool hit(const ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; a++) {
			float invD = 1.0f / r.direction()[a];
			float t0 = (min()[a] - r.origin()[a]) * invD;
			float t1 = (max()[a] - r.origin()[a]) * invD;
			if (invD < 0.0f) {
				float tmp = t0;
				t0 = t1;
				t1 = tmp;
			}
			//std::swap(t0, t1);
			tmin = t0 > tmin ? t0 : tmin;
			tmax = t1 < tmax ? t1 : tmax;
			if (tmax < tmin)
				return false;
		}
		return true;
	};

	vec3 _min;
	vec3 _max;
};




// //compute the bounding box of two boxes on the fly
__host__ __device__ aabb surrounding_box(aabb box0, aabb box1);

//compute the bounding box of two boxes on the fly
// __host__ __device__ aabb surrounding_box(aabb box0, aabb box1) {
// 	vec3 small(ffmin(box0.min().x(), box1.min().x()),
// 		ffmin(box0.min().y(), box1.min().y()),
// 		ffmin(box0.min().z(), box1.min().z()));
// 	vec3 big(ffmax(box0.max().x(), box1.max().x()),
// 		ffmax(box0.max().y(), box1.max().y()),
// 		ffmax(box0.max().z(), box1.max().z()));
// 	return aabb(small, big);
// }
#endif