#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "ray.cuh"
#include "vec3.cuh"
#include "random.cuh"

// #define PI 3.1415926535898

class camera {
public:
	/*__device__ camera() {
		lower_left_corner = vec3(-2.0, -1.0, -1.0);
		horizontal = vec3(4.0, 0.0, 0.0);
		vertical = vec3(0.0, 2.0, 0.0);
		origin = vec3(0.0, 0.0, 0.0);
	}*/

	__host__ __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect,
		float aperture, float focus_dist) {
		lens_radius = aperture / 2;
		float theta = vfov * 3.1415926535898 / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin
			- half_width * focus_dist * u
			- half_height * focus_dist * v
			- focus_dist * w;
		horizontal = 2 * half_width*focus_dist*u;
		vertical = 2 * half_height*focus_dist*v;
	}

	__device__ ray get_ray(float s, float t, curandState *pixel_random_seed) {
		vec3 rd = lens_radius * random_in_unit_disk(pixel_random_seed);
		vec3 offset = u * rd.x() + v * rd.y();
		return ray(origin + offset,
			lower_left_corner + s * horizontal + t * vertical
			- origin - offset);
	}



	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};
#endif