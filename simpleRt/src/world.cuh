#ifndef WORLD_CUH
#define WORLD_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "cuda_error_handle.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include <stdio.h>

__global__ void create_world(hittable **obj_list, hittable **world, camera **cam, int dx,int dy,curandState *random_seed) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		vec3 lookfrom(-85, -62, 56);
		vec3 lookat(-10, 37, -9);
		float dist_to_focus = 120;
		float aperture = 0.1;

		*cam = new camera(lookfrom, lookat, vec3(0, 0, 1), 55,
			float(dx) / float(dy), aperture, dist_to_focus);
	}
}

__global__ void free_world(hittable **obj_list, hittable **world, camera **cam) {
	printf("ok1\n");
	for (int i = 0; i < ((hittable_list*)(*world))->list_size; i++) {
		delete ((sphere*)obj_list[i])->mat_ptr;
		delete obj_list[i];
	}
	printf("ok2\n"); 
	delete (*world);
	printf("ok3\n"); 
	delete (*cam);
}

#endif