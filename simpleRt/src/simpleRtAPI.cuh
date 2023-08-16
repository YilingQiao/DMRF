#ifndef SIMPLERTAPI_CUH
#define SIMPLERTAPI_CUH

#include <iostream>
#include <time.h>
#include <float.h>
#include <random>
#include <curand_kernel.h>

#include "camera.cuh"
#include "sphere.cuh"
#include "hittable_list.cuh"
#include "material.cuh"

#include "bvh.cuh"
#include "FileReader.cuh"


void create_world_cpu(hittable **d_list, camera *&d_camera, int nx, int ny);
void create_ray_trace_scene(hittable **&d_world, hittable **&d_lightsrc, hittable **&d_shadow);
__global__ void rand_init(curandState *rand_state);

#endif