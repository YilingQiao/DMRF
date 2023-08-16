#include "hittable_list.cuh"
#include "aabb.cuh"
#include "triangle.cuh"
/*
//copy objs to gpu
hittable** copy_objs_to_gpu(hittable **obj_list_host, int list_size) {
	hittable **obj_list_tmp = new hittable*[list_size];
	for (int i = 0; i < list_size; i++) {
		obj_list_tmp[i] = obj_list_host[i]->copy_to_gpu();
	}
	hittable **obj_list_device;
	checkCudaErrors(cudaMalloc((void**)&obj_list_device, list_size * sizeof(hittable*)));
	checkCudaErrors(cudaMemcpy(obj_list_device, obj_list_tmp, list_size * sizeof(hittable*), cudaMemcpyHostToDevice));
	//free(obj_list_tmp);
	delete[] obj_list_tmp;
	return obj_list_device;
}

hittable* hittable_list::copy_to_gpu() const {
	hittable_list *tmp, *device;
	checkCudaErrors(cudaMalloc((void**)&device, sizeof(hittable_list)));
	tmp = new hittable_list();
	memcpy(tmp, this, sizeof(hittable));
	tmp->list = copy_objs_to_gpu(this->list, this->list_size);
	checkCudaErrors(cudaMemcpy(device, tmp, sizeof(hittable_list), cudaMemcpyHostToDevice));
	return device;
}
*/

hittable** copy_objs_to_gpu(hittable **obj_list_host, int list_size) {
	hittable **obj_list_tmp = new hittable*[list_size];
	for (int i = 0; i < list_size; i++) {
		obj_list_tmp[i] = obj_list_host[i]->copy_to_gpu();
	}
	hittable **obj_list_device;
	checkCudaErrors(cudaMalloc((void**)&obj_list_device, list_size * sizeof(hittable*)));
	checkCudaErrors(cudaMemcpy(obj_list_device, obj_list_tmp, list_size * sizeof(hittable*), cudaMemcpyHostToDevice));
	//free(obj_list_tmp);
	delete[] obj_list_tmp;
	return obj_list_device;
}


__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max,
	hit_record& rec) const {

	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = t_max;
	for (int i = 0; i < list_size; i++) {
		if (type_lightsource == list[i]->mat_ptr->type)
			continue;
		if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

__host__ __device__ bool hittable_list::bounding_box(float t0, float t1, aabb& box){
	if (list_size < 1) return false;
	aabb temp_box;
	bool first_true = list[0]->bounding_box(t0, t1, temp_box);
	if (!first_true)
		return false;
	else
		box = temp_box;
	for (int i = 1; i < list_size; i++) {
		if (list[i]->bounding_box(t0, t1, temp_box)) {
			box = surrounding_box(box, temp_box);
		}
		else
			return false;
	}
	return true;
}

__device__ vec3 hittable_list::sample(curandState *local_rand_state) const {

	// int idx = 0;
	// n primitives in the light source
	int idx = (int)(curand_uniform(local_rand_state) * (list_size-1)); 
	return list[idx]->sample(local_rand_state);
}

