#ifndef RENDER_CUH
#define RENDER_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "cuda_error_handle.cuh"

__global__ void render(float *fb, int max_x, int max_y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 3 + i * 3;
	fb[pixel_index + 0] = float(i) / max_x;
	fb[pixel_index + 1] = float(j) / max_y;
	fb[pixel_index + 2] = 0.2;
}

class Render {
	int nx, ny;
	char* save_dir;

public:
	Render(int nx, int ny, char* save_dir) {
		this->nx = nx;
		this->ny = ny;
		this->save_dir = (char*)malloc(sizeof(save_dir) + 1);
		strcpy(this->save_dir, save_dir);
	}

	void GetImage() {
		FILE *stream1;
		freopen_s(&stream1, save_dir, "w", stdout);
		if (stream1 == NULL) {
			exit(9);
		}
		size_t FB_size = 3 * nx*ny * sizeof(float); //the size of a frame buffer
		float *fb; //frame buffer
		checkCudaErrors(cudaMallocManaged((void**)&fb, FB_size));
		int tx = 8, ty = 8;
		dim3 blocks(nx / tx + 1, ny / ty + 1);
		dim3 threads(tx, ty);
		render << <blocks, threads >> > (fb, nx, ny);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		std::cout << "P3\n" << nx << " " << ny << "\n255\n";
		for (int j = ny - 1; j >= 0; j--) {
			for (int i = 0; i < nx; i++) {
				int pixel_idx = j * 3 * nx + i * 3;
				float r = fb[pixel_idx + 0];
				float g = fb[pixel_idx + 1];
				float b = fb[pixel_idx + 2];
				int ir = int(255.99*r);
				int ig = int(255.99*g);
				int ib = int(255.99*b);
				std::cout << ir << " " << ig << " " << ib << std::endl;
			}
		}
		checkCudaErrors(cudaFree(fb));
		fclose(stream1);
	}

};

#endif // !RENDER_CUH
