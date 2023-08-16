#include "simpleRtAPI.cuh"

#define RND_CPU (distribution(generator))

void create_world_cpu(hittable **d_list) {
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    std::default_random_engine generator;

    // int i = 0;

    // d_list[i++] = new sphere(vec3(0,-1001.0,-1), 1000,
    //                        new lambertian(vec3(0.5, 0.5, 0.5)));
    // for(int a = -11; a < 11; a++) {
    //   for(int b = -11; b < 11; b++) {
    //     float choose_mat = RND_CPU;
    //     vec3 center(a+RND_CPU,0.2,b+RND_CPU);
    //     if(choose_mat < 0.8f) {
    //       d_list[i++] = new sphere(center, 0.2,
    //                                new lambertian(vec3(RND_CPU*RND_CPU, RND_CPU*RND_CPU, RND_CPU*RND_CPU)));
    //     }
    //     else if(choose_mat < 0.95f) {
    //       d_list[i++] = new sphere(center, 0.2,
    //                                new metal(vec3(0.5f*(1.0f+RND_CPU), 0.5f*(1.0f+RND_CPU), 0.5f*(1.0f+RND_CPU)), 0.5f*RND_CPU));
    //     }
    //     else {
    //       d_list[i++] = new sphere(center, 0.2, new dielectric(vec3(1.0, 1.0, 1.0), 1.5));
    //     }
    //   }
    // }
    // d_list[i++] = new sphere(vec3(0, 1, 0),  1.0, new dielectric(vec3(0.0, 1.0, 0.0), 1.5));
    // d_list[i++] = new sphere(vec3(-2, 1, 0), 1.0, new metal(vec3(0.0, 0.0, 1.0), 0.0));
    // d_list[i++] = new sphere(vec3(-2, 1, 0), 1.0, new lambertian(vec3(0.0, 1.0, 0.0))); 
    // d_list[i++] = new sphere(vec3(2, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

  
}


// void create_without_bvh2(hittable **&d_world) {
//   // --------------------- AABB
//   int num_hittables = 0; //22*22+1+3;

//   // ---------------- json scene
//   camera *h_camera = NULL; 
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/family_siggraph.json";
//   // char dir[200] = "/media/yiling/ljb_backup/yiling/22hybrid_extra/instant-ngp/data/_car/4.json";

//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/monkey.json"; // bowl

//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/scale/scale.json"; // scale
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/cloth/cloth.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/fem/fem.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/game/game.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/kitchen_compare/kitchen_compare.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/jump/jump.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/rendering/render_bicycle.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/rendering/render_garden.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/tryon/tryon.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser/teaser_diamond.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_ball/teaser_ball.json"; // cloth

//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror_iccv.json"; // cloth
//   char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/move_chair/chair.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/game_counter/game.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/diagram_bunny/diagram.json"; // fem
//   // char dir[200] = "/media/yiling/ljb_backup/yiling/22hybrid_extra/instant-ngp/data/alexander/furniture/base.json"; // fem
//   std::vector<hittable*> vec_obj_from_json;
//   std::vector<int> vec_lightsource_idx;
//   int nx, ny, ns;
//   FileReader::readfile_to_render(
//     vec_obj_from_json, vec_lightsource_idx,
//     dir, nx, ny, ns, h_camera);
//   int size_obj_from_json = vec_obj_from_json.size();
//   printf("size_obj_from_json %d\n", size_obj_from_json);
//   // ---------------- json scene

//   // ---------------- create scene
//   hittable **h_list = (hittable **) malloc(
//     (num_hittables + size_obj_from_json) * sizeof(hittable *));
//   create_world_cpu(h_list);

//   // aggreate scene
//   for (int i = num_hittables; i < num_hittables + size_obj_from_json; i++) {
//     h_list[i] = vec_obj_from_json[i - num_hittables];
//   }

//   // initialize world in cpu
//   hittable *world_host;
//   // world_host = new bvh_node(h_list, 0, num_hittables + size_obj_from_json, 0, 1, 0);

//   // copy to gpu
//   checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
//   hittable **tmp = new hittable*[1];
//   // use bvh
//   // *tmp = world_host->copy_to_gpu();

//   // do not use bvh
//   hittable *h_world_list;
//   // malloc((void **)&h_world_list, sizeof(hittable *));
//   h_world_list  = new hittable_list(h_list, num_hittables + size_obj_from_json);
//   *tmp = h_world_list->copy_to_gpu();


//   checkCudaErrors(cudaMemcpy(d_world, tmp, sizeof(hittable*), cudaMemcpyHostToDevice));

//   // int *count_device;
//   // int *count_host = new int;
//   // checkCudaErrors(cudaMalloc((void**)&count_device, sizeof(int)));
//   // visit <<<1, 1 >>>((bvh_node**)d_world, count_device);
//   // std::cout << "=========== 5\n";
//   // checkCudaErrors(cudaGetLastError());
//   // checkCudaErrors(cudaDeviceSynchronize());
//   // checkCudaErrors(cudaMemcpy(count_host, count_device, sizeof(int), cudaMemcpyDeviceToHost));
//   // std::cerr << "dbg:count=" << *count_host << std::endl;
// }


void create_with_bvh_geometry_shadow(hittable **&d_world, hittable **&d_lightsrc, hittable **&d_shadow) {
  // --------------------- AABB

  // ---------------- json scene
  camera *h_camera = NULL; 
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror_test.json"; // cloth
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/bonsai/bonsai.json"; // cloth
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/lamp/lamp_geometry.json"; // 
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/family_light_source/family_light_source.json"; // 
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/kitchen_compare/kitchen_material.json"; // 
  char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/move_chair/chair.json"; // fem 
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/jump/jump_light.json"; // fem
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/jump/tmp.json"; // fem
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/rendering/render_garden_lightsource.json"; // fem
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror_iccv_lightsource_geom.json"; // cloth
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/family_siggraph.json";
  // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/monkey.json";

  std::vector<hittable*> vec_obj_from_json;
  std::vector<hittable*> vec_lightsrc_list;
  std::vector<hittable*> vec_shadow_list;

  std::vector<int> vec_lightsource_idx;
  int nx, ny, ns;
  FileReader::readfile_to_render(
    vec_obj_from_json, vec_lightsrc_list, vec_shadow_list,
    dir, nx, ny, ns, h_camera);
  int size_obj_from_json = vec_obj_from_json.size();
  int size_lightsrc_list = vec_lightsrc_list.size();
  int size_shadow_list = vec_shadow_list.size();

  printf("size_obj_from_json %d\n", size_obj_from_json);
  printf("size_lightsrc_list %d\n", size_lightsrc_list);
  printf("size_shadow_list %d\n", size_shadow_list);
  // ---------------- json scene

  // ---------------- create scene
  hittable **h_obj = (hittable **) malloc(
    size_obj_from_json * sizeof(hittable *));
  hittable **h_lightsrc = (hittable **) malloc(
    size_lightsrc_list * sizeof(hittable *));
  hittable **h_shadow = (hittable **) malloc(
    size_shadow_list * sizeof(hittable *));


  // aggreate scene
  for (int i = 0; i < size_obj_from_json; i++) {
    h_obj[i] = vec_obj_from_json[i];
  }
  for (int i = 0; i < size_lightsrc_list; i++) {
    h_lightsrc[i] = vec_lightsrc_list[i];
  }
  for (int i = 0; i < size_shadow_list; i++) {
    h_shadow[i] = vec_shadow_list[i];
  }

  // initialize world in cpu - bvh
  // hittable *world_host;
  // world_host = new bvh_node(h_obj, 0, size_obj_from_json, 0, 1, 0);
  // checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
  // hittable **tmp = new hittable*[1];
  // *tmp = world_host->copy_to_gpu();
  // checkCudaErrors(cudaMemcpy(d_world, tmp, sizeof(hittable*), cudaMemcpyHostToDevice));


  // initialize world in cpu - list
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
  hittable **tmp = new hittable*[1];
  hittable *h_obj_list;
  h_obj_list  = new hittable_list(h_obj, size_obj_from_json);
  *tmp = h_obj_list->copy_to_gpu();
  checkCudaErrors(cudaMemcpy(d_world, tmp, sizeof(hittable*), cudaMemcpyHostToDevice));


  // initialize lightsrc in cpu
  checkCudaErrors(cudaMalloc((void**)&d_lightsrc, sizeof(hittable*)));
  hittable **tmp_lightsrc = new hittable*[1];
  hittable *h_lightsrc_list;
  h_lightsrc_list  = new hittable_list(h_lightsrc, size_lightsrc_list);
  *tmp_lightsrc = h_lightsrc_list->copy_to_gpu();
  checkCudaErrors(cudaMemcpy(d_lightsrc, tmp_lightsrc, sizeof(hittable*), cudaMemcpyHostToDevice));

  // initialize shadow in cpu
  // checkCudaErrors(cudaMalloc((void**)&d_shadow, sizeof(hittable*)));
  // hittable **tmp_shadow = new hittable*[1];
  // hittable *h_shadow_list;
  // h_shadow_list  = new hittable_list(h_shadow, size_shadow_list);
  // *tmp_shadow = h_shadow_list->copy_to_gpu();
  // checkCudaErrors(cudaMemcpy(d_shadow, tmp_shadow, sizeof(hittable*), cudaMemcpyHostToDevice));

  // initialize shadow in cpu
  hittable *world_host_shadow;
  world_host_shadow = new bvh_node(h_shadow, 0, size_shadow_list, 0, 1, 0);
  checkCudaErrors(cudaMalloc((void**)&d_shadow, sizeof(hittable*)));
  hittable **tmp_shadow = new hittable*[1];
  *tmp_shadow = world_host_shadow->copy_to_gpu();
  checkCudaErrors(cudaMemcpy(d_shadow, tmp_shadow, sizeof(hittable*), cudaMemcpyHostToDevice));

}


// void create_with_bvh(hittable **&d_world) {
//   // --------------------- AABB
//   int num_hittables = 0; //22*22+1+3;

//   // ---------------- json scene
//   camera *h_camera = NULL; 
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror_test.json"; // cloth
//   char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror_iccv_lightsource.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/family_siggraph.json";
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/monkey.json";

//   std::vector<hittable*> vec_obj_from_json;
//   std::vector<int> vec_lightsource_idx;
//   int nx, ny, ns;
//   FileReader::readfile_to_render(
//     vec_obj_from_json, vec_lightsource_idx,
//     dir, nx, ny, ns, h_camera);
//   int size_obj_from_json = vec_obj_from_json.size();
//   printf("size_obj_from_json %d\n", size_obj_from_json);
//   // ---------------- json scene

//   // ---------------- create scene
//   hittable **h_list = (hittable **) malloc(
//     (num_hittables + size_obj_from_json) * sizeof(hittable *));
//   create_world_cpu(h_list);

//   // aggreate scene
//   for (int i = num_hittables; i < num_hittables + size_obj_from_json; i++) {
//     h_list[i] = vec_obj_from_json[i - num_hittables];
//   }

//   // initialize world in cpu
//   hittable *world_host;
//   world_host = new bvh_node(h_list, 0, num_hittables + size_obj_from_json, 0, 1, 0);
//   // exit(0);
//   // copy to gpu
//   checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
//   hittable **tmp = new hittable*[1];
//   *tmp = world_host->copy_to_gpu();
//   checkCudaErrors(cudaMemcpy(d_world, tmp, sizeof(hittable*), cudaMemcpyHostToDevice));

//   int *count_device;
//   int *count_host = new int;
//   checkCudaErrors(cudaMalloc((void**)&count_device, sizeof(int)));
//   visit <<<1, 1 >>>((bvh_node**)d_world, count_device);
//   std::cout << "=========== 5\n";
//   checkCudaErrors(cudaGetLastError());
//   checkCudaErrors(cudaDeviceSynchronize());
//   checkCudaErrors(cudaMemcpy(count_host, count_device, sizeof(int), cudaMemcpyDeviceToHost));
//   std::cerr << "dbg:count=" << *count_host << std::endl;
// }



#define RND (curand_uniform(&local_rand_state))

// __global__ void create_world_gpu(
//   hittable **d_list, hittable **d_world, curandState *rand_state, int num_hittables) {
//   if (threadIdx.x == 0 && blockIdx.x == 0) {
//     curandState local_rand_state = *rand_state;
//     int i = 0;
//     // d_list[i++] = new sphere(vec3(0,-1000.0,-1), 1000,
//     //                        new lambertian(vec3(0.5, 0.5, 0.5)));
//     // for(int a = -11; a < 11; a++) {
//     //   for(int b = -11; b < 11; b++) {
//     //     float choose_mat = RND;
//     //     vec3 center(a+RND,0.2,b+RND);
//     //     if(choose_mat < 0.8f) {
//     //       d_list[i++] = new sphere(center, 0.2,
//     //                                new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
//     //     }
//     //     else if(choose_mat < 0.95f) {
//     //       d_list[i++] = new sphere(center, 0.2,
//     //                                new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
//     //     }
//     //     else {
//     //       d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
//     //     }
//     //   }
//     // }

//     // d_list[i++] = new sphere(vec3(0, 0, 1.5),  0.3, new lambertian(vec3(0.5, 0.5, 0.5)));
//     // d_list[i++] = new sphere(vec3(0, 0.5, 1.0),  0.2, new dielectric(vec3(0.0, 1.0, 0.0), 1.0)); // fox
//     // d_list[i++] = new sphere(vec3(-2, 1, 0), 1.0, new metal(vec3(0.0, 0.0, 1.0), 0.0));


//     // bicycle
//     d_list[i++] = new sphere(vec3(1.2, 0.5, 0.3),  0.5, new metal(vec3(1.0, 0.7, 0.3), 0.0));

//     // garden
//     // d_list[i++] = new sphere(vec3(1.2, 1.1, 0.),  0.6, new metal(vec3(0.6, 0.6, 0.6), 0.0));

//     *rand_state = local_rand_state;
//     *d_world  = new hittable_list(d_list, num_hittables);

//   }
// }



__global__ void rand_init(curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
  }
}

// void create_without_bvh(hittable **&d_world) {
//   // allocate random state
//   curandState *d_rand_state2;
//   checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

//   // we need that 2nd random state to be initialized for the world creation
//   rand_init<<<1,1>>>(d_rand_state2);
//   checkCudaErrors(cudaGetLastError());
//   checkCudaErrors(cudaDeviceSynchronize());

//   // make our world of hittables & the camera
//   hittable **d_list;
//   int num_hittables = 1; // 22*22+1+3
//   checkCudaErrors(cudaMalloc((void **)&d_list, num_hittables*sizeof(hittable *)));
//   checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
//   create_world_gpu<<<1,1>>>(d_list, d_world, d_rand_state2, num_hittables);
//   checkCudaErrors(cudaGetLastError());
//   checkCudaErrors(cudaDeviceSynchronize());


// }




// void create_without_bvh2_light_source(hittable **&d_world) {
//   // --------------------- AABB
//   int num_hittables = 0; //22*22+1+3;

//   // ---------------- json scene
//   camera *h_camera = NULL; 
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/family_siggraph.json";
//   // char dir[200] = "/media/yiling/ljb_backup/yiling/22hybrid_extra/instant-ngp/data/_car/4.json";

//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/simpleRt/assets/monkey.json"; // bowl

//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/scale/scale.json"; // scale
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/cloth/cloth.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/fem/fem.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/game/game.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/kitchen_compare/kitchen_compare.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/jump/jump_light.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/rendering/render_bicycle.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/rendering/render_garden_lightsource.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/rendering/render_garden.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/tryon/tryon.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser/teaser_diamond.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_ball/teaser_ball.json"; // cloth

//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/lamp/base.json"; // fem
//   char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/conference/base.json"; // fem
  
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/teaser_garden_mirror/teaser_garden_mirror_iccv_lightsource.json"; // cloth
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/game_counter/game.json"; // fem
//   // char dir[200] = "/home/yiling/Desktop/research/22hybrid/instant-ngp/scripts/exp/diagram_bunny/diagram.json"; // fem
//   // char dir[200] = "/media/yiling/ljb_backup/yiling/22hybrid_extra/instant-ngp/data/alexander/furniture/base.json"; // fem
//   std::vector<hittable*> vec_obj_from_json;
//   std::vector<int> vec_lightsource_idx;
//   int nx, ny, ns;
//   FileReader::readfile_to_render(
//     vec_obj_from_json, vec_lightsource_idx,
//     dir, nx, ny, ns, h_camera);
//   int size_obj_from_json = vec_obj_from_json.size();
//   printf("size_obj_from_json %d\n", size_obj_from_json);
//   // ---------------- json scene

//   // ---------------- create scene
//   hittable **h_list = (hittable **) malloc(
//     (num_hittables + size_obj_from_json) * sizeof(hittable *));
//   create_world_cpu(h_list);

//   // aggreate scene
//   for (int i = num_hittables; i < num_hittables + size_obj_from_json; i++) {
//     h_list[i] = vec_obj_from_json[i - num_hittables];
//   }

//   // initialize world in cpu
//   hittable *world_host;
//   // world_host = new bvh_node(h_list, 0, num_hittables + size_obj_from_json, 0, 1, 0);

//   // copy to gpu
//   checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
//   hittable **tmp = new hittable*[1];
//   // use bvh
//   // *tmp = world_host->copy_to_gpu();

//   // do not use bvh
//   hittable *h_world_list;
//   // malloc((void **)&h_world_list, sizeof(hittable *));
//   h_world_list  = new hittable_list(h_list, num_hittables + size_obj_from_json);
//   *tmp = h_world_list->copy_to_gpu();


//   checkCudaErrors(cudaMemcpy(d_world, tmp, sizeof(hittable*), cudaMemcpyHostToDevice));


// }



void create_ray_trace_scene(hittable **&d_world, hittable **&d_lightsrc, hittable **&d_shadow) {
  // create_with_bvh(d_world);
  create_with_bvh_geometry_shadow(d_world, d_lightsrc, d_shadow);

  // create_without_bvh2(d_world);
  // create_without_bvh2_light_source(d_world);

  // create_without_bvh(d_world);
}