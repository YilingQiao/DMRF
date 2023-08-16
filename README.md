

# Dynamic Mesh-Aware Radiance Fields

[Yi-Ling Qiao*](https://ylqiao.net/), [Alexander Gao*](https://gaoalexander.github.io/), [Yiran Xu](https://twizwei.github.io/), [Yue Feng](https://yuefeng21.github.io/), [Jia-Bin Huang](https://jbhuang0604.github.io/), [Ming C. Lin](https://www.cs.umd.edu/~lin/)

## Introduction
This is an implementation of the ICCV 2023 paper Dynamic Mesh-Aware Radiance Fields.
The NeRF volume rendering is largely based on NVIDA's great work [InstantNGP](https://github.com/NVlabs/instant-ngp). The physics part also runs with NVIDIA's [Warp](https://github.com/NVIDIA/warp) and/or DeepMind's [Mujoco](https://github.com/deepmind/mujoco). A more detailed tutorial for ray tracing can be found in [Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io).


## Structure
We have now uploaded all core C++/CUDA files but are still cleaning the python scripts, config files, and datasets for running each experiment.

The main loop for raytracing-NeRF coupling is `render_nerf_rt` function at `./src/testbed_nerf.cu`. `./simpleRt/` also has raytracing-related functions.


## Setup
TODO

## Demos
TODO
