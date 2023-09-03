

# Dynamic Mesh-Aware Radiance Fields

[Yi-Ling Qiao*](https://ylqiao.net/), [Alexander Gao*](https://gaoalexander.github.io/), [Yiran Xu](https://twizwei.github.io/), [Yue Feng](https://yuefeng21.github.io/), [Jia-Bin Huang](https://jbhuang0604.github.io/), [Ming C. Lin](https://www.cs.umd.edu/~lin/)

[[Project]](https://sites.google.com/view/mesh-aware-rf/home) [[Paper]]()



## Introduction
This is an implementation of the ICCV 2023 paper Dynamic Mesh-Aware Radiance Fields.
The NeRF volume rendering is largely based on NVIDA's great work [InstantNGP](https://github.com/NVlabs/instant-ngp). The physics part also runs with NVIDIA's [Warp](https://github.com/NVIDIA/warp) and/or DeepMind's [Mujoco](https://github.com/deepmind/mujoco). A more detailed tutorial for ray tracing can be found in [Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io).


## Structure
We have now uploaded all core C++/CUDA files but are still cleaning the python scripts, config files, and datasets for running each experiment.

The entry to the main rendering loop for raytracing-NeRF coupling is the `render_nerf_rt` function at `./src/testbed_nerf.cu`. 

The folder `./simpleRt/` also has CUDA-based raytracing-related functions. 

Physics simulation module will be called in python scripts.

## Setup
I build this project with Ubuntu 20.04, gcc 9.4.0, CUDA 11.8.
```bash
git clone git@github.com:YilingQiao/DMRF.git
cd DMRF
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd ..
```


## Demos

### Reflective ball
We can add a reflective ball into the `nerf360/garden` scene. The ball also cast shdows onto the ground. Scene desctiption of the ball, lightsource, and shadow mesh can be found in the configuration file `./scripts/exp/garden_ball.json"`.

```
python ./scripts/exp_garden_ball.py --mode nerf --load_snapshot ./extra_data/nerf360/garden/35000.msgpack  --width 800 --height 600 --gui
```


### Infinite mirror room
Since the scene is complex, we choose not to render it in the gui. Images will be saved to `./render_output/garden_mirror/`

```
python ./scripts/exp_garden_mirror.py --mode nerf --load_snapshot ./extra_data/nerf360/garden/35000.msgpack --width 200 --height 150 --video_spp 4 
```


### Interactive game
Use `W/S/A/D/Z/X` to control the ball's moving direction. Use `space` to stop the ball

```
python ./scripts/exp_game.py --mode nerf --load_snapshot ./extra_data/nerf360/kitchen/35000.msgpack --width 800 --height 600 --gui --simulation 
```
