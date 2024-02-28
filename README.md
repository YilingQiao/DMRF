


# Dynamic Mesh-Aware Radiance Fields

[Yi-Ling Qiao*](https://ylqiao.net/), [Alexander Gao*](https://gaoalexander.github.io/), [Yiran Xu](https://twizwei.github.io/), [Yue Feng](https://yuefeng21.github.io/), [Jia-Bin Huang](https://jbhuang0604.github.io/), [Ming C. Lin](https://www.cs.umd.edu/~lin/)

[[Project]](https://mesh-aware-rf.github.io/) [[Paper]](https://arxiv.org/abs/2309.04581)  [[Video]](https://youtu.be/v9k_UzUKtcE) 



## Introduction
This is an implementation of the ICCV 2023 paper Dynamic Mesh-Aware Radiance Fields.
The NeRF volume rendering is largely based on NVIDA's great work [InstantNGP](https://github.com/NVlabs/instant-ngp). The physics part also runs with NVIDIA's [Warp](https://github.com/NVIDIA/warp) and/or DeepMind's [Mujoco](https://github.com/deepmind/mujoco). A more detailed tutorial for ray tracing can be found in [Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io).


## Structure
We have now uploaded all core C++/CUDA files but are still cleaning the python scripts, config files, and datasets for running each experiment.

The entry to the main rendering loop for raytracing-NeRF coupling is the `render_nerf_rt` function at `./src/testbed_nerf.cu`. 

The folder `./simpleRt/` also has CUDA-based raytracing-related functions. 

Physics simulation module will be called in python scripts.

## Setup
This project has been tested with Ubuntu 20.04, gcc 9.4.0, CUDA 11.8.
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

If you have any issues building/running this project, please try to make sure you can run [InstantNGP](https://github.com/NVlabs/instant-ngp) first (especially its commit `c4d622e`, which is a start point of our project).

## TODOs
1. An end-to-end tutorial for training NeRF, estimating light source and geometry, inserting mesh, and rendering.

2. An introduction to the config file system and code structure.

3. Other scenes presented in the paper.

## Demos

Let's download the data and pretrained NeRF from this [google drive](https://drive.google.com/drive/folders/1n8BJhkSCBqXTN-mkdMdfrM5IpsMzkj61?usp=drive_link).

Put them under `DMRF/extra_data`.

### Reflective ball
We can insert a reflective ball into the `nerf360/garden` scene. The ball will also cast shdows onto the ground. 

Scene desctiption of the ball, lightsource, and shadow mesh can be found in the configuration file `./scripts/exp/garden_ball.json"`.

```
python ./scripts/exp_garden_ball.py --mode nerf --load_snapshot ./extra_data/nerf360/garden/35000.msgpack  --width 800 --height 600 --gui
```

<div align="center">
<img width="400px" src="https://github.com/YilingQiao/linkfiles/raw/master/23DMRF/garden_ball_shadow.gif"> 
</div>


### Infinite mirror room
Let's go one step further and add more objects into the scene.

With relfective walls around, now the ray can bounce multiple times inside a mirror room.

Since the scene is complex, we choose not to render it in GUI. Images will be saved to `./render_output/garden_mirror/`.

```
python ./scripts/exp_garden_mirror.py --mode nerf --load_snapshot ./extra_data/nerf360/garden/35000.msgpack --width 200 --height 150 --video_spp 4 
```

<div align="center">
<img width="400px" src="https://github.com/YilingQiao/linkfiles/raw/master/23DMRF/garden_mirror_monkey.gif"> 
</div>

### Interactive game
We can also add a controllable object into NeRF and see how it interact with the scene.

Here we use [Warp](https://github.com/NVIDIA/warp) to compute the dynamics. You might need to install this package first.

Use `W/S/A/D/Z/X` to control the ball's moving direction. Use `space` to stop the ball

```
python ./scripts/exp_game.py --mode nerf --load_snapshot ./extra_data/nerf360/kitchen/35000.msgpack --width 800 --height 600 --gui --simulation 
```

<div align="center">
<img width="400px" src="https://github.com/YilingQiao/linkfiles/raw/master/23DMRF/game_counter.gif"> 
</div>

## Run with your own data

### Preprocess
You can refer to this [tutorial](preprocess/README.md) for preprocessing your own data.

### Config file
Scene configuration files are stored in `scripts/exp/*.json`.

Geometries in the scene can be provided to the scene by setting `spheres` and `objfile`. You can set the sphere's `center` and `radius`.

Rendering materials of the geometries can be set using `materials` entries. Available types include `dielectric`, `lambertian`, `metal`, `shadow`, `lightsource`. Note that the last two are used for cast shadows and not actually visible in the scene.

The `yiling/231018_lightsrc` branch has more experimental features for controlling the light source, and it can render faster for lambertian materials.

Note that the material `id` in `geometries` and `materials` are different by 1, which might be a little confusing. (i.e. material id in geometries =  `id` in `mateirals` + 1)



## BibTex
```
@article{qiao2023dmrf,
  author    = {Yi-Ling Qiao and Alexander Gao and Yiran Xu and Yue Feng and Jia-Bin Huang and Ming C. Lin},
  title     = {Dynamic Mesh-Aware Radiance Fields},
  journal   = {ICCV},
  year      = {2023},
}
```
