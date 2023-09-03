#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_loop_animation", action="store_true", help="Connect the last and first keyframes in a continuous loop.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video.")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--simulation", action="store_true", help="Run simulation.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")


	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	args.mode = args.mode or mode_from_scene(args.scene) or mode_from_scene(args.load_snapshot)
	if not args.mode:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	if args.mode == "sdf":
		mode = ngp.TestbedMode.Sdf
		configs_dir = os.path.join(ROOT_DIR, "configs", "sdf")
		scenes = scenes_sdf
	elif args.mode == "nerf":
		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
	elif args.mode == "image":
		mode = ngp.TestbedMode.Image
		configs_dir = os.path.join(ROOT_DIR, "configs", "image")
		scenes = scenes_image
	elif args.mode == "volume":
		mode = ngp.TestbedMode.Volume
		configs_dir = os.path.join(ROOT_DIR, "configs", "volume")
		scenes = scenes_volume
	else:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	base_network = os.path.join(configs_dir, "base.json")
	if args.scene in scenes:
		network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
		base_network = os.path.join(configs_dir, network+".json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	testbed = ngp.Testbed(mode)
	testbed.init_rt(config_path="./scripts/exp/garden_mirror_shadow.json")

	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene = args.scene
		if not os.path.exists(args.scene) and args.scene in scenes:
			scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
		# testbed.load_training_data(scene)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw*sh > 1920*1080*4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.hybrid_render = 1
		testbed.init_window(sw, sh, second_window = args.second_window or False)


	if args.load_snapshot:
		snapshot = args.load_snapshot
		if not os.path.exists(snapshot) and snapshot in scenes:
			snapshot = default_snapshot_filename(scenes[snapshot])
		print("Loading snapshot ", snapshot)
		testbed.load_snapshot(snapshot)
	else:
		testbed.reload_network_from_file(network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_lens_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]
	if args.mode == "sdf":
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Optionally match nerf paper behaviour and train on a
		# fixed white bg. We prefer training on random BG colors.
		# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		# testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = 35000

	# bicycle
	testbed.view_dir =  [0.994,-0.105,-0.008]
	testbed.scale = 1.24

	# garden
	# testbed.view_dir = [0.919,-0.354,0.176]
	# testbed.scale = 1.5



	def view_circle(a, b, t, hT):
		o = (a + b) / 2
		or1 = a - o  
		c = np.cross(a, b)
		or2 = c / np.linalg.norm(c) * np.linalg.norm(or1)
		return o + or1 * np.cos(t/hT*np.pi) + or2 * np.sin(t/hT*np.pi)

	# view_dir_1 = np.array([0.994,-0.105,-0.008])
	# view_dir_2 = np.array([0.902,0.012,-0.432])	
	# testbed.scale = 1.24

	# inside view
	view_dir_1 = np.array([-0.896,-0.443,0.012])
	view_dir_2 = np.array( [-0.845,-0.499,0.192])	
	testbed.look_at = [0.539,0.811,0.440]
	testbed.scale = 0.847


	# outside view
	# view_dir_1 = np.array([0.474,-0.588,0.656])
	# view_dir_2 = np.array([0.358,-0.536,0.765])	
	# testbed.look_at = [-0.055,0.093,0.385]
	# testbed.scale = 2.658

	# outside to inside
	# view_dir_1 = np.array([0.474,-0.588,0.656])
	# view_dir_2 = np.array([-0.896,-0.443,0.012])	
	# look_at_1 =  np.array([-0.055,0.093,0.385])
	# look_at_2 =  np.array([0.539,0.811,0.440])
	# scale_1 = 2.658
	# scale_2 = 0.847

	# triangle view
	# view_dir_1 = np.array([0.684,-0.376,0.625])
	# view_dir_2 = np.array([0.902,0.012,-0.432])	
	# testbed.look_at = [0.702,0.472,0.708]
	# testbed.scale = 1.127


	testbed.background_color = [1.0, 1.0, 1.0, 1.0]
	half_circle = 40	
	testbed.view_dir = view_dir_1
	testbed.rt_set_shadow_decay(0.3)
	# render image
	if not args.gui:
		testbed.hybrid_render = 1
		n_frames = 80
		resolution = [args.width or 1920, args.height or 1080]

		path_output = "./render_output/garden_mirror/"
		os.makedirs("./render_output", exist_ok=True)
		os.makedirs(path_output, exist_ok=True)
		
		for i in tqdm(list(range(0, min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):

			testbed.view_dir = view_circle(view_dir_1, view_dir_2, i, half_circle)
			

			
			# ------------ view_interpolate
			# testbed.look_at = look_at_1*(n_frames-i)/n_frames + look_at_2*i/n_frames 
			# testbed.scale = scale_1*(n_frames-min(i*3,n_frames))/n_frames + scale_2*min(i*3,n_frames)/n_frames 
			# testbed.view_dir = view_dir_1*(n_frames-i)/n_frames + view_dir_2*i/n_frames  
			# ------------ view_interpolate



			frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
				

			write_image(f"{path_output}/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
	# end video



	# python ./scripts/exp_garden_mirror.py --scene ./extra_data/nerf360/garden --mode nerf --load_snapshot ./extra_data/nerf360/garden/35000.msgpack --width 200 --height 150 --video_spp 4 
	render_frame = 0
	tqdm_last_update = 0
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.frame():

				# testbed.view_dir = 	view_circle(view_dir_1, view_dir_2, render_frame, half_circle)
			
				render_frame += 1

				# print("!!!!!!!111", testbed.rt_world_size())
				# print("!!!!!!!222", testbed.rt_hittable_center(0))


				# if render_frame % 2 == 0 and args.simulation:
				# 	state = hybrid_sim.call_step()
				# 	print("------- render_frame", render_frame, state[0][:3])

				# 	# if render_frame % 100 == 0:
				# 	# print("!!!!!!!111", testbed.rt_world_size())
				# 	print("!!!!!!!222", testbed.rt_hittable_center(0))
				# 	new_center = state[0][:3]
				# 	print("!!!!!!!333", testbed.rt_set_hittable_center(0, new_center))
				# print("!!!!!!!222", testbed.rt_hittable_center(0))
				# exit()


				# if testbed.want_repl():
				# 	repl(testbed)
				# # What will happen when training is done?
				# if testbed.training_step >= n_steps:
				# 	if args.gui:
				# 		testbed.shall_train = False
				# 	else:
				# 		break

				# # Update progress bar
				# if testbed.training_step < old_training_step or old_training_step == 0:
				# 	old_training_step = 0
				# 	t.reset()

				# now = time.monotonic()
				# if now - tqdm_last_update > 0.1:
				# 	t.update(testbed.training_step - old_training_step)
				# 	t.set_postfix(loss=testbed.loss)
				# 	old_training_step = testbed.training_step
				# 	tqdm_last_update = now

