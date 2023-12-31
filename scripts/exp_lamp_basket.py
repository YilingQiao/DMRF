#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import sys
build_path = "/home/awgao/git/hybrid_nerf_demos/hybrid_nerf/build"
sys.path.append(build_path)
print(sys.path)
import commentjson as json
import numpy as np
import pyngp as ngp  # noqa
from tqdm import tqdm

from scenes import *


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
    parser.add_argument("--start_frame", type=int, default=0, help="Frame number in sequence to start rendering")

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
        print("NeRF compatibility mode enabled")

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

    def view_circle(a, b, t, hT):
        o = (a + b) / 2
        or1 = a - o
        c = np.cross(a, b)
        or2 = c / np.linalg.norm(c) * np.linalg.norm(or1)
        return o + or1 * np.cos(t/hT*np.pi) + or2 * np.sin(t/hT*np.pi)

    view_dir_1 = np.array([0.383,-0.798,0.465])
    view_dir_2 = np.array([0.196,-0.562,-0.804])
    half_circle = 40
    testbed.view_dir = view_dir_1
    testbed.scale = 1.24

    # render image
    if not args.gui:
        testbed.hybrid_render = 1
        n_frames = 1
        start_frame = args.start_frame
        resolution_low = [20, 10]
        resolution = [args.width or 1920, args.height or 1080]

        testbed.background_color = [1.000, 1.000, 1.000, 0.000]
        testbed.exposure = -5.0
        testbed.fov, testbed.aperture_size, testbed.slice_plane_z = 50.625, 0.000, 0.000
        testbed.autofocus = True

        start_view_dir = [-0.867, -0.120, -0.426]
        start_look_at = [0.603, 0.471, 0.389]
        start_scale = 0.636

        end_view_dir = [0.559, -0.107, 0.822]
        end_look_at = [0.624, 0.474, 0.412]
        end_scale = 0.700

        testbed.hybrid_render = 1

        os.makedirs("tmp", exist_ok=True)

        for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc="Rendering video"):
            def interpolate_scalar(start_scale, end_scale, frame_idx, n_frames):
                print("Computing scale at frame {}/{}.".format(frame_idx, n_frames))

                interpolated_scale = start_scale + (end_scale - start_scale) * (frame_idx / (n_frames - 1))
                return interpolated_scale
            def interpolate_vec3(start_vec, end_vec, frame_idx, n_frames):
                print("Computing view dir at frame {}/{}.".format(frame_idx, n_frames))

                start = np.array(start_vec)
                end = np.array(end_vec)

                interpolated = start + (end - start) * (frame_idx / (n_frames - 1))
                return interpolated.tolist()

            testbed.view_dir = [-0.393, -0.12, 0.906]
            testbed.look_at = [0.494, 0.48, 0.495]
            testbed.scale = 0.52
            frame = testbed.render(resolution[0], resolution[1], args.video_spp, True,
                                   float(i)/n_frames, float(i + 1)/n_frames, args.video_fps,
                                   shutter_fraction=0.0001)
            write_image(f"tmp/{i:04d}.jpg", frame, quality=100)
    # end video

    render_frame = 0
    tqdm_last_update = 0
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():
                render_frame += 1
