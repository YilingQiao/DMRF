import json
import os

import numpy as np


def load_event_data(events_json_path):
    with open(events_json_path, 'r') as json_file:
        event_data = json.load(json_file)
        offset = 0.5
        for event in event_data["events"]:
            event["bbox"] = np.array(event["bbox"]).astype(np.float32)

            # align LERF coordinate system with Instant-NGP		
            event["bbox"] += offset
            event["bbox"][0:2], event["bbox"][2:4], event["bbox"][4:6] = event["bbox"][2:4], event["bbox"][4:6], event[
                                                                                                                     "bbox"][
                                                                                                                 0:2]  # XYZ --> ZXY

            # event["bbox"][2:] = -event["bbox"][2:]

            event["bbox"] = np.array(event["bbox"])
        print(f"Loaded event data!\n{event_data}")
        return event_data


def get_lighting_from_bbox(camera_position,
                           bbox_center,
                           bbox_h,
                           bbox_w,
                           bbox_l,
                           area_light_intensity=4.0,
                           spot_light_intensity=24.0,
                           spot_light_start_angle=np.pi / 16,
                           spot_light_elevation=3 * np.pi / 8):
    view_dir = bbox_center - camera_position
    view_dir = np.array([view_dir[0], view_dir[1], 0.0])
    view_dir = view_dir / np.linalg.norm(view_dir)
    min_lw = min(bbox_w, bbox_l)
    mean_lw = (bbox_w + bbox_l) / 2.0

    light_sources = []
    spot_light_position = bbox_center + (1.5 * min_lw) * view_dir
    spot_light_z = np.linalg.norm(spot_light_position - bbox_center) * np.tan(np.array([spot_light_elevation]))
    spot_light_position = spot_light_position + np.array([0.0, 0.0, spot_light_z])

    # spot_light_position[0] = np.clip(spot_light_position[0], -7.0, 7.0)
    # spot_light_position[1] = np.clip(spot_light_position[1], -7.0, 7.0)
    # spot_light_position[2] = np.min(spot_light_position[2], np.array(4.0))
    #
    # look_at = bbox_center
    # look_at[0] = np.clip(spot_light_position[0], -7.8, 7.8)
    # look_at[1] = np.clip(spot_light_position[0], -7.8, 7.8)
    # look_at[2] = np.clip(spot_light_position[0], -2.0, 2.0)

    light_sources.append(
        {
            "type": "spot",
            "intensity": np.array([spot_light_intensity,
                                   spot_light_intensity,
                                   spot_light_intensity * 0.75]),
            "position": spot_light_position,
            "look_at": bbox_center,
            "falloff_start_angle_rad": np.array(spot_light_start_angle),
            "total_width_angle_rad": np.array(spot_light_start_angle * 1.1)
        }
    )
    # area_position_z = np.min(bbox_h * 1.5, np.array(2.0))
    # area_position_z = np.clip(area_position_z, -2.0, 2.0)

    # light_sources.append(
    #     {
    #         "type": "area",
    #         "intensity": np.array([area_light_intensity,
    #                                area_light_intensity,
    #                                area_light_intensity]),
    #         "position": bbox_center + np.array([0.0, 0.0, area_position_z]),
    #         "look_at": bbox_center,
    #         "axis_x": np.array([1.0, 0.0, 0.0]),
    #         "axis_y": np.array([0.0, 1.0, 0.0]),
    #         "area": np.array([mean_lw ** 2]),
    #         "n_samples": 128
    #     }
    # )
    return light_sources


def get_global_lights(camera_position, scene_events):
    """
    This function will take in a series of bounding box / time stamps,
    and output a lighting configuration.

    Bounding box data will be used to determine lighting positions, taking into account the camera viewing angle.

    Input spec:
    [
      {
        "keyword": "table",
        "start_frame": 9,
        "end_frame": 26,
        "bbox": []==================
      },
      { ... },
    ]

    Output spec:
    [
      {
        "start_frame": 9,
        "end_frame": 26,
        "sources": [
            {
                "type": "spot",
                "intensity": np.array([16.0, 16.0, 16.0]),
                "position": np.array([-0.1, -2.0, 2.0]),
                "look_at": np.array([-0.1, -0.2, 0.3]),
                "falloff_start_angle_rad": np.array(np.pi / 32),
                "total_width_angle_rad": np.array(3 * np.pi / 64)
            },
            {
                "type": "area",
                "intensity": np.array([1.2, 1.2, 1.2]),
                "position": np.array([0.4, 0.0, 1.5]),
                "look_at": np.array([0.4, 0.0, 0.0]),
                "area": np.array([0.7 ** 2]),
                "n_samples": 128
            }
        ]
      },
      { ... },
    ]
    """
    global_light_sources = []
    for event in scene_events["events"]:
        bbox = event["bbox"]
        bbox_center = np.array([(bbox[0] + bbox[1]) / 2.0,
                                (bbox[2] + bbox[3]) / 2.0,
                                (bbox[4] + bbox[5]) / 2.0])
        bbox_h = np.abs(bbox[5] - bbox[4])
        bbox_w = np.abs(bbox[3] - bbox[2])
        bbox_l = np.abs(bbox[1] - bbox[0])
        event["sources"] = get_lighting_from_bbox(camera_position,
                                                  bbox_center,
                                                  bbox_h,
                                                  bbox_w,
                                                  bbox_l)
        global_light_sources.append(event)
    return global_light_sources


def modify_config_for_timestep(original_config_path, tmp_config_path, current_frame_idx):
    # load base config for scene
    with open(original_config_path, "r") as original_config_file:
        scene_config = json.load(original_config_file)

    # modify base config's OBJ filepaths for the current frame
    for mesh in scene_config["objfile"]:
        if "shadow_catcher" not in mesh["dir"]:
            mesh["dir"] = mesh["dir"].replace(".obj", f"{current_frame_idx}.obj")

    with open(tmp_config_path, "w") as tmp_config_file:
        json.dump(scene_config, tmp_config_file)
