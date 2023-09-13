import json


def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.endswith(".json")
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def load_segmentation_masks(segmentation_masks_path):
    print("\nLoading segmentation masks...")
    return load_from_json(segmentation_masks_path)


def load_nerf_camera_data(camera_poses_json_path):
    print("\nLoading camera pose data...")
    return load_from_json(camera_poses_json_path)

