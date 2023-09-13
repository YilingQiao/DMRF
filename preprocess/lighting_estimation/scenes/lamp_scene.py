OBSMAP_PREVIEW_DIR = "mitsuba3/outputs/observation_map/lamp"
BASE_DIR = "dmrf/outputs/data-lamp_sdfstudio/monosdf/2023-04-27_093146"
EMISSION_TEXTURE_FILEPATH = "mitsuba3/outputs/lamp_2/6_0_uv_emission.exr"
METADATA_PATH = "dmrf/data/lamp_sdfstudio/meta_data.json"

EMISSION_CLIP_THRESHOLD = 0.1
# idea - brightness clustering to separate the primary light sources from the secondary bounce lighting.
# The idea is to find the optimal low clipping threshold to separate the main light sources.
# Kind of like pruning as the optimization proceeds. The question is how to choose the right parameter automatically.

EMISSION_LR = 0.01
SPEC_ROUGH_LR = 0.05
ALBEDO_LR = 0.1

MAX_DEPTH = 3
L1_WEIGHT = 25
USE_OBS_FREQ_WEIGHT = True
WEIGHT_CLIP_LOW = 0.2
UV_CLIP_AFTER_EPOCH = True
UV_CLIP_RATE = 2
CLIP_EXTREME_COLORS = True

SCENE = {
	"type": "scene",
	"local_scene": {
		"type": "obj",
		"filename": "dmrf/meshes/textured/lamp/mesh.obj",
		"emission_texture": {
			"type": "area",
			"radiance": {
				"type": "bitmap",
				"filename": "mitsuba3/scenes/texture_templates/emission_texture_1024.exr",
				"raw": True
			}
		},
		"principled_bsdf": {
			"type": "principled",
			"base_color": {
				"type": "bitmap",
				"filename": "dmrf/meshes/textured/lamp/albedo_initial.exr",
				"raw": True
			},
			"roughness": {
				"type": "bitmap",
				"filename": "mitsuba3/scenes/texture_templates/roughness_texture_1024.exr",
				"raw": True
			},
			"spec_trans": {
				"type": "bitmap",
				"filename": "mitsuba3/scenes/texture_templates/specular_texture_mid_1024.exr",
				"raw": True
			},
			"specular": 1.0
		}
	}
}
