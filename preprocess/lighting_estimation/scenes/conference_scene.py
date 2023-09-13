OBSMAP_PREVIEW_DIR = "mitsuba3/outputs/observation_map/conference"
BASE_DIR = "dmrf/outputs/data-conference_sdfstudio/monosdf/2023-04-26_170215"
EMISSION_TEXTURE_FILEPATH = "mitsuba3/outputs/conference_2/8_0_uv_emission.exr"
METADATA_PATH = "dmrf/data/conference_sdfstudio/meta_data.json"

EMISSION_CLIP_THRESHOLD = 0.2

EMISSION_LR = 0.01
SPEC_ROUGH_LR = 0.05
ALBEDO_LR = 0.1

MAX_DEPTH = 3
L1_WEIGHT = 10
USE_OBS_FREQ_WEIGHT = True
WEIGHT_CLIP_LOW = 0.1
UV_CLIP_AFTER_EPOCH = True
UV_CLIP_RATE = 2
CLIP_EXTREME_COLORS = False

SCENE = {
	"type": "scene",
	"local_scene": {
		"type": "obj",
		"filename": "dmrf/meshes/textured/conference/mesh.obj",
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
				"filename": "dmrf/meshes/textured/conference/material_0.png",
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
