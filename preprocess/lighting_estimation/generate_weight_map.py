import drjit as dr
import math
import mitsuba as mi
import numpy as np
import os
import shutil
from datetime import datetime
from experiments import conference_scene
from tqdm import tqdm

mi.set_variant('cuda_ad_rgb')
from utils.sensor_utils import load_nerf_camera_data, load_from_json, load_segmentation_masks

import lamp_scene as current_scene

REFERENCE_SCENE = {
	"type": "scene",
	"integrator": {
		"type": "prb",
		# "max_depth": 12
	},
	"local_light_sources": {
		"type": "obj",
		"filename": "/home/awgao/git/nerfstudio/meshes/garden/mesh.obj",
		"emission_texture": {
			"type": "area",
			"radiance": {
				"type": "bitmap",
				"filename": "/home/awgao/git/gi_nerf/meshes/textured/garden2/material_0.png"
				# "filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/emission_texture_1024.exr",
			}
		}
	},
	"local_scene": {
		"type": "obj",
		"filename": "/home/awgao/git/nerfstudio/meshes/garden/mesh.obj",
		"principled_bsdf": {
			"type": "principled",
			"base_color": {
				"type": "bitmap",
				"filename": "/home/awgao/git/gi_nerf/meshes/textured/garden2/material_0.png"
			},
			"roughness": {
				"type": "bitmap",
				"filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/roughness_texture_1024.exr"
			},
			"spec_trans": {
				"type": "bitmap",
				"filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/specular_texture_1024.exr"
			},
			"specular": 0.5
		}
	}
}

EMISSION_ONLY_SCENE = {
	"type": "scene",
	"integrator": {
		"type": "prb",
		# "max_depth": 12
	},
	"local_light_sources": {
		"type": "obj",
		"filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/mesh.obj",
		"emission_texture": {
			"type": "area",
			"radiance": {
				"type": "bitmap",
				# "filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/material_0.png"
				"filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/replica-scan2-emission_uv.png",
			}
		}
	}
}

PATH_INTEGRATOR = {
	"type": "prb",
	"max_depth": -1,
	"hide_emitters": False
}

PRB_INTEGRATOR = {
	"type": "prb"
}

AOV_INTEGRATOR = {
	"type": "aov",
	"aovs": "uv:uv"
}

SCENE = {
	"type": "scene",
	# "emitter_sphere": {
	#     'type': 'sphere',
	#     'emitter': {
	#         'type': 'area',
	#         'radiance': {
	#             'type': 'rgb',
	#             'value': 100.0,
	#         }
	#     },
	#     "radius": 0.05,
	#     "center": [0, 0, 0]
	# },
	# "area": {
	#     "type": "obj",
	#     "filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/mesh.obj",
	#     "emission_texture": {
	#         "type": "area",
	#         "radiance": {
	#             "type": "bitmap",
	#             # "filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/material_0_crushed.exr"
	#             "filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/checkerboard.exr",
	#         }
	#     }
	# },
	# "directional": {
	#     "type": "directional",
	#     "direction": [1.0, 1.0, 1.0],
	#     "irradiance": {
	#         "type":"rgb",
	#         "value": 0.25
	#     }
	# },
	"local_scene": {
		"type": "obj",
		"filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/mesh.obj",
		# "blend": {
		#     "type": "blendbsdf",
		#     "weight": {
		#         "type": "bitmap",
		#         "filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/material_0_crushed.exr"
		#     },
		#     "bsdf_0": {
		#
		#     },
		#     "bsdf_1": {
		#
		#     }
		# }
		# "mask": {
		#     "type": "mask",
		#     "opacity": {
		#         "type": "bitmap",
		#         "filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/material_0_inverted.png"
		#     },
		#     "principled_bsdf": {
		#         "type": "principled",
		#         "base_color": {
		#             "type": "bitmap",
		#             # "filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/specular_texture_1024.exr",
		#             "filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/material_0.png"
		#         },
		#         "roughness": {
		#             "type": "bitmap",
		#             "filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/roughness_texture_1024.exr"
		#         },
		#         "spec_trans": {
		#             "type": "bitmap",
		#             "filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/specular_texture_1024.exr"
		#         },
		#         "specular": 0.5
		#     },
		# },
		"emission_texture": {
			"type": "area",
			"radiance": {
				"type": "bitmap",
				"filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/emission_texture_1024.exr",
				# "filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/material_0_crushed_bw.exr",
				"raw": True
			}
		},
		"principled_bsdf": {
			"type": "principled",
			"base_color": {
				"type": "bitmap",
				# "filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/specular_texture_1024.exr",
				"filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/material_0.png"
			},
			"roughness": {
				"type": "bitmap",
				"filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/roughness_texture_1024.exr"
			},
			"spec_trans": {
				"type": "bitmap",
				"filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/specular_texture_1024.exr"
			},
			"specular": 0.5
		}
	}
}

# EMISSION_KEY = "area.emitter.radiance.data"
EMISSION_KEY = "local_scene.emitter.radiance.data"


# SPECULAR_KEY = "local_scene.bsdf.spec_trans.data"
# ROUGHNESS_KEY = "local_scene.bsdf.roughness.data"
# ALBEDO_KEY = "local_scene.bsdf.base_color.data"


class Optimizer:
	def __init__(self, scene, base_dir, preview_dir):
		self.spp = 512
		self.num_epochs = 1
		self.save_preview_rate = 2
		self.preview_dir = preview_dir
		os.makedirs(self.preview_dir, exist_ok=True)

		print("\nLoading scene...")
		self.scene = mi.load_dict(scene)
		# self.emission_only_scene = mi.load_dict(EMISSION_ONLY_SCENE)

		# self.reference_scene = mi.load_dict(REFERENCE_SCENE)
		self.params_to_optimize = [
			EMISSION_KEY,
			# ALBEDO_KEY,
			# SPECULAR_KEY,
			# ROUGHNESS_KEY
		]

		print("\nTraversing mitsuba parameters...")
		self.params = mi.traverse(self.scene)
		print(self.params)
		self.emission_lr = 0.01
		self.spec_rough_lr = 0.005
		self.albedo_lr = 0.1

		self.emission_optimizer = mi.ad.Adam(lr=self.emission_lr)
		self.spec_rough_optimizer = mi.ad.Adam(lr=self.spec_rough_lr)
		self.albedo_optimizer = mi.ad.Adam(lr=self.albedo_lr)

		self.emission_optimizer[EMISSION_KEY] = self.params[EMISSION_KEY]

		self.params.update(self.emission_optimizer)

		self.cameras = load_nerf_camera_data(os.path.join(base_dir, "transformed_cameras.json"))

		self.image_filenames = np.array(self.cameras["image_filenames"])
		self.camera_to_worlds = np.array(self.cameras["cameras"]["camera_to_worlds"])
		self.fx = self.cameras["cameras"]["fx"]
		self.width = self.cameras["cameras"]["width"]
		self.height = self.cameras["cameras"]["height"]
		self.observation_heatmap = np.zeros((1024, 1024))

	def save(scene):
		pass

	# def load_reference_images(self):
	#     print("\nLoading reference images...")
	#     images = []
	#     for path in tqdm(self.image_filenames):
	#         images.append(mi.Bitmap(path))
	#     print("\nDone loading reference images.")
	#     return images

	def disable_grads(self):
		for each in self.params_to_optimize:
			dr.disable_grad(each)

	# def optimize_albedo(self):
	#     """
	#     Base color -coarse-to-fine.
	#     """
	#     print("\nOptimizing base color texture...")
	#
	#     self.disable_grads()
	#     dr.enable_grad(self.params[ALBEDO_KEY])
	#     self.params.update();
	#
	#     self.gradient_descent_loop(self.albedo_optimizer, [ALBEDO_KEY])
	#
	# def optimize_specular_roughness(self):
	#     print("\nOptimizing specular and roughness textures...")
	#
	#     """
	#     Specular & Roughness - course-to-fine.
	#     """
	#     self.disable_grads()
	#     dr.enable_grad(self.params[SPECULAR_KEY])
	#     dr.enable_grad(self.params[ROUGHNESS_KEY])
	#     self.params.update();
	#
	#     self.gradient_descent_loop(self.spec_rough_optimizer, [SPECULAR_KEY, ROUGHNESS_KEY])

	def optimize_emission(self):
		"""
		1 - Binary-mask pass (with segmentation regularizer).
		2 - Spatially-varying pass.

		Sample code:
		opt = mi.ad.Adam(lr=0.05)
		opt[key] = params[key]
		params.update(opt);
		"""
		print("\nOptimizing emission texture...")

		# self.disable_grads()
		# dr.enable_grad(self.params[EMISSION_KEY])
		# self.params.update();

		# Initial pass
		self.gradient_descent_loop(self.emission_optimizer, [EMISSION_KEY])

	def mse(self, rendered, reference_image):
		return dr.mean(dr.sqr(rendered - reference_image))

	def l1_loss(self, arr):
		"""
		Encourage most emission values to be zero.
		"""
		return dr.mean(dr.abs(arr))  # / (self.width[0][0] * self.height[0][0])

	def segmentation_emission_loss(self, masks, uv_coords):
		"""
		For the first pass of emission optimization, only allow one emission value per segmented object.
		This can be enforced as a soft constraint, by introducing an L1/L2 deviation loss term.

		mask format (e.g. for a 3x4 image):
		[[bool, bool, bool, bool],
		 [bool, bool, bool, bool],
		 [bool, bool, bool, bool]]

		Pseudocode:
		for each segmentation mask (its associated bundle of rays):
			loss += MSE/MAE( texel_values - mean_texel_value )
		"""
		loss = 0
		uv_coords = np.array(np.round(uv_coords * self.params[EMISSION_KEY].shape[0])).astype(np.uint8)
		raveled = dr.ravel(self.emission_optimizer[EMISSION_KEY])

		for i, mask in enumerate(masks):
			# if i > 0: break
			print("\nComputing segmentation loss for mask {} / {}...".format(i + 1, len(masks)))
			mask = mask["segmentation"]

			r_indices = []
			g_indices = []
			b_indices = []
			for j in range(len(mask)):
				for k in range(len(mask[0])):
					if mask[j][k]:
						u = uv_coords[j, k, 0]
						v = uv_coords[j, k, 1]
						flattened_start_index = (u * len(mask[0]) * 3) + (3 * v)
						r_indices.append(flattened_start_index)
						g_indices.append(flattened_start_index + 1)
						b_indices.append(flattened_start_index + 2)

			texel_values_r = dr.gather(dtype=type(raveled), source=raveled, index=r_indices)
			texel_values_g = dr.gather(dtype=type(raveled), source=raveled, index=g_indices)
			texel_values_b = dr.gather(dtype=type(raveled), source=raveled, index=b_indices)

			mask_mse = dr.mean(dr.sqr(texel_values_r - dr.mean(texel_values_r))) + \
					   dr.mean(dr.sqr(texel_values_g - dr.mean(texel_values_g))) + \
					   dr.mean(dr.sqr(texel_values_b - dr.mean(texel_values_b)))

			loss += mask_mse
			print(loss)
		return loss

	# def uv_to_px(self, u, v):

	def accumulate_observation_heatmap(self, uv_coords):
		# uv_coords = np.array(uv_coords)
		# print(uv_coords.max(), uv_coords.min())

		uv_coords = np.array(np.floor(uv_coords * 1024)).astype(np.uint16)
		for i in range(len(uv_coords)):
			for j in range(len(uv_coords[0])):
				u = uv_coords[i, j, 0]
				v = uv_coords[i, j, 1]
				self.observation_heatmap[u, v] += (1 / 255)

	def tv_loss(self):
		"""
		TV Denoising regularizer over general texture images.
		"""
		pass

	def compute_fov(self, width, fx):
		return 2 * math.atan(width / 2.0 / fx) * (180 / math.pi)

	def load_uv_sensor(self, img_index):
		camera_to_world = mi.ScalarTransform4f(np.concatenate((self.camera_to_worlds[img_index],
															   np.array([[0, 0, 0, 1]])), axis=0))
		correction = mi.ScalarTransform4f.rotate([1, 0, 0], 0).rotate([0, 1, 0], 180).rotate([0, 0, 1], 0)
		camera_to_world = camera_to_world @ correction
		fov = self.compute_fov(self.width[img_index][0], self.fx[img_index][0])

		return mi.load_dict({
			'type': 'perspective',
			'fov': fov,
			'to_world': camera_to_world,
			'sampler': {
				'type': 'independent',
				'sample_count': 1
			},
			'film': {
				'type': 'hdrfilm',
				'width': self.width[img_index][0],
				'height': self.height[img_index][0],
				'rfilter': {
					'type': 'box',
				},
				'pixel_format': 'rgb',
			}
		})

	def gradient_descent_loop(self, optimizer, params_to_optimize):  # , keys):
		for i in range(self.num_epochs):
			print("Epoch {} / {}".format(i, self.num_epochs))
			j = 0
			for img_index in tqdm(range(len(self.camera_to_worlds))):
				reference_image = mi.Bitmap(self.image_filenames[img_index]).convert(
					component_format=mi.Struct.Type.Float32)

				# Forward render
				uv_sensor = self.load_uv_sensor(img_index)

				self.disable_grads()
				uv_coords = mi.render(self.scene,
									  self.params,
									  spp=1,
									  sensor=uv_sensor,
									  integrator=mi.load_dict(AOV_INTEGRATOR))  # [:, :, 3:5]
				uv_coords = mi.TensorXf(dict(uv_sensor.film().bitmap().split())["uv"])
				self.accumulate_observation_heatmap(uv_coords)

				if j % self.save_preview_rate == 0:
					image_outpath = os.path.join(self.preview_dir,
												 "{}_observation_heatmap.png".format(str(datetime.now())))
					mi.util.convert_to_bitmap(self.observation_heatmap).write(path=image_outpath)

				# Update scene state to the new optimized values
				self.params.update(optimizer)

				j += 1

		print('\nOptimization complete.')

	def optimize_scene(self):
		# Prepare the scene
		shutil.rmtree(self.preview_dir)
		os.makedirs(self.preview_dir, exist_ok=True)

		"""
		Optimization procedure:
		"""
		self.optimize_emission()
		# self.optimize_specular_roughness()
		self.optimize_albedo()


if __name__ == "__main__":
	scene_optimizer = Optimizer(current_scene.SCENE, base_dir=current_scene.BASE_DIR,
								preview_dir=current_scene.OBSMAP_PREVIEW_DIR)

	print("\nMain optimization loop:")
	scene_optimizer.optimize_scene()
	scene_optimizer.save()
