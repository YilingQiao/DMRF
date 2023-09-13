import drjit as dr
import math
import mitsuba as mi
import numpy as np
import os
import shutil
from datetime import datetime
from tqdm import tqdm

mi.set_variant('cuda_ad_rgb')
from utils.sensor_utils import load_nerf_camera_data, load_from_json, load_segmentation_masks

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

SCENE = {
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
				"filename": "/home/awgao/git/mitsuba3/scenes/texture_templates/specular_texture_1024.exr",
			}
		}
	},
	"local_scene": {
		"type": "obj",
		"filename": "/home/awgao/git/gi_nerf/meshes/textured/replica-scan2/mesh.obj",
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

EMISSION_KEY = "local_light_sources.emitter.radiance.data"
SPECULAR_KEY = "local_scene.bsdf.spec_trans.data"
ROUGHNESS_KEY = "local_scene.bsdf.roughness.data"
BASE_COLOR_KEY = "local_scene.bsdf.base_color.data"


class Optimizer:
	def __init__(self, scene, base_dir):
		self.spp = 64
		self.num_epochs = 100
		self.save_preview_rate = 2
		self.preview_dir = "/home/awgao/git/mitsuba3/outputs/preview_emission_only"

		print("\nLoading scene...")
		self.scene = mi.load_dict(scene)
		# self.emission_only_scene = mi.load_dict(EMISSION_ONLY_SCENE)

		# self.reference_scene = mi.load_dict(REFERENCE_SCENE)
		self.params_to_optimize = [
			EMISSION_KEY,
			BASE_COLOR_KEY,
			# SPECULAR_KEY,
			# ROUGHNESS_KEY
		]

		print("\nTraversing mitsuba parameters...")
		self.params = mi.traverse(self.scene)

		self.emission_lr = 1.0
		self.spec_rough_lr = 0.005
		self.albedo_lr = 0.1

		self.emission_optimizer = mi.ad.Adam(lr=self.emission_lr)
		self.spec_rough_optimizer = mi.ad.Adam(lr=self.spec_rough_lr)
		self.albedo_optimizer = mi.ad.Adam(lr=self.albedo_lr)

		self.emission_optimizer[EMISSION_KEY] = self.params[EMISSION_KEY]
		# self.albedo_optimizer[BASE_COLOR_KEY] = self.params[BASE_COLOR_KEY]
		# self.spec_rough_optimizer[SPECULAR_KEY] = self.params[SPECULAR_KEY]
		# self.spec_rough_optimizer[ROUGHNESS_KEY] = self.params[ROUGHNESS_KEY]

		self.params.update(self.emission_optimizer)
		# self.params.update(self.albedo_optimizer)
		# self.params.update(self.spec_rough_optimizer)

		self.cameras = load_nerf_camera_data(os.path.join(base_dir, "transformed_cameras.json"))
		# self.segmentation_masks = load_segmentation_masks(os.path.join(base_dir, "segmentation_masks.json"))["masks"]

		self.image_filenames = np.array(self.cameras["image_filenames"])
		self.camera_to_worlds = np.array(self.cameras["cameras"]["camera_to_worlds"])
		self.fx = self.cameras["cameras"]["fx"]
		self.width = self.cameras["cameras"]["width"]
		self.height = self.cameras["cameras"]["height"]

		# self.segmentation_masks = self.load_segmentation_masks()

	def save(scene):
		pass

	def load_segmentation_masks(self):
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

	def optimize_albedo(self):
		"""
		Base color -coarse-to-fine.
		"""
		print("\nOptimizing base color texture...")

		self.disable_grads()
		dr.enable_grad(self.params[BASE_COLOR_KEY])
		self.params.update();

		self.gradient_descent_loop(self.albedo_optimizer, [BASE_COLOR_KEY])

	def optimize_specular_roughness(self):
		print("\nOptimizing specular and roughness textures...")

		"""
		Specular & Roughness - course-to-fine.
		"""
		self.disable_grads()
		dr.enable_grad(self.params[SPECULAR_KEY])
		dr.enable_grad(self.params[ROUGHNESS_KEY])
		self.params.update();

		self.gradient_descent_loop(self.spec_rough_optimizer, [SPECULAR_KEY, ROUGHNESS_KEY])

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

		self.disable_grads()
		dr.enable_grad(self.params[EMISSION_KEY])
		self.params.update();

		# Initial pass
		self.gradient_descent_loop(self.emission_optimizer, [EMISSION_KEY])

	def mse(self, rendered, reference_image):
		return dr.mean(dr.sqr(rendered - reference_image))

	def l1_loss(self, arr):
		"""
		Encourage most emission values to be zero.
		"""
		return dr.mean(dr.abs(arr))  # / (self.width[0][0] * self.height[0][0])

	# def segmentation_emission_loss(self, masks, emission_only_render):
	#     """
	#     For the first pass of emission optimization, only allow one emission value per segmented object.
	#     This can be enforced as a soft constraint, by introducing an L1/L2 deviation loss term.
	#
	#     mask format (e.g. for a 3x4 image):
	#     [[bool, bool, bool, bool],
	#      [bool, bool, bool, bool],
	#      [bool, bool, bool, bool]]
	#
	#     Pseudocode:
	#     for each segmentation mask (its associated bundle of rays):
	#         loss += MSE/MAE( texel_values - mean_texel_value )
	#     """
	#     loss = 0
	#     for mask in masks:
	#         mask = dr.cuda.TensorXf(mask["segmentation"])
	#         masked_image = dr.select(mask, emission_only_render)
	#         mask_mae = dr.mean(dr.abs(masked_image - dr.mean(masked_image)))
	#         loss += mask_mae
	#     return loss

	def tv_loss(self):
		"""
		TV Denoising regularizer over general texture images.
		"""
		pass

	def compute_fov(self, width, fx):
		return 2 * math.atan(width / 2.0 / fx) * (180 / math.pi)

	def load_sensor(self, img_index):
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
					'type': 'tent',
				},
				'pixel_format': 'rgb',
			}
		})

	def sample_texture_space(self):
		"""
		- Uniformly sample in texture space.
		- Determine inverse-UV mapping from texture coordinates to 3D coordinates
		- Render pixels from training poses where those 3D coordinates are visible
		- Optimization step
		"""
		pass

	def gradient_descent_loop(self, optimizer, params_to_optimize):  # , keys):
		for i in range(self.num_epochs):
			print("Epoch {} / {}".format(i, self.num_epochs))
			j = 0
			for img_index in tqdm(range(len(self.camera_to_worlds))):
				reference_image = mi.Bitmap(self.image_filenames[img_index]).convert(
					component_format=mi.Struct.Type.Float32)

				# Forward render
				sensor = self.load_sensor(img_index)
				# reference_image = dr.detached_t(mi.render(self.reference_scene, self.params, spp=self.spp, sensor=sensor))
				rendered = mi.render(self.scene, self.params, spp=self.spp, sensor=sensor)
				# emission_only_render = mi.render(self.emission_only_scene, self.params, spp=self.spp, sensor=sensor)

				# Optimization step
				loss = self.mse(rendered, reference_image) + 0.1 * self.l1_loss(self.params[EMISSION_KEY])
				# self.segmentation_emission_loss(self.segmentation_masks[img_index], emission_only_render)
				# + self.segmentation_loss()

				# dr.backward(loss)
				# optimizer.step()

				if j % self.save_preview_rate == 0:
					image_outpath = os.path.join(self.preview_dir, "{}.png".format(str(datetime.now())))
					mi.util.convert_to_bitmap(rendered).write(path=image_outpath)

					image_outpath = os.path.join(self.preview_dir, "{}_ref.png".format(str(datetime.now())))
					mi.util.convert_to_bitmap(reference_image).write(path=image_outpath)

					image_outpath = os.path.join(self.preview_dir, "{}_emission_uv.png".format(str(datetime.now())))
					mi.util.convert_to_bitmap(self.params[EMISSION_KEY]).write(path=image_outpath)

					# image_outpath = os.path.join(self.preview_dir, "{}_albedo_uv.png".format(str(datetime.now())))
					# mi.util.convert_to_bitmap(self.params[BASE_COLOR_KEY]).write(path=image_outpath)
					#
					# image_outpath = os.path.join(self.preview_dir, "{}_specular_uv.png".format(str(datetime.now())))
					# mi.util.convert_to_bitmap(self.params[SPECULAR_KEY]).write(path=image_outpath)
					#
					# image_outpath = os.path.join(self.preview_dir, "{}_roughness_uv.png".format(str(datetime.now())))
					# mi.util.convert_to_bitmap(self.params[ROUGHNESS_KEY]).write(path=image_outpath)

				# Post-process optimized parameters to ensure legal color values
				for param in params_to_optimize:
					optimizer[param] = dr.clamp(optimizer[param], 0.0, 10.0)

				# Update scene state to the new optimized values
				# self.params.update(optimizer)

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
	base_dir = "/home/awgao/git/gi_nerf/outputs/data-replica-scan2/monosdf/2023-04-25_230533"
	scene_optimizer = Optimizer(EMISSION_ONLY_SCENE, base_dir=base_dir)

	print("\nMain optimization loop:")
	scene_optimizer.optimize_scene()
	scene_optimizer.save()
