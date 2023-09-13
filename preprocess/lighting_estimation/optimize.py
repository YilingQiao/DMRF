import lamp scene as current_scene
import contextlib
import drjit as dr
import math
import mitsuba as mi
import numpy as np
import os
import random
import shutil
from tqdm import tqdm

mi.set_variant('cuda_ad_rgb')
from utils.sensor_utils import load_nerf_camera_data, load_segmentation_masks
from utils.post_process_obj import export_lighting_for_hybrid_nerf

PATH_INTEGRATOR = {
	"type": "prb",
	"max_depth": current_scene.MAX_DEPTH,
	"hide_emitters": False
}

AOV_INTEGRATOR = {
	"type": "aov",
	"aovs": "uv:uv"
}

EMISSION_KEY = "local_scene.emitter.radiance.data"
SPECULAR_KEY = "local_scene.bsdf.specular"
SPECULAR_TRANS_KEY = "local_scene.bsdf.spec_trans.data"
ROUGHNESS_KEY = "local_scene.bsdf.roughness.data"
ALBEDO_KEY = "local_scene.bsdf.base_color.data"


class Optimizer:
	def __init__(self, scene, base_dir):
		self.base_dir = base_dir
		self.spp = 16
		self.num_epochs = 16
		self.save_preview_rate = 5
		self.preview_dir = "/home/awgao/git/mitsuba3/outputs/preview_renders"
		os.makedirs(self.preview_dir, exist_ok=True)

		print("\nLoading scene...")
		self.scene = mi.load_dict(scene)
		self.params_to_optimize = [
			EMISSION_KEY,
			ALBEDO_KEY,
			SPECULAR_KEY,
			ROUGHNESS_KEY
		]

		print("\nTraversing mitsuba parameters...")
		self.params = mi.traverse(self.scene)
		self.emission_lr = current_scene.EMISSION_LR
		self.spec_rough_lr = current_scene.SPEC_ROUGH_LR
		self.albedo_lr = current_scene.ALBEDO_LR
		print(self.params)

		self.emission_optimizer = mi.ad.Adam(lr=self.emission_lr, mask_updates=True)
		self.spec_rough_optimizer = mi.ad.Adam(lr=self.spec_rough_lr, mask_updates=True)
		self.albedo_optimizer = mi.ad.Adam(lr=self.albedo_lr, mask_updates=True)

		# self.params[EMISSION_KEY] = (self.params[EMISSION_KEY] ** 2) * 10
		# self.params.update()
		self.emission_optimizer[EMISSION_KEY] = self.params[EMISSION_KEY]
		self.spec_rough_optimizer[SPECULAR_KEY] = self.params[SPECULAR_KEY]
		self.spec_rough_optimizer[SPECULAR_TRANS_KEY] = self.params[SPECULAR_TRANS_KEY]
		self.spec_rough_optimizer[ROUGHNESS_KEY] = self.params[ROUGHNESS_KEY]
		self.albedo_optimizer[ALBEDO_KEY] = self.params[ALBEDO_KEY]

		self.params.update(self.emission_optimizer)
		self.params.update(self.spec_rough_optimizer)
		self.params.update(self.albedo_optimizer)

		self.cameras = load_nerf_camera_data(os.path.join(base_dir, "transformed_cameras.json"))
		self.segmentation_masks = None
		weights_arr = np.load(os.path.join(base_dir, "uv_weights.npy"))
		self.uv_weight_map = mi.TensorXf(
			weights_arr)  # / np.min(weights_arr[np.nonzero(weights_arr)])) # Only boost gradients (multiplicative factor greater than 1)
		self.uv_coords = None

		self.image_filenames = np.array(self.cameras["image_filenames"])
		self.camera_to_worlds = np.array(self.cameras["cameras"]["camera_to_worlds"])
		self.fx = self.cameras["cameras"]["fx"]
		self.width = self.cameras["cameras"]["width"]
		self.height = self.cameras["cameras"]["height"]

		self.aov_integrator = mi.load_dict(AOV_INTEGRATOR)
		self.prb_integrator = mi.load_dict(PATH_INTEGRATOR)

		self.uv_sensors = []
		self.rgb_sensors = []
		self.uv_coords = []
		self.reference_images = []
		for img_index in range(len(self.camera_to_worlds)):
			self.uv_sensors.append(self.load_uv_sensor(img_index))
			self.rgb_sensors.append(self.load_rgb_sensor(img_index))

			mi.render(self.scene,
					  self.params,
					  spp=1,
					  sensor=self.uv_sensors[img_index],
					  integrator=self.aov_integrator)[:, :, 3:5]
			self.uv_coords.append(mi.TensorXf(dict(self.uv_sensors[img_index].film().bitmap().split())["uv"]))
			self.reference_images.append(mi.Bitmap(self.image_filenames[img_index]).convert(
				component_format=mi.Struct.Type.Float32))

	def disable_grads(self):
		for each in self.params_to_optimize:
			dr.disable_grad(each)

	def optimize_albedo(self):
		"""
		Base color -coarse-to-fine.
		"""
		print("\nOptimizing base color texture...")
		self.gradient_descent_loop(self.albedo_optimizer, [ALBEDO_KEY])

	def optimize_specular_roughness(self):
		print("\nOptimizing specular and roughness textures...")
		self.segmentation_masks = load_segmentation_masks(os.path.join(self.base_dir, "segmentation_masks.json"))[
			"masks"]
		self.gradient_descent_loop(self.spec_rough_optimizer, [SPECULAR_KEY, SPECULAR_TRANS_KEY, ROUGHNESS_KEY])

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

		# Initial pass
		self.gradient_descent_loop(self.emission_optimizer, [EMISSION_KEY])

	def mse(self, rendered, reference_image, uv_coords):
		diff = rendered - reference_image

		if current_scene.USE_OBS_FREQ_WEIGHT:
			square = dr.sqr(diff)  # , 0.3, 1.0)
			square_r = self.apply_weighting_to_flattened_image(dr.ravel(square[:, :, 0]), uv_coords)
			square_g = self.apply_weighting_to_flattened_image(dr.ravel(square[:, :, 1]), uv_coords)
			square_b = self.apply_weighting_to_flattened_image(dr.ravel(square[:, :, 2]), uv_coords)
			return (dr.mean(square_r) + dr.mean(square_g) + dr.mean(square_b)) / 3
		else:
			return dr.mean(dr.sqr(diff))

	def l1_loss(self, arr):
		"""
		Encourage most emission values to be zero.

		Expected dim(arr) = (1024, 1024)
		i.e. arr is a texture-space image NOT render-space image
		"""
		return dr.mean(dr.abs(arr))

	def segmentation_loss(self):
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
		threshold = 2500
		total_area = 0
		for i, mask in enumerate(self.segmentation_masks):
			area = mask["area"]
			if area >= threshold:
				total_area += area

		loss = 0
		for param in self.params_to_optimize:
			if param.shape[0] == 1:
				continue

			tex_h, tex_w = self.params[param].shape[0], self.params[param].shape[1]
			uv_coords = np.array(np.floor(self.uv_coords * tex_h)).astype(np.uint16)
			raveled = dr.ravel(self.emission_optimizer[param])
			raveled_r = dr.ravel(self.emission_optimizer[param][:, :, 0])
			raveled_g = dr.ravel(self.emission_optimizer[param][:, :, 1])
			raveled_b = dr.ravel(self.emission_optimizer[param][:, :, 2])
			raveled_weights = dr.ravel(self.uv_weight_map)

			# For debugging purposes
			assert raveled[0] == self.emission_optimizer[param][0, 0, 0]
			assert raveled[1] == self.emission_optimizer[param][0, 0, 1]
			assert raveled[2] == self.emission_optimizer[param][0, 0, 2]
			assert raveled[3] == self.emission_optimizer[param][0, 1, 0]
			assert raveled[4] == self.emission_optimizer[param][0, 1, 1]
			assert raveled[5] == self.emission_optimizer[param][0, 1, 2]
			assert raveled[(10 * tex_w + 180) * 3 + 1] == self.emission_optimizer[param][10, 180, 1]

			for i, mask in enumerate(self.segmentation_masks):
				area = mask["area"]
				mask = mask["segmentation"]

				mask_h, mask_w = len(mask), len(mask[0])
				# r_indices = []
				# g_indices = []
				# b_indices = []
				indices = []
				if area >= threshold:
					# flattened_indices = (uv_coords[:, :, 0] * tex_w + uv_coords[:, :, 1]).flatten()
					for j in range(mask_h):
						for k in range(mask_w):
							if mask[j][k] == 1:
								u = uv_coords[j, k, 0]
								v = uv_coords[j, k, 1]
								indices.append(u * tex_w + v)
							# flattened_start_index = (u * tex_w + v) * 3
							# r_indices.append(flattened_start_index)
							# g_indices.append(flattened_start_index + 1)
							# b_indices.append(flattened_start_index + 2)

					# texel_values_r = dr.gather(dtype=type(raveled), source=raveled, index=r_indices)
					# texel_values_g = dr.gather(dtype=type(raveled), source=raveled, index=g_indices)
					# texel_values_b = dr.gather(dtype=type(raveled), source=raveled, index=b_indices)

					texel_values_r = dr.gather(dtype=type(raveled_r), source=raveled_r, index=indices)
					texel_values_g = dr.gather(dtype=type(raveled_g), source=raveled_g, index=indices)
					texel_values_b = dr.gather(dtype=type(raveled_b), source=raveled_b, index=indices)

					sqr_r = dr.sqr(texel_values_r - dr.mean(texel_values_r))
					sqr_g = dr.sqr(texel_values_g - dr.mean(texel_values_g))
					sqr_b = dr.sqr(texel_values_b - dr.mean(texel_values_b))

					# dim(absolute_r) = (mask_area, 1)
					# dim(indices) = (mask_area, 1)
					sqr_r = sqr_r * dr.gather(dtype=type(raveled_weights), source=raveled_weights, index=indices)
					sqr_g = sqr_g * dr.gather(dtype=type(raveled_weights), source=raveled_weights, index=indices)
					sqr_b = sqr_b * dr.gather(dtype=type(raveled_weights), source=raveled_weights, index=indices)

					mask_mae = dr.mean(sqr_r) + dr.mean(sqr_g) + dr.mean(sqr_b)
					weighted_mae = mask_mae / 3 * (area / total_area)
					loss += weighted_mae
		return loss

	def tv_loss(self):
		"""
		TV Denoising regularizer over general texture images.
		"""
		for param in self.params_to_optimize:
			if param.shape[0] == 1:
				continue

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

	def load_rgb_sensor(self, img_index):
		camera_to_world = mi.ScalarTransform4f(np.concatenate((self.camera_to_worlds[img_index],
															   np.array([[0, 0, 0, 1]])), axis=0))
		correction = mi.ScalarTransform4f.rotate([1, 0, 0], 0).rotate([0, 1, 0], 180).rotate([0, 0, 1], 0)
		camera_to_world = camera_to_world @ correction
		fov = self.compute_fov(self.width[img_index][0], self.fx[img_index][0])

		# camera_to_world = mi.ScalarTransform4f.translate([1, 1, 1]).look_at(target=[0, 0, 0], origin=[1, 1, 1], up=[0, 1, 0])

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

	def apply_weighting_to_flattened_uv(self, uv_flattened):
		weights_flattened = dr.ravel(self.uv_weight_map)
		return uv_flattened * weights_flattened

	def apply_weighting_to_flattened_image(self, image_flattened, uv_coords):
		tex_h, tex_w = self.params[EMISSION_KEY].shape[0], self.params[EMISSION_KEY].shape[1]
		uv_coords = np.array(np.floor(uv_coords * tex_h)).astype(np.uint16)

		u_slice = uv_coords[:, :, 0]
		v_slice = uv_coords[:, :, 1]
		flattened_indices = (u_slice * tex_w + v_slice).flatten()

		raveled = dr.clamp(dr.ravel(self.uv_weight_map), current_scene.WEIGHT_CLIP_LOW, 1.0)
		weights_flattened = dr.gather(dtype=type(raveled), source=raveled, index=flattened_indices)

		return image_flattened * weights_flattened

	def clip_emission_below_threshold(self, threshold=current_scene.EMISSION_CLIP_THRESHOLD):
		self.disable_grads()

		channel_average = np.mean(np.array(self.params[EMISSION_KEY].array).reshape((1024, 1024, 3)), axis=2)
		is_above_threshold = (channel_average / channel_average.max() > threshold)
		is_above_threshold = np.expand_dims(is_above_threshold, axis=2)
		is_above_threshold = mi.TensorXb(np.repeat(is_above_threshold, 3, axis=-1))

		self.emission_optimizer.reset(EMISSION_KEY)
		self.params[EMISSION_KEY] = dr.select(is_above_threshold, self.params[EMISSION_KEY],
											  0)  # self.params[EMISSION_KEY] * 0.05)

		if current_scene.CLIP_EXTREME_COLORS:
			min = np.amin(np.array(self.params[EMISSION_KEY].array).reshape((1024, 1024, 3)), axis=2)
			max = np.amax(np.array(self.params[EMISSION_KEY].array).reshape((1024, 1024, 3)), axis=2)

			is_extreme_color = max > 8 * min
			is_extreme_color = np.expand_dims(is_extreme_color, axis=2)
			is_extreme_color = np.repeat(is_extreme_color, 3, axis=-1)
			is_extreme_color = mi.TensorXb(is_extreme_color)
			self.params[EMISSION_KEY] = dr.select(is_extreme_color, 0, self.params[EMISSION_KEY])

		self.emission_optimizer[EMISSION_KEY] = self.params[EMISSION_KEY]
		self.params.update(self.emission_optimizer)

	def boost_remaining_emission(self):
		self.disable_grads()

		rand_idx = random.randint(0, len(self.reference_images) - 1)
		with contextlib.redirect_stdout(None):
			rendered = mi.render(self.scene,
								 self.params,
								 spp=self.spp,
								 sensor=self.rgb_sensors[rand_idx],
								 integrator=self.prb_integrator)

		self.params[EMISSION_KEY] = dr.clamp((self.params[EMISSION_KEY] * 2), 0.0, 100.0)
		self.params.update()
		with contextlib.redirect_stdout(None):
			rendered2 = mi.render(self.scene,
								  self.params,
								  spp=self.spp,
								  sensor=self.rgb_sensors[rand_idx],
								  integrator=self.prb_integrator)

		mse1 = self.mse(rendered, self.reference_images[rand_idx], self.uv_coords[rand_idx])
		mse2 = self.mse(rendered2, self.reference_images[rand_idx], self.uv_coords[rand_idx])

		# tqdm.write("mse1 vs. mse2: " + str(mse1) + "   " + str(mse2))

		if mse2[0] < mse1[0] * 1.5:
			self.emission_optimizer[EMISSION_KEY] *= self.params[EMISSION_KEY]
			self.params.update(self.emission_optimizer)
		else:
			self.params[EMISSION_KEY] /= 2
			self.params.update()

	def save_preview_images(self, rendered, epoch, img_index):
		image_outpath = os.path.join(self.preview_dir, "{}_{}.png".format(epoch, img_index))
		mi.util.convert_to_bitmap(rendered).write(path=image_outpath)

		image_outpath = os.path.join(self.preview_dir, "{}_{}_ref.png".format(epoch, img_index))
		mi.util.convert_to_bitmap(self.reference_images[img_index]).write(path=image_outpath)

		image_outpath = os.path.join(self.preview_dir, "{}_{}_uv_emission.png".format(epoch, img_index))
		mi.util.convert_to_bitmap(self.params[EMISSION_KEY]).write(path=image_outpath)

		image_outpath = os.path.join(self.preview_dir, "{}_{}_uv_emission.exr".format(epoch, img_index))
		mi.util.convert_to_bitmap(self.params[EMISSION_KEY], uint8_srgb=False).write(path=image_outpath)

		image_outpath = os.path.join(self.preview_dir, "{}_{}_uv_albedo.png".format(epoch, img_index))
		mi.util.convert_to_bitmap(self.params[ALBEDO_KEY]).write(path=image_outpath)

		image_outpath = os.path.join(self.preview_dir, "{}_{}_uv_spectrans.png".format(epoch, img_index))
		mi.util.convert_to_bitmap(self.params[SPECULAR_TRANS_KEY]).write(path=image_outpath)

		image_outpath = os.path.join(self.preview_dir, "{}_{}_uv_roughness.png".format(epoch, img_index))
		mi.util.convert_to_bitmap(self.params[ROUGHNESS_KEY]).write(path=image_outpath)

	def copy_code_checkpoint(self):
		shutil.copy(current_scene.__file__, os.path.join(self.preview_dir, "checkpoint"))
		shutil.copy(__file__, os.path.join(self.preview_dir, "checkpoint"))

	def gradient_descent_loop(self, optimizer, params_to_optimize):  # , keys):
		for i in tqdm(range(self.num_epochs), desc="Epoch", leave=True, position=0):
			print("Epoch {} / {}".format(i, self.num_epochs))

			for img_index in tqdm(range(len(self.camera_to_worlds)), desc="Iteration", leave=True, position=0):
				# Forward render
				self.disable_grads()
				dr.enable_grad(params_to_optimize)
				self.params.update();

				with contextlib.redirect_stdout(None):
					rendered = mi.render(self.scene,
										 self.params,
										 spp=self.spp,
										 sensor=self.rgb_sensors[img_index],
										 integrator=self.prb_integrator)
				# Optimization step
				mse = self.mse(rendered, self.reference_images[img_index], self.uv_coords[img_index])
				if SPECULAR_KEY in params_to_optimize:
					segmentation_loss = self.segmentation_loss()
					loss = mse + segmentation_loss
					print("\nMSE: {}\nSeg. loss: {}".format(mse, l1_emission, segmentation_loss))
				elif EMISSION_KEY in params_to_optimize:
					l1_emission = current_scene.L1_WEIGHT * self.l1_loss(self.params[EMISSION_KEY])
					loss = mse + l1_emission  # + segmentation_emission
				# print("\nMSE: {}\nEmission L1 loss: {}".format(mse, l1_emission))
				else:
					loss = mse
				# loss = segmentation_emission
				# print("\nMSE: {}\nEmission L1 loss: {}\nEmission seg. loss: {}".format(mse, l1_emission, segmentation_emission))

				with contextlib.redirect_stdout(None):
					dr.backward(loss)
				optimizer.step()

				if img_index % self.save_preview_rate == 0:
					self.save_preview_images(rendered, i, img_index)

				# Post-process optimized parameters to ensure legal color values
				for param in params_to_optimize:
					if param == EMISSION_KEY:
						optimizer[param] = dr.clamp(optimizer[param], 0.0, 100.0)
					else:
						optimizer[param] = dr.clamp(optimizer[param], 0.0, 1.0)

				# Update scene state to the new optimized values
				self.params.update(optimizer)
			if EMISSION_KEY in params_to_optimize:
				if current_scene.UV_CLIP_AFTER_EPOCH and (i + 1) % current_scene.UV_CLIP_RATE == 0 and img_index == len(
					self.reference_images) - 1:
					self.clip_emission_below_threshold()
					self.boost_remaining_emission()

			dr.flush_kernel_cache()
			dr.flush_malloc_cache()
		print('\nOptimization complete.')

	def optimize_scene(self):
		# Prepare the scene
		shutil.rmtree(self.preview_dir)
		os.makedirs(self.preview_dir, exist_ok=True)
		os.makedirs(os.path.join(self.preview_dir, "checkpoint"), exist_ok=True)
		self.copy_code_checkpoint()

		"""
		Optimization procedure:
		"""
		self.optimize_emission()


if __name__ == "__main__":
	scene_optimizer = Optimizer(current_scene.SCENE, base_dir=current_scene.BASE_DIR)

	print("\nMain optimization loop:")
	scene_optimizer.optimize_scene()

	# export optimized result
	export_lighting_for_hybrid_nerf(current_scene)
