# Preprocess Data

## Setup
- Ensure that necessary dependencies have been installed (repo-level requirements).
- Download our provided [**example datasets**](https://drive.google.com/drive/folders/1mGEc9dztIyxDxjUzpN22ay34HgXJya9H?usp=sharing).

## Overview:
1. Preprocess multi-exposure HDR image captures.
2. Estimate geometry.
3. Estimate light sources.

## Process Multi-Exposure HDR Image Captures
- Follow the Jupyter Notebook provided in:
`preprocess_multiexposure_hdr/multi_exposure_hdr_preprocess.ipynb`

## Estimate Geometry
- Follow the [SDFStudio installation instructions](https://github.com/autonomousvision/sdfstudio#1-installation-setup-the-environment).

- Extract geometry using the MonoSDF implementation, using the following command:
```commandline
ns-train monosdf --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/<your-dataset> --include-mono-prior True
```

Note that argument `inside-outside` should be set to `True` for an indoor scene.

## Estimate Light Sources
- **PREREQUISITE**: [Estimate Geometry](#Estimate Geometry)
- Configure the lighting optimizer for your scene.  We suggest starting with one of our provided example scenes:
  - `lighting_estimation/scenes/lamp_scene.py`
  - `lighting_estimation/scenes/conference_scene.py`

- Modify `lighting_estimation/optimize.py` to set your desired scene as `current_scene` in `L1`.

- Estimate light sources using the following command:
```commandline
python lighting_estimation/optimize.py
```

