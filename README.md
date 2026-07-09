# PVS Segmentation Module

A Python module for detecting enlarged perivascular spaces (PVS) in
T2-weighted brain MRI and reducing false-positive detections.

The module processes the basal ganglia (BG) and centrum semiovale (CSO)
separately, allowing region-specific Frangi scales and thresholds. It combines
native-space SynthSeg inference, image preprocessing, 2D Frangi filtering,
anatomical exclusion masks, and connected-component filtering in a single
pipeline.

## Pipeline

1. Run native-space SynthSeg and generate brain/tissue masks.
2. Generate BG and CSO regions of interest.
3. Normalize T2 intensity.
4. Adjust Frangi sigmas for the input in-plane resolution.
5. Generate PVS candidates separately for BG and CSO.
6. Apply CSF, gray matter, and ventricular exclusion masks.
7. Reduce false positives using 2D area and shape criteria.
8. Merge the BG and CSO results into a final PVS segmentation mask.

Preprocessing is performed only once per subject and shared between the BG and
CSO branches.

## Requirements

- Windows
- Python 3.10
- TensorFlow 2.10 for CPU
- [Native SynthSeg](https://github.com/Iceberg6618/Native_Synthseg)

Native SynthSeg includes the original
[BBillot/SynthSeg](https://github.com/BBillot/SynthSeg) source code and model.

## Installation

Create and activate a Conda environment:

```bash
conda create -n PVS_Seg_Module python=3.10 -y
conda activate PVS_Seg_Module
```

Install Native SynthSeg first:

```bash
pip install git+https://github.com/Iceberg6618/Native_Synthseg.git
```

Clone this repository and install the remaining dependencies:

```bash
git clone <THIS_REPOSITORY_URL>
cd PVS_Seg_Module_Local
pip install -r requirements.txt
```

This repository does not currently contain a `setup.py` or `pyproject.toml`.
Run the examples below from the parent directory of
`PVS_Seg_Module_Local`, or add that parent directory to `PYTHONPATH`.

## Full Segmentation

```python
import numpy as np

from PVS_Seg_Module_Local.runner import PVSSegRunner


runner = PVSSegRunner(
    t2_path=r"input\subj01.nii.gz",
    out_dir=r"output\subj01",
    save_all=True,
    bg_thr=0.01,
    cso_thr=0.02,
)

result = runner.run(
    bg_sigmas=np.array([1.0, 2.0]),
    cso_sigmas=np.array([2.0, 3.0]),
    preproc_overwrite=False,
    fp_mode="2d",
    csf_gm_dilation_mm=2.0,
    ventricle_dilation_mm=1.5,
    min_area_mm2=0.5,
    max_area_mm2=15.0,
)

final_mask = result["pvs_segmentation_mask"]
```

`bg_sigmas` and `cso_sigmas` are specified in millimeters. The runner
automatically converts them to voxel units using the input image's in-plane
spacing. For example, a sigma of 1.0 mm becomes 2.0 voxels when the in-plane
spacing is 0.5 mm.

## Candidate Generation

Use `PVSCandidateRunner` to generate Frangi candidates without false-positive
reduction:

```python
from PVS_Seg_Module_Local.runner import PVSCandidateRunner


runner = PVSCandidateRunner(
    t2_path=r"input\subj01.nii.gz",
    out_dir=r"output\subj01",
    bg_thr=0.01,
    cso_thr=0.02,
)

result = runner.run(
    bg_sigmas=[1.0, 2.0],
    cso_sigmas=[2.0, 3.0],
)
```

The common `sigmas` and `thr` arguments apply the same values to BG and CSO.
Region-specific arguments (`bg_sigmas`, `cso_sigmas`, `bg_thr`, and
`cso_thr`) take precedence over the common values.

## Output Structure

The full pipeline produces the following output structure:

```text
output/subj01/
|-- preproc/
|   |-- brain_mask.nii.gz
|   |-- native_lobe.nii.gz
|   |-- native_roi_mask.nii.gz
|   |-- native_synthseg.nii.gz
|   |-- t2_raw.nii.gz
|   `-- t2_norm.nii.gz
|-- BG/
|   |-- frangi_params.json
|   |-- raw_frangi.nii.gz
|   |-- pvs_candidates.nii.gz
|   `-- pvs_fp_reduced.nii.gz
|-- CSO/
|   |-- frangi_params.json
|   |-- raw_frangi.nii.gz
|   |-- pvs_candidates.nii.gz
|   `-- pvs_fp_reduced.nii.gz
`-- pvs_segmentation_mask.nii.gz
```

`pvs_segmentation_mask.nii.gz` is the final binary mask obtained by merging
the false-positive-reduced BG and CSO results.

## Main Parameters

| Parameter | Description | Default |
|---|---|---:|
| `bg_sigmas` | BG Frangi scales in millimeters | `[1, 5]` |
| `cso_sigmas` | CSO Frangi scales in millimeters | `[1, 5]` |
| `bg_thr` | BG Frangi response threshold | `0.01` |
| `cso_thr` | CSO Frangi response threshold | `0.01` |
| `csf_gm_dilation_mm` | CSF/GM exclusion-mask dilation radius | `2.0 mm` |
| `ventricle_dilation_mm` | Ventricular exclusion-mask dilation radius | `1.5 mm` |
| `min_area_mm2` | Minimum 2D component area | `0.5 mm2` |
| `max_area_mm2` | Maximum 2D component area | `15.0 mm2` |
| `preproc_overwrite` | Recompute existing preprocessing outputs | `False` |

Area filtering and exclusion-mask dilation are performed independently on each
slice. No through-plane dilation is applied.

## Notes

- The input must be a 3D NIfTI file (`.nii` or `.nii.gz`).
- Output NIfTI files preserve the input image affine and header.
- Existing preprocessing products are reused when
  `preproc_overwrite=False`.
- Frangi cache reuse is controlled by `frangi_params.json`, which records the
  spacing, sigma, and threshold.
- Parameter values should be validated for the scanner, sequence, image
  resolution, and target cohort before research use.

## License

Before public distribution, add a license file that is compatible with this
repository, Native SynthSeg, the bundled SynthSeg source code and model, and
the included atlas data.
