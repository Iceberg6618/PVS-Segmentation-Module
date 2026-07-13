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
5. Generate raw Frangi maps using the brain mask as the Frangi support mask.
6. Generate BG and CSO candidates with region-specific thresholds.
7. Reduce false positives using 2D connected-component filtering.
8. Merge the BG and CSO results into a final PVS segmentation mask.

Preprocessing is performed only once per subject and shared between the BG and
CSO branches.

The BG and CSO candidate masks are saved before ROI clipping. Region ROI masks
are applied at the final false-positive-reduction output stage, which keeps
candidate components intact for anatomical exclusion checks near ROI borders.

## Requirements

- Windows
- Python 3.10
- TensorFlow 2.10 for CPU
- Native SynthSeg

For the Native SynthSeg implementation and installation instructions, refer to
[Iceberg6618/Native_Synthseg](https://github.com/Iceberg6618/Native_Synthseg.git).

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
from PVS_Seg_Module_Local.runner import PVSSegRunner


runner = PVSSegRunner(
    t2_path=r"input\subj01.nii.gz",
    out_dir=r"output\subj01",
    t1_path=None,
    save_all=True,
)

result = runner.run(
    preproc_overwrite=False,
)

final_mask = result["pvs_segmentation_mask"]
```

`bg_sigmas` and `cso_sigmas` can be specified in millimeters. The runner
automatically converts them to voxel units using the input image's in-plane
spacing. For example, a sigma of 1.0 mm becomes 2.0 voxels when the in-plane
spacing is 0.5 mm.

By default, the Frangi sigmas are `[0.215, 0.430, 0.645]` mm. These reproduce
voxel-level sigmas `[1, 2, 3]` for images with 0.215 mm in-plane spacing.

The default candidate thresholds are:

```python
bg_thr=0.002
cso_thr=0.003
```

If `t1_path` is provided, SynthSeg is run on the T1 image and the resulting
segmentation is registered to the T2 native space. If `t1_path` is omitted,
SynthSeg is run directly on the T2 image.

False-positive reduction is performed slice-by-slice. Components are removed
when they fall outside the configured 2D area range, intersect dilated CSF or
dilated ventricle masks, or overlap non-dilated gray matter by at least 50%.
For BG, components with elongation greater than `bg_max_elongation` are also
removed by default. CSO does not use the BG elongation criterion.

The default false-positive-reduction parameters are:

```python
csf_dilation_mm=None  # one in-plane voxel
ventricle_dilation_mm=2.0
cso_min_area_mm2=1.0
cso_max_area_mm2=12.0
bg_min_area_mm2=1.5
bg_max_area_mm2=18.0
bg_max_elongation=5.0
```

## Candidate Generation

Use `PVSCandidateRunner` to generate Frangi candidates without false-positive
reduction:

```python
from PVS_Seg_Module_Local.runner import PVSCandidateRunner


runner = PVSCandidateRunner(
    t2_path=r"input\subj01.nii.gz",
    out_dir=r"output\subj01",
)

result = runner.run()
```

The common `sigmas` argument applies the same Frangi scales to BG and CSO.
Region-specific arguments (`bg_sigmas`, `cso_sigmas`, `bg_thr`, and
`cso_thr`) take precedence over the default values.

## Output Structure

The full pipeline produces the following output structure:

```text
output/subj01/
|-- preproc/
|   |-- brain_mask.nii.gz
|   |-- native_lobe.nii.gz
|   |-- native_roi_mask.nii.gz
|   |-- native_synthseg.nii.gz
|   |-- t1_raw.nii.gz              # only when t1_path is provided
|   |-- t1_to_t2.nii.gz            # only when t1_path is provided
|   |-- t2_raw.nii.gz
|   `-- t2_norm.nii.gz
|-- BG/
|   |-- frangi_params.json
|   |-- roi_mask.nii.gz
|   |-- raw_frangi.nii.gz
|   |-- pvs_candidates.nii.gz
|   `-- pvs_fp_reduced.nii.gz
|-- CSO/
|   |-- frangi_params.json
|   |-- roi_mask.nii.gz
|   |-- raw_frangi.nii.gz
|   |-- pvs_candidates.nii.gz
|   `-- pvs_fp_reduced.nii.gz
`-- pvs_segmentation_mask.nii.gz
```

`pvs_segmentation_mask.nii.gz` is the final binary mask obtained by merging
the false-positive-reduced BG and CSO results.

`raw_frangi.nii.gz` is computed within the brain mask support. `pvs_candidates.nii.gz`
is thresholded from the raw Frangi map without ROI clipping. `pvs_fp_reduced.nii.gz`
is the region-specific output after false-positive reduction and final ROI
masking.

## Notes

- The input must be a 3D NIfTI file (`.nii` or `.nii.gz`).
- Output NIfTI files preserve the input image affine and header.
- Existing preprocessing products are reused when
  `preproc_overwrite=False`.
- Frangi cache reuse is controlled by `frangi_params.json`, which records the
  spacing, sigma in millimeters, voxel sigma, and threshold.
- Parameter values should be validated for the scanner, sequence, image
  resolution, and target cohort before research use.
