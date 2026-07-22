# PVS Segmentation Module

A Python module for detecting enlarged perivascular spaces (PVS) in
T2-weighted brain MRI and reducing false-positive detections.

The module processes the basal ganglia (BG) and centrum semiovale (CSO)
separately. BG and CSO can use different Frangi scales, candidate thresholds,
size filters, and shape filters. The pipeline combines native-space SynthSeg,
MNI lobar registration, T2 intensity normalization, 2D Frangi filtering,
anatomical exclusion masks, and slice-wise connected-component filtering.

## Requirements

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
Run the examples below from the parent directory of `PVS_Seg_Module_Local`, or
add that parent directory to `PYTHONPATH`.

## Full Pipeline

The full segmentation pipeline is run by `PVSSegRunner`. It performs candidate
generation and false-positive reduction for both BG and CSO, then merges the two
region-specific masks.

```python
from PVS_Seg_Module_Local.runner import PVSSegRunner


runner = PVSSegRunner(
    t2_path=r"input\subj01_t2.nii.gz",
    out_dir=r"output\subj01",
    t1_path=None,
    save_all=True,
)

result = runner.run(preproc_overwrite=False)
final_mask = result["pvs_segmentation_mask"]
```

The pipeline runs the following steps.

1. **Load input images**
   - `t2_path` is required and is used as the final native target space.
   - `t1_path` is optional. If present, SynthSeg is run on T1 and then the T1
     SynthSeg output is registered to T2 native space.

2. **Native SynthSeg**
   - If `t1_path=None`, SynthSeg is run directly on T2.
   - If `t1_path` is provided, SynthSeg is run on T1, T1 is affinely registered
     to T2, and the SynthSeg label map is transformed to T2 using label
     interpolation.
   - Output: `preproc/native_synthseg.nii.gz`.

3. **MNI lobar registration**
   - The MNI T2 template and lobar atlas are registered to the subject T2 native
     space using ANTs SyN registration.
   - Label interpolation is used for the lobar atlas.
   - Output: `preproc/native_lobe.nii.gz`.

4. **Brain, tissue, and ROI mask generation**
   - `brain_mask` is generated from nonzero SynthSeg labels.
   - BG ROI is generated from SynthSeg basal ganglia labels.
   - CSO ROI is generated from lobar atlas labels overlapped with SynthSeg white
     matter.
   - CSF, ventricle, and gray-matter masks are generated from SynthSeg and used
     later during false-positive reduction.
   - Outputs: `preproc/brain_mask.nii.gz`, `preproc/native_roi_mask.nii.gz`.

5. **T2 intensity normalization**
   - T2 intensity is normalized using the normalization mask made from SynthSeg
     white matter and basal ganglia labels.
   - The default normalization uses percentile scaling with `qmin=0` and
     `qmax=99`, without upper clipping.
   - Outputs: `preproc/t2_raw.nii.gz`, `preproc/t2_norm.nii.gz`.

6. **In-plane resolution handling**
   - The module reads the image affine and finds the Superior-Inferior axis.
   - The two axes that are not Superior-Inferior are treated as the in-plane
     axes.
   - Frangi sigmas are provided in millimeters and converted to voxel sigmas
     using the mean in-plane spacing.
   - 2D Frangi filtering is performed slice-by-slice along the Superior-Inferior
     axis, not blindly along array axis 2.

7. **Raw Frangi filtering**
   - BG and CSO raw Frangi maps are generated separately.
   - Frangi support is the brain mask, not the final region ROI. This keeps
     candidate components intact near ROI boundaries.
   - Outputs: `BG/raw_frangi.nii.gz`, `CSO/raw_frangi.nii.gz`.

8. **Candidate generation**
   - BG and CSO candidates are created by thresholding each region's raw Frangi
     map.
   - Candidate masks are intentionally saved before ROI clipping.
   - Outputs: `BG/pvs_candidates.nii.gz`, `CSO/pvs_candidates.nii.gz`.

9. **False-positive reduction**
   - Connected components are evaluated slice-by-slice on anatomical axial
     slices, using the same Superior-Inferior axis logic as Frangi filtering.
   - Components are removed if their 2D area is outside the configured range.
   - Components are removed if they intersect dilated CSF or dilated ventricle
     masks.
   - Components are removed if at least `gm_overlap_thres` of the component
     overlaps non-dilated gray matter.
   - BG additionally removes components whose elongation is greater than
     `bg_max_elongation`.
   - ROI masking is applied at the final FP-reduced output stage.
   - Outputs: `BG/pvs_fp_reduced.nii.gz`, `CSO/pvs_fp_reduced.nii.gz`.

10. **Final aggregation**
    - The final mask is the union of the BG and CSO FP-reduced masks.
    - Output: `pvs_segmentation_mask.nii.gz`.

## Default Values

The default Frangi sigmas are:

```python
DEFAULT_SIGMAS_MM = [0.215, 0.430, 0.645]
```

These reproduce voxel-level sigmas `[1, 2, 3]` for T2 images with 0.215 mm
in-plane spacing. For an image with 0.5 mm in-plane spacing, the same default
physical sigmas become approximately `[0.43, 0.86, 1.29]` voxels.

The default candidate thresholds are:

```python
bg_thr = 0.002
cso_thr = 0.003
```

The default false-positive-reduction parameters are:

```python
csf_dilation_voxels = 1        # one in-plane voxel
ventricle_dilation_mm = 2.0     # converted to in-plane voxel iterations
cso_min_area_mm2 = 1.0
cso_max_area_mm2 = 12.0
bg_min_area_mm2 = 1.5
bg_max_area_mm2 = 18.0
bg_max_elongation = 5.0
```

## Main Classes

### `PVSCandidateRunner`

Use this class when you want preprocessing and candidate generation only.
False-positive reduction and final aggregation are not performed.

```python
from PVS_Seg_Module_Local.runner import PVSCandidateRunner


runner = PVSCandidateRunner(
    t2_path=r"input\subj01_t2.nii.gz",
    out_dir=r"output\subj01",
    t1_path=None,
    save_all=True,
    bg_thr=None,
    cso_thr=None,
)

result = runner.run(
    sigmas=None,
    bg_sigmas=None,
    cso_sigmas=None,
    bg_thr=None,
    cso_thr=None,
    preproc_overwrite=False,
)
```

### `PVSSegRunner`

Use this class for the full segmentation pipeline.

```python
from PVS_Seg_Module_Local.runner import PVSSegRunner


runner = PVSSegRunner(
    t2_path=r"input\subj01_t2.nii.gz",
    out_dir=r"output\subj01",
    t1_path=r"input\subj01_t1.nii.gz",
    save_all=True,
)

result = runner.run(
    bg_sigmas=None,
    cso_sigmas=None,
    bg_thr=None,
    cso_thr=None,
    preproc_overwrite=False,
    csf_dilation_voxels=1,
    ventricle_dilation_mm=2.0,
    cso_min_area_mm2=1.0,
    cso_max_area_mm2=12.0,
    bg_min_area_mm2=1.5,
    bg_max_area_mm2=18.0,
    bg_max_elongation=5.0,
)
```

## Parameter Reference

### Constructor Parameters

| Parameter | Used by | Default | Description |
| --- | --- | --- | --- |
| `t2_path` | `PVSCandidateRunner`, `PVSSegRunner` | required | Path to the input T2-weighted NIfTI. This image defines the native output space, affine, header, voxel spacing, and anatomical slicing axis. |
| `out_dir` | `PVSCandidateRunner`, `PVSSegRunner` | required | Output directory for preprocessing products, region outputs, and the final mask. |
| `t1_path` | `PVSCandidateRunner`, `PVSSegRunner` | `None` | Optional T1-weighted NIfTI. If provided, SynthSeg runs on T1 and the segmentation is registered to T2 native space. If omitted, SynthSeg runs on T2. |
| `save_all` | `PVSCandidateRunner`, `PVSSegRunner` | `True` | Whether to save intermediate and final NIfTI outputs. |
| `bg_thr` | `PVSCandidateRunner`, `PVSSegRunner` | `0.002` | Default BG candidate threshold. Can be overridden during `run()`. |
| `cso_thr` | `PVSCandidateRunner`, `PVSSegRunner` | `0.003` | Default CSO candidate threshold. Can be overridden during `run()`. |

### Candidate Generation Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `sigmas` | `None` | Common Frangi sigmas in millimeters for both BG and CSO. Used only when region-specific sigmas are not provided. If all sigma arguments are `None`, the module uses `DEFAULT_SIGMAS_MM`. |
| `bg_sigmas` | `None` | BG-specific Frangi sigmas in millimeters. Takes precedence over `sigmas` for BG. |
| `cso_sigmas` | `None` | CSO-specific Frangi sigmas in millimeters. Takes precedence over `sigmas` for CSO. |
| `bg_thr` | `None` | Runtime BG threshold override. If `None`, the constructor-level `bg_thr` is used. |
| `cso_thr` | `None` | Runtime CSO threshold override. If `None`, the constructor-level `cso_thr` is used. |
| `preproc_overwrite` | `False` | If `False`, existing preprocessing outputs such as `native_synthseg.nii.gz` and `native_lobe.nii.gz` are reused when available. If `True`, preprocessing is recomputed. |

### False-Positive Reduction Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `csf_dilation_voxels` | `1` | CSF mask dilation before exclusion, directly specified as in-plane voxel iterations. `0` disables CSF dilation. |
| `ventricle_dilation_mm` | `2.0` | Ventricle mask dilation before exclusion. This value is converted from millimeters to in-plane voxel iterations using the mean in-plane spacing. `0` disables ventricle dilation. |
| `cso_min_area_mm2` | `1.0` | Minimum allowed 2D connected-component area for CSO candidates. Components smaller than this are removed. |
| `cso_max_area_mm2` | `12.0` | Maximum allowed 2D connected-component area for CSO candidates. Components larger than this are removed. |
| `bg_min_area_mm2` | `1.5` | Minimum allowed 2D connected-component area for BG candidates. BG defaults are slightly larger because BG PVS are often thicker than CSO PVS. |
| `bg_max_area_mm2` | `18.0` | Maximum allowed 2D connected-component area for BG candidates. |
| `bg_max_elongation` | `5.0` | Maximum allowed BG component elongation. Components with `major_axis_length / minor_axis_length` greater than this value are removed. Set to `None` to disable this rule. |

### Internal `FPReduction` Parameters

`FPReduction` is usually called through `PVSSegRunner`, but it can also be used
directly when candidate masks already exist.

| Parameter | Default | Description |
| --- | --- | --- |
| `t2_nib` | required | Reference T2 NIfTI. Used for affine, voxel spacing, and anatomical slicing axis. |
| `seg_array` | required | Binary candidate mask. Nonzero voxels are treated as candidate voxels. |
| `roi_mask` | required | Region ROI mask. It is applied only to the final output, after connected-component filtering. |
| `csf_mask` | `None` | CSF exclusion mask before dilation. |
| `gm_mask` | `None` | Gray-matter exclusion mask. This mask is not dilated. |
| `ventricle_mask` | `None` | Ventricle exclusion mask before dilation. |
| `gm_overlap_thres` | `0.5` | Component-level GM overlap threshold. A component is removed when `GM-overlap voxels / component voxels >= gm_overlap_thres`. |
| `max_elongation` | `None` | Optional elongation filter. Used for BG by default through `bg_max_elongation`; disabled for CSO. |

## How Parameters Affect the Result

- Increasing `bg_thr` or `cso_thr` makes candidate generation stricter and
  usually decreases sensitivity and false positives.
- Decreasing `bg_thr` or `cso_thr` keeps more faint Frangi responses and usually
  increases sensitivity and false positives.
- Larger `bg_sigmas` or `cso_sigmas` emphasize thicker, tube-like structures.
- Smaller sigmas emphasize thinner structures but can increase noise-like
  candidates.
- Larger `csf_dilation_voxels` removes more candidates near sulci and CSF spaces.
- Larger `ventricle_dilation_mm` removes more candidates near ventricles and
  periventricular regions.
- Increasing `min_area_mm2` removes more small components.
- Decreasing `max_area_mm2` removes more large components that may represent
  vessels, WMH, sulcal partial volume, or other non-PVS structures.
- Lowering `gm_overlap_thres` makes GM exclusion stricter.
- Lowering `bg_max_elongation` removes more long, vessel-like BG components.

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

Output meanings:

| File | Description |
| --- | --- |
| `preproc/brain_mask.nii.gz` | Brain mask from SynthSeg nonzero labels. |
| `preproc/native_synthseg.nii.gz` | SynthSeg segmentation in T2 native space. |
| `preproc/native_lobe.nii.gz` | MNI lobar atlas transformed to T2 native space. |
| `preproc/native_roi_mask.nii.gz` | Combined region ROI mask. BG is label `1`; lobar WM regions are labels greater than `1`. |
| `preproc/t2_norm.nii.gz` | Normalized T2 image used for Frangi filtering. |
| `BG/raw_frangi.nii.gz`, `CSO/raw_frangi.nii.gz` | Raw Frangi vesselness maps computed inside brain-mask support. |
| `BG/roi_mask.nii.gz`, `CSO/roi_mask.nii.gz` | Region-specific ROI masks used at the final FP-reduction stage. |
| `BG/pvs_candidates.nii.gz`, `CSO/pvs_candidates.nii.gz` | Thresholded candidate masks before ROI clipping. |
| `BG/pvs_fp_reduced.nii.gz`, `CSO/pvs_fp_reduced.nii.gz` | Region-specific candidate masks after false-positive reduction and final ROI masking. |
| `pvs_segmentation_mask.nii.gz` | Final union of BG and CSO FP-reduced masks. |
| `frangi_params.json` | Frangi metadata used for cache validation: spacing, threshold, sigmas in mm, and sigmas in voxels. |

## Cache and Overwrite Behavior

- Existing preprocessing products are reused when `preproc_overwrite=False`.
- `native_synthseg.nii.gz` and `native_lobe.nii.gz` are loaded from disk when
  present and overwrite is disabled.
- Raw Frangi maps are reused only when `frangi_params.json` matches the current
  spacing, threshold, sigma in millimeters, and voxel sigma.
- If Frangi parameters do not match, raw Frangi is recomputed.
- Progress messages are printed during execution, including cache loading,
  Frangi computation, candidate generation, FP reduction, and final aggregation.

## Notes

- The input must be a 3D NIfTI file (`.nii` or `.nii.gz`).
- Output NIfTI files preserve the input image affine and header.
- Frangi filtering and FP reduction are both performed slice-by-slice along the
  anatomical Superior-Inferior axis derived from the NIfTI affine.
- Area thresholds are defined in `mm2`, but component filtering is performed in
  2D on each anatomical axial slice.
- Candidate masks are not ROI-clipped before false-positive reduction. ROI masks
  are applied at the final FP-reduced output stage.
- Parameter values should be validated for the scanner, sequence, image
  resolution, and target cohort before research use.
