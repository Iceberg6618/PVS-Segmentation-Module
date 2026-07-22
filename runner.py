"""Runners for region-specific PVS segmentation."""

from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np

from .pvs_base_module import FPReduction, PVSPreprocessing, frangi2d
from .pvs_base_module.pvs_preprocessing import TARGET_ROI_LABELS


MIN_SIGMA_VOX = 0.25
REFERENCE_INPLANE_MM = 0.215
DEFAULT_SIGMAS_MM = np.array([1.0, 2.0, 3.0], dtype=np.float64) * REFERENCE_INPLANE_MM
DEFAULT_BG_THR = 0.002
DEFAULT_CSO_THR = 0.003


def get_inplane_spacing(nii: nib.Nifti1Image) -> float:
    """
    Return mean in-plane voxel spacing in mm.

    For axial acquisitions, the in-plane axes are the two axes that are not the
    Superior-Inferior direction. If orientation cannot be determined, fall back
    to the first header zoom.
    """
    zooms = np.array(nii.header.get_zooms()[:3], dtype=np.float64)
    try:
        codes = np.array(nib.orientations.aff2axcodes(nii.affine))
        si_axis = int(np.where((codes == "S") | (codes == "I"))[0][0])
        inplane = [z for ax, z in enumerate(zooms) if ax != si_axis]
        return float(np.mean(inplane))
    except Exception:
        return float(zooms[0])


def sigmas_mm_to_voxels(sigmas_mm, spacing: float) -> np.ndarray:
    """Convert physical Frangi scales in mm to per-image voxel sigmas."""
    sigmas_vox = np.asarray(sigmas_mm, dtype=np.float64) / float(spacing)
    return np.maximum(sigmas_vox, MIN_SIGMA_VOX)


def get_superior_inferior_axis(nii: nib.Nifti1Image) -> int:
    """Return the voxel axis corresponding to anatomical Superior-Inferior."""
    try:
        codes = np.array(nib.orientations.aff2axcodes(nii.affine))
        return int(np.where((codes == "S") | (codes == "I"))[0][0])
    except Exception:
        return 2


class PVSCandidateRunner:
    """
    Run preprocessing once, then generate BG and CSO PVS candidates separately.

    Output layout:
        out_dir/
            preproc/
            BG/
                raw_frangi.nii.gz
                pvs_candidates.nii.gz
            CSO/
                raw_frangi.nii.gz
                pvs_candidates.nii.gz
    """

    def __init__(
        self,
        t2_path: str,
        out_dir: str,
        t1_path: str | None = None,
        save_all: bool = True,
        bg_thr: float | None = None,
        cso_thr: float | None = None,
    ):
        if out_dir is None:
            raise ValueError("out_dir must be a valid directory path.")

        self.t2_path = t2_path
        self.t2_nib = nib.load(t2_path)
        self.t1_path = t1_path
        self.t1_nib = nib.load(t1_path) if t1_path is not None else None
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.save = save_all
        self.bg_thr = DEFAULT_BG_THR if bg_thr is None else bg_thr
        self.cso_thr = DEFAULT_CSO_THR if cso_thr is None else cso_thr
        self.slice_axis = get_superior_inferior_axis(self.t2_nib)
        self.inplane_spacing = get_inplane_spacing(self.t2_nib)
        print(
            f"[PVS] Initialized runner | T2={os.path.basename(t2_path)} | "
            f"SI axis={self.slice_axis} | in-plane spacing={self.inplane_spacing:.4f} mm | out={self.out_dir}"
        )

    def run(
        self,
        sigmas=None,
        bg_sigmas=None,
        cso_sigmas=None,
        bg_thr: float | None = None,
        cso_thr: float | None = None,
        preproc_overwrite: bool = False,
    ) -> dict[str, np.ndarray]:
        print("[PVS] Candidate generation started.")
        preproc = self._run_preprocessing(preproc_overwrite=preproc_overwrite)
        bg_threshold, cso_threshold = self._resolve_thresholds(bg_thr, cso_thr)
        bg_mask = self._bg_mask(preproc)
        cso_mask = self._cso_mask(preproc)
        frangi_support_mask = self._frangi_support_mask(preproc)

        bg_result = self._run_region_candidate(
            preproc=preproc,
            region_name="BG",
            region_mask=bg_mask,
            frangi_support_mask=frangi_support_mask,
            sigmas=self._resolve_sigmas(bg_sigmas, sigmas),
            threshold=bg_threshold,
        )
        cso_result = self._run_region_candidate(
            preproc=preproc,
            region_name="CSO",
            region_mask=cso_mask,
            frangi_support_mask=frangi_support_mask,
            sigmas=self._resolve_sigmas(cso_sigmas, sigmas),
            threshold=cso_threshold,
        )

        print("[PVS] Candidate generation done.")
        return {
            "BG": bg_result,
            "CSO": cso_result,
            "brain_mask": preproc.brain_mask,
            "roi_mask": preproc.target_roi_mask,
            "exclusion_mask": preproc.exclusion_mask,
        }

    def _run_preprocessing(self, preproc_overwrite: bool) -> PVSPreprocessing:
        preproc_out = os.path.join(self.out_dir, "preproc")
        print(f"[PVS] Preprocessing started | overwrite={preproc_overwrite} | out={preproc_out}")
        preproc = PVSPreprocessing(
            t2_nib=self.t2_nib,
            t1_nib=self.t1_nib,
            out_dir=preproc_out,
            overwrite=preproc_overwrite,
        )
        preproc.run_all_preprocesses()
        print("[PVS] Preprocessing done.")
        return preproc

    @staticmethod
    def _resolve_sigmas(region_sigmas, fallback_sigmas):
        if region_sigmas is not None:
            return region_sigmas
        if fallback_sigmas is not None:
            return fallback_sigmas
        return DEFAULT_SIGMAS_MM

    def _resolve_thresholds(self, bg_thr: float | None, cso_thr: float | None) -> tuple[float, float]:
        bg_threshold = self.bg_thr if bg_thr is None else bg_thr
        cso_threshold = self.cso_thr if cso_thr is None else cso_thr
        return float(bg_threshold), float(cso_threshold)

    def _adapt_sigmas(self, sigmas):
        sigmas_mm = np.asarray(sigmas, dtype=np.float64)
        sigmas_vox = sigmas_mm_to_voxels(sigmas_mm, self.inplane_spacing)
        return sigmas_mm, sigmas_vox

    @staticmethod
    def _bg_mask(preproc: PVSPreprocessing) -> np.ndarray:
        return (preproc.target_roi_mask == TARGET_ROI_LABELS["BG"]).astype(np.uint8)

    @staticmethod
    def _cso_mask(preproc: PVSPreprocessing) -> np.ndarray:
        return (preproc.target_roi_mask > TARGET_ROI_LABELS["BG"]).astype(np.uint8)

    @staticmethod
    def _frangi_support_mask(preproc: PVSPreprocessing) -> np.ndarray:
        return (preproc.brain_mask > 0).astype(np.uint8)

    def _normalised_array(self, preproc: PVSPreprocessing) -> np.ndarray:
        if isinstance(preproc.t2_normalized, nib.Nifti1Image):
            return preproc.t2_normalized.get_fdata()
        return np.asarray(preproc.t2_normalized)

    def _run_region_candidate(
        self,
        preproc: PVSPreprocessing,
        region_name: str,
        region_mask: np.ndarray,
        frangi_support_mask: np.ndarray,
        sigmas,
        threshold: float,
    ) -> dict[str, np.ndarray]:
        region_out = os.path.join(self.out_dir, region_name)
        os.makedirs(region_out, exist_ok=True)

        sigmas_mm, sigmas_vox = self._adapt_sigmas(sigmas)
        local_frangi_path = os.path.join(region_out, "raw_frangi.nii.gz")
        metadata_path = os.path.join(region_out, "frangi_params.json")
        metadata = self._frangi_metadata(threshold, sigmas_mm, sigmas_vox)

        if os.path.exists(local_frangi_path) and self._frangi_metadata_matches(metadata_path, metadata):
            print(f"[PVS] {region_name} raw_frangi exists. Loading file: {local_frangi_path}")
            frangi_filtered = nib.load(local_frangi_path).get_fdata()
        else:
            print(
                f"[PVS] {region_name} Frangi filtering started | "
                f"sigmas_mm={sigmas_mm.tolist()} | sigmas_vox={sigmas_vox.tolist()} | threshold={threshold}"
            )
            frangi_filtered = frangi2d(
                image=self._normalised_array(preproc),
                black_ridges=False,
                sigmas=sigmas_vox,
                mask=frangi_support_mask,
                slice_axis=self.slice_axis,
            )
            if self.save:
                self._save_nifti(frangi_filtered, local_frangi_path)
                self._save_frangi_metadata(metadata_path, metadata)
                print(f"[PVS] {region_name} raw_frangi saved: {local_frangi_path}")

        pvs_candidates = (frangi_filtered > threshold).astype(np.uint8)
        if self.save:
            self._save_nifti(region_mask, os.path.join(region_out, "roi_mask.nii.gz"))
            self._save_nifti(pvs_candidates, os.path.join(region_out, "pvs_candidates.nii.gz"))
        print(f"[PVS] {region_name} candidate generation done | voxels={int(np.count_nonzero(pvs_candidates))}")

        return {
            "frangi": frangi_filtered,
            "pvs_candidates": pvs_candidates,
            "region_mask": region_mask,
            "frangi_support_mask": frangi_support_mask,
            "sigmas_mm": sigmas_mm,
            "sigmas_vox": sigmas_vox,
            "inplane_spacing": self.inplane_spacing,
        }

    def _save_nifti(self, array: np.ndarray, path: str) -> None:
        nib.save(nib.Nifti1Image(array, self.t2_nib.affine, self.t2_nib.header), path)

    def _frangi_metadata(self, threshold: float, sigmas_mm: np.ndarray, sigmas_vox: np.ndarray) -> dict:
        return {
            "spacing_mm": float(self.inplane_spacing),
            "threshold": float(threshold),
            "sigmas_mm": [float(v) for v in sigmas_mm],
            "sigmas_vox": [float(v) for v in sigmas_vox],
        }

    @staticmethod
    def _frangi_metadata_matches(metadata_path: str, current: dict) -> bool:
        if not os.path.exists(metadata_path):
            return False
        try:
            with open(metadata_path, "r") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        if not np.isclose(existing.get("spacing_mm", np.nan), current["spacing_mm"]):
            return False
        if not np.isclose(existing.get("threshold", np.nan), current["threshold"]):
            return False
        try:
            if not np.allclose(existing.get("sigmas_mm", []), current["sigmas_mm"]):
                return False
            if not np.allclose(existing.get("sigmas_vox", []), current["sigmas_vox"]):
                return False
        except ValueError:
            return False
        return True

    @staticmethod
    def _save_frangi_metadata(metadata_path: str, metadata: dict) -> None:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


class PVSSegRunner(PVSCandidateRunner):
    """Run region-specific candidate generation and false-positive reduction."""

    def run(
        self,
        sigmas=None,
        bg_sigmas=None,
        cso_sigmas=None,
        bg_thr: float | None = None,
        cso_thr: float | None = None,
        preproc_overwrite: bool = False,
        csf_dilation_voxels: int = 1,
        ventricle_dilation_mm: float = 2.0,
        cso_min_area_mm2: float = 1.0,
        cso_max_area_mm2: float = 12.0,
        bg_min_area_mm2: float = 1.5,
        bg_max_area_mm2: float = 18.0,
        bg_max_elongation: float | None = 5.0,
    ) -> dict[str, np.ndarray]:
        print("[PVS] Full segmentation started.")
        preproc = self._run_preprocessing(preproc_overwrite=preproc_overwrite)
        bg_threshold, cso_threshold = self._resolve_thresholds(bg_thr, cso_thr)
        bg_mask = self._bg_mask(preproc)
        cso_mask = self._cso_mask(preproc)
        frangi_support_mask = self._frangi_support_mask(preproc)

        bg_result = self._run_region_candidate(
            preproc=preproc,
            region_name="BG",
            region_mask=bg_mask,
            frangi_support_mask=frangi_support_mask,
            sigmas=self._resolve_sigmas(bg_sigmas, sigmas),
            threshold=bg_threshold,
        )
        cso_result = self._run_region_candidate(
            preproc=preproc,
            region_name="CSO",
            region_mask=cso_mask,
            frangi_support_mask=frangi_support_mask,
            sigmas=self._resolve_sigmas(cso_sigmas, sigmas),
            threshold=cso_threshold,
        )

        bg_fp = self._run_fp_reduction(
            preproc=preproc,
            region_name="BG",
            candidates=bg_result["pvs_candidates"],
            roi_mask=bg_mask,
            csf_dilation_voxels=csf_dilation_voxels,
            ventricle_dilation_mm=ventricle_dilation_mm,
            min_area_mm2=bg_min_area_mm2,
            max_area_mm2=bg_max_area_mm2,
            max_elongation=bg_max_elongation,
        )
        cso_fp = self._run_fp_reduction(
            preproc=preproc,
            region_name="CSO",
            candidates=cso_result["pvs_candidates"],
            roi_mask=cso_mask,
            csf_dilation_voxels=csf_dilation_voxels,
            ventricle_dilation_mm=ventricle_dilation_mm,
            min_area_mm2=cso_min_area_mm2,
            max_area_mm2=cso_max_area_mm2,
            max_elongation=None,
        )

        segmentation_mask = ((bg_fp > 0) | (cso_fp > 0)).astype(np.uint8)
        if self.save:
            self._save_nifti(segmentation_mask, os.path.join(self.out_dir, "pvs_segmentation_mask.nii.gz"))
        print(f"[PVS] Full segmentation done | final voxels={int(np.count_nonzero(segmentation_mask))}")

        bg_result["pvs_fp_reduced"] = bg_fp
        cso_result["pvs_fp_reduced"] = cso_fp

        return {
            "BG": bg_result,
            "CSO": cso_result,
            "pvs_segmentation_mask": segmentation_mask,
            "brain_mask": preproc.brain_mask,
            "roi_mask": preproc.target_roi_mask,
            "exclusion_mask": preproc.exclusion_mask,
        }

    def _run_fp_reduction(
        self,
        preproc: PVSPreprocessing,
        region_name: str,
        candidates: np.ndarray,
        roi_mask: np.ndarray,
        csf_dilation_voxels: int,
        ventricle_dilation_mm: float,
        min_area_mm2: float,
        max_area_mm2: float,
        max_elongation: float | None,
    ) -> np.ndarray:
        print(
            f"[PVS] {region_name} FP reduction started | "
            f"min_area={min_area_mm2}, max_area={max_area_mm2}, max_elongation={max_elongation}"
        )
        fp_reduction = FPReduction(
            t2_nib=self.t2_nib,
            seg_array=candidates,
            roi_mask=roi_mask,
            csf_mask=preproc.csf_mask,
            gm_mask=preproc.gm_mask,
            ventricle_mask=preproc.ventricle_mask,
            csf_dilation_voxels=csf_dilation_voxels,
            ventricle_dilation_mm=ventricle_dilation_mm,
            min_area_mm2=min_area_mm2,
            max_area_mm2=max_area_mm2,
            gm_overlap_thres=0.5,
            max_elongation=max_elongation,
        )
        fp_reduced = fp_reduction.run()
        if self.save:
            out_path = os.path.join(self.out_dir, region_name, "pvs_fp_reduced.nii.gz")
            self._save_nifti(fp_reduced, out_path)
            print(f"[PVS] {region_name} FP-reduced mask saved: {out_path}")
        print(f"[PVS] {region_name} FP reduction done | voxels={int(np.count_nonzero(fp_reduced))}")
        return fp_reduced

if __name__ == "__main__":
    pass
