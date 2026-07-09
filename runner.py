"""Runners for region-specific PVS segmentation."""

from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np

from .pvs_base_module import FPReduction, PVSPreprocessing, frangi2d
from .pvs_base_module.pvs_preprocessing import TARGET_ROI_LABELS


MIN_SIGMA_VOX = 0.25


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
        save_all: bool = True,
        thr: float = 0.01,
        bg_thr: float | None = None,
        cso_thr: float | None = None,
    ):
        if out_dir is None:
            raise ValueError("out_dir must be a valid directory path.")

        self.t2_path = t2_path
        self.t2_nib = nib.load(t2_path)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.save = save_all
        self.thr = thr
        self.bg_thr = thr if bg_thr is None else bg_thr
        self.cso_thr = thr if cso_thr is None else cso_thr
        self.inplane_spacing = get_inplane_spacing(self.t2_nib)

    def run(
        self,
        sigmas=None,
        bg_sigmas=None,
        cso_sigmas=None,
        bg_thr: float | None = None,
        cso_thr: float | None = None,
        preproc_overwrite: bool = False,
        frangi_filtered_path: str | None = None,
    ) -> dict[str, np.ndarray]:
        
        preproc = self._run_preprocessing(preproc_overwrite=preproc_overwrite)
        bg_threshold, cso_threshold = self._resolve_thresholds(bg_thr, cso_thr)
        bg_result = self._run_region_candidate(
            preproc=preproc,
            region_name="BG",
            region_mask=self._bg_mask(preproc),
            sigmas=self._resolve_sigmas(bg_sigmas, sigmas),
            threshold=bg_threshold,
            frangi_filtered_path=frangi_filtered_path,
        )
        cso_result = self._run_region_candidate(
            preproc=preproc,
            region_name="CSO",
            region_mask=self._cso_mask(preproc),
            sigmas=self._resolve_sigmas(cso_sigmas, sigmas),
            threshold=cso_threshold,
            frangi_filtered_path=None,
        )

        return {
            "BG": bg_result,
            "CSO": cso_result,
            "brain_mask": preproc.brain_mask,
            "roi_mask": preproc.target_roi_mask,
            "exclusion_mask": preproc.exclusion_mask,
        }

    def _run_preprocessing(self, preproc_overwrite: bool) -> PVSPreprocessing:
        preproc_out = os.path.join(self.out_dir, "preproc")
        preproc = PVSPreprocessing(
            t2_nib=self.t2_nib,
            out_dir=preproc_out,
            overwrite=preproc_overwrite,
        )
        preproc.run_all_preprocesses()
        return preproc

    @staticmethod
    def _resolve_sigmas(region_sigmas, fallback_sigmas):
        if region_sigmas is not None:
            return region_sigmas
        if fallback_sigmas is not None:
            return fallback_sigmas
        return np.linspace(1, 5, 2)

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

    def _normalised_array(self, preproc: PVSPreprocessing) -> np.ndarray:
        if isinstance(preproc.t2_normalized, nib.Nifti1Image):
            return preproc.t2_normalized.get_fdata()
        return np.asarray(preproc.t2_normalized)

    def _run_region_candidate(
        self,
        preproc: PVSPreprocessing,
        region_name: str,
        region_mask: np.ndarray,
        sigmas,
        threshold: float,
        frangi_filtered_path: str | None = None,
    ) -> dict[str, np.ndarray]:
        region_out = os.path.join(self.out_dir, region_name)
        os.makedirs(region_out, exist_ok=True)

        sigmas_mm, sigmas_vox = self._adapt_sigmas(sigmas)
        local_frangi_path = os.path.join(region_out, "raw_frangi.nii.gz")
        metadata_path = os.path.join(region_out, "frangi_params.json")
        metadata = self._frangi_metadata(threshold, sigmas_mm, sigmas_vox)

        if os.path.exists(local_frangi_path) and self._frangi_metadata_matches(metadata_path, metadata):
            frangi_filtered = nib.load(local_frangi_path).get_fdata()
        elif frangi_filtered_path is not None and os.path.exists(frangi_filtered_path):
            frangi_filtered = nib.load(frangi_filtered_path).get_fdata()
        else:
            frangi_filtered = frangi2d(
                image=self._normalised_array(preproc),
                black_ridges=False,
                sigmas=sigmas_vox,
                mask=region_mask,
            )
            if self.save:
                self._save_nifti(frangi_filtered, local_frangi_path)
                self._save_frangi_metadata(metadata_path, metadata)

        pvs_candidates = ((frangi_filtered > threshold) & (region_mask > 0)).astype(np.uint8)
        if self.save:
            self._save_nifti(pvs_candidates, os.path.join(region_out, "pvs_candidates.nii.gz"))

        return {
            "frangi": frangi_filtered,
            "pvs_candidates": pvs_candidates,
            "region_mask": region_mask,
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
        frangi_filtered_path: str | None = None,
        fp_mode: str = "2d",
        csf_gm_dilation_mm: float = 2.0,
        ventricle_dilation_mm: float = 1.5,
        min_area_mm2: float = 0.5,
        max_area_mm2: float = 15.0,
    ) -> dict[str, np.ndarray]:
        preproc = self._run_preprocessing(preproc_overwrite=preproc_overwrite)
        bg_threshold, cso_threshold = self._resolve_thresholds(bg_thr, cso_thr)

        bg_result = self._run_region_candidate(
            preproc=preproc,
            region_name="BG",
            region_mask=self._bg_mask(preproc),
            sigmas=self._resolve_sigmas(bg_sigmas, sigmas),
            threshold=bg_threshold,
            frangi_filtered_path=frangi_filtered_path,
        )
        cso_result = self._run_region_candidate(
            preproc=preproc,
            region_name="CSO",
            region_mask=self._cso_mask(preproc),
            sigmas=self._resolve_sigmas(cso_sigmas, sigmas),
            threshold=cso_threshold,
            frangi_filtered_path=None,
        )

        bg_fp = self._run_fp_reduction(
            preproc=preproc,
            region_name="BG",
            candidates=bg_result["pvs_candidates"],
            roi_mask=self._bg_mask(preproc),
            fp_mode=fp_mode,
            csf_gm_dilation_mm=csf_gm_dilation_mm,
            ventricle_dilation_mm=ventricle_dilation_mm,
            min_area_mm2=min_area_mm2,
            max_area_mm2=max_area_mm2,
        )
        cso_fp = self._run_fp_reduction(
            preproc=preproc,
            region_name="CSO",
            candidates=cso_result["pvs_candidates"],
            roi_mask=self._cso_mask(preproc),
            fp_mode=fp_mode,
            csf_gm_dilation_mm=csf_gm_dilation_mm,
            ventricle_dilation_mm=ventricle_dilation_mm,
            min_area_mm2=min_area_mm2,
            max_area_mm2=max_area_mm2,
        )

        segmentation_mask = ((bg_fp > 0) | (cso_fp > 0)).astype(np.uint8)
        if self.save:
            self._save_nifti(segmentation_mask, os.path.join(self.out_dir, "pvs_segmentation_mask.nii.gz"))

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
        fp_mode: str,
        csf_gm_dilation_mm: float,
        ventricle_dilation_mm: float,
        min_area_mm2: float,
        max_area_mm2: float,
    ) -> np.ndarray:
        fp_reduction = FPReduction(
            t2_nib=self.t2_nib,
            seg_array=candidates,
            roi_mask=roi_mask,
            exclusion_mask=preproc.exclusion_mask,
            csf_mask=preproc.csf_mask,
            gm_mask=preproc.gm_mask,
            ventricle_mask=preproc.ventricle_mask,
            csf_gm_dilation_mm=csf_gm_dilation_mm,
            ventricle_dilation_mm=ventricle_dilation_mm,
            min_area_mm2=min_area_mm2,
            max_area_mm2=max_area_mm2,
        )
        fp_reduced = fp_reduction.run(mode=fp_mode)
        if self.save:
            self._save_nifti(
                fp_reduced,
                os.path.join(self.out_dir, region_name, "pvs_fp_reduced.nii.gz"),
            )
        return fp_reduced

if __name__ == "__main__":
    pass
    # Usage
    # -----
    # 이 파일은 직접 실행하기보다는 아래처럼 import해서 사용하는 것을 권장합니다.
    #
    # 1. Candidate만 생성하는 경우
    #
    # from PVS_Seg_Module_Local.runner import PVSCandidateRunner
    # import numpy as np
    #
    # runner = PVSCandidateRunner(
    #     t2_path="E:/code_organization/input_samples/T2/subj01.nii.gz",
    #     out_dir="E:/code_organization/output_samples/pvs_segmentation/subj01/candidate",
    #     save_all=True,
    #     bg_thr=0.01,
    #     cso_thr=0.02,
    # )
    # outputs = runner.run(
    #     bg_sigmas=np.linspace(1.0, 2.0, 2),
    #     cso_sigmas=np.linspace(2.0, 3.0, 2),
    #     preproc_overwrite=False,
    # )
    #
    # 저장 구조:
    # out_dir/
    #     preproc/
    #         brain_mask.nii.gz
    #         native_lobe.nii.gz
    #         native_roi_mask.nii.gz
    #         native_synthseg.nii.gz
    #         t2_raw.nii.gz
    #         t2_norm.nii.gz
    #     BG/
    #         raw_frangi.nii.gz
    #         pvs_candidates.nii.gz
    #     CSO/
    #         raw_frangi.nii.gz
    #         pvs_candidates.nii.gz
    #
    # 2. False-positive reduction까지 포함한 full pipeline
    #
    # from PVS_Seg_Module_Local.runner import PVSSegRunner
    # import numpy as np
    #
    # runner = PVSSegRunner(
    #     t2_path="E:/code_organization/input_samples/T2/subj01.nii.gz",
    #     out_dir="E:/code_organization/output_samples/pvs_segmentation/subj01/full",
    #     save_all=True,
    #     bg_thr=0.01,
    #     cso_thr=0.02,
    # )
    # outputs = runner.run(
    #     bg_sigmas=np.linspace(1.0, 2.0, 2),
    #     cso_sigmas=np.linspace(2.0, 3.0, 2),
    #     preproc_overwrite=False,
    #     fp_mode="2d",
    #     csf_gm_dilation_mm=2.0,
    #     ventricle_dilation_mm=1.5,
    #     min_area_mm2=0.5,
    #     max_area_mm2=15.0,
    # )
    #
    # 저장 구조:
    # out_dir/
    #     preproc/
    #         brain_mask.nii.gz
    #         native_lobe.nii.gz
    #         native_roi_mask.nii.gz
    #         native_synthseg.nii.gz
    #         t2_raw.nii.gz
    #         t2_norm.nii.gz
    #     BG/
    #         raw_frangi.nii.gz
    #         pvs_candidates.nii.gz
    #         pvs_fp_reduced.nii.gz
    #     CSO/
    #         raw_frangi.nii.gz
    #         pvs_candidates.nii.gz
    #         pvs_fp_reduced.nii.gz
    #     pvs_segmentation_mask.nii.gz
    #
    # Notes
    # -----
    # - BG와 CSO는 각각 다른 sigmas/threshold를 사용할 수 있습니다.
    # - sigmas는 mm 단위로 입력합니다. runner가 input T2의 in-plane spacing을
    #   기준으로 voxel sigma로 변환한 뒤 Frangi filter에 전달합니다.
    #   예: spacing=0.5 mm, sigma=1.0 mm -> sigma=2.0 vox
    # - raw_frangi.nii.gz는 frangi_params.json의 spacing/sigmas/threshold가
    #   현재 설정과 일치할 때만 재사용됩니다.
    # - preprocessing은 한 번만 수행되고 BG/CSO 처리에서 공유됩니다.
    # - preproc_overwrite=False이면 이미 저장된 native_synthseg, t2_norm 등을 재사용합니다.
