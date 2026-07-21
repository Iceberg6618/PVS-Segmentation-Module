"""Preprocessing pipeline for T2-weighted MRI prior to PVS segmentation."""

from __future__ import annotations

import os
import tempfile

import ants
import nibabel as nib
import numpy as np

from Native_Synthseg_Module import NativeSynthSegConfig, NativeSynthSegRunner

MNI_LABEL_MAPS = {
    "Frontal": [1, 2],
    "Temporal": [17, 18],
    "Parietal": [3, 4],
    "Occipital": [7, 8],
}

SYNTHSEG_LABEL_MAPS = {
    "CSF": [24],
    "Ventricle": [4, 5, 14, 15, 43, 44],
    "Gray Matter": [3, 42],
    "White Matter": [2, 41],
    "Basal Ganglia": [10, 11, 12, 13, 49, 50, 51, 52],
}

NORMALIZATION_LABEL_MAPS = {
    "White Matter": [2, 41, 77],
    "Basal Ganglia": [10, 11, 12, 13, 49, 50, 51, 52, 26, 58],
}

TARGET_ROI_LABELS = {
    "BG": 1,
    "Frontal": 2,
    "Parietal": 3,
    "Temporal": 4,
    "Occipital": 5,
}


def _ants_image_from_nifti(nifti_image: nib.Nifti1Image):
    if hasattr(ants, "from_nibabel"):
        return ants.from_nibabel(nifti_image)

    temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    try:
        nib.save(nifti_image, temp_path)
        return ants.image_read(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _nifti_from_ants_image(ants_image) -> nib.Nifti1Image:
    if hasattr(ants, "to_nibabel"):
        return ants.to_nibabel(ants_image)

    temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    try:
        ants.image_write(ants_image, temp_path)
        loaded = nib.load(temp_path)
        return nib.Nifti1Image(loaded.get_fdata(), loaded.affine, loaded.header)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class PVSPreprocessing:
    """
    Orchestrates all pre-processing steps required before Frangi filtering.

    Steps:
      1. Cached SynthSeg parcellation in native MRI space.
      2. MNI lobar atlas registration to native space.
      3. Brain, ROI, and exclusion-mask generation.
      4. ROI-based T2 intensity normalisation.
    """

    def __init__(
        self,
        t2_nib: nib.Nifti1Image,
        t1_nib: nib.Nifti1Image | None = None,
        mni_src=os.path.join(os.path.dirname(__file__), "MNI_Atlas"),
        out_dir=None,
        save=True,
        overwrite=True,
        synthseg_runner: NativeSynthSegRunner | None = None,
        synthseg_config: NativeSynthSegConfig | None = None,
    ):
        self.t2_nib = t2_nib
        self.t2_nib.header["qform_code"] = 1
        self.t2_ants = _ants_image_from_nifti(self.t2_nib)
        self.img_np_arr = self.t2_nib.get_fdata()
        self.t1_nib = t1_nib
        self.t1_ants = None
        if self.t1_nib is not None:
            self.t1_nib.header["qform_code"] = 1
            self.t1_ants = _ants_image_from_nifti(self.t1_nib)

        self.out_dir = out_dir if out_dir is not None else "./preproc"
        os.makedirs(self.out_dir, exist_ok=True)

        self.mni_src = mni_src
        self.save = save
        self.overwrite = overwrite
        self.synthseg_runner = synthseg_runner
        self.synthseg_config = synthseg_config

        self.native_synthseg = None
        self.native_lobe = None
        self.brain_mask = None
        self.exclusion_mask = None
        self.csf_mask = None
        self.ventricle_mask = None
        self.gm_mask = None
        self.target_roi_mask = None
        self.mask = None
        self.normalization_mask = None
        self.int_norm_nib = None
        self.t2_normalized = None
        self.t1_to_t2_nib = None

    def synthseg(self) -> np.ndarray:
        """
        Run SynthSeg using a cached model, then register output to native space.

        The native-space SynthSeg module caches the model and registers the
        segmentation back to the source T2 image space.
        """
        cache_path = os.path.join(self.out_dir, "native_synthseg.nii.gz")

        if os.path.exists(cache_path) and not self.overwrite:
            print(f"[PVS][Preproc] native_synthseg exists. Loading file: {cache_path}")
            return nib.load(cache_path).get_fdata().squeeze().astype(np.uint8)

        print("[PVS][Preproc] SynthSeg started.")
        runner = self.synthseg_runner or NativeSynthSegRunner.get(self.synthseg_config)
        if self.t1_nib is None:
            synthseg_native_nib = runner.run_native(
                self.t2_nib,
                output_path=cache_path if (self.save or self.overwrite) else None,
                overwrite=True,
            )
        else:
            synthseg_native_nib = self._synthseg_from_t1_to_t2(runner, cache_path)

        print(f"[PVS][Preproc] SynthSeg done: {cache_path}")
        return synthseg_native_nib.get_fdata().squeeze().astype(np.uint8)

    def _synthseg_from_t1_to_t2(
        self,
        runner: NativeSynthSegRunner,
        cache_path: str,
    ) -> nib.Nifti1Image:
        print("[PVS][Preproc] T1 SynthSeg started.")
        t1_seg_nib = runner.run_native(self.t1_nib, output_path=None, overwrite=True)
        print("[PVS][Preproc] T1-to-T2 registration started.")

        t1_to_t2 = ants.registration(
            fixed=self.t2_ants,
            moving=self.t1_ants,
            type_of_transform="Affine",
            verbose=False,
        )
        self.t1_to_t2_nib = _nifti_from_ants_image(
            ants.apply_transforms(
                fixed=self.t2_ants,
                moving=self.t1_ants,
                transformlist=t1_to_t2["fwdtransforms"],
                interpolator="linear",
            )
        )

        t1_seg_ants = _ants_image_from_nifti(t1_seg_nib)
        synthseg_t2_ants = ants.apply_transforms(
            fixed=self.t2_ants,
            moving=t1_seg_ants,
            transformlist=t1_to_t2["fwdtransforms"],
            interpolator="genericLabel",
        )
        synthseg_t2_nib = _nifti_from_ants_image(synthseg_t2_ants)
        synthseg_t2_nib.header["qform_code"] = 1

        if self.save or self.overwrite:
            nib.save(synthseg_t2_nib, cache_path)
            nib.save(self.t1_to_t2_nib, os.path.join(self.out_dir, "t1_to_t2.nii.gz"))
        print("[PVS][Preproc] T1 SynthSeg registered to T2 native space done.")

        return synthseg_t2_nib

    def lobar_segmentation(self) -> np.ndarray:
        cache_path = os.path.join(self.out_dir, "native_lobe.nii.gz")

        if os.path.exists(cache_path) and not self.overwrite:
            print(f"[PVS][Preproc] native_lobe exists. Loading file: {cache_path}")
            return nib.load(cache_path).get_fdata().squeeze().astype(np.uint8)

        print("[PVS][Preproc] MNI lobe registration started.")
        mni_t2 = ants.image_read(
            os.path.join(
                self.mni_src,
                "mni_icbm152_nlin_asym_09c",
                "mni_icbm152_t2_tal_nlin_asym_09c.nii",
            )
        )
        mni_atlas = ants.image_read(os.path.join(self.mni_src, "Lobar_Map.nii"))

        mni_2_t2 = ants.registration(
            fixed=self.t2_ants,
            moving=mni_t2,
            type_of_transform="SyN",
        )

        native_lobe_ants = ants.apply_transforms(
            fixed=self.t2_ants,
            moving=mni_atlas,
            transformlist=mni_2_t2["fwdtransforms"],
            interpolator="genericLabel",
        )

        native_lobe_nib = _nifti_from_ants_image(native_lobe_ants)

        if self.save or self.overwrite:
            nib.save(native_lobe_nib, cache_path)
        print(f"[PVS][Preproc] MNI lobe registration done: {cache_path}")

        return native_lobe_nib.get_fdata().squeeze().astype(np.uint8)

    def get_masks(self):
        print("[PVS][Preproc] Mask generation started.")
        if self.native_synthseg is None:
            raise RuntimeError("native_synthseg is None. Call synthseg() first.")
        if self.native_lobe is None:
            raise RuntimeError("native_lobe is None. Call lobar_segmentation() first.")

        brain_mask = (self.native_synthseg > 0).astype(np.uint8)

        bg_mask = np.isin(self.native_synthseg, SYNTHSEG_LABEL_MAPS["Basal Ganglia"]).astype(np.uint8)
        wm_mask = np.isin(self.native_synthseg, SYNTHSEG_LABEL_MAPS["White Matter"]).astype(np.uint8)
        norm_bg_mask = np.isin(self.native_synthseg, NORMALIZATION_LABEL_MAPS["Basal Ganglia"]).astype(np.uint8)
        norm_wm_mask = np.isin(self.native_synthseg, NORMALIZATION_LABEL_MAPS["White Matter"]).astype(np.uint8)
        normalization_mask = norm_bg_mask + norm_wm_mask

        lobar_roi = np.zeros_like(self.native_lobe, dtype=np.uint8)
        for lobe in ("Frontal", "Parietal", "Temporal", "Occipital"):
            lobar_roi[np.isin(self.native_lobe, MNI_LABEL_MAPS[lobe])] = TARGET_ROI_LABELS[lobe]
        lobar_roi *= wm_mask
        roi_mask = (lobar_roi + bg_mask).astype(np.uint8)

        csf = np.isin(self.native_synthseg, SYNTHSEG_LABEL_MAPS["CSF"]).astype(np.uint8)
        ventricle = np.isin(self.native_synthseg, SYNTHSEG_LABEL_MAPS["Ventricle"]).astype(np.uint8)
        gm = np.isin(self.native_synthseg, SYNTHSEG_LABEL_MAPS["Gray Matter"]).astype(np.uint8)
        exclusion_mask = (csf + ventricle + gm).astype(np.uint8)

        self.csf_mask = csf
        self.ventricle_mask = ventricle
        self.gm_mask = gm
        self.normalization_mask = normalization_mask
        self.mask = normalization_mask

        if self.save:
            affine, header = self.t2_nib.affine, self.t2_nib.header
            nib.save(nib.Nifti1Image(brain_mask, affine, header), os.path.join(self.out_dir, "brain_mask.nii.gz"))
            nib.save(nib.Nifti1Image(roi_mask, affine, header), os.path.join(self.out_dir, "native_roi_mask.nii.gz"))
        print(
            f"[PVS][Preproc] Mask generation done | brain={int(brain_mask.sum())}, "
            f"roi={int(np.count_nonzero(roi_mask))}"
        )

        return brain_mask, roi_mask, exclusion_mask

    def intensity_normalization(self, qmin: float = 0, qmax: float = 99) -> nib.Nifti1Image:
        assert self.mask is not None, "unable to process intensity normalization without mask..."
        print(f"[PVS][Preproc] Intensity normalization started | qmin={qmin}, qmax={qmax}")
        img_norm = self._normalize_with_mask(qmin=qmin, qmax=qmax, clip_upper=False)
        self.int_norm_nib = nib.Nifti1Image(img_norm, self.t2_nib.affine, self.t2_nib.header)

        if self.save:
            nib.save(self.int_norm_nib, os.path.join(self.out_dir, "t2_norm.nii.gz"))
            nib.save(self.t2_nib, os.path.join(self.out_dir, "t2_raw.nii.gz"))
            if self.t1_nib is not None:
                nib.save(self.t1_nib, os.path.join(self.out_dir, "t1_raw.nii.gz"))
                if self.t1_to_t2_nib is not None:
                    nib.save(self.t1_to_t2_nib, os.path.join(self.out_dir, "t1_to_t2.nii.gz"))
        print(f"[PVS][Preproc] Intensity normalization done: {os.path.join(self.out_dir, 't2_norm.nii.gz')}")

        return self.int_norm_nib

    def _normalize_with_mask(self, qmin: float, qmax: float, clip_upper: bool) -> np.ndarray:
        in_min, in_max = np.percentile(self.img_np_arr[self.mask > 0], [qmin, qmax])
        img_norm = (self.img_np_arr - in_min) / (in_max - in_min)
        img_norm = np.where(img_norm < 0, 0, img_norm)
        if clip_upper:
            img_norm = np.where(img_norm > 1, 1, img_norm)
        return img_norm

    def save_normalization_options(self, output_dir: str) -> dict[int, str]:
        assert self.mask is not None, "unable to save normalization options without mask..."
        os.makedirs(output_dir, exist_ok=True)

        options = {
            1: self._normalize_with_mask(qmin=0, qmax=95, clip_upper=False),
            2: self._normalize_with_mask(qmin=0, qmax=95, clip_upper=True),
            3: self._normalize_with_mask(qmin=0, qmax=99, clip_upper=False),
        }

        saved_paths = {}
        for option, img_norm in options.items():
            path = os.path.join(output_dir, f"t2_norm_option{option}.nii.gz")
            nib.save(nib.Nifti1Image(img_norm, self.t2_nib.affine, self.t2_nib.header), path)
            saved_paths[option] = path

        return saved_paths

    def run_all_preprocesses(self):
        print(f"[PVS][Preproc] Full preprocessing pipeline started | out={self.out_dir}")
        self.native_synthseg = self.synthseg()
        self.native_lobe = self.lobar_segmentation()
        self.brain_mask, self.target_roi_mask, self.exclusion_mask = self.get_masks()
        self.t2_normalized = self.intensity_normalization()
        print("[PVS][Preproc] Full preprocessing pipeline done.")
