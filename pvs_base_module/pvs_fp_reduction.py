"""False-positive reduction for Frangi-based PVS candidate masks."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import measure


class FPReduction:
    """
    Slice-wise connected-component filtering for PVS candidate masks.

    Current filtering rules are intentionally simple:
    - remove components outside the configured 2-D area range,
    - remove components touching dilated CSF or dilated ventricles,
    - remove components with >= gm_overlap_thres overlap with non-dilated GM,
    - optionally remove components with elongation > max_elongation,
    - apply the ROI mask only at the final output stage.

    CSF/ventricle dilation is performed only in-plane. This matches the 2-D
    Frangi candidate generation and avoids suppressing candidates through thick
    slice direction. When ``csf_dilation_mm`` is ``None``, CSF is dilated by
    exactly one in-plane voxel. Ventricle dilation is controlled directly by
    ``ventricle_dilation_voxels``.
    """

    def __init__(
        self,
        t2_nib: nib.Nifti1Image,
        seg_array: np.ndarray,
        roi_mask: np.ndarray,
        csf_mask: np.ndarray | None = None,
        gm_mask: np.ndarray | None = None,
        ventricle_mask: np.ndarray | None = None,
        csf_dilation_mm: float | None = None,
        ventricle_dilation_voxels: int = 1,
        min_area_mm2: float = 1.0,
        max_area_mm2: float = 12.0,
        gm_overlap_thres: float = 0.5,
        max_elongation: float | None = None,
    ):
        self.t2_nib = t2_nib
        self.voxel_size = np.array(t2_nib.header.get_zooms()[:3], dtype=float)
        self.slice_axis = self._superior_inferior_axis(t2_nib)
        self.inplane_axes = tuple(ax for ax in range(3) if ax != self.slice_axis)
        self.inplane_shape = tuple(int(t2_nib.shape[ax]) for ax in self.inplane_axes)

        self.seg_array = (np.asarray(seg_array) > 0).astype(np.uint8)
        self.roi_mask = (np.asarray(roi_mask) > 0).astype(np.uint8)

        self.csf_mask = self._as_mask(csf_mask)
        self.gm_mask = self._as_mask(gm_mask)
        self.ventricle_mask = self._as_mask(ventricle_mask)

        self.csf_dilation_mm = csf_dilation_mm
        self.ventricle_dilation_voxels = int(ventricle_dilation_voxels)
        self.min_area_mm2 = min_area_mm2
        self.max_area_mm2 = max_area_mm2
        self.gm_overlap_thres = gm_overlap_thres
        self.max_elongation = max_elongation

        self._validate_shapes()
        self._set_area_thresholds()

    @staticmethod
    def _superior_inferior_axis(nii: nib.Nifti1Image) -> int:
        try:
            codes = np.array(nib.orientations.aff2axcodes(nii.affine))
            return int(np.where((codes == "S") | (codes == "I"))[0][0])
        except Exception:
            return 2

    @staticmethod
    def _as_mask(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        return (np.asarray(mask) > 0).astype(np.uint8)

    def _validate_shapes(self) -> None:
        expected_shape = self.seg_array.shape
        if self.roi_mask.shape != expected_shape:
            raise ValueError("roi_mask shape must match seg_array shape.")

        for name in ("csf_mask", "gm_mask", "ventricle_mask"):
            mask = getattr(self, name)
            if mask is not None and mask.shape != expected_shape:
                raise ValueError(f"{name} shape must match seg_array shape.")

        if self.csf_mask is None and self.ventricle_mask is None and self.gm_mask is None:
            raise ValueError("Provide at least one of csf_mask, ventricle_mask, or gm_mask.")

    def _set_area_thresholds(self) -> None:
        pix_area = float(np.prod(self.voxel_size[list(self.inplane_axes)]))
        self.min_area_px = max(1, int(np.ceil(self.min_area_mm2 / pix_area)))
        self.max_area_px = int(np.floor(self.max_area_mm2 / pix_area))
        if self.max_area_px < self.min_area_px:
            raise ValueError(
                "max_area_mm2 is smaller than min_area_mm2 after voxel-size conversion."
            )

    def _csf_radius_px(self, radius_mm: float | None) -> int:
        if radius_mm is None:
            return 1
        if radius_mm <= 0:
            return 0
        return max(1, int(round(radius_mm / float(np.mean(self.voxel_size[list(self.inplane_axes)])))))

    @staticmethod
    def _voxel_radius_px(radius_voxels: int) -> int:
        return max(0, int(radius_voxels))

    def _dilate_2d(self, mask: np.ndarray, radius_px: int) -> np.ndarray:
        if radius_px == 0:
            return mask.astype(bool)

        structure = ndi.generate_binary_structure(2, 1)
        return ndi.binary_dilation(mask.astype(bool), structure=structure, iterations=radius_px)

    def _take_slice(self, array: np.ndarray, slice_idx: int) -> np.ndarray:
        return np.take(array, slice_idx, axis=self.slice_axis)

    def _write_slice(self, array: np.ndarray, slice_idx: int, plane: np.ndarray) -> None:
        index = [slice(None)] * array.ndim
        index[self.slice_axis] = slice_idx
        array[tuple(index)] = plane

    def _slice_exclusion_masks(self, slice_idx: int) -> tuple[np.ndarray, np.ndarray]:
        csf_vent_masks = []
        if self.csf_mask is not None:
            csf_vent_masks.append(
                self._dilate_2d(
                    self._take_slice(self.csf_mask, slice_idx),
                    self._csf_radius_px(self.csf_dilation_mm),
                )
            )
        if self.ventricle_mask is not None:
            csf_vent_masks.append(
                self._dilate_2d(
                    self._take_slice(self.ventricle_mask, slice_idx),
                    self._voxel_radius_px(self.ventricle_dilation_voxels),
                )
            )

        csf_vent = (
            np.logical_or.reduce(csf_vent_masks)
            if csf_vent_masks
            else np.zeros(self.inplane_shape, dtype=bool)
        )

        gm = (
            self._take_slice(self.gm_mask, slice_idx).astype(bool)
            if self.gm_mask is not None
            else np.zeros(self.inplane_shape, dtype=bool)
        )
        return csf_vent, gm

    def _fp_comp_2d(
        self,
        seg: np.ndarray,
        csf_vent_mask: np.ndarray,
        gm_mask: np.ndarray,
    ) -> np.ndarray:
        labeled = measure.label(seg, connectivity=2)
        keep_mask = np.zeros_like(seg, dtype=np.uint8)

        for prop in measure.regionprops(labeled):
            area_px = int(prop.area)
            if not (self.min_area_px <= area_px <= self.max_area_px):
                continue

            comp_mask = labeled == prop.label
            if np.count_nonzero(comp_mask & csf_vent_mask) > 0:
                continue

            gm_overlap = np.count_nonzero(comp_mask & gm_mask) / float(max(1, area_px))
            if gm_overlap >= self.gm_overlap_thres:
                continue

            if self.max_elongation is not None:
                minor_px = float(prop.minor_axis_length)
                major_px = float(prop.major_axis_length)
                elongation = major_px / minor_px if minor_px > 1e-6 else np.inf
                if elongation > self.max_elongation:
                    continue

            keep_mask[comp_mask] = 1

        return keep_mask

    def run(self, mode: str = "2d") -> np.ndarray:
        if mode != "2d":
            raise NotImplementedError(
                f"FPReduction mode {mode!r} is not implemented. Only '2d' is supported."
            )

        filtered = np.zeros_like(self.seg_array, dtype=np.uint8)
        for idx in range(self.seg_array.shape[self.slice_axis]):
            csf_vent_mask, gm_mask = self._slice_exclusion_masks(idx)
            filtered_slice = self._fp_comp_2d(
                self._take_slice(self.seg_array, idx),
                csf_vent_mask,
                gm_mask,
            )
            self._write_slice(filtered, idx, filtered_slice)

        return (filtered * self.roi_mask).astype(np.uint8)
