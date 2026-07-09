"""False-positive reduction for Frangi-based PVS candidate masks."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from .pvs_preprocessing import TARGET_ROI_LABELS


class FPReduction:
    """
    Slice-wise connected-component filtering for PVS candidate masks.

    The exclusion mask is dilated only in-plane. This matches the axial,
    slice-wise PVS candidate generation and avoids suppressing candidates across
    thick-slice z direction. When split SynthSeg-derived masks are provided,
    CSF/GM and ventricles can use different dilation radii.
    """

    def __init__(
        self,
        t2_nib: nib.Nifti1Image,
        seg_array: np.ndarray,
        roi_mask: np.ndarray,
        exclusion_mask: np.ndarray | None = None,
        csf_mask: np.ndarray | None = None,
        gm_mask: np.ndarray | None = None,
        ventricle_mask: np.ndarray | None = None,
        dilation: bool = True,
        csf_gm_dilation_mm: float = 2.0,
        ventricle_dilation_mm: float = 1.5,
        fallback_dilation_mm: float = 1.5,
        min_area_mm2: float = 0.5,
        max_area_mm2: float = 15.0,
        circularity_thres: float = 2.0,
        cso_elongation_thres: float = 2.0,
        allow_single_voxel: bool = False,
    ):
        self.t2_nib = t2_nib
        self.voxel_size = np.array(t2_nib.header.get_zooms()[:3], dtype=float)

        self.seg_array = (np.asarray(seg_array) > 0).astype(np.uint8)
        self.bg_mask = (np.asarray(roi_mask) == TARGET_ROI_LABELS["BG"]).astype(np.uint8)
        self.roi_mask = (np.asarray(roi_mask) > 0).astype(np.uint8)

        self.exclusion_mask = self._as_mask(exclusion_mask)
        self.csf_mask = self._as_mask(csf_mask)
        self.gm_mask = self._as_mask(gm_mask)
        self.ventricle_mask = self._as_mask(ventricle_mask)

        self.dilation = dilation
        self.csf_gm_dilation_mm = csf_gm_dilation_mm
        self.ventricle_dilation_mm = ventricle_dilation_mm
        self.fallback_dilation_mm = fallback_dilation_mm
        self.min_area_mm2 = min_area_mm2
        self.max_area_mm2 = max_area_mm2
        self.circularity_thres = circularity_thres
        self.elongation_thres = cso_elongation_thres
        self.allow_single_voxel = allow_single_voxel

        self._validate_shapes()

    def _as_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        return (np.asarray(mask) > 0).astype(np.uint8)

    def _validate_shapes(self) -> None:
        expected_shape = self.seg_array.shape
        if self.roi_mask.shape != expected_shape:
            raise ValueError("roi_mask shape must match seg_array shape.")

        for name in ("exclusion_mask", "csf_mask", "gm_mask", "ventricle_mask"):
            mask = getattr(self, name)
            if mask is not None and mask.shape != expected_shape:
                raise ValueError(f"{name} shape must match seg_array shape.")

        has_split_masks = any(
            mask is not None for mask in (self.csf_mask, self.gm_mask, self.ventricle_mask)
        )
        if self.exclusion_mask is None and not has_split_masks:
            raise ValueError("Provide exclusion_mask or at least one split exclusion mask.")

    def _radius_px(self, radius_mm: float) -> int:
        if radius_mm <= 0:
            return 0
        return max(1, int(round(radius_mm / float(np.mean(self.voxel_size[:2])))))

    def _dilate_2d(self, mask: np.ndarray, radius_mm: float) -> np.ndarray:
        if not self.dilation:
            return mask.astype(bool)

        radius_px = self._radius_px(radius_mm)
        if radius_px == 0:
            return mask.astype(bool)

        structure = ndi.generate_binary_structure(2, 1)
        return ndi.binary_dilation(mask.astype(bool), structure=structure, iterations=radius_px)

    def _slice_exclusion_mask(self, slice_idx: int) -> np.ndarray:
        masks = []
        if self.csf_mask is not None:
            masks.append(self._dilate_2d(self.csf_mask[:, :, slice_idx], self.csf_gm_dilation_mm))
        if self.gm_mask is not None:
            masks.append(self._dilate_2d(self.gm_mask[:, :, slice_idx], self.csf_gm_dilation_mm))
        if self.ventricle_mask is not None:
            masks.append(self._dilate_2d(self.ventricle_mask[:, :, slice_idx], self.ventricle_dilation_mm))

        if masks:
            return np.logical_or.reduce(masks)

        return self._dilate_2d(self.exclusion_mask[:, :, slice_idx], self.fallback_dilation_mm)

    def _fp_comp_2d(
        self,
        seg: np.ndarray,
        exclusion_mask: np.ndarray,
        bg_mask: np.ndarray,
    ) -> np.ndarray:
        pix_area = float(np.prod(self.voxel_size[:2]))
        min_area_px = max(1, int(np.ceil(self.min_area_mm2 / pix_area)))
        max_area_px = int(np.floor(self.max_area_mm2 / pix_area))

        labeled = measure.label(seg, connectivity=2)
        keep_mask = np.zeros_like(seg, dtype=np.uint8)

        for prop in measure.regionprops(labeled):
            area_px = prop.area
            if not self.allow_single_voxel and area_px == 1:
                continue

            if not (min_area_px <= area_px <= max_area_px):
                continue

            comp_mask = labeled == prop.label
            if np.count_nonzero(comp_mask & exclusion_mask) > 0:
                continue

            minor_px = prop.minor_axis_length
            major_px = prop.major_axis_length
            elongation = (major_px / minor_px) if minor_px > 1e-6 else np.inf

            boundary = prop.filled_image ^ ndi.binary_erosion(prop.filled_image)
            perimeter = np.count_nonzero(boundary)
            circularity = 4 * np.pi * area_px / (perimeter**2) if perimeter > 0 else 0.0

            bg_overlap = np.count_nonzero(comp_mask & bg_mask) / float(max(1, area_px))
            is_bg = bg_overlap > 0.5

            if is_bg:
                if elongation > self.elongation_thres:
                    continue
            elif elongation < self.elongation_thres and circularity > self.circularity_thres:
                continue

            keep_mask[comp_mask] = 1

        return keep_mask

    def run(self, mode: str = "2d") -> np.ndarray:
        if mode != "2d":
            raise NotImplementedError(
                f"FPReduction mode {mode!r} is not implemented. Only '2d' is supported."
            )

        filtered = np.stack(
            [
                self._fp_comp_2d(
                    self.seg_array[:, :, i],
                    self._slice_exclusion_mask(i),
                    self.bg_mask[:, :, i],
                )
                for i in range(self.seg_array.shape[-1])
            ],
            axis=-1,
        )
        return (filtered * self.roi_mask).astype(np.uint8)
