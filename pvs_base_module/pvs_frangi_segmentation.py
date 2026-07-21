"""2-D slice-wise Frangi vesselness filtering for PVS candidate detection."""

from __future__ import annotations

import functools
import math
from itertools import combinations_with_replacement

import numpy as np
from scipy import ndimage as ndi
from skimage._shared.utils import _supported_float_type
from skimage.util import img_as_float


def _hessian_matrix_with_gaussian(image, sigma=1, mode="reflect", cval=0, order="rc"):
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if image.ndim > 2 and order == "xy":
        raise ValueError("order='xy' is only supported for 2-D images.")
    if order not in ("rc", "xy"):
        raise ValueError(f"Unrecognised order: {order!r}. Must be 'rc' or 'xy'.")

    if np.isscalar(sigma):
        sigma = (sigma,) * image.ndim

    truncate = 8 if all(s > 1 for s in sigma) else 100
    sigma_scaled = tuple(math.sqrt(2) * s for s in sigma)
    gaussian_ = functools.partial(
        ndi.gaussian_filter,
        sigma=sigma_scaled,
        mode=mode,
        cval=cval,
        truncate=truncate,
    )

    ndim = image.ndim
    orders = tuple([0] * d + [1] + [0] * (ndim - d - 1) for d in range(ndim))
    gradients = [gaussian_(image, order=orders[d]) for d in range(ndim)]

    axes = range(ndim)
    if order == "xy":
        axes = reversed(axes)

    return [
        gaussian_(gradients[ax0], order=orders[ax1])
        for ax0, ax1 in combinations_with_replacement(axes, 2)
    ]


def hessian_matrix(image, sigma=1, mode="constant", cval=0, order="rc"):
    return _hessian_matrix_with_gaussian(
        image,
        sigma=sigma,
        mode=mode,
        cval=cval,
        order=order,
    )


def _symmetric_image(s_elems):
    image = s_elems[0]
    sym = np.zeros(image.shape + (image.ndim, image.ndim), dtype=s_elems[0].dtype)
    for idx, (row, col) in enumerate(combinations_with_replacement(range(image.ndim), 2)):
        sym[..., row, col] = s_elems[idx]
        sym[..., col, row] = s_elems[idx]
    return sym


def _symmetric_compute_eigenvalues(s_elems):
    if len(s_elems) == 3:
        m00, m01, m11 = s_elems
        eigs = np.empty((2, *m00.shape), dtype=m00.dtype)
        half_trace = (m00 + m11) / 2
        half_sqrt_disc = np.sqrt(m01**2 + ((m00 - m11) / 2) ** 2)
        eigs[0] = half_trace + half_sqrt_disc
        eigs[1] = half_trace - half_sqrt_disc
        return eigs

    matrices = _symmetric_image(s_elems)
    eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
    leading_axes = tuple(range(eigs.ndim - 1))
    return np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)


def hessian_matrix_eigvals(h_elems):
    return _symmetric_compute_eigenvalues(h_elems)


def frangi2d(
    image: np.ndarray,
    sigmas,
    alpha=0.5,
    beta=0.5,
    gamma=None,
    black_ridges=False,
    mode="reflect",
    cval=0,
    mask=None,
    slice_axis: int = 2,
) -> np.ndarray:
    """Apply Frangi filtering independently along the requested slice axis."""
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    slice_axis = int(slice_axis)
    if slice_axis < 0:
        slice_axis += image.ndim
    if image.ndim != 3 or slice_axis not in (0, 1, 2):
        raise ValueError("frangi2d expects a 3-D image and slice_axis in {0, 1, 2}.")

    if not black_ridges:
        image = -image

    mask_arr = np.asarray(mask) if mask is not None else np.ones_like(image)
    image_work = np.moveaxis(image, slice_axis, 2)
    mask_work = np.moveaxis(mask_arr, slice_axis, 2)
    filtered_max = np.zeros_like(image_work)

    for sigma in sigmas:
        eigvals = np.zeros((2,) + image_work.shape, dtype=image_work.dtype)
        for i in range(image_work.shape[2]):
            eigvals[:, :, :, i] = hessian_matrix_eigvals(
                hessian_matrix(image_work[:, :, i], sigma, mode=mode, cval=cval)
            )

        eigvals = np.take_along_axis(eigvals, np.abs(eigvals).argsort(0), axis=0)
        lambda1 = eigvals[0]
        lambda2 = np.maximum(eigvals[1], 1e-10)

        r_b = np.abs(lambda1) / lambda2
        s = np.sqrt((eigvals**2).sum(axis=0))

        if gamma is None:
            gamma = (s * mask_work).max() / 2
            if gamma == 0:
                gamma = 1

        vals = np.ones_like(image_work)
        vals *= np.exp(-(r_b**2) / (2 * beta**2))
        vals *= 1.0 - np.exp(-(s**2) / (2 * gamma**2))
        vals *= mask_work

        filtered_max = np.maximum(filtered_max, vals)

    return np.moveaxis(filtered_max, 2, slice_axis)
