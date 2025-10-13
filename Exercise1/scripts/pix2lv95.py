from typing import Optional, List, Dict, Tuple

import numpy as np


def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes a set of 2D points for numerical stability in geometric transformations.
    The normalization centers the points at the origin and scales them so that the average distance
    from the origin is sqrt(2). Returns the normalized points and the normalization matrix.

    Parameters
    ----------
    pts : array-like
        Array of 2D points, shape (N, 2).

    Returns
    -------
    norm : np.ndarray
        Normalized points, shape (N, 2).
    T : np.ndarray
        Normalization transformation matrix, shape (3, 3).
    """
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    shifted = pts - mean
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    pts_h = np.c_[pts, np.ones(len(pts))]
    norm = (T @ pts_h.T).T
    return norm[:, :2], T


def _dlt_homography(pix: np.ndarray, geo: np.ndarray) -> np.ndarray:
    """
    Computes the homography (projective transformation) matrix from pixel coordinates to geocoordinates
    using the Direct Linear Transform (DLT) algorithm with at least four ground control points (GCPs).

    Parameters
    ----------
    pix : array-like
        Array of pixel coordinates, shape (N, 2).
    geo : array-like
        Array of geocoordinates (e.g., LV95), shape (N, 2).

    Returns
    -------
    np.ndarray
        Homography transformation matrix of shape (3, 3) that maps pixel coordinates to geocoordinates.

    Raises
    ------
    ValueError
        If fewer than 4 GCPs are provided.
    """
    if len(pix) < 4:
        raise ValueError("Need at least 4 GCPs for homography.")
    pix = np.asarray(pix, float)
    geo = np.asarray(geo, float)

    # Normalize both sets
    pix_n, T_pix = _normalize_points(pix)
    geo_n, T_geo = _normalize_points(geo)

    # Build DLT system: A h = 0
    A = []
    for (x, y), (E, N) in zip(pix_n, geo_n):
        A.append([0, 0, 0, -x, -y, -1, N * x, N * y, N])
        A.append([x, y, 1, 0, 0, 0, -E * x, -E * y, -E])
    A = np.asarray(A, float)

    # Solve via SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # last row
    Hn = h.reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T_geo) @ Hn @ T_pix
    H /= H[2, 2]
    return H


def _affine_from_gcp(pix: np.ndarray, geo: np.ndarray) -> np.ndarray:
    """
    Computes the affine transformation matrix from pixel coordinates to geocoordinates
    using at least three ground control points (GCPs).

    Parameters
    ----------
    pix : array-like
        Array of pixel coordinates, shape (N, 2).
    geo : array-like
        Array of geocoordinates (e.g., LV95), shape (N, 2).

    Returns
    -------
    np.ndarray
        Affine transformation matrix of shape (2, 3) that maps pixel coordinates to geocoordinates.

    Raises
    ------
    ValueError
        If fewer than 3 GCPs are provided.
    """
    if len(pix) < 3:
        raise ValueError("Need at least 3 GCPs for affine.")
    pix = np.asarray(pix, float)
    geo = np.asarray(geo, float)
    X = np.c_[pix, np.ones(len(pix))]  # N x 3
    # Solve two independent least squares for E and N
    A_E, _, _, _ = np.linalg.lstsq(X, geo[:, 0], rcond=None)
    A_N, _, _, _ = np.linalg.lstsq(X, geo[:, 1], rcond=None)
    A = np.vstack([A_E, A_N])  # 2x3
    return A


def fit_pixel_to_lv95_transform(gcp: List[Dict], method: Optional[str] = "homography") -> Dict:
    """
    Computes a transformation from pixel coordinates to LV95 geocoordinates using ground control points (GCPs).
    Supports both affine and homography (projective) transformations.

    Parameters
    ----------
    gcp : list of dict
        List of ground control points, each with keys:
            - "pixel": (x, y) pixel coordinates
            - "lv95": (E, N) LV95 geocoordinates
    method : str, optional
        Transformation type: "affine" for affine or "homography" for projective (default: "homography").

    Returns
    -------
    dict
        Dictionary with keys:
            - "type": "A" for affine or "H" for homography
            - "mat": transformation matrix (2x3 for affine, 3x3 for homography)
            - "rmse": root mean squared reprojection error (float)
    """
    pix = np.array([p["pixel"] for p in gcp], float)
    geo = np.array([p["lv95"] for p in gcp], float)

    # Optional: drop exact duplicates if present
    keep = []
    seen = set()
    for i, (px, gy) in enumerate(zip(map(tuple, pix), map(tuple, geo))):
        key = (px, gy)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    pix = pix[keep]
    geo = geo[keep]

    if method == "affine":
        A = _affine_from_gcp(pix, geo)
        # reprojection
        X = np.c_[pix, np.ones(len(pix))]
        pred = (A @ X.T).T
        err = np.linalg.norm(pred - geo, axis=1)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        return {"type": "A", "mat": A, "rmse": rmse}

    # default: homography
    H = _dlt_homography(pix, geo)
    # reprojection error
    pix_h = np.c_[pix, np.ones(len(pix))]
    proj = (H @ pix_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - geo, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"type": "H", "mat": H, "rmse": rmse}


def apply_transform_to_points(pts_pix: np.ndarray, T: dict) -> np.ndarray:
    """
    Applies a geometric transformation (affine or homography) to a set of pixel coordinates.

    Parameters
    ----------
    pts_pix : array-like
        Array of pixel coordinates to transform, shape (N, 2).
    T : dict
        Transformation dictionary. Must contain:
            - "type": either "A" (affine) or "H" (homography)
            - "mat": transformation matrix (2x3 for affine, 3x3 for homography)

    Returns
    -------
    np.ndarray
        Transformed coordinates, shape (N, 2).
    """
    pts_pix = np.asarray(pts_pix, float)
    if T["type"] == "A":
        X = np.c_[pts_pix, np.ones(len(pts_pix))]
        out = (T["mat"] @ X.T).T
        return out
    else:
        H = T["mat"]
        X = np.c_[pts_pix, np.ones(len(pts_pix))]
        Y = (H @ X.T).T
        Y = Y[:, :2] / Y[:, 2:3]
        return Y