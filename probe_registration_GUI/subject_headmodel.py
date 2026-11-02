"""
Subject head model from an SPM12 segmentation.

Tissue labels (0 air, 1 skin, 2 skull, 3 CSF, 4 gray, 5 white) are read either
from a single indexed label image (spm12_segment_for_NIRS.m) or combined from
SPM12 tissue probability maps (c1..c6, most probable tissue per voxel).  An
optional manually drawn lesion mask can be attached; for light modelling it is
folded in as a 6th tissue type (label 6 = lesion, see ``labels_with_lesion``).
A label image that already contains label 6 is split back into tissue labels
plus a lesion mask on loading.  The labels are turned
into surfaces in scanner/world mm (NIfTI affine).  The template probe and the
10-20 landmarks are defined on the atlas head (atlas_ICBM152_and_probe.mat), so
they are transferred to the subject by registering the atlas scalp to the
subject scalp (9-dof: rigid + per-axis scale, trimmed point-to-plane ICP) and
projecting onto the subject scalp.  The result is an ``Atlas`` object, so the
digitizer registration code in digitizer_registration.py works unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes

import digitizer_registration as dr

LABELS = {0: "air", 1: "skin", 2: "skull", 3: "CSF", 4: "gray", 5: "white"}
LESION_LABEL = 6
ALL_LABELS = {**LABELS, LESION_LABEL: "lesion"}
LESION_TISSUES = (3, 4, 5)          # tissues a lesion may replace (intracranial: CSF, gray, white)
LABEL_DESCRIP = b"NIRS seg 0air 1skin 2skull 3CSF 4GM 5WM 6lesion"
# SPM12 tissue classes c1..c6 (GM, WM, CSF, bone, soft tissue, air) -> NIRS labels
SPM_TO_NIRS = {1: 4, 2: 5, 3: 3, 4: 2, 5: 1, 6: 0}


def _as_list(paths):
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(p) for p in paths]


def spm_class_of(path):
    """SPM tissue class (1-6) from a file name such as c1T1w.nii, rc2sub.nii.gz; None if not recognised."""
    m = re.match(r"^(?:r|w|mw|m)?c([1-6])", Path(path).name)
    return int(m.group(1)) if m else None


def _same_grid(img, shape, affine, tol=1e-3):
    return tuple(img.shape[:3]) == tuple(shape) and np.allclose(img.affine, affine, atol=tol)


def combine_tissue_probabilities(paths, keep_largest=True):
    """Most probable tissue per voxel from SPM12 probability maps -> (labels uint8, affine).

    ``paths`` is either the separate class images (c1..c6 / c1..c5, class taken
    from the c<k> prefix, otherwise from the order given) or one 4-D image with
    the classes as volumes.  If the air class (c6) is missing it is taken as
    1 - sum(other classes).
    """
    paths = _as_list(paths)
    vols = []                                        # (spm class, callable returning the 3-D map)
    if len(paths) == 1:
        img = nib.load(paths[0])
        if img.ndim != 4 or img.shape[3] < 5:
            raise ValueError(f"{Path(paths[0]).name} is a single 3-D map; select all of the SPM tissue "
                             "probability maps (c1..c6) or one 4-D file with the classes as volumes.")
        shape, aff = img.shape[:3], img.affine
        for k in range(min(img.shape[3], 6)):
            vols.append((k + 1, lambda k=k: np.asarray(img.dataobj[..., k], dtype=np.float32)))
    else:
        classes = [spm_class_of(p) for p in paths]
        if None in classes or len(set(classes)) != len(classes):
            if len(paths) not in (5, 6):
                raise ValueError("Could not tell which SPM tissue class each file is (expected names starting "
                                 "with c1..c6, e.g. c1T1w.nii), and the files are not exactly 5 or 6 maps in "
                                 "the order GM, WM, CSF, bone, soft tissue[, air].")
            classes = list(range(1, len(paths) + 1))
            order = list(range(len(paths)))
        else:
            order = list(np.argsort(classes))
        missing = {1, 2, 3, 4, 5} - set(classes)
        if missing:
            raise ValueError(f"Missing SPM tissue class(es) {sorted(missing)} (need GM, WM, CSF, bone and soft tissue).")
        ref = nib.load(paths[order[0]])
        shape, aff = ref.shape[:3], ref.affine
        for i in order:
            img = nib.load(paths[i])
            if not _same_grid(img, shape, aff):
                raise ValueError(f"{Path(paths[i]).name} is not on the same voxel grid as "
                                 f"{Path(paths[order[0]]).name}; use the native-space c* maps.")
            vols.append((classes[i], lambda img=img: np.asarray(img.get_fdata(dtype=np.float32)).reshape(shape)))

    # running arg-max (avoids holding all six float volumes at once)
    best = np.full(shape, -1.0, np.float32)
    lab = np.zeros(shape, np.uint8)
    total = np.zeros(shape, np.float32)
    have_air = any(c == 6 for c, _ in vols)
    for c, get in vols:
        p = get()
        total += p
        upd = p > best
        best[upd] = p[upd]
        lab[upd] = SPM_TO_NIRS[c]
    if not have_air:
        lab[(1.0 - total) > best] = 0
    lab[best <= 0] = 0

    if keep_largest:
        cc, n = ndimage.label(lab > 0, structure=np.ones((3, 3, 3)))
        if n > 1:
            lab[cc != (np.argmax(np.bincount(cc.ravel())[1:]) + 1)] = 0
    return lab, aff


def load_segmentation(paths):
    """Tissue labels (0 air, 1 skin, 2 skull, 3 CSF, 4 GM, 5 WM, optionally 6 lesion) and voxel->world affine.

    Accepts either a single indexed label image (spm12_segment_for_NIRS.m) or
    SPM12 tissue probability maps (several c* files or one 4-D file), which are
    combined into the most probable tissue per voxel.  Use ``split_lesion_label``
    to separate a label-6 lesion from the tissue labels.
    """
    paths = _as_list(paths)
    if len(paths) > 1:
        return combine_tissue_probabilities(paths)
    img = nib.load(paths[0])
    if img.ndim == 4 and img.shape[3] > 1:
        return combine_tissue_probabilities(paths)
    data = np.asarray(img.dataobj)
    data = data.reshape(data.shape[:3])
    if not np.allclose(data, np.round(data)):
        raise ValueError(f"{Path(paths[0]).name} is not an indexed label image (non-integer values). "
                         "To use SPM probability maps select all of the c1..c6 files together.")
    if data.min() < 0 or data.max() > LESION_LABEL:
        raise ValueError(f"{Path(paths[0]).name} has labels outside 0-{LESION_LABEL} "
                         "(expected 0 air, 1 skin, 2 skull, 3 CSF, 4 gray, 5 white, 6 lesion).")
    return data.astype(np.uint8), img.affine


def split_lesion_label(lab):
    """Labels with an embedded lesion (6) -> (tissue labels 0-5, lesion mask or None).

    The tissue under the lesion is unknown, so lesion voxels take the label of the
    nearest non-lesion tissue voxel (used only for surfaces / display / QC).
    """
    les = lab == LESION_LABEL
    if not les.any():
        return lab, None
    _, idx = ndimage.distance_transform_edt(les, return_indices=True)
    tissue = lab[tuple(idx)]
    tissue[~les] = lab[~les]
    return tissue.astype(np.uint8), les


def labels_with_lesion(lab, lesion_mask, within=LESION_TISSUES):
    """Tissue labels with the lesion as a 6th tissue type (label 6), e.g. for light modelling.

    Only lesion voxels currently labelled as one of ``within`` (default CSF, gray,
    white) are relabelled; lesion voxels in air/skin/skull are left unchanged.
    Returns (labels, number of lesion voxels left unchanged).
    """
    out = lab.copy()
    if lesion_mask is None or not lesion_mask.any():
        return out, 0
    sel = lesion_mask & np.isin(lab, within)
    out[sel] = LESION_LABEL
    return out, int(lesion_mask.sum() - sel.sum())


def describe_source(paths):
    paths = _as_list(paths)
    if len(paths) > 1:
        return "SPM tissue probability maps"
    img = nib.load(paths[0])
    return "SPM tissue probability maps (4-D)" if img.ndim == 4 and img.shape[3] > 1 else "tissue label image"


def save_labels(lab, affine, path, lesion_mask=None):
    """Write an indexed uint8 label image; a lesion mask is folded in as label 6.

    Returns the number of lesion voxels that were left unchanged (in air/skin/skull).
    """
    lab, n_out = labels_with_lesion(lab, lesion_mask)
    img = nib.Nifti1Image(lab.astype(np.uint8), affine)
    img.header.set_data_dtype(np.uint8)
    img.header["descrip"] = LABEL_DESCRIP
    nib.save(img, str(path))
    return n_out


def load_lesion(path, shape, affine):
    """Binary lesion mask resampled (if needed) onto the segmentation grid."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.nan_to_num(data)
    if np.allclose(data, np.round(data)):
        mask = data > 0                       # 0/1, 0/255 or several lesion labels
    else:
        mask = data >= 0.5 * min(1.0, float(data.max()))   # fractional / probabilistic mask
    if _same_grid(img, shape, affine):
        return mask
    # lesion drawn on a different grid (e.g. raw vs resliced T1): resample into the segmentation grid
    M = np.linalg.inv(img.affine) @ affine                    # segmentation voxel -> lesion voxel
    ijk = np.indices(shape, dtype=np.float32).reshape(3, -1)
    src = M[:3, :3] @ ijk + M[:3, 3:4]
    return (ndimage.map_coordinates(mask.astype(np.float32), src, order=1, cval=0) >= 0.5).reshape(shape)


def surface_from_mask(mask, affine, sigma=1.0, target_faces=40000, smooth_iter=20, keep_largest=True):
    """Smoothed, decimated isosurface of a binary mask in world mm (nodes, 0-based faces)."""
    import pyvista as pv

    pad = np.pad(mask.astype(np.float32), 2)
    if sigma > 0:
        pad = ndimage.gaussian_filter(pad, sigma)
    verts, faces, _, _ = marching_cubes(pad, 0.5)
    verts = verts - 2
    mesh = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel()).clean()
    if keep_largest:
        mesh = mesh.connectivity("largest")
    mesh = mesh.extract_surface().triangulate()
    # clean() can collapse degenerate triangles into lines/points; keep proper triangles only
    f = mesh.regular_faces
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    mesh = pv.PolyData(np.asarray(mesh.points), np.c_[np.full(len(f), 3), f].ravel()).clean(point_merging=False)
    if mesh.n_cells > target_faces:
        mesh = mesh.decimate_pro(1 - target_faces / mesh.n_cells, preserve_topology=True)
    if smooth_iter:
        mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.1)
    nodes = np.asarray(mesh.points) @ affine[:3, :3].T + affine[:3, 3]
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    return nodes, faces


SCALP_CLOSE_MM = 3.0     # default scalp closing radius (GUI surface and MCX volume)


def _enclosed_air(air):
    """Air voxels not connected to the volume border."""
    comp, _ = ndimage.label(air)
    border = np.unique(np.concatenate([comp[[0, -1]].ravel(), comp[:, [0, -1]].ravel(),
                                       comp[:, :, [0, -1]].ravel()]))
    return air & ~np.isin(comp, border[border > 0])


def ball_closing(mask, radius_vox):
    """Morphological closing with a ball of ``radius_vox`` (distance-transform based, padded)."""
    pad = int(np.ceil(radius_vox)) + 1
    m = np.pad(mask, pad)
    dil = ndimage.distance_transform_edt(~m) <= radius_vox
    closed = ndimage.distance_transform_edt(dil) > radius_vox
    return closed[pad:-pad, pad:-pad, pad:-pad]


def clean_head_labels(lab, voxel_mm=1.0, close_mm=SCALP_CLOSE_MM):
    """Solid, closed head for display and light modelling.

    1. tissue islands not connected to the head (specks in the air) are removed;
    2. the head is closed with a ball of radius ``close_mm`` (holes, cracks and
       crevices narrower than ~2x``close_mm``; tissue is never removed);
    3. air enclosed by the head (sinuses, ear canals sealed by the closing,
       segmentation holes) is filled.
    Added voxels take the label of the nearest tissue voxel.  Returns (labels, info);
    ``labels > 0`` is the head mask used for the GUI scalp surface.
    """
    lab = np.asarray(lab).copy()
    comp, n = ndimage.label(lab > 0)
    islands = 0
    if n > 1:
        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        speck = (comp != sizes.argmax()) & (comp > 0)
        islands = int(speck.sum())
        lab[speck] = 0
    del comp
    air = lab == 0
    head = ~air
    closed = ball_closing(head, close_mm / voxel_mm) if close_mm > 0 else head
    solid = closed | _enclosed_air(~closed)
    fill = solid & air
    if fill.any():
        idx = ndimage.distance_transform_edt(air, return_distances=False, return_indices=True)
        lab[fill] = lab[tuple(i[fill] for i in idx)]
    n_enclosed = int((solid & ~closed).sum())
    info = {"tissue_islands_removed": islands, "surface_crevices_filled": int((closed & air).sum()),
            "enclosed_air_filled": n_enclosed, "closing_radius_mm": close_mm}
    return lab, info


def head_mask(lab, close_mm=SCALP_CLOSE_MM, voxel_mm=1.0):
    """Solid head mask (same cleaning as the MCX export volume, see ``clean_head_labels``)."""
    return clean_head_labels(lab, voxel_mm, close_mm)[0] > 0


@dataclass
class SubjectModel(dr.Atlas):
    affine: np.ndarray = None           # voxel -> world
    atlas_to_subject: np.ndarray = None  # 4x4, atlas mm -> subject world mm (before scalp projection)
    fit_rms_mm: float = np.nan
    seg_path: str = ""                   # display string (one file, or "; "-joined probability maps)
    scalp_surface: dr.Surface = None
    seg_files: list = None               # files the tissue labels came from
    seg_source: str = ""                 # "tissue label image" / "SPM tissue probability maps"
    labels: np.ndarray = None            # uint8 tissue labels on the segmentation grid
    lesion_path: str = ""
    lesion_mask: np.ndarray = None       # bool, on the segmentation grid (None = no lesion file)
    lesion_surface: tuple = None         # (nodes, faces) world mm
    lesion_tree: cKDTree = None          # lesion voxel centres, for distances
    head: np.ndarray = None              # bool solid head mask (clean_head_labels; same as the MCX volume)
    _voxel_scalp: tuple = None           # cached unsmoothed scalp surface (see voxel_scalp)

    @property
    def head_mask(self):
        if self.head is None:
            vox = float(np.mean(np.linalg.norm(self.affine[:3, :3], axis=0)))
            self.head = head_mask(self.labels, SCALP_CLOSE_MM, vox)
        return self.head

    def voxel_scalp(self):
        """Unsmoothed, undecimated boundary of the head mask (nodes, faces), i.e. the scalp as the
        MCX volume sees it (voxel faces, to within the marching-cubes half-voxel)."""
        if self._voxel_scalp is None:
            self._voxel_scalp = surface_from_mask(self.head_mask, self.affine, sigma=0,
                                                  target_faces=np.inf, smooth_iter=0)
        return self._voxel_scalp

    @property
    def has_lesion(self):
        return self.lesion_mask is not None and self.lesion_mask.any()

    def lesion_volume_ml(self):
        if self.lesion_mask is None:
            return 0.0
        return float(self.lesion_mask.sum() * abs(np.linalg.det(self.affine[:3, :3])) / 1000)

    def lesion_tissue_fraction(self):
        """Fraction of lesion voxels in each tissue label (QC that the mask is in the right space)."""
        if not self.has_lesion:
            return {}
        lab = self.labels[self.lesion_mask]
        return {LABELS[k]: float(np.mean(lab == k)) for k in LABELS}

    def forward_labels(self):
        """Tissue labels with the lesion as label 6 (0 air ... 5 white, 6 lesion) for light modelling."""
        return labels_with_lesion(self.labels, self.lesion_mask)[0]

    def lesion_distance(self, P):
        """Distance (mm) from points P [N,3] to the nearest lesion voxel (NaN without a lesion)."""
        P = np.atleast_2d(P)
        if not self.has_lesion:
            return np.full(len(P), np.nan)
        return self.lesion_tree.query(P)[0]


def attach_lesion(model: SubjectModel, path, progress=None) -> SubjectModel:
    """Load a lesion mask NIfTI, put it on the model's grid and build its surface (in place)."""
    say = progress or (lambda s: None)
    if path is None or str(path) == "":
        model.lesion_path, model.lesion_mask, model.lesion_surface, model.lesion_tree = "", None, None, None
        return model
    if isinstance(path, np.ndarray):              # mask already on the segmentation grid
        mask, path = path.astype(bool), f"label {LESION_LABEL} in {Path(model.seg_files[0]).name}"
    else:
        say(f"Loading lesion mask {Path(path).name}")
        mask = load_lesion(path, model.labels.shape, model.affine)
    model.lesion_path = str(path)
    model.lesion_mask = mask
    model.lesion_surface, model.lesion_tree = None, None
    if mask.any():
        ijk = np.argwhere(mask)
        model.lesion_tree = cKDTree(ijk @ model.affine[:3, :3].T + model.affine[:3, 3])
        say("Extracting lesion surface")
        try:
            model.lesion_surface = surface_from_mask(mask, model.affine, sigma=0.5, target_faces=20000,
                                                     smooth_iter=10, keep_largest=False)
        except (ValueError, RuntimeError):          # e.g. a lesion of only a few voxels
            model.lesion_surface = None
    return model


def _atlas_fit_points(atlas: dr.Atlas):
    """Atlas scalp points used to fit the subject scalp (drop the closed bottom cap)."""
    n, f = atlas.scalp
    s = dr.Surface(n, f, n_sub=2)
    zcut = min(atlas.fiducials["spmlpa"][2], atlas.fiducials["spmrpa"][2]) - 20
    keep = (s.samples[:, 2] > zcut) & (s.normals[:, 2] > -0.6)
    return s.samples[keep]


def _T_of(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix() @ np.diag(np.exp(x[6:9]))
    T[:3, 3] = x[3:6]
    return T


def _icp(pairs, x, n_iter=40, trim=0.85):
    """Trimmed point-to-plane ICP (9-dof) over several (points, target Surface, weight) pairs."""
    for _ in range(n_iter):
        sets = []
        for A, surf, w in pairs:
            P = dr.apply(_T_of(x), A)
            q, nrm = surf.closest(P)
            d = np.abs(np.sum((P - q) * nrm, 1)) + 0.2 * np.linalg.norm(P - q, axis=1)
            keep = d <= np.quantile(d, trim)
            sets.append((A[keep], q[keep], nrm[keep], w / np.sqrt(keep.sum())))

        def resid(y):
            T = _T_of(y)
            r = [s * np.sum((dr.apply(T, Ak) - qk) * nk, 1) for Ak, qk, nk, s in sets]
            return np.concatenate(r + [0.5 * y[6:9]])
        x_new = least_squares(resid, x, loss="soft_l1", f_scale=0.05).x
        done = np.max(np.abs(x_new - x)) < 1e-5
        x = x_new
        if done:
            break
    return x


def _trimmed_rms(T, A, surf, trim=0.85):
    d = np.abs(surf.signed_distance(dr.apply(T, A)))
    return float(np.sqrt(np.mean(np.sort(d)[: int(trim * len(d))] ** 2)))


def fit_atlas_to_subject(atlas: dr.Atlas, scalp: dr.Surface, brain: dr.Surface = None, trim=0.85):
    """9-dof (rigid + per-axis scale) atlas -> subject transform.

    The subject scalp from a T1 includes the face, neck and shoulders, so a
    scalp-only fit can trade pitch against those regions.  The brain is compact
    and well segmented, so it is fitted first (atlas cortex -> subject brain,
    multistart over pitch/yaw) and then refined jointly with the upper scalp.
    """
    A_scalp = _atlas_fit_points(atlas)
    if brain is None:
        pairs0 = [(A_scalp, scalp, 1.0)]
        a_c = A_scalp[A_scalp[:, 2] > A_scalp[:, 2].max() - 60].mean(0)
        s_c = scalp.samples[scalp.samples[:, 2] > scalp.samples[:, 2].max() - 60].mean(0)
    else:
        bn, bf = atlas.layers[-1]          # atlas cortex surface
        A_brain = dr.Surface(bn, bf, n_sub=1).samples
        if len(A_brain) > 6000:
            A_brain = A_brain[np.random.default_rng(0).choice(len(A_brain), 6000, replace=False)]
        pairs0 = [(A_brain, brain, 1.0)]
        a_c = A_brain.mean(0)
        # subject brain includes brainstem; use the cerebrum (top 120 mm) for the centroid
        S = brain.samples
        s_c = S[S[:, 2] > S[:, 2].max() - 110].mean(0)
        a_c = A_brain[A_brain[:, 2] > A_brain[:, 2].max() - 110].mean(0)

    best = None
    for pitch in (-30, -15, 0, 15, 30):
        for yaw in (-10, 0, 10):
            R0 = Rotation.from_euler("xz", [pitch, yaw], degrees=True)
            x = _icp(pairs0, np.r_[R0.as_rotvec(), s_c - R0.apply(a_c), 0, 0, 0], trim=trim)
            score = _trimmed_rms(_T_of(x), *pairs0[0][:2], trim=trim)
            if best is None or score < best[0]:
                best = (score, x)
    x = best[1]
    if brain is not None:
        x = _icp([(A_brain, brain, 1.0), (A_scalp, scalp, 1.0)], x, trim=trim)
    T = _T_of(x)
    return T, _trimmed_rms(T, A_scalp, scalp, trim)


def build_subject_model(seg_path, atlas: dr.Atlas, target_faces=40000, progress=None,
                        lesion_path=None) -> SubjectModel:
    """Build scalp/skull/brain surfaces and transfer the probe + landmarks.

    ``seg_path`` is a tissue label image (segmented.nii.gz) or a list of SPM12
    tissue probability maps (see ``load_segmentation``).  ``lesion_path`` is an
    optional lesion mask NIfTI; it takes precedence over a label-6 lesion
    embedded in the label image.
    """
    say = progress or (lambda s: None)
    files = _as_list(seg_path)
    say("Loading segmentation" if len(files) == 1 else f"Combining {len(files)} tissue probability maps")
    lab, aff = load_segmentation(files)
    lab, embedded_lesion = split_lesion_label(lab)
    vox = float(np.mean(np.linalg.norm(aff[:3, :3], axis=0)))
    say("Cleaning head (specks, holes, enclosed air)")
    head = head_mask(lab, SCALP_CLOSE_MM, vox)
    say("Extracting scalp surface")
    scalp_nf = surface_from_mask(head, aff, sigma=1.0, target_faces=target_faces)
    say("Extracting skull / brain surfaces")
    skull_nf = surface_from_mask(ndimage.binary_fill_holes(lab >= 2), aff, sigma=1.0, target_faces=target_faces // 2)
    brain_nf = surface_from_mask(ndimage.binary_fill_holes(lab >= 4), aff, sigma=0.7, target_faces=target_faces * 2, smooth_iter=10)
    scalp = dr.Surface(*scalp_nf, n_sub=3)
    brain = dr.Surface(*brain_nf, n_sub=1)

    say("Registering atlas head to subject (brain + scalp)")
    T, rms = fit_atlas_to_subject(atlas, scalp, brain)
    fid = {k: scalp.project(dr.apply(T, v[None]))[0] for k, v in atlas.fiducials.items()}
    probe = atlas.probe.copy()
    probe[["X", "Y", "Z"]] = scalp.project(dr.apply(T, probe[["X", "Y", "Z"]].values.astype(float)))
    model = SubjectModel(layers=[scalp_nf, skull_nf, brain_nf], fiducials=fid, probe=probe, link=atlas.link,
                         affine=aff, atlas_to_subject=T, fit_rms_mm=float(rms), seg_path="; ".join(files),
                         scalp_surface=scalp, seg_files=files, seg_source=describe_source(files), labels=lab,
                         head=head)
    if lesion_path:
        attach_lesion(model, lesion_path, progress)
    elif embedded_lesion is not None:
        attach_lesion(model, embedded_lesion, progress)
    return model
