"""
Repair and register Polhemus Fastrak (NIRS_Digitizer.mlapp) digitizer sessions.

Hemisphere-flip model
---------------------
NIRS_Digitizer collects head landmarks (LPA/RPA/Cz) with a fixed hemisphere
(``H1,0,0,1``), then switches to hemisphere *tracking* (``H1,0,0,0``) for the
optodes and extra head points.  When tracking fails the Fastrak solves for the
mirror solution of the stylus *sensor* position (s -> -s) while the orientation
(az, el, roll) stays correct.  Because the firmware adds the stylus-tip offset
``t`` (fixed in the stylus frame) after solving for the sensor position, the
reported value is ``reported = -s + R t`` while the truth is ``true = s + R t``:

    true = -reported + 2 R(az, el, roll) t

This is why a flip is not a simple reflection: the error depends on how the
stylus was held.  ``t`` is a hardware constant, estimated jointly from the
sessions in this folder (``estimate_stylus_offset``) as ~(6.97, -0.18, -0.08) cm,
i.e. a ~7 cm tip offset along the stylus axis.

Raw digitizer frame (cm): x = subject right, y = up, z = posterior, origin at
the transmitter (near the nasion).  Atlas frame (mm): ICBM152 MNI (RAS).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

STYLUS_TIP_OFFSET_CM = np.array([6.97, -0.18, -0.08])


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Atlas:
    layers: list            # list of (nodes [N,3] mm, faces [M,3] 0-based)
    fiducials: dict         # name -> xyz (mm)
    probe: pd.DataFrame     # template optodes registered to the atlas
    link: np.ndarray        # [K,2] source, detector (1-based)

    @property
    def scalp(self):
        return self.layers[0]

    @property
    def cortex(self):
        return self.layers[-1]


def load_atlas(path) -> Atlas:
    """Load the file written by export_atlas_for_python.m."""
    s = sio.loadmat(path, squeeze_me=True)
    layers = []
    k = 1
    while f"nodes{k}" in s:
        layers.append((np.asarray(s[f"nodes{k}"], float), np.asarray(s[f"faces{k}"], int) - 1))
        k += 1
    fid = {str(n): np.asarray(x, float) for n, x in zip(s["fid_name"], s["fid_xyz"])}
    probe = pd.DataFrame(s["opt_xyz"], columns=["X", "Y", "Z"])
    probe.insert(0, "Type", [str(x) for x in s["opt_type"]])
    probe.insert(0, "Name", [str(x) for x in s["opt_name"]])
    probe = probe[probe.Type.isin(["Source", "Detector"])].reset_index(drop=True)
    return Atlas(layers, fid, probe, np.atleast_2d(s["link"]).astype(int))


@dataclass
class Session:
    name: str
    head: np.ndarray | None     # [N,6] x,y,z (cm), az, el, roll (deg)
    optodes: np.ndarray | None
    extra: np.ndarray | None
    note: str = ""


def load_session(path) -> Session:
    path = Path(path)
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "PtsRaw" not in d:
        return Session(path.stem, None, None, None,
                       "No raw digitizer points (PtsRaw) were saved in this file; cannot be repaired.")
    r = d["PtsRaw"]
    get = lambda a: np.atleast_2d(np.asarray(a, float)) if np.size(a) else np.zeros((0, 6))
    return Session(path.stem, get(r.HeadPoints), get(r.OptodePoints), get(r.ExtraPoints))


# --------------------------------------------------------------------------
# Hemisphere-flip repair
# --------------------------------------------------------------------------
def stylus_rotations(P6):
    """Fastrak Euler angles (azimuth Z, elevation Y, roll X) -> rotation matrices."""
    return Rotation.from_euler("ZYX", P6[:, 3:6], degrees=True).as_matrix()


def unflip(P6, t=STYLUS_TIP_OFFSET_CM):
    """Recover true stylus-tip positions (cm) from hemisphere-flipped records."""
    return -P6[:, :3] + 2 * np.einsum("nij,j->ni", stylus_rotations(P6), t)


def _ellipsoid_resid(P, e):
    c, a = e[:3], np.abs(e[3:6])
    return (np.linalg.norm((P - c) / a, axis=1) - 1) * a.mean()


def cluster_landmarks(head_xyz, thr=2.0):
    """Average repeated clicks into LPA / RPA / Cz (same rule as NIRS_Digitizer)."""
    lab = fcluster(linkage(head_xyz, "single"), thr, "distance")
    ids, counts = np.unique(lab, return_counts=True)
    ids = ids[np.argsort(-counts)][:3]
    pts = np.array([head_xyz[lab == i].mean(0) for i in ids])
    return {"LPA": pts[pts[:, 0].argmin()], "RPA": pts[pts[:, 0].argmax()], "Cz": pts[pts[:, 1].argmax()]}


@dataclass
class RepairedSession:
    session: Session
    landmarks: dict                  # cm, raw frame
    optodes: np.ndarray              # corrected optode clicks (cm)
    extra: np.ndarray                # corrected extra head points (cm)
    optode_flipped: np.ndarray
    extra_flipped: np.ndarray
    optode_outlier: np.ndarray
    extra_outlier: np.ndarray
    ellipsoid: np.ndarray            # head ellipsoid fit (cm), raw frame
    optode_height: float             # cm above scalp
    rms: dict = field(default_factory=dict)


def repair_session(ses: Session, t=STYLUS_TIP_OFFSET_CM, flip_thr=1.5, outlier_thr=2.5) -> RepairedSession:
    """Detect and undo hemisphere flips point-by-point.

    A head ellipsoid is fit to the landmarks (fixed hemisphere -> trusted) and to
    every optode/extra point, letting each point take whichever interpretation
    (as recorded or un-flipped) is closer to the surface.  A point is declared
    flipped if the un-flipped version is closer and the recorded one is more
    than ``flip_thr`` cm off; points off by ``outlier_thr`` cm either way are
    dropped from the registration.
    """
    lm = cluster_landmarks(ses.head[:, :3])
    L = np.array(list(lm.values()))
    O_raw, E_raw = ses.optodes[:, :3], ses.extra[:, :3]
    O_flp = unflip(ses.optodes, t) if len(ses.optodes) else O_raw
    E_flp = unflip(ses.extra, t) if len(ses.extra) else E_raw

    def best(P0, P1, e):
        return np.minimum(np.abs(_ellipsoid_resid(P0, e)), np.abs(_ellipsoid_resid(P1, e)))

    def resid(x):
        e, h = x[:6], x[6]
        eo = e.copy(); eo[3:6] = np.abs(eo[3:6]) + h
        return np.concatenate([3 * _ellipsoid_resid(L, e), best(O_raw, O_flp, eo), best(E_raw, E_flp, e)])

    ear_mid = (lm["LPA"] + lm["RPA"]) / 2
    x0 = np.r_[ear_mid + 0.3 * (lm["Cz"] - ear_mid), 9.5, 9.5, 9.5, 1.5]
    x = least_squares(resid, x0, loss="soft_l1", f_scale=0.5).x
    e, h = x[:6].copy(), x[6]
    e[3:6] = np.abs(e[3:6])
    eo = e.copy(); eo[3:6] += h

    def classify(P0, P1, ee):
        r0, r1 = np.abs(_ellipsoid_resid(P0, ee)), np.abs(_ellipsoid_resid(P1, ee))
        flipped = (r1 < r0) & (r0 > flip_thr)
        P = np.where(flipped[:, None], P1, P0)
        return P, flipped, np.minimum(r0, r1) > outlier_thr, np.where(flipped, r1, r0)

    O, of, oo, ro = classify(O_raw, O_flp, eo)
    E, ef, eo_, re = classify(E_raw, E_flp, e)
    return RepairedSession(ses, lm, O, E, of, ef, oo, eo_, e, h,
                           {"landmarks": float(np.sqrt(np.mean(_ellipsoid_resid(L, e) ** 2))),
                            "optodes": float(np.sqrt(np.mean(ro[~oo] ** 2))) if (~oo).any() else np.nan,
                            "extra": float(np.sqrt(np.mean(re[~eo_] ** 2))) if (~eo_).any() else np.nan})


def estimate_stylus_offset(sessions, flipped_mask, t0=(7.0, 0.0, 0.0)):
    """Jointly estimate the stylus tip offset from several sessions.

    ``flipped_mask[i]`` says whether the extra points of session i are flipped.
    Each session gets its own head ellipsoid; optodes are assumed un-flipped and
    a common optode height above the scalp is estimated.
    """
    n = len(sessions)

    def unpack(x):
        return x[:3], x[3], [x[4 + 6 * i: 10 + 6 * i] for i in range(n)]

    def resid(x):
        t, h, El = unpack(x)
        out = []
        for s, e, fl in zip(sessions, El, flipped_mask):
            L = np.array(list(cluster_landmarks(s.head[:, :3]).values()))
            eo = e.copy(); eo[3:6] = np.abs(eo[3:6]) + h
            out += [3 * _ellipsoid_resid(L, e), _ellipsoid_resid(s.optodes[:, :3], eo),
                    _ellipsoid_resid(unflip(s.extra, t) if fl else s.extra[:, :3], e)]
        return np.concatenate(out)

    x0 = np.r_[t0, 1.5]
    for s in sessions:
        x0 = np.r_[x0, np.r_[s.head[:, :3], s.optodes[:, :3]].mean(0), 9, 9, 10]
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.5)
    t, h, _ = unpack(sol.x)
    return t, h


# --------------------------------------------------------------------------
# Surface helpers
# --------------------------------------------------------------------------
class Surface:
    """Densely sampled triangle mesh with outward normals for fast closest-point queries."""

    def __init__(self, nodes, faces, n_sub=7):
        self.nodes, self.faces = nodes, faces
        v0, v1, v2 = (nodes[faces[:, i]] for i in range(3))
        fn = np.cross(v1 - v0, v2 - v0)
        cen = (v0 + v1 + v2) / 3
        if np.mean(np.sum(fn * (cen - nodes.mean(0)), 1) > 0) < 0.5:
            fn = -fn
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        ij = [(i, j) for i in range(n_sub + 1) for j in range(n_sub + 1 - i)]
        b = np.array([[i, j, n_sub - i - j] for i, j in ij], float) / n_sub
        self.samples = (b[:, 0, None, None] * v0 + b[:, 1, None, None] * v1 + b[:, 2, None, None] * v2).reshape(-1, 3)
        self.normals = np.broadcast_to(fn, (len(b),) + fn.shape).reshape(-1, 3)
        self.tree = cKDTree(self.samples)

    def closest(self, P):
        _, i = self.tree.query(P)
        return self.samples[i], self.normals[i]

    def signed_distance(self, P):
        q, n = self.closest(P)
        return np.sum((P - q) * n, 1)

    def project(self, P):
        return self.closest(P)[0]


def _similarity(A, B):
    """Umeyama: B ~= s R A + t."""
    ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / np.sum((A - ma) ** 2)
    return s, R, mb - s * R @ ma


def _rigid(A, B):
    _, R, _ = _similarity(A, B)
    return R, B.mean(0) - R @ A.mean(0)


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------
# Head registration (subject raw frame -> atlas)
# --------------------------------------------------------------------------
def register_head(rep: RepairedSession, atlas: Atlas, scalp: Surface | None = None,
                  lm_weight=1.0, n_iter=30, allow_scaling=True):
    """Fit x_atlas = diag(scales) R p_mm + b (9 dof: rigid + per-axis scaling).

    With ``allow_scaling=False`` (subject's own MRI) the fit is rigid (6 dof).

    Initialised from LPA/RPA/Cz, refined with point-to-plane ICP of the extra
    head points (on the scalp) and the optode clicks (a fitted height above it)
    against the atlas scalp, with weak landmark priors.
    """
    scalp = scalp or Surface(*atlas.scalp)
    lm_sub = np.array([rep.landmarks[k] for k in ("LPA", "RPA", "Cz")]) * 10
    lm_atl = np.array([atlas.fiducials[k] for k in ("spmlpa", "spmrpa", "Cz")])
    if allow_scaling:
        s0, R0, b0 = _similarity(lm_sub, lm_atl)
    else:
        s0 = 1.0
        R0, b0 = _rigid(lm_sub, lm_atl)
    E = rep.extra[~rep.extra_outlier] * 10
    O = rep.optodes[~rep.optode_outlier] * 10

    def T_of(x):
        T = np.eye(4)
        T[:3, :3] = np.diag(np.exp(x[6:9])) @ Rotation.from_rotvec(x[:3]).as_matrix() @ R0 * s0
        T[:3, 3] = x[3:6] + b0
        return T

    x = np.zeros(10)
    x[9] = rep.optode_height * 10 * s0
    for _ in range(n_iter):
        T = T_of(x)
        qe, ne = scalp.closest(apply(T, E))
        qo, no = scalp.closest(apply(T, O))

        def resid(y):
            Ty = T_of(y)
            return np.concatenate([
                np.sum((apply(Ty, E) - qe) * ne, 1),
                np.sum((apply(Ty, O) - qo) * no, 1) - y[9],
                lm_weight * (apply(Ty, lm_sub) - lm_atl).ravel(),
                [3.0 * np.sum(y[6:9])],        # keep overall size set by the landmarks/scalp, not drift
                np.zeros(3) if allow_scaling else 100.0 * y[6:9],
            ])
        x_new = least_squares(resid, x, loss="soft_l1", f_scale=3.0).x
        if np.max(np.abs(x_new - x)) < 1e-5:
            x = x_new
            break
        x = x_new
    T = T_of(x)
    info = {"scales": np.exp(x[6:9]) * s0, "optode_height_mm": x[9],
            "extra_rms_mm": float(np.sqrt(np.mean(scalp.signed_distance(apply(T, E)) ** 2))) if len(E) else np.nan,
            "optode_rms_mm": float(np.sqrt(np.mean((scalp.signed_distance(apply(T, O)) - x[9]) ** 2))),
            "landmark_err_mm": dict(zip(("LPA", "RPA", "Cz"), np.linalg.norm(apply(T, lm_sub) - lm_atl, axis=1)))}
    return T, info


# --------------------------------------------------------------------------
# Probe registration
# --------------------------------------------------------------------------
def dedupe_clicks(P, thr=1.0):
    """Average repeated clicks on the same optode (cm). Returns centers and click counts."""
    if len(P) < 2:
        return P, np.ones(len(P), int)
    lab = fcluster(linkage(P, "single"), thr, "distance")
    order = pd.unique(lab)                         # keep click order
    return np.array([P[lab == i].mean(0) for i in order]), np.array([(lab == i).sum() for i in order])


def short_separation_pairs(probe: pd.DataFrame, max_dist=12.0):
    """Detectors sitting within ``max_dist`` mm of a source are short-separation detectors."""
    src = probe[probe.Type == "Source"]
    pairs = {}
    for i, r in probe[probe.Type == "Detector"].iterrows():
        d = np.linalg.norm(src[["X", "Y", "Z"]].values - r[["X", "Y", "Z"]].values.astype(float), axis=1)
        if d.min() < max_dist:
            pairs[i] = src.index[d.argmin()]
    return pairs


def _match_patch(Xt, C, gate, n_iter=30):
    """Rigid ICP + Hungarian matching of template patch Xt onto clicks C from many starts.

    Starts cover in-plane shifts (+/-30 mm) and rotations (+/-40 deg) about the
    patch normal so that a whole-patch offset of one optode row cannot trap the
    solution.  The start with the most gated matches (then lowest error) wins.
    """
    cen = Xt.mean(0)
    nrm = cen / np.linalg.norm(cen)
    u = np.cross(nrm, [0, 0, 1.0]); u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    best = None
    for ang in np.deg2rad(np.arange(-40, 41, 10)):
        R0 = Rotation.from_rotvec(nrm * ang).as_matrix()
        for du in (-30, -15, 0, 15, 30):
            for dv in (-30, -15, 0, 15, 30):
                R = R0
                b = cen + du * u + dv * v - R0 @ cen
                for it in range(n_iter):
                    D = cdist(Xt @ R.T + b, C)
                    r_, c_ = linear_sum_assignment(np.minimum(D, gate))
                    ok = D[r_, c_] < gate
                    r_, c_ = r_[ok], c_[ok]
                    if len(r_) < 3:
                        break
                    R_new, b_new = _rigid(Xt[r_], C[c_])
                    done = np.allclose(R_new, R, atol=1e-6) and np.allclose(b_new, b, atol=1e-4)
                    R, b = R_new, b_new
                    if done:
                        break
                D = cdist(Xt @ R.T + b, C)
                r_, c_ = linear_sum_assignment(np.minimum(D, gate))
                ok = D[r_, c_] < gate
                r_, c_ = r_[ok], c_[ok]
                score = (len(r_), -np.mean(D[r_, c_]) if len(r_) else 0)
                if best is None or score > best[0]:
                    best = (score, R, b, r_, c_, D[r_, c_])
    _, R, b, r_, c_, _ = best
    # Regularised affine refinement absorbs cap stretch so stretched-but-correct optodes pass the gate.
    A, a = R, b
    for _ in range(10):
        if len(r_) < 4:
            break
        Xh = np.c_[Xt[r_], np.ones(len(r_))]
        lam = 50.0 * len(r_) / 31
        reg = np.c_[np.sqrt(lam) * np.eye(3), np.zeros((3, 1))]
        M = np.linalg.lstsq(np.r_[Xh, reg], np.r_[C[c_], np.sqrt(lam) * R.T], rcond=None)[0].T
        A_new, a_new = M[:, :3], M[:, 3]
        D = cdist(Xt @ A_new.T + a_new, C)
        rn, cn = linear_sum_assignment(np.minimum(D, gate))
        ok = D[rn, cn] < gate
        if ok.sum() < len(r_):
            break
        A, a, r_, c_ = A_new, a_new, rn[ok], cn[ok]
    D = cdist(Xt @ A.T + a, C)
    return R, b, A, a, r_, c_, D[r_, c_]


def register_probe(rep: RepairedSession, T, atlas: Atlas, scalp: Surface | None = None,
                   gate=15.0, sigma=25.0):
    """Label the digitized optodes and build the session-specific probe on the atlas scalp.

    Each hemisphere patch of the template is rigidly aligned to the clicks
    (projected to the scalp) with multi-start Hungarian matching gated at
    ``gate`` mm.  Matched optodes take their measured position; unclicked
    optodes follow the aligned template plus a Gaussian-weighted (``sigma`` mm)
    interpolation of neighbouring displacements; short-separation detectors
    follow their source.
    """
    scalp = scalp or Surface(*atlas.scalp)
    tpl = atlas.probe.copy()
    ss = short_separation_pairs(tpl)
    long_idx = np.array([i for i in tpl.index if i not in ss])
    Xt_all = tpl[["X", "Y", "Z"]].values.astype(float)

    clicks_cm, counts = dedupe_clicks(rep.optodes[~rep.optode_outlier])
    C = scalp.project(apply(T, clicks_cm * 10))

    final = Xt_all.copy()
    match = pd.Series(-1, index=tpl.index)
    resid = pd.Series(np.nan, index=tpl.index)
    rigid = {}
    # hemisphere split along the ear-to-ear axis (head need not be centred at x=0, e.g. MRI scanner space)
    lpa, rpa = atlas.fiducials["spmlpa"], atlas.fiducials["spmrpa"]
    lr = (rpa - lpa) / np.linalg.norm(rpa - lpa)
    mid = (lpa + rpa) / 2
    side_of = lambda P: np.sign((np.atleast_2d(P) - mid) @ lr)
    for side in (-1, 1):
        ti = long_idx[side_of(Xt_all[long_idx]) == side]
        ci = np.where(side_of(C) == side)[0]
        if len(ci) == 0:
            continue
        Xt = Xt_all[ti]
        R, b, A, a, r_, c_, dd = _match_patch(Xt, C[ci], gate)
        rigid[side] = (R, b)
        Xa = Xt @ A.T + a
        match[ti[r_]] = ci[c_]
        resid[ti[r_]] = dd
        disp = C[ci][c_] - Xa[r_]
        for k, i in enumerate(ti):
            if i in ti[r_]:
                final[i] = C[ci][c_][list(ti[r_]).index(i)]
            else:
                w = np.exp(-np.sum((Xa[r_] - Xa[k]) ** 2, 1) / (2 * sigma ** 2))
                final[i] = Xa[k] + (w @ disp) / max(w.sum(), 1e-9) * min(1.0, w.sum())
    for d, s in ss.items():
        R = rigid.get(side_of(Xt_all[s])[0], (np.eye(3), 0))[0]
        final[d] = final[s] + R @ (Xt_all[d] - Xt_all[s])
    final = scalp.project(final)

    out = tpl[["Name", "Type"]].copy()
    out[["X", "Y", "Z"]] = final
    out["template_X"], out["template_Y"], out["template_Z"] = Xt_all.T
    out["shift_from_template_mm"] = np.linalg.norm(final - Xt_all, axis=1)
    out["status"] = np.where(match >= 0, "digitized", "interpolated")
    out.loc[list(ss), "status"] = "short-sep (follows source)"
    out["click_index"] = match.values
    out["match_residual_mm"] = resid.values
    clicks = pd.DataFrame(C, columns=["X", "Y", "Z"])
    clicks["n_clicks"] = counts
    clicks["label"] = ""
    for i, j in match[match >= 0].items():
        clicks.loc[j, "label"] = tpl.Name[i]
    return out, clicks


def click_order_consistency(clicks: pd.DataFrame, probe: pd.DataFrame, max_step=50.0):
    """Fraction of consecutive labelled clicks that land on neighbouring template optodes.

    Operators click optodes in a path through the cap, so a correct labelling
    gives consecutive clicks on template neighbours (independent sanity check).
    """
    xyz = probe.set_index("Name")[["template_X", "template_Y", "template_Z"]]
    lab = [l for l in clicks.label if l]
    steps = [np.linalg.norm(xyz.loc[a].values - xyz.loc[b].values) for a, b in zip(lab[:-1], lab[1:])]
    steps = [d for d in steps if d < 100]          # ignore the hop between hemispheres
    return float(np.mean(np.array(steps) < max_step)) if steps else np.nan


def project_to_cortex(probe_df, atlas: Atlas):
    """Closest cortex point under each optode (useful for later MRI-based checks)."""
    cortex = Surface(*atlas.cortex, n_sub=2)
    return cortex.project(probe_df[["X", "Y", "Z"]].values)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    repaired: RepairedSession | None = None
    T: np.ndarray | None = None
    head_info: dict | None = None
    probe: pd.DataFrame | None = None
    clicks: pd.DataFrame | None = None
    note: str = ""


def process(path, atlas: Atlas, scalp: Surface | None = None, t=STYLUS_TIP_OFFSET_CM,
            allow_scaling=True) -> Result:
    ses = load_session(path)
    if ses.head is None:
        return Result(ses.name, note=ses.note)
    scalp = scalp or Surface(*atlas.scalp)
    rep = repair_session(ses, t)
    T, info = register_head(rep, atlas, scalp, allow_scaling=allow_scaling)
    probe, clicks = register_probe(rep, T, atlas, scalp)
    return Result(ses.name, rep, T, info, probe, clicks)


def save_result(res: Result, outdir):
    """Write <name>_registered_probe.csv and a MATLAB-readable .mat."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    if res.probe is None:
        return
    res.probe.to_csv(outdir / f"{res.name}_registered_probe.csv", index=False)
    rep = res.repaired
    sio.savemat(outdir / f"{res.name}_registered.mat", {
        "optode_name": np.array(res.probe.Name.tolist(), dtype=object),
        "optode_type": np.array(res.probe.Type.tolist(), dtype=object),
        "optode_xyz_mm": res.probe[["X", "Y", "Z"]].values,
        "optode_status": np.array(res.probe.status.tolist(), dtype=object),
        # Column-vector convention: x_atlas = T * [p_raw_cm*10; 1]  (MATLAB app uses row-vector p*T, i.e. T')
        "T_rawmm_to_atlas": res.T,
        "corrected_OptodePoints_cm": rep.optodes,
        "corrected_ExtraPoints_cm": rep.extra,
        "optode_flipped": rep.optode_flipped,
        "extra_flipped": rep.extra_flipped,
        "stylus_tip_offset_cm": STYLUS_TIP_OFFSET_CM,
        "landmarks_cm": np.array([rep.landmarks[k] for k in ("LPA", "RPA", "Cz")]),
    })
