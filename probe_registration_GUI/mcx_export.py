"""
Export registered probes + the subject's segmented head for Monte Carlo
photon simulations with MCX (CUDA, https://mcx.space) or MCX-CL (OpenCL).

The export folder is self-contained and portable:

    head/head_labels.bin          uint8 tissue labels (MCX volume, x fastest)
    head/head_labels.nii.gz       the same labels with the voxel->world affine
    optical_properties.csv        mua/mus/g/n per tissue label and wavelength
    settings.json                 every export setting, affine, label/tissue map
    simulations.txt               list of simulations run by the scripts
    simulations.csv               the same list with optode/detector details
    sessions/<session>/optodes.csv, channels.csv
    sessions/<session>/<wl>nm/<optode>.json   one MCX input per launch optode
    run_analysis.sh / .command    macOS & Linux (the .command opens in Terminal on double-click)
    run_analysis.bat              Windows
    README.txt

Each registered probe (session) is a separate set of simulations on the same
head volume: one simulation per wavelength and launch optode.  Sources are
always simulated (detected photons with partial path lengths per tissue at all
detectors + the fluence volume); with ``simulate="both"`` the detectors are
also simulated as adjoint sources (fluence only is needed, but photons
reaching the sources are recorded too), so channel sensitivity (Jacobian)
maps can be formed as phi_source * phi_detector.

Tissue optical properties come from the NIRS-toolbox (nirs.media.tissues),
tabulated by export_tissue_optics.m into nirs_tissue_optics.json:
    1 skin -> skin, 2 skull -> bone, 3 CSF -> water,
    4 gray / 5 white / 6 lesion -> brain (SpectralProp, SO2 and HbT adjustable)
The skin and bone tables hold μs′; they are converted to MCX's μs = μs′/(1-g).
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import subject_headmodel as shm

HERE = Path(__file__).resolve().parent
OPTICS_FILE = HERE / "assets" / "nirs_tissue_optics.json"

# tissue label -> nirs.media.tissues model
TISSUE_MODEL = {0: "air", 1: "skin", 2: "bone", 3: "water", 4: "brain", 5: "brain", 6: "brain"}
# nirs.media.tissues.skin/bone tabulate the reduced scattering coefficient μs′ (1/mm) in their
# 'mus' field (the toolbox's diffusion models use kappa = 1/(3(mua+mus)); skin.m cites Jacques'
# μs′ fit).  MCX needs μs, so these are divided by (1-g).  water's mus (0.01 /mm) is used as given.
REDUCED_SCATTERING_TABLES = ("skin", "bone")
LABEL_NAMES = shm.ALL_LABELS
MCX_SEED = 1648335518          # MCX default RNG seed (simulation k uses seed + k)
MCX_DEFAULT_BINARIES = [       # where the run scripts also look for MCX (besides PATH)
    "$HOME/Desktop/MCXStudio/MCXSuite",
    "$HOME/MCXStudio/MCXSuite",
    "$HOME/Downloads/MCXStudio/MCXSuite",
    "/Applications/MCXStudio/MCXSuite",
    "/Applications/MCXStudio.app/Contents/MacOS/MCXSuite",
    "/opt/MCXStudio/MCXSuite",
    "/usr/local/MCXStudio/MCXSuite",
]


# --------------------------------------------------------------------------
# Optical properties (NIRS-toolbox nirs.media.tissues)
# --------------------------------------------------------------------------
class TissueOptics:
    """mua/mus (1/mm), g and refractive index from the NIRS-toolbox tables."""

    def __init__(self, path=OPTICS_FILE):
        with open(path) as f:
            d = json.load(f)
        self.source = d.get("source", str(path))
        self.lam = np.asarray(d["lambda_nm"], float)
        self.tab = {k: {"mua": np.asarray(d[k]["mua"], float), "mus": np.asarray(d[k]["mus"], float),
                        "g": float(d[k]["g"]), "ri": float(d[k]["ri"])} for k in ("skin", "bone", "water")}
        self.spectra = {k: np.asarray(v, float) for k, v in d["spectra"].items()}
        self.brain = dict(d["brain_defaults"])
        self.check = d.get("check")

    @property
    def wavelength_range(self):
        lo = max(self.lam[0], self.brain["lambda_min"])
        hi = min(self.lam[-1], self.brain["lambda_max"])
        return float(lo), float(hi)

    def _interp(self, y, lam):
        return float(np.interp(lam, self.lam, y))

    def brain_props(self, lam, so2=None, hbt=None):
        """nirs.media.SpectralProp: mua from HbO/HbR/water/fat, mus = a*(lambda/500)^-b/(1-g)."""
        b = self.brain
        so2 = b["so2"] if so2 is None else so2
        hbt = b["hbt"] if hbt is None else hbt
        s = {k: self._interp(v, lam) for k, v in self.spectra.items()}
        mua = (s["ext_hbo"] * so2 * hbt * 1e-6 + s["ext_hbr"] * (1 - so2) * hbt * 1e-6
               + s["mua_water"] * b["water"] + s["mua_fat"] * b["lipid"])
        mus = b["a"] * (lam / 500.0) ** -b["b"] / (1 - b["g"])
        return mua, mus, b["g"], b["ri"]

    def props(self, model, lam, so2=None, hbt=None):
        lo, hi = self.wavelength_range
        if not lo < lam < hi:
            raise ValueError(f"wavelength {lam} nm is outside the tabulated range ({lo:.0f}-{hi:.0f} nm)")
        if model == "air":
            return 0.0, 0.0, 1.0, 1.0
        if model == "brain":
            return self.brain_props(lam, so2, hbt)
        t = self.tab[model]
        mus = self._interp(t["mus"], lam)
        if model in REDUCED_SCATTERING_TABLES:
            mus /= 1 - t["g"]
        return self._interp(t["mua"], lam), mus, t["g"], t["ri"]

    def table(self, wavelengths, so2=None, hbt=None, labels=TISSUE_MODEL):
        """Long table: wavelength_nm, label, tissue, model, mua, mus, g, n, musp (1/mm); mus is MCX's (unreduced) μs."""
        rows = []
        for lam in wavelengths:
            for lab, model in labels.items():
                mua, mus, g, n = self.props(model, float(lam), so2, hbt)
                rows.append(dict(wavelength_nm=float(lam), label=lab, tissue=LABEL_NAMES[lab], model=model,
                                 mua=mua, mus=mus, g=g, n=n, musp=mus * (1 - g)))
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class McxSettings:
    wavelengths: list = field(default_factory=lambda: [780.0, 850.0])   # NIRx
    photons: float = 1e8               # per simulation (MCX example/benchmark value)
    simulate: str = "both"             # "sources" or "both" (sources + detectors as adjoint sources)
    save_fluence: bool = True          # -S 1
    output_type: str = "x"             # -O: x flux (MCX default), f fluence, e energy
    output_format: str = "jnii"        # -F: jnii (MCX default), bnii, nii, mc2
    save_detected: bool = True         # -d 1
    detected_fields: str = "DP"        # -w: D detector id, P partial path lengths (MCX default)
    max_detected: int = 10_000_000     # -H (MCX default 1e6 overflows for short-separation pairs at 1e8 photons)
    reflect: bool = True               # -b 1 (MCX default): Fresnel reflection at the boundary
    normalize: bool = True             # -U 1 (MCX default)
    t0: float = 0.0                    # s
    t1: float = 5e-9                   # s
    dt: float = 5e-9                   # s (one gate = continuous wave)
    seed: int = MCX_SEED               # -E (simulation k uses seed + k)
    detector_radius_mm: float = 2.0
    close_scalp_mm: float = shm.SCALP_CLOSE_MM  # ball closing of the scalp (same as the GUI scalp surface)
    margin_mm: float = 10.0            # pad the grid with air so the head is at least this far from its faces
    brain_so2: float = 0.7
    brain_hbt_uM: float = 50.0
    engine: str = "auto"               # run-script default: auto, mcx (CUDA) or mcxcl (OpenCL)
    gpu: str = ""                      # passed as -G if set (e.g. "1"); empty = MCX picks

    def n_gates(self):
        return max(1, int(round((self.t1 - self.t0) / self.dt)))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def voxel_size(affine):
    vs = np.linalg.norm(affine[:3, :3], axis=0)
    if np.ptp(vs) > 0.01 * vs.mean():
        raise ValueError(f"MCX needs isotropic voxels; the segmentation has voxel size {np.round(vs, 3)} mm. "
                         "Reslice the T1/segmentation to isotropic voxels first.")
    return float(vs.mean())


def world_to_mcx(affine, P):
    """World mm -> MCX grid coordinates (OriginType 1: voxel i spans [i, i+1))."""
    P = np.atleast_2d(P)
    ijk = (P - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T
    return ijk + 0.5


def world_dir_to_mcx(affine, D):
    D = np.atleast_2d(D) @ np.linalg.inv(affine[:3, :3]).T
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def snap_to_voxel_surface(labels, affine, P, N, outside=10.0, inside=10.0, step=0.1, solid_mm=2.0):
    """Move each point along its outward normal to the voxelised head boundary.

    Returns the last air point before the first tissue voxel that is followed
    by at least ``solid_mm`` of continuous tissue, coming from ``outside`` mm
    along the normal, and the distance moved.
    """
    t = np.arange(outside, -inside - step / 2, -step)
    out = P.copy()
    moved = np.zeros(len(P))
    Ainv = np.linalg.inv(affine)
    shape = np.array(labels.shape)
    for i, (p, n) in enumerate(zip(P, N)):
        Q = p + t[:, None] * n
        ijk = np.rint(Q @ Ainv[:3, :3].T + Ainv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        lab = np.zeros(len(t), np.uint8)
        lab[ok] = labels[tuple(ijk[ok].T)]
        # first tissue sample followed by >= solid_mm of continuous tissue (skips thin skin ridges)
        tissue = lab > 0
        m = max(1, int(round(solid_mm / step)))
        run = np.convolve(tissue.astype(int), np.ones(m, int), "full")[m - 1:]
        hit = np.flatnonzero(tissue & (run >= m))
        if len(hit) and hit[0] > 0:
            out[i] = Q[hit[0] - 1]
            moved[i] = t[hit[0] - 1]
    return out, moved


def prepare_mcx_volume(labels, voxel_mm=1.0, close_mm=shm.SCALP_CLOSE_MM, margin_mm=10.0):
    """Make the label volume safe for MCX, which stops photons entering label 0 or leaving the grid.

    The head is cleaned exactly as the GUI scalp surface (``subject_headmodel.clean_head_labels``):
    tissue specks removed, scalp closed with a ball of radius ``close_mm`` and all air enclosed by
    the head filled with the nearest tissue.  The grid is then padded with air so the head is
    >= ``margin_mm`` from every face (where it is not already cut by the MRI field of view).
    Returns (labels, pad_lo (voxels added before index 0 on each axis), info dict).
    """
    lab, info = shm.clean_head_labels(labels, voxel_mm, close_mm)
    m = int(np.ceil(margin_mm / voxel_mm))
    tissue = np.nonzero(lab.any(axis=(1, 2)))[0], np.nonzero(lab.any(axis=(0, 2)))[0], np.nonzero(lab.any(axis=(0, 1)))[0]
    pad = [(max(0, m - int(t[0])), max(0, m - (lab.shape[a] - 1 - int(t[-1])))) for a, t in enumerate(tissue)]
    if any(p for pp in pad for p in pp):
        lab = np.pad(lab, pad)
    info["padding_voxels"] = [list(p) for p in pad]
    return lab, np.array([p[0] for p in pad]), info


def padded_affine(affine, pad_lo):
    """Affine of a grid with ``pad_lo`` voxels prepended on each axis."""
    A = affine.copy()
    A[:3, 3] = affine[:3, 3] - affine[:3, :3] @ np.asarray(pad_lo, float)
    return A


def optode_table(model, probe: pd.DataFrame, labels, affine=None):
    """Registered optodes with outward normals, positions snapped to the voxel surface and MCX coordinates.

    ``affine`` is the voxel->world affine of ``labels`` (default: the model's).
    """
    affine = model.affine if affine is None else affine
    P = probe[["X", "Y", "Z"]].values.astype(float)
    # per-triangle normals averaged over a few mm for a stable launch direction
    surf = model.scalp_surface
    N = np.array([surf.normals[surf.tree.query_ball_point(p, 4.0) or [surf.tree.query(p)[1]]].mean(0) for p in P])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B, moved = snap_to_voxel_surface(labels, affine, P, N)
    t = probe[["Name", "Type"]].copy().reset_index(drop=True)
    t["index"] = t.groupby("Type").cumcount() + 1
    t["X"], t["Y"], t["Z"] = P.T
    t["normal_x"], t["normal_y"], t["normal_z"] = N.T
    t["surf_X"], t["surf_Y"], t["surf_Z"] = B.T
    t["snap_mm"] = moved
    G = world_to_mcx(affine, B)
    D = world_dir_to_mcx(affine, -N)
    t["mcx_x"], t["mcx_y"], t["mcx_z"] = G.T
    t["mcx_dir_x"], t["mcx_dir_y"], t["mcx_dir_z"] = D.T
    if "status" in probe:
        t["status"] = probe["status"].values
    if model.has_lesion:
        t["lesion_dist_mm"] = model.lesion_distance(P)
    return t


def channel_table(link, opt: pd.DataFrame):
    src = opt[opt.Type == "Source"].set_index("index")
    det = opt[opt.Type == "Detector"].set_index("index")
    rows = []
    for s, d in np.atleast_2d(link):
        a, b = src.loc[s], det.loc[d]
        rows.append(dict(source=int(s), detector=int(d), source_name=a.Name, detector_name=b.Name,
                         distance_mm=float(np.linalg.norm(a[["X", "Y", "Z"]].values.astype(float)
                                                          - b[["X", "Y", "Z"]].values.astype(float)))))
    return pd.DataFrame(rows)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "session"


# --------------------------------------------------------------------------
# MCX input
# --------------------------------------------------------------------------
DETFLAG_BITS = {"D": 1, "S": 2, "P": 4, "M": 8, "X": 16, "V": 32, "W": 64}


def detflag_mask(flags):
    bad = set(flags.upper()) - set(DETFLAG_BITS)
    if bad:
        raise ValueError(f"unknown detected-photon field(s) {sorted(bad)}; use letters from DSPMXVW")
    return sum(DETFLAG_BITS[c] for c in set(flags.upper()))


def media_list(optics_wl: pd.DataFrame, n_labels):
    m = optics_wl.set_index("label").sort_index()
    out = []
    for lab in range(n_labels):
        r = m.loc[lab]
        out.append({"mua": round(float(r.mua), 8), "mus": round(float(r.mus), 6),
                    "g": round(float(r.g), 6), "n": round(float(r.n), 6)})
    return out


def mcx_config(sim_id, volume_rel, dim, unit_mm, media, src_row, det_rows, settings: McxSettings, seed):
    det = [{"Pos": [round(float(v), 4) for v in r[["mcx_x", "mcx_y", "mcx_z"]]],
            "R": round(settings.detector_radius_mm / unit_mm, 4)} for _, r in det_rows.iterrows()]
    cfg = {
        "Session": {
            "ID": sim_id,
            "Photons": int(settings.photons),
            "RNGSeed": int(seed),
            "DoMismatch": bool(settings.reflect),
            "DoNormalize": bool(settings.normalize),
            "DoSaveVolume": bool(settings.save_fluence),
            "DoPartialPath": bool(settings.save_detected and len(det) > 0),
            "SaveDataMask": detflag_mask(settings.detected_fields),
            "DoAutoThread": True,
            "OutputType": settings.output_type,
            "OutputFormat": settings.output_format,
        },
        "Forward": {"T0": settings.t0, "T1": settings.t1, "Dt": settings.dt},
        "Domain": {
            "VolumeFile": volume_rel,
            "Dim": [int(x) for x in dim],
            "OriginType": 1,
            "LengthUnit": unit_mm,
            "MediaFormat": "byte",
            "Media": media,
        },
        "Optode": {
            "Source": {"Type": "pencil",
                       "Pos": [round(float(v), 4) for v in src_row[["mcx_x", "mcx_y", "mcx_z"]]],
                       "Dir": [round(float(v), 6) for v in src_row[["mcx_dir_x", "mcx_dir_y", "mcx_dir_z"]]]},
        },
    }
    if det:
        cfg["Optode"]["Detector"] = det
    return cfg


def output_files(sim_id, settings: McxSettings):
    ext = {"jnii": ".jnii", "bnii": ".bnii", "nii": ".nii", "mc2": ".mc2", "hdr": ".hdr"}[settings.output_format]
    out = []
    if settings.save_fluence:
        out.append(sim_id + ext)
    if settings.save_detected:
        out.append(sim_id + ("_detp.jdat" if settings.output_format in ("jnii", "bnii") else ".mch"))
    return out


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def plan(sessions: dict, settings: McxSettings):
    """(n simulations, total photons) for a dict of session name -> probe table."""
    n = 0
    for probe in sessions.values():
        k = int((probe.Type == "Source").sum())
        if settings.simulate == "both":
            k += int((probe.Type == "Detector").sum())
        n += k * len(settings.wavelengths)
    return n, n * settings.photons


def export_mcx(folder, model, sessions: dict, settings: McxSettings, optics: pd.DataFrame | None = None,
               mcx_hint=None, progress=None):
    """Write the MCX simulation folder.

    ``sessions`` maps a session name to its registered probe (Name, Type, X, Y, Z
    in the subject's world mm, as produced by digitizer_registration).
    ``optics`` is a long table (wavelength_nm, label, mua, mus, g, n); by default
    it is computed from the NIRS-toolbox tissues.  Returns a summary dict.
    """
    say = progress or (lambda s: None)
    folder = Path(folder)
    if not sessions:
        raise ValueError("No registered sessions to export.")
    if settings.save_fluence is False and settings.save_detected is False:
        raise ValueError("Nothing to save: enable the fluence volume and/or detected photons.")
    if settings.simulate not in ("sources", "both"):
        raise ValueError("simulate must be 'sources' or 'both'")
    wls = [float(w) for w in settings.wavelengths]
    if optics is None:
        optics = TissueOptics().table(wls, settings.brain_so2, settings.brain_hbt_uM)
    missing = [w for w in wls if not np.isclose(optics.wavelength_nm, w).any()]
    if missing:
        raise ValueError(f"No optical properties for wavelength(s) {missing}")

    unit = voxel_size(model.affine)
    labels = model.forward_labels().astype(np.uint8)
    say("Cleaning the head volume (specks, enclosed air, scalp closing, padding)")
    labels, pad_lo, vol_fix = prepare_mcx_volume(labels, unit, settings.close_scalp_mm, settings.margin_mm)
    affine = padded_affine(model.affine, pad_lo)
    n_labels = max(int(labels.max()) + 1, max(TISSUE_MODEL) + 1)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "head").mkdir(exist_ok=True)
    say("Writing head volume")
    labels.ravel(order="F").tofile(folder / "head" / "head_labels.bin")
    shm.save_labels(labels, affine, folder / "head" / "head_labels.nii.gz")
    optics.to_csv(folder / "optical_properties.csv", index=False)

    sim_rows, sim_lines = [], []
    k = 0
    for sname, probe in sessions.items():
        sdir_rel = Path("sessions") / safe_name(sname)
        sdir = folder / sdir_rel
        sdir.mkdir(parents=True, exist_ok=True)
        say(f"Writing simulations for {sname}")
        opt = optode_table(model, probe, labels, affine)
        opt.to_csv(sdir / "optodes.csv", index=False)
        if model.link is not None and len(model.link):
            channel_table(model.link, opt).to_csv(sdir / "channels.csv", index=False)
        src = opt[opt.Type == "Source"]
        det = opt[opt.Type == "Detector"]
        launches = [(r, det) for _, r in src.iterrows()]
        if settings.simulate == "both":
            launches += [(r, src) for _, r in det.iterrows()]
        for wl in wls:
            wdir_rel = sdir_rel / f"{wl:g}nm"
            (folder / wdir_rel).mkdir(exist_ok=True)
            media = media_list(optics[np.isclose(optics.wavelength_nm, wl)], n_labels)
            for r, dets in launches:
                sim_id = safe_name(r.Name)
                dets_used = dets if settings.save_detected else dets.iloc[:0]
                cfg = mcx_config(sim_id, "../../../head/head_labels.bin", labels.shape, unit, media, r,
                                 dets_used, settings, settings.seed + k)
                with open(folder / wdir_rel / f"{sim_id}.json", "w") as f:
                    json.dump(cfg, f, indent=1)
                (folder / wdir_rel / f"{sim_id}.done").unlink(missing_ok=True)  # new inputs -> rerun
                sim_lines.append(f"{wdir_rel.as_posix()}|{sim_id}")
                sim_rows.append(dict(session=sname, wavelength_nm=wl, launch_optode=r.Name, launch_type=r.Type,
                                     folder=wdir_rel.as_posix(), json=f"{sim_id}.json",
                                     outputs=";".join(output_files(sim_id, settings)),
                                     detectors=";".join(dets_used.Name), rng_seed=settings.seed + k,
                                     photons=int(settings.photons)))
                k += 1

    (folder / "simulations.txt").write_text("\n".join(sim_lines) + "\n")
    pd.DataFrame(sim_rows).to_csv(folder / "simulations.csv", index=False)

    info = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "mcx_settings": asdict(settings),
        "n_simulations": len(sim_lines),
        "sessions": list(sessions),
        "volume": {"file": "head/head_labels.bin", "nifti": "head/head_labels.nii.gz", "dim": list(labels.shape),
                   "dtype": "uint8", "order": "x fastest (Fortran/MATLAB order)", "voxel_mm": unit,
                   "voxel_to_world_affine": affine.tolist(),
                   "mcx_grid_to_world": "world = affine @ [mcx_x-0.5, mcx_y-0.5, mcx_z-0.5, 1]",
                   "cleanup": vol_fix},
        "labels": {str(k): v for k, v in LABEL_NAMES.items()},
        "tissue_model_per_label": {str(k): v for k, v in TISSUE_MODEL.items()},
        "optical_properties_source": TissueOptics().source if OPTICS_FILE.exists() else "",
        "segmentation": model.seg_path,
        "lesion_mask": model.lesion_path,
        "notes": ["Media[0] is air/background; Media[k] is tissue label k.",
                  "Lesion (label 6) has the same optical properties as brain.",
                  "Source simulations record photons at all detectors; detector (adjoint) "
                  "simulations record photons at all sources.",
                  "Detected photon detector IDs follow the order in simulations.csv 'detectors'."],
    }
    with open(folder / "settings.json", "w") as f:
        json.dump(info, f, indent=1)

    write_scripts(folder, settings, mcx_hint)
    write_readme(folder, settings, info)
    return {"folder": str(folder), "n_simulations": len(sim_lines), "total_photons": len(sim_lines) * settings.photons}


# --------------------------------------------------------------------------
# Run scripts
# --------------------------------------------------------------------------
SH_TEMPLATE = r'''#!/bin/bash
# Run all MCX / MCX-CL simulations in this folder (macOS / Linux).
#
#   ./run_analysis.sh            auto-detect MCX (CUDA) or MCX-CL (OpenCL)
#   ./run_analysis.sh mcxcl      force MCX-CL;  ./run_analysis.sh mcx   force MCX
#   MCX_BIN=/path/to/mcxcl ./run_analysis.sh          use a specific executable
#   MCX_ARGS="-G 1" ./run_analysis.sh                  extra MCX command-line options
#
# Finished simulations are skipped (<name>.done), so the script can be stopped
# and restarted.  Delete the .done files (or run with FORCE=1) to rerun.

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
PREF="${1:-${MCX_ENGINE:-__ENGINE__}}"
GPU_ARGS="__GPU_ARGS__"
EXTRA_ARGS="${MCX_ARGS:-}"
FORCE="${FORCE:-0}"

SUITES=(
__SUITES__
)

try_bin() {   # $1 executable, $2 engine name
    [ -x "$1" ] || return 1
    "$1" -L >/dev/null 2>&1 || return 1
    BIN="$1"; ENGINE="$2"; return 0
}

find_engine() {   # $1 = mcx | mcxcl
    local name="$1" p
    p="$(command -v "$name" 2>/dev/null)" && try_bin "$p" "$name" && return 0
    for s in "${SUITES[@]}"; do
        s="${s/\$HOME/$HOME}"
        try_bin "$s/$name/bin/$name" "$name" && return 0
    done
    return 1
}

BIN=""; ENGINE=""
if [ -n "${MCX_BIN:-}" ]; then
    case "$(basename "$MCX_BIN")" in mcxcl*) e=mcxcl ;; *) e=mcx ;; esac
    try_bin "$MCX_BIN" "$e" || { echo "MCX_BIN=$MCX_BIN does not run ('$MCX_BIN -L' failed)."; exit 1; }
else
    case "$PREF" in
        mcx)   find_engine mcx ;;
        mcxcl) find_engine mcxcl ;;
        *)     find_engine mcx || find_engine mcxcl ;;   # CUDA first (NVIDIA), then OpenCL (Apple/AMD/Intel/NVIDIA)
    esac
fi
if [ -z "$BIN" ]; then
    cat <<EOF
ERROR: no working MCX installation found.
  Looked for '$PREF' on the PATH and in:
$(printf '    %s\n' "${SUITES[@]}")
  Install MCX Studio from https://mcx.space (it contains mcx, mcxcl and mmc), then either
  put its MCXSuite/mcx/bin or MCXSuite/mcxcl/bin folder on the PATH or run
      MCX_BIN=/path/to/MCXSuite/mcxcl/bin/mcxcl ./run_analysis.sh
  Notes: mcx (CUDA) needs an NVIDIA GPU and driver; on a Mac use mcxcl (OpenCL).
EOF
    exit 1
fi

ENGINE_ARGS="$GPU_ARGS"
if [ "$ENGINE" = "mcxcl" ] && ! "$BIN" -L 2>/dev/null | grep -q "GPU device"; then
    echo "WARNING: MCX-CL found no GPU - running on the CPU (much slower)."
    [ "$(uname)" = "Darwin" ] && ENGINE_ARGS="-A 0 -t 1024 -T 1"   # Apple CPU OpenCL needs a work-group size of 1
fi
echo "Using $ENGINE: $BIN"
"$BIN" -L 2>/dev/null | grep -E "Device [0-9]|GPU device|CPU device" | head -4

[ -f head/head_labels.bin ] || { echo "ERROR: head/head_labels.bin is missing"; exit 1; }
[ -f simulations.txt ] || { echo "ERROR: simulations.txt is missing"; exit 1; }

N=$(grep -c '|' simulations.txt); K=0; DONE=0; SKIP=0; FAIL=0; NODET=0
START=$(date +%s)
while IFS='|' read -r DIR ID; do
    [ -z "$DIR" ] && continue
    K=$((K + 1))
    if [ "$FORCE" != "1" ] && [ -f "$DIR/$ID.done" ]; then SKIP=$((SKIP + 1)); continue; fi
    printf '[%d/%d] %s/%s ... ' "$K" "$N" "$DIR" "$ID"
    T0=$(date +%s)
    if ( cd "$DIR" && "$BIN" -f "$ID.json" $ENGINE_ARGS $EXTRA_ARGS < /dev/null > "$ID.log" 2>&1 ); then
        echo "$ENGINE $(date)" > "$DIR/$ID.done"
        DONE=$((DONE + 1))
        echo "done ($(( $(date +%s) - T0 )) s)"
    elif [ -f "$DIR/$ID.jnii" ] && grep -aq "error when converting to JSON" "$DIR/$ID.log"; then
        # MCX-CL aborts writing an empty _detp file when no photon reached a detector; the fluence is fine
        echo "$ENGINE $(date) - 0 detected photons" > "$DIR/$ID.done"
        DONE=$((DONE + 1)); NODET=$((NODET + 1))
        echo "done ($(( $(date +%s) - T0 )) s) - no photons reached a detector, no _detp file"
    else
        FAIL=$((FAIL + 1))
        echo "FAILED - see $DIR/$ID.log"
        tail -3 "$DIR/$ID.log" | sed 's/^/      /'
    fi
done < simulations.txt

echo
echo "Finished in $(( ($(date +%s) - START) / 60 )) min: $DONE run, $SKIP already done, $FAIL failed (of $N)."
[ "$NODET" -gt 0 ] && echo "$NODET run(s) detected no photons (fluence saved, no _detp file) - consider more photons."
[ "$FAIL" -eq 0 ]
'''

COMMAND_TEMPLATE = r'''#!/bin/bash
# Double-click in Finder to run the simulations in Terminal (macOS).
cd "$(dirname "$0")" || exit 1
chmod +x run_analysis.sh 2>/dev/null
./run_analysis.sh "$@"
echo
read -r -p "Press Enter to close this window..."
'''

BAT_TEMPLATE = r'''@echo off
rem Run all MCX / MCX-CL simulations in this folder (Windows).
rem
rem   run_analysis.bat           auto-detect MCX (CUDA) or MCX-CL (OpenCL)
rem   run_analysis.bat mcxcl     force MCX-CL;  run_analysis.bat mcx   force MCX
rem   set MCX_BIN=C:\path\to\mcxcl.exe   use a specific executable
rem   set MCX_ARGS=-G 1                  extra MCX command-line options
rem
rem Finished simulations are skipped (name.done); delete them (or set FORCE=1) to rerun.

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "PREF=%~1"
if "%PREF%"=="" if defined MCX_ENGINE set "PREF=%MCX_ENGINE%"
if "%PREF%"=="" set "PREF=__ENGINE__"
set "GPU_ARGS=__GPU_ARGS__"
if not defined FORCE set "FORCE=0"
set "BIN="
set "ENGINE="

if defined MCX_BIN (
    set "E=mcx"
    echo "%MCX_BIN%" | findstr /i "mcxcl" >nul && set "E=mcxcl"
    call :try "%MCX_BIN%" !E!
    if not defined BIN (
        echo MCX_BIN=%MCX_BIN% does not run ^("%MCX_BIN%" -L failed^).
        goto :fail
    )
) else (
    if /i "%PREF%"=="mcx" (
        call :find mcx
    ) else if /i "%PREF%"=="mcxcl" (
        call :find mcxcl
    ) else (
        call :find mcx
        if not defined BIN call :find mcxcl
    )
)
if not defined BIN (
    echo ERROR: no working MCX installation found.
    echo   Looked for %PREF% on the PATH and in the usual MCXStudio folders:
    echo     %USERPROFILE%\Desktop\MCXStudio, %USERPROFILE%\MCXStudio, %USERPROFILE%\Downloads\MCXStudio,
    echo     %ProgramFiles%\MCXStudio, C:\MCXStudio
    echo   Install MCX Studio from https://mcx.space ^(contains mcx, mcxcl and mmc^), then either add
    echo   MCXSuite\mcx\bin or MCXSuite\mcxcl\bin to the PATH, or run
    echo       set MCX_BIN=C:\path\to\MCXSuite\mcxcl\bin\mcxcl.exe
    echo       run_analysis.bat
    echo   Note: mcx ^(CUDA^) needs an NVIDIA GPU and driver; otherwise use mcxcl ^(OpenCL^).
    goto :fail
)

set "ENGINE_ARGS=%GPU_ARGS%"
if /i "%ENGINE%"=="mcxcl" (
    "%BIN%" -L 2>nul | findstr /c:"GPU device" >nul || echo WARNING: MCX-CL found no GPU - running on the CPU ^(much slower^).
)
echo Using %ENGINE%: %BIN%

if not exist "head\head_labels.bin" ( echo ERROR: head\head_labels.bin is missing & goto :fail )
if not exist "simulations.txt" ( echo ERROR: simulations.txt is missing & goto :fail )

set /a N=0, K=0, DONE=0, SKIP=0, FAIL=0, NODET=0
for /f "usebackq tokens=1,2 delims=|" %%a in ("simulations.txt") do set /a N+=1
for /f "usebackq tokens=1,2 delims=|" %%a in ("simulations.txt") do (
    set /a K+=1
    set "D=%%a"
    set "D=!D:/=\!"
    set "RUN=1"
    if "!FORCE!"=="0" if exist "!D!\%%b.done" set "RUN=0"
    if "!RUN!"=="0" (
        set /a SKIP+=1
    ) else (
        echo [!K!/!N!] !D!\%%b
        pushd "!D!"
        "!BIN!" -f "%%b.json" !ENGINE_ARGS! %MCX_ARGS% > "%%b.log" 2>&1
        if errorlevel 1 (
            set "ZERO=0"
            if exist "%%b.jnii" findstr /c:"error when converting to JSON" "%%b.log" >nul 2>&1 && set "ZERO=1"
            if "!ZERO!"=="1" (
                rem MCX-CL aborts writing an empty _detp file when no photon reached a detector; the fluence is fine
                echo !ENGINE! %DATE% %TIME% - 0 detected photons> "%%b.done"
                set /a DONE+=1, NODET+=1
                echo     done - no photons reached a detector, no _detp file
            ) else (
                set /a FAIL+=1
                echo     FAILED - see !D!\%%b.log
            )
        ) else (
            echo !ENGINE! %DATE% %TIME%> "%%b.done"
            set /a DONE+=1
        )
        popd
    )
)
echo.
echo Finished: !DONE! run, !SKIP! already done, !FAIL! failed ^(of !N!^).
if !NODET! gtr 0 echo !NODET! run^(s^) detected no photons ^(fluence saved, no _detp file^) - consider more photons.
pause
if !FAIL! gtr 0 exit /b 1
exit /b 0

:fail
pause
exit /b 1

:find
rem %1 = mcx or mcxcl
for %%p in (%1.exe) do if not "%%~$PATH:p"=="" call :try "%%~$PATH:p" %1
if defined BIN goto :eof
for %%d in ("%USERPROFILE%\Desktop\MCXStudio" "%USERPROFILE%\MCXStudio" "%USERPROFILE%\Downloads\MCXStudio" "%ProgramFiles%\MCXStudio" "%LOCALAPPDATA%\MCXStudio" "C:\MCXStudio") do (
    if not defined BIN if exist "%%~d\MCXSuite\%1\bin\%1.exe" call :try "%%~d\MCXSuite\%1\bin\%1.exe" %1
)
goto :eof

:try
rem %1 = executable, %2 = engine name; accept it if "-L" (list devices) succeeds
"%~1" -L >nul 2>&1
if not errorlevel 1 (
    set "BIN=%~1"
    set "ENGINE=%2"
)
goto :eof
'''


def write_scripts(folder, settings: McxSettings, mcx_hint=None):
    """run_analysis.sh / .command / .bat.  ``mcx_hint``: MCXSuite folder known at export time (searched too)."""
    folder = Path(folder)
    suites = list(MCX_DEFAULT_BINARIES)
    if mcx_hint:
        h = str(Path(mcx_hint))
        home = str(Path.home())
        h = "$HOME" + h[len(home):] if h.startswith(home) else h
        if h not in suites:
            suites.insert(0, h)
    engine = settings.engine if settings.engine in ("mcx", "mcxcl") else "auto"
    gpu = f"-G {settings.gpu}" if str(settings.gpu).strip() else ""
    if settings.save_detected:  # buffer size has no JSON key, so it goes on the command line
        gpu = f"{gpu} -H {int(settings.max_detected)}".strip()
    sh = (SH_TEMPLATE.replace("__ENGINE__", engine).replace("__GPU_ARGS__", gpu)
          .replace("__SUITES__", "\n".join(f'    "{s}"' for s in suites)))
    for name, text in (("run_analysis.sh", sh), ("run_analysis.command", COMMAND_TEMPLATE)):
        p = folder / name
        with open(p, "w", newline="\n") as f:
            f.write(text)
        p.chmod(0o755)
    bat = (BAT_TEMPLATE.replace("__ENGINE__", engine).replace("__GPU_ARGS__", gpu)
           )
    with open(folder / "run_analysis.bat", "w", newline="\r\n") as f:
        f.write(bat)


def write_readme(folder, settings: McxSettings, info):
    wl = ", ".join(f"{w:g}" for w in settings.wavelengths)
    txt = f"""MCX photon-migration simulations for fNIRS probe registrations
Created {info['created']} by probe_registration_gui.py (mcx_export.py)

RUNNING
  macOS : double-click run_analysis.command (or in Terminal: ./run_analysis.sh)
  Linux : ./run_analysis.sh
  Windows: double-click run_analysis.bat
  The scripts look for MCX (CUDA, NVIDIA GPUs) first and then MCX-CL (OpenCL: Apple,
  AMD, Intel and NVIDIA GPUs), on the PATH and in the usual MCXStudio folders.  Force one
  with "run_analysis.sh mcxcl" / "run_analysis.bat mcxcl", or point to an executable
  with the MCX_BIN environment variable.  MCX Studio: https://mcx.space
  Finished simulations are skipped when the script is run again (<name>.done files).
  If macOS refuses to run the script after copying, run:  chmod +x run_analysis.sh run_analysis.command

CONTENTS
  head/head_labels.bin      uint8 tissue labels {list(info['volume']['dim'])}, x fastest, {info['volume']['voxel_mm']:g} mm voxels
  head/head_labels.nii.gz   same labels with the voxel->world (scanner mm) affine
     labels: 0 air, 1 skin, 2 skull, 3 CSF, 4 gray matter, 5 white matter, 6 lesion
     MCX stops photons that enter label 0 or leave the grid, so the volume was cleaned:
     {info['volume']['cleanup']['tissue_islands_removed']} voxels of floating tissue specks removed, {info['volume']['cleanup']['enclosed_air_filled']} voxels of enclosed air and
     {info['volume']['cleanup']['surface_crevices_filled']} voxels of scalp holes/crevices (ball closing, radius {info['volume']['cleanup']['closing_radius_mm']:g} mm) filled with the
     nearest tissue, and the grid padded with air by {info['volume']['cleanup']['padding_voxels']} voxels (before/after, per axis).
  optical_properties.csv    mua, mus (1/mm), g, n per label and wavelength (MCX Media[label])
     musp = mus*(1-g) is listed for reference (MCX uses mus and g).  Skin and bone tables in the
     toolbox are μs′ and were converted to mus = μs′/(1-g).
     from the NIRS-toolbox nirs.media.tissues: skin, bone (skull), water (CSF),
     brain (gray, white and lesion; SO2 {settings.brain_so2:g}, HbT {settings.brain_hbt_uM:g} uM)
  settings.json             all settings, voxel->world affine, label map
  simulations.txt / .csv    one line per simulation (folder|name); the .csv lists the
                            detector order used for the detected-photon detector IDs
  sessions/<session>/optodes.csv   registered optodes: world mm (X,Y,Z), point on the voxel
                            surface (surf_*), MCX grid position (mcx_*) and launch direction
  sessions/<session>/channels.csv  source-detector channels and distances
  sessions/<session>/<wavelength>nm/<optode>.json   MCX input, one per launch optode

SIMULATIONS
  {info['n_simulations']} simulations: {len(info['sessions'])} session(s) x wavelengths [{wl}] nm x launch optodes
  ({'sources and detectors (adjoint)' if settings.simulate == 'both' else 'sources only'}), {settings.photons:.3g} photons each.
  Pencil beam launched from the scalp surface along the inward normal; detectors are discs
  of radius {settings.detector_radius_mm:g} mm on the scalp.  Time window {settings.t0:g}-{settings.t1:g} s, step {settings.dt:g} s.
  Outputs next to each .json: <optode>.{settings.output_format} (fluence/flux volume on the head grid,
  {'saved' if settings.save_fluence else 'not saved'}) and <optode>_detp.jdat (detected photons: detector ID and
  partial path length in each label, {'saved' if settings.save_detected else 'not saved'}); <optode>.log is the MCX log.
  If no photon reaches any detector, MCX-CL cannot write an empty _detp file; the run scripts still count it as
  done (its .done file says "0 detected photons") since the fluence is valid - use more photons if you need it.
  Read .jnii/.jdat in Python with "pip install jdata" (jdata.load) or in MATLAB with JSONLab/JNIfTI.
  MCX grid position -> world mm:  world = affine @ [mcx - 0.5, 1]   (affine in settings.json)
  Channel sensitivity (adjoint): J(r) ~ phi_source(r) * phi_detector(r) on the head grid.
"""
    (Path(folder) / "README.txt").write_text(txt)


def find_local_mcx():
    """MCXSuite folder on this computer (for the run-script search list), or None."""
    cands = [shutil.which("mcxcl"), shutil.which("mcx")]
    for c in cands:
        if c:
            return str(Path(c).resolve().parents[2])
    for s in MCX_DEFAULT_BINARIES:
        p = Path(s.replace("$HOME", str(Path.home())))
        if (p / "mcxcl" / "bin").exists() or (p / "mcx" / "bin").exists():
            return str(p)
    return None
