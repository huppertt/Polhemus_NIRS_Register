# NIRS probe registration GUI

Repair Polhemus Fastrak hemisphere flips, register fNIRS sessions to a subject's
segmented MRI, compare placements in an interactive 3D viewer, and export portable
MCX / MCX-CL simulation folders.

## Layout

- `probe_registration_GUI/`: GUI, registration, subject head model, and MCX exporter.
- `probe_registration_GUI/assets/`: required atlas/template and numerical tissue
  optical-property tables. These are distributable runtime assets, not subject data.
- `probe_registration_GUI/matlab/`: optional SPM12 preprocessing and asset-generation helpers.
- `tests/`: automated regression tests; no subject/session data required.
- `local_data/`: local MRI, digitizer sessions, original probe file, registration
  results, and MCX exports. **Ignored by Git; not included in a clone.**
- `local_demo/`: local exploratory notebook and original MATLAB digitizer app.
  **Ignored by Git; not included in a clone.**

## Install and run

Use Python 3.10 or newer in a dedicated environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python probe_registration_GUI/probe_registration_gui.py
```

The GUI's file dialogs can load data from any location. For command-line loading:

```bash
python probe_registration_GUI/probe_registration_gui.py \
  --mri /path/to/segmented.nii.gz \
  --t1 /path/to/T1w.nii \
  --lesion /path/to/lesion.nii.gz \
  -- /path/to/sessions/Test*.mat
```

`--t1` and `--lesion` are optional. `--mri` also accepts multiple SPM probability
maps (`c1` GM, `c2` WM, `c3` CSF, `c4` skull, `c5` scalp, optional `c6` other).
Use `--` before session arguments so `--mri` does not consume them.

On the original development machine, the preserved example is now:

```bash
python probe_registration_GUI/probe_registration_gui.py \
  --mri local_data/Pilot003/segmented.nii.gz \
  --t1 local_data/Pilot003/T1w.nii \
  -- local_data/Test*.mat
```

The application locates its assets relative to its source, independent of the
working directory. The bundled atlas includes the fixed probe template used by
this dataset; it is not a generic template for arbitrary probe layouts.

## MRI preprocessing and Monte Carlo export

SPM12 segmentation is a separate MATLAB preprocessing step. Add
`probe_registration_GUI/matlab/` to the MATLAB path, then run:

```matlab
spm12_segment_for_NIRS('/path/to/T1w.nii', [], '/path/to/spm12');
```

No MATLAB or SPM installation is needed to load an existing segmentation in the GUI.
Tissue labels are 0 air, 1 skin, 2 skull, 3 CSF, 4 GM, 5 WM, and optional 6 lesion.

The 3D scalp and default MCX export share head-mask cleaning (3 mm closing and
enclosed-air filling). The display can show the smoothed registration mesh or the
unsmoothed voxel-boundary mesh.

**Export to MCExtreme** creates scripts and all simulation inputs in a folder you
choose. Keep these exports under `local_data/` or outside this repository.
MCX/MCX-CL must be installed on the simulation machine; it is not bundled.
The default is 1e8 photons per optode per wavelength at 780 and 850 nm.
Skin/skull table values are reduced scattering (1/mm), converted to MCX's
scattering coefficient using `mus = musp / (1 - g)`. Lesion, GM, and WM share brain
optics. The export dialog allows optical-property and simulation-setting changes.

To regenerate runtime assets with the NIRS-toolbox on the MATLAB path:

```matlab
export_atlas_for_python('/path/to/digitizer_session.mat', ...
    '/path/to/sample_probeInfo.mat');
export_tissue_optics();
```

The atlas exporter copies only atlas meshes, atlas landmarks, template optodes,
and channel links; not measured session points. It replaces the fixed probe
template for this application. The original source MAT files stay local.

## Tests

From the repository root (macOS/Linux):

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=probe_registration_GUI \
  python -m unittest discover -s tests -v
```

On Windows, set `QT_QPA_PLATFORM=offscreen` and
`PYTHONPATH=probe_registration_GUI` before the same unittest command.
Rendering smoke tests need a native display/OpenGL context.

## GitHub scope

Commit the application, approved runtime assets, MATLAB helper code, tests,
dependency list, this README, and `.gitignore`. Do not force-add local data.
MRI/MAT files are ignored everywhere, with one explicit exception for the approved
atlas/template asset. Nothing was previously tracked when this layout was created.
No commit or push is performed automatically.
