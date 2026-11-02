"""
NIRS probe registration viewer.

Load a subject's segmented MRI, optionally the T1 for slice display and a
manually drawn lesion mask, and one or more NIRS_Digitizer session files
(.mat).  The segmented MRI is either one indexed tissue label image
(``segmented.nii.gz`` from spm12_segment_for_NIRS.m) or the SPM12 tissue
probability maps (c1..c6), combined into the most probable tissue per voxel.
The tissue labels can be saved with the lesion as a 6th tissue type (label 6)
for light modelling; such a file restores the lesion when loaded again.
Each session is repaired (Polhemus
hemisphere flips), registered rigidly to the subject's scalp and its probe is
labelled against the (hard-coded) template probe.  All sessions are drawn on
the subject's 3D scalp so placement variability across sessions can be seen.
"Export to MCExtreme…" writes a portable folder of MCX / MCX-CL Monte Carlo
simulations (head volume, optodes, NIRS-toolbox optical properties, one run
per launch optode, session and wavelength, and run_analysis.sh / .bat).
Background results and errors are delivered through GUI-thread Qt slots;
only the numerical work runs in a worker thread.

    python probe_registration_GUI/probe_registration_gui.py
    python probe_registration_GUI/probe_registration_gui.py --mri /path/to/segmented.nii.gz -- /path/to/Test*.mat
    python probe_registration_GUI/probe_registration_gui.py --mri Sub01/c[1-6]T1w.nii --lesion Sub01/lesion.nii.gz -- Sub01/Test*.mat
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtGui, QtWidgets
from scipy.ndimage import map_coordinates

import digitizer_registration as dr
import subject_headmodel as shm

HERE = Path(__file__).resolve().parent
TEMPLATE_FILE = HERE / "assets" / "atlas_ICBM152_and_probe.mat"
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f"]
SEG_COLORS = np.array([[0, 0, 0], [240, 200, 170], [235, 235, 220], [90, 160, 230], [170, 120, 120], [245, 245, 245], [220, 30, 30]], float) / 255


# --------------------------------------------------------------------------
# Background worker
# --------------------------------------------------------------------------
class Worker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self.fn(*self.args, progress=self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


class BackgroundJob(QtCore.QObject):
    """Keep worker callbacks on the GUI thread, including with PySide6."""

    def __init__(self, window, fn, args, on_done):
        super().__init__(window)
        self.window = window
        self.on_done = on_done
        self.thread = QtCore.QThread(self)
        self.worker = Worker(fn, *args)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        # Plain Python closures have no reliable receiver affinity in PySide6.
        queued = QtCore.Qt.QueuedConnection
        self.worker.progress.connect(self._progress, queued)
        self.worker.finished.connect(self._succeeded, queued)
        self.worker.failed.connect(self._failed, queued)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._cleanup, queued)

    @QtCore.Slot(str)
    def _progress(self, message):
        self.window.status.showMessage(message)

    @QtCore.Slot(object)
    def _succeeded(self, result):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.thread.quit()
        self.on_done(result)

    @QtCore.Slot(str)
    def _failed(self, message):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.thread.quit()
        self.window.status.showMessage("Error")
        QtWidgets.QMessageBox.critical(self.window, "Error", message)

    @QtCore.Slot()
    def _cleanup(self):
        self.thread.wait()
        self.window._threads.remove(self)
        self.deleteLater()


# --------------------------------------------------------------------------
# Session bookkeeping
# --------------------------------------------------------------------------
class SessionEntry:
    def __init__(self, path, color):
        self.path = Path(path)
        self.name = self.path.stem
        self.color = color
        self.visible = True
        self.result: dr.Result | None = None
        self.error = ""

    @property
    def ok(self):
        return self.result is not None and self.result.probe is not None


def register_sessions(model, paths, progress=None):
    out = {}
    for i, p in enumerate(paths):
        if progress:
            progress(f"Registering {Path(p).name} ({i + 1}/{len(paths)})")
        try:
            out[p] = (dr.process(p, model, model.scalp_surface, allow_scaling=False), "")
        except Exception as e:
            out[p] = (None, f"{type(e).__name__}: {e}")
    return out


def build_model(seg_paths, atlas, lesion_path=None, progress=None):
    return shm.build_subject_model(seg_paths, atlas, progress=progress, lesion_path=lesion_path)


def attach_lesion(model, lesion_path, progress=None):
    # work on a shallow copy so the GUI thread never sees a half-attached lesion
    return shm.attach_lesion(dataclasses.replace(model), lesion_path, progress=progress)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIRS probe registration – subject MRI")
        self.resize(1500, 950)
        self.atlas = dr.load_atlas(TEMPLATE_FILE)
        self.model: shm.SubjectModel | None = None
        self.t1 = None                     # (data, affine)
        self.sessions: dict[str, SessionEntry] = {}
        self._threads = []
        self._mcx_folder = ""
        self._camera_set = False

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.enable_anti_aliasing("ssaa")

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self.plotter.interactor)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 1020])
        self.setCentralWidget(splitter)
        self.status = self.statusBar()
        self._update_enabled()

    # ---------------- UI construction ----------------
    def _build_controls(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        box = QtWidgets.QGroupBox("Subject MRI")
        g = QtWidgets.QGridLayout(box)
        self.btn_mri = QtWidgets.QPushButton("Load segmented MRI…")
        self.btn_mri.setToolTip("Select ONE indexed tissue label image (e.g. segmented.nii.gz),\n"
                                "or ALL of the SPM12 tissue probability maps (c1*.nii … c6*.nii)\n"
                                "to combine them into the most probable tissue per voxel.")
        self.btn_mri.clicked.connect(self.load_mri_dialog)
        self.btn_t1 = QtWidgets.QPushButton("Load T1 (optional)…")
        self.btn_t1.clicked.connect(self.load_t1_dialog)
        self.btn_lesion = QtWidgets.QPushButton("Load lesion mask (optional)…")
        self.btn_lesion.clicked.connect(self.load_lesion_dialog)
        self.btn_clear_lesion = QtWidgets.QPushButton("Remove lesion")
        self.btn_clear_lesion.clicked.connect(lambda: self.load_lesion(None))
        self.btn_save_labels = QtWidgets.QPushButton("Save tissue labels (lesion = 6)…")
        self.btn_save_labels.setToolTip("Write the subject's tissue labels as one indexed NIfTI for light modelling / faster loading:\n"
                                        "0 air, 1 skin, 2 skull, 3 CSF, 4 GM, 5 WM, 6 lesion.\n"
                                        "Lesion voxels in CSF/GM/WM become 6; lesion voxels in air/skin/skull are left unchanged.\n"
                                        "Loading this file again restores the lesion mask from label 6.")
        self.btn_save_labels.clicked.connect(self.save_labels_dialog)
        self.lbl_mri = QtWidgets.QLabel("No MRI loaded")
        self.lbl_mri.setWordWrap(True)
        self.lbl_lesion = QtWidgets.QLabel("No lesion mask")
        self.lbl_lesion.setWordWrap(True)
        g.addWidget(self.btn_mri, 0, 0)
        g.addWidget(self.btn_t1, 0, 1)
        g.addWidget(self.btn_lesion, 1, 0)
        g.addWidget(self.btn_clear_lesion, 1, 1)
        g.addWidget(self.btn_save_labels, 2, 0, 1, 2)
        g.addWidget(self.lbl_mri, 3, 0, 1, 2)
        g.addWidget(self.lbl_lesion, 4, 0, 1, 2)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Digitizer sessions")
        v = QtWidgets.QVBoxLayout(box)
        h = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add sessions…")
        self.btn_add.clicked.connect(self.add_sessions_dialog)
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self.remove_selected)
        h.addWidget(self.btn_add)
        h.addWidget(self.btn_remove)
        v.addLayout(h)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Show", "Color", "Session", "Flipped extra", "Extra→scalp RMS",
                                              "Optode height", "Digitized", "Status"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._table_item_changed)
        self.table.cellDoubleClicked.connect(self._table_double_clicked)
        self.table.setMinimumHeight(150)
        v.addWidget(self.table)
        v.addWidget(QtWidgets.QLabel("Double-click a colour to change it."))
        lay.addWidget(box, 2)

        box = QtWidgets.QGroupBox("Display")
        g = QtWidgets.QGridLayout(box)
        self.chk = {}
        opts = [("scalp", "Scalp", True), ("skull", "Skull", False), ("brain", "Brain", False),
                ("channels", "Channels", True), ("interp", "Interpolated optodes", True),
                ("template", "Template probe (MRI)", False), ("mean", "Mean probe across sessions", False),
                ("labels", "Optode labels", False), ("extra", "Extra head points", False),
                ("landmarks", "LPA/RPA/Cz (clicked + MRI)", False), ("slices", "MRI slices", False),
                ("lesion", "Lesion", True)]
        for i, (k, label, val) in enumerate(opts):
            c = QtWidgets.QCheckBox(label)
            c.setChecked(val)
            c.toggled.connect(self.redraw)
            self.chk[k] = c
            g.addWidget(c, i // 2, i % 2)
        r = (len(opts) + 1) // 2
        g.addWidget(QtWidgets.QLabel("Colour optodes by"), r, 0)
        self.cmb_color = QtWidgets.QComboBox()
        self.cmb_color.addItems(["Session", "Deviation from session mean (mm)", "Shift from template (mm)",
                                 "Distance to lesion (mm)"])
        self.cmb_color.currentIndexChanged.connect(self.redraw)
        g.addWidget(self.cmb_color, r, 1)
        g.addWidget(QtWidgets.QLabel("Scalp opacity"), r + 1, 0)
        self.sld_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_opacity.setRange(5, 100)
        self.sld_opacity.setValue(100)
        self.sld_opacity.valueChanged.connect(self.redraw)
        g.addWidget(self.sld_opacity, r + 1, 1)
        g.addWidget(QtWidgets.QLabel("Scalp surface"), r + 2, 0)
        self.cmb_scalp = QtWidgets.QComboBox()
        self.cmb_scalp.addItems(["Smoothed (used for registration)", "As simulated (MCX voxel boundary)"])
        self.cmb_scalp.setToolTip(
            "Both are extracted from the same cleaned head mask as the MCX export volume\n"
            f"(specks removed, {shm.SCALP_CLOSE_MM:g} mm ball closing, enclosed air filled).\n"
            "Smoothed: blurred, decimated, Taubin-smoothed mesh used to register the probes.\n"
            "As simulated: raw voxel boundary, flat-shaded - what the photons actually see.")
        self.cmb_scalp.currentIndexChanged.connect(self.redraw)
        g.addWidget(self.cmb_scalp, r + 2, 1)
        r += 1
        self.sld_slice = {}
        for j, ax in enumerate("XYZ"):
            g.addWidget(QtWidgets.QLabel(f"Slice {ax} (mm)"), r + 2 + j, 0)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(-120, 120)
            s.valueChanged.connect(self._slice_moved)
            self.sld_slice[ax] = s
            g.addWidget(s, r + 2 + j, 1)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("View")
        h = QtWidgets.QGridLayout(box)
        for i, (name, vec, up) in enumerate([("Left", (-1, 0, 0), (0, 0, 1)), ("Right", (1, 0, 0), (0, 0, 1)),
                                             ("Front", (0, 1, 0), (0, 0, 1)), ("Back", (0, -1, 0), (0, 0, 1)),
                                             ("Top", (0, 0, 1), (0, 1, 0))]):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda _=False, v=vec, u=up: self.set_view(v, u))
            h.addWidget(b, 0, i)
        b = QtWidgets.QPushButton("Save screenshot…")
        b.clicked.connect(self.save_screenshot)
        h.addWidget(b, 1, 0, 1, 2)
        self.btn_export = QtWidgets.QPushButton("Export registrations…")
        self.btn_export.clicked.connect(self.export_dialog)
        h.addWidget(self.btn_export, 1, 2, 1, 3)
        self.btn_mcx = QtWidgets.QPushButton("Export to MCExtreme…")
        self.btn_mcx.setToolTip("Write a portable folder of MCX / MCX-CL Monte Carlo simulations "
                                "(head volume, optodes, inputs, run_analysis.sh/.bat)")
        self.btn_mcx.clicked.connect(self.export_mcx_dialog)
        h.addWidget(self.btn_mcx, 2, 0, 1, 5)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Placement variability across visible sessions")
        v = QtWidgets.QVBoxLayout(box)
        self.var_table = QtWidgets.QTableWidget(0, 4)
        self.var_table.setHorizontalHeaderLabels(["Optode", "n", "Mean dev (mm)", "Max dev (mm)"])
        self.var_table.verticalHeader().setVisible(False)
        self.var_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.var_table.setSortingEnabled(True)
        v.addWidget(self.var_table)
        lay.addWidget(box, 1)
        return w

    def _update_enabled(self):
        has = self.model is not None
        self.btn_add.setEnabled(has)
        self.btn_lesion.setEnabled(has)
        self.btn_clear_lesion.setEnabled(has and self.model.lesion_mask is not None)
        self.btn_save_labels.setEnabled(has)
        self.chk["lesion"].setEnabled(has and self.model.has_lesion)
        # "Distance to lesion" colour mode only with a lesion
        item = self.cmb_color.model().item(3)
        item.setEnabled(has and self.model.has_lesion)
        if not item.isEnabled() and self.cmb_color.currentIndex() == 3:
            self.cmb_color.setCurrentIndex(0)
        self.btn_export.setEnabled(any(s.ok for s in self.sessions.values()))
        self.btn_mcx.setEnabled(has and any(s.ok for s in self.sessions.values()))
        for s in self.sld_slice.values():
            s.setEnabled(has)

    # ---------------- threading helper ----------------
    def _run(self, fn, args, on_done):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        job = BackgroundJob(self, fn, args, on_done)
        self._threads.append(job)
        job.thread.start()

    # ---------------- loading ----------------
    def load_mri_dialog(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Segmented MRI: one tissue label image, or all SPM tissue probability maps (c1–c6)",
            str(HERE), "NIfTI (*.nii *.nii.gz)")
        if files:
            self.load_mri(files)

    def load_mri(self, paths, lesion=None, then=None):
        """Build the head model from a label image or SPM probability maps (+ optional lesion mask).

        Loading a new MRI drops any lesion mask of the previous subject.
        """
        paths = [str(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        names = ", ".join(Path(p).name for p in paths)
        self.lbl_mri.setText(f"Building head model from {names} …")

        def done(model):
            self.model = model
            self._seg_cache = None
            self._camera_set = False
            self.lbl_mri.setText(f"{names}\n({model.seg_source}; folder {Path(paths[0]).parent})\n"
                                 f"Atlas→subject scalp fit RMS: {model.fit_rms_mm:.1f} mm")
            self._update_lesion_label()
            c = model.scalp[0].mean(0)
            for ax, s in zip("XYZ", self.sld_slice.values()):
                s.blockSignals(True)
                s.setValue(int(round(c["XYZ".index(ax)])))
                s.blockSignals(False)
            self._update_enabled()
            self.status.showMessage("Head model ready")
            sessions = list(self.sessions)
            if sessions:
                self.register(sessions, then)
            else:
                self.redraw()
                if then:
                    then()

        self._run(build_model, (paths, self.atlas, lesion), done)

    def load_lesion_dialog(self):
        start = Path(self.model.seg_files[0]).parent if self.model else HERE
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Lesion mask", str(start), "NIfTI (*.nii *.nii.gz)")
        if f:
            self.load_lesion(f)

    def load_lesion(self, path, then=None):
        """Attach (path) or remove (None) the lesion mask of the current subject."""
        if self.model is None:
            return

        def done(model):
            self.model = model
            self._update_lesion_label()
            self._update_enabled()
            if model.lesion_mask is not None and not model.has_lesion:
                QtWidgets.QMessageBox.warning(self, "Lesion mask", "The lesion mask is empty on this MRI's grid "
                                              "(is it in the same space as the segmentation?).")
            self.redraw()
            if then:
                then()

        self._run(attach_lesion, (self.model, path), done)

    def _update_lesion_label(self):
        m = self.model
        if m is None or m.lesion_mask is None:
            self.lbl_lesion.setText("No lesion mask")
            return
        frac = m.lesion_tissue_fraction()
        tissue = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(frac.items(), key=lambda kv: -kv[1]) if v >= 0.01)
        txt = f"Lesion: {Path(m.lesion_path).name} – {m.lesion_volume_ml():.1f} mL"
        if tissue:
            txt += f"\n  tissue under lesion: {tissue}"
            if frac.get("air", 0) + frac.get("skin", 0) > 0.2:
                txt += "\n  ⚠ much of the lesion is outside the head/brain – check it is in this MRI's space"
        self.lbl_lesion.setText(txt)

    def save_labels_dialog(self):
        if self.model is None:
            return
        start = Path(self.model.seg_files[0]).parent / "segmented.nii.gz"
        if self.model.has_lesion:
            start = start.with_name("segmented_lesion.nii.gz")
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save tissue labels", str(start), "NIfTI (*.nii.gz *.nii)")
        if f:
            n_out = shm.save_labels(self.model.labels, self.model.affine, f, self.model.lesion_mask)
            msg = f"Saved tissue labels to {f}"
            if self.model.has_lesion:
                msg += f" (lesion = label 6; {n_out} lesion voxels outside CSF/GM/WM left unchanged)"
            self.status.showMessage(msg)

    def load_t1_dialog(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "T1 MRI", str(HERE), "NIfTI (*.nii *.nii.gz)")
        if f:
            self.load_t1(f)

    def load_t1(self, path):
        import nibabel as nib
        img = nib.load(str(path))
        self.t1 = (np.asarray(img.dataobj, dtype=np.float32), img.affine)
        v = self.t1[0]
        self._t1_clim = np.percentile(v[v > 0], [1, 99.5])
        self.chk["slices"].setChecked(True)
        if self.sld_opacity.value() > 50:
            self.sld_opacity.setValue(40)      # so the slices are visible through the scalp
        self.redraw()

    def add_sessions_dialog(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Digitizer sessions", str(HERE), "MAT files (*.mat)")
        if files:
            self.add_sessions(files)

    def add_sessions(self, files, then=None):
        new = []
        for f in files:
            f = str(Path(f).resolve())
            if f in self.sessions:
                continue
            self.sessions[f] = SessionEntry(f, PALETTE[len(self.sessions) % len(PALETTE)])
            new.append(f)
        self._refresh_table()
        if new and self.model is not None:
            self.register(new, then)
        elif then:
            then()

    def register(self, paths, then=None):
        def done(out):
            for p, (res, err) in out.items():
                if p in self.sessions:
                    self.sessions[p].result, self.sessions[p].error = res, err
            self._refresh_table()
            self._update_enabled()
            self.status.showMessage(f"Registered {len(out)} session(s)")
            self.redraw()
            if then:
                then()
        self._run(register_sessions, (self.model, paths), done)

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        keys = list(self.sessions)
        for r in rows:
            self.sessions.pop(keys[r], None)
        self._refresh_table()
        self._update_enabled()
        self.redraw()

    # ---------------- session table ----------------
    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.sessions))
        for row, s in enumerate(self.sessions.values()):
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            it.setCheckState(QtCore.Qt.Checked if s.visible else QtCore.Qt.Unchecked)
            self.table.setItem(row, 0, it)
            it = QtWidgets.QTableWidgetItem("")
            it.setBackground(QtGui.QColor(s.color))
            it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 1, it)
            vals = [s.name, "", "", "", "", ""]
            if s.ok:
                r, rp, hi = s.result, s.result.repaired, s.result.head_info
                P = r.probe
                n_long = (P.status != "short-sep (follows source)").sum()
                vals = [s.name, f"{rp.extra_flipped.sum()}/{len(rp.extra)}", f"{hi['extra_rms_mm']:.1f} mm",
                        f"{hi['optode_height_mm']:.0f} mm", f"{(P.status == 'digitized').sum()}/{n_long}", "ok"]
            elif s.result is not None:
                vals[-1] = s.result.note
            elif s.error:
                vals[-1] = s.error
            elif self.model is None:
                vals[-1] = "waiting for MRI"
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                if c == 5:
                    it.setToolTip(v)
                self.table.setItem(row, c + 2, it)
        self.table.blockSignals(False)

    def _table_item_changed(self, item):
        if item.column() == 0:
            s = list(self.sessions.values())[item.row()]
            s.visible = item.checkState() == QtCore.Qt.Checked
            self.redraw()

    def _table_double_clicked(self, row, col):
        if col != 1:
            return
        s = list(self.sessions.values())[row]
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(s.color), self)
        if c.isValid():
            s.color = c.name()
            self._refresh_table()
            self.redraw()

    # ---------------- drawing ----------------
    def _visible(self):
        return [s for s in self.sessions.values() if s.ok and s.visible]

    def _lift(self, P, d):
        """Move points d mm outward from the scalp so they are not hidden inside it."""
        q, n = self.model.scalp_surface.closest(P)
        return q + d * n

    def _deviation_table(self, vis):
        if len(vis) < 1:
            return None, None
        names = vis[0].result.probe.Name.values
        X = np.stack([s.result.probe.set_index("Name").loc[names, ["X", "Y", "Z"]].values for s in vis])
        mean = X.mean(0)
        dev = np.linalg.norm(X - mean, axis=2)
        return mean, dev

    def redraw(self, *_):
        if self.model is None:
            return
        pl = self.plotter
        cam = pl.camera_position if self._camera_set else None
        # (Plotter.clear() would also delete the light kit and leave surfaces unshaded)
        pl.clear_actors()
        pl.remove_legend()
        for bar in list(pl.scalar_bars.keys()):
            pl.remove_scalar_bar(bar)
        m = self.model
        op = self.sld_opacity.value() / 100

        def mesh(nf):
            n, f = nf
            return pv.PolyData(n, np.c_[np.full(len(f), 3), f].ravel())

        if self.chk["scalp"].isChecked():
            if self.cmb_scalp.currentIndex() == 1:
                if getattr(self, "_vox_scalp_cache", (None,))[0] is not m:
                    self.statusBar().showMessage("Extracting voxel scalp surface…")
                    QtWidgets.QApplication.processEvents()
                    self._vox_scalp_cache = (m, mesh(m.voxel_scalp()))
                    self.statusBar().clearMessage()
                pl.add_mesh(self._vox_scalp_cache[1], color="peachpuff", opacity=op, smooth_shading=False,
                            name="scalp", specular=0.1)
            else:
                pl.add_mesh(mesh(m.scalp), color="peachpuff", opacity=op, smooth_shading=True, name="scalp",
                            specular=0.1)
        if self.chk["skull"].isChecked():
            pl.add_mesh(mesh(m.layers[1]), color="ivory", opacity=0.5, smooth_shading=True, name="skull")
        if self.chk["brain"].isChecked():
            pl.add_mesh(mesh(m.layers[-1]), color="#d8a0a0", smooth_shading=True, name="brain")
        if m.lesion_surface is not None and self.chk["lesion"].isChecked():
            pl.add_mesh(mesh(m.lesion_surface), color="red", opacity=0.85, smooth_shading=True, name="lesion")
        if self.chk["slices"].isChecked():
            self._draw_slices()

        vis = self._visible()
        mean, dev = self._deviation_table(vis)
        mode = self.cmb_color.currentIndex()
        if mode == 3 and not m.has_lesion:
            mode = 0
        clim, cmap = None, "plasma"
        if mode == 1 and dev is not None:
            clim = [0, max(5.0, float(np.ceil(dev.max())))]
        elif mode == 2 and vis:
            clim = [0, max(5.0, float(np.ceil(max(s.result.probe.shift_from_template_mm.max() for s in vis))))]
        elif mode == 3 and vis:
            clim = [0, min(80.0, max(10.0, float(np.ceil(max(
                m.lesion_distance(s.result.probe[["X", "Y", "Z"]].values).max() for s in vis)))))]
            cmap = "RdYlGn"                      # red = optode close to the lesion
        sphere, cube = pv.Sphere(radius=3.2), pv.Cube(x_length=5.2, y_length=5.2, z_length=5.2)
        bar_added = False

        for k, s in enumerate(vis):
            P = s.result.probe
            xyz = self._lift(P[["X", "Y", "Z"]].values, 2.5)
            if mode == 1:
                scal = dev[k]
            elif mode == 2:
                scal = P.shift_from_template_mm.values
            elif mode == 3:
                scal = m.lesion_distance(P[["X", "Y", "Z"]].values)
            else:
                scal = None
            for typ, geom in (("Source", sphere), ("Detector", cube)):
                for interp in (False, True):
                    sel = (P.Type.values == typ) & ((P.status.values == "interpolated") == interp)
                    if interp and not self.chk["interp"].isChecked():
                        continue
                    if not sel.any():
                        continue
                    pd_ = pv.PolyData(xyz[sel])
                    kw = dict(opacity=0.35 if interp else 1.0, smooth_shading=True)
                    if scal is None:
                        g = pd_.glyph(geom=geom, scale=False, orient=False)
                        pl.add_mesh(g, color=s.color, **kw)
                    else:
                        pd_["mm"] = scal[sel]
                        g = pd_.glyph(geom=geom, scale=False, orient=False)
                        pl.add_mesh(g, scalars="mm", cmap=cmap, clim=clim, show_scalar_bar=not bar_added,
                                    scalar_bar_args=dict(title="mm", vertical=True,
                                                         position_x=0.88, position_y=0.1), **kw)
                        bar_added = True
            if self.chk["channels"].isChecked():
                idx = {nm: i for i, nm in enumerate(P.Name)}
                lines = []
                for a, b in m.link:
                    lines += [2, idx[f"Source-{a:04d}"], idx[f"Detector-{b:04d}"]]
                pl.add_mesh(pv.PolyData(self._lift(P[["X", "Y", "Z"]].values, 1.5), lines=np.array(lines)),
                            color=s.color, line_width=3)
            if self.chk["extra"].isChecked():
                rp = s.result.repaired
                E = dr.apply(s.result.T, rp.extra[~rp.extra_outlier] * 10)
                if len(E):
                    pl.add_points(E, color=s.color, point_size=5, render_points_as_spheres=True)
            if self.chk["landmarks"].isChecked():
                rp = s.result.repaired
                L = dr.apply(s.result.T, np.array([rp.landmarks[k_] for k_ in ("LPA", "RPA", "Cz")]) * 10)
                pl.add_points(L, color=s.color, point_size=16, render_points_as_spheres=True)

        if self.chk["template"].isChecked():
            T = self._lift(m.probe[["X", "Y", "Z"]].values, 2.0)
            pl.add_points(T, color="gray", point_size=10, render_points_as_spheres=True)
        if self.chk["mean"].isChecked() and mean is not None:
            pl.add_points(self._lift(mean, 3.0), color="black", point_size=12, render_points_as_spheres=True)
        if self.chk["landmarks"].isChecked():
            A = np.array([m.fiducials[k] for k in ("spmlpa", "spmrpa", "Cz", "spmnas")])
            pl.add_point_labels(A, ["LPA (MRI)", "RPA (MRI)", "Cz (MRI)", "Nasion (MRI)"], point_color="magenta",
                                point_size=14, render_points_as_spheres=True, font_size=11, shape_opacity=0.5)
        if self.chk["labels"].isChecked():
            names = (vis[0].result.probe.Name.values if vis else m.probe.Name.values)
            pos = mean if mean is not None else m.probe[["X", "Y", "Z"]].values
            short = [n.replace("Source-00", "S").replace("Detector-00", "D") for n in names]
            pl.add_point_labels(self._lift(pos, 9.0), short, font_size=10, point_size=1, shape_opacity=0.4,
                                always_visible=True)

        legend = [[s.name, s.color] for s in vis]
        if legend and mode == 0:
            pl.add_legend(legend, bcolor="white", face="circle", size=(0.28, 0.04 * len(legend) + 0.02),
                          loc="upper right")
        pl.add_text("sphere = source   cube = detector" + ("   (faded = interpolated)" if self.chk["interp"].isChecked() else ""),
                    position="lower_left", font_size=9, color="black")
        if cam is None:
            self.set_view((-1, 0.4, 0.3), (0, 0, 1))
            self._camera_set = True
        else:
            pl.camera_position = cam
        pl.render()
        self._update_variability(vis, dev)

    def _slice_moved(self, *_):
        if self.chk["slices"].isChecked():
            self.redraw()

    def _seg(self):
        """(labels, affine, head mask) of the current model; the head mask is cached per model."""
        m = self.model
        cache = getattr(self, "_seg_cache", None)
        if cache is None or cache[0] is not m.labels:
            self._seg_cache = (m.labels, m.head_mask.astype(np.uint8))
        return m.labels, m.affine, self._seg_cache[1]

    def _draw_slices(self):
        """Three world-axis-aligned planes resampled from the T1 (or the tissue labels if no T1),
        with the lesion (if loaded and shown) in red."""
        lab, lab_aff, head = self._seg()
        lesion = self.model.lesion_mask if (self.model.has_lesion and self.chk["lesion"].isChecked()) else None
        if self.t1 is not None:
            vol, aff = self.t1
            is_seg = False
        else:
            vol, aff = lab, lab_aff
            is_seg = True
        inv = np.linalg.inv(aff)
        inv_lab = np.linalg.inv(lab_aff)
        lo, hi = self.model.scalp[0].min(0) - 10, self.model.scalp[0].max(0) + 10
        res = 1.0
        for ax in range(3):
            c = self.sld_slice["XYZ"[ax]].value()
            u, v = [i for i in range(3) if i != ax]
            gu = np.arange(lo[u], hi[u], res)
            gv = np.arange(lo[v], hi[v], res)
            U, V = np.meshgrid(gu, gv, indexing="ij")
            W = np.zeros((3,) + U.shape)
            W[ax], W[u], W[v] = c, U, V
            ijk = np.tensordot(inv[:3, :3], W, 1) + inv[:3, 3][:, None, None]
            vals = map_coordinates(vol, ijk, order=0 if is_seg else 1, cval=0).ravel(order="F")
            # only show voxels inside the head so the planes don't hide the scalp/probes
            ijk_lab = np.tensordot(inv_lab[:3, :3], W, 1) + inv_lab[:3, 3][:, None, None]
            inside = map_coordinates(head, ijk_lab, order=0, cval=0).ravel(order="F") > 0
            if not inside.any():
                continue
            if is_seg:
                rgb = SEG_COLORS[vals.astype(int).clip(0, 6)]
            else:
                g = np.clip((vals - self._t1_clim[0]) / (self._t1_clim[1] - self._t1_clim[0]), 0, 1)
                rgb = np.repeat(g[:, None], 3, 1)
            if lesion is not None:
                les = map_coordinates(lesion.astype(np.uint8), ijk_lab, order=0, cval=0).ravel(order="F") > 0
                rgb[les] = 0.35 * rgb[les] + 0.65 * np.array([1.0, 0.0, 0.0])
            grid = pv.StructuredGrid(*[W[i] for i in range(3)])
            grid["rgb"] = (rgb * 255).astype(np.uint8)
            grid["inside"] = inside.astype(np.uint8)
            self.plotter.add_mesh(grid.threshold(0.5, scalars="inside"), scalars="rgb", rgb=True)

    def _update_variability(self, vis, dev):
        self.var_table.setSortingEnabled(False)
        if dev is None:
            self.var_table.setRowCount(0)
            return
        P = vis[0].result.probe
        keep = (P.status.values != "short-sep (follows source)")
        names = P.Name.values[keep]
        d = dev[:, keep]
        self.var_table.setRowCount(len(names))
        for r, (n, col) in enumerate(zip(names, d.T)):
            for c, v in enumerate([n.replace("Source-00", "S").replace("Detector-00", "D"), len(vis),
                                   round(float(col.mean()), 1), round(float(col.max()), 1)]):
                it = QtWidgets.QTableWidgetItem()
                it.setData(QtCore.Qt.DisplayRole, v)
                self.var_table.setItem(r, c, it)
        self.var_table.setSortingEnabled(True)
        if len(vis) > 1:
            self.status.showMessage(f"{len(vis)} sessions shown – mean optode deviation from session mean "
                                    f"{d.mean():.1f} mm (max {d.max():.1f} mm)")

    def set_view(self, vec, up):
        self.plotter.view_vector(vec, viewup=up)
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.3)

    # ---------------- output ----------------
    def save_screenshot(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", str(HERE / "registration.png"),
                                                     "PNG (*.png)")
        if f:
            self.plotter.screenshot(f)

    def export_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export folder", str(HERE))
        if d:
            self.export(d)
            self.status.showMessage(f"Exported to {d}")

    def export(self, folder):
        folder = Path(folder)
        m = self.model
        rows = []
        for s in self.sessions.values():
            if not s.ok:
                continue
            res = s.result
            if m.has_lesion:
                probe = res.probe.copy()
                probe["lesion_dist_mm"] = m.lesion_distance(probe[["X", "Y", "Z"]].values)
                res = dataclasses.replace(res, probe=probe)
            dr.save_result(res, folder)
            P = res.probe.copy()
            P.insert(0, "session", s.name)
            rows.append(P)
        if rows:
            allp = pd.concat(rows)
            allp["mri"] = m.seg_path
            allp["lesion_mask"] = m.lesion_path
            allp.to_csv(folder / "all_sessions_registered_probe.csv", index=False)

    def export_mcx_dialog(self):
        import mcx_export as mx
        from mcx_export_dialog import McxExportDialog
        ses = {s.name: s.result.probe for s in self.sessions.values() if s.ok}
        if self.model is None or not ses:
            return
        suite = mx.find_local_mcx()
        dlg = McxExportDialog(ses, folder=self._mcx_folder, has_lesion=self.model.has_lesion,
                              mcx_found=suite, parent=self)
        if not dlg.exec_():
            return
        self._mcx_folder = dlg.folder
        chosen = {k: ses[k] for k in dlg.selected_sessions()}
        settings, optics = dlg.settings(), dlg.optics()

        def done(info):
            self.status.showMessage(f"MCX export: {info['n_simulations']} simulations in {info['folder']}")
            QtWidgets.QMessageBox.information(
                self, "Export to MCExtreme",
                f"Wrote {info['n_simulations']} simulations ({info['total_photons']:.3g} photons) to\n"
                f"{info['folder']}\n\nRun them with run_analysis.command / run_analysis.sh (macOS, Linux) "
                "or run_analysis.bat (Windows).  See README.txt in that folder.")

        self._run(mx.export_mcx, (self._mcx_folder, self.model, chosen, settings, optics, suite), done)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mri", nargs="+", metavar="NIFTI",
                    help="tissue label image (segmented.nii.gz from spm12_segment_for_NIRS.m) "
                         "or all SPM12 tissue probability maps (c1*.nii ... c6*.nii)")
    ap.add_argument("--lesion", help="lesion mask NIfTI (optional)")
    ap.add_argument("--t1", help="T1 image (optional, for slice display)")
    ap.add_argument("--screenshot", help="render, save a screenshot to this file and exit (testing)")
    ap.add_argument("sessions", nargs="*", help="NIRS_Digitizer .mat files")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    def finish():
        if args.screenshot:
            app.processEvents()
            win.plotter.screenshot(args.screenshot)
            p = Path(args.screenshot)
            win.grab().save(str(p.with_name(p.stem + "_window" + p.suffix)))
            if win.sessions:
                print(pd.DataFrame([{"session": s.name, "ok": s.ok,
                                     "note": s.error or (s.result.note if s.result else "")}
                                    for s in win.sessions.values()]).to_string())
            app.quit()

    def start():
        if args.t1:
            win.load_t1(args.t1)
        if args.sessions:
            win.add_sessions(args.sessions)
        if args.mri:
            win.load_mri(args.mri, lesion=args.lesion, then=finish)
        else:
            finish()

    QtCore.QTimer.singleShot(100, start)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
