"""
"Export to MCExtreme" settings dialog for probe_registration_gui.py.

Collects the export folder, the sessions to simulate, the wavelengths and the
tissue optical properties (from the NIRS-toolbox, editable) and the MCX
simulation options, pre-filled with the MCX / MCX-CL defaults
(mcx_export.McxSettings).
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy import QtCore, QtGui, QtWidgets

import mcx_export as mx

# rough MCX-CL throughput on an Apple M4 Pro GPU (colin27 benchmark); for the time estimate only
PHOTONS_PER_MS = 3000.0
OUTPUT_TYPES = [("x", "x - flux (fluence rate), MCX default"), ("f", "f - fluence"), ("e", "e - energy deposit")]
OUTPUT_FORMATS = [("jnii", "jnii - JNIfTI (MCX default)"), ("bnii", "bnii - binary JNIfTI"),
                  ("nii", "nii - NIfTI"), ("mc2", "mc2 - raw float")]
ENGINES = [("auto", "auto - MCX (CUDA) if it works, else MCX-CL (OpenCL)"),
           ("mcxcl", "mcxcl - OpenCL (Apple/AMD/Intel/NVIDIA)"), ("mcx", "mcx - CUDA (NVIDIA only)")]
OPT_COLS = ["wavelength_nm", "label", "tissue", "model", "mua", "mus", "g", "n", "musp"]
OPT_HEAD = ["λ (nm)", "Label", "Tissue", "Model", "μa (1/mm)", "μs (1/mm)", "g", "n", "μs′ (1/mm)"]
EDITABLE = {"mua", "mus", "g", "n"}


def _parse_list(text):
    return [float(t) for t in text.replace(";", ",").replace(" ", ",").split(",") if t.strip()]


def _fmt_time(sec):
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    return f"{sec / 3600:.1f} h"


class McxExportDialog(QtWidgets.QDialog):
    """Settings form; after ``exec_()`` use ``folder``, ``selected_sessions``, ``settings`` and ``optics``."""

    def __init__(self, sessions: dict, folder="", has_lesion=False, mcx_found=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export to MCExtreme (MCX / MCX-CL)")
        self.sessions = sessions            # name -> registered probe dataframe
        self.optics_src = mx.TissueOptics()
        self._loading = False
        d = mx.McxSettings()
        lay = QtWidgets.QVBoxLayout(self)

        # ---- folder and sessions
        box = QtWidgets.QGroupBox("Export")
        g = QtWidgets.QGridLayout(box)
        g.addWidget(QtWidgets.QLabel("Folder"), 0, 0)
        self.ed_folder = QtWidgets.QLineEdit(str(folder))
        self.ed_folder.setPlaceholderText("choose an (empty) export folder…")
        g.addWidget(self.ed_folder, 0, 1)
        b = QtWidgets.QPushButton("Browse…")
        b.clicked.connect(self.browse)
        g.addWidget(b, 0, 2)
        g.addWidget(QtWidgets.QLabel("Sessions"), 1, 0, QtCore.Qt.AlignTop)
        self.lst_sessions = QtWidgets.QListWidget()
        for name in sessions:
            it = QtWidgets.QListWidgetItem(name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.lst_sessions.addItem(it)
        self.lst_sessions.setMaximumHeight(90)
        self.lst_sessions.itemChanged.connect(self.update_summary)
        g.addWidget(self.lst_sessions, 1, 1, 1, 2)
        note = ("MCX found: " + str(mcx_found)) if mcx_found else \
            "MCX / MCX-CL not found on this computer (the run scripts search for it when run)."
        lbl = QtWidgets.QLabel(note)
        lbl.setStyleSheet("color: gray")
        g.addWidget(lbl, 2, 1, 1, 2)
        lay.addWidget(box)

        cols = QtWidgets.QHBoxLayout()
        lay.addLayout(cols, 1)

        # ---- optical properties
        box = QtWidgets.QGroupBox("Wavelengths and tissue optical properties (NIRS-toolbox nirs.media.tissues)")
        v = QtWidgets.QVBoxLayout(box)
        f = QtWidgets.QFormLayout()
        self.ed_wl = QtWidgets.QLineEdit(", ".join(f"{w:g}" for w in d.wavelengths))
        self.ed_wl.setToolTip("Comma-separated wavelengths in nm (NIRx: 780, 850). "
                              f"Tabulated range {self.optics_src.wavelength_range[0]:.0f}-"
                              f"{self.optics_src.wavelength_range[1]:.0f} nm.")
        f.addRow("Wavelengths (nm)", self.ed_wl)
        h = QtWidgets.QHBoxLayout()
        self.sp_so2 = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0, maximum=1, singleStep=0.05, value=d.brain_so2)
        self.sp_hbt = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0, maximum=500, singleStep=5, value=d.brain_hbt_uM)
        h.addWidget(QtWidgets.QLabel("SO₂"))
        h.addWidget(self.sp_so2)
        h.addWidget(QtWidgets.QLabel("HbT (µM)"))
        h.addWidget(self.sp_hbt)
        h.addStretch(1)
        f.addRow("Brain (gray, white, lesion)", h)
        v.addLayout(f)
        self.tbl = QtWidgets.QTableWidget(0, len(OPT_COLS))
        self.tbl.setHorizontalHeaderLabels(OPT_HEAD)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.itemChanged.connect(self._cell_changed)
        v.addWidget(self.tbl, 1)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Reset from NIRS-toolbox")
        b.setToolTip("Recompute the table from the wavelengths, SO₂ and HbT (discards edits)")
        b.clicked.connect(self.refresh_optics)
        h.addWidget(b)
        lbl = QtWidgets.QLabel("μa, μs, g, n are editable (MCX uses μs; μs′ = μs(1−g)). Skin/bone tables are "
                               "μs′ in the toolbox. Lesion = brain"
                               + ("" if has_lesion else " (no lesion loaded)") + ".")
        lbl.setStyleSheet("color: gray")
        h.addWidget(lbl, 1)
        v.addLayout(h)
        cols.addWidget(box, 3)

        # ---- simulation options
        right = QtWidgets.QVBoxLayout()
        box = QtWidgets.QGroupBox("Simulation")
        f = QtWidgets.QFormLayout(box)
        self.ed_photons = QtWidgets.QLineEdit(f"{d.photons:g}")
        self.ed_photons.setValidator(QtGui.QDoubleValidator(1, 1e12, 0))
        self.ed_photons.setToolTip("Photons per simulation (-n); MCX examples use 1e8")
        f.addRow("Photons per simulation", self.ed_photons)
        self.cmb_launch = QtWidgets.QComboBox()
        self.cmb_launch.addItem("Sources and detectors (adjoint)", "both")
        self.cmb_launch.addItem("Sources only", "sources")
        self.cmb_launch.setToolTip("Launching from detectors too gives the adjoint fields needed for "
                                   "channel sensitivity (Jacobian) maps")
        f.addRow("Launch from", self.cmb_launch)
        self.sp_detr = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0.5, maximum=10, singleStep=0.5,
                                                value=d.detector_radius_mm, suffix=" mm")
        f.addRow("Detector radius", self.sp_detr)
        self.sp_t0 = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0, maximum=100, value=d.t0 * 1e9, suffix=" ns")
        self.sp_t1 = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=100, value=d.t1 * 1e9, suffix=" ns")
        self.sp_dt = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=100, value=d.dt * 1e9, suffix=" ns")
        h = QtWidgets.QHBoxLayout()
        for w in (self.sp_t0, self.sp_t1, self.sp_dt):
            h.addWidget(w)
        f.addRow("T0 / T1 / Δt", h)
        self.ed_seed = QtWidgets.QLineEdit(str(d.seed))
        self.ed_seed.setValidator(QtGui.QIntValidator(1, 2 ** 31 - 1))
        f.addRow("RNG seed (+k per run)", self.ed_seed)
        self.chk_reflect = QtWidgets.QCheckBox("Fresnel reflection at boundaries (-b 1)")
        self.chk_reflect.setChecked(d.reflect)
        f.addRow(self.chk_reflect)
        self.chk_norm = QtWidgets.QCheckBox("Normalize output (-U 1)")
        self.chk_norm.setChecked(d.normalize)
        f.addRow(self.chk_norm)
        self.sp_close = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0, maximum=10, singleStep=0.5,
                                                 value=d.close_scalp_mm, suffix=" mm")
        self.sp_close.setToolTip("Radius of a ball closing of the scalp: fills holes, cracks and crevices narrower "
                                 "than ~2× this so photons are not lost in air gaps (tissue is never removed). "
                                 "Floating tissue specks are removed and enclosed air is always filled.")
        f.addRow("Close scalp holes (radius)", self.sp_close)
        self.sp_margin = QtWidgets.QDoubleSpinBox(decimals=0, minimum=0, maximum=50, singleStep=5,
                                                  value=d.margin_mm, suffix=" mm")
        self.sp_margin.setToolTip("Pad the grid with air so the head is at least this far from its edges")
        f.addRow("Air margin around head", self.sp_margin)
        right.addWidget(box)

        box = QtWidgets.QGroupBox("Output")
        f = QtWidgets.QFormLayout(box)
        self.chk_fluence = QtWidgets.QCheckBox("Save volume (-S 1)")
        self.chk_fluence.setChecked(d.save_fluence)
        f.addRow(self.chk_fluence)
        self.cmb_otype = QtWidgets.QComboBox()
        for k, t in OUTPUT_TYPES:
            self.cmb_otype.addItem(t, k)
        f.addRow("Output type (-O)", self.cmb_otype)
        self.cmb_ofmt = QtWidgets.QComboBox()
        for k, t in OUTPUT_FORMATS:
            self.cmb_ofmt.addItem(t, k)
        f.addRow("Output format (-F)", self.cmb_ofmt)
        self.chk_det = QtWidgets.QCheckBox("Save detected photons (-d 1)")
        self.chk_det.setChecked(d.save_detected)
        f.addRow(self.chk_det)
        self.ed_fields = QtWidgets.QLineEdit(d.detected_fields)
        self.ed_fields.setToolTip("-w flags: D detector ID, S scattering events, P partial path lengths, "
                                  "M momentum transfer, X exit position, V exit direction, W initial weight")
        f.addRow("Detected fields (-w)", self.ed_fields)
        self.ed_maxdet = QtWidgets.QLineEdit(f"{d.max_detected:g}")
        self.ed_maxdet.setValidator(QtGui.QDoubleValidator(1, 1e9, 0))
        self.ed_maxdet.setToolTip("-H; the MCX default (1e6) overflows for short-separation detectors at 1e8 photons")
        f.addRow("Max detected photons (-H)", self.ed_maxdet)
        right.addWidget(box)

        box = QtWidgets.QGroupBox("Run scripts")
        f = QtWidgets.QFormLayout(box)
        self.cmb_engine = QtWidgets.QComboBox()
        for k, t in ENGINES:
            self.cmb_engine.addItem(t, k)
        f.addRow("Engine", self.cmb_engine)
        self.ed_gpu = QtWidgets.QLineEdit(d.gpu)
        self.ed_gpu.setPlaceholderText("auto")
        self.ed_gpu.setToolTip("-G: device number (1 = first GPU) or a mask such as 1101; empty lets MCX choose")
        f.addRow("GPU (-G)", self.ed_gpu)
        right.addWidget(box)
        right.addStretch(1)
        cols.addLayout(right, 2)

        # ---- summary and buttons
        self.lbl_summary = QtWidgets.QLabel()
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                                        | QtWidgets.QDialogButtonBox.RestoreDefaults)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        lay.addWidget(bb)

        for w in (self.ed_wl,):
            w.editingFinished.connect(self.refresh_optics)
        for w in (self.sp_so2, self.sp_hbt):
            w.valueChanged.connect(self.refresh_optics)
        for w in (self.ed_photons, self.ed_wl):
            w.textChanged.connect(self.update_summary)
        self.cmb_launch.currentIndexChanged.connect(self.update_summary)
        self.chk_fluence.toggled.connect(self._toggle)
        self.chk_det.toggled.connect(self._toggle)
        self.refresh_optics()
        self._toggle()
        self.resize(1250, 760)

    # ---------------- helpers ----------------
    def browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "MCX export folder", self.ed_folder.text() or str(Path.home()))
        if d:
            self.ed_folder.setText(d)

    def _toggle(self, *_):
        for w in (self.cmb_otype, self.cmb_ofmt):
            w.setEnabled(self.chk_fluence.isChecked())
        for w in (self.ed_fields, self.ed_maxdet):
            w.setEnabled(self.chk_det.isChecked())

    def wavelengths(self):
        return _parse_list(self.ed_wl.text())

    def refresh_optics(self, *_):
        try:
            wl = self.wavelengths()
            if not wl:
                raise ValueError("Enter at least one wavelength")
            tab = self.optics_src.table(wl, self.sp_so2.value(), self.sp_hbt.value())
        except ValueError as e:
            self.ed_wl.setStyleSheet("background: #ffd0d0")
            self.ed_wl.setToolTip(str(e))
            return
        self.ed_wl.setStyleSheet("")
        self._loading = True
        self.tbl.setRowCount(len(tab))
        for i, r in enumerate(tab.itertuples(index=False)):
            for j, c in enumerate(OPT_COLS):
                val = getattr(r, c)
                txt = f"{val:g}" if c in ("wavelength_nm", "label") else \
                    (f"{val:.6g}" if c in EDITABLE or c == "musp" else str(val))
                it = QtWidgets.QTableWidgetItem(txt)
                if c not in EDITABLE:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#555")))
                self.tbl.setItem(i, j, it)
        self._loading = False
        self.update_summary()

    def _cell_changed(self, item):
        if self._loading:
            return
        try:
            v = float(item.text())
            ok = v >= 0
        except ValueError:
            ok = False
        item.setBackground(QtGui.QBrush(QtGui.QColor("white" if ok else "#ffd0d0")))
        r = item.row()
        try:
            musp = float(self.tbl.item(r, OPT_COLS.index("mus")).text()) * \
                (1 - float(self.tbl.item(r, OPT_COLS.index("g")).text()))
            txt = f"{musp:.6g}"
        except (ValueError, AttributeError):
            txt = "?"
        cell = self.tbl.item(r, OPT_COLS.index("musp"))
        if cell is not None:
            self._loading = True
            cell.setText(txt)
            self._loading = False

    def optics(self) -> pd.DataFrame:
        rows = []
        for i in range(self.tbl.rowCount()):
            r = {c: self.tbl.item(i, j).text() for j, c in enumerate(OPT_COLS)}
            for c in ("wavelength_nm", *EDITABLE):
                try:
                    r[c] = float(r[c])
                except ValueError:
                    raise ValueError(f"Invalid {c} '{r[c]}' for {r['tissue']} at {r['wavelength_nm']} nm")
                if r[c] < 0:
                    raise ValueError(f"Negative {c} for {r['tissue']} at {r['wavelength_nm']} nm")
            r["label"] = int(float(r["label"]))
            r["musp"] = r["mus"] * (1 - r["g"])
            rows.append(r)
        return pd.DataFrame(rows, columns=OPT_COLS)

    def selected_sessions(self):
        return [self.lst_sessions.item(i).text() for i in range(self.lst_sessions.count())
                if self.lst_sessions.item(i).checkState() == QtCore.Qt.Checked]

    @property
    def folder(self):
        return self.ed_folder.text().strip()

    def settings(self) -> mx.McxSettings:
        return mx.McxSettings(
            wavelengths=self.wavelengths(),
            photons=float(self.ed_photons.text() or 0),
            simulate=self.cmb_launch.currentData(),
            save_fluence=self.chk_fluence.isChecked(),
            output_type=self.cmb_otype.currentData(),
            output_format=self.cmb_ofmt.currentData(),
            save_detected=self.chk_det.isChecked(),
            detected_fields=self.ed_fields.text().strip().upper() or "DP",
            max_detected=int(float(self.ed_maxdet.text() or 1e6)),
            reflect=self.chk_reflect.isChecked(),
            normalize=self.chk_norm.isChecked(),
            t0=self.sp_t0.value() * 1e-9, t1=self.sp_t1.value() * 1e-9, dt=self.sp_dt.value() * 1e-9,
            seed=int(self.ed_seed.text() or mx.MCX_SEED),
            detector_radius_mm=self.sp_detr.value(),
            close_scalp_mm=self.sp_close.value(), margin_mm=self.sp_margin.value(),
            brain_so2=self.sp_so2.value(), brain_hbt_uM=self.sp_hbt.value(),
            engine=self.cmb_engine.currentData(),
            gpu=self.ed_gpu.text().strip(),
        )

    def update_summary(self, *_):
        try:
            s = self.settings()
            ses = {k: self.sessions[k] for k in self.selected_sessions()}
            n, tot = mx.plan(ses, s)
        except (ValueError, KeyError):
            self.lbl_summary.setText("")
            return
        per = 0
        if ses:
            p = next(iter(ses.values()))
            per = int((p.Type == "Source").sum()) + (int((p.Type == "Detector").sum()) if s.simulate == "both" else 0)
        self.lbl_summary.setText(
            f"<b>{n} simulations</b> ({len(ses)} session(s) × {len(s.wavelengths)} wavelength(s) × {per} launch "
            f"optodes), {tot:.3g} photons in total.  Estimated run time ≈ {_fmt_time(tot / PHOTONS_PER_MS / 1000)} "
            f"on an Apple M4 Pro GPU with MCX-CL (~{PHOTONS_PER_MS:g} photons/ms; NVIDIA GPUs with MCX are "
            f"typically faster, CPUs 50-100× slower).")

    def restore_defaults(self):
        d = mx.McxSettings()
        self.ed_wl.setText(", ".join(f"{w:g}" for w in d.wavelengths))
        self.sp_so2.setValue(d.brain_so2)
        self.sp_hbt.setValue(d.brain_hbt_uM)
        self.ed_photons.setText(f"{d.photons:g}")
        self.cmb_launch.setCurrentIndex(self.cmb_launch.findData(d.simulate))
        self.sp_detr.setValue(d.detector_radius_mm)
        self.sp_t0.setValue(d.t0 * 1e9)
        self.sp_t1.setValue(d.t1 * 1e9)
        self.sp_dt.setValue(d.dt * 1e9)
        self.ed_seed.setText(str(d.seed))
        self.chk_reflect.setChecked(d.reflect)
        self.chk_norm.setChecked(d.normalize)
        self.sp_close.setValue(d.close_scalp_mm)
        self.sp_margin.setValue(d.margin_mm)
        self.chk_fluence.setChecked(d.save_fluence)
        self.cmb_otype.setCurrentIndex(self.cmb_otype.findData(d.output_type))
        self.cmb_ofmt.setCurrentIndex(self.cmb_ofmt.findData(d.output_format))
        self.chk_det.setChecked(d.save_detected)
        self.ed_fields.setText(d.detected_fields)
        self.ed_maxdet.setText(f"{d.max_detected:g}")
        self.cmb_engine.setCurrentIndex(self.cmb_engine.findData(d.engine))
        self.ed_gpu.setText(d.gpu)
        self.refresh_optics()

    # ---------------- validation ----------------
    def validate(self):
        """Return an error message, or '' when the form is valid."""
        try:
            s = self.settings()
            self.optics()
        except ValueError as e:
            return str(e)
        if not s.wavelengths:
            return "Enter at least one wavelength."
        lo, hi = self.optics_src.wavelength_range
        bad = [w for w in s.wavelengths if not lo < w < hi]
        if bad:
            return f"Wavelength(s) {bad} outside the tabulated range {lo:.0f}-{hi:.0f} nm."
        tab_wl = set(np.round(self.optics().wavelength_nm.unique(), 6))
        if tab_wl != set(np.round(s.wavelengths, 6)):
            return "The optical property table does not match the wavelengths; press 'Reset from NIRS-toolbox'."
        if s.photons < 1:
            return "Number of photons must be positive."
        if not self.selected_sessions():
            return "Select at least one session."
        if not s.save_fluence and not s.save_detected:
            return "Nothing to save: enable the volume and/or detected photons."
        if s.t1 <= s.t0 or s.dt <= 0 or s.dt > s.t1 - s.t0 + 1e-15:
            return "Time window: need T1 > T0 and 0 < Δt ≤ T1 - T0."
        if s.save_detected and not set(s.detected_fields) <= set(mx.DETFLAG_BITS):
            return f"Detected fields must be letters from {''.join(mx.DETFLAG_BITS)}."
        if s.gpu and not s.gpu.isdigit():
            return "GPU must be a device number or a 0/1 mask (e.g. 1 or 1101)."
        return ""

    def accept(self):
        err = self.validate()
        if err:
            QtWidgets.QMessageBox.warning(self, "Export to MCExtreme", err)
            return
        if not self.folder:
            self.browse()
            if not self.folder:
                return
        p = Path(self.folder)
        if p.exists() and any(p.iterdir()):
            r = QtWidgets.QMessageBox.question(
                self, "Export to MCExtreme",
                f"{p} is not empty.\nSimulation inputs will be overwritten and the exported simulations "
                "marked as not run (old results are replaced when the scripts run).\n\nExport here anyway?")
            if r != QtWidgets.QMessageBox.Yes:
                return
        super().accept()
