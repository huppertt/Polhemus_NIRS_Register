"""Verify that the distributable assets work without local demo/subject data."""
import unittest

import numpy as np
from scipy.io import whosmat

import digitizer_registration as dr
import mcx_export as mx
from probe_registration_gui import TEMPLATE_FILE


class RuntimeAssetTests(unittest.TestCase):
    def test_atlas_contains_only_template_and_atlas_fields(self):
        names = {name for name, shape, dtype in whosmat(TEMPLATE_FILE)}
        metadata = {"fid_name", "fid_xyz", "opt_name", "opt_type",
                    "opt_xyz", "opt2d", "link"}
        geometry = {name for name in names
                    if (name.startswith("nodes") and name[5:].isdigit())
                    or (name.startswith("faces") and name[5:].isdigit())}
        self.assertEqual(names, metadata | geometry)
        atlas = dr.load_atlas(TEMPLATE_FILE)
        self.assertGreater(len(atlas.layers), 0)
        self.assertGreater(len(atlas.probe), 0)
        self.assertGreater(len(atlas.link), 0)
        self.assertTrue(atlas.fiducials)
        for nodes, faces in atlas.layers:
            self.assertTrue(np.isfinite(nodes).all())
            self.assertGreaterEqual(faces.min(), 0)
            self.assertLess(faces.max(), len(nodes))

    def test_optics_load_at_default_wavelengths(self):
        optics = mx.TissueOptics().table([780, 850])
        self.assertEqual(len(optics), 14)
        for wavelength in (780, 850):
            rows = optics[optics.wavelength_nm == wavelength]
            self.assertEqual(set(rows.label), set(range(7)))
            self.assertTrue(np.isfinite(rows[["mua", "mus", "g", "n"]]).all().all())


if __name__ == "__main__":
    unittest.main()
