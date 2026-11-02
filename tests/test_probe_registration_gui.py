"""Regression tests for GUI worker delivery (no 3D render window required)."""
import threading
import time
import unittest
from unittest.mock import patch

from qtpy import QtCore, QtWidgets

from probe_registration_gui import MainWindow, build_model


class WorkerWindow(MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.status = self.statusBar()
        self._threads = []


class GuiWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = WorkerWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def wait_for_workers(self):
        deadline = time.monotonic() + 10
        while self.window._threads and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(self.window._threads, "Worker did not finish and clean up")
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.assertIsNone(self.app.overrideCursor())

    def test_success_and_progress_are_delivered_on_gui_thread(self):
        main_thread = threading.get_ident()
        progress_threads = []
        results = []
        self.window.status.messageChanged.connect(
            lambda message: progress_threads.append((message, threading.get_ident())))

        def work(progress):
            progress("Working")
            return threading.get_ident()

        self.window._run(work, (), lambda result: results.append(
            (result, threading.get_ident())))
        self.wait_for_workers()
        self.assertEqual(len(results), 1)
        worker_thread, callback_thread = results[0]
        self.assertNotEqual(worker_thread, main_thread)
        self.assertEqual(callback_thread, main_thread)
        self.assertEqual(progress_threads, [("Working", main_thread)])

    def test_failure_dialog_is_delivered_on_gui_thread(self):
        main_thread = threading.get_ident()
        errors = []

        def work(progress):
            raise ValueError("Invalid MRI input")

        def critical(parent, title, message):
            errors.append((parent, title, message, threading.get_ident()))

        with patch.object(QtWidgets.QMessageBox, "critical", side_effect=critical):
            self.window._run(work, (), lambda result: self.fail("Unexpected success"))
            self.wait_for_workers()
        self.assertEqual(len(errors), 1)
        parent, title, message, callback_thread = errors[0]
        self.assertIs(parent, self.window)
        self.assertEqual(title, "Error")
        self.assertIn("ValueError: Invalid MRI input", message)
        self.assertEqual(callback_thread, main_thread)
        self.assertEqual(self.window.status.currentMessage(), "Error")

    def test_completion_can_start_next_job(self):
        main_thread = threading.get_ident()
        callbacks = []

        def work(progress):
            return threading.get_ident()

        def second(result):
            callbacks.append(threading.get_ident())

        def first(result):
            callbacks.append(threading.get_ident())
            self.window._run(work, (), second)

        self.window._run(work, (), first)
        self.wait_for_workers()
        self.assertEqual(callbacks, [main_thread, main_thread])

    def test_mri_loading_error_preserves_traceback(self):
        errors = []
        missing = str(__file__) + ".missing.nii.gz"
        with patch.object(QtWidgets.QMessageBox, "critical",
                          side_effect=lambda parent, title, message: errors.append(message)):
            self.window._run(build_model, ([missing], None), lambda result: self.fail(
                "Missing MRI unexpectedly loaded"))
            self.wait_for_workers()
        self.assertEqual(len(errors), 1)
        self.assertIn("Traceback", errors[0])
        self.assertIn("FileNotFoundError", errors[0])
        self.assertIn(missing, errors[0])


if __name__ == "__main__":
    unittest.main()
