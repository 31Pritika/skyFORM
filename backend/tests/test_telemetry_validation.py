import importlib.util
import os
from pathlib import Path
import unittest

os.environ.setdefault('SKYFORM_JOB_ID', 'unit-test')
path = Path(__file__).resolve().parents[1] / 'new_pipeline/scripts/10_validate_telemetry.py'
spec = importlib.util.spec_from_file_location('telemetry_validation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TelemetryValidationTests(unittest.TestCase):
    def row(self, timestamp=0, latitude=12):
        return dict(timestamp=str(timestamp), latitude=str(latitude), longitude='77', altitude='100')

    def test_valid_samples(self):
        module.validate_rows([self.row(0), self.row(1), self.row(2)])

    def test_invalid_coordinates_and_nonfinite(self):
        for latitude in [91, float('nan')]:
            with self.assertRaises(RuntimeError):
                module.validate_rows([self.row(latitude=latitude)])

    def test_timestamps_must_increase(self):
        with self.assertRaises(RuntimeError):
            module.validate_rows([self.row(1), self.row(1)])
