import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_pipeline/scripts'))
from image_geometry import original_to_prediction, prediction_to_original


class CropGeometryTests(unittest.TestCase):
    def test_square_crop_is_not_stretched(self):
        camera = {'width': 1024, 'height': 1024}
        self.assertEqual(original_to_prediction(511.5, 511.5, camera, 512, 384), (255.5, 191.5))
        self.assertEqual(prediction_to_original(0, 0, camera, 512, 384), (.5, 128.5))

    def test_roundtrip_non_multiple_of_patch(self):
        camera = {'width': 1920, 'height': 1080}
        x, y = np.array([10., 959.5, 1900]), np.array([45., 539.5, 1000])
        mx, my = original_to_prediction(x, y, camera, 512, 288)
        ox, oy = prediction_to_original(mx, my, camera, 512, 288)
        np.testing.assert_allclose(ox, x)
        np.testing.assert_allclose(oy, y)
