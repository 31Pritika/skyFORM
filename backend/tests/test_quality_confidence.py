import struct
import tempfile
import unittest
from pathlib import Path
from app.services.quality_service import load_sparse_confidence


class SparseConfidenceTests(unittest.TestCase):
    def test_tracks_stay_aligned_and_scores_reflect_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sfm/sparse/0/points3D.bin'
            path.parent.mkdir(parents=True)
            with path.open('wb') as stream:
                stream.write(struct.pack('<Q', 2))
                for point_id, error, length in [(1, 0.0, 6), (2, 2.0, 2)]:
                    stream.write(struct.pack('<QdddBBBd', point_id, point_id, 2, 3, 255, 0, 0, error))
                    stream.write(struct.pack('<Q', length))
                    stream.write(struct.pack('<ii', 1, 0) * length)
            points = load_sparse_confidence(Path(directory))
            self.assertEqual(len(points), 2)
            self.assertEqual(points[1]['position'], [2, 2, 3])
            self.assertEqual(points[0]['confidence'], 1)
            self.assertAlmostEqual(points[1]['confidence'], 1 / 6)

    def test_missing_and_truncated_data(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_sparse_confidence(Path(directory)), [])
            path = Path(directory) / 'sfm/sparse/0/points3D.bin'
            path.parent.mkdir(parents=True)
            path.write_bytes(struct.pack('<Q', 1) + b'broken')
            self.assertEqual(load_sparse_confidence(Path(directory)), [])


if __name__ == '__main__':
    unittest.main()

class QualityAvailabilityTests(unittest.TestCase):
    def test_quality_works_before_final_report_and_uses_selected_model(self):
        import json
        from app.services.quality_service import get_quality_metrics
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'outputs').mkdir()
            (base / 'outputs/pipeline_state.json').write_text(json.dumps({'video_id': 'job'}))
            model = base / 'outputs/jobs/job/sfm/sparse/2'
            model.mkdir(parents=True)
            (model / 'images.bin').write_bytes(struct.pack('<Q', 12))
            (model / 'points3D.bin').write_bytes(struct.pack('<Q', 1) +
                struct.pack('<QdddBBBd', 1, 1., 2., 3., 255, 0, 0, .5) +
                struct.pack('<Q', 3) + struct.pack('<ii', 1, 0) * 3)
            result = get_quality_metrics(base)
            self.assertTrue(result['available'])
            self.assertEqual(len(result['points']), 1)
