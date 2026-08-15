import unittest

from jrdb_sphere.data import shift_json_y


class AnnotationTest(unittest.TestCase):
    def test_shift_visible_2d_annotations(self):
        source = {
            "bbox": [10, 20, 30, 40],
            "keypoints": [1, 2, 2, 0, 0, 0],
            "segmentation": [[1, 2, 3, 4]],
            "box_3d": [1, 2, 3],
        }
        result = shift_json_y(source, 120)
        self.assertEqual([10, 140, 30, 40], result["bbox"])
        self.assertEqual([1, 122, 2, 0, 0, 0], result["keypoints"])
        self.assertEqual([[1, 122, 3, 124]], result["segmentation"])
        self.assertEqual([1, 2, 3], result["box_3d"])


if __name__ == "__main__":
    unittest.main()
