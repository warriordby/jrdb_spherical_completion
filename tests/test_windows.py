import unittest

from jrdb_sphere.windows import make_windows


class WindowTest(unittest.TestCase):
    def test_243_frames_use_deterministic_81_17_windows(self):
        windows = make_windows(243, 81, 17)
        self.assertEqual([0, 64, 128, 192], [window.start for window in windows])
        self.assertEqual([81, 81, 81, 51], [window.valid_frames for window in windows])
        self.assertEqual([0, 17, 17, 17], [window.overlap_with_previous for window in windows])
        self.assertEqual(81, len(windows[-1].padded_indices))
        self.assertEqual(242, windows[-1].padded_indices[-1])

    def test_short_sequence_is_padded_without_changing_indices(self):
        window = make_windows(3, 81, 17)[0]
        self.assertEqual([0, 1, 2], window.padded_indices[:3])
        self.assertTrue(all(index == 2 for index in window.padded_indices[3:]))

    def test_frame_count_must_be_4n_plus_1(self):
        with self.assertRaises(ValueError):
            make_windows(100, 80, 17)


if __name__ == "__main__":
    unittest.main()
