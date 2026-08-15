import unittest

import numpy as np

from jrdb_sphere.geometry import (
    enforce_contract,
    observed_row_bounds,
    output_synthetic_mask,
    proxy_generation_mask,
    proxy_to_output,
    seam_repair_blend,
    source_to_proxy,
    tile_starts,
    verify_frame,
)


class GeometryTest(unittest.TestCase):
    def test_720_geometry_has_exact_120_row_caps(self):
        self.assertEqual((120, 600), observed_row_bounds(720, 60.0, -60.0))

    def test_high_resolution_geometry_is_derived_from_latitude(self):
        self.assertEqual((171, 853), observed_row_bounds(1024, 60.0, -60.0))

    def test_proxy_round_trip_restores_observed_source_exactly(self):
        source = np.arange(480 * 1440 * 3, dtype=np.uint32).reshape(480, 1440, 3)
        source = np.mod(source, 251).astype(np.uint8)
        proxy = source_to_proxy(source, 1440, 720)
        self.assertTrue(np.array_equal(proxy[120:600], source))
        output = proxy_to_output(proxy, source, 720)
        self.assertTrue(np.array_equal(output[120:600], source))
        self.assertEqual([], verify_frame(output, source, 120, 720))

    def test_contract_only_restores_source(self):
        source = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
        generated = np.full((8, 8, 3), 77, dtype=np.uint8)
        result = enforce_contract(generated, source, 2)
        self.assertEqual([], verify_frame(result, source, 2, 8))
        self.assertTrue(np.all(result[:2] == 77))
        self.assertTrue(np.all(result[6:] == 77))

    def test_masks_only_select_synthetic_latitudes(self):
        proxy_mask = proxy_generation_mask(8, 720)
        output_mask = output_synthetic_mask(8, 720)
        self.assertTrue(np.array_equal(proxy_mask, output_mask))
        self.assertTrue(np.all(proxy_mask[:120] == 255))
        self.assertTrue(np.all(proxy_mask[120:600] == 0))
        self.assertTrue(np.all(proxy_mask[600:] == 255))

    def test_seam_repair_does_not_touch_observed_band(self):
        primary = np.zeros((8, 8, 3), dtype=np.uint8)
        rolled = np.full_like(primary, 200)
        mask = np.full_like(primary, 255)
        mask[2:6] = 0
        result = seam_repair_blend(primary, rolled, mask)
        self.assertTrue(np.all(result[2:6] == 0))
        self.assertTrue(np.any(result[:2] != 0))

    def test_tiles_cover_panorama_for_legacy_baseline(self):
        starts = tile_starts(3760, 1024, 256)
        covered = np.zeros(3760, dtype=bool)
        for start in starts:
            covered[np.arange(start, start + 1024) % 3760] = True
        self.assertTrue(covered.all())


if __name__ == "__main__":
    unittest.main()
