import json
import re
import tempfile
import unittest
from pathlib import Path

from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_sdf_cache as cache


class PrepareSDFCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.assembly = root / "training_assembly" / "00001"
        self.assembly.mkdir(parents=True)
        for part in ("0", "1", "2"):
            (self.assembly / (part + ".obj")).write_text(
                "o part\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
        (self.assembly / "contact_graph.json").write_text("{}\n", encoding="utf-8")
        self.input_sha256 = labels._hash_assembly(self.assembly)[0]
        self.cache_dir = root / "cache" / "sdf" / self.input_sha256

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def fake_simulation(model, _asset_folder):
        for obj_path in re.findall(r'filename="([^"]+[.]obj)"', model):
            Path(obj_path).with_suffix(".sdf").write_bytes(
                ("sdf:" + Path(obj_path).stem).encode("ascii")
            )

    def test_generates_every_part_in_scratch_and_reuses_valid_cache(self):
        before = labels._hash_files(
            self.assembly,
            sorted(path for path in self.assembly.rglob("*") if path.is_file()),
        )[0]
        result = cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            expected_input_sha256=self.input_sha256,
            simulation_factory=self.fake_simulation,
        )
        self.assertEqual(result["schema_version"], cache.SCHEMA_VERSION)
        self.assertEqual(result["part_ids"], ["0", "1", "2"])
        self.assertEqual(result["file_count"], 3)
        self.assertTrue(result["source_directory"]["unchanged"])
        self.assertEqual(
            sorted(path.name for path in self.cache_dir.glob("*.sdf")),
            ["0.sdf", "1.sdf", "2.sdf"],
        )
        after = labels._hash_files(
            self.assembly,
            sorted(path for path in self.assembly.rglob("*") if path.is_file()),
        )[0]
        self.assertEqual(before, after)
        self.assertFalse(any(self.assembly.glob("*.sdf")))

        repeated = cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            expected_input_sha256=self.input_sha256,
            simulation_factory=lambda *_: self.fail("cache should be reused"),
        )
        self.assertTrue(repeated["idempotent"])

    def test_rejects_corrupt_cache_and_wrong_key(self):
        cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            simulation_factory=self.fake_simulation,
        )
        (self.cache_dir / "1.sdf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            cache.validate_cache(self.cache_dir)

        with self.assertRaisesRegex(ValueError, "RedMax"):
            cache.validate_cache(
                self.cache_dir,
                expected_simulator_sha256="0" * 64,
            )

        wrong = self.cache_dir.parent / ("0" * 64)
        with self.assertRaisesRegex(ValueError, "directory name"):
            cache.prepare_cache(
                self.assembly,
                wrong,
                simulation_factory=self.fake_simulation,
            )

    def test_manifest_is_complete_publication_marker(self):
        cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            simulation_factory=self.fake_simulation,
        )
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["input"]["sha256"], self.input_sha256)
        self.assertEqual(
            manifest["cache_key"],
            labels._sha256_bytes(
                labels._canonical_json(manifest["identity"]).encode("utf-8")
            ),
        )


if __name__ == "__main__":
    unittest.main()
