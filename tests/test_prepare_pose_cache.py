import json
import tempfile
import unittest
from pathlib import Path

from plan_sequence import generate_sim_labels as labels
from plan_sequence import prepare_pose_cache as cache


POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class PreparePoseCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.assembly = root / "training_assembly" / "00001"
        self.assembly.mkdir(parents=True)
        for part in ("0", "1"):
            (self.assembly / (part + ".obj")).write_text(
                "o part\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
        (self.assembly / "contact_graph.json").write_text("{}\n", encoding="utf-8")
        self.input_sha256 = labels._hash_assembly(self.assembly)[0]
        self.cache_dir = root / "cache" / "stable-poses" / self.input_sha256

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def poses(*_):
        return [(POSE, 0.75), (POSE, 0.25)]

    def test_publishes_valid_cache_and_reuses_it(self):
        before = labels._hash_assembly(self.assembly)[0]
        result = cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            expected_input_sha256=self.input_sha256,
            max_poses=2,
            probability_threshold=0.9,
            pose_factory=self.poses,
        )
        self.assertEqual(result["schema_version"], cache.SCHEMA_VERSION)
        self.assertEqual(result["pose_count"], 2)
        self.assertEqual(result["poses"][0]["pose"], [list(row) for row in POSE])
        self.assertEqual(sorted(path.name for path in self.cache_dir.iterdir()), ["manifest.json"])
        self.assertEqual(before, labels._hash_assembly(self.assembly)[0])

        repeated = cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            max_poses=2,
            probability_threshold=0.9,
            pose_factory=lambda *_: self.fail("valid cache should be reused"),
        )
        self.assertTrue(repeated["idempotent"])

    def test_rejects_changed_identity_and_corrupt_pose(self):
        cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            max_poses=2,
            pose_factory=self.poses,
        )
        with self.assertRaisesRegex(ValueError, "probability threshold"):
            cache.validate_cache(
                self.cache_dir, expected_probability_threshold=0.8
            )

        manifest_path = self.cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["poses"][0]["pose"] = [[1.0]]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matrix"):
            cache.validate_cache(self.cache_dir)

    def test_identity_fallback_is_explicit(self):
        result = cache.prepare_cache(
            self.assembly,
            self.cache_dir,
            max_poses=1,
            pose_factory=lambda *_: [],
        )
        self.assertEqual(
            result["poses"],
            [{
                "index": 0,
                "pose_id": "pose-000",
                "pose": None,
                "probability": None,
                "source": "identity_fallback",
            }],
        )


if __name__ == "__main__":
    unittest.main()
