import tempfile
import unittest
from pathlib import Path

from fast import ablation_closure as closure
from fast import ablation_common as common


def _ids():
    return ["{:05d}".format(index) for index in range(20000, 20000 + closure.SMALL_VALIDATION_COUNT)]


def _audit(extra=()):
    inventory = [{"assembly_id": value, "dataset_split": "validation", "source_split": "training_assembly", "tier": "small"} for value in _ids()]
    inventory += [{"assembly_id": "3{:04d}".format(index), "dataset_split": "validation", "source_split": "training_assembly", "tier": "medium"} for index in range(76)]
    inventory += [{"assembly_id": "4{:04d}".format(index), "dataset_split": "train", "source_split": "training_assembly", "tier": "small"} for index in range(1702)]
    inventory.append({"assembly_id": closure.REPLAY_FIXTURE, "dataset_split": "train", "source_split": "training_assembly", "tier": "small"})
    inventory += list(extra)
    return {"status": "passed", "test_rows_loaded": 0, "inventory": inventory}


def _listing(requests, extra=None):
    listing = {}
    for namespace, _item, kind, path in requests:
        if kind == "file":
            listing[(namespace, path)] = 10
        elif kind == "tree":
            listing[(namespace, path + "/summary.json")] = 10
            listing[(namespace, path + "/state-a/pose-000/part-0.jsonl")] = 10
        else:
            for name in ("0.obj", "1.obj", "0.sdf", "contact_graph.json", "notes.txt"):
                listing[(namespace, path + "/" + name)] = 10
    listing.update(extra or {})
    return listing


class AblationClosureTest(unittest.TestCase):
    def test_small_validation_ids_come_only_from_development_audit(self):
        self.assertEqual(closure.small_validation_ids(_audit()), _ids())
        with self.assertRaisesRegex(ValueError, "non-development"):
            closure.small_validation_ids(_audit([{"assembly_id": "99999", "dataset_split": "test", "source_split": "test_assembly", "tier": "small"}]))

    def test_geometry_rule_keeps_obj_and_contact_graph_but_excludes_sdf(self):
        requests = closure.closure_requests(_ids())
        files, excluded = closure.select_files(requests, _listing(requests))
        geometry = [row["path"] for row in files if row["namespace"] == "asap-data"]
        self.assertTrue(all(path.endswith((".obj", "contact_graph.json")) for path in geometry))
        self.assertEqual(len(geometry), 3 * (closure.SMALL_VALIDATION_COUNT + 1))
        self.assertEqual(excluded, {".sdf": closure.SMALL_VALIDATION_COUNT + 1, ".txt": closure.SMALL_VALIDATION_COUNT + 1})

    def test_missing_requested_file_fails_closed(self):
        requests = closure.closure_requests(_ids())
        listing = _listing(requests)
        del listing[("fast-data", closure.REPLAY_TRACES[0])]
        with self.assertRaisesRegex(ValueError, "absent"):
            closure.select_files(requests, listing)

    def test_test_bearing_paths_are_rejected(self):
        requests = closure.closure_requests(_ids())
        listing = _listing(requests, {("fast-data", closure.GRAPH_ROOT + "/shards/test/test_assembly/00000.jsonl"): 1})
        with self.assertRaisesRegex(ValueError, "test-bearing"):
            closure.select_files(requests, listing)
        with self.assertRaisesRegex(ValueError, "test-bearing"):
            common.require_development_path("labels/fast-labels-v1/test_assembly/00000/summary.json")

    def test_unexpected_label_payload_is_rejected(self):
        requests = closure.closure_requests(_ids())
        label = closure.LABEL_ROOT + "/" + _ids()[0] + "/0.sdf"
        with self.assertRaisesRegex(ValueError, "unexpected label payload"):
            closure.select_files(requests, _listing(requests, {("fast-data", label): 1}))

    def test_wrong_assembly_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            closure.closure_requests(_ids()[:-1])

    def test_local_listing_rejects_symlinks_and_reports_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fast-data" / "a").mkdir(parents=True)
            (root / "fast-data" / "a" / "x.json").write_text("{}", encoding="utf-8")
            self.assertEqual(set(closure._local_files(root)), {("fast-data", "a/x.json")})
            (root / "fast-data" / "link.json").symlink_to(root / "fast-data" / "a" / "x.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                closure._local_files(root)

    def test_write_json_once_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            common.write_json_once(path, {"a": 1})
            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                common.write_json_once(path, {"a": 2})
            self.assertEqual(common.read_json(path), {"a": 1})


if __name__ == "__main__":
    unittest.main()
