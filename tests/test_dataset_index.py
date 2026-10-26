import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plan_sequence import build_dataset_index as dataset_index
from plan_sequence import production_sim_labels as production


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(production._canonical_json(value) + "\n", encoding="utf-8")


def _write_candidate(path, assembly_id):
    header = {
        "schema_version": dataset_index.LABEL_SCHEMA_VERSION,
        "record_type": "header",
        "assembly_id": assembly_id,
        "state_id": "state-root",
        "parts": ["0", "1", "2"],
        "parts_after": ["1", "2"],
        "part_move": "0",
        "pose_id": "pose-000",
        "pose_index": 0,
        "pose_probability": 1.0,
        "pose": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    stability = {
        "schema_version": dataset_index.LABEL_SCHEMA_VERSION,
        "record_type": "stability",
        "queried": True,
        "T": False,
        "T_status": "unstable",
        "reason": "max_fix_exceeded",
    }
    actions = []
    for index, (direction_id, direction) in enumerate(dataset_index.DIRECTIONS):
        timed_out = index == 0
        actions.append(
            {
                "schema_version": dataset_index.LABEL_SCHEMA_VERSION,
                "record_type": "action",
                "direction_id": direction_id,
                "direction": direction,
                "R": None if timed_out else index % 2 == 0,
                "R_status": "timeout" if timed_out else "success",
                "R_reason": "timeout" if timed_out else "disassembled",
                "Y": None if timed_out else False,
                "Y_status": "removal_timeout" if timed_out else "unstable",
            }
        )
    summary = {
        "schema_version": dataset_index.LABEL_SCHEMA_VERSION,
        "record_type": "summary",
        "complete": True,
        "record_count": 9,
    }
    records = [header, stability] + actions + [summary]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(production._canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


class DatasetIndexTest(unittest.TestCase):
    def test_frozen_split(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "dataset_index.json").read_text(
                encoding="utf-8"
            )
        )
        prepared = production._read_spec(root / config["production_manifest"])
        jobs = production._read_jobs(prepared)
        assignments = dataset_index._split_jobs(jobs, config)
        self.assertEqual(
            dataset_index._assembly_counts(jobs, assignments),
            config["expected_assemblies"],
        )
        duplicate_splits = {}
        for row in jobs:
            if row["split"] == "training_assembly":
                duplicate_splits.setdefault(row["input_sha256"], set()).add(
                    assignments[(row["split"], row["assembly_id"])]
                )
        self.assertTrue(all(len(splits) == 1 for splits in duplicate_splits.values()))


    def test_build_indexes_only_authoritative_candidates_and_masks_nulls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            data_root = root / "data"
            manifest_path = repository / "configs" / "dataset-index.json"
            production_manifest = repository / "configs" / "production.json"
            exclusions = [
                {
                    "split": "test_assembly",
                    "assembly_id": "00004",
                    "reason": "exact input duplicate of training_assembly/00001",
                }
            ]
            config = {
                "schema_version": dataset_index.CONFIG_SCHEMA_VERSION,
                "phase": "test-index",
                "production_manifest": "configs/production.json",
                "combined_audit": "runs/production/audit.json",
                "output_root": "datasets/index",
                "validation": {
                    "fraction": 0.1,
                    "seed": 0,
                    "group_by": "input_sha256",
                    "part_count_bins": [
                        [3, 4],
                        [5, 6],
                        [7, 10],
                        [11, 15],
                        [16, 20],
                        [21, 30],
                        [31, 40],
                        [41, 70],
                    ],
                },
                "primary_test_exclusions": exclusions,
                "expected_assemblies": {
                    "train": {"total": 1, "small": 1, "medium": 0, "large": 0},
                    "validation": {
                        "total": 1,
                        "small": 1,
                        "medium": 0,
                        "large": 0,
                    },
                    "test": {"total": 1, "small": 1, "medium": 0, "large": 0},
                    "excluded": {
                        "total": 1,
                        "small": 1,
                        "medium": 0,
                        "large": 0,
                    },
                },
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            _write_json(production_manifest, {})

            hashes = {"00001": "1" * 64, "00002": "2" * 64, "00003": "3" * 64}
            hashes["00004"] = hashes["00001"]
            jobs = [
                {
                    "split": "training_assembly" if index < 3 else "test_assembly",
                    "assembly_id": assembly_id,
                    "tier": "small",
                    "part_count": 3,
                    "input_sha256": hashes[assembly_id],
                }
                for index, assembly_id in enumerate(("00001", "00002", "00003", "00004"), 1)
            ]
            jobs[2]["split"] = "test_assembly"
            production_prepared = {
                "manifest_path": production_manifest,
                "manifest_sha256": "a" * 64,
                "inventory_sha256": "b" * 64,
                "manifest": {
                    "paths": {
                        "labels_root": "labels/production",
                        "audit_root": "runs/production/assemblies",
                    },
                    "excluded_assemblies": [],
                },
            }
            combined = {
                "schema_version": production.AUDIT_SCHEMA_VERSION,
                "status": "passed",
                "integrity_status": "passed",
                "manifest_sha256": "a" * 64,
                "inventory_sha256": "b" * 64,
                "audit_count": 4,
                "excluded_assemblies": [],
                "unseen_test_exclusions": exclusions,
            }
            _write_json(data_root / "runs/production/audit.json", combined)

            for row in jobs:
                assembly_root = (
                    data_root / "labels/production" / row["split"] / row["assembly_id"]
                )
                candidate_path = assembly_root / "state-root/pose-000/part-0.jsonl"
                _write_candidate(candidate_path, row["assembly_id"])
                _write_candidate(
                    assembly_root / "state-orphan/pose-000/part-1.jsonl",
                    row["assembly_id"],
                )
                summary = {
                    "schema_version": dataset_index.LABEL_SCHEMA_VERSION,
                    "complete": True,
                    "input": {"sha256": row["input_sha256"]},
                    "accounting": {"candidate_count": 1},
                    "candidate_files": [
                        {
                            "path": "state-root/pose-000/part-0.jsonl",
                            "sha256": production._sha256_file(candidate_path),
                            "state_id": "state-root",
                            "pose_id": "pose-000",
                            "part_move": "0",
                        }
                    ],
                }
                summary_path = assembly_root / "summary.json"
                _write_json(summary_path, summary)
                audit_path = (
                    data_root
                    / "runs/production/assemblies"
                    / row["split"]
                    / (row["assembly_id"] + ".json")
                )
                _write_json(
                    audit_path,
                    {
                        "schema_version": dataset_index.AUDIT_SCHEMAS["small"],
                        "status": "passed",
                        "integrity_status": "passed",
                        "completion_sha256": production._sha256_file(summary_path),
                        "coverage": {"complete_sequence_found": True},
                        "orphan_state_checkpoints": [],
                    },
                )

            with mock.patch.object(
                dataset_index.production, "_read_spec", return_value=production_prepared
            ), mock.patch.object(dataset_index.production, "_read_jobs", return_value=jobs):
                result = dataset_index.build(manifest_path, data_root)
                repeated = dataset_index.build(manifest_path, data_root)

            self.assertEqual(result["artifacts"]["assemblies"]["row_count"], 4)
            self.assertEqual(result["artifacts"]["candidates"]["row_count"], 3)
            self.assertEqual(result["artifacts"]["actions"]["row_count"], 18)
            self.assertTrue(repeated["idempotent"])

            output = data_root / "datasets/index"
            candidates = dataset_index._read_jsonl(output / "candidates.jsonl")
            actions = dataset_index._read_jsonl(output / "actions.jsonl")
            profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
            self.assertNotIn("00004", {row["assembly_id"] for row in candidates})
            self.assertFalse(actions[0]["R_valid"])
            self.assertFalse(actions[0]["Y_valid"])
            self.assertEqual(profile["overall"]["labels"]["R"]["null"], 3)
            self.assertEqual(profile["overall"]["labels"]["Y"]["null"], 3)
            self.assertEqual(profile["overall"]["statuses"]["R"]["timeout"], 3)
            self.assertEqual(profile["overall"]["reasons"]["T"]["max_fix_exceeded"], 3)
            self.assertEqual(
                profile["overall"]["label_rates"]["R"][
                    "positive_fraction_among_valid"
                ],
                0.4,
            )

            assemblies = dataset_index._read_jsonl(output / "assemblies.jsonl")
            self.assertFalse({"audit", "summary"} & set(assemblies[0]) or "source" in candidates[0])
            for row in (row for row in assemblies if row["included"]):
                labels_dir = data_root / "labels/production" / row["source_split"] / row["assembly_id"]
                summary = json.loads((labels_dir / "summary.json").read_text(encoding="utf-8"))
                expected = [dataset_index._validate_candidate(labels_dir / entry["path"], entry["sha256"],
                                                              {"assembly_id": row["assembly_id"]}, entry)
                            for entry in summary["candidate_files"]]
                actual = dataset_index.index_candidates(output, row)
                self.assertEqual(len(actual), len(expected))
                for (header, stability, actions_), (raw_header, raw_stability, raw_actions) in zip(actual, expected):
                    self.assertEqual(header, {key: raw_header[key] for key in header})
                    self.assertEqual((stability["T"], stability["queried"]), (raw_stability.get("T"), raw_stability.get("queried")))
                    for key in ("direction_id", "direction", "R", "Y"):
                        self.assertEqual([action[key] for action in actions_], [action.get(key) for action in raw_actions])
            with self.assertRaisesRegex(ValueError, "index candidates differ"):
                dataset_index.index_candidates(output, dict(assemblies[0], candidate_count=99))


if __name__ == "__main__":
    unittest.main()
