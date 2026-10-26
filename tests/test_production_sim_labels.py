import hashlib
import json
import unittest
from pathlib import Path

from plan_sequence import production_sim_labels as production


class ProductionLabelsTest(unittest.TestCase):
    def test_frozen_inventory(self):
        root = Path(__file__).resolve().parents[1]
        prepared = production._read_spec(root / "configs" / "label_production.json")
        jobs = production._read_jobs(prepared)
        self.assertEqual(production._job_counts(jobs)["total"], 2132)
        self.assertEqual(
            production._job_counts(jobs)["per_split"]["training_assembly"],
            {"total": 1894, "small": 1151, "medium": 641, "large": 102},
        )
        self.assertEqual(
            production._job_counts(jobs)["per_split"]["test_assembly"],
            {"total": 238, "small": 20, "medium": 129, "large": 89},
        )
        self.assertEqual(len(prepared["manifest"]["excluded_assemblies"]), 14)
        self.assertFalse(
            {
                (entry["split"], entry["assembly_id"])
                for entry in prepared["manifest"]["excluded_assemblies"]
            }
            & {(row["split"], row["assembly_id"]) for row in jobs}
        )
        self.assertEqual(
            hashlib.sha256(prepared["inventory_path"].read_bytes()).hexdigest(),
            prepared["manifest"]["inventory"]["sha256"],
        )
        self.assertEqual(len(prepared["manifest"]["known_duplicate_groups"]), 8)
        self.assertIn(
            [["training_assembly", "07385"], ["test_assembly", "07386"]],
            [group["members"] for group in prepared["manifest"]["known_duplicate_groups"]],
        )


    def test_tier_manifests_match_the_approved_workers(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads(
            (root / "configs" / "label_production.json").read_text(encoding="utf-8")
        )
        protocols = {tier: {"tier": tier} for tier in production.TIER_SPECS}
        rows = {
            "small": {
                "split": "training_assembly",
                "assembly_id": "1",
                "tier": "small",
                "part_count": 3,
                "input_sha256": "1" * 64,
            },
            "medium": {
                "split": "training_assembly",
                "assembly_id": "2",
                "tier": "medium",
                "part_count": 10,
                "input_sha256": "2" * 64,
            },
            "large": {
                "split": "test_assembly",
                "assembly_id": "3",
                "tier": "large",
                "part_count": 30,
                "input_sha256": "3" * 64,
            },
        }
        small = production._job_manifest(spec, rows["small"], ["0", "1", "2"], protocols)
        medium = production._job_manifest(
            spec, rows["medium"], [str(index) for index in range(10)], protocols
        )
        large = production._job_manifest(
            spec, rows["large"], [str(index) for index in range(30)], protocols
        )
        self.assertEqual(small["parameters"]["max_states"], 7)
        self.assertEqual(small["expected"]["root_action_count"], 54)
        self.assertEqual(medium["parameters"]["candidate_time_budget_s"], 3600.0)
        self.assertEqual(medium["expected"]["state_part_attempt_budget"], 18)
        self.assertEqual(medium["expected"]["candidate_evaluation_budget"], 54)
        self.assertEqual(large["expected"]["state_part_attempt_budget"], 400)
        self.assertEqual(large["expected"]["candidate_evaluation_budget"], 400)


if __name__ == "__main__":
    unittest.main()
