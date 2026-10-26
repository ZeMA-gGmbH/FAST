import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fast import complete_case


PRIMARY_METHODS = ("asap_random", "asap_heur_out", "asap_released_gatv2",
                   "cebulla_gcn", "heur_out_direction_prior", "fast")
LEGACY_METHODS = PRIMARY_METHODS[:3]


class PlanningStudyCompleteCaseTest(unittest.TestCase):
    def setUp(self):
        self.jobs = []
        for assembly_index in range(237):
            assembly_id = "{:05d}".format(assembly_index)
            for counter, methods in (("action_validation", PRIMARY_METHODS),
                                     ("asap_legacy_evaluation", LEGACY_METHODS)):
                for method in methods:
                    for seed in (0, 1, 2):
                        self.jobs.append({"counter": counter, "method": method, "seed": str(seed),
                                          "assembly_id": assembly_id})
        self.primary_retry = [row for row in self.jobs
                              if row["counter"] == "action_validation"
                              and row["assembly_id"] == "00000"][:8]
        self.legacy_retry = (
            [row for row in self.jobs if row["counter"] == "asap_legacy_evaluation"
             and row["assembly_id"] == "00000"]
            + [row for row in self.jobs if row["counter"] == "asap_legacy_evaluation"
               and row["assembly_id"] == "00001"][:7]
            + [row for row in self.jobs if row["counter"] == "asap_legacy_evaluation"
               and row["assembly_id"] == "00002"][:1]
        )

    def test_selects_one_complete_234_assembly_population(self):
        excluded = complete_case.exclusions(self.primary_retry, self.legacy_retry)
        retained, assemblies = complete_case.retained_jobs(
            self.jobs, excluded, PRIMARY_METHODS, LEGACY_METHODS
        )
        self.assertEqual(excluded, frozenset(("00000", "00001", "00002")))
        self.assertEqual((len(assemblies), len(retained)), (234, 6318))
        self.assertEqual(sum(row["counter"] == "action_validation" for row in retained), 4212)
        self.assertEqual(sum(row["counter"] == "asap_legacy_evaluation" for row in retained), 2106)

    def test_rejects_incomplete_population(self):
        excluded = complete_case.exclusions(self.primary_retry, self.legacy_retry)
        with self.assertRaisesRegex(ValueError, "assembly-complete population differs"):
            complete_case.retained_jobs(
                self.jobs[:-1], excluded, PRIMARY_METHODS, LEGACY_METHODS
            )

    def test_population_reads_one_fast_seed_zero_row_per_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ("process", "counter", "method", "seed", "assembly_id", "tier", "part_count", "input_sha256")
            with (root / "jobs.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for process, row in enumerate(self.jobs[:400]):
                    writer.writerow(dict(row, process=process, tier="small", part_count=3, input_sha256="0" * 64))
            assemblies = {row["assembly_id"] for row in self.jobs[:400]}
            (root / "setup.json").write_text(json.dumps({
                "job_table": "jobs.tsv", "test_assembly_count": len(assemblies),
                "job_table_sha256": hashlib.sha256((root / "jobs.tsv").read_bytes()).hexdigest()}), encoding="utf-8")
            rows = complete_case.population(root, {"00000"})
            self.assertEqual([row["assembly_id"] for row in rows], sorted(assemblies - {"00000"}))
            with self.assertRaisesRegex(ValueError, "test population differs"):
                complete_case.population(root, {"unknown"})


if __name__ == "__main__":
    unittest.main()
