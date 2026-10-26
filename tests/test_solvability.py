import importlib.util
import json
import re
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("locked Torch runtime is unavailable")

from fast import solvability as solvability


def _summary(solution_at, max_checks=24):
    return {"solution_action_validation": solution_at, "max_action_validations": max_checks,
            "budgets": {"8": {"success": solution_at is not None and solution_at <= max_checks}}}


def _records(solved_by_assembly):
    records = []
    for assembly_id, (tier, solved_trace) in solved_by_assembly.items():
        for index in range(18):
            records.append((assembly_id, tier, _summary(5 if index == solved_trace else None)))
    return records


class SolvabilityTest(unittest.TestCase):
    def test_aggregate_only_output_and_independent_audit_agree(self):
        records = _records({"secret-A": ("small", 17), "secret-B": ("small", None), "secret-C": ("large", 0)})
        result = solvability.aggregate(records)
        self.assertEqual(tuple(result), solvability.OUTPUT_FIELDS)
        self.assertEqual((result["demonstrated_solvable"], result["denominator"], result["not_demonstrated"]), (2, 3, 1))
        self.assertEqual(result["by_tier"]["small"]["demonstrated_solvable"], 1)
        self.assertEqual(result, solvability.aggregate(records, independent=True))
        self.assertIsNone(re.search("secret", json.dumps(result)))

    def test_incomplete_or_inconsistent_evidence_is_rejected(self):
        records = _records({"A": ("small", None)})
        with self.assertRaisesRegex(Exception, "six methods x three seeds"):
            solvability.aggregate(records[:-1])
        records[0] = ("A", "large", records[0][2])
        with self.assertRaisesRegex(Exception, "tier differs"):
            solvability.aggregate(records)

if __name__ == "__main__":
    unittest.main()
