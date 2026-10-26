"""Assembly-complete population selection for the planning study.

An assembly with a technical failure that persisted after its one retry is excluded from every
method, so all comparisons stay paired. In the thesis run, 3 of 237 test assemblies were excluded.
"""

import csv
import hashlib
import json
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def exclusions(primary_retry_jobs, legacy_retry_jobs):
    """Assemblies with a failed primary or legacy-counter attempt after retry."""
    return frozenset(row["assembly_id"] for row in primary_retry_jobs + legacy_retry_jobs)


def retained_jobs(jobs, excluded, primary_methods, legacy_methods):
    retained = [row for row in jobs if row["assembly_id"] not in excluded]
    assemblies = {row["assembly_id"] for row in retained}
    actual = {(row["counter"], row["method"], row["seed"], row["assembly_id"]) for row in retained}
    expected = {
        (counter, method, str(seed), assembly_id)
        for assembly_id in assemblies
        for counter, methods in (("action_validation", primary_methods),
                                 ("asap_legacy_evaluation", legacy_methods))
        for method in methods for seed in (0, 1, 2)
    }
    _require(
        excluded.isdisjoint(assemblies) and len(retained) == len(actual) == len(expected) and actual == expected,
        "assembly-complete population differs",
    )
    return retained, assemblies


def population(run_root, excluded=()):
    """Retained test assemblies of a planning-study run: one FAST seed-0 job row per assembly."""
    run_root = Path(run_root)
    setup = json.loads((run_root / "setup.json").read_text(encoding="utf-8"))
    jobs_path = run_root / setup["job_table"]
    _require(hashlib.sha256(jobs_path.read_bytes()).hexdigest() == setup["job_table_sha256"],
             "planning-study job table differs")
    with jobs_path.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t")
                if row["counter"] == "action_validation" and row["method"] == "fast" and row["seed"] == "0"]
    _require(len(rows) == setup["test_assembly_count"] and set(excluded) <= {row["assembly_id"] for row in rows},
             "planning-study test population differs")
    return sorted((row for row in rows if row["assembly_id"] not in excluded), key=lambda row: row["assembly_id"])
