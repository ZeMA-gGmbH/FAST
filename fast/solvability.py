"""Aggregate-only demonstrated-solvability lower bound over the primary traces of a planning study.

An assembly is demonstrated_solvable if at least one of its 18 primary traces (six methods x
three seeds) holds a complete mandatory-RedMax-validated plan within the 8*(N-1) budget.
This is a lower bound under finite evidence, not a feasibility ceiling.
"""

import argparse
import json
from pathlib import Path

from fast import complete_case, graph_contract
from fast.planner_integration import METHODS
from plan_sequence import production_sim_labels as production


MANIFEST_SCHEMA = "fast.solvability-input-manifest.v1"
OUTPUT_SCHEMA = "fast.solvability-lower-bound.v1"
OUTPUT_FIELDS = ("schema_version", "estimand", "demonstrated_solvable", "denominator",
                 "proportion", "not_demonstrated", "by_tier")
SEEDS = (0, 1, 2)
TIERS = ("small", "medium", "large")
_require = graph_contract._require


def _trace(run_root, method, seed, assembly_id):
    return Path(run_root) / "traces" / "primary" / method / "seed-{}".format(seed) / (assembly_id + ".jsonl")


def manifest(run_root, excluded=()):
    """Bind every expected trace to its digest sidecar; no trace content is read."""
    entries = []
    for row in complete_case.population(run_root, excluded):
        for method in METHODS:
            for seed in SEEDS:
                trace = _trace(run_root, method, seed, row["assembly_id"])
                digest = Path(str(trace) + ".sha256").read_text(encoding="ascii").split()[0]
                entries.append({"assembly_id": row["assembly_id"], "tier": row["tier"], "method": method,
                                "seed": seed, "trace_sha256": digest})
    return {"schema_version": MANIFEST_SCHEMA, "excluded_assemblies": sorted(excluded),
            "entry_count": len(entries), "entries": entries, "outcomes_accessed": False}


def _summary(run_root, entry):
    trace = _trace(run_root, entry["method"], entry["seed"], entry["assembly_id"])
    _require(production._sha256_file(trace) == entry["trace_sha256"], "planning-study trace digest differs")
    lines = trace.read_text(encoding="utf-8").splitlines()
    header, summary = json.loads(lines[0]), json.loads(lines[-1])
    _require(header["method"] == entry["method"] and header["seed"] == entry["seed"]
             and header["assembly_id"] == entry["assembly_id"] and summary["complete"] is True
             and summary["mandatory_redmax_validation"] is True and summary["unvalidated_leaf_shortcut"] is False,
             "planning-study trace contract differs")
    return summary


def aggregate(records, independent=False):
    """records: (assembly_id, tier, summary) for all 18 traces of every assembly; returns aggregate only."""
    by_assembly = {}
    for assembly_id, tier, summary in records:
        if independent:  # audit path: recompute from the first-solution check count, not the budget flag
            solved = (summary["solution_action_validation"] is not None
                      and summary["solution_action_validation"] <= summary["max_action_validations"])
        else:
            solved = summary["budgets"]["8"]["success"] is True
        tier_seen, any_solved, count = by_assembly.get(assembly_id, (tier, False, 0))
        _require(tier_seen == tier, "assembly tier differs")
        by_assembly[assembly_id] = (tier, any_solved or solved, count + 1)
    _require(all(count == len(METHODS) * len(SEEDS) for _, _, count in by_assembly.values()),
             "every assembly needs all six methods x three seeds")
    def cell(values):
        solved = sum(values)
        return {"demonstrated_solvable": solved, "denominator": len(values), "proportion": solved / len(values) if values else None,
                "not_demonstrated": len(values) - solved}
    overall = cell([solved for _, solved, _ in by_assembly.values()])
    return dict(
        {"schema_version": OUTPUT_SCHEMA,
         "estimand": "demonstrated-solvability lower bound under the finite planning-study method/seed/budget evidence; "
                     "not a feasibility ceiling or proof of infeasibility"},
        **overall,
        by_tier={tier: cell([solved for t, solved, _ in by_assembly.values() if t == tier]) for tier in TIERS},
    )


def analyze(run_root, manifest_path, manifest_sha256, output_path, independent=False):
    """Writes aggregate fields only; `audit` recomputes them on an independent path."""
    _require(production._sha256_file(manifest_path) == manifest_sha256, "solvability manifest differs")
    output_path = Path(output_path)
    _require(not output_path.exists(), "solvability output cannot be overwritten")
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["entries"]
    result = aggregate([(e["assembly_id"], e["tier"], _summary(run_root, e)) for e in entries], independent)
    tiers = {e["assembly_id"]: e["tier"] for e in entries}
    _require(result["denominator"] == len(tiers) and tuple(result) == OUTPUT_FIELDS
             and all(c["denominator"] == sum(t == tier for t in tiers.values()) for tier, c in result["by_tier"].items()),
             "solvability output differs")
    output_path.write_text(production._canonical_json(result) + "\n", encoding="utf-8")
    return {"output_sha256": production._sha256_file(output_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("manifest")
    sub.add_argument("--run-root", required=True)
    sub.add_argument("--exclude-assembly", action="append", default=[])
    sub.add_argument("--output", required=True)
    for name in ("analyze", "audit"):
        sub = commands.add_parser(name)
        for flag in ("run-root", "manifest", "manifest-sha256", "output"):
            sub.add_argument("--" + flag, required=True)
    args = parser.parse_args(argv)
    if args.command == "manifest":
        output = Path(args.output)
        _require(not output.exists(), "solvability manifest cannot be overwritten")
        output.write_text(production._canonical_json(manifest(args.run_root, args.exclude_assembly)) + "\n", encoding="utf-8")
        print(production._canonical_json({"manifest_sha256": production._sha256_file(output)}))
    else:
        print(production._canonical_json(analyze(args.run_root, args.manifest, args.manifest_sha256, args.output,
                                                  independent=args.command == "audit")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
