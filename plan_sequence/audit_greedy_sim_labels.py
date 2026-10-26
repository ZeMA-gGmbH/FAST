"""Audit ASAP-compatible DFS + heur-out labels."""

import argparse
import sys
from pathlib import Path

from plan_sequence import audit_budgeted_sim_labels as shared_audit
from plan_sequence import generate_greedy_sim_labels as greedy
from plan_sequence import generate_sim_labels as labels


SCHEMA_VERSION = "fast.greedy-label-audit.v1"


def audit(manifest_path, source_root, output_root):
    prepared = greedy._validate_manifest(manifest_path, source_root, output_root)
    return shared_audit.audit_prepared(prepared, SCHEMA_VERSION)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report = audit(args.manifest, args.source_root, args.output_root)
    print(labels._canonical_json(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
