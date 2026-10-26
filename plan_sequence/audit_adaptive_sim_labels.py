"""Audit runtime-adaptive FAST labels without treating outcomes as corruption."""

import argparse
import sys
from pathlib import Path

from plan_sequence import audit_budgeted_sim_labels as budgeted_audit
from plan_sequence import generate_adaptive_sim_labels as adaptive
from plan_sequence import generate_sim_labels as labels


SCHEMA_VERSION = "fast.adaptive-label-audit.v1"


def audit(manifest_path, source_root, output_root):
    prepared = adaptive._validate_manifest(manifest_path, source_root, output_root)
    return budgeted_audit.audit_prepared(prepared, SCHEMA_VERSION)


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
