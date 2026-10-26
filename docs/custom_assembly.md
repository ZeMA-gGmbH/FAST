# Planning a custom assembly

This guide runs FAST with a release checkpoint on your own assembly. All
commands run from the repository root, in the environment of the README, after
RedMax is built and the checkpoints are downloaded. The folders below
`data/` are examples; any path works.

## What FAST needs

One folder per assembly:

```text
data/custom/widget/
├── 0.obj
├── 1.obj
├── ...
├── config.json          optional: assembled poses of parts stored in a local frame
└── contact_graph.json   built in step 3
```

- **One mesh per part.** Each part is a triangle mesh in Wavefront OBJ format,
  named `<id>.obj`. Each `<id>` must be an integer, because the simulator
  derives part colours from it. Parts are rigid; FAST does not model joints,
  springs, or flexible parts.
- **Assembled pose.** Either every OBJ already sits in its assembled position
  (as in the ASAP assemblies, no `config.json`), or `config.json` places each
  part (see below).
- **Watertight meshes.** RedMax uses signed distance fields, which need closed
  meshes. Step 1 repairs small defects and reports meshes it cannot close.
- **Units and size.** The code converts no units. FAST was trained on
  assemblies scaled to a largest dimension of 10, in centimetres, and the
  simulator thresholds are fixed in these units (0.5 separation for a removed
  part, 0.2 drift and 0.1 contact for stability, gravity 9.8). Scale the
  assembly so that its largest dimension is about 10 (step 1).
- **Contacts.** Step 3 computes which parts touch and writes
  `contact_graph.json`. FAST reads the contact graph for its features.
- **Part count.** FAST was trained and evaluated on assemblies with 3 to 70
  parts. It runs on other sizes, but its behaviour there is not tested.

### `config.json`

`config.json` maps each part ID to its states. A state is
`[x, y, z, rx, ry, rz]`: a translation followed by XYZ Euler angles in radians.
`final_state` is the assembled pose. Set `initial_state` to `null`: a removed
part with an `initial_state` stays in the scene at that pose as a fixed
obstacle, whereas the thesis data has no `config.json`, so removed parts leave
the scene.

```json
{
  "0": {"initial_state": null, "final_state": [12.245, 3.8152, 7.0889, 0, 0, 1.570796]},
  "6": {"initial_state": null, "final_state": [16.69, 3.8152, 0.6321, 0, 0, 1.570796]}
}
```

A part missing from `config.json` keeps the pose of its OBJ file.

## Step 1: check and scale the meshes

```bash
python -m assets.process_mesh --source-dir data/custom/widget-raw --target-dir data/custom/widget
```

The command loads every OBJ, closes holes and fixes normals where needed, and
writes the meshes to the target folder. It prints `non-watertight mesh not
fixed` and writes nothing when a mesh stays open; repair that mesh in a mesh
tool first. Options: `--rescale FACTOR` multiplies all vertices,
`--subdivide` splits long edges for denser contact points, and `--recenter`
moves each part to its own centroid.

The command writes OBJ files only. Copy `config.json` yourself. When you use
`--rescale`, multiply the three translations of every state in `config.json` by
the same factor.

Check the size of the assembled product:

```bash
python - data/custom/widget <<'EOF'
import sys
import numpy as np
from assets.load import load_assembly
assembly = load_assembly(sys.argv[1])
vertices = np.vstack([part["mesh"].vertices for part in assembly.values()])
extent = np.ptp(vertices, axis=0)
print("parts:", len(assembly), "largest dimension:", float(extent.max()))
print("watertight:", all(part["mesh"].is_watertight for part in assembly.values()))
EOF
```

If the largest dimension is far from 10, rerun step 1 with
`--rescale 10/<largest dimension>` (the value, for example `--rescale 0.6562`)
and scale `config.json` as described above.

## Step 2: what is computed automatically

- **Stable poses.** For each state, the planner computes at most three stable
  poses of the remaining parts on a horizontal support plane (the most
  probable ones, until their cumulative probability exceeds 0.9).
- **SDF caches.** RedMax writes `<id>.sdf` files next to the meshes on first
  use and reuses them later. Delete them after changing a mesh.

The pose and SDF cache commands in `plan_sequence/` serve the label production
and are not needed here.

## Step 3: build the contact graph

```bash
python -m plan_sequence.assets.build_contact_graph --dir data/custom/widget --save-json
```

This writes `data/custom/widget/contact_graph.json` (NetworkX node-link JSON).
Two parts are in contact when their distance in the assembled pose is below
0.1. A part without any contact is still a node. Check the `links` list: each
pair should be two parts that touch.

## Step 4: plan

```bash
python -m fast.plan --assembly-dir data/custom/widget \
  --checkpoint checkpoints/fast_gt_seed0.pth --normalization checkpoints/normalization.json \
  --output outputs/widget/trace.jsonl
```

`scripts/smoke.sh data/custom/widget outputs` runs the same command and writes
`outputs/smoke/widget.jsonl`. Set
`MPLBACKEND=Agg` without a display. FAST computes the features of each state,
ranks its actions, and validates them in RedMax until one part is left, at most
`8 × (parts − 1)` actions are checked, or two hours have passed
(`--wall-timeout` changes this limit). The command prints a summary:

```json
{
  "solution_action_validation": 6,
  "action_validation_count": 6,
  "stop_reason": "solution_found",
  "budgets": {"1": false, "2": true, "4": true, "8": true}
}
```

`stop_reason` is `solution_found`, `budget_reached`, `frontier_exhausted` (no
feasible action is left), or `wall_timeout`. `budgets` says whether a complete
plan was found within `1, 2, 4, 8 × (parts − 1)` checked actions. Seeds 1 and
2 (`fast_gt_seed1.pth`, `fast_gt_seed2.pth`) are the other thesis checkpoints.

## Step 5: read the sequence

The trace has a header line, one line per checked action, and a summary line;
`trace.jsonl.sha256` holds its hash. Print the plan:

```bash
python - outputs/widget/trace.jsonl <<'EOF'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
summary = rows[-1]
removals = [(step["part"], step["direction"]) for step in summary["transitions"]]
removed = {part for part, _ in removals}
last = [part for part in summary["root_parts"] if part not in removed]
print("complete plan:", summary["solution_action_validation"] is not None)
print("checked actions:", summary["action_validation_count"])
print("removal order (part, direction):", removals)
print("assembly order:", last + [part for part, _ in reversed(removals)])
EOF
```

For the beam example this prints:

```text
complete plan: True
checked actions: 6
removal order (part, direction): [('0', '+Z'), ('2', '+Y'), ('1', '+Z'), ('6', '-Y')]
assembly order: ['3', '6', '1', '2', '0']
```

Each removal moves the part in a straight line along one of `+X, -X, +Y, -Y,
+Z, -Z`. The direction is in the frame of the stable pose of that state, with
`+Z` pointing away from the support plane; each action line of the trace
records the pose (`pose_id`). Assembling inserts the parts in reverse order,
each along the opposite direction. The action lines also show rejected
actions: `R` is the removability check and `T` the stability check of the
remaining parts, each with `label` `true`, `false`, or `null` (unresolved).

## Limits

- Motions are straight lines along six axes. Rotations and motions with
  several segments are not considered.
- Stability may use at most one held part, which with the gripper of the
  removed part means at most two grippers.
- Every accepted step is checked only in simulation (RedMax, ASAP settings,
  gravity about 1% of Earth gravity in centimetre units). A plan is not a
  guarantee for a real robot: grasps, robot reach, and tolerances are not
  checked.
- FAST was trained on Fusion 360 Gallery assemblies with 3 to 70 parts. Other
  part counts, sizes far from 10, and very different shapes are not tested.

## Common errors

| Message | Fix |
|---|---|
| `non-watertight mesh not fixed` | close the mesh in a mesh tool, then rerun step 1 |
| `No such file or directory: '.../contact_graph.json'` | run step 3 |
| `contact graph part IDs differ` | rebuild the contact graph after adding, removing, or renaming parts |
| `invalid literal for int()` | rename the OBJ files to integer IDs |
| `No module named 'redmax_py'` | build `./simulation` in the active environment |
| checkpoint `SHA256SUMS` mismatch or load error | download the release asset again |
