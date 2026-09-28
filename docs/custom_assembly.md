# Preparing a custom assembly

FAST expects one directory per assembly below `assets/`:

```text
assets/<collection>/<assembly-id>/
├── 0.obj
├── 1.obj
├── ...
├── config.json
└── contact_graph.json
```

## Mesh and identifier requirements

- Each part is a triangular Wavefront OBJ mesh named `<part-id>.obj`.
- Part identifiers must be unique decimal integers because the simulation
  runtime converts them to integers for display colors.
- Meshes must be watertight for SDF collision handling. The preprocessing
  command attempts normal and hole repair, then rejects meshes that remain
  non-watertight.
- Each OBJ is expressed in its part-local frame. `config.json` places it in the
  assembled frame.
- The code performs no unit conversion. OBJ vertices and configuration
  translations must use the same unit. Physics uses fixed numerical thresholds
  (`0.05` SDF spacing, `0.1` contact tolerance, and `0.5` separation), so scale
  all meshes and translations together if changing units.

Optional mesh preprocessing:

```bash
python -m assets.process_mesh \
  --source-dir incoming/widget \
  --target-dir assets/custom/widget \
  --subdivide
```

`--rescale FACTOR` multiplies OBJ geometry only; apply the same factor to every
translation in `config.json`. `--recenter` recenters every part independently
and therefore requires corresponding configuration translations.

## Assembly configuration

`config.json` maps string part identifiers to poses. A state is
`[x, y, z, rx, ry, rz]`: translation followed by XYZ Euler angles in radians.
`final_state` is the assembled pose. `initial_state` is optional and is used to
place parts already removed from the current subassembly.

```json
{
  "0": {
    "initial_state": null,
    "final_state": [0, 0, 0, 0, 0, 0]
  },
  "1": {
    "initial_state": [3, 0, 0, 0, 0, 0],
    "final_state": [1, 0, 0, 0, 0, 0]
  }
}
```

The file itself is required by FAST inference. Parts omitted from it use an
identity assembled pose, but explicit entries are recommended.

## Contact graph

Build the contact graph with the native RedMax runtime:

```bash
python -m plan_sequence.assets.build_contact_graph \
  --dir assets/custom/widget --save-json
```

This writes NetworkX node-link JSON. `nodes` contains each part identifier and
`links` contains contacting `source`/`target` pairs. Do not hand-author the graph
unless it uses exactly the same identifiers as the OBJ filenames.

## Generate a sequence

Choose an optional base part that must remain in the assembly, then run with one
process:

```bash
python -m plan_sequence.run_seq_plan \
  --dir custom \
  --id widget \
  --checkpoint checkpoints/pretrained_model_gt.pth \
  --base-part 0 \
  --num-proc 1 \
  --max-gripper 2 \
  --budget 400 \
  --early-term \
  --log-dir outputs/widget
```

Omit `--base-part` when no part is fixed. FAST automatically uses CUDA when
available; RedMax physics and its build remain CPU/native. Rendering is off
unless `--render` is supplied and requires a working OpenGL display.

By default RedMax caches `.sdf` files beside the meshes. Use `--clear-sdf` to
remove them after the run, or `--disable-save-sdf` to avoid caching.

## Outputs

The command above creates:

```text
outputs/widget/dfs-fast/s0/widget/
├── setup.json
├── stats.json
└── tree.pkl
```

In `stats.json`, `success` reports whether a complete path was found and
`sequence` is the disassembly removal order. Reverse that list to obtain the
corresponding assembly order, starting with the final remaining part or the
chosen base part. `tree.pkl` is Python pickle data; only open trees produced by
your own trusted run.

## Common errors

- `Missing contact graph`: run the contact-graph command with `--save-json`.
- `Missing assembly configuration`: add `config.json`, even when every pose is
  the identity transform.
- `Contact graph references missing meshes`: make node IDs and OBJ stems match.
- `non-watertight mesh not fixed`: repair the OBJ in a mesh tool before retrying.
- `No module named redmax_py`: build `./simulation` in the active environment.
- Checkpoint loading failure: delete it and download the matching release asset
  again.
- No sequence within the budget: increase `--budget` or set `--timeout`; this is
  a planner search limit, not a checkpoint-loading problem.
