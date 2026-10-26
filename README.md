# FAST 2.0

FAST plans the disassembly, and so the assembly, of a product made of rigid
parts. A Graph Transformer on the contact graph ranks every action, one part
moved along one of six directions, by its predicted feasibility. A depth-first
planner tries the actions in this order. Every accepted removal is checked in
the RedMax simulator of ASAP: the part must come free (removability, R), and
the remaining parts must stay supported with at most one held part (stability,
T).

This is FAST 2.0, the version of the master's thesis "Feasibility Aware
Assembly Sequence Planning using Graph Transformer" (Saarland University,
2026). It contains everything needed to rerun the thesis: label generation,
the dataset index, features, models, training, calibration, the planner with
its baselines, and the evaluation and analysis code.

## Versions

| Version | Publication | Model | Code |
|---|---|---|---|
| [`v0.1.0`](https://github.com/ZeMA-gGmbH/FAST/tree/v0.1.0) | CATS 2026 article and IROS 2026 paper | part ranker inside ASAP's part-level search | inference and sequence generation |
| `v2.0.0` (this version) | master's thesis, 2026 | action ranker `Y = σ(R)·σ(T)` with mandatory RedMax validation of every accepted action | full reproduction path |

Differences from `v0.1.0`:

- FAST ranks actions (part and direction), not parts, and factorizes joint
  feasibility into removability and stability heads.
- The inputs are 57 CAD features (41 per part, 16 per action) on the contact
  graph, computed before any simulation.
- FAST is trained on new simulation labels for 2,131 ASAP assemblies, published
  as a separate dataset.
- The planner validates every accepted action in RedMax. It is compared with
  five baselines: ASAP random, ASAP outside-in, ASAP's released GATv2, a
  Cebulla-style GCN, and outside-in with a direction prior.
- The runtime is PyTorch 2.4.1 with PyTorch Geometric 2.6.1.

### Papers based on `v0.1.0`

The IROS 2026 paper and the CATS 2026 article are based on `v0.1.0`, not on
this version. To use that code and its checkpoint:

```bash
git clone --branch v0.1.0 https://github.com/ZeMA-gGmbH/FAST.git FAST-v0.1.0
curl -L -o FAST-v0.1.0/checkpoints/pretrained_model_gt.pth \
  https://github.com/ZeMA-gGmbH/FAST/releases/download/v0.1.0/pretrained_model_gt.pth
```

The release page is <https://github.com/ZeMA-gGmbH/FAST/releases/tag/v0.1.0>;
`v0.1.0`'s README describes its installation (PyTorch 1.13.1).

## Repository layout

| Path | Content |
|---|---|
| `fast/` | features, graph records, models, training, validation, PointNet extension, planner, evaluation and analyses |
| `plan_sequence/` | label search per size tier, label audits, dataset index builder, ASAP planner and simulation interface |
| `simulation/` | RedMax (C++ with Python bindings), built locally |
| `assets/`, `utils/` | ASAP helpers: mesh loading and processing, rendering, parallel jobs |
| `configs/` | frozen specifications of the thesis run (labels, index, features, models, PointNet, ablations, test quality) |
| `environment/` | locked environments: `cu121` (thesis training and planning), `cu128` (retraining on RTX 50-series GPUs) |
| `checkpoints/` | checksums and feature normalization of the released FAST checkpoints |
| `scripts/` | local runners: `smoke.sh` (one assembly), `pipeline.sh` (all stages) |
| `assets/beam_assembly/` | small example assembly for a first run |
| `docs/reproduction.md` | how each thesis table and figure is produced |
| `docs/custom_assembly.md` | how to plan your own assembly with a release checkpoint |
| `tests/` | unit and contract tests; no simulator run, data, or GPU needed |

## Install

Linux and Python 3.8. Install the build prerequisites of RedMax (Ubuntu):

```bash
sudo apt-get install build-essential cmake libgl1-mesa-dev libglu1-mesa-dev xorg-dev
```

Create an environment and install FAST:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
python -m pip install -e .
python -m pip install --no-build-isolation ./simulation   # FAST_BUILD_JOBS=2 limits parallel compiles (default 8)
python -c "import redmax_py; print('RedMax import OK')"
```

The thesis used the locked CUDA 12.1 environment in
`environment/cu121` (`uv sync --project environment/cu121`), which training
checks against. Set `MPLBACKEND=Agg` on machines without a display.

## Checkpoints

The three FAST checkpoints of the thesis (Graph Transformer, seeds 0, 1, 2, at
their selected epochs) are assets of release `v2.0.0`. Each file is a
weights-only state dictionary without training metadata, loaded with
`torch.load(..., weights_only=True)`.

```bash
for seed in 0 1 2; do
  curl -L -o checkpoints/fast_gt_seed$seed.pth \
    https://github.com/ZeMA-gGmbH/FAST/releases/download/v2.0.0/fast_gt_seed$seed.pth
done
(cd checkpoints && sha256sum -c SHA256SUMS)
```

`checkpoints/normalization.json` holds the feature means and standard
deviations from the training split. All three checkpoints use it.

## Data

- **Geometry**: the ASAP training and test assemblies (Fusion 360 Gallery
  assemblies processed by ASAP). Download `training_assembly.zip` and
  `test_assembly.zip` from the [ASAP repository](https://github.com/yunshengtian/ASAP)
  and unpack both into one folder, `$FAST_SOURCE_ROOT`. Use the original
  meshes, not the subdivided ones. The meshes and contact graphs of exactly the
  labelled assemblies are also on Hugging Face as
  [`jaybhagiya/FAST-assembly-meshes`](https://huggingface.co/datasets/jaybhagiya/FAST-assembly-meshes);
  its `place_meshes.py` unpacks them into `$FAST_SOURCE_ROOT` and checks each
  `input_sha256`.
- **Labels and splits**: the Hugging Face dataset
  [`jaybhagiya/FAST-assembly-labels`](https://huggingface.co/datasets/jaybhagiya/FAST-assembly-labels)
  (CC BY-NC 4.0 plus the Fusion 360 Gallery conditions). Restore the index
  that the code reads:

```bash
python -m pip install "huggingface_hub>=0.30" pyarrow==17.0.0
hf download jaybhagiya/FAST-assembly-labels --repo-type dataset --local-dir data/hf-labels
python data/hf-labels/convert_index.py restore --dataset-dir data/hf-labels \
  --index-dir "$FAST_DATA_ROOT/datasets/fast-labels-v1"
```

The restore checks every file against the SHA-256 in `manifest.json`
(`b57465c0…`), which the feature and model specifications pin. Each assembly
row has an `input_sha256` of its ASAP input files, which FAST checks before
use.

## Plan one assembly

```bash
scripts/smoke.sh assets/beam_assembly/original outputs                   # included example, no download
scripts/smoke.sh "$FAST_SOURCE_ROOT/training_assembly/00014" outputs     # an ASAP assembly
# or: python -m fast.plan --assembly-dir <assembly> --checkpoint checkpoints/fast_gt_seed0.pth \
#       --normalization checkpoints/normalization.json --output outputs/trace.jsonl
```

This computes the features of each visited state, ranks its actions with FAST,
and validates them in RedMax until one part is left or `8(N−1)` actions were
checked. The trace (JSON Lines and a `.sha256` sidecar) records every checked
action with its R and T outcome, and the summary records success at the
budgets `1, 2, 4, 8 × (N−1)`. `--method` also runs the baselines
(`asap_random`, `asap_heur_out`, `heur_out_direction_prior` with
`--direction-prior`, `asap_released_gatv2` with `--released-checkpoint`, and
`cebulla_gcn` or `mlp_check` with their own checkpoint). RedMax writes SDF
caches next to the meshes.

On one CPU, the included beam assembly takes about 1.5 minutes and 0.5 GB of
memory, and the five-part assembly `00014` takes 4 to 9 minutes and up to 1 GB.
Large assemblies take up to the two-hour planning limit.

To plan your own assembly (mesh requirements, scaling, contact graph, reading
the sequence), see [docs/custom_assembly.md](docs/custom_assembly.md).

## Reproduce the thesis

`scripts/pipeline.sh <stage>` runs each stage locally and sequentially. Set
`FAST_SOURCE_ROOT` (ASAP assemblies) and `FAST_DATA_ROOT` (all outputs).

| Stage | What it does |
|---|---|
| `labels` | SDF caches, label search per size tier, per-assembly and combined audits |
| `index` | dataset index with the train/validation/test splits |
| `features` | the 57 CAD features of every candidate and action |
| `screen` | validation-only feature-group screen (gradient boosting, logistic regression) |
| `graphs` | graph records and the training-split normalization |
| `ranker` | training of MLP, GNN, GATv2, Graph Transformer, and direct-Y (3 seeds each), selection, calibration, bootstrap |
| `pointnet` | point clouds, PointNet extension and Cebulla-style GCN, representation decision |
| `planner-prep` | direction prior, development smoke on a training assembly, RedMax readiness |
| `study-setup`, `study-run`, `study-audit` | six methods × three seeds on the 237 test assemblies, paired bootstrap with Holm correction |

The released dataset replaces the `labels` and `index` stages: the `features`
and `graphs` stages read the candidates and actions from the index.

The specifications in `configs/` are the record of the thesis run. Besides
fixed settings, they pin the SHA-256 of their inputs. Inputs that are files of
this repository (configs, lock file) and the dataset manifest match. Pins of
generated artifacts, for example `feature_build_sha256` or
`ranker_training_audit_sha256`, are the thesis values. A new run produces new
artifacts with timestamps, so update each pin with the hash that the previous
stage reports before you start the next one. Test-set access is locked: the
planning study only runs after a registry is frozen (`planner_integration
freeze-registry`). The study runs 6,399 jobs of up to two hours each. The
thesis ran them on a cluster.

The analyses after the study, how each thesis table and figure is computed,
and the environment variables of individual modules are described in
[docs/reproduction.md](docs/reproduction.md).

## Tests

```bash
MPLBACKEND=Agg python -m unittest discover -s tests
```

The tests use synthetic fixtures. They need neither the dataset nor a GPU nor
a simulator run. One test skips without `wandb`.

## Citation

```bibtex
@mastersthesis{bhagiya2026fast,
  author = {Bhagiya, Jaykumar},
  title  = {Feasibility Aware Assembly Sequence Planning using Graph Transformer},
  school = {Saarland University},
  year   = {2026}
}

@article{xu2026cats,
  author  = {Xu, Xiaomei and Bhagigya, Jaykumar and Schneider, Marco and M{\"u}ller, Rainer},
  title   = {A Physical Feasibility-Aware Sequence Planner for Robotic Assembly-Picking Systems},
  journal = {Procedia CIRP},
  volume  = {147},
  pages   = {266--271},
  year    = {2026},
  doi     = {10.1016/j.procir.2026.02.149}
}

@inproceedings{xu2026iros,
  author    = {Xu, Xiaomei and Bhagiya, Jaykumar and Schneider, Marco and M{\"u}ller, Rainer},
  title     = {{FAST}: Action-Aware Feasibility in Assembly Sequence Planning using {Graph Transformer}},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  note      = {Accepted, to appear}
}
```

Please also cite ASAP (Tian et al., ICRA 2024), whose simulator, assets, and
search FAST builds on.

## License

The code is distributed under the MIT license in [LICENSE](LICENSE). Bundled
third-party components keep their licenses; see [NOTICE](NOTICE). The dataset
and the ASAP assemblies have their own terms.
