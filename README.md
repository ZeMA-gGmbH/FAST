# FAST

FAST ranks candidate part removals with a pretrained graph transformer and
passes that ranking to a depth-first sequence planner. RedMax then checks
disassembly motion and intermediate stability. This repository contains
inference and sequence-generation code only.

## Install

The supported build path is Linux with Python 3.8. Install the native build
prerequisites on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libgl1-mesa-dev libglu1-mesa-dev xorg-dev
```

Create and activate an environment:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CPU-only inference, install:

```bash
python -m pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter==2.1.0 torch-sparse==0.6.16 \
  -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
```

For an NVIDIA GPU with a CUDA 11.7-compatible driver, install instead:

```bash
python -m pip install torch==1.13.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install torch-scatter==2.1.0+pt113cu117 \
  torch-sparse==0.6.16+pt113cu117 \
  -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

Then install FAST and build RedMax:

```bash
python -m pip install -e .
python -m pip install --no-build-isolation ./simulation
python -c "import redmax_py; print('RedMax import OK')"
```

## Checkpoint

Download `pretrained_model_gt.pth` from the GitHub Release matching this
repository version:

```bash
curl -L -o checkpoints/pretrained_model_gt.pth \
  https://github.com/ZeMA-gGmbH/FAST/releases/download/v0.1.0/pretrained_model_gt.pth
```

The file is 81,120,721 bytes. It contains model tensors only and is loaded with
PyTorch's `weights_only=True` mode.

## Included example

Build the example contact graph:

```bash
python -m plan_sequence.assets.build_contact_graph \
  --dir assets/beam_assembly/original --save-json
```

Run model-only FAST inference (the ranked choices are printed):

```bash
python -m fast_network.inference \
  --assembly_id original \
  --asset-dir assets/beam_assembly \
  --checkpoint_path checkpoints/pretrained_model_gt.pth
```

Generate a physics-checked sequence:

```bash
python -m plan_sequence.run_seq_plan \
  --dir beam_assembly \
  --id original \
  --checkpoint checkpoints/pretrained_model_gt.pth \
  --base-part 6 \
  --num-proc 1 \
  --max-gripper 2 \
  --budget 400 \
  --early-term \
  --log-dir outputs/beam
```

Results are written under `outputs/beam/dfs-fast/s0/original/`. Read
`stats.json` for success, disassembly sequence, gripper count, and evaluation
count; `setup.json` records the planner settings. `tree.pkl` is the complete
search tree and should only be unpickled when it was produced locally.

See [docs/custom_assembly.md](docs/custom_assembly.md) for the custom-assembly
layout, preprocessing, configuration, and output details.

## Tests

After downloading the checkpoint and building RedMax:

```bash
FAST_CHECKPOINT=checkpoints/pretrained_model_gt.pth python -m unittest discover -s tests -v
```

The tests cover safe checkpoint loading, one bounded model inference, and DFS
sequence construction.

## License and attribution

The code is distributed under the MIT license in [LICENSE](LICENSE). Bundled
third-party components retain their license files; see [NOTICE](NOTICE).
