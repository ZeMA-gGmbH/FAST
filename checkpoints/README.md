# FAST checkpoints

The three Graph Transformer checkpoints of the thesis are assets of GitHub
release `v2.0.0`. They are not stored in Git. Download them into this folder
and check them with `sha256sum -c SHA256SUMS`.

| File | Seed | Selected checkpoint | Thesis checkpoint (SHA-256 of the full training file) |
|---|---:|---|---|
| `fast_gt_seed0.pth` | 0 | `epoch-034` | `5f44170fb749f4f71963599bf6a28b613f91d9b79c7df56dfbec394e27a0c1dc` |
| `fast_gt_seed1.pth` | 1 | `epoch-099` | `8ccfee223d9bd854034de14693ca43bde7c67bba0569b61ed3095f460d310e0e` |
| `fast_gt_seed2.pth` | 2 | `epoch-044` | `38cd41a9a98854fc3977aba64104e0189a0d4880bb49a38374f4bb13a396d422` |

Each release file holds only the 21 model tensors (90,142 parameters) of the
thesis checkpoint as a state dictionary. Optimizer state, random-number state,
and training provenance are removed, so the file hash differs from the thesis
checkpoint, while every tensor is identical. Load a checkpoint with
`torch.load(path, weights_only=True)` into `fast.ranker_models.build_model("graph_transformer")`.

`normalization.json` holds the per-feature mean and standard deviation of the
41 part features and 16 action features on the training split. Its canonical
JSON hash, `c81a653f52e617dc4edbdfa9fd536aa6643faeed07828b733f93b368eea9b806`,
is the normalization hash recorded in the thesis training audit.
