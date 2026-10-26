"""Named action-feature masks and the masked Graph Transformer for the component-ablation D variants.

One mask mechanism serves D0/D1/D2. Indices are derived by name from the frozen scalar-feature
action schema and the frozen component-ablation spec, and fail closed if names, groups, or order differ.
The model keeps the frozen module construction order, so D0 reproduces the frozen initialization.
"""

import numpy as np
import torch
from torch import nn

from fast import graph_contract, ranker_models
from fast import ablation_common as common
from fast.cad_features import ACTION_FEATURES


VARIANTS = ("D0", "D1", "D2")


def action_groups(spec):
    groups = spec["experiments"]["D"]["action_feature_groups"]
    names = [name for group in ("direction", "directional_clearance") for name in groups[group]]
    common.require(tuple(names) == tuple(ACTION_FEATURES), "frozen action-feature schema differs from the spec")
    for group, members in groups.items():
        common.require(all(name.split("__", 1)[0] == group for name in members), "action-feature group membership differs: " + group)
    return groups


def action_mask(spec, variant):
    """Return (indices, names) of the retained action features for a D variant."""
    common.require(variant in VARIANTS, "unknown D variant")
    groups = action_groups(spec)
    definition = spec["experiments"]["D"]["variants"][variant]
    retained = set(definition["retained_groups"])
    common.require(retained and retained.issubset(groups), "retained groups differ")
    indices = [index for index, name in enumerate(ACTION_FEATURES) if name.split("__", 1)[0] in retained]
    names = [ACTION_FEATURES[index] for index in indices]
    common.require(names == [name for group in ("direction", "directional_clearance") if group in retained for name in groups[group]],
                   "mask order differs from the frozen schema")
    common.require(len(indices) == definition["input_width"], "mask width differs from the spec")
    return np.asarray(indices, dtype=np.int64), names


def sliced_normalization(normalization, indices):
    """Slice the frozen training-only action normalization with the named mask; never recompute it."""
    mean = np.asarray(normalization["action"]["mean"], dtype=np.float32)
    std = np.asarray(normalization["action"]["std"], dtype=np.float32)
    common.require(mean.shape == (len(ACTION_FEATURES),) and std.shape == mean.shape, "action normalization shape differs")
    return {"node": normalization["node"], "action": {"mean": mean[indices].tolist(), "std": std[indices].tolist()}}


def masked_torch_batch(records, normalization, indices, device):
    """ranker_models.torch_batch with the action features restricted to the mask."""
    batch = graph_contract.batch_development_records(records)
    sliced = sliced_normalization(normalization, indices)
    node_mean = np.asarray(sliced["node"]["mean"], dtype=np.float32)
    node_std = np.asarray(sliced["node"]["std"], dtype=np.float32)
    action_mean = np.asarray(sliced["action"]["mean"], dtype=np.float32)
    action_std = np.asarray(sliced["action"]["std"], dtype=np.float32)
    batch["node_x"] = (batch["node_x"] - node_mean) / node_std
    batch["action_x"] = (np.ascontiguousarray(batch["action_x"][..., indices]) - action_mean) / action_std
    return {name: torch.as_tensor(value, device=device) if isinstance(value, np.ndarray) else value for name, value in batch.items()}


class MaskedGraphTransformer(nn.Module):
    """ranker_models.RankerModel('graph_transformer') with a configurable action-feature width."""

    def __init__(self, action_width):
        super().__init__()
        self.architecture = "graph_transformer"
        self.action_width = int(action_width)
        self.trunk = ranker_models._TransformerTrunk()
        width = self.trunk.width
        self.action_head = nn.Sequential(nn.Linear(width + self.action_width, width), nn.ReLU(), nn.Linear(width, 1))
        self.t_head = nn.Linear(width, 1)

    def forward(self, batch):
        node = self.trunk(batch["node_x"], batch["edge_index"], batch["node_ptr"])
        candidate = node[batch["candidate_index"]]
        common.require(batch["action_x"].shape[-1] == self.action_width, "masked action width differs")
        repeated = candidate[:, None, :].expand(-1, 6, -1)
        action_logits = self.action_head(torch.cat((repeated, batch["action_x"]), dim=-1)).squeeze(-1)
        return {"R": action_logits, "T": self.t_head(candidate).squeeze(-1)}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(spec, variant):
    indices, _ = action_mask(spec, variant)
    model = MaskedGraphTransformer(len(indices))
    expected = spec["experiments"]["D"]["variants"][variant]["parameter_count"]
    common.require(parameter_count(model) == expected, "{} parameter count differs: {}".format(variant, parameter_count(model)))
    return model
