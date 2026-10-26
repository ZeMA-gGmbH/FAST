"""Frozen ranker PyTorch trunks, heads, batching, and losses."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, GCNConv

from fast import graph_contract


PRIMARY_ARCHITECTURES = ("mlp", "gnn", "gatv2", "graph_transformer")
DIRECT_Y_ARCHITECTURE = "graph_transformer_direct_y"
ARCHITECTURES = PRIMARY_ARCHITECTURES + (DIRECT_Y_ARCHITECTURE,)
EXPECTED_PARAMETERS = {
    "mlp": 91266,
    "gnn": 91266,
    "gatv2": 91490,
    "graph_transformer": 90142,
    "graph_transformer_direct_y": 90045,
}


class _LinearBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.dropout(self.norm(F.relu(self.linear(x))))


class _IndependentTrunk(nn.Module):
    width = 128

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(41, self.width)
        self.blocks = nn.ModuleList(_LinearBlock(self.width) for _ in range(4))

    def forward(self, node_x, edge_index, node_ptr):
        del edge_index, node_ptr
        x = self.input(node_x)
        for block in self.blocks:
            x = block(x)
        return x


class _GCNBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = GCNConv(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        return x + self.dropout(self.norm(F.relu(self.conv(x, edge_index))))


class _GCNTrunk(nn.Module):
    width = 128

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(41, self.width)
        self.blocks = nn.ModuleList(_GCNBlock(self.width) for _ in range(4))

    def forward(self, node_x, edge_index, node_ptr):
        del node_ptr
        x = self.input(node_x)
        for block in self.blocks:
            x = block(x, edge_index)
        return x


class _GATBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = GATv2Conv(width, 24, heads=4)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        return x + self.dropout(self.norm(F.relu(self.conv(x, edge_index))))


class _GATTrunk(nn.Module):
    width = 96

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(41, self.width)
        self.input_norm = nn.LayerNorm(self.width)
        self.blocks = nn.ModuleList(_GATBlock(self.width) for _ in range(4))
        self.output_norm = nn.LayerNorm(self.width)

    def forward(self, node_x, edge_index, node_ptr):
        del node_ptr
        x = self.input_norm(self.input(node_x))
        for block in self.blocks:
            x = block(x, edge_index)
        return self.output_norm(x)


class _TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(96, 4, dropout=0.1, batch_first=True)
        self.distance_bias = nn.Embedding(71, 4)
        self.norm1 = nn.LayerNorm(96)
        self.norm2 = nn.LayerNorm(96)
        self.linear1 = nn.Linear(96, 192)
        self.linear2 = nn.Linear(192, 96)
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def _distances(node_count, edge_index, device):
        distance = torch.full((node_count, node_count), 70, dtype=torch.long, device=device)
        distance.fill_diagonal_(0)
        if edge_index.numel():
            distance[edge_index[0], edge_index[1]] = 1
        for intermediate in range(node_count):
            distance = torch.minimum(
                distance,
                distance[:, intermediate, None] + distance[None, intermediate, :],
            )
        return distance.clamp_max(70)

    def forward(self, x, edge_index):
        distance = self._distances(len(x), edge_index, x.device)
        bias = self.distance_bias(distance).permute(2, 0, 1)
        attended, _ = self.attention(x[None], x[None], x[None], attn_mask=bias, need_weights=False)
        x = self.norm1(x + self.dropout(attended[0]))
        feed_forward = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout(feed_forward))


class _TransformerTrunk(nn.Module):
    width = 96

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(41, self.width)
        self.block = _TransformerBlock()

    def forward(self, node_x, edge_index, node_ptr):
        outputs = []
        for context in range(len(node_ptr) - 1):
            start, stop = int(node_ptr[context]), int(node_ptr[context + 1])
            mask = (
                (edge_index[0] >= start)
                & (edge_index[0] < stop)
                & (edge_index[1] >= start)
                & (edge_index[1] < stop)
            )
            outputs.append(self.block(self.input(node_x[start:stop]), edge_index[:, mask] - start))
        return torch.cat(outputs)


class RankerModel(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        graph_contract._require(architecture in ARCHITECTURES, "unknown architecture")
        self.architecture = architecture
        if architecture == "mlp":
            self.trunk = _IndependentTrunk()
        elif architecture == "gnn":
            self.trunk = _GCNTrunk()
        elif architecture == "gatv2":
            self.trunk = _GATTrunk()
        else:
            self.trunk = _TransformerTrunk()
        width = self.trunk.width
        self.action_head = nn.Sequential(nn.Linear(width + 16, width), nn.ReLU(), nn.Linear(width, 1))
        self.t_head = None if architecture == DIRECT_Y_ARCHITECTURE else nn.Linear(width, 1)

    def forward(self, batch):
        node = self.trunk(batch["node_x"], batch["edge_index"], batch["node_ptr"])
        candidate = node[batch["candidate_index"]]
        repeated = candidate[:, None, :].expand(-1, 6, -1)
        action_logits = self.action_head(torch.cat((repeated, batch["action_x"]), dim=-1)).squeeze(-1)
        if self.architecture == DIRECT_Y_ARCHITECTURE:
            return {"Y": action_logits}
        return {"R": action_logits, "T": self.t_head(candidate).squeeze(-1)}


def build_model(architecture):
    model = RankerModel(architecture)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    graph_contract._require(count == EXPECTED_PARAMETERS[architecture], "{} parameter count differs: {}".format(architecture, count))
    return model


def torch_batch(records, normalization, device):
    batch = graph_contract.batch_development_records(records)
    node_mean = np.asarray(normalization["node"]["mean"], dtype=np.float32)
    node_std = np.asarray(normalization["node"]["std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action"]["mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action"]["std"], dtype=np.float32)
    batch["node_x"] = (batch["node_x"] - node_mean) / node_std
    batch["action_x"] = (batch["action_x"] - action_mean) / action_std
    result = {}
    for name, value in batch.items():
        if isinstance(value, np.ndarray):
            result[name] = torch.as_tensor(value, device=device)
        else:
            result[name] = value
    return result


def _joint_nll(r_logits, t_logits, targets):
    positive = F.softplus(-r_logits) + F.softplus(-t_logits)
    negative = F.softplus(r_logits) + F.softplus(t_logits) - torch.logsumexp(
        torch.stack((torch.zeros_like(r_logits), r_logits, t_logits)), dim=0
    )
    return torch.where(targets == 1, positive, negative)


def model_loss(batch, outputs, component_assembly_counts, direct_y=False):
    total_assemblies = component_assembly_counts["total"]
    assembly_count = len(batch["assembly_keys"])
    candidate_assembly = batch["context_assembly"][batch["candidate_context"]]
    component_values = {name: [] for name in (("rank", "Y") if direct_y else ("rank", "R", "T", "Y"))}
    denominators = {name: 0 for name in component_values}
    if direct_y:
        action_score = F.logsigmoid(outputs["Y"])
        pointwise = F.binary_cross_entropy_with_logits(outputs["Y"], batch["Y"], reduction="none")
    else:
        r_logits, t_logits = outputs["R"], outputs["T"]
        action_score = F.logsigmoid(r_logits) + F.logsigmoid(t_logits)[:, None]
        r_pointwise = F.binary_cross_entropy_with_logits(r_logits, batch["R"], reduction="none")
        t_pointwise = F.binary_cross_entropy_with_logits(t_logits, batch["T"], reduction="none")
        pointwise = _joint_nll(r_logits, t_logits[:, None].expand_as(r_logits), batch["Y"])

    for assembly in range(assembly_count):
        candidates = candidate_assembly == assembly
        if not direct_y:
            for name, values, mask in (
                ("R", r_pointwise, candidates[:, None] & batch["R_valid"]),
                ("T", t_pointwise, candidates & batch["T_valid"]),
                ("Y", pointwise, candidates[:, None] & batch["R_valid"] & batch["T_valid"][:, None]),
            ):
                denominators[name] += int(mask.sum())
                if mask.any():
                    component_values[name].append(values[mask].mean())
        else:
            mask = candidates[:, None] & batch["Y_valid"]
            denominators["Y"] += int(mask.sum())
            if mask.any():
                component_values["Y"].append(pointwise[mask].mean())

        ranking = []
        for context in torch.nonzero(batch["context_assembly"] == assembly, as_tuple=False).flatten().tolist():
            context_candidates = batch["candidate_context"] == context
            if direct_y:
                valid = batch["Y_valid"][context_candidates].all()
            else:
                valid = batch["R_valid"][context_candidates].all() and batch["T_valid"][context_candidates].all()
            targets = batch["Y"][context_candidates].reshape(-1)
            if not valid or not targets.any() or targets.all():
                continue
            scores = action_score[context_candidates].reshape(-1)
            ranking.append(torch.logsumexp(scores, 0) - torch.logsumexp(scores[targets == 1], 0))
        denominators["rank"] += len(ranking)
        if ranking:
            component_values["rank"].append(torch.stack(ranking).mean())

    weights = {"rank": 1.0, "R": 1.0 / 3.0, "T": 1.0 / 3.0, "Y": 1.0 if direct_y else 1.0 / 3.0}
    loss = torch.zeros((), device=batch["node_x"].device)
    components = {}
    for name, values in component_values.items():
        if values:
            value = torch.stack(values).sum() / assembly_count
            loss = loss + weights[name] * total_assemblies / component_assembly_counts[name] * value
            components[name] = float(torch.stack(values).mean().detach())
        else:
            components[name] = None
    graph_contract._require(torch.isfinite(loss).item(), "non-finite loss")
    return loss, {"components": components, "denominators": denominators}


def probabilities(outputs):
    if set(outputs) == {"Y"}:
        return {"Y": torch.sigmoid(outputs["Y"])}
    r = torch.sigmoid(outputs["R"])
    t = torch.sigmoid(outputs["T"])
    return {"R": r, "T": t, "Y": r * t[:, None]}
