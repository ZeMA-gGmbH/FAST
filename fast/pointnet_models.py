"""PointNet ranker and Cebulla-style control."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fast import graph_contract, ranker_models


POINTNET_PARAMETERS = 21376
EXPECTED_PARAMETERS = 93902


class PointNet(nn.Module):
    """Isolated copy of the released ASAP PN topology, fixed to XYZ -> 32."""

    def __init__(self):
        super().__init__()
        channels = (3, 64, 64, 64, 128, 32)
        self.convs = nn.ModuleList(
            nn.Conv1d(source, target, 1, bias=False)
            for source, target in zip(channels[:-1], channels[1:])
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(channel) for channel in channels[1:])

    def forward(self, points):
        x = points.permute(0, 2, 1)
        for convolution, normalization in zip(self.convs, self.norms):
            x = F.relu(normalization(convolution(x)))
        return x.max(dim=-1).values


class _TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(84, 4, dropout=0.1, batch_first=True)
        self.distance_bias = nn.Embedding(71, 4)
        self.norm1 = nn.LayerNorm(84)
        self.norm2 = nn.LayerNorm(84)
        self.linear1 = nn.Linear(84, 168)
        self.linear2 = nn.Linear(168, 84)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        distance = ranker_models._TransformerBlock._distances(len(x), edge_index, x.device)
        bias = self.distance_bias(distance).permute(2, 0, 1)
        attended, _ = self.attention(x[None], x[None], x[None], attn_mask=bias, need_weights=False)
        x = self.norm1(x + self.dropout(attended[0]))
        return self.norm2(x + self.dropout(self.linear2(self.dropout(F.relu(self.linear1(x))))))


class _TransformerTrunk(nn.Module):
    width = 84

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(73, self.width)
        self.block = _TransformerBlock()

    def forward(self, node_x, edge_index, node_ptr):
        outputs = []
        for context in range(len(node_ptr) - 1):
            start, stop = int(node_ptr[context]), int(node_ptr[context + 1])
            mask = ((edge_index[0] >= start) & (edge_index[0] < stop)
                    & (edge_index[1] >= start) & (edge_index[1] < stop))
            outputs.append(self.block(self.input(node_x[start:stop]), edge_index[:, mask] - start))
        return torch.cat(outputs)


class PointNetGraphTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointnet = PointNet()
        self.trunk = _TransformerTrunk()
        self.action_head = nn.Sequential(nn.Linear(100, 84), nn.ReLU(), nn.Linear(84, 1))
        self.t_head = nn.Linear(84, 1)

    def forward(self, batch):
        unique_embeddings = self.pointnet(batch["point_x"])
        node_x = torch.cat((batch["node_x"], unique_embeddings[batch["node_point_index"]]), dim=1)
        node = self.trunk(node_x, batch["edge_index"], batch["node_ptr"])
        candidate = node[batch["candidate_index"]]
        repeated = candidate[:, None, :].expand(-1, 6, -1)
        action_logits = self.action_head(torch.cat((repeated, batch["action_x"]), dim=-1)).squeeze(-1)
        return {"R": action_logits, "T": self.t_head(candidate).squeeze(-1)}


def build_pointnet_model():
    model = PointNetGraphTransformer()
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    graph_contract._require(count == EXPECTED_PARAMETERS, "PointNet parameter count differs")
    graph_contract._require(
        sum(parameter.numel() for parameter in model.pointnet.parameters() if parameter.requires_grad) == POINTNET_PARAMETERS,
        "PointNet encoder parameter count differs",
    )
    return model


def torch_batch(records, normalization, point_assemblies, device):
    """Batch each unique assembly part once and gather it into context nodes."""
    numpy_batch = graph_contract.batch_development_records(records)
    node_mean = np.asarray(normalization["node"]["mean"], dtype=np.float32)
    node_std = np.asarray(normalization["node"]["std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action"]["mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action"]["std"], dtype=np.float32)
    numpy_batch["node_x"] = (numpy_batch["node_x"] - node_mean) / node_std
    numpy_batch["action_x"] = (numpy_batch["action_x"] - action_mean) / action_std

    unique, points, gather = {}, [], []
    for record in records:
        key = (record["source_split"], record["assembly_id"])
        part_ids, tensor = point_assemblies[key]
        graph_contract._require(part_ids == sorted(part_ids) and tensor.shape == (len(part_ids), 1000, 3), "point assembly differs")
        lookup = {part_id: index for index, part_id in enumerate(part_ids)}
        for part_id in record["part_ids"]:
            unique_key = key + (part_id,)
            if unique_key not in unique:
                unique[unique_key] = len(points)
                points.append(tensor[lookup[part_id]])
            gather.append(unique[unique_key])
    numpy_batch["point_x"] = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    numpy_batch["node_point_index"] = np.asarray(gather, dtype=np.int64)
    graph_contract._require(len(gather) == len(numpy_batch["node_x"]), "point gather length differs")
    return {
        name: torch.as_tensor(value, device=device) if isinstance(value, np.ndarray) else value
        for name, value in numpy_batch.items()
    }


class CebullaGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = ranker_models._GCNTrunk()
        self.classifier = nn.Linear(self.trunk.width, 1)

    def forward(self, batch):
        return self.classifier(self.trunk(batch["node_x"], batch["edge_index"], batch["node_ptr"])).squeeze(-1)


def cebulla_targets(batch):
    node_count = len(batch["node_x"])
    targets = torch.zeros(node_count, dtype=torch.float32, device=batch["node_x"].device)
    valid = torch.zeros(node_count, dtype=torch.bool, device=batch["node_x"].device)
    positive = (batch["R_valid"] & (batch["R"] == 1)).any(dim=1)
    negative = batch["R_valid"].all(dim=1) & (batch["R"] == 0).all(dim=1)
    indices = batch["candidate_index"]
    targets[indices] = positive.float()
    valid[indices] = positive | negative
    return targets, valid


def cebulla_loss(batch, logits):
    targets, valid = cebulla_targets(batch)
    values = []
    for assembly in range(len(batch["assembly_keys"])):
        contexts = torch.nonzero(batch["context_assembly"] == assembly, as_tuple=False).flatten()
        mask = valid & torch.isin(batch["node_context"], contexts)
        if mask.any():
            values.append(F.binary_cross_entropy_with_logits(logits[mask], targets[mask]))
    graph_contract._require(values, "Cebulla batch has no resolved targets")
    return torch.stack(values).mean()
