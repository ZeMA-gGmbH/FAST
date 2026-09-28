import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

# --- Stage 1, Part A: Geometric Encoder ---
class PointNetEncoder(nn.Module):
    """
    Encodes a point cloud into a single feature vector.
    This is a deeper version, similar to the one used in the ASAP paper.
    """

    def __init__(self, input_dim=3, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_dim)

    def forward(self, x):
        # x shape: [B * N, num_points, 3] -> [B * N, 3, num_points]
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(x.size(0), -1)  # [B * N, feat_dim]
        return x


# --- Stage 1, Part B: Custom Feature Encoder ---
class CustomFeatureEncoder(nn.Module):
    """
    Encodes scalar and categorical features using a deeper MLP.
    """

    def __init__(self, input_dim, feat_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# --- Stage 2: Graph Transformer Core ---
class GraphTransformerEncoder(nn.Module):
    """
    Processes node features using Graph Transformer layers to learn graph structure.
    """

    def __init__(self, input_dim, transformer_dim, num_heads, dropout):
        super().__init__()
        self.conv1 = TransformerConv(input_dim, transformer_dim, heads=num_heads, dropout=dropout)
        self.conv2 = TransformerConv(transformer_dim * num_heads, transformer_dim, heads=num_heads, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# --- Stage 3: Graph-Level Classifier ---
class GraphClassifier(nn.Module):
    """
    Takes a graph-level feature vector and predicts a single logit.
    """

    def __init__(self, input_dim, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(x)


# --- The Main Assembled Model ---
class FeasibilityTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # --- Instantiate all the modules ---
        self.pointnet_encoder = PointNetEncoder(feat_dim=args.pointnet_dim)

        # volume (1) + center_mass (3) + distance (1) + is_action_target (1) + action_vector (7) = 13
        custom_feature_dim = 1 + 3 + 1 + 1 + 7
        self.custom_feature_encoder = CustomFeatureEncoder(
            input_dim=custom_feature_dim,
            feat_dim=args.custom_mlp_dim
        )

        fused_dim = args.pointnet_dim + args.custom_mlp_dim
        self.graph_transformer = GraphTransformerEncoder(
            input_dim=fused_dim,
            transformer_dim=args.transformer_dim,
            num_heads=args.num_heads,
            dropout=args.dropout
        )

        classifier_input_dim = args.transformer_dim * args.num_heads
        self.classifier = GraphClassifier(
            input_dim=classifier_input_dim,
            dropout=args.dropout
        )

    def forward(self, data):
        """
        The forward pass connecting all the modules.
        This function returns the raw logits for the graph classification task.

        Args:
            data (torch_geometric.data.Batch): A batch of graph data.
        """
        if isinstance(data.pc, list):
            point_clouds = torch.stack(data.pc)
        else:
            point_clouds = data.pc
        edge_index = data.edge_index
        batch = data.batch

        # --- Stage 1: Encode and Fuse Features ---
        geometric_embedding = self.pointnet_encoder(point_clouds)

        custom_features = torch.cat([
            data.volume,
            data.center_mass,
            data.distance,
            data.is_action_target,
            data.action_vector
        ], dim=1)
        context_embedding = self.custom_feature_encoder(custom_features)

        x = torch.cat([geometric_embedding, context_embedding], dim=1)

        # --- Stage 2: Process with Graph Transformer ---
        x = self.graph_transformer(x, edge_index)

        # --- Stage 3: Graph Pooling and Classification ---
        graph_embedding = global_mean_pool(x, batch)
        output = self.classifier(graph_embedding)

        return output
