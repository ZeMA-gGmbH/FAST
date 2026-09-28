import json
import argparse
import torch
import numpy as np
import trimesh
import networkx as nx
from pathlib import Path

from torch_geometric.data import Data, Batch
from networkx.readwrite import node_link_graph

from .model import FeasibilityTransformer
from .args import get_args as get_model_args

# Define the 6 canonical disassembly directions
ACTIONS_3D = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
]


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Safely load the trusted FAST checkpoint and require an exact match."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model


def get_action_vector_7d(action_vector_3d):
    """
    Converts a 3D action vector into the 7D encoding our model expects.
    """
    action_7d = [0.0] * 7
    if action_vector_3d is None:
        action_7d[6] = 1.0  # is_blocked flag
        return action_7d

    mapping = {
        (1, 0, 0): 0, (-1, 0, 0): 1,
        (0, 1, 0): 2, (0, -1, 0): 3,
        (0, 0, 1): 4, (0, 0, -1): 5
    }
    action_tuple = tuple(action_vector_3d)
    if action_tuple in mapping:
        action_7d[mapping[action_tuple]] = 1.0
    return action_7d


class InferenceHelper:
    """
    A helper class to prepare a single data point for inference.
    This encapsulates the data processing logic from data.py.
    """

    def __init__(self, assembly_id, asset_dir, max_pc_size=1000):
        self.assembly_id = assembly_id
        self.asset_dir = Path(asset_dir)
        self.max_pc_size = max_pc_size
        self.part_meshes = {}
        self.contact_graph = None
        self.config = None
        self._load_assembly_data()

    def _load_assembly_data(self):
        """Loads the contact graph and all part meshes for the assembly."""
        assembly_dir = self.asset_dir / self.assembly_id

        contact_graph_path = assembly_dir / 'contact_graph.json'
        if not contact_graph_path.is_file():
            raise FileNotFoundError(f'Missing contact graph: {contact_graph_path}')
        with contact_graph_path.open('r') as f:
            graph_dict = json.load(f)
            mapping = {node['id']: str(node['id']) for node in graph_dict['nodes']}
            self.contact_graph = nx.relabel_nodes(node_link_graph(graph_dict), mapping)

        # Load config.json for part poses
        config_path = assembly_dir / 'config.json'
        if not config_path.is_file():
            raise FileNotFoundError(f'Missing assembly configuration: {config_path}')
        with config_path.open('r') as f:
            self.config = json.load(f)

        for part_file in assembly_dir.glob('*.obj'):
            part_name = part_file.stem
            self.part_meshes[part_name] = trimesh.load(str(part_file), force='mesh', process=False)

        missing_meshes = set(self.contact_graph.nodes) - set(self.part_meshes)
        if missing_meshes:
            raise ValueError(f'Contact graph references missing meshes: {sorted(missing_meshes)}')

    def prepare_data_for_action(self, current_parts, action_part, action_direction):
        """Creates a PyG Data object for a given state and action."""
        subgraph = self.contact_graph.subgraph(current_parts).copy()

        subgraph_meshes = []
        for name in subgraph.nodes():
            mesh = self.part_meshes[name].copy()
            part_transform_info = self.config.get(name, {}).get('final_state')
            if part_transform_info:
                # Create a 4x4 transformation matrix from the [tx, ty, tz, rx, ry, rz] vector
                part_transform_matrix = trimesh.transformations.euler_matrix(
                    part_transform_info[3], part_transform_info[4], part_transform_info[5]
                )
                part_transform_matrix[:3, 3] = part_transform_info[:3]
                mesh.apply_transform(part_transform_matrix)
            subgraph_meshes.append(mesh)

        if not subgraph_meshes: return None
        combined_mesh = trimesh.util.concatenate(subgraph_meshes)

        bbox = combined_mesh.bounding_box
        center = torch.from_numpy(bbox.centroid.copy()).float()
        scale = 2.0 / torch.from_numpy(bbox.extents.copy()).float().max()

        pcs, volumes, center_masses, distances, targets, actions = [], [], [], [], [], []
        node_map = {node_name: i for i, node_name in enumerate(subgraph.nodes())}
        action_vec_7d = torch.tensor(get_action_vector_7d(action_direction), dtype=torch.float)

        for node_name in subgraph.nodes():
            mesh = self.part_meshes[node_name]
            pc = torch.from_numpy(trimesh.sample.sample_surface(mesh, self.max_pc_size)[0]).float()

            pc = (pc - center) * scale
            center_mass = (torch.from_numpy(mesh.center_mass).float() - center) * scale
            volume = torch.tensor([mesh.volume * (scale ** 3)], dtype=torch.float)
            distance = torch.linalg.norm(center_mass).unsqueeze(0)

            is_action_target = torch.tensor([1.0 if node_name == action_part else 0.0], dtype=torch.float)
            current_action_vector = action_vec_7d if node_name == action_part else torch.zeros_like(action_vec_7d)

            pcs.append(pc)
            volumes.append(volume)
            center_masses.append(center_mass)
            distances.append(distance)
            targets.append(is_action_target)
            actions.append(current_action_vector)

        edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in subgraph.edges()],
                                  dtype=torch.long).t().contiguous()

        return Data(
            edge_index=edge_index,
            pc=pcs,
            volume=torch.stack(volumes),
            center_mass=torch.stack(center_masses),
            distance=torch.stack(distances),
            is_action_target=torch.stack(targets),
            action_vector=torch.stack(actions),
            num_nodes=len(subgraph.nodes())
        )


def generate_sequence(args):
    """
    Generates a disassembly sequence for a given assembly using the trained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = FeasibilityTransformer(args).to(device)
    print(f"Loading model weights from: {args.checkpoint_path}")
    load_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    helper = InferenceHelper(args.assembly_id, args.asset_dir, args.max_pc_size)

    assembly_dir = Path(args.asset_dir) / args.assembly_id
    assembly_info_path = assembly_dir / 'assembly_info.json'
    base_part = None
    if assembly_info_path.exists():
        with assembly_info_path.open('r') as f:
            assembly_info = json.load(f)
        base_part = assembly_info.get('base_part')

    current_parts = [Path(p).stem for p in assembly_dir.glob('*.obj')]
    if base_part:
        current_parts.remove(base_part)

    print(f"\nStarting inference for assembly: {args.assembly_id}")
    print(f"Initial parts: {current_parts}")
    print(f"Base part: {base_part}")

    disassembly_sequence = []

    step = 1
    while len(current_parts) > 1:
        possible_actions = []

        # for part_to_remove in tqdm(current_parts, desc=f"Step {step}: Evaluating parts"):
        for part_to_remove in current_parts:
            for direction in ACTIONS_3D:
                data = helper.prepare_data_for_action(current_parts, part_to_remove, direction)
                if data is None: continue

                data_batch = Batch.from_data_list([data]).to(device)

                with torch.no_grad():
                    logit = model(data_batch)
                    score = torch.sigmoid(logit).item()

                possible_actions.append({'part': part_to_remove, 'direction': direction, 'score': score})

        if not possible_actions:
            print("Error: No possible actions found. Cannot proceed.")
            break

        best_action = max(possible_actions, key=lambda x: x['score'])
        disassembly_sequence.append(best_action)

        print(f"\n--- Step {step} Decision ---")
        print(f"Best Action: Remove part '{best_action['part']}' in direction {best_action['direction']}")
        print(f"Feasibility Score: {best_action['score']:.4f}")

        current_parts.remove(best_action['part'])
        # print(f"Remaining parts: {current_parts}")
        step += 1

    print("\n=====================================")
    print("Disassembly Complete.")
    assembly_sequence = [action['part'] for action in reversed(disassembly_sequence)]
    if base_part:
        assembly_sequence.insert(0, base_part)
    print(f"Final Assembly Sequence: {assembly_sequence}")
    print("=====================================")


if __name__ == '__main__':
    # --- Get Model Architecture Args ---
    args = get_model_args()

    # --- Parser for inference ---
    parser = argparse.ArgumentParser(description="Generate an assembly sequence using the trained model.")
    parser.add_argument('--assembly_id', type=str, required=True, help="ID of the assembly to process (e.g., '00232').")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint (.pth file).')
    parser.add_argument('--asset-dir', type=str, default='./assets',
                        help='Directory containing assembly subdirectories.')

    inference_args = parser.parse_args()
    for key, value in vars(inference_args).items():
        setattr(args, key, value)

    generate_sequence(args)
