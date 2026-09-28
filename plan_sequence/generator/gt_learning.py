import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Batch

from .base import Generator
from fast_network.model import FeasibilityTransformer
from fast_network.args import get_args
from fast_network.inference import ACTIONS_3D, InferenceHelper, load_checkpoint

class TransformerBasedGenerator(Generator):
    """
    Generate candidate parts using our trained FeasibilityTransformer model.
    """
    def __init__(self, asset_folder, assembly_dir, checkpoint_path, base_part=None, save_sdf=False):
        super().__init__(asset_folder, assembly_dir, base_part=base_part, save_sdf=save_sdf)

        # --- Load Model ---
        self.args = get_args()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = FeasibilityTransformer(self.args).to(self.device)
        load_checkpoint(self.model, checkpoint_path, self.device)
        self.model.eval()

        # --- Prepare Data Helper ---
        assembly_id = Path(assembly_dir).name
        data_dir = Path(assembly_dir).parent
        self.inference_helper = InferenceHelper(assembly_id, data_dir, self.args.max_pc_size)

    def generate_candidate_part(self, assembled_parts):
        """
        Generate the next candidate part to disassemble by finding the part
        with the highest-scoring feasible action.

        Input:
            assembled_parts: A list of parts currently in the assembly.
        Output:
            A list of parts, ordered from most likely to least likely to be removed next.
        """
        parts_to_consider = self._remove_base_part(assembled_parts)

        part_scores = {}

        # Evaluate all possible actions for the current state
        for part_to_remove in parts_to_consider:
            max_score_for_part = -1.0

            for direction in ACTIONS_3D:
                # Prepare the data for this specific part and direction
                data = self.inference_helper.prepare_data_for_action(
                    assembled_parts, part_to_remove, direction
                )
                if data is None: continue

                # Batch the single data object and send to device
                data_batch = Batch.from_data_list([data]).to(self.device)

                # Get the model's prediction
                with torch.no_grad():
                    logit = self.model(data_batch)
                    score = torch.sigmoid(logit).item()

                # Keep track of the best score found for this part
                if score > max_score_for_part:
                    max_score_for_part = score

            part_scores[part_to_remove] = max_score_for_part

        if not part_scores:
            return np.random.permutation(np.array(parts_to_consider, dtype=object))

        # Sort the parts in descending order based on their best score
        # The part with the highest feasibility score for any action is ranked first
        ordered_parts = sorted(part_scores, key=part_scores.get, reverse=True)

        return ordered_parts
