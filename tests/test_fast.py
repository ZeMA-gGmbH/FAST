import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch_geometric.data import Batch

from fast_network.args import get_args
from fast_network.inference import InferenceHelper, load_checkpoint
from fast_network.model import FeasibilityTransformer


CHECKPOINT = Path(os.environ.get('FAST_CHECKPOINT', 'checkpoints/pretrained_model_gt.pth'))


class FASTTests(unittest.TestCase):
    @unittest.skipUnless(CHECKPOINT.is_file(), 'download the FAST checkpoint first')
    def test_safe_checkpoint_load_and_inference(self):
        args = get_args()
        model = FeasibilityTransformer(args)
        load_checkpoint(model, CHECKPOINT)
        model.eval()

        with tempfile.TemporaryDirectory() as root:
            assembly = Path(root) / 'example'
            assembly.mkdir()
            trimesh.creation.box().export(assembly / '0.obj')
            trimesh.creation.box().export(assembly / '1.obj')
            (assembly / 'config.json').write_text(json.dumps({
                '0': {'final_state': [0, 0, 0, 0, 0, 0]},
                '1': {'final_state': [1, 0, 0, 0, 0, 0]},
            }))
            (assembly / 'contact_graph.json').write_text(json.dumps({
                'directed': False,
                'multigraph': False,
                'graph': {},
                'nodes': [{'id': '0'}, {'id': '1'}],
                'links': [{'source': '0', 'target': '1'}],
            }))

            helper = InferenceHelper('example', root, args.max_pc_size)
            torch.manual_seed(0)
            data = helper.prepare_data_for_action(['0', '1'], '1', [1, 0, 0])
            with torch.no_grad():
                output = model(Batch.from_data_list([data]))

        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_dfs_sequence_generation(self):
        try:
            from plan_sequence.planner.dfs import DFSSequencePlanner
        except ImportError as error:
            self.skipTest(str(error))

        class Generator:
            base_part = '0'
            asset_folder = ''
            assembly_dir = ''

            @staticmethod
            def generate_candidate_part(parts):
                return [part for part in parts if part != '0']

        class Planner(DFSSequencePlanner):
            def __init__(self):
                self.seq_generator = Generator()
                self.asset_folder = 'assets'
                self.assembly_dir = 'assets/beam_assembly/original'
                self.base_part = '0'
                self.parts = ['0', '1', '2']
                self.part_mass = {'0': 2.0, '1': 1.0, '2': 1.0}
                self.num_proc = 1
                self.save_sdf = False
                self.t_start = self.n_eval = self.stop_msg = None

            def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers, **kwargs):
                return {
                    'feasible': True,
                    'action': np.array([1, 0, 0]),
                    'base_part': self.base_part,
                    'parts_fix': [],
                    'part_move': part_move,
                    'pose': pose,
                }

        planner = Planner()
        tree = planner.plan(4, 2, early_term=True)
        stats = planner.get_stats(tree)
        self.assertTrue(stats['success'])
        self.assertEqual(len(stats['sequence']), 2)


if __name__ == '__main__':
    unittest.main()
