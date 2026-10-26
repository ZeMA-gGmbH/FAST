"""Structured ASAP simulation labels for one disassembly state."""

from time import time

from assets.load import load_part_ids
from plan_sequence.feasibility_check import get_R3_actions
from plan_sequence.physics_planner import MultiPartPathPlanner, MultiPartStabilityPlanner


_DIRECTION_NAMES = ('+Z', '-Z', '+X', '-X', '+Y', '-Y')


def _error_info(error):
    return {'type': type(error).__name__, 'message': str(error)}


class ASAPLabelOracle:
    """Evaluate ASAP A6 removability and serial-greedy G=2 stability."""

    def __init__(self, asset_folder, assembly_dir, *, timeout=None, save_sdf=False, allow_gap=False):
        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.timeout = timeout
        self.save_sdf = save_sdf
        self.allow_gap = allow_gap
        self.all_parts = load_part_ids(assembly_dir)

    def _validate_parts(self, parts):
        parts = list(parts)
        if not parts:
            raise ValueError('parts must not be empty')
        if len(parts) != len(set(parts)):
            raise ValueError('parts must not contain duplicates')
        unknown = [part for part in parts if part not in self.all_parts]
        if unknown:
            raise ValueError(f'unknown parts: {unknown}')
        return parts

    def evaluate_removability(self, parts, part_move, pose=None):
        """Evaluate all six ASAP directions using one path-planner instance."""
        parts = self._validate_parts(parts)
        if part_move not in parts:
            raise ValueError(f'part_move {part_move!r} is not in parts')
        if len(parts) < 2:
            raise ValueError('removability requires at least two parts')

        parts_fix = [part for part in parts if part != part_move]
        parts_removed = [part for part in self.all_parts if part not in parts]
        actions = get_R3_actions()
        t_start = time()

        try:
            planner = MultiPartPathPlanner(
                self.asset_folder,
                self.assembly_dir,
                parts_fix,
                part_move,
                parts_removed=parts_removed,
                pose=pose,
                save_sdf=self.save_sdf,
            )
        except Exception as error:
            runtime_s = time() - t_start
            error_info = _error_info(error)
            directions = [
                {
                    'direction': name,
                    'action': action.tolist(),
                    'label': None,
                    'status': 'error',
                    'reason': 'planner_initialization',
                    'rollout_count': 0,
                    'forward_steps': 0,
                    'path_length': 0,
                    'runtime_s': 0.0,
                    'error': error_info,
                }
                for name, action in zip(_DIRECTION_NAMES, actions)
            ]
            return {
                'part_move': part_move,
                'parts_fix': parts_fix,
                'label': None,
                'status': 'error',
                'reason': 'planner_initialization',
                'rollout_count': 0,
                'forward_steps': 0,
                'runtime_s': runtime_s,
                'directions': directions,
                'error': error_info,
            }

        directions = []
        for name, action in zip(_DIRECTION_NAMES, actions):
            rollout_start = time()
            try:
                detail = planner.check_success_detailed(action, timeout=self.timeout)
                directions.append({
                    'direction': name,
                    'action': action.tolist(),
                    'label': detail['success'] if detail['status'] != 'timeout' else None,
                    'status': detail['status'],
                    'reason': detail['reason'],
                    'rollout_count': 1,
                    'forward_steps': detail['forward_steps'],
                    'path_length': len(detail['path']),
                    'runtime_s': detail['runtime_s'],
                })
            except Exception as error:
                directions.append({
                    'direction': name,
                    'action': action.tolist(),
                    'label': None,
                    'status': 'error',
                    'reason': 'simulator_error',
                    'rollout_count': 1,
                    'forward_steps': 0,
                    'path_length': 0,
                    'runtime_s': time() - rollout_start,
                    'error': _error_info(error),
                })

        labels = [result['label'] for result in directions]
        if any(label is True for label in labels):
            label, status, reason = True, 'success', 'some_direction_succeeded'
        elif all(label is False for label in labels):
            label, status, reason = False, 'blocked', 'all_directions_blocked'
        elif any(result['status'] == 'error' for result in directions):
            label, status, reason = None, 'error', 'direction_error'
        else:
            label, status, reason = None, 'timeout', 'direction_timeout'

        return {
            'part_move': part_move,
            'parts_fix': parts_fix,
            'label': label,
            'status': status,
            'reason': reason,
            'rollout_count': sum(result['rollout_count'] for result in directions),
            'forward_steps': sum(result['forward_steps'] for result in directions),
            'runtime_s': time() - t_start,
            'directions': directions,
        }

    def evaluate_stability(self, parts_after, pose=None):
        """Reproduce ASAP's serial-greedy holding search with base=None and G=2."""
        parts_after = self._validate_parts(parts_after)
        max_fix = min(1, len(parts_after))
        parts_fix = []
        rollout_count = 0
        forward_steps = 0
        rollouts = []
        t_start = time()

        while True:
            remaining_timeout = None
            if self.timeout is not None:
                remaining_timeout = self.timeout - (time() - t_start)
                if remaining_timeout < 0:
                    return {
                        'label': None,
                        'status': 'timeout',
                        'reason': 'stability_timeout',
                        'parts_fix': None,
                        'rollout_count': rollout_count,
                        'forward_steps': forward_steps,
                        'runtime_s': time() - t_start,
                        'rollouts': rollouts,
                    }

            parts_move = [part for part in parts_after if part not in parts_fix]
            rollout_count += 1
            rollout_start = time()
            try:
                planner = MultiPartStabilityPlanner(
                    self.asset_folder,
                    self.assembly_dir,
                    parts_fix,
                    parts_move,
                    pose=pose,
                    save_sdf=self.save_sdf,
                    allow_gap=self.allow_gap,
                )
                detail = planner.check_success_detailed(timeout=remaining_timeout)
            except Exception as error:
                error_info = _error_info(error)
                rollouts.append({
                    'parts_fix': parts_fix.copy(),
                    'parts_move': parts_move,
                    'status': 'error',
                    'reason': 'simulator_error',
                    'parts_fall': None,
                    'forward_steps': 0,
                    'runtime_s': time() - rollout_start,
                    'error': error_info,
                })
                return {
                    'label': None,
                    'status': 'error',
                    'reason': 'simulator_error',
                    'parts_fix': None,
                    'rollout_count': rollout_count,
                    'forward_steps': forward_steps,
                    'runtime_s': time() - t_start,
                    'rollouts': rollouts,
                    'error': error_info,
                }

            forward_steps += detail['forward_steps']
            rollouts.append({
                'parts_fix': parts_fix.copy(),
                'parts_move': parts_move,
                'status': detail['status'],
                'reason': detail['reason'],
                'parts_fall': detail['parts_fall'],
                'forward_steps': detail['forward_steps'],
                'runtime_s': detail['runtime_s'],
            })

            if detail['success']:
                return {
                    'label': True,
                    'status': 'success',
                    'reason': 'stable',
                    'parts_fix': parts_fix,
                    'rollout_count': rollout_count,
                    'forward_steps': forward_steps,
                    'runtime_s': time() - t_start,
                    'rollouts': rollouts,
                }
            if detail['status'] == 'timeout':
                return {
                    'label': None,
                    'status': 'timeout',
                    'reason': 'stability_timeout',
                    'parts_fix': None,
                    'rollout_count': rollout_count,
                    'forward_steps': forward_steps,
                    'runtime_s': time() - t_start,
                    'rollouts': rollouts,
                }

            if not detail['parts_fall']:
                return {
                    'label': None,
                    'status': 'error',
                    'reason': 'invalid_simulator_result',
                    'parts_fix': None,
                    'rollout_count': rollout_count,
                    'forward_steps': forward_steps,
                    'runtime_s': time() - t_start,
                    'rollouts': rollouts,
                }
            parts_fix.extend(detail['parts_fall'])
            if len(parts_fix) > max_fix:
                return {
                    'label': False,
                    'status': 'unstable',
                    'reason': 'max_fix_exceeded',
                    'parts_fix': None,
                    'rollout_count': rollout_count,
                    'forward_steps': forward_steps,
                    'runtime_s': time() - t_start,
                    'rollouts': rollouts,
                }

    def evaluate_part(self, parts, part_move, pose=None):
        """Evaluate R and T independently for one candidate child state."""
        parts = self._validate_parts(parts)
        removability = self.evaluate_removability(parts, part_move, pose)
        parts_after = [part for part in parts if part != part_move]
        stability = self.evaluate_stability(parts_after, pose)
        return {
            'part_move': part_move,
            'parts_after': parts_after,
            'removability': removability,
            'stability': stability,
            'label': self._combined_label(removability['label'], stability['label']),
        }

    @staticmethod
    def _combined_label(removability, stability):
        if removability is False or stability is False:
            return False
        if removability is True and stability is True:
            return True
        return None
