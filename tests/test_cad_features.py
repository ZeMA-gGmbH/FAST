import json
import math
import unittest

import networkx as nx
import numpy as np
import trimesh

from fast import cad_features as features


def _part(center, scale=1.0):
    mesh = trimesh.creation.box(extents=np.array([1.0, 1.0, 1.0]) * scale)
    mesh.apply_translation(np.asarray(center) * scale)
    inertia = np.linalg.eigvalsh(np.asarray(mesh.moment_inertia, dtype=np.float64))
    return {
        "vertices": np.asarray(mesh.vertices, dtype=np.float64),
        "volume": abs(float(mesh.volume)),
        "area": float(mesh.area),
        "center": np.asarray(mesh.center_mass, dtype=np.float64),
        "inertia": inertia,
        "extent": np.ptp(np.asarray(mesh.vertices), axis=0),
    }


def _record(scale=1.0, x_translation=0.0):
    static = {
        "0": _part((0.0, 0.0, 0.5), scale),
        "1": _part((1.0, 0.0, 0.5), scale),
        "2": _part((2.0, 0.0, 0.5), scale),
    }
    graph = nx.Graph([("0", "1"), ("1", "2")])
    pose = np.eye(4)
    pose[0, 3] = x_translation
    posed = features._posed_parts(static, pose, {})
    header = {
        "parts": ["0", "1", "2"],
        "parts_after": ["1", "2"],
        "part_move": "0",
        "state_id": "state-root",
        "pose_id": "pose-000",
    }
    actions = []
    for direction_id, direction in features.dataset_index.DIRECTIONS:
        removable = direction_id == "-X"
        actions.append(
            {
                "direction_id": direction_id,
                "direction": direction,
                "R": removable,
                "Y": removable,
            }
        )
    row = {
        "source_split": "training_assembly",
        "assembly_id": "00001",
        "dataset_split": "train",
        "tier": "small",
    }
    return features._candidate_record(
        row,
        header,
        {"queried": True, "T": True},
        actions,
        static,
        posed,
        graph,
        sum(part["volume"] for part in static.values()),
        0.001,
    )


class ScalarFeatureFeatureTest(unittest.TestCase):
    def test_canonical_json_round_trip_preserves_feature_schema(self):
        record = json.loads(features.production._canonical_json(_record()))
        self.assertNotEqual(tuple(record["features"]), features.CANDIDATE_FEATURES)
        features._validate_features(record["features"], features.CANDIDATE_FEATURES)
        for action in record["actions"]:
            features._validate_features(action["features"], features.ACTION_FEATURES)

    def test_identity_fallback_pose_matches_explicit_identity(self):
        fallback_key, fallback = features._pose_key(None)
        identity_key, identity = features._pose_key(np.eye(4))
        self.assertEqual(fallback_key, identity_key)
        np.testing.assert_array_equal(fallback, identity)

    def test_three_valued_joint_target(self):
        true = {"value": True, "valid": True}
        false = {"value": False, "valid": True}
        null = {"value": None, "valid": False}
        self.assertEqual(features._joint_target(true, true), true)
        self.assertEqual(features._joint_target(false, null), false)
        self.assertEqual(features._joint_target(null, false), false)
        self.assertEqual(features._joint_target(true, null), null)

    def test_candidate_has_one_stability_target_and_six_directional_actions(self):
        record = _record()
        self.assertEqual(record["targets"]["T"], {"value": True, "valid": True})
        self.assertEqual(len(record["actions"]), 6)
        self.assertEqual(
            [action["direction_id"] for action in record["actions"]],
            [item[0] for item in features.dataset_index.DIRECTIONS],
        )
        self.assertTrue(
            all(
                math.isfinite(value)
                for value in record["features"].values()
                for _ in (0,)
            )
        )

    def test_directional_blockers_and_normalization(self):
        record = _record()
        by_direction = {action["direction_id"]: action for action in record["actions"]}
        self.assertGreater(
            by_direction["+X"]["features"][
                "directional_clearance__blocker_count_fraction"
            ],
            0.0,
        )
        self.assertEqual(
            by_direction["-X"]["features"][
                "directional_clearance__has_no_aabb_blocker"
            ],
            1.0,
        )
        translated = _record(x_translation=10.0)
        scaled = _record(scale=2.0)
        np.testing.assert_allclose(
            list(record["features"].values()),
            list(translated["features"].values()),
            rtol=1e-7,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            list(record["features"].values()),
            list(scaled["features"].values()),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_feature_names_exclude_simulator_outputs(self):
        names = features.CANDIDATE_FEATURES + features.ACTION_FEATURES
        self.assertEqual(set(features._feature_groups(names)), set(features.FEATURE_GROUPS))
        for name in names:
            self.assertFalse(
                any(term in name for term in features.FORBIDDEN_FEATURE_TERMS), name
            )


if __name__ == "__main__":
    unittest.main()
