import argparse

def get_args():
    """
    Returns a namespace containing the arguments that define the
    FeasibilityTransformer model architecture.

    These values should match the ones used during the training
    of the model checkpoint you intend to load.
    """
    args = argparse.Namespace()

    # --- PointNet Encoder Hyperparameters ---
    # feat_dim is the output dimension of the final global feature vector
    args.pointnet_dim = 256

    # --- Custom Feature Encoders Hyperparameters ---
    # feat_dim for the MLP encoding properties (volume, etc.) and actions
    args.custom_mlp_dim = 128

    # Bias correction (for [0,0,-1] action direction skew)
    args.use_bias_correction = False

    # --- Graph Transformer Hyperparameters ---
    # The internal dimension of the transformer layers
    args.transformer_dim = 256
    # Number of attention heads in each transformer layer
    args.num_heads = 8
    # Number of transformer layers to stack
    args.num_transformer_layers = 4

    # --- General Model Hyperparameters ---
    args.dropout = 0.1

    # --- Data parameters relevant for the model ---
    args.max_pc_size = 1000

    return args
