import numpy as np

HYPERPARAMETER_SPACE_NN = {
    "hidden_size": [100, 200, 300, 400, 500],  # Discrete
    "epochs": np.linspace(1, 10, 10, dtype=int).tolist(),  # Linear: [1, 2, ..., 10]
    "learning_rate": np.logspace(-5, -1, 20).tolist(),  # Log: 10^-5 to 10^-1
    "batch_size": [16, 32, 64, 128, 256],  # Discrete
    "dropout_rate": np.linspace(0, 0.5, 20).tolist(),  # Linear: [0, 0.026, ..., 0.5]
    "weight_decay": np.logspace(-6, -2, 20).tolist(),  # Log: 10^-6 to 10^-2
}


def sample_random_hyperparams_NN(space):
    """Sample random hyperparameters from the space."""
    return {
        "hidden_size": int(np.random.choice(space["hidden_size"])),
        "epochs": int(np.random.choice(space["epochs"])),
        "learning_rate": float(np.random.choice(space["learning_rate"])),
        "batch_size": int(np.random.choice(space["batch_size"])),
        "dropout_rate": float(np.random.choice(space["dropout_rate"])),
        "weight_decay": float(np.random.choice(space["weight_decay"])),
    }


def hyperparams_to_vector_NN(hyperparams):
    """Convert hyperparameters to a normalized vector for GP."""
    # Normalize hyperparameters to [0, 1] range
    # Hidden size: map [100, 500] to [0, 1]
    # Epochs: map [1, 10] to [0, 1]
    # Learning rate: log scale already, map log10([1e-5, 1e-1]) = [-5, -1] to [0, 1]
    # Batch size: map [16, 256] to [0, 1] (log scale)
    # Dropout: map [0, 0.5] to [0, 1]
    # Weight decay: log scale already, map log10([1e-6, 1e-2]) = [-6, -2] to [0, 1]

    vec = np.array(
        [
            (hyperparams["hidden_size"] - 100) / 400.0,  # [0, 1]
            (hyperparams["epochs"] - 1) / 9.0,  # [0, 1]
            (np.log10(hyperparams["learning_rate"]) + 5) / 4.0,  # [0, 1]
            (np.log2(hyperparams["batch_size"]) - 4) / 4.0,  # [0, 1] (log2)
            hyperparams["dropout_rate"] / 0.5,  # [0, 1]
            (np.log10(hyperparams["weight_decay"]) + 6) / 4.0,  # [0, 1]
        ]
    )
    return vec


def vector_to_hyperparams_NN(vec, space):
    """Convert normalized vector back to hyperparameters (snap to nearest discrete value)."""
    # Denormalize
    hidden_size_val = vec[0] * 400 + 100
    epochs_val = vec[1] * 9 + 1
    lr_val = 10 ** (vec[2] * 4 - 5)
    batch_size_val = 2 ** (vec[3] * 4 + 4)
    dropout_val = vec[4] * 0.5
    weight_decay_val = 10 ** (vec[5] * 4 - 6)

    # Snap to nearest discrete values
    hidden_size = min(space["hidden_size"], key=lambda x: abs(x - hidden_size_val))
    epochs = int(min(space["epochs"], key=lambda x: abs(x - epochs_val)))
    learning_rate = min(space["learning_rate"], key=lambda x: abs(x - lr_val))
    batch_size = min(space["batch_size"], key=lambda x: abs(x - batch_size_val))
    dropout_rate = min(space["dropout_rate"], key=lambda x: abs(x - dropout_val))
    weight_decay = min(space["weight_decay"], key=lambda x: abs(x - weight_decay_val))

    return {
        "hidden_size": int(hidden_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "dropout_rate": float(dropout_rate),
        "weight_decay": float(weight_decay),
    }


HYPERPARAMETER_UTILS_NN = {
    "space": HYPERPARAMETER_SPACE_NN,
    "num_candidates": (
        len(HYPERPARAMETER_SPACE_NN["hidden_size"])
        * len(HYPERPARAMETER_SPACE_NN["epochs"])
        * len(HYPERPARAMETER_SPACE_NN["learning_rate"])
        * len(HYPERPARAMETER_SPACE_NN["batch_size"])
        * len(HYPERPARAMETER_SPACE_NN["dropout_rate"])
        * len(HYPERPARAMETER_SPACE_NN["weight_decay"])
    ),
    "sample_random_hyperparams": sample_random_hyperparams_NN,
    "hyperparams_to_vector": hyperparams_to_vector_NN,
    "vector_to_hyperparams": vector_to_hyperparams_NN,
}

HYPERPARAMETER_SPACE_CNN = {
    "hidden_size_fc": [100, 200, 300, 400, 500],  # Discrete
    "epochs": np.linspace(1, 10, 10, dtype=int).tolist(),  # Linear: [1, 2, ..., 10]
    "learning_rate": np.logspace(-5, -1, 20).tolist(),  # Log: 10^-5 to 10^-1
    "batch_size": [16, 32, 64, 128, 256],  # Discrete
    "weight_decay": np.logspace(-6, -2, 20).tolist(),  # Log: 10^-6 to 10^-2
    "dropout_rate_conv": np.linspace(
        0, 0.5, 20
    ).tolist(),  # Linear: [0, 0.026, ..., 0.5]
    "dropout_rate_fc": np.linspace(0, 0.5, 20).tolist(),  # Linear: [0, 0.026, ..., 0.5]
    "kernel_size": [3, 5, 7],  # Discrete
}


def sample_random_hyperparams_CNN(space):
    """Sample random hyperparameters from the space."""
    return {
        "hidden_size_fc": int(np.random.choice(space["hidden_size_fc"])),
        "epochs": int(np.random.choice(space["epochs"])),
        "learning_rate": float(np.random.choice(space["learning_rate"])),
        "batch_size": int(np.random.choice(space["batch_size"])),
        "weight_decay": float(np.random.choice(space["weight_decay"])),
        "dropout_rate_conv": float(np.random.choice(space["dropout_rate_conv"])),
        "dropout_rate_fc": float(np.random.choice(space["dropout_rate_fc"])),
        "kernel_size": int(np.random.choice(space["kernel_size"])),
    }


def hyperparams_to_vector_CNN(hyperparams):
    """Convert hyperparameters to a normalized vector for GP."""
    # Normalize hyperparameters to [0, 1] range
    # Hidden size: map [100, 500] to [0, 1]
    # Epochs: map [1, 10] to [0, 1]
    # Learning rate: log scale already, map log10([1e-5, 1e-1]) = [-5, -1] to [0, 1]
    # Batch size: map [16, 256] to [0, 1] (log scale)
    # Dropout: map [0, 0.5] to [0, 1]
    # Weight decay: log scale already, map log10([1e-6, 1e-2]) = [-6, -2] to [0, 1]

    vec = np.array(
        [
            (hyperparams["hidden_size_fc"] - 100) / 400.0,  # [0, 1]
            (hyperparams["epochs"] - 1) / 9.0,  # [0, 1]
            (np.log10(hyperparams["learning_rate"]) + 5) / 4.0,  # [0, 1]
            (np.log2(hyperparams["batch_size"]) - 4) / 4.0,  # [0, 1] (log2)
            (np.log10(hyperparams["weight_decay"]) + 6) / 4.0,  # [0, 1]
            hyperparams["dropout_rate_conv"] / 0.5,  # [0, 1]
            hyperparams["dropout_rate_fc"] / 0.5,  # [0, 1]
            (hyperparams["kernel_size"] - 3) / 2.0,  # [0, 1]
        ]
    )
    return vec


def vector_to_hyperparams_CNN(vec, space):
    """Convert normalized vector back to hyperparameters (snap to nearest discrete value)."""
    # Denormalize
    hidden_size_fc_val = vec[0] * 400 + 100
    epochs_val = vec[1] * 9 + 1
    lr_val = 10 ** (vec[2] * 4 - 5)
    batch_size_val = 2 ** (vec[3] * 4 + 4)
    weight_decay_val = 10 ** (vec[4] * 4 - 6)
    dropout_rate_conv_val = vec[5] * 0.5
    dropout_rate_fc_val = vec[6] * 0.5
    kernel_size_val = vec[7] * 2 + 3

    # Snap to nearest discrete values
    hidden_size_fc = min(
        space["hidden_size_fc"], key=lambda x: abs(x - hidden_size_fc_val)
    )
    epochs = int(min(space["epochs"], key=lambda x: abs(x - epochs_val)))
    learning_rate = min(space["learning_rate"], key=lambda x: abs(x - lr_val))
    batch_size = min(space["batch_size"], key=lambda x: abs(x - batch_size_val))
    weight_decay = min(space["weight_decay"], key=lambda x: abs(x - weight_decay_val))
    dropout_rate_conv = min(
        space["dropout_rate_conv"], key=lambda x: abs(x - dropout_rate_conv_val)
    )
    dropout_rate_fc = min(
        space["dropout_rate_fc"], key=lambda x: abs(x - dropout_rate_fc_val)
    )
    kernel_size = min(space["kernel_size"], key=lambda x: abs(x - kernel_size_val))

    return {
        "hidden_size_fc": int(hidden_size_fc),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "dropout_rate_conv": float(dropout_rate_conv),
        "dropout_rate_fc": float(dropout_rate_fc),
        "kernel_size": int(kernel_size),
    }


HYPERPARAMETER_UTILS_CNN = {
    "space": HYPERPARAMETER_SPACE_CNN,
    "num_candidates": (
        len(HYPERPARAMETER_SPACE_CNN["hidden_size_fc"])
        * len(HYPERPARAMETER_SPACE_CNN["epochs"])
        * len(HYPERPARAMETER_SPACE_CNN["learning_rate"])
        * len(HYPERPARAMETER_SPACE_CNN["batch_size"])
        * len(HYPERPARAMETER_SPACE_CNN["weight_decay"])
        * len(HYPERPARAMETER_SPACE_CNN["dropout_rate_conv"])
        * len(HYPERPARAMETER_SPACE_CNN["dropout_rate_fc"])
        * len(HYPERPARAMETER_SPACE_CNN["kernel_size"])
    ),
    "sample_random_hyperparams": sample_random_hyperparams_CNN,
    "hyperparams_to_vector": hyperparams_to_vector_CNN,
    "vector_to_hyperparams": vector_to_hyperparams_CNN,
}
