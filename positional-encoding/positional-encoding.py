import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model)
    containing sinusoidal positional encodings.
    """
    # Positions: (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]

    # Dimension indices: (1, d_model)
    dimensions = np.arange(d_model)[np.newaxis, :]

    # Angle rates
    angle_rates = 1 / np.power(base, (2 * (dimensions // 2)) / d_model)

    # Angles for each position and dimension
    angles = positions * angle_rates

    # Apply sin to even dimensions and cos to odd dimensions
    pe = np.empty((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe