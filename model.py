"""
Clustering from Scratch in NumPy: HAC, K-Means++, Medoids, GMMs and Spectral Cuts

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - pairwise_distances
import numpy as np

def pairwise_distances(X, metric="euclidean"):
    """
    Compute the pairwise dissimilarity matrix between rows of X.

    Supported metrics:
        - euclidean:    Euclidean distance
        - sqeuclidean:  Squared Euclidean distance
        - cityblock:    Manhattan (L1) distance
        - correlation:  1 - Pearson correlation

    Parameters
    ----------
    X : array-like of shape (N, D)
        Input data matrix containing N observations and D features.

    metric : str, default="euclidean"
        Distance metric to use.

    Returns
    -------
    D : numpy.ndarray of shape (N, N)
        Symmetric pairwise dissimilarity matrix with a zero diagonal
        and no negative entries.
    """

    # Validate the requested distance metric.
    valid_metrics = ("euclidean", "sqeuclidean", "cityblock", "correlation")

    if metric not in valid_metrics:
        raise ValueError(f"Unsupported metric: {metric}")

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Ensure that the input is a two-dimensional data matrix.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    n_samples, n_features = X.shape

    # Handle empty dimensions.
    if n_samples == 0 or n_features == 0:
        return np.zeros((n_samples, n_samples), dtype=float)

    # ---------------------------------------------------------
    # 1. Euclidean and Squared Euclidean Distances
    # ---------------------------------------------------------
    if metric in ("euclidean", "sqeuclidean"):

        # Calculate the squared L2 norm of each observation.
        squared_norms = np.sum(X ** 2, axis=1)

        # Use the identity:
        # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2(xi . xj)
        squared_distances = (
            squared_norms[:, np.newaxis]
            + squared_norms[np.newaxis, :]
            - 2.0 * (X @ X.T)
        )

        # Remove small negative values caused by floating-point rounding.
        np.maximum(squared_distances, 0.0, out=squared_distances)

        if metric == "euclidean":
            D = np.sqrt(squared_distances)
        else:
            D = squared_distances

    # ---------------------------------------------------------
    # 2. Cityblock (Manhattan) Distance
    # ---------------------------------------------------------
    elif metric == "cityblock":

        # Allocate the pairwise distance matrix.
        D = np.empty((n_samples, n_samples), dtype=float)

        # Process observations in blocks to avoid allocating a
        # very large (N, N, D) intermediate array.
        max_elements = 8_000_000

        block_size = max(
            1,
            min(
                n_samples,
                max_elements // (n_samples * n_features)
            )
        )

        for start in range(0, n_samples, block_size):

            end = min(start + block_size, n_samples)

            # Broadcast the current block against all observations.
            differences = (
                X[start:end, np.newaxis, :]
                - X[np.newaxis, :, :]
            )

            # Sum absolute feature-wise differences.
            D[start:end] = np.sum(np.abs(differences), axis=2)

    # ---------------------------------------------------------
    # 3. Correlation Distance
    # ---------------------------------------------------------
    elif metric == "correlation":

        # Center each observation by subtracting its feature mean.
        centered = X - np.mean(X, axis=1, keepdims=True)

        # Calculate the L2 norm of each centered observation.
        norms = np.linalg.norm(centered, axis=1, keepdims=True)

        # Normalize each centered observation.
        # Constant rows have zero norm; use zero vectors for them
        # to avoid division by zero.
        normalized = np.divide(
            centered,
            norms,
            out=np.zeros_like(centered),
            where=norms > 0
        )

        # Pearson correlation is the dot product of normalized,
        # mean-centered observations.
        correlations = normalized @ normalized.T

        # Account for minor floating-point errors.
        np.clip(correlations, -1.0, 1.0, out=correlations)

        # Convert correlation into dissimilarity.
        D = 1.0 - correlations

    # ---------------------------------------------------------
    # 4. Final Numerical Corrections
    # ---------------------------------------------------------

    # Enforce exact symmetry.
    D = 0.5 * (D + D.T)

    # Ensure that no negative dissimilarities remain.
    np.maximum(D, 0.0, out=D)

    # Ensure that every observation has exactly zero distance
    # from itself.
    np.fill_diagonal(D, 0.0)

    return D

# Step 2 - contingency_table
def contingency_table(labels_pred, labels_true):
    """
    Construct a contingency table showing the overlap between
    predicted cluster labels and true class labels.

    Parameters
    ----------
    labels_pred : array-like of shape (N,)
        Predicted cluster labels.

    labels_true : array-like of shape (N,)
        Ground-truth class labels.

    Returns
    -------
    N : numpy.ndarray of shape (n_pred, n_true)
        Integer contingency table where N[i, j] represents
        the number of observations belonging to predicted
        cluster i and true class j.

        Rows and columns correspond to sorted distinct labels.
    """

    # Convert inputs into NumPy arrays.
    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)

    # Ensure both inputs are one-dimensional label arrays.
    if labels_pred.ndim != 1 or labels_true.ndim != 1:
        raise ValueError("Both label arrays must be one-dimensional.")

    # Both label arrays must contain the same number of observations.
    if len(labels_pred) != len(labels_true):
        raise ValueError("Both label arrays must have the same length.")

    # Obtain sorted unique predicted labels and their corresponding indices.
    pred_labels, pred_indices = np.unique(
        labels_pred,
        return_inverse=True
    )

    # Obtain sorted unique true labels and their corresponding indices.
    true_labels, true_indices = np.unique(
        labels_true,
        return_inverse=True
    )

    # Determine the number of predicted clusters and true classes.
    n_pred = len(pred_labels)
    n_true = len(true_labels)

    # Initialize the contingency table with integer zeros.
    N = np.zeros((n_pred, n_true), dtype=int)

    # Count the occurrences of each predicted-cluster / true-class pair.
    # np.add.at handles repeated indices correctly.
    np.add.at(N, (pred_indices, true_indices), 1)

    return N

