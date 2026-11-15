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

# Step 3 - pair_counts
def pair_counts(N):
    """
    Calculate pairwise clustering agreement counts from a contingency table.

    Parameters
    ----------
    N : array-like of shape (n_pred, n_true)
        Contingency table where N[i, j] represents the number of
        observations in predicted cluster i and true class j.

    Returns
    -------
    tuple
        (TP, FP, FN, TN), where all four values are Python integers.

        TP: Pairs in the same predicted cluster and same true class.
        FP: Pairs in the same predicted cluster but different true classes.
        FN: Pairs in different predicted clusters but the same true class.
        TN: Pairs in different predicted clusters and different true classes.
    """

    # Convert the contingency table into a NumPy array.
    N = np.asarray(N)

    # Ensure that the contingency table is two-dimensional.
    if N.ndim != 2:
        raise ValueError("N must be a two-dimensional contingency table.")

    # Count the total number of pairs within each individual cell.
    # These pairs belong to both the same predicted cluster
    # and the same true class.
    TP = sum(
        int(n) * (int(n) - 1) // 2
        for n in N.flat
    )

    # Calculate the row sums (total observations in each predicted cluster).
    row_sums = [
        sum(int(value) for value in row)
        for row in N
    ]

    # Calculate the column sums (total observations in each true class).
    col_sums = [
        sum(int(value) for value in N[:, j])
        for j in range(N.shape[1])
    ]

    # Count all pairs belonging to the same predicted cluster.
    same_cluster_pairs = sum(
        n * (n - 1) // 2
        for n in row_sums
    )

    # Count all pairs belonging to the same true class.
    same_class_pairs = sum(
        n * (n - 1) // 2
        for n in col_sums
    )

    # False positives: same predicted cluster, different true classes.
    FP = same_cluster_pairs - TP

    # False negatives: different predicted clusters, same true class.
    FN = same_class_pairs - TP

    # Calculate the total number of unordered pairs.
    total_observations = sum(row_sums)
    total_pairs = (
        total_observations * (total_observations - 1) // 2
    )

    # True negatives: pairs that differ in both cluster and class.
    TN = total_pairs - TP - FP - FN

    # Return all counts as standard Python integers.
    return (int(TP), int(FP), int(FN), int(TN))

# Step 4 - cluster_scores
def cluster_scores(labels_pred, labels_true):
    """
    Compute external clustering evaluation scores.

    Parameters
    ----------
    labels_pred : array-like of shape (N,)
        Predicted cluster labels.

    labels_true : array-like of shape (N,)
        Ground-truth reference labels.

    Returns
    -------
    dict
        Dictionary containing the following scores:
            purity : Cluster purity.
            rand   : Rand Index.
            ari    : Adjusted Rand Index.
            mi     : Mutual Information (in nats).
            nmi    : Normalized Mutual Information.
    """

    # Construct the contingency table using the previously implemented function.
    N = contingency_table(labels_pred, labels_true)

    # Calculate the total number of observations.
    n_samples = int(np.sum(N))

    # Purity and information-theoretic scores are undefined for empty data.
    if n_samples == 0:
        raise ValueError("Clustering labels must contain at least one observation.")

    # ---------------------------------------------------------
    # 1. Purity
    # ---------------------------------------------------------

    # For each predicted cluster, find the number of observations
    # belonging to its most frequent true class.
    purity = float(np.sum(np.max(N, axis=1)) / n_samples)

    # ---------------------------------------------------------
    # 2. Rand Index and Adjusted Rand Index
    # ---------------------------------------------------------

    # Obtain True Positive, False Positive, False Negative,
    # and True Negative pair counts.
    TP, FP, FN, TN = pair_counts(N)

    # Calculate the total number of unordered observation pairs.
    total_pairs = TP + FP + FN + TN

    # Rand Index: proportion of pairs on which both partitions agree.
    # When there are no pairs, use the conventional perfect-agreement
    # value of 1.0.
    if total_pairs == 0:
        rand = 1.0
    else:
        rand = float((TP + TN) / total_pairs)

    # Expected number of agreements under the generalized
    # random-partition model.
    if total_pairs == 0:
        expected_index = 0.0
    else:
        expected_index = (
            (TP + FP) * (TP + FN)
        ) / total_pairs

    # Maximum index used in the Adjusted Rand Index denominator.
    max_index = 0.5 * (
        (TP + FP) + (TP + FN)
    )

    # Adjusted Rand Index corrects the Rand Index for chance agreement.
    ari_denominator = max_index - expected_index

    # Handle the zero-denominator edge case.
    if ari_denominator == 0.0:
        ari = 1.0
    else:
        ari = float(
            (TP - expected_index) / ari_denominator
        )

    # ---------------------------------------------------------
    # 3. Mutual Information (MI)
    # ---------------------------------------------------------

    # Calculate the row and column marginal counts.
    row_sums = np.sum(N, axis=1).astype(float)
    col_sums = np.sum(N, axis=0).astype(float)

    # Convert the contingency table into joint probabilities.
    joint_prob = N.astype(float) / n_samples

    # Calculate marginal probabilities.
    row_prob = row_sums / n_samples
    col_prob = col_sums / n_samples

    # Only nonzero joint probabilities contribute to MI.
    nonzero = joint_prob > 0

    # Mutual Information:
    # MI(U, V) = sum_ij p(i,j) * log(p(i,j) / (p(i)*p(j)))
    mi = float(
        np.sum(
            joint_prob[nonzero]
            * np.log(
                joint_prob[nonzero]
                / (
                    row_prob[:, np.newaxis]
                    * col_prob[np.newaxis, :]
                )[nonzero]
            )
        )
    )

    # Remove any small negative value caused by floating-point rounding.
    mi = max(mi, 0.0)

    # ---------------------------------------------------------
    # 4. Normalized Mutual Information (NMI)
    # ---------------------------------------------------------

    # Calculate the entropy of the predicted clustering U.
    H_pred = -np.sum(
        row_prob[row_prob > 0]
        * np.log(row_prob[row_prob > 0])
    )

    # Calculate the entropy of the reference labels V.
    H_true = -np.sum(
        col_prob[col_prob > 0]
        * np.log(col_prob[col_prob > 0])
    )

    # Use the arithmetic mean of the two entropies for normalization.
    nmi_denominator = 0.5 * (H_pred + H_true)

    # Handle the case where both partitions contain no uncertainty.
    if nmi_denominator == 0.0:
        nmi = 1.0
    else:
        nmi = float(mi / nmi_denominator)

    # ---------------------------------------------------------
    # 5. Return all evaluation scores
    # ---------------------------------------------------------

    return {
        "purity": float(purity),
        "rand": float(rand),
        "ari": float(ari),
        "mi": float(mi),
        "nmi": float(nmi)
    }

# Step 5 - agglomerate
def agglomerate(D, linkage="average"):
    """
    Perform hierarchical agglomerative clustering using a dissimilarity matrix.

    Parameters
    ----------
    D : array-like of shape (N, N)
        Symmetric pairwise dissimilarity matrix.

    linkage : str, default="average"
        Linkage criterion:
            - "single": Minimum pairwise distance.
            - "complete": Maximum pairwise distance.
            - "average": Size-weighted average pairwise distance.

    Returns
    -------
    Z : numpy.ndarray of shape (N - 1, 4)
        Linkage matrix where each row contains:
            [cluster_a, cluster_b, merge_height, cluster_size]

        Original observations have IDs 0 to N-1.
        The cluster created at step t has ID N+t.

    Notes
    -----
    The input dissimilarity matrix is not modified.
    Ties are resolved by choosing the lexicographically smallest
    pair of active cluster IDs.
    """

    # Validate the linkage criterion.
    valid_linkages = ("single", "complete", "average")

    if linkage not in valid_linkages:
        raise ValueError(f"Unsupported linkage: {linkage}")

    # Convert the input into a floating-point NumPy array.
    D = np.asarray(D, dtype=float)

    # Ensure that the dissimilarity matrix is square.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square two-dimensional matrix.")

    # Determine the number of original observations.
    n = D.shape[0]

    if n == 0:
        raise ValueError("D must contain at least one observation.")

    # Allocate the linkage matrix.
    # Each merge records [a, b, height, cluster_size].
    Z = np.empty((n - 1, 4), dtype=float)

    # No merges are required for a single observation.
    if n == 1:
        return Z

    # ---------------------------------------------------------
    # 1. Initialize the working dissimilarity matrix
    # ---------------------------------------------------------

    # Reserve space for all original and newly created clusters.
    # The maximum cluster ID is 2*N - 2.
    work = np.full(
        (2 * n - 1, 2 * n - 1),
        np.inf,
        dtype=float
    )

    # Copy the original dissimilarities.
    # The input matrix D remains unchanged.
    work[:n, :n] = D

    # Track the number of original observations in each cluster.
    sizes = np.zeros(2 * n - 1, dtype=np.int64)
    sizes[:n] = 1

    # Track which clusters are currently active.
    active = np.zeros(2 * n - 1, dtype=bool)
    active[:n] = True

    # ---------------------------------------------------------
    # 2. Perform N - 1 agglomerative merges
    # ---------------------------------------------------------

    for t in range(n - 1):

        # Obtain the IDs of all currently active clusters.
        # np.flatnonzero returns them in ascending order.
        active_ids = np.flatnonzero(active)

        # Generate upper-triangular indices to consider each
        # unordered cluster pair exactly once.
        upper_i, upper_j = np.triu_indices(
            len(active_ids),
            k=1
        )

        # Retrieve the current dissimilarities between active pairs.
        pair_distances = work[
            active_ids[upper_i],
            active_ids[upper_j]
        ]

        # Locate the smallest dissimilarity.
        # np.argmin selects the first occurrence, ensuring that
        # ties are resolved lexicographically by cluster IDs.
        best_idx = int(np.argmin(pair_distances))

        # Identify the two clusters to merge.
        a = int(active_ids[upper_i[best_idx]])
        b = int(active_ids[upper_j[best_idx]])

        # Record the dissimilarity at which the clusters merge.
        height = float(pair_distances[best_idx])

        # Assign the new cluster ID.
        new_id = n + t

        # Calculate the size of the resulting cluster.
        new_size = sizes[a] + sizes[b]

        # Store the merge information in the linkage matrix.
        Z[t] = [a, b, height, new_size]

        # -----------------------------------------------------
        # 3. Update distances using the selected linkage method
        # -----------------------------------------------------

        # Remove the two merged clusters from the active set.
        active[a] = False
        active[b] = False

        # Identify all remaining active clusters.
        other_ids = np.flatnonzero(active)

        # Retrieve distances from the two merged clusters
        # to every remaining active cluster.
        dist_a = work[a, other_ids]
        dist_b = work[b, other_ids]

        if linkage == "single":

            # Single linkage:
            # Distance to the new cluster is the minimum distance
            # to either of the two merged clusters.
            new_distances = np.minimum(dist_a, dist_b)

        elif linkage == "complete":

            # Complete linkage:
            # Distance to the new cluster is the maximum distance
            # to either of the two merged clusters.
            new_distances = np.maximum(dist_a, dist_b)

        else:

            # Average linkage:
            # Calculate the size-weighted average of the distances.
            new_distances = (
                sizes[a] * dist_a +
                sizes[b] * dist_b
            ) / new_size

        # Store the updated distances symmetrically.
        work[new_id, other_ids] = new_distances
        work[other_ids, new_id] = new_distances

        # Update the size of the newly created cluster.
        sizes[new_id] = new_size

        # Activate the newly created cluster.
        active[new_id] = True

    return Z

# Step 6 - cut_tree
def cut_tree(Z, n_clusters=None, height=None):
    """
    Convert a hierarchical clustering linkage matrix into flat cluster labels.

    Parameters
    ----------
    Z : array-like of shape (N - 1, 4)
        Linkage matrix produced by agglomerate().
        Each row contains [a, b, height, size].

    n_clusters : int, optional
        Desired number of clusters. Applies the first N - n_clusters merges.

    height : float, optional
        Distance threshold. Applies merges whose heights are
        less than or equal to this threshold.

    Returns
    -------
    labels : numpy.ndarray of shape (N,)
        Integer cluster labels assigned in order of first appearance.

    Raises
    ------
    ValueError
        If both or neither of n_clusters and height are provided,
        or if n_clusters is outside the valid range.
    """

    # Exactly one cutting criterion must be provided.
    if (n_clusters is None) == (height is None):
        raise ValueError(
            "Provide exactly one of n_clusters or height."
        )

    # Convert the linkage matrix to a NumPy array.
    Z = np.asarray(Z, dtype=float)

    # Validate the linkage matrix dimensions.
    if Z.ndim != 2 or Z.shape[1] != 4:
        raise ValueError("Z must have shape (N - 1, 4).")

    # Infer the number of original observations.
    n = Z.shape[0] + 1

    # ---------------------------------------------------------
    # 1. Determine how many merges to apply
    # ---------------------------------------------------------

    if n_clusters is not None:

        # Validate that the requested cluster count is an integer.
        if (
            isinstance(n_clusters, (bool, np.bool_))
            or not isinstance(n_clusters, (int, np.integer))
        ):
            raise ValueError("n_clusters must be an integer.")

        # A valid clustering must contain between 1 and N clusters.
        if not 1 <= n_clusters <= n:
            raise ValueError(
                "n_clusters must be between 1 and N."
            )

        # Apply exactly N - n_clusters merges.
        n_merges = n - n_clusters

    else:

        # Ensure that the height threshold is a scalar.
        if not np.isscalar(height):
            raise ValueError("height must be a scalar.")

        try:
            height = float(height)
        except (TypeError, ValueError):
            raise ValueError("height must be numeric.")

        # Count the consecutive merges whose heights are within
        # the specified threshold.
        merge_heights = Z[:, 2]

        eligible = merge_heights <= height

        # HAC linkage heights are non-decreasing, so eligible
        # merges form a prefix of the linkage matrix.
        n_merges = int(np.sum(eligible))

    # ---------------------------------------------------------
    # 2. Build the cluster parent structure
    # ---------------------------------------------------------

    # Original observations have IDs 0, 1, ..., N-1.
    # New clusters have IDs N, N+1, ..., 2*N-2.
    #
    # parent[c] stores the ID of the cluster that contains c.
    # A value of -1 indicates that the cluster has no parent.
    parent = np.full(2 * n - 1, -1, dtype=int)

    # Apply the selected merges in chronological order.
    for t in range(n_merges):

        # Retrieve the two clusters being merged.
        a = int(Z[t, 0])
        b = int(Z[t, 1])

        # The new cluster receives ID N + t.
        new_id = n + t

        # Record the parent of both merged clusters.
        parent[a] = new_id
        parent[b] = new_id

    # ---------------------------------------------------------
    # 3. Find the final cluster of each original observation
    # ---------------------------------------------------------

    labels = np.empty(n, dtype=int)

    # Map final cluster IDs to consecutive labels.
    cluster_to_label = {}

    for i in range(n):

        # Start from the original observation.
        root = i

        # Follow parent pointers until reaching the final
        # cluster containing this observation.
        while parent[root] != -1:
            root = parent[root]

        # Assign labels in order of first appearance.
        if root not in cluster_to_label:
            cluster_to_label[root] = len(cluster_to_label)

        labels[i] = cluster_to_label[root]

    return labels

