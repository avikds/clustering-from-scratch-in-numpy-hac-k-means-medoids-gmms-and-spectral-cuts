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

# Step 7 - single_link_mst
def single_link_mst(D):
    """
    Compute the single-link hierarchical clustering tree using
    a Minimum Spanning Tree (MST).

    Parameters
    ----------
    D : array-like of shape (N, N)
        Symmetric pairwise dissimilarity matrix.

    Returns
    -------
    Z : numpy.ndarray of shape (N - 1, 4)
        Linkage matrix where each row contains:
            [cluster_a, cluster_b, merge_height, cluster_size]

        Original observations have IDs 0 to N-1.
        The cluster created at step t has ID N+t.

    Notes
    -----
    - Prim's algorithm constructs the MST in O(N^2) time.
    - MST edges are processed in increasing order of weight.
    - Union-Find constructs the merge tree.
    - The input matrix D is not modified.
    """

    # Convert the input into a floating-point NumPy array.
    D = np.asarray(D, dtype=float)

    # Validate that the dissimilarity matrix is square.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square two-dimensional matrix.")

    n = D.shape[0]

    # Handle matrices containing zero or one observation.
    if n <= 1:
        return np.empty((0, 4), dtype=float)

    # ---------------------------------------------------------
    # 1. Construct the Minimum Spanning Tree using Prim's algorithm
    # ---------------------------------------------------------

    # Track which observations have already been added to the MST.
    in_tree = np.zeros(n, dtype=bool)

    # Store the minimum distance connecting each outside observation
    # to the currently constructed MST.
    best_distance = np.full(n, np.inf, dtype=float)

    # Store the corresponding parent observation in the MST.
    best_parent = np.full(n, -1, dtype=int)

    # Start constructing the MST from observation 0.
    best_distance[0] = 0.0

    # Store the selected MST edges as (u, v, weight).
    mst_edges = []

    for _ in range(n):

        # Find all observations that have not yet entered the MST.
        candidates = np.flatnonzero(~in_tree)

        # Select the observation with the smallest connection distance.
        # np.argmin selects the first occurrence, ensuring deterministic ties.
        v = int(candidates[np.argmin(best_distance[candidates])])

        # Add the selected observation to the MST.
        in_tree[v] = True

        # Record its connecting edge, except for the initial root.
        if best_parent[v] != -1:

            u = int(best_parent[v])

            # Store edge endpoints in ascending order.
            edge_a, edge_b = sorted((u, v))

            mst_edges.append(
                (edge_a, edge_b, float(best_distance[v]))
            )

        # Update the cheapest connection for every observation
        # that has not yet entered the MST.
        outside = ~in_tree

        candidate_distances = D[v, outside]
        current_distances = best_distance[outside]
        current_parents = best_parent[outside]

        # Update when a strictly smaller connection is found.
        better = candidate_distances < current_distances

        # For equal weights, prefer the smaller parent ID.
        tied_with_smaller_parent = (
            (candidate_distances == current_distances)
            & ((current_parents == -1) | (v < current_parents))
        )

        update = better | tied_with_smaller_parent

        # Apply the updates to the corresponding observations.
        outside_ids = np.flatnonzero(outside)
        update_ids = outside_ids[update]

        best_distance[update_ids] = D[v, update_ids]
        best_parent[update_ids] = v

    # ---------------------------------------------------------
    # 2. Sort MST edges by increasing weight
    # ---------------------------------------------------------

    # Endpoint ordering provides deterministic tie-breaking
    # when multiple MST edges have identical weights.
    mst_edges.sort(
        key=lambda edge: (edge[2], edge[0], edge[1])
    )

    # ---------------------------------------------------------
    # 3. Initialize Union-Find (Disjoint Set Union)
    # ---------------------------------------------------------

    # Initially, every original observation belongs to its own component.
    uf_parent = np.arange(n, dtype=int)

    # Track the size of each Union-Find component.
    uf_size = np.ones(n, dtype=int)

    # Track the dendrogram cluster ID represented by each component root.
    cluster_id = np.arange(n, dtype=int)

    # Track the number of original observations in each dendrogram cluster.
    cluster_sizes = np.ones(2 * n - 1, dtype=int)

    def find(x):
        """
        Find the representative of a Union-Find component
        using path compression.
        """
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]

        return x

    # Allocate the final linkage matrix.
    Z = np.empty((n - 1, 4), dtype=float)

    merge_count = 0

    # ---------------------------------------------------------
    # 4. Process sorted MST edges using Union-Find
    # ---------------------------------------------------------

    for u, v, weight in mst_edges:

        # Find the components containing the two edge endpoints.
        root_u = find(u)
        root_v = find(v)

        # Ignore edges that already belong to the same component.
        if root_u == root_v:
            continue

        # Retrieve the current dendrogram IDs of both components.
        a = int(cluster_id[root_u])
        b = int(cluster_id[root_v])

        # Ensure that the smaller cluster ID comes first.
        if a > b:
            a, b = b, a

        # Assign the ID of the newly created cluster.
        new_id = n + merge_count

        # Calculate the total number of observations in the new cluster.
        new_size = int(cluster_sizes[a] + cluster_sizes[b])

        # Record the merge in the linkage matrix.
        Z[merge_count] = [
            a,
            b,
            weight,
            new_size
        ]

        # Update the new cluster's size.
        cluster_sizes[new_id] = new_size

        # -----------------------------------------------------
        # 5. Union the two components by size
        # -----------------------------------------------------

        # Keep the larger component as the representative.
        if uf_size[root_u] < uf_size[root_v]:
            root_u, root_v = root_v, root_u

        # Merge the smaller component into the larger component.
        uf_parent[root_v] = root_u

        uf_size[root_u] += uf_size[root_v]

        # Associate the resulting component with the new cluster ID.
        cluster_id[root_u] = new_id

        merge_count += 1

        # The MST contains exactly N-1 useful merging edges.
        if merge_count == n - 1:
            break

    return Z

# Step 8 - same_hierarchy
def same_hierarchy(Z1, Z2):
    """
    Determine whether two hierarchical clustering linkage matrices
    describe the same family of clusters, ignoring merge heights,
    merge order, and intermediate cluster IDs.

    Parameters
    ----------
    Z1 : array-like of shape (N - 1, 4)
        First linkage matrix.

    Z2 : array-like of shape (N - 1, 4)
        Second linkage matrix.

    Returns
    -------
    bool
        True if both linkage matrices create the same family of clusters,
        otherwise False.
    """

    # Convert both linkage matrices into NumPy arrays.
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    # Validate the dimensions of both linkage matrices.
    if Z1.ndim != 2 or Z1.shape[1] != 4:
        raise ValueError("Z1 must have shape (N - 1, 4).")

    if Z2.ndim != 2 or Z2.shape[1] != 4:
        raise ValueError("Z2 must have shape (N - 1, 4).")

    # Both trees must describe the same number of original observations.
    if Z1.shape[0] != Z2.shape[0]:
        return False

    # Infer the number of original observations.
    n = Z1.shape[0] + 1

    def get_clusters(Z):
        """
        Reconstruct the family of clusters created by a linkage matrix.

        Each cluster is represented by a frozenset of original
        observation indices.
        """

        # Initially, every observation is its own cluster.
        members = {
            i: frozenset([i])
            for i in range(n)
        }

        # Store all clusters created by the merges.
        clusters = set()

        # Process merges in chronological order.
        for t, row in enumerate(Z):

            # Retrieve the two cluster IDs being merged.
            a = int(row[0])
            b = int(row[1])

            # Combine the original observations belonging to both clusters.
            merged_cluster = members[a] | members[b]

            # The newly created cluster receives ID N + t.
            new_id = n + t

            # Store the membership of the newly created cluster.
            members[new_id] = merged_cluster

            # Record this cluster independently of its numerical ID.
            clusters.add(merged_cluster)

        return clusters

    # Reconstruct the cluster families for both linkage matrices.
    clusters1 = get_clusters(Z1)
    clusters2 = get_clusters(Z2)

    # Two hierarchical trees are equivalent if their cluster families match.
    return clusters1 == clusters2

# Step 9 - linkage_report
def max_diameter(D, labels):
    """
    Calculate the maximum pairwise dissimilarity between
    any two observations belonging to the same cluster.

    Parameters
    ----------
    D : array-like of shape (N, N)
        Pairwise dissimilarity matrix.

    labels : array-like of shape (N,)
        Cluster assignments for the observations.

    Returns
    -------
    float
        Largest within-cluster dissimilarity.
        Returns 0.0 if every cluster is a singleton.
    """

    # Convert inputs into NumPy arrays.
    D = np.asarray(D, dtype=float)
    labels = np.asarray(labels)

    # Validate the dissimilarity matrix.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square two-dimensional matrix.")

    # Validate the cluster labels.
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")

    if len(labels) != D.shape[0]:
        raise ValueError("D and labels must contain the same number of observations.")

    # Obtain unique cluster labels and the cluster index of each observation.
    unique_labels, inverse = np.unique(
        labels,
        return_inverse=True
    )

    # Initialize the maximum within-cluster diameter.
    largest_diameter = 0.0

    # Examine each cluster.
    for cluster_idx in range(len(unique_labels)):

        # Find the observations belonging to the current cluster.
        members = np.flatnonzero(inverse == cluster_idx)

        # A singleton cluster has no distinct observation pairs.
        if len(members) < 2:
            continue

        # Extract the pairwise dissimilarities within this cluster.
        cluster_distances = D[np.ix_(members, members)]

        # Consider only the upper triangle, excluding the diagonal.
        pair_distances = cluster_distances[
            np.triu_indices(len(members), k=1)
        ]

        # Update the largest within-cluster dissimilarity.
        cluster_diameter = float(np.max(pair_distances))

        largest_diameter = max(
            largest_diameter,
            cluster_diameter
        )

    return float(largest_diameter)


def linkage_report(D, K):
    """
    Compare single, complete, and average hierarchical clustering.

    Parameters
    ----------
    D : array-like of shape (N, N)
        Pairwise dissimilarity matrix.

    K : int
        Desired number of clusters.

    Returns
    -------
    dict
        Dictionary containing one report for each linkage method.

        Each report contains:
            top_height   : Height of the final merge.
            largest      : Size of the largest cluster.
            singletons   : Number of singleton clusters.
            max_diameter : Maximum within-cluster dissimilarity.
    """

    # Convert the input into a floating-point NumPy array.
    D = np.asarray(D, dtype=float)

    # Validate the dissimilarity matrix.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square two-dimensional matrix.")

    # Define the three linkage methods to compare.
    linkages = ("single", "complete", "average")

    # Store the evaluation results for all three methods.
    report = {}

    # ---------------------------------------------------------
    # Compare each hierarchical clustering linkage method
    # ---------------------------------------------------------

    for linkage in linkages:

        # Build the hierarchical clustering tree.
        Z = agglomerate(D, linkage=linkage)

        # Cut the dendrogram into exactly K clusters.
        labels = cut_tree(
            Z,
            n_clusters=K
        )

        # Determine the size of each resulting cluster.
        _, cluster_sizes = np.unique(
            labels,
            return_counts=True
        )

        # Find the size of the largest cluster.
        largest = int(np.max(cluster_sizes))

        # Count the number of singleton clusters.
        singletons = int(np.sum(cluster_sizes == 1))

        # Obtain the height of the final merge.
        # A single-observation dataset has no merges.
        if len(Z) > 0:
            top_height = float(Z[-1, 2])
        else:
            top_height = 0.0

        # Calculate the largest within-cluster dissimilarity.
        diameter = max_diameter(D, labels)

        # Store the statistics for the current linkage method.
        report[linkage] = {
            "top_height": top_height,
            "largest": largest,
            "singletons": singletons,
            "max_diameter": float(diameter)
        }

    return report

# Step 10 - kmeans_lloyd
def assign_clusters(X, mu):
    """
    Assign each observation to its nearest cluster centroid
    and calculate the total K-means distortion.

    Parameters
    ----------
    X : array-like of shape (N, D)
        Input dataset containing N observations and D features.

    mu : array-like of shape (K, D)
        Current cluster centroids.

    Returns
    -------
    z : numpy.ndarray of shape (N,)
        Integer cluster assignments. Ties are assigned to
        the centroid with the lowest index.

    J : float
        Total K-means distortion, calculated as the sum
        of squared Euclidean distances to assigned centroids.
    """

    # Convert inputs to floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # Validate the input dimensions.
    if X.ndim != 2 or mu.ndim != 2:
        raise ValueError("X and mu must be two-dimensional arrays.")

    # Ensure that observations and centroids have matching features.
    if X.shape[1] != mu.shape[1]:
        raise ValueError("X and mu must have the same number of features.")

    # At least one centroid is required for cluster assignment.
    if mu.shape[0] == 0:
        raise ValueError("mu must contain at least one centroid.")

    # Calculate squared Euclidean distances between every observation
    # and every centroid using NumPy broadcasting.
    squared_distances = np.sum(
        (X[:, np.newaxis, :] - mu[np.newaxis, :, :]) ** 2,
        axis=2
    )

    # Assign each observation to its nearest centroid.
    # np.argmin selects the lowest centroid index when distances tie.
    z = np.argmin(squared_distances, axis=1).astype(int)

    # Extract the squared distance to each observation's assigned centroid.
    assigned_distances = squared_distances[
        np.arange(X.shape[0]),
        z
    ]

    # Calculate the total within-cluster sum of squared distances.
    J = float(np.sum(assigned_distances))

    return z, J


def kmeans_lloyd(X, mu0, max_iter=100):
    """
    Perform K-means clustering using Lloyd's alternating minimization.

    Parameters
    ----------
    X : array-like of shape (N, D)
        Input dataset containing N observations and D features.

    mu0 : array-like of shape (K, D)
        Initial cluster centroids.

    max_iter : int, default=100
        Maximum number of assignment and centroid-update iterations.

    Returns
    -------
    mu : numpy.ndarray of shape (K, D)
        Final cluster centroids.

    z : numpy.ndarray of shape (N,)
        Final cluster assignments.

    history : list of float
        Distortion recorded after every assignment step.
        Includes a final assignment if convergence is not reached
        within max_iter iterations.
    """

    # Convert the dataset into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Create an independent copy of the initial centroids.
    # This ensures that mu0 is never modified.
    mu = np.array(mu0, dtype=float, copy=True)

    # Validate input dimensions.
    if X.ndim != 2 or mu.ndim != 2:
        raise ValueError("X and mu0 must be two-dimensional arrays.")

    if X.shape[1] != mu.shape[1]:
        raise ValueError("X and mu0 must have the same number of features.")

    if mu.shape[0] == 0:
        raise ValueError("At least one initial centroid is required.")

    # Validate the maximum iteration count.
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 0
    ):
        raise ValueError("max_iter must be a non-negative integer.")

    # Store the distortion after each assignment step.
    history = []

    # Keep track of the assignments from the previous iteration.
    previous_z = None

    # ---------------------------------------------------------
    # Lloyd's alternating minimization
    # ---------------------------------------------------------

    for iteration in range(max_iter):

        # Step 1: Assign observations to their nearest centroids
        # and calculate the current distortion.
        z, J = assign_clusters(X, mu)

        # Record the distortion for this iteration.
        history.append(J)

        # Step 2: Check whether cluster assignments have converged.
        # If assignments are unchanged, the algorithm has converged.
        if (
            previous_z is not None
            and np.array_equal(z, previous_z)
        ):
            return mu, z, history

        # Step 3: Update each centroid using the mean of its
        # currently assigned observations.
        #
        # Start with a copy so empty clusters retain their old centres.
        new_mu = mu.copy()

        for k in range(mu.shape[0]):

            # Identify observations assigned to cluster k.
            cluster_points = X[z == k]

            # Update the centroid only if the cluster is non-empty.
            if len(cluster_points) > 0:

                # Calculate the arithmetic mean of assigned observations.
                new_mu[k] = np.mean(cluster_points, axis=0)

        # Update the centroids for the next iteration.
        mu = new_mu

        # Save the current assignments for the convergence check.
        previous_z = z.copy()

    # ---------------------------------------------------------
    # Final assignment when max_iter is reached
    # ---------------------------------------------------------

    # Perform one final assignment using the latest updated centroids.
    z, J = assign_clusters(X, mu)

    # Record the final distortion.
    history.append(J)

    return mu, z, history

# Step 11 - kmeanspp_init
def kmeanspp_init(X, K, rng):
    """
    Initialize K cluster centres using the K-Means++ algorithm.

    Parameters
    ----------
    X : array-like of shape (N, D)
        Input dataset containing N observations and D features.

    K : int
        Number of cluster centres to initialize.

    rng : numpy.random.Generator
        Random number generator used for reproducible initialization.

    Returns
    -------
    mu : numpy.ndarray of shape (K, D)
        Initial cluster centres selected from rows of X.
        The returned array has floating-point dtype.
    """

    # Convert the input dataset into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Validate that X is a two-dimensional data matrix.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    N, D = X.shape

    # Ensure that the dataset contains at least one observation.
    if N == 0:
        raise ValueError("X must contain at least one observation.")

    # Validate the requested number of cluster centres.
    if (
        isinstance(K, (bool, np.bool_))
        or not isinstance(K, (int, np.integer))
        or K < 1
    ):
        raise ValueError("K must be a positive integer.")

    # Allocate an array to store the K initial centres.
    mu = np.empty((K, D), dtype=float)

    # ---------------------------------------------------------
    # 1. Select the first centre uniformly at random
    # ---------------------------------------------------------

    # Choose an observation index using the supplied generator.
    first_idx = rng.integers(N)

    # Use the selected observation as the first centre.
    mu[0] = X[first_idx]

    # ---------------------------------------------------------
    # 2. Calculate squared distances to the nearest centre
    # ---------------------------------------------------------

    # Calculate the squared Euclidean distance from every
    # observation to the first selected centre.
    D2 = np.sum(
        (X - mu[0]) ** 2,
        axis=1
    )

    # ---------------------------------------------------------
    # 3. Select the remaining K - 1 centres
    # ---------------------------------------------------------

    for k in range(1, K):

        # Calculate the total squared distance.
        total_distance = np.sum(D2)

        if total_distance == 0.0:

            # If every observation already has zero distance
            # to its nearest selected centre, choose uniformly
            # at random as specified in the instructions.
            next_idx = rng.integers(N)

        else:

            # Calculate the probability of selecting each observation.
            # Observations farther from existing centres receive
            # higher selection probabilities.
            probabilities = D2 / total_distance

            # Select the next centre using the D-squared distribution.
            next_idx = rng.choice(
                N,
                p=probabilities
            )

        # Add the selected observation as the next centre.
        mu[k] = X[next_idx]

        # -----------------------------------------------------
        # 4. Update nearest-centre squared distances
        # -----------------------------------------------------

        # Calculate squared distances to the newly selected centre.
        new_D2 = np.sum(
            (X - mu[k]) ** 2,
            axis=1
        )

        # Retain the minimum squared distance to any selected centre.
        D2 = np.minimum(D2, new_D2)

    return mu

# Step 12 - kmeans_restarts
def kmeans_restarts(X, K, n_init, rng, init="kmeans++", max_iter=100):
    """
    Run K-means multiple times with different initializations
    and return the solution with the lowest final distortion.

    Parameters
    ----------
    X : numpy.ndarray of shape (N, D)
        Input dataset.

    K : int
        Number of clusters.

    n_init : int
        Number of independent K-means runs.

    rng : numpy.random.Generator
        Random number generator for reproducible initialization.

    init : str, default="kmeans++"
        Initialization method: "kmeans++" or "random".

    max_iter : int, default=100
        Maximum number of Lloyd iterations per run.

    Returns
    -------
    dict
        mu    : Best cluster centroids.
        z     : Best cluster assignments.
        J     : Lowest final distortion.
        all_J : Final distortions from all runs, in execution order.
    """

    # Validate the initialization method.
    if init not in ("kmeans++", "random"):
        raise ValueError("init must be 'kmeans++' or 'random'.")

    # Validate the number of initializations.
    if (
        isinstance(n_init, (bool, np.bool_))
        or not isinstance(n_init, (int, np.integer))
        or n_init < 1
    ):
        raise ValueError("n_init must be a positive integer.")

    # Convert the dataset into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    N = X.shape[0]

    if N == 0:
        raise ValueError("X must contain at least one observation.")

    # Store the final distortion from every run.
    all_J = []

    # Initialize the best solution.
    best_mu = None
    best_z = None
    best_J = np.inf

    # Run K-means from multiple initializations.
    for _ in range(n_init):

        # Select initial cluster centres.
        if init == "kmeans++":

            mu0 = kmeanspp_init(X, K, rng)

        else:

            # Randomly select K distinct observations.
            indices = rng.choice(
                N,
                size=K,
                replace=False
            )

            mu0 = X[indices].copy()

        # Run Lloyd's K-means algorithm.
        mu, z, history = kmeans_lloyd(
            X,
            mu0,
            max_iter=max_iter
        )

        # Obtain the final distortion of the current run.
        J = float(history[-1])

        # Record the distortion of this run.
        all_J.append(J)

        # Keep the solution with the lowest distortion.
        # Strict comparison preserves the earlier run when tied.
        if J < best_J:

            best_J = J
            best_mu = mu.copy()
            best_z = z.copy()

    # Return the best solution and the distortions of all runs.
    return {
        "mu": best_mu,
        "z": best_z,
        "J": float(best_J),
        "all_J": all_J
    }

# Step 13 - vq_bits
def vq_bits(N, D, K, B=8):
    """
    Calculate the number of bits required for vector quantization.

    Parameters
    ----------
    N : int
        Number of input vectors.

    D : int
        Number of scalars per vector.

    K : int
        Number of prototypes in the codebook.

    B : int, default=8
        Number of bits used to represent each scalar.

    Returns
    -------
    dict
        raw_bits      : Uncompressed storage size.
        code_bits     : Storage required for vector indices.
        codebook_bits : Storage required for the codebook.
        total_bits    : Total compressed storage size.
        ratio         : Compression ratio (raw_bits / total_bits).
    """

    # Validate the input parameters.
    values = (N, D, K, B)

    if any(
        isinstance(v, bool) or not isinstance(v, int)
        for v in values
    ):
        raise ValueError("N, D, K and B must be integers.")

    if N < 0 or D < 1 or K < 1 or B < 1:
        raise ValueError("Invalid vector quantization parameters.")

    # Calculate the uncompressed storage size.
    raw_bits = N * D * B

    # Calculate the number of bits required for each code index.
    # bit_length() computes ceil(log2(K)) exactly for K >= 1.
    bits_per_index = (K - 1).bit_length()

    # Calculate the total storage required for all code indices.
    code_bits = N * bits_per_index

    # Calculate the storage required for all codebook prototypes.
    codebook_bits = K * D * B

    # Calculate the total compressed storage size.
    total_bits = code_bits + codebook_bits

    # Calculate the compression ratio.
    ratio = float(raw_bits / total_bits)

    return {
        "raw_bits": int(raw_bits),
        "code_bits": int(code_bits),
        "codebook_bits": int(codebook_bits),
        "total_bits": int(total_bits),
        "ratio": ratio
    }

# Step 14 - block_vq
def block_vq(image, b, K, rng, n_init=1, max_iter=30):
    """
    Compress a grayscale image using block-based vector quantization.

    Parameters
    ----------
    image : array-like of shape (H, W)
        Grayscale image with pixel values between 0 and 255.

    b : int
        Side length of each non-overlapping square block.

    K : int
        Number of codebook prototypes.

    rng : numpy.random.Generator
        Random number generator for reproducible initialization.

    n_init : int, default=1
        Number of K-means initialization attempts.

    max_iter : int, default=30
        Maximum number of Lloyd iterations per initialization.

    Returns
    -------
    dict
        recon      : Reconstructed image.
        codes      : Integer index assigned to each block.
        codebook   : Rounded codebook of shape (K, b*b).
        mse        : Mean squared reconstruction error.
        psnr       : Peak signal-to-noise ratio in decibels.
        bpp_codes  : Bits per pixel used by block indices.
        bpp_total  : Bits per pixel including codebook storage.
    """

    # Convert the input image into a floating-point NumPy array.
    image = np.asarray(image, dtype=float)

    # Validate the image dimensions.
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional array.")

    # Validate the block size.
    if (
        isinstance(b, (bool, np.bool_))
        or not isinstance(b, (int, np.integer))
        or b <= 0
    ):
        raise ValueError("b must be a positive integer.")

    # Validate the number of codebook prototypes.
    if (
        isinstance(K, (bool, np.bool_))
        or not isinstance(K, (int, np.integer))
        or K < 1
    ):
        raise ValueError("K must be a positive integer.")

    # ---------------------------------------------------------
    # 1. Crop the image to the largest multiples of b
    # ---------------------------------------------------------

    H = (image.shape[0] // b) * b
    W = (image.shape[1] // b) * b

    # Keep only complete blocks.
    cropped = image[:H, :W]

    if H == 0 or W == 0:
        raise ValueError("Image must contain at least one complete block.")

    # ---------------------------------------------------------
    # 2. Extract non-overlapping b x b blocks
    # ---------------------------------------------------------

    # Reshape into block rows, within-block rows,
    # block columns, and within-block columns.
    blocks = cropped.reshape(
        H // b,
        b,
        W // b,
        b
    )

    # Group each square block together.
    # The order is left to right, then top to bottom.
    blocks = blocks.transpose(0, 2, 1, 3)

    # Flatten each block row by row into a vector of length b*b.
    blocks = blocks.reshape(-1, b * b)

    # Count the number of available blocks.
    n_blocks = blocks.shape[0]

    # The codebook cannot contain more prototypes than blocks.
    if K > n_blocks:
        raise ValueError("K cannot exceed the number of image blocks.")

    # ---------------------------------------------------------
    # 3. Learn the codebook using K-means++
    # ---------------------------------------------------------

    result = kmeans_restarts(
        blocks,
        K,
        n_init,
        rng,
        max_iter=max_iter
    )

    # Retrieve the learned cluster centres.
    mu = result["mu"]

    # ---------------------------------------------------------
    # 4. Round and clip the codebook
    # ---------------------------------------------------------

    # Round the centres to integer-valued pixel levels and
    # clip them to the valid grayscale range.
    #
    # IMPORTANT: Keep the floating-point dtype.
    # np.rint() produces rounded values without converting
    # the array into an integer dtype.
    codebook = np.clip(
        np.rint(mu),
        0.0,
        255.0
    )

    # ---------------------------------------------------------
    # 5. Re-encode blocks using the rounded codebook
    # ---------------------------------------------------------

    # Assign each original block to its nearest stored prototype.
    codes, _ = assign_clusters(
        blocks,
        codebook
    )

    # ---------------------------------------------------------
    # 6. Reconstruct the compressed image
    # ---------------------------------------------------------

    # Replace each block with its corresponding codebook vector.
    reconstructed_blocks = codebook[codes]

    # Restore the two-dimensional block arrangement.
    reconstructed_blocks = reconstructed_blocks.reshape(
        H // b,
        W // b,
        b,
        b
    )

    # Move the within-block row dimension back into image rows
    # and the within-block column dimension back into image columns.
    recon = reconstructed_blocks.transpose(0, 2, 1, 3)

    # Reconstruct the cropped grayscale image.
    # The result remains a floating-point array.
    recon = recon.reshape(H, W).astype(float)

    # ---------------------------------------------------------
    # 7. Calculate reconstruction quality
    # ---------------------------------------------------------

    # Calculate mean squared error against the cropped original image.
    mse = float(np.mean((cropped - recon) ** 2))

    # Calculate peak signal-to-noise ratio.
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(
            10.0 * np.log10((255.0 ** 2) / mse)
        )

    # ---------------------------------------------------------
    # 8. Calculate compression storage requirements
    # ---------------------------------------------------------

    # Calculate index and codebook storage using 8 bits per scalar.
    bits = vq_bits(
        n_blocks,
        b * b,
        K,
        B=8
    )

    # Total number of pixels in the cropped image.
    total_pixels = H * W

    # Bits per pixel used by the block indices.
    bpp_codes = float(
        bits["code_bits"] / total_pixels
    )

    # Total bits per pixel, including the codebook.
    bpp_total = float(
        bits["total_bits"] / total_pixels
    )

    # ---------------------------------------------------------
    # 9. Return the complete compression results
    # ---------------------------------------------------------

    return {
        "recon": recon,
        "codes": codes,
        "codebook": codebook,
        "mse": mse,
        "psnr": psnr,
        "bpp_codes": bpp_codes,
        "bpp_total": bpp_total
    }

# Step 15 - edit_distance_matrix
def edit_distance(a, b):
    """
    Calculate the Levenshtein edit distance between two strings.

    The edit distance is the minimum number of single-character
    insertions, deletions, and substitutions required to transform
    string a into string b.

    Parameters
    ----------
    a : str
        First input string.

    b : str
        Second input string.

    Returns
    -------
    int
        Minimum number of editing operations required.
    """

    # Use the shorter string as the second string to minimize memory usage.
    if len(a) < len(b):
        a, b = b, a

    m = len(a)
    n = len(b)

    # Initialize the previous dynamic programming row.
    # Converting an empty string into b requires n insertions.
    previous = list(range(n + 1))

    # Process each character of the first string.
    for i in range(1, m + 1):

        # Initialize the current row.
        # Converting the first i characters of a into an empty
        # string requires i deletions.
        current = [i] + [0] * n

        for j in range(1, n + 1):

            # No substitution is required when characters match.
            cost = 0 if a[i - 1] == b[j - 1] else 1

            # Calculate the minimum cost among:
            # 1. Deletion: previous[j] + 1
            # 2. Insertion: current[j - 1] + 1
            # 3. Substitution: previous[j - 1] + cost
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost
            )

        # Move to the next dynamic programming row.
        previous = current

    # The final cell contains the minimum edit distance.
    return int(previous[n])


def edit_distance_matrix(strings):
    """
    Compute the pairwise Levenshtein distance matrix for a collection
    of strings.

    Parameters
    ----------
    strings : array-like of shape (N,)
        Collection of input strings.

    Returns
    -------
    D : numpy.ndarray of shape (N, N)
        Symmetric integer matrix where D[i, j] is the edit distance
        between strings[i] and strings[j].

        The diagonal contains zeros.
    """

    # Convert the collection into a list for indexed access.
    strings = list(strings)

    # Determine the number of input strings.
    N = len(strings)

    # Initialize an integer distance matrix with zeros.
    D = np.zeros((N, N), dtype=int)

    # Calculate each unordered pair exactly once.
    for i in range(N):
        for j in range(i + 1, N):

            # Calculate the Levenshtein distance.
            distance = edit_distance(strings[i], strings[j])

            # Store the result symmetrically.
            D[i, j] = distance
            D[j, i] = distance

    return D

# Step 16 - kmedoids_voronoi
def kmedoids_voronoi(D, m0, max_iter=100):
    """
    Perform K-Medoids clustering using Voronoi iteration
    on a precomputed dissimilarity matrix.

    Parameters
    ----------
    D : array-like of shape (N, N)
        Pairwise dissimilarity matrix.

    m0 : array-like of shape (K,)
        Initial medoid indices. Each value identifies an
        original observation in the dataset.

    max_iter : int, default=100
        Maximum number of assignment and medoid-update rounds.

    Returns
    -------
    m : numpy.ndarray of shape (K,)
        Final medoid indices, in cluster order.

    z : numpy.ndarray of shape (N,)
        Integer cluster assignments.

    history : list of float
        Total assignment cost recorded after every assignment.
    """

    # Convert the dissimilarity matrix to a floating-point array.
    D = np.asarray(D, dtype=float)

    # Create an independent integer copy of the initial medoids.
    # This ensures that m0 is never modified.
    m = np.array(m0, dtype=int, copy=True)

    # Validate the dissimilarity matrix.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square two-dimensional matrix.")

    N = D.shape[0]

    if N == 0:
        raise ValueError("D must contain at least one observation.")

    # Validate the initial medoid indices.
    if m.ndim != 1 or len(m) == 0:
        raise ValueError("m0 must be a non-empty one-dimensional array.")

    if np.any(m < 0) or np.any(m >= N):
        raise ValueError("Medoid indices must be valid observation indices.")

    # Validate the maximum iteration count.
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 0
    ):
        raise ValueError("max_iter must be a non-negative integer.")

    # Store the assignment cost after each iteration.
    history = []

    # ---------------------------------------------------------
    # 1. Voronoi assignment and medoid-update iterations
    # ---------------------------------------------------------

    for iteration in range(max_iter):

        # Assign every observation to its nearest medoid.
        # Columns correspond to medoids in their existing cluster order.
        # np.argmin resolves ties in favour of the lowest cluster index.
        distances = D[:, m]

        z = np.argmin(distances, axis=1).astype(int)

        # Calculate the total dissimilarity to the assigned medoids.
        assigned_cost = distances[np.arange(N), z]

        # Record the assignment cost as a Python float.
        history.append(float(np.sum(assigned_cost)))

        # Preserve the previous medoids for the convergence check.
        old_m = m.copy()

        # -----------------------------------------------------
        # 2. Update each medoid independently
        # -----------------------------------------------------

        for k in range(len(m)):

            # Find all observations assigned to cluster k.
            members = np.flatnonzero(z == k)

            # Empty clusters retain their existing medoid.
            if len(members) == 0:
                continue

            # Extract pairwise dissimilarities within this cluster.
            cluster_distances = D[np.ix_(members, members)]

            # Calculate the total dissimilarity from each member
            # to every other member in the same cluster.
            total_distances = np.sum(
                cluster_distances,
                axis=1
            )

            # Select the member with the smallest total dissimilarity.
            # Since members are sorted, argmin resolves ties by
            # selecting the lowest original observation index.
            best_member = int(
                members[np.argmin(total_distances)]
            )

            # Update the medoid for the current cluster.
            m[k] = best_member

        # -----------------------------------------------------
        # 3. Check for convergence
        # -----------------------------------------------------

        # If no medoid has changed, the clustering has converged.
        if np.array_equal(m, old_m):
            return m, z, history

    # ---------------------------------------------------------
    # 4. Final assignment when max_iter is reached
    # ---------------------------------------------------------

    # Assign observations once more using the latest medoids.
    distances = D[:, m]

    z = np.argmin(distances, axis=1).astype(int)

    # Calculate the final assignment cost.
    final_cost = float(
        np.sum(distances[np.arange(N), z])
    )

    # Append the final cost to the history.
    history.append(final_cost)

    return m, z, history

# Step 17 - pam
def pam(D, m0, max_iter=100):
    """
    Perform Partitioning Around Medoids (PAM) using a swap search.

    Parameters
    ----------
    D : array-like of shape (N, N)
        Precomputed pairwise dissimilarity matrix.

    m0 : array-like of shape (K,)
        Initial medoid indices.

    max_iter : int, default=100
        Maximum number of complete medoid-swap passes.

    Returns
    -------
    m : numpy.ndarray of shape (K,)
        Final medoid indices, in cluster order.

    z : numpy.ndarray of shape (N,)
        Cluster assignment of every observation.

    history : list of float
        Initial cost followed by the cost after each completed pass.
    """

    # Convert the dissimilarity matrix to a floating-point array.
    D = np.asarray(D, dtype=float)

    # Validate the dissimilarity matrix.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square two-dimensional matrix.")

    N = D.shape[0]

    if N == 0:
        raise ValueError("D must contain at least one observation.")

    # Validate and independently copy the initial medoids.
    m = np.asarray(m0)

    if m.ndim != 1 or len(m) == 0:
        raise ValueError("m0 must be a non-empty one-dimensional array.")

    if not np.issubdtype(m.dtype, np.integer):
        raise ValueError("Medoid indices must be integers.")

    m = m.astype(int, copy=True)

    K = len(m)

    # Ensure that every medoid index is valid.
    if np.any(m < 0) or np.any(m >= N):
        raise ValueError("Medoid indices must be between 0 and N-1.")

    # Initial medoids must represent distinct observations.
    if len(np.unique(m)) != K:
        raise ValueError("Initial medoids must be distinct.")

    # Validate the maximum number of passes.
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 0
    ):
        raise ValueError("max_iter must be a non-negative integer.")

    # ---------------------------------------------------------
    # 1. Calculate the initial clustering cost
    # ---------------------------------------------------------

    def calculate_cost(medoids):
        """
        Calculate the total dissimilarity to the nearest medoid.
        """

        # Obtain distances from every observation to every medoid.
        distances = D[:, medoids]

        # Each observation is assigned to its nearest medoid.
        nearest_distances = np.min(distances, axis=1)

        # Return the total clustering cost as a Python float.
        return float(np.sum(nearest_distances))

    # Start the history with the cost of the initial medoids.
    current_cost = calculate_cost(m)

    history = [current_cost]

    # ---------------------------------------------------------
    # 2. Perform PAM medoid-swap passes
    # ---------------------------------------------------------

    for _ in range(max_iter):

        # Track whether at least one swap is accepted in this pass.
        swapped = False

        # Visit each medoid position in cluster order.
        for k in range(K):

            # Try replacing the current medoid with every observation.
            for o in range(N):

                # A medoid cannot be replaced by an observation
                # that is already serving as a medoid.
                if o in m:
                    continue

                # Create a candidate medoid configuration.
                candidate = m.copy()

                # Replace medoid k with observation o.
                candidate[k] = o

                # Calculate the cost of the candidate configuration.
                candidate_cost = calculate_cost(candidate)

                # Accept the swap only when the cost decreases
                # by more than the specified tolerance.
                if candidate_cost < current_cost - 1e-12:

                    # Immediately retain the improved medoid.
                    m[k] = o

                    # Update the current cost.
                    current_cost = candidate_cost

                    # Mark that an improvement occurred.
                    swapped = True

        # Record the cost after completing the entire pass.
        history.append(float(current_cost))

        # Stop when an entire pass produces no accepted swaps.
        if not swapped:
            break

    # ---------------------------------------------------------
    # 3. Calculate final cluster assignments
    # ---------------------------------------------------------

    # Calculate distances from every observation to the final medoids.
    final_distances = D[:, m]

    # Assign observations to their nearest medoids.
    # Ties are resolved in favour of the lowest cluster index.
    z = np.argmin(final_distances, axis=1).astype(int)

    return m, z, history

