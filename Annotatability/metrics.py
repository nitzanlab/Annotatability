import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import normalize
from sklearn.metrics import  pairwise_distances
import torch
import scipy.sparse as sp
from numba import jit


def rank_genes_conf(adata, power=1, power_low=1):
    """
    Rank genes based on confidence scores and specified parameters.

    Parameters
    ----------
    adata : AnnData
        Anndata object containing the data and annotations.
    power : float, optional
        Power for calculating confidence score for high confidence, by default 1.
    power_low : float, optional
        Power for calculating confidence score for low confidence, by default 1.

    Returns
    -------
    adata : AnnData
        Anndata object with additional gene-wise confidence scores.
    """
    scaled_adata = adata.copy()

    # Normalize gene expression data
    A = scaled_adata.X
    mean = np.mean(A, axis=0)
    A = A - mean
    A = normalize(A, axis=0)

    # Calculate confidence vectors
    conf_vector = np.array(scaled_adata.obs['conf'])
    low_conf_vector = 1 - conf_vector
    conf_vector = conf_vector / np.sum(conf_vector)
    low_conf_vector = low_conf_vector / np.sum(low_conf_vector)

    # Create an indicator for ambiguous confidence scores
    ambiguous_indicator = create_ambiguous_indicator(conf_vector)

    # Calculate confidence scores for high, low, and mid confidence
    adata.var['conf_score_high'] = A.T @ np.power(conf_vector, power)
    adata.var['conf_score_low'] = A.T @ np.power(low_conf_vector, power_low)
    adata.var['conf_score_mid'] = A.T @ ambiguous_indicator

    return adata


def rank_genes_conf_min_counts(adata, power=1, power_low=1, min_counts=100):
    """
    Rank genes based on confidence scores and specified parameters,
    with a minimum count threshold for filtering genes.

    Parameters
    ----------
    adata : AnnData
        Anndata object containing the data and annotations.
    power : float, optional
        Power for calculating confidence score for high confidence, by default 1.
    power_low : float, optional
        Power for calculating confidence score for low confidence, by default 1.
    min_counts : int, optional
        Minimum counts for filtering genes, by default 100.

    Returns
    -------
    scaled_adata : AnnData
        Anndata object with additional gene-wise confidence scores.
    """
    scaled_adata = adata.copy()

    # Filter genes based on minimum counts
    sc.pp.filter_genes(scaled_adata, min_counts=min_counts)

    # Normalize gene expression data
    A = scaled_adata.X
    mean = np.mean(A, axis=0)
    A = A - mean
    A = normalize(A, axis=0)

    # Calculate confidence vectors
    conf_vector = np.array(scaled_adata.obs['conf'])
    low_conf_vector = 1 - conf_vector
    conf_vector = conf_vector / np.sum(conf_vector)
    low_conf_vector = low_conf_vector / np.sum(low_conf_vector)

    # Create an indicator for ambiguous confidence scores
    ambiguous_indicator = create_ambiguous_indicator(conf_vector)

    # Calculate confidence scores for high, low, and mid confidence
    scaled_adata.var['conf_score_high'] = A.T @ np.power(conf_vector, power)
    scaled_adata.var['conf_score_low'] = A.T @ np.power(low_conf_vector, power_low)
    scaled_adata.var['conf_score_mid'] = A.T @ ambiguous_indicator

    return scaled_adata


def create_ambiguous_indicator(conf):
    """
    Create an indicator for ambiguous confidence scores based on mean and variance.

    Parameters
    ----------
    conf : np.ndarray
        Confidence scores.

    Returns
    -------
    ambiguous_indicator : np.ndarray
        An indicator array where 1 indicates ambiguous confidence scores,
        and 0 indicates non-ambiguous confidence scores.
    """
    mean = np.mean(conf)
    var = np.var(conf)

    # Initialize indicator array with ones
    ambiguous_indicator = np.ones(len(conf))

    # Set values to zero for ambiguous confidence scores
    ambiguous_indicator[(conf > mean + var) | (conf < mean - var)] = 0

    return ambiguous_indicator



def make_conf_graph(adata, alpha=0.9, k=15):
    """
    Create a confidence-based graph using K-nearest neighbors (KNN) approach.

    Parameters
    ----------
    adata : AnnData
        Anndata object containing the data and annotations.
    alpha : float, optional
        Alpha value for confidence-based distance adjustment, by default 0.9.
    k : int, optional
        Number of nearest neighbors to consider, by default 15.

    Returns
    -------
    adj_matrix : np.ndarray
        Adjacency matrix representing the KNN graph.
    distance_m : np.ndarray
        Distance matrix after confidence-based adjustment.
    """
    # Sort indices based on confidence
    sorted_indices = np.argsort(adata.obs['conf'])

    # Calculate positions relative to the sorted array
    positions = np.empty_like(sorted_indices)
    positions[sorted_indices] = np.arange(len(adata.obs['conf']))
    adata.obs['conf_position'] = positions

    # Calculate pairwise distances in PCA space
    distance_m = pairwise_distances(adata.obsm['X_pca'])

    # Adjust distances based on confidence
    distance_m = distance_matrix_with_conf(distance_m, positions, alpha)

    # Find K-nearest neighbors
    indices = np.argsort(distance_m, axis=-1)[:, 1:k + 1]

    # Create the adjacency matrix based on KNN
    adj_matrix = np.zeros(distance_m.shape)
    adj_matrix = knn_graph_from_distance(distance_m, adj_matrix, indices, distance_m.shape[0])

    return adj_matrix, distance_m



def distance_matrix_with_conf(distance_m, conf_array, alpha):
    """
    Adjust the distance matrix based on confidence scores.

    Parameters
    ----------
    distance_m : np.ndarray
        The original distance matrix.
    conf_array : np.ndarray
        Array of confidence scores.
    alpha : float
        Weighting parameter for adjusting distances based on confidence scores.

    Returns
    -------
    adjusted_distance_m : np.ndarray
        The adjusted distance matrix incorporating confidence scores.
    """
    # Precompute mean distances
    mean_distance_ge = np.mean(distance_m)
    mean_distance_conf = mean_absolute_distance(len(conf_array))

    # Normalize distance matrix
    normalized_distance_m = distance_m / mean_distance_ge

    # Compute confidence difference matrix and normalize
    conf_diff_matrix = np.abs(conf_array[:, np.newaxis] - conf_array)
    normalized_conf_diff_matrix = conf_diff_matrix / mean_distance_conf

    # Adjusted distance matrix with alpha blending
    adjusted_distance_m = alpha * normalized_distance_m + (1 - alpha) * normalized_conf_diff_matrix

    return adjusted_distance_m




def knn_graph_from_distance(distance_m, adj_matrix, indices, n):
    """
    Create a K-nearest neighbors (KNN) graph based on the distance matrix.

    Parameters
    ----------
    distance_m : np.ndarray
        Distance matrix.
    adj_matrix : np.ndarray
        Adjacency matrix to be updated based on KNN connections.
    indices : np.ndarray
        Indices of the K-nearest neighbors for each data point.
    n : int
        Number of data points.

    Returns
    -------
    adj_matrix : np.ndarray
        Updated adjacency matrix representing the KNN graph.
    """
    mean_distance = np.mean(distance_m)

    for i in range(n):
        # Update adjacency matrix based on KNN connections
        adj_matrix[i, indices[i]] = np.exp(-distance_m[i, indices[i]] / mean_distance)
        adj_matrix[indices[i], i] = np.exp(-distance_m[i, indices[i]] / mean_distance)

    return adj_matrix



def mean_absolute_distance(n):
    """
    Calculate the mean absolute distance for a given number of elements.

    Parameters
    ----------
    n : int
        Number of elements for which to calculate the mean absolute distance.

    Returns
    -------
    float
        Mean absolute distance for the given number of elements.
    """
    # Total number of pairs
    num_pairs = n * (n - 1) / 2

    # Sum of absolute differences for elements from 0 to n-1
    total_absolute_distance = sum(i * (n - i) for i in range(n))

    return total_absolute_distance / num_pairs



