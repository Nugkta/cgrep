# -*- coding: utf-8 -*-
import torch
from torch import nn
#import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as nnf
import collections
from collections import namedtuple
from tqdm import tqdm

from sklearn.datasets import fetch_20newsgroups
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import umap

from sklearn.mixture import GaussianMixture


from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device: {device}')


class Initializer:
  def __init__(self, embeddings_vocab, n_topics, n_dims = 11):
    """
    Args: 
      embeddings_vocab: embedding of each word in the vocabulary 
      n_topics: number of topics to find
      n_dims: Number of dimensions to project data into with UMAP 
    """
    self.embeddings_vocab = embeddings_vocab
    self.n_topics = n_topics
    self.n_dims = n_dims 

  def reduce_dimensionality(self, umap_hyperparams = {'n_neighbors': 15, 'min_dist': 0.01}):
    """
    Take the tensor embeddings of the embeddings of the vocabulary and reduce their dimensionality. 
    The paramters n_neighbors and min_dist change the behavour of UMAP. 
    """
    
    umap1 = umap.UMAP(n_components=self.n_dims, metric = 'cosine', **umap_hyperparams)
    proj_embeddings = umap1.fit_transform(self.embeddings_vocab)
    self.proj_embeddings = proj_embeddings

    return proj_embeddings

  def fit_gmm(self, embeddings, random_state = 42):
    """
    Fit Gaussian Mixture model to the embeddings (with dimensionality reduction)
    and return the means and covariances of the Gaussians and the bic of this model
    """

    # Fixed logic: if embeddings is None, use self.proj_embeddings
    if embeddings is None:
      embeddings = self.proj_embeddings

    # Check for valid data
    if len(embeddings) < self.n_topics:
      raise ValueError(f"Not enough data points ({len(embeddings)}) for {self.n_topics} topics. "
                      f"Reduce n_topics or increase dataset size.")

    # fit gmm to embeddings with regularization
    # Use 'tied' or 'diag' covariance for stability with many components
    gmm1 = GaussianMixture(
        n_components=self.n_topics,
        covariance_type='full',
        random_state=random_state,
        max_iter=200,
        n_init=3,
        reg_covar=1e-6  # Add regularization to prevent singular matrices
    )

    try:
      gmm1.fit(embeddings)
    except Exception as e:
      print(f"Warning: GMM fitting failed with error: {e}")
      print("Falling back to diagonal covariance...")
      gmm1 = GaussianMixture(
          n_components=self.n_topics,
          covariance_type='diag',  # Simpler covariance structure
          random_state=random_state,
          max_iter=200,
          n_init=3,
          reg_covar=1e-6
      )
      gmm1.fit(embeddings)

    mus_init = torch.tensor(gmm1.means_, dtype=torch.float32)
    sigmas_init = torch.tensor(gmm1.covariances_, dtype=torch.float32)

    bic = gmm1.bic(embeddings)  # Changed from score to bic for consistency

    return mus_init, sigmas_init, bic

  def get_reparametrization_parameters(self, sigmas_init, eps=0.1):
    """
    Compute the parameters for the reparameterization of the sigmas, such that
    \sigma = L_lower_init @ L_lower_init.T + torch.exp(log_diag_init)
    where log_diag_init is a diagonal matrix
    """

    # Add small regularization to diagonal for numerical stability
    regularization = torch.eye(self.n_dims) * 1e-5
    sigmas_regularized = sigmas_init + regularization.unsqueeze(0)

    # Try Cholesky decomposition with error handling
    try:
      L_lower_init = torch.linalg.cholesky(sigmas_regularized)
    except RuntimeError as e:
      print(f"Warning: Cholesky decomposition failed: {e}")
      print("Using diagonal approximation instead...")
      # Fall back to diagonal matrices if Cholesky fails
      diag_vals = torch.diagonal(sigmas_init, dim1=-2, dim2=-1)
      diag_vals = torch.clamp(diag_vals, min=1e-6)  # Ensure positive
      L_lower_init = torch.zeros_like(sigmas_init)
      for i in range(self.n_topics):
        L_lower_init[i] = torch.diag(torch.sqrt(diag_vals[i]))

    # Check for NaN or Inf in L_lower_init
    if torch.isnan(L_lower_init).any() or torch.isinf(L_lower_init).any():
      print("Warning: NaN or Inf detected in L_lower_init, using small random values")
      L_lower_init = torch.randn(self.n_topics, self.n_dims, self.n_dims) * 0.01
      # Make it lower triangular
      L_lower_init = torch.tril(L_lower_init)

    # Initialize log_diag with proper value
    log_diag_init = torch.log(torch.ones(self.n_topics, self.n_dims) * eps)

    # Check for invalid values in log_diag_init
    if torch.isnan(log_diag_init).any() or torch.isinf(log_diag_init).any():
      print(f"Warning: Invalid log_diag_init with eps={eps}, using default")
      log_diag_init = torch.ones(self.n_topics, self.n_dims) * (-2.0)  # log(0.135) ≈ -2

    return L_lower_init, log_diag_init


  def reduce_dim_and_cluster(self, eps = 1e-4, umap_hyperparams = {'n_neighbors': 15, 'min_dist': 0.01}):
    """
    Reduce the dimensionality with UMAP of the embeddings and fit a GMM model, which yields the means and covariances (albeit reparameterized)
    of the GMM.
    Args:
      n_neigbors: Number of neighbors to consider in UMAP
      min_dist: Minimal distance of points in space with lower dimensionality for UMAP
      eps: Regularization parameter for the covariance matrices.

    Return:
      emb_dim_red: UMAP-reduced embeddings
      mus_init: means of topic-specific covariances
      L_lower_init: factor of covariance matrix
      log_diag_init: log of diagonal matrix to add to L_lower_init @ L_lower_init.T
      bic: Bayesian information criterion of GMM
    """

    print(f"Initializing with {self.n_topics} topics and {self.n_dims} dimensions")
    print(f"Input embeddings shape: {self.embeddings_vocab.shape}")

    # Check if we have enough data
    n_samples = self.embeddings_vocab.shape[0]
    if n_samples < self.n_topics * 2:
      print(f"WARNING: Only {n_samples} samples for {self.n_topics} topics!")
      print(f"Recommendation: Use at least {self.n_topics * 10} samples or reduce n_topics")

    emb_dim_red = self.reduce_dimensionality(umap_hyperparams = umap_hyperparams)
    print(f"UMAP reduction completed. Shape: {emb_dim_red.shape}")

    mus_init, sigmas_init, bic = self.fit_gmm(emb_dim_red)
    print(f"GMM fitting completed. BIC: {bic:.2f}")

    # Pass the eps parameter here
    L_lower_init, log_diag_init = self.get_reparametrization_parameters(sigmas_init, eps=eps)
    print(f"Reparameterization completed. L_lower shape: {L_lower_init.shape}, log_diag shape: {log_diag_init.shape}")

    # Final validation
    if torch.isnan(mus_init).any() or torch.isnan(L_lower_init).any() or torch.isnan(log_diag_init).any():
      print("ERROR: NaN values detected in initialization!")
      print(f"  mus_init NaN: {torch.isnan(mus_init).sum().item()}")
      print(f"  L_lower_init NaN: {torch.isnan(L_lower_init).sum().item()}")
      print(f"  log_diag_init NaN: {torch.isnan(log_diag_init).sum().item()}")
      raise ValueError("Initialization failed with NaN values. Try reducing n_topics or increasing dataset size.")

    return emb_dim_red, mus_init, L_lower_init, log_diag_init, bic

