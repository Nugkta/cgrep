import torch
from torch import nn
#import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random  # Add this import at the top of the file

from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as nnf
import collections
from collections import namedtuple
from tqdm import tqdm

from sklearn.datasets import fetch_20newsgroups
import pickle
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import time
import os


from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device: {device}')




class Linear_skip_block(nn.Module):
  """
  Block of linear layer + softplus + skip connection +  dropout  + batchnorm 
  """
  def __init__(self, n_input, dropout_rate):
    super(Linear_skip_block, self).__init__()

    self.fc = nn.Linear(n_input, n_input)
    self.act = torch.nn.LeakyReLU()

    self.bn = nn.BatchNorm1d(n_input, affine = True) 
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x0 = x
    x = self.fc(x)
    x = self.act(x)
    x = x0 + x
    x = self.drop(x)
    x = self.bn(x)

    return x


class Linear_block(nn.Module):
  """
  Block of linear layer dropout  + batchnorm 
  """
  def __init__(self, n_input, n_output, dropout_rate):
    super(Linear_block, self).__init__()

    self.fc = nn.Linear(n_input, n_output)
    self.act = torch.nn.LeakyReLU()
    self.bn = nn.BatchNorm1d(n_output, affine = True) 
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.fc(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.bn(x)

    return x


class Encoder_NVLDA(nn.Module):
  """
  Encoder for TLDA, takes tokenized bow representation of a batch of documents and returns the mean and log-variance of the corresponding distributions over theta 
  """
  def __init__(self, config):
    super(Encoder_NVLDA, self).__init__()

    self.config = config

    self.linear1 = Linear_block(config.num_input, config.n_hidden_block, config.drop_rate_en)    # initial linear layer 
    self.hidden_layers = torch.nn.Sequential(*[Linear_skip_block(config.n_hidden_block, config.drop_rate_en) for _ in range(config.n_skip_layers)])  #hidden skip-layers
    self.mean_fc = nn.Linear(config.n_hidden_block, config.n_topics)              #linear layer to get mean of topic-distribution per document
    self.logvar_fc = nn.Linear(config.n_hidden_block, config.n_topics)            #linear layer to get log of diagonal of covariance of topic-distribution per document
    self.act = nnf.softplus                                                       #softplus activation function

  def forward(self, x):
    x = self.linear1(x)
    x = self.hidden_layers(x)
  
    posterior_mean = self.mean_fc(x)                                            #calculate posterior mean from output of second linear layer
    posterior_logvar = self.logvar_fc(x)                                        # calculate posterior logvar from output of seconf linear layer

    return posterior_mean, posterior_logvar

class Encoder_SentenceTransformer(nn.Module):
  """
  Encoder for TLDA, takes tokenized bow representation of a batch of documents and returns the mean and log-variance of the corresponding distributions over theta 
  """
  def __init__(self, config, sentence_transformer, extract_from_sentence_transformer):
    """
    Use a sentence transformer in the encoder, as in Bianchi et al. 2021.
    Args: 
        config: config file
        sentence_transformer: sentence transformer to use in the encoder
        extract_from_sentence_transformer: function that maps the output of the sentence transformer into something useable 
        
    """
    super(Encoder_NVLDA, self).__init__()

    self.config = config
    self.sentence_transformer = sentence_transformer
    
    self.linear_layer = nn.Linear(config.sentence_transformer.hidden_dim, config.n_hidden_block) #linear layer from sentence transformer to mean and logvar layers    
    self.mean_fc = nn.Linear(config.n_hidden_block, config.n_topics)              #linear layer to get mean of topic-distribution per document
    self.logvar_fc = nn.Linear(config.n_hidden_block, config.n_topics)            #linear layer to get log of diagonal of covariance of topic-distribution per document
    
    self.act = nnf.softplus 


  def forward(self, x):
    """
    Takes batches of sentences as input and outputs the variational posterior mean and variational posterior logvariance to sample from. 
    """
    x = self.sentence_transformer(x)
    x = self.linear_layer(x)
    x = self.act(x)
  
    posterior_mean = self.mean_fc(x)                                            #calculate posterior mean from output of second linear layer
    posterior_logvar = self.logvar_fc(x)                                        # calculate posterior logvar from output of seconf linear layer

    return posterior_mean, posterior_logvar
    
    
class Encoder_SentenceTransformer_precomputed(nn.Module):
  """
  Encoder for TLDA, takes tokenized bow representation of a batch of documents and returns the mean and log-variance of the corresponding distributions over theta 
  """
  def __init__(self, config):
    """
    Use the embedding obtained by a sentence transformer
    Args: 
        config: config file
        sentence_transformer: sentence transformer to use in the encoder
        extract_from_sentence_transformer: function that maps the output of the sentence transformer into something useable 
        
    """
    super(Encoder_SentenceTransformer_precomputed, self).__init__()

    self.config = config
  
    self.linear_layer = nn.Linear(config.sentence_transformer_hidden_dim, config.n_hidden_block) #linear layer from sentence transformer to mean and logvar layers    
    self.mean_fc = nn.Linear(config.n_hidden_block, config.n_topics)              #linear layer to get mean of topic-distribution per document
    self.logvar_fc = nn.Linear(config.n_hidden_block, config.n_topics)            #linear layer to get log of diagonal of covariance of topic-distribution per document
    
    self.act = nnf.softplus 


  def forward(self, x):
    """
    Takes batches of embeddings of sentences as input and outputs the variational posterior mean and variational posterior logvariance to sample from. 
    """
    x = self.linear_layer(x)
    x = self.act(x)
  
    posterior_mean = self.mean_fc(x)                                            #calculate posterior mean from output of second linear layer
    posterior_logvar = self.logvar_fc(x)                                        # calculate posterior logvar from output of seconf linear layer

    return posterior_mean, posterior_logvar
    
    


def calc_beta(mus, L_lower, log_diag, embeddings, config, eps=1e-6):
  """
  take parameters of topic-specific normal distributions of shape (n_topics, embedding_dim), i.e. mus and L_lower
  and return probability of each word embedding among the embeddings.
  L_lower is a (n_embedding_dim, n_embedding_dim) matrix, but only the part below the diagonal is used

  Return log-probabilities of each embedding under each normal distribution
  """

  # Get device from mus
  device = mus.device

  # Add a small epsilon to ensure positive diagonal values
  diag = torch.exp(log_diag.clamp(min=-10, max=10)) + eps  # Clamp to prevent overflow/underflow

  # Create log_probs on the same device as mus
  log_probs = torch.zeros(config.n_topics, config.vocab_size, device=device, dtype=mus.dtype)

  for i, (mu, lower, D) in enumerate(zip(mus, L_lower, diag)):
    # Check for NaN or Inf in parameters
    if torch.isnan(mu).any() or torch.isinf(mu).any():
      print(f"Warning: NaN or Inf detected in mu for topic {i}, using uniform distribution")
      log_probs[i] = torch.zeros(config.vocab_size, device=device, dtype=mus.dtype)
      continue

    if torch.isnan(D).any() or torch.isinf(D).any() or (D <= 0).any():
      print(f"Warning: Invalid diagonal values for topic {i}, using uniform distribution")
      log_probs[i] = torch.zeros(config.vocab_size, device=device, dtype=mus.dtype)
      continue

    try:
      dist = LowRankMultivariateNormal(mu, cov_factor=lower, cov_diag=D)
      log_probs[i] = dist.log_prob(embeddings)
    except Exception as e:
      # Fall back to a simpler diagonal normal distribution if there's an issue
      try:
        std = torch.sqrt(D).clamp(min=eps)  # Ensure positive std
        dist = Independent(Normal(mu, std), 1)
        log_probs[i] = dist.log_prob(embeddings)
      except Exception as e2:
        print(f"Warning: Both distributions failed for topic {i}, using uniform: {e2}")
        log_probs[i] = torch.zeros(config.vocab_size, device=device, dtype=mus.dtype)

  return log_probs


class Decoder_TNTM(nn.Module):
  """
    embeddings: The embeddings of every word in the corpus
    mus_init: What to initialize the means with
    L_lower_init: What to initialize the L matrix for the variance with
    log_diag_init: What to initialize the log of the diagonal component of the variance with. The covariance is reparametrized as sigma = L_lower_init.T @ L_lower_init + exp(log_diag_init)
    config: config dict
    use_hybrid: Whether to use hybrid beta (Gaussian + ProdLDA)
    """
  def __init__(self, embeddings, mus_init, L_lower_init, log_diag_init, config, use_hybrid=False, beta_prodlda_init=None):
    """
    embeddings: The precomputed embeddings of every word in the corpus
    mus_init: What to initialize the means with
    L_lower_init: What to initialize the L matrix for covariance
    log_diag_init: What to initialize the log diagonal with
    use_hybrid: Whether to use hybrid approach (Gaussian + ProdLDA beta)
    beta_prodlda_init: Initial values for ProdLDA beta matrix (n_topics x vocab_size)
    """
    super(Decoder_TNTM, self).__init__()

    self.config = config
    self.embeddings = embeddings
    self.use_hybrid = use_hybrid

    # Gaussian-based parameters (embedding space)
    self.mus = nn.Parameter(mus_init)   #create topic means as learnable paramter
    self.L_lower = nn.Parameter(L_lower_init)   # factor of covariance per topic
    self.log_diag = nn.Parameter(log_diag_init)  # summand for diagonal of covariance

    # ProdLDA-style parameters (if hybrid mode is enabled)
    if self.use_hybrid:
      # Get device from mus_init to ensure consistency
      device = mus_init.device

      if beta_prodlda_init is None:
        # Initialize with uniform distribution + small noise on the correct device
        beta_prodlda_init = torch.ones(config.n_topics, config.vocab_size, device=device) / config.vocab_size
        beta_prodlda_init += torch.randn(config.n_topics, config.vocab_size, device=device) * 0.01

      # Ensure beta_prodlda_init is positive
      beta_prodlda_init = torch.clamp(beta_prodlda_init, min=1e-10)

      # Store log-space for numerical stability, will softmax in forward
      self.log_beta_prodlda = nn.Parameter(torch.log(beta_prodlda_init))

      # Lambda parameter to weight the two betas (will be passed through sigmoid)
      self.lambda_logit = nn.Parameter(torch.tensor(0.0, device=device))  # sigmoid(0) = 0.5

      print(f"Hybrid mode enabled: beta_prodlda shape={self.log_beta_prodlda.shape}, device={self.log_beta_prodlda.device}")

  def forward(self, theta_hat):
    """
    ProdLDA-style forward pass: reconstruction = softmax(θ̂ @ β)
    where θ̂ is unconstrained (no softmax applied to theta_hat)

    Args:
        theta_hat: [batch, n_topics] - unconstrained document-topic representation

    Returns:
        log_recon: [batch, vocab_size] - log probabilities of reconstructed word distribution
    """
    # Get device from theta_hat to ensure consistency
    device = theta_hat.device

    # Calculate Gaussian-based log_beta (embedding space)
    log_beta_gaussian = calc_beta(self.mus, self.L_lower, self.log_diag, self.embeddings, self.config)

    # Check for NaN in log_beta_gaussian
    if torch.isnan(log_beta_gaussian).any() or torch.isinf(log_beta_gaussian).any():
      print("WARNING: NaN/Inf in log_beta_gaussian, using uniform distribution")
      log_beta_gaussian = torch.zeros(self.config.n_topics, self.config.vocab_size, device=device)

    # Convert to probabilities: softmax over vocabulary for each topic
    beta_gaussian = torch.nn.functional.softmax(log_beta_gaussian, dim=-1)  # [n_topics, vocab_size]

    # Check for NaN after softmax
    if torch.isnan(beta_gaussian).any():
      print("WARNING: NaN in beta_gaussian after softmax, using uniform distribution")
      beta_gaussian = torch.ones(self.config.n_topics, self.config.vocab_size, device=device) / self.config.vocab_size

    if self.use_hybrid:
      # Calculate ProdLDA-style beta (raw word probabilities)
      # Apply softmax to ensure proper probability distribution over vocabulary
      beta_prodlda = torch.nn.functional.softmax(self.log_beta_prodlda, dim=-1)  # [n_topics, vocab_size]

      # Check for NaN
      if torch.isnan(beta_prodlda).any():
        print("WARNING: NaN in beta_prodlda, using uniform distribution")
        beta_prodlda = torch.ones(self.config.n_topics, self.config.vocab_size, device=device) / self.config.vocab_size

      # Get lambda weight (constrained to [0, 1])
      lambda_weight = torch.sigmoid(self.lambda_logit)

      # Check for NaN in lambda
      if torch.isnan(lambda_weight):
        print("WARNING: NaN in lambda_weight, using 0.5")
        lambda_weight = torch.tensor(0.5, device=device)

      # Combine the two betas in probability space
      # β_hybrid = λ * β_gaussian + (1-λ) * β_prodlda
      beta = lambda_weight * beta_gaussian + (1 - lambda_weight) * beta_prodlda  # [n_topics, vocab_size]
    else:
      # Use only Gaussian-based beta
      beta = beta_gaussian

    # Check beta validity
    if torch.isnan(beta).any() or torch.isinf(beta).any():
      print("WARNING: NaN/Inf in final beta, using uniform distribution")
      beta = torch.ones(self.config.n_topics, self.config.vocab_size, device=device) / self.config.vocab_size

    # ProdLDA scheme: multiply first, then softmax
    # logits = θ̂ @ β
    logits = torch.matmul(theta_hat, beta)  # [batch, n_topics] @ [n_topics, vocab_size] = [batch, vocab_size]

    # Check logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
      print("WARNING: NaN/Inf in logits after matmul")
      print(f"  theta_hat: min={theta_hat.min():.4f}, max={theta_hat.max():.4f}, nan={torch.isnan(theta_hat).any()}")
      print(f"  beta: min={beta.min():.4f}, max={beta.max():.4f}, nan={torch.isnan(beta).any()}")

    # Apply log-softmax at the end
    log_recon = torch.nn.functional.log_softmax(logits, dim=-1)  # [batch, vocab_size]

    return log_recon


class TNTM_bow(nn.Module):
  """
  Combine encoder and decoder to one model
  """

  def __init__(self, config, embeddings, mus_init, lower_init, log_diag_init, prior_mean, prior_variance,
               use_hybrid=False, beta_prodlda_init=None):
    super(TNTM_bow, self).__init__()

    self.config = config
    self.use_hybrid = use_hybrid

    self.encoder = Encoder_NVLDA(config)  # use same encoder as for NVLDA
    self.decoder = Decoder_TNTM(embeddings, mus_init, lower_init, log_diag_init, config,
                                use_hybrid=use_hybrid, beta_prodlda_init=beta_prodlda_init)  # Use decoder with optional hybrid mode

    self.prior_mean = prior_mean
    self.prior_variance = prior_variance


  def forward(self, x):
    # Encoder: maps BoW to parameters of logistic-normal distribution over topics
    posterior_mean, posterior_logvar = self.encoder(x)
    posterior_std = torch.exp(0.5*posterior_logvar)

    # Reparameterization trick: sample theta_hat from N(posterior_mean, posterior_std)
    eps = torch.randn_like(posterior_std)  # Sample from standard normal
    theta_hat = posterior_mean + eps*posterior_std  # Unconstrained topic representation

    # Decoder: ProdLDA-style reconstruction = softmax(theta_hat @ beta)
    # Note: theta_hat is NOT passed through softmax before multiplying with beta
    log_recon = self.decoder(theta_hat)

    return log_recon, posterior_mean, posterior_logvar 



  def infer_topic_of_doc(self, docs, word2idx, top_k=3):
      """
      Infers the top-K topics of document(s) using a trained Variational Autoencoder (VAE) topic model.

      Parameters
      ----------
      docs : list of list of str
          A list of documents, where each document is represented as a list of words.
      word2idx : dict
          A dictionary mapping words (str) to their corresponding indices (int) in the vocabulary.
      top_k : int, optional
          The number of most probable topics to select for each document (default is 3).

      Returns
      -------
      list of np.ndarray or np.ndarray
          - If multiple documents are provided, returns a list where each element is a NumPy array 
            containing the indices of the top-K most probable topics for that document.
          - If only a single document is provided, returns a single NumPy array of top-K topic indices.
      """

      # Step 1: Build the Bag-of-Words (BoW) representation
      num_docs = len(docs)
      vocab_size = len(word2idx)
      bow_tensor = torch.zeros(num_docs, vocab_size, dtype=torch.float32)

      # Convert each document into indices and fill the BoW tensor.
      for i, doc in enumerate(docs):
          for word in doc:
              if word in word2idx:
                  bow_tensor[i, word2idx[word]] += 1

      # Normalize the BoW tensor (ensuring numerical stability)
      bow_tensor /= (bow_tensor.sum(dim=1, keepdim=True) + 1e-8)  # Avoid division by zero

      # Step 2: Move the BoW tensor to the same device as the encoder.
      device = next(self.encoder.parameters()).device
      bow_tensor = bow_tensor.to(device)

      # Step 3: Encode the document(s) into the topic space using the VAE encoder.
      posterior_mean, posterior_logvar = self.encoder(bow_tensor)

      # Step 4: Apply the Reparameterization Trick.
      posterior_var = torch.exp(posterior_logvar)  # Convert log variance to variance
      epsilon = torch.randn_like(posterior_mean)   # Sample from standard normal
      theta_tilde = posterior_mean + torch.sqrt(posterior_var) * epsilon

      # Step 5: For inference, use the mean of the posterior as topic distribution
      # Note: In ProdLDA, theta is not constrained to sum to 1 during training,
      # but for inference we can use softmax to get interpretable topic proportions
      theta = torch.softmax(theta_tilde, dim=-1)  # Convert to probability distribution for interpretability
      theta = theta.cpu().detach().numpy()  # Convert to NumPy array

      # Step 6: Select the top-K most probable topics for each document.
      if theta.ndim == 1:
          # Single document case
          top_topics = np.argsort(-theta)[:top_k]  # Sort indices in descending order and take top-K
      else:
          # Multiple documents case
          top_topics = [np.argsort(-doc_theta)[:top_k] for doc_theta in theta]

          # Ensure single document case returns a single NumPy array
          if num_docs == 1:
              top_topics = top_topics[0]

      return top_topics







class TNTM_sentence_transformer_precomputed(nn.Module):
  """
  Combine encoder and decoder to one model
  """

  def __init__(self, config, embeddings, mus_init, lower_init, log_diag_init, prior_mean, prior_variance,
               use_hybrid=False, beta_prodlda_init=None):
    super(TNTM_sentence_transformer_precomputed, self).__init__()

    self.config = config
    self.use_hybrid = use_hybrid

    self.encoder = Encoder_SentenceTransformer_precomputed(config)  # use same encoder as for NVLDA
    self.decoder = Decoder_TNTM(embeddings, mus_init, lower_init, log_diag_init, config,
                                use_hybrid=use_hybrid, beta_prodlda_init=beta_prodlda_init)  # Use decoder with optional hybrid mode

    self.prior_mean = prior_mean
    self.prior_variance = prior_variance


  def forward(self, x):
    # Encoder: maps sentence embeddings to parameters of logistic-normal distribution over topics
    posterior_mean, posterior_logvar = self.encoder(x)
    posterior_std = torch.exp(0.5*posterior_logvar)

    # Reparameterization trick: sample theta_hat from N(posterior_mean, posterior_std)
    eps = torch.randn_like(posterior_std)  # Sample from standard normal
    theta_hat = posterior_mean + eps*posterior_std  # Unconstrained topic representation

    # Decoder: ProdLDA-style reconstruction = softmax(theta_hat @ beta)
    # Note: theta_hat is NOT passed through softmax before multiplying with beta
    log_recon = self.decoder(theta_hat)

    return log_recon, posterior_mean, posterior_logvar 


def loss_elbo(input, log_recon, posterior_mean, posterior_logvar, 
              prior_mean, prior_var, n_topics, model, 
              reg_lambda=0.1, trace_min=5.0, log_probability=0.1):
    """
    Compute the Evidence Lower Bound (ELBO) loss for the Variational Autoencoder (VAE) topic model.

    The ELBO consists of two main components:
    1. Negative Log-Likelihood (NLL): Measures the reconstruction loss between the input and the model's output.
    2. Kullback-Leibler Divergence (KLD): Measures the divergence between the variational posterior and the prior.

    Additionally, a soft regularization penalty is applied to ensure the trace of the decoder's covariance matrix 
    does not fall below a specified minimum value (`trace_min`).

    Args:
        input (torch.Tensor): The input data (e.g., bag-of-words representation of documents).
        log_recon (torch.Tensor): The log of the reconstructed output from the decoder.
        posterior_mean (torch.Tensor): The mean of the variational posterior distribution.
        posterior_logvar (torch.Tensor): The log-variance of the variational posterior distribution.
        prior_mean (torch.Tensor): The mean of the prior distribution.
        prior_var (torch.Tensor): The variance of the prior distribution.
        n_topics (int): The number of topics in the model.
        model (nn.Module): The VAE model containing the encoder and decoder.
        reg_lambda (float, optional): The regularization coefficient for the trace penalty. Default is 0.1.
        trace_min (float, optional): The minimum allowed value for the trace of the covariance matrix. Default is 5.0.
        log_probability (float, optional): Probability of logging the metrics. Default is 0.1.

    Returns:
        tuple: A tuple containing:
            - loss (torch.Tensor): The total loss (ELBO + regularization penalty).
            - NL_avg (torch.Tensor): The average Negative Log-Likelihood (NLL).
            - KLD_avg (torch.Tensor): The average Kullback-Leibler Divergence (KLD).
    """
    # Negative log-likelihood (reconstruction loss)
    NL = -(input * log_recon).sum(1)
    
    # KLD between the variational posterior and the prior
    posterior_var = posterior_logvar.exp()
    prior_mean = prior_mean.expand_as(posterior_mean)
    prior_var = prior_var.expand_as(posterior_mean)
    prior_logvar = torch.log(prior_var)
    
    var_division = posterior_var / prior_var
    diff = posterior_mean - prior_mean
    diff_term = diff * diff / prior_var
    logvar_division = prior_logvar - posterior_logvar
    KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - n_topics)
    
    NL_avg = torch.mean(NL)
    KLD_avg = torch.mean(KLD)
    base_loss = (NL + KLD).mean()
    
    # Trace of covariance = sum of squares in L_lower + sum of exp(log_diag)
    cov_trace = (model.decoder.L_lower**2).sum() + torch.exp(model.decoder.log_diag).sum()
    
    # Soft penalty if trace < trace_min
    below_min = torch.clamp(trace_min - cov_trace, min=0)
    trace_penalty = below_min**2
    
    # Total loss
    loss = base_loss + reg_lambda * trace_penalty

    # Logging the loss and regularization term with a random probability
    if random.random() < log_probability:
        reg_penalty = reg_lambda * trace_penalty
        print(f"Total Loss = {loss.item():.4f}, Base Loss = {base_loss.item():.4f}, "
              f"Trace Penalty = {trace_penalty.item():.4f}, Reg Penalty = {reg_penalty.item():.4f}, "
              f"Covariance Trace = {cov_trace.item():.4f}")

    return loss, NL_avg, KLD_avg
  
  
  
def train_test_split(dataset, train_frac, val_frac, batch_size):
    """
    Split dataset into train and validation sets.
    Note: test set is not created as it's not used in training.

    Args:
        dataset: The full dataset
        train_frac: Fraction for training (e.g., 0.8)
        val_frac: Fraction for validation (e.g., 0.2)
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader, None (test_loader placeholder)
    """
    tot_len = len(dataset)

    # Calculate sizes ensuring they sum to total length
    train_size = int(tot_len * train_frac)
    val_size = tot_len - train_size  # Use remainder for validation to avoid rounding issues

    # Split into train and validation only (test set not needed)
    train, val = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

    # Return None for test_loader since it's not used
    return train_loader, val_loader, None
    
def validate(model, dataloader, prior_mean, prior_var, n_topics, sparse_ten = False, reg_lambda=0.1, trace_min=5.0):
    val_loss_lis = []
    val_nl_lis = []
    val_kld_lis = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            
            try: 
              sample_encode, sample_decode = batch 
            except:
              sample_encode = sample_decode = batch 
              
            sample_encode = sample_encode.to_dense().float().to(device)
            sample_decode = sample_decode.to_dense().float().to(device)
               
                
            log_recon, posterior_mean, posterior_logvar = model(sample_encode)

            loss , NL, KLD = loss_elbo(input = sample_decode, log_recon = log_recon, posterior_mean = posterior_mean, posterior_logvar = posterior_logvar,
                            prior_mean = prior_mean, prior_var = prior_var, n_topics = n_topics, model=model, reg_lambda=reg_lambda, trace_min=trace_min)

            val_loss_lis.append(loss.cpu().detach())
            val_nl_lis.append(NL.cpu().detach())
            val_kld_lis.append(KLD.cpu().detach())


    return np.mean(np.array(val_loss_lis)), np.mean(np.array(val_nl_lis)), np.mean(np.array(val_kld_lis))
    
def validate_median(model, dataloader, prior_mean, prior_var, n_topics, sparse_ten = False, reg_lambda=0.1, trace_min=5.0):
    val_loss_lis = []
    val_nl_lis = []
    val_kld_lis = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            try: 
              sample_encode, sample_decode = batch 
            except:
              sample_encode = sample_decode = batch
            sample_encode = sample_encode.to_dense().float().to(device)
            sample_decode = sample_decode.to_dense().float().to(device)
               
                
            log_recon, posterior_mean, posterior_logvar = model(sample_encode)

            loss, NL, KLD = loss_elbo(
                input=sample_decode, 
                log_recon=log_recon, 
                posterior_mean=posterior_mean, 
                posterior_logvar=posterior_logvar,
                prior_mean=prior_mean, 
                prior_var=prior_var, 
                n_topics=n_topics, 
                model=model, 
                reg_lambda=reg_lambda, 
                trace_min=trace_min
            )

            val_loss_lis.append(loss.cpu().detach())
            val_nl_lis.append(NL.cpu().detach())
            val_kld_lis.append(KLD.cpu().detach())


    return np.median(np.array(val_loss_lis)), np.median(np.array(val_nl_lis)), np.median(np.array(val_kld_lis))





def train_loop(model, optimizer1, optimizer2, trainset, valset, print_mod, device, n_epochs,
               save_path, config, tensorboard_log_dir, clip_value=1e5, topic_num=None,
               save_checkpoints=False, save_interval=1, reg_lambda=0.1, trace_min=5.0, optimizer3=None):
    """
    Train the model.

    Args:
        model: The TLDA model to train.
        optimizer1: The optimizer for the encoder.
        optimizer2: The optimizer for the topic-specific normal distributions (Gaussian parameters).
        trainset: The training dataset.
        valset: The validation dataset.
        print_mod: Frequency (in epochs) at which to print and log metrics.
        device: Device to use ("cpu" or "cuda").
        n_epochs: Total number of epochs to train.
        save_path: Directory where to save model checkpoints.
        config: Configuration object (must include n_topics, early_stopping, n_epochs_early_stopping).
        tensorboard_log_dir: Directory to save TensorBoard logs.
        clip_value: Value above which the gradient norm will be clipped.
        topic_num: Optional topic number for naming the checkpoint file.
        save_checkpoints: Whether to save checkpoints during training (default: False).
        save_interval: How often to save checkpoints (in epochs) if save_checkpoints is True (default: 1).
        reg_lambda (float): Regularization coefficient for variance penalty.
        trace_min (float): Minimum variance threshold for regularization.
        optimizer3: Optional optimizer for ProdLDA parameters (for hybrid mode).

    Returns:
        A dictionary with training and validation metrics.
    """
    # Set up TensorBoard logging.
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    
    # Early stopping settings.
    early_stopping = config.early_stopping
    n_early_stopping = config.n_epochs_early_stopping if early_stopping else None
    past_val_losses = []

    # Dictionaries to store epoch-level metrics.
    train_metrics = {"loss": [], "nll": [], "kld": []}
    val_metrics = {"loss": [], "nll": [], "kld": []}

    model.train()

    # Print total trainable parameters once at the start.
    total_params = count_parameters(model)
    print(f"Trainable model parameters: {total_params}")

    for epoch in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):
        epoch_start_time = time.time()

        # Lists for accumulating per-batch metrics during the epoch.
        batch_losses = []
        batch_nlls = []
        batch_klds = []
        batch_grad_norms = []

        for batch in trainset:
            # Unpack the batch. If batch has a single tensor, use it for both encoding and decoding.
            try:
                sample_encode, sample_decode = batch
            except Exception:
                sample_encode = sample_decode = batch

            # Convert sparse tensors to dense and move to the target device.
            sample_encode = sample_encode.to_dense().float().to(device)
            sample_decode = sample_decode.to_dense().float().to(device)

            # Zero the gradients.
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            if optimizer3 is not None:
                optimizer3.zero_grad()

            # Forward pass.
            log_recon, posterior_mean, posterior_logvar = model(sample_encode)

            # Compute loss and its components.
            loss, nll, kld = loss_elbo(
                input=sample_decode,
                log_recon=log_recon,
                posterior_mean=posterior_mean,
                posterior_logvar=posterior_logvar,
                prior_mean=model.prior_mean,
                prior_var=model.prior_variance,
                n_topics=config.n_topics,
                model=model,
                reg_lambda=reg_lambda,
                trace_min=trace_min
            )

            # Backward pass.
            loss.backward()

            # Clip gradients to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer1.step()
            optimizer2.step()
            if optimizer3 is not None:
                optimizer3.step()

            # Store per-batch metrics.
            batch_losses.append(loss.item())
            batch_nlls.append(nll.item())
            batch_klds.append(kld.item())

            # Compute and record the gradient norm.
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            batch_grad_norms.append(total_norm)

        # Compute epoch-level metrics.
        train_loss_mean = np.mean(batch_losses)
        train_nll_mean = np.mean(batch_nlls)
        train_kld_mean = np.mean(batch_klds)
        grad_norm_mean = np.mean(batch_grad_norms)
        grad_norm_median = np.median(batch_grad_norms)
        grad_norm_max = np.max(batch_grad_norms)

        # Log training metrics to TensorBoard.
        writer.add_scalar('Train/Loss', train_loss_mean, epoch)
        writer.add_scalar('Train/NLL', train_nll_mean, epoch)
        writer.add_scalar('Train/KLD', train_kld_mean, epoch)
        writer.add_scalar('Train/GradNorm_mean', grad_norm_mean, epoch)
        writer.add_scalar('Train/GradNorm_median', grad_norm_median, epoch)
        writer.add_scalar('Train/GradNorm_max', grad_norm_max, epoch)

        # Log lambda weight if using hybrid mode
        if hasattr(model.decoder, 'lambda_logit'):
            lambda_weight = torch.sigmoid(model.decoder.lambda_logit).item()
            writer.add_scalar('Train/Lambda_weight', lambda_weight, epoch)

        # Save the epoch metrics.
        train_metrics["loss"].append(train_loss_mean)
        train_metrics["nll"].append(train_nll_mean)
        train_metrics["kld"].append(train_kld_mean)

        # --- Validation ---
        # Assume that 'validate' returns mean metrics, and 'validate_median' returns median metrics.
        val_loss_mean, val_nll_mean, val_kld_mean = validate(
            model, valset, model.prior_mean, model.prior_variance,
            n_topics=config.n_topics, sparse_ten=True,
            reg_lambda=reg_lambda, trace_min=trace_min  # Pass parameters here
        )
        val_loss_median, val_nll_median, val_kld_median = validate_median(
            model, valset, model.prior_mean, model.prior_variance,
            n_topics=config.n_topics, sparse_ten=True,
            reg_lambda=reg_lambda, trace_min=trace_min  # Pass parameters here too
        )

        # Log validation metrics.
        writer.add_scalar('Validation/Loss_mean', val_loss_mean, epoch)
        writer.add_scalar('Validation/NLL_mean', val_nll_mean, epoch)
        writer.add_scalar('Validation/KLD_mean', val_kld_mean, epoch)
        writer.add_scalar('Validation/Loss_median', val_loss_median, epoch)
        writer.add_scalar('Validation/NLL_median', val_nll_median, epoch)
        writer.add_scalar('Validation/KLD_median', val_kld_median, epoch)

        # Save validation metrics.
        val_metrics["loss"].append(val_loss_mean)
        val_metrics["nll"].append(val_nll_mean)
        val_metrics["kld"].append(val_kld_mean)

        elapsed = time.time() - epoch_start_time

        # Print metrics every 'print_mod' epochs.
        if epoch % print_mod == 0:
            base_msg = (f"Epoch {epoch}: "
                       f"Train Loss: {train_loss_mean:.4f}, NLL: {train_nll_mean:.4f}, KLD: {train_kld_mean:.4f} | "
                       f"Validation Loss: {val_loss_mean:.4f} (median: {val_loss_median:.4f}) | "
                       f"Grad Norms (mean: {grad_norm_mean:.4f}, median: {grad_norm_median:.4f}, max: {grad_norm_max:.4f})")

            # Add lambda weight if using hybrid mode
            if hasattr(model.decoder, 'lambda_logit'):
                lambda_weight = torch.sigmoid(model.decoder.lambda_logit).item()
                base_msg += f" | λ (Gauss weight): {lambda_weight:.4f}"

            base_msg += f" | Elapsed time: {elapsed:.2f}s"
            print(base_msg)

        # Early stopping based on median validation loss, if enabled.
        if early_stopping:
            if len(past_val_losses) >= n_early_stopping:
                if val_loss_median > max(past_val_losses):
                    print(f"Early stopping: No improvement in median validation loss for {n_early_stopping} epochs.")
                    writer.close()
                    return {"train": train_metrics, "validation": val_metrics}
                else:
                    past_val_losses = past_val_losses[1:] + [val_loss_median]
            else:
                past_val_losses.append(val_loss_median)

        # Save a checkpoint if enabled and it's the right interval
        if save_checkpoints and epoch % save_interval == 0:
            checkpoint_name = f'epoch_{epoch}_topics_{topic_num}_model.pth'
            save_f_name = os.path.join(save_path, checkpoint_name)
            torch.save(model.state_dict(), save_f_name)

    writer.close()
    return {"train": train_metrics, "validation": val_metrics}

    
def smooth_loss(data, window = 100 ):
    """
    smooth the loss
    """
    if isinstance(data, list):
        data = np.array(data)
    
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out
    

def get_topwords(n_topwords, mus_res, L_lower_res, D_log_res, emb_vocab_mat, idx2word, config):
    """
    Compute the topwords according to the paramters of the TLDA model
    
    Args: 
        n_topwords: Number of topwords per topic
        mus_res: means of topics
        L_lower_res: Matrix parametrizing the covariance matrix
        D_log_res: Log of diagonal to parametrize the covariance matrix
        emb_vocab_mat: Matrix with embeddings of each word in the vocabulary, where the words are sorted alphabetically
        idx2word: maps each index to the word
        config: config dict for the model
        
    Return a numpy array of shape (n_topics, n_topwords) that contains the topwords of each topic
    """
    
    # probs1 = torch.exp(calc_beta(mus_res, L_lower_res, D_log_res, emb_vocab_mat, config))
    # probs_np = probs1.detach().cpu().numpy()
    
    # # build vocab_arr in the original idx order:
    # vocab_arr = np.array([idx2word[i] for i in range(len(idx2word))])

    # # Get indices sorted by descending probability (only once):
    # args1 = np.argsort(-probs_np, axis=1)
    
    # # Limit to top-n words per topic:
    # args1_topn = args1[:, :n_topwords]
    
    # # Create row indices corresponding to each topic 
    # topic_indices = np.arange(args1_topn.shape[0])[:, np.newaxis]
    
    # # words1_sort: get top words per topic
    # words1_sort = vocab_arr[args1_topn]
    
    # # probs1_sort: get corresponding probabilities using proper 2D indexing
    # probs1_sort = probs_np[topic_indices, args1_topn]

    probs1 = torch.exp(calc_beta(mus_res, L_lower_res, D_log_res, emb_vocab_mat, config))
    args1 = np.argsort(-probs1.detach().cpu().numpy(), axis = 1)
    
    vocab_arr_sorted = np.array(sorted(list(idx2word.values()))) # this line make the vocab sorted alphabetically
    # check if the vocab_arr is changed by the sorting
    vocab_arr = np.array(list(idx2word.values()))
    if not np.array_equal(vocab_arr, vocab_arr_sorted):
        print("Warning: The vocab_arr is not in the original order. Please check the idx2word mapping.")
    else:
        print("The vocab_arr is in the original order")


    # therefore no matter what the input vocab order is the output with prob-sorted index will be the same here
    words1_sort = np.empty(args1.shape, dtype = vocab_arr.dtype)
    
    for t in range(config.n_topics):
      words1_sort[t] = vocab_arr[args1[t]]
    
    probs1_sort = np.empty(probs1.shape)
    
    for i in range(len(probs1)):
      probs1_sort[i] = probs1[i].detach().cpu()[args1[i]]

    return words1_sort, probs1_sort

# def get_topwords(n_topwords, mus_res, L_lower_res, D_log_res, emb_vocab_mat, vocab, config):
#     """
#     Compute the topwords according to the parameters of the TLDA model.
    
#     Args: 
#         n_topwords (int): Number of topwords per topic.
#         mus_res (torch.Tensor): Means of topics.
#         L_lower_res (torch.Tensor): Lower-triangular factors of the covariance matrices.
#         D_log_res (torch.Tensor): Log of diagonal components for the covariances.
#         emb_vocab_mat (torch.Tensor): Embedding matrix for the vocabulary.
#         vocab (list): List of vocabulary words.
#         config: Configuration object with attributes or keys including n_topics and vocab_size.
        
#     Returns:
#         tuple: (words1_sort, probs1_sort) where:
#             - words1_sort is a NumPy array of shape (n_topics, n_topwords) containing the top words for each topic.
#             - probs1_sort is a NumPy array of shape (n_topics, n_topwords) containing the corresponding probabilities.
#     """
#     # Calculate probabilities for each word under each topic
#     probs1 = torch.exp(calc_beta(mus_res, L_lower_res, D_log_res, emb_vocab_mat, config))
#     probs_np = probs1.detach().cpu().numpy()
    
#     # Convert the vocabulary list to a numpy array.
#     # If alphabetical order is desired, replace vocab with sorted(vocab).
#     vocab_arr = np.array(vocab)
    
#     # Get indices sorted by descending probability:
#     args1 = np.argsort(-probs_np, axis=1)
    
#     # Limit to top-n words per topic:
#     args1_topn = args1[:, :n_topwords]
    
#     # Create row indices corresponding to each topic:
#     topic_indices = np.arange(args1_topn.shape[0])[:, np.newaxis]
    
#     # Retrieve the top words and their probabilities:
#     words1_sort = vocab_arr[args1_topn]
#     probs1_sort = probs_np[topic_indices, args1_topn]
    
#     return words1_sort, probs1_sort

def load_tntm_model(save_path: str, device: torch.device = None):
    """
    Loads a TNTM model from saved metadata and a state dictionary.
    
    Args:
        save_path (str): The directory path where 'metadata.pkl' and 'model_state.pth' are stored.
        device (torch.device, optional): The device on which to load the model. 
                                         If None, uses CUDA if available, else CPU.
    
    Returns:
        model (TNTM_bow): The TNTM model loaded with trained parameters and set to evaluation mode.
        train_config (namedtuple): The training configuration used to create the model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths to the saved metadata and model state
    metadata_path = os.path.join(save_path, "metadata.pkl")
    state_path = os.path.join(save_path, "model_state.pth")
    
    # Load metadata
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    vocab = metadata['vocab']
    word2idx = metadata['word2idx']
    idx2word = metadata['idx2word']
    train_config_dict = metadata['train_config']  # This is a dictionary
    embeddings_proj_np = metadata['embeddings_proj']
    prior_mean_np = metadata['prior_mean']
    prior_variance_np = metadata['prior_variance']
    
    # Convert numpy arrays to torch tensors
    embeddings_proj = torch.tensor(embeddings_proj_np, dtype=torch.float32).to(device)
    prior_mean = torch.tensor(prior_mean_np, dtype=torch.float32).to(device)
    prior_variance = torch.tensor(prior_variance_np, dtype=torch.float32).to(device)
    
    # Reconstruct the training configuration as a namedtuple
    TrainConfig = namedtuple("TrainConfig", list(train_config_dict.keys()))
    train_config = TrainConfig(**train_config_dict)
    
    # Create dummy initializations with the correct shapes.
    n_topics = train_config.n_topics        # e.g., 20
    embedding_dim = train_config.embedding_dim  # e.g., 11
    
    # These dummy tensors are only used to initialize the model; they will be overwritten 
    # when loading the saved state dictionary.
    dummy_mus = torch.zeros(n_topics, embedding_dim, dtype=torch.float32).to(device)           # Shape: [n_topics, embedding_dim]
    dummy_lower = torch.zeros(n_topics, embedding_dim, embedding_dim, dtype=torch.float32).to(device)  # Shape: [n_topics, embedding_dim, embedding_dim]
    dummy_log_diag = torch.zeros(n_topics, embedding_dim, dtype=torch.float32).to(device)        # Shape: [n_topics, embedding_dim]
    
    # Instantiate the model using the loaded parameters and dummy initializations.
    model = TNTM_bow(
        embeddings=embeddings_proj,
        mus_init=dummy_mus,
        lower_init=dummy_lower,
        log_diag_init=dummy_log_diag,
        config=train_config,
        prior_mean=prior_mean,
        prior_variance=prior_variance
    ).to(device)
    
    # Load the saved state dictionary into the model.
    model_state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(model_state_dict)
    
    # Set the model to evaluation mode.
    model.eval()
    
    return model, train_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)