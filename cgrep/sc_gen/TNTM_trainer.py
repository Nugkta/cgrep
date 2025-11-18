from multiprocessing.process import parent_process
import torch
from torch import nn
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sklearn

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



from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

import octis
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

from . import Initialization as init
from . import TNTM_inference





import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import namedtuple
from tqdm import tqdm
import os

from . import Initialization as init
from . import TNTM_inference  # This module contains TNTM_bow, train_loop, get_topwords, etc.

class TNTMTrainer:
    def __init__(self,
                 n_topics: int,
                 save_path: str,
                 n_dims: int = 11,
                 n_hidden_units: int = 200,
                 n_encoder_layers: int = 3,
                 enc_lr: float = 1e-3,
                 dec_lr: float = 1e-3,
                 n_epochs: int = 100,
                 batch_size: int = 128,
                 dropout_rate_encoder: float = 0.3,
                 prior_variance=0.995,
                 prior_mean=None,
                 n_topwords: int = 10,
                 device: str = None,
                 validation_set_size: float = 0.2,
                 early_stopping: bool = True,
                 n_epochs_early_stopping: int = 10,
                 log_diag_init_eps: float = 0.1,    # Initial value for log_diag initialization
                 reg_lambda: float = 0.1,           # Regularization coefficient
                 trace_min: float = 0.5,            # Minimum trace threshold for regularization
                 use_hybrid: bool = False,          # Whether to use hybrid beta (Gaussian + ProdLDA)
                 prodlda_only: bool = False,        # Whether to use only ProdLDA decoder
                 prodlda_lr: float = 1e-3):         # Learning rate for ProdLDA parameters
        """
        Trainer for the TNTM model. This class is responsible for preparing the data,
        building the model using the provided configuration, running training, and saving
        the model's state. The TNTM model architecture itself is defined in TNTM_inference.

        Parameters:
            n_topics (int): number of topics.
            save_path (str): Path for saving the model state (and related outputs).
            n_dims (int): Dimensionality of the word embedding space.
            n_hidden_units (int): Number of hidden units in the encoder.
            n_encoder_layers (int): Number of skip layers in the encoder.
            enc_lr (float): Learning rate for the encoder.
            dec_lr (float): Learning rate for the decoder.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            dropout_rate_encoder (float): Dropout rate for the encoder.
            prior_variance: Either a float (symmetric prior variance) or a tensor of shape (1, n_topics).
            prior_mean: Tensor of shape (1, n_topics) or None (default zeros).
            n_topwords (int): Number of top words per topic.
            device (str): "cpu" or "cuda". If None, use cuda if available.
            validation_set_size (float): Fraction of dataset for validation.
            early_stopping (bool): Whether to use early stopping based on validation loss.
            n_epochs_early_stopping (int): Patience parameter for early stopping.
            log_diag_init_eps (float): Epsilon value for initializing the log diagonal matrix.
            reg_lambda (float): Regularization coefficient for the variance penalty.
            trace_min (float): Minimum trace threshold for regularization.
            use_hybrid (bool): Whether to use hybrid beta combining Gaussian and ProdLDA approaches.
            prodlda_lr (float): Learning rate for ProdLDA parameters (beta_prodlda and lambda).
        """
        self.n_topics = n_topics
        self.save_path = save_path
        self.n_dims = n_dims
        self.n_hidden_units = n_hidden_units
        self.n_encoder_layers = n_encoder_layers
        self.enc_lr = enc_lr
        self.dec_lr = dec_lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_topwords = n_topwords
        self.dropout_rate_encoder = dropout_rate_encoder
        self.validation_set_size = validation_set_size
        self.early_stopping = early_stopping
        self.n_epochs_early_stopping = n_epochs_early_stopping

        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Handle prior variance
        assert type(prior_variance) in [float, torch.Tensor], "prior_variance must be a float or a tensor"
        if isinstance(prior_variance, float):
            self.prior_var = torch.Tensor(1, self.n_topics).fill_(prior_variance).to(self.device)
        else:
            self.prior_var = prior_variance.to(self.device)
        self.prior_logvar = torch.log(self.prior_var).to(self.device)

        # Handle prior mean
        if prior_mean is None:
            self.prior_mean = torch.Tensor(1, self.n_topics).fill_(0).to(self.device)
        else:
            self.prior_mean = prior_mean.to(self.device)

        # New initializations
        self.log_diag_init_eps = log_diag_init_eps
        self.reg_lambda = reg_lambda
        self.trace_min = trace_min  # Updated from v_min to trace_min

        # Hybrid mode parameters
        if use_hybrid and prodlda_only:
            raise ValueError("use_hybrid and prodlda_only cannot both be True.")
        self.use_hybrid = use_hybrid
        self.prodlda_only = prodlda_only
        self.prodlda_lr = prodlda_lr
        
        # These will be set later during data preparation and model building.
        self.vocab = None
        self.corpus = None
        self.embeddings = None
        self.word2idx = None
        self.idx2word = None
        self.embedding_ten = None
        self.bow_ten = None
        self.train_config = None
        self.model = None
        self.train_ds = self.val_ds = self.test_ds = None
        self.embeddings_proj_ten = None
        self.mus_init = None
        self.L_lower_init = None
        self.log_diag_init = None

    def prepare_data(self, corpus: list, vocab: list, embeddings: torch.Tensor):
        """
        Prepares data from corpus, vocabulary, and word embeddings.
        
        Parameters:
            corpus (list[list[str]]): List of documents (each document is a list of words).
            vocab (list[str]): List of unique words.
            embeddings (torch.Tensor): Word embeddings with shape (len(vocab), embedding_dim).
        """
        self.corpus = corpus
        self.vocab = vocab
        self.embeddings = embeddings

        # Create word-index mappings.
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for i, word in enumerate(vocab)}
        self.embedding_ten = embeddings.to(self.device)

        # Build bag-of-words tensor.
        bow_ten = torch.zeros(len(corpus), len(vocab))
        corpus_idx = [[self.word2idx[word] for word in doc] for doc in corpus]
        for i, doc in tqdm(enumerate(corpus_idx), desc="Building BoW representation"):
            for word in doc:
                bow_ten[i, word] += 1
        self.bow_ten = bow_ten

        # Initialize low-dimensional embeddings and initial topic assignments.
        init_in = init.Initializer(self.embedding_ten.cpu().detach().numpy(), n_topics=self.n_topics, n_dims=self.n_dims)
        
        # Pass the custom epsilon value for log_diag initialization
        embeddings_proj, mus_init, L_lower_init, log_diag_init, bic = init_in.reduce_dim_and_cluster(eps=self.log_diag_init_eps)
        
        # Fix tensor conversions to avoid warnings
        self.embeddings_proj_ten = torch.as_tensor(embeddings_proj).to(self.device)
        
        # Properly handle tensor conversion based on input type
        if isinstance(mus_init, torch.Tensor):
            self.mus_init = mus_init.clone().detach().to(self.device)
        else:
            self.mus_init = torch.as_tensor(mus_init).to(self.device)
        
        if isinstance(L_lower_init, torch.Tensor):
            self.L_lower_init = L_lower_init.clone().detach().to(self.device)
        else:
            self.L_lower_init = torch.as_tensor(L_lower_init).to(self.device)
        
        if isinstance(log_diag_init, torch.Tensor):
            self.log_diag_init = log_diag_init.clone().detach().to(self.device)
        else:
            self.log_diag_init = torch.as_tensor(log_diag_init).to(self.device)

        # Split data into training, validation, and test sets.
        self.train_ds, self.val_ds, self.test_ds = TNTM_inference.train_test_split(
            self.bow_ten, 1 - self.validation_set_size, self.validation_set_size, self.batch_size)

        # Build the training configuration.
        config_dict = {
            "num_input": len(self.vocab),
            "n_hidden_block": self.n_hidden_units,
            "n_skip_layers": self.n_encoder_layers,
            "n_topics": self.n_topics,
            "drop_rate_en": self.dropout_rate_encoder,
            "init_mult": 1,
            "vocab_size": len(self.vocab),
            "embedding_dim": self.n_dims,
            "early_stopping": self.early_stopping,
            "n_epochs_early_stopping": self.n_epochs_early_stopping,
            "n_topwords": self.n_topwords
        }
        self.train_config = namedtuple("TrainConfig", config_dict.keys())(*config_dict.values())

    def build_model(self):
        """
        Builds the TNTM model using the low-dimensional embeddings and initializations
        computed during data preparation.
        """
        # Check for NaN/Inf in initialization parameters
        if torch.isnan(self.mus_init).any() or torch.isinf(self.mus_init).any():
            print("WARNING: NaN or Inf detected in mus_init, replacing with zeros")
            self.mus_init = torch.zeros_like(self.mus_init)

        if torch.isnan(self.L_lower_init).any() or torch.isinf(self.L_lower_init).any():
            print("WARNING: NaN or Inf detected in L_lower_init, replacing with small values")
            self.L_lower_init = torch.randn_like(self.L_lower_init) * 0.01

        if torch.isnan(self.log_diag_init).any() or torch.isinf(self.log_diag_init).any():
            print("WARNING: NaN or Inf detected in log_diag_init, replacing with default")
            self.log_diag_init = torch.ones_like(self.log_diag_init) * self.log_diag_init_eps

        self.model = TNTM_inference.TNTM_bow(
            embeddings      = self.embeddings_proj_ten.to(self.device),
                   mus_init       = self.mus_init.to(self.device),
                   lower_init     = self.L_lower_init.to(self.device),
                   log_diag_init  = self.log_diag_init.to(self.device),
                   config         = self.train_config,
                   prior_mean     = self.prior_mean.to(self.device),
                   prior_variance = self.prior_var.to(self.device),
                   use_hybrid     = self.use_hybrid,
                   prodlda_only   = self.prodlda_only,
                   beta_prodlda_init = None  # Let the model initialize it
                   ).to(self.device)

        print(f"Model built successfully on device: {self.device}")
        print(f"Model parameters: encoder={sum(p.numel() for p in self.model.encoder.parameters())}, "
              f"decoder={sum(p.numel() for p in self.model.decoder.parameters())}")


    def train(self):
        """
        Trains the TNTM model using the prepared data and training configuration.
        This method performs the following steps:
        1. Initializes optimizers for the encoder and decoder.
        2. Creates necessary directories for saving model states and TensorBoard logs.
        3. Runs the training loop, which includes regularization parameters.
        4. Retrieves final topic parameters from the model's decoder.
        5. Extracts top words and their probabilities for each topic.
        6. Saves the top words and probabilities to CSV files.
        7. Filters the top words and probabilities based on a cumulative threshold.
        8. Saves the filtered top words and probabilities as PKL files.
        9. Saves the model's state dictionary.
        10. Prepares and saves metadata including vocabulary, word mappings, and training configuration.

        Returns:
            tuple: (filtered_topwords, filtered_probs, metrics)
                - filtered_topwords (dict): Filtered top words for each topic.
                - filtered_probs (dict): Filtered probabilities for each top word.
                - metrics (dict): Training metrics including losses and other relevant information.
        """
        # Create optimizers for the encoder and decoder.
        opt1 = torch.optim.Adam(self.model.encoder.parameters(), lr=self.enc_lr, betas=(0.99, 0.999))

        # Configure decoder optimizers based on chosen mode
        if self.prodlda_only:
            for param in [self.model.decoder.mus,
                          self.model.decoder.L_lower,
                          self.model.decoder.log_diag]:
                param.requires_grad_(False)
            prodlda_params = [self.model.decoder.log_beta_prodlda]
            opt2 = torch.optim.Adam(prodlda_params, lr=self.prodlda_lr)
            opt3 = None
        elif self.use_hybrid:
            # Gaussian parameters (mus, L_lower, log_diag)
            gaussian_params = [
                self.model.decoder.mus,
                self.model.decoder.L_lower,
                self.model.decoder.log_diag
            ]
            opt2 = torch.optim.Adam(gaussian_params, lr=self.dec_lr)

            # ProdLDA parameters (log_beta_prodlda, lambda_logit)
            prodlda_params = [
                self.model.decoder.log_beta_prodlda,
                self.model.decoder.lambda_logit
            ]
            opt3 = torch.optim.Adam(prodlda_params, lr=self.prodlda_lr)
        else:
            # Use all decoder parameters
            opt2 = torch.optim.Adam(self.model.decoder.parameters(), lr=self.dec_lr)
            opt3 = None

        # Check if the save directory exists; if not, create it.
        if not os.path.exists(self.save_path):
            print(f"Saving directory is not detected, creating directory: {self.save_path}")
            os.makedirs(self.save_path)

        # Create a directory for TensorBoard logs if it does not exist.
        if not os.path.exists(os.path.join(self.save_path, 'tensorboard')):
            print(f"Creating Tensorboard Directory: {os.path.join(self.save_path, 'tensorboard')}")
            os.makedirs(os.path.join(self.save_path, 'tensorboard'))
        
        self.save_path_tensorboard = os.path.join(self.save_path, 'tensorboard')

        # Run the training loop.
        # All regularization parameters (reg_lambda and trace_min) are passed here to ensure they are used consistently.
        metrics = TNTM_inference.train_loop(
            model=self.model,
            optimizer1=opt1,
            optimizer2=opt2,
            trainset=self.train_ds,
            valset=self.val_ds,
            print_mod=1,
            device=self.device,
            n_epochs=self.n_epochs,
            save_path=self.save_path,
            config=self.train_config,
            topic_num=self.n_topics,
            tensorboard_log_dir=self.save_path_tensorboard,
            # Pass the variance regularization parameters:
            reg_lambda=self.reg_lambda,
            trace_min=self.trace_min,  # Updated from v_min to trace_min
            optimizer3=opt3  # Pass the ProdLDA optimizer if using hybrid mode
        )

        # Retrieve final topic parameters from the model's decoder.
        self.mus_res = self.model.decoder.mus.detach()
        self.L_lower_res = self.model.decoder.L_lower.detach()
        self.log_diag_res = self.model.decoder.log_diag.detach()

        # If using hybrid mode, save lambda weights
        if self.use_hybrid:
            lambda_weights = torch.sigmoid(self.model.decoder.lambda_logit).detach().cpu()
            self.lambda_weights = lambda_weights  # Store all weights
            print(f"\nFinal Lambda weights (Gaussian) - Mean: {lambda_weights.mean():.4f}, "
                  f"Std: {lambda_weights.std():.4f}, Min: {lambda_weights.min():.4f}, Max: {lambda_weights.max():.4f}")
            print(f"Final Lambda weights (ProdLDA) - Mean: {(1 - lambda_weights).mean():.4f}, "
                  f"Std: {(1 - lambda_weights).std():.4f}, Min: {(1 - lambda_weights).min():.4f}, Max: {(1 - lambda_weights).max():.4f}")

        # Get top words and their probabilities using the TNTM_inference.get_topwords() function.
        log_beta_prodlda = None
        lambda_logit = None
        if self.use_hybrid or self.prodlda_only:
            log_beta_prodlda = self.model.decoder.log_beta_prodlda.detach()
        if self.use_hybrid:
            lambda_logit = self.model.decoder.lambda_logit.detach()

        topwords, probs = TNTM_inference.get_topwords(
            n_topwords=self.n_topwords,
            mus_res=self.mus_res,
            L_lower_res=self.L_lower_res,
            D_log_res=self.log_diag_res,
            emb_vocab_mat=self.embeddings_proj_ten,
            idx2word=self.idx2word,
            config=self.train_config,
            log_beta_prodlda=log_beta_prodlda,
            lambda_logit=lambda_logit
        )
        self.topwords = topwords
        self.probs = probs

        # Save the topic words and probabilities into CSV files.
        pd.DataFrame(self.topwords).to_csv(os.path.join(self.save_path, 'topwords.csv'), index=False)
        pd.DataFrame(self.probs).to_csv(os.path.join(self.save_path, 'probs.csv'), index=False)

        # Filter the top words and probabilities based on the cumulative threshold.
        filtered_topwords, filtered_probs = filter_topic_word_matrices(
            pd.DataFrame(self.topwords),
            pd.DataFrame(self.probs)
        )
        # Save filtered matrices as PKL files.
        pickle.dump(filtered_topwords, open(os.path.join(self.save_path, 'filtered_topwords.pkl'), 'wb'))
        pickle.dump(filtered_probs, open(os.path.join(self.save_path, 'filtered_probs.pkl'), 'wb'))
        
        # Save the model's state dictionary for later use or reloading.
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_state.pth'))

        # Prepare metadata to save, including vocabulary, word mappings, and training configuration.
        metadata = {
            'vocab': self.vocab,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'train_config': self.train_config._asdict(),  # Convert namedtuple to dictionary
            'embeddings_proj': self.embeddings_proj_ten.cpu().detach().numpy(),
            'prior_mean': self.prior_mean.cpu().detach().numpy(),
            'prior_variance': self.prior_var.cpu().detach().numpy()
        }

        # Save metadata using pickle.
        with open(os.path.join(self.save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        return filtered_topwords, filtered_probs, metrics


def filter_topic_word_matrices(topwords, probs, cumulative_threshold=0.90, min_contrib=0.01):
    """
    Filter the top words and probabilities based on a threshold.
    
    Parameters:
        topwords (list[list[str]]): List of top words for each topic.
        probs (list[list[float]]): List of probabilities for each top word.
        threshold (float): Minimum probability for a word to be included.
    
    Returns:
        filtered_words (dict): Dictionary of filtered top words for each topic.
        filtered_probs (dict): Dictionary of filtered probabilities for each top words for each topic
    """

    # topwords = topwords.iloc[1:, 1:] # remove the first row and column which are indices
    # top_word_prob = probs.iloc[1:, 1:]
    top_word_prob = probs
    top_word_prob_normalized = top_word_prob.div(top_word_prob.sum(axis=1), axis=0)
    filtered_topword, filtered_prob = filter_top_words_by_cumulative_and_individual_threshold(topwords, top_word_prob_normalized,
                                                                            cumulative_threshold, min_contrib)
    return filtered_topword, filtered_prob







def filter_top_words_by_cumulative_and_individual_threshold(top_words_df, top_word_prob_normalized_df,
                                                              cumulative_threshold=.8, min_contrib=1e-3):
    """
    For each topic, select the top words (assumed to be sorted in descending order by contribution) 
    that meet two conditions:
      1. Each word must have an individual contribution >= min_contrib.
      2. The cumulative sum of contributions of the selected words reaches or exceeds cumulative_threshold.
    
    Parameters:
        top_words_df: pd.DataFrame
            A DataFrame of shape (n_topics, n_possible_top_words) containing topic words in descending order.
        top_word_prob_normalized_df: pd.DataFrame
            A DataFrame of the same shape as top_words_df containing the normalized contribution (probability) 
            of each word.
        cumulative_threshold: float, default=0.8
            The cumulative contribution threshold (e.g., 0.8 means we stop once the selected words add up to 80%).
        min_contrib: float, default=1e-4
            The minimum individual contribution required for a word to be considered.
    
    Returns:
        filtered_words: dict
            A dictionary where each key is the topic index (integer) and the value is a list of words that 
            pass the individual contribution threshold and collectively meet the cumulative threshold.
    """
    filtered_words = {}
    filtered_probs = {}
    n_topics = top_word_prob_normalized_df.shape[0]
    for i in range(n_topics):
        # Extract the sorted words and probabilities for this topic.
        # print('deal with topic:', i)
        topic_words = top_words_df.iloc[i].values
        topic_probs = top_word_prob_normalized_df.iloc[i].values
        
        selected_words = []
        selected_probs = []
        cum_sum = 0.0
        
        
        # Iterate over words and their corresponding probability (assumed sorted descending)
        for word, prob in zip(topic_words, topic_probs):
            # Discard and break if the word's individual contribution is too low.
            if pd.isna(prob) or prob < min_contrib:
                # print('the word:', word, 'has a prob:', prob, 'which is less than the min_contrib:', min_contrib)
                # print('termintate the loop')
                break
            selected_words.append(word)
            selected_probs.append(prob)
            # print('selected_words:', selected_words)

            cum_sum += prob
            # Stop once we have accumulated enough contribution.
            if cum_sum >= cumulative_threshold:
                break
        
        filtered_words[i] = selected_words
        filtered_probs[i] = selected_probs

    
    return filtered_words, filtered_probs
