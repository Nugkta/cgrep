import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm.auto import tqdm
import pandas as pd
import math
from sklearn.model_selection import KFold


# THE MLP MODEL
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Note: No softmax at the end - using sigmoid implicitly with BCEWithLogitsLoss
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class BGCDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class MultiLabelBGCDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Float for BCE loss
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SequenceBGCDataset(Dataset):
    """Dataset for handling sequence embeddings for BGC classification."""
    def __init__(self, embeddings, labels, max_seq_len=None):
        """
        Args:
            embeddings: List of sequence embeddings of shape (seq_len, embed_dim)
            labels: Multi-hot encoded labels
            max_seq_len: Optional maximum sequence length for padding/truncation
        """
        self.max_seq_len = max_seq_len
        self.embeddings = embeddings
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        # Get sequence embedding
        embed = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        
        # Handle padding/truncation if max_seq_len is specified
        if self.max_seq_len is not None:
            seq_len, embed_dim = embed.shape
            if seq_len > self.max_seq_len:
                # Truncate
                embed = embed[:self.max_seq_len, :]
            elif seq_len < self.max_seq_len:
                # Pad with zeros
                pad = torch.zeros(self.max_seq_len - seq_len, embed_dim)
                embed = torch.cat([embed, pad], dim=0)
            
        return embed, self.labels[idx], (embed.shape[0] if self.max_seq_len is None else min(embed.shape[0], self.max_seq_len))


class MultiLabelBGCClassifier:
    def __init__(self, 
                 hidden_dim=256, 
                 dropout_rate=0.5, 
                 lr=0.001,
                 batch_size=32, 
                 early_stopping_patience=10,
                 max_epochs=100,
                 random_seed=42,
                 device=None):
        """Initialize a multi-label BGC classifier using an MLP architecture."""
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Set random seed
        self.set_seed(random_seed)
        
        # Initialize other attributes
        self.model = None
        self.mlb = None  # MultiLabelBinarizer
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self, X, y, test_split=0.15, n_folds=5, show_progress=False):
        """Train the multi-label model using k-fold cross-validation."""
        print(f"Using device: {self.device}")
        
        # Process multi-label data
        self.mlb = MultiLabelBinarizer()
        processed_labels = [label.split(';') for label in y]
        y_encoded = self.mlb.fit_transform(processed_labels)
        num_classes = len(self.mlb.classes_)
        
        print(f"Total samples: {len(X)}")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {self.mlb.classes_}")
        
        # Create dataset for all data
        dataset = MultiLabelBGCDataset(X, y_encoded)
        
        # Split off test set first
        test_size = int(test_split * len(dataset))
        train_val_size = len(dataset) - test_size
        train_val_dataset, test_dataset = random_split(
            dataset, 
            [train_val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        # Prepare test loader
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        print(f"Train/Val samples: {len(train_val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Initialize K-Fold cross-validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        # Extract features and labels from train_val_dataset for cross-validation
        train_val_indices = train_val_dataset.indices
        X_train_val = X[train_val_indices]
        y_train_val = y_encoded[train_val_indices]
        
        # Track best model across all folds
        best_val_loss = float('inf')
        best_model_state = None
        fold_histories = []
        fold_metrics = []
        
        print(f"Starting {n_folds}-fold cross-validation...")
        
        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val)):
            print(f"\nFold {fold+1}/{n_folds}")
            
            # Create train and validation datasets for this fold
            X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
            y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]
            
            train_dataset_fold = MultiLabelBGCDataset(X_train_fold, y_train_fold)
            val_dataset_fold = MultiLabelBGCDataset(X_val_fold, y_val_fold)
            
            train_loader = DataLoader(train_dataset_fold, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset_fold, batch_size=self.batch_size)
            
            print(f"  Train samples: {len(train_dataset_fold)}")
            print(f"  Validation samples: {len(val_dataset_fold)}")
            
            # Build model for this fold
            input_dim = X.shape[1]
            self.model = MultiLabelMLP(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=num_classes,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            if fold == 0:  # Only print architecture once
                print(f"Building Multi-Label MLP with input_dim={input_dim}, hidden_dim={self.hidden_dim}, output_dim={num_classes}")
            
            # Define loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()  # Standard BCE loss
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=False
            )
            
            # History for this fold
            fold_history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
            
            # Training loop for this fold
            fold_best_val_loss = float('inf')
            fold_best_model_state = None
            no_improve_count = 0
            
            for epoch in tqdm(range(self.max_epochs), 
                             desc=f"Training Fold {fold+1}/{n_folds}",
                             disable=not show_progress):
                # Train
                train_loss, _ = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validate
                val_loss, val_metrics = self._validate(val_loader, criterion)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Calculate F1 score for display
                val_f1 = val_metrics.get('f1', 0.0)
                
                # Print results
                if show_progress and (epoch+1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.max_epochs}, "
                         f"Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                
                # Save fold history
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['val_f1'].append(val_f1)
                
                # Check if this is the best model for this fold
                if val_loss < fold_best_val_loss:
                    fold_best_val_loss = val_loss
                    fold_best_model_state = copy.deepcopy(self.model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stopping for this fold
                if no_improve_count >= self.early_stopping_patience:
                    if show_progress:
                        print(f"  No improvement for {self.early_stopping_patience} epochs. Stopping training for fold {fold+1}.")
                    break
            
            # Load best model for this fold and evaluate
            self.model.load_state_dict(fold_best_model_state)
            fold_val_results = self._evaluate(val_loader)
            
            # Record metrics for this fold
            fold_metrics.append({
                'fold': fold+1,
                'val_loss': fold_best_val_loss,
                'val_accuracy': fold_val_results['accuracy'],
                'val_hamming_loss': fold_val_results['hamming_loss'],
                'val_auroc_micro': fold_val_results['auroc_micro'],
                'val_auroc_macro': fold_val_results['auroc_macro']
            })
            
            print(f"  Fold {fold+1} best validation loss: {fold_best_val_loss:.4f}")
            print(f"  Fold {fold+1} validation accuracy: {fold_val_results['accuracy']:.4f}")
            print(f"  Fold {fold+1} validation AUROC (micro): {fold_val_results['auroc_micro']:.4f}")
            
            # Store fold history
            fold_histories.append(fold_history)
            
            # Update best overall model if this fold's model is better
            if fold_best_val_loss < best_val_loss:
                best_val_loss = fold_best_val_loss
                best_model_state = fold_best_model_state
                print(f"  New best model from fold {fold+1}")
        
        # Print cross-validation summary
        print("\nCross-Validation Summary:")
        avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) 
                      for key in ['val_loss', 'val_accuracy', 'val_hamming_loss', 
                                 'val_auroc_micro', 'val_auroc_macro']}
        
        print(f"Average validation loss: {avg_metrics['val_loss']:.4f}")
        print(f"Average validation accuracy: {avg_metrics['val_accuracy']:.4f}")
        print(f"Average validation hamming loss: {avg_metrics['val_hamming_loss']:.4f}")
        print(f"Average validation AUROC (micro): {avg_metrics['val_auroc_micro']:.4f}")
        print(f"Average validation AUROC (macro): {avg_metrics['val_auroc_macro']:.4f}")
        
        # FIX: More robust history aggregation
        if fold_histories:
            # Find minimum length across all fold histories (ensure it's not 0)
            min_train_len = min(len(h['train_loss']) for h in fold_histories)
            min_val_len = min(len(h['val_loss']) for h in fold_histories)
            min_val_f1_len = min(len(h['val_f1']) for h in fold_histories)
            
            # Use the minimum length that's above 0
            min_len = max(1, min(min_train_len, min_val_len, min_val_f1_len))
            
            # Truncate to same length and then average
            self.history = {
                'train_loss': np.mean([[h['train_loss'][i] for i in range(min(min_len, len(h['train_loss'])))] 
                                      for h in fold_histories], axis=0).tolist(),
                'val_loss': np.mean([[h['val_loss'][i] for i in range(min(min_len, len(h['val_loss'])))]
                                    for h in fold_histories], axis=0).tolist(),
                'val_acc': np.mean([[h['val_f1'][i] for i in range(min(min_len, len(h['val_f1'])))]
                                   for h in fold_histories], axis=0).tolist()
            }
            
            # Ensure train_acc exists with proper length
            self.history['train_acc'] = [0.0] * len(self.history['train_loss'])
        else:
            # Fallback if no histories available
            self.history = {
                'train_loss': [0.0],
                'val_loss': [0.0],
                'train_acc': [0.0],
                'val_acc': [0.0]
            }
            print("Warning: No training history available for plotting.")
        
        # Load best model from all folds for final evaluation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Evaluate best model on test set
        if test_split > 0.0 and len(test_dataset) > 0:
            test_results = self._evaluate(test_loader)
            print(f"\nTest Results (Best Model):")
            print(f"Test exact match accuracy: {test_results['accuracy']:.4f}")
            print(f"Test hamming loss: {test_results['hamming_loss']:.4f}")
            print(f"Test AUROC (micro-avg): {test_results['auroc_micro']:.4f}")
            print(f"Test AUROC (macro-avg): {test_results['auroc_macro']:.4f}")
            print(f"Test AUROC (weighted): {test_results['auroc_weighted']:.4f}")

            # Print per-class AUROC
            print("\nPer-class AUROC:")
            for class_name, auroc in test_results['auroc_per_class']:
                print(f"{class_name}: {auroc:.4f}")
        else:
            print("\nNo test set provided (test_split=0.0). Skipping test evaluation.")

        return self

    def _train_epoch(self, loader, optimizer, criterion):
        """Train for one epoch with multi-label data."""
        self.model.train()
        running_loss = 0.0
        total = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device).float()
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
        
        epoch_loss = running_loss / total
        return epoch_loss, {}

    def _validate(self, loader, criterion):
        """Validate multi-label model."""
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
                
                # Store outputs and labels for metric calculation
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Calculate metrics
        outputs = torch.cat(all_outputs, dim=0).sigmoid().numpy()  # Apply sigmoid to get probabilities
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Apply threshold to get binary predictions
        predictions = (outputs > 0.5).astype(int)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, predictions, average='macro')
        
        epoch_loss = running_loss / total
        return epoch_loss, {'f1': f1}

    def _evaluate(self, loader):
        """Evaluate multi-label model with detailed metrics."""
        self.model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Convert to numpy for evaluation
        outputs = torch.sigmoid(torch.cat(all_outputs, dim=0)).numpy()  # Apply sigmoid for probabilities
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Apply threshold to get binary predictions
        predictions = (outputs > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import (hamming_loss, accuracy_score, roc_auc_score,
                                    precision_score, recall_score, f1_score)
        
        # Calculate AUROC scores for multi-label classification
        try:
            auroc_micro = roc_auc_score(labels, outputs, average='micro')
            auroc_macro = roc_auc_score(labels, outputs, average='macro')
            auroc_weighted = roc_auc_score(labels, outputs, average='weighted')
            auroc_per_class = []
            
            # Per-class AUROC
            for i in range(labels.shape[1]):
                if len(np.unique(labels[:, i])) > 1:  # Only calculate if both classes present
                    auroc = roc_auc_score(labels[:, i], outputs[:, i])
                    auroc_per_class.append((self.mlb.classes_[i], auroc))
        except Exception as e:
            print(f"Warning: Could not compute AUROC: {e}")
            auroc_micro = auroc_macro = auroc_weighted = 0.0
            auroc_per_class = []
        
        # Calculate precision, recall and F1 scores
        try:
            precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
            recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
            f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
            
            # Calculate support (number of occurrences of each class)
            support_per_class = np.sum(labels, axis=0)
            
            # Create detailed per-class metrics report
            class_metrics = []
            for i, class_name in enumerate(self.mlb.classes_):
                class_metrics.append({
                    'class': class_name,
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1': f1_per_class[i],
                    'support': support_per_class[i],
                    'auroc': next((auroc for name, auroc in auroc_per_class if name == class_name), 0.0)
                })
        except Exception as e:
            print(f"Warning: Could not compute precision/recall metrics: {e}")
            class_metrics = []
        
        # Sample-wise accuracy (exact matches)
        accuracy = accuracy_score(labels, predictions)
        
        # Convert to original label format for reporting
        pred_labels = [';'.join(sorted([self.mlb.classes_[i] for i in range(len(pred)) if pred[i]])) 
                       for pred in predictions]
        true_labels = [';'.join(sorted([self.mlb.classes_[i] for i in range(len(true)) if true[i]]))
                       for true in labels]
        
        return {
            'accuracy': accuracy,
            'hamming_loss': hamming_loss(labels, predictions),
            'auroc_micro': auroc_micro,
            'auroc_macro': auroc_macro,
            'auroc_weighted': auroc_weighted,
            'auroc_per_class': auroc_per_class,
            'class_metrics': class_metrics,
            'predictions': pred_labels,
            'true_labels': true_labels
        }
        
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
        # Convert to original class names
        pred_labels = []
        for pred in predictions:
            classes = [self.mlb.classes_[i] for i in range(len(pred)) if pred[i]]
            pred_labels.append(';'.join(sorted(classes)) if classes else '')
            
        return pred_labels
    
    def plot_history(self):
        """Plot training history."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_acc'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Validation F1 Score')
        
        plt.tight_layout()
        plt.show()
        
    def save(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No trained model to save.")
            
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'mlb': self.mlb,
            'hyperparams': {
                'hidden_dim': self.hidden_dim,
                'dropout_rate': self.dropout_rate,
                'input_dim': self.model.fc1.in_features,
                'output_dim': self.model.fc3.out_features
            },
            'history': self.history
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath, device=None):
        """Load model from file."""
        save_dict = torch.load(filepath, map_location='cpu')
        
        # Create instance with same hyperparameters
        instance = cls(
            hidden_dim=save_dict['hyperparams']['hidden_dim'],
            dropout_rate=save_dict['hyperparams']['dropout_rate'],
            device=device
        )
        
        # Create model with correct dimensions
        input_dim = save_dict['hyperparams']['input_dim']
        output_dim = save_dict['hyperparams']['output_dim']
        
        instance.model = MultiLabelMLP(input_dim, instance.hidden_dim, output_dim, instance.dropout_rate)
        instance.model.load_state_dict(save_dict['model_state_dict'])
        instance.model.to(instance.device)
        
        # Load other attributes
        instance.mlb = save_dict['mlb']
        instance.history = save_dict['history']
        
        return instance






class PositionalEncoding(nn.Module):
    """Implement the standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class SequenceTransformer(nn.Module):
    """Transformer-based encoder for multi-label classification of sequence data."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 hidden_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 100,
                 pooling_strategy: str = 'mean'):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension in feed-forward networks
            num_classes: Number of output classes
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            pooling_strategy: How to pool sequence representations ('mean', 'first', or 'max')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pooling_strategy = pooling_strategy
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=False  # PyTorch expects (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
            src_key_padding_mask: Optional mask for padded positions (batch_size, seq_len)
                                 with True indicating positions to be masked
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects shape (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        
        # Encode sequence
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Reshape back to (batch_size, seq_len, embed_dim) for pooling
        encoded = encoded.transpose(0, 1)
        
        # Apply pooling strategy
        if self.pooling_strategy == 'first':
            # Use first token as representation (like CLS token)
            pooled = encoded[:, 0, :]
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence dimension
            pooled, _ = torch.max(encoded, dim=1)
        else:  # Default to mean pooling
            # Mean pooling over sequence dimension
            if src_key_padding_mask is not None:
                # Create a mask of shape (batch_size, seq_len, 1) with 0s for masked positions
                mask = (~src_key_padding_mask).float().unsqueeze(-1)
                # Apply mask and calculate mean only over unmasked positions
                pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = encoded.mean(dim=1)
        
        # Apply classification head
        logits = self.classifier(pooled)
        return logits


class MultiLabelSequenceClassifier:
    """Multi-label classifier using a transformer for sequence inputs."""
    
    def __init__(self, 
                 embed_dim=256,
                 num_heads=4, 
                 num_layers=2,
                 hidden_dim=512,
                 dropout_rate=0.1,
                 pooling_strategy='mean',
                 max_seq_len=None, 
                 lr=0.0005,
                 batch_size=16, 
                 early_stopping_patience=10,
                 max_epochs=100,
                 random_seed=42,
                 device=None):
        """Initialize a multi-label sequence classifier using a transformer architecture.
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension in feed-forward networks
            dropout_rate: Dropout rate
            pooling_strategy: Strategy for pooling transformer outputs ('mean', 'first', or 'max')
            max_seq_len: Maximum sequence length (None for variable length)
            lr: Learning rate
            batch_size: Batch size for training
            early_stopping_patience: Number of epochs with no improvement before early stopping
            max_epochs: Maximum number of training epochs
            random_seed: Random seed for reproducibility
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.pooling_strategy = pooling_strategy
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Set random seed
        self.set_seed(random_seed)
        
        # Initialize other attributes
        self.model = None
        self.mlb = None  # MultiLabelBinarizer
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def _create_model(self, num_classes):
        """Create and return a transformer model."""
        model = SequenceTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            dropout=self.dropout_rate,
            max_seq_len=self.max_seq_len or 1000,
            pooling_strategy=self.pooling_strategy
        ).to(self.device)
        return model
    
    def _collate_fn(self, batch):
        """Custom collate function for handling variable-length sequences."""
        # Separate embeddings, labels, and lengths
        embeddings, labels, lengths = zip(*batch)
        
        # Get batch max length if not using a fixed max_seq_len
        if self.max_seq_len is None:
            max_len = max(lengths)
            # Pad all sequences to max length
            padded_embeddings = []
            for embed, length in zip(embeddings, lengths):
                if length < max_len:
                    pad = torch.zeros(max_len - length, embed.shape[1])
                    padded_embed = torch.cat([embed, pad], dim=0)
                else:
                    padded_embed = embed
                padded_embeddings.append(padded_embed)
            
            # Create padding mask (True where padded)
            padding_mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool)
            for i, length in enumerate(lengths):
                padding_mask[i, length:] = True
        else:
            # Use predetermined max_seq_len
            padded_embeddings = embeddings
            padding_mask = torch.zeros(len(embeddings), self.max_seq_len, dtype=torch.bool)
            for i, length in enumerate(lengths):
                if length < self.max_seq_len:
                    padding_mask[i, length:] = True
        
        # Stack into tensors
        embeddings_tensor = torch.stack(padded_embeddings)
        labels_tensor = torch.stack(labels)
        
        return embeddings_tensor, labels_tensor, padding_mask

    def fit(self, X, y, test_split=0.15, n_folds=5, show_progress=False):
        """Train the multi-label sequence model using k-fold cross-validation."""
        print(f"Using device: {self.device}")
        
        # Process multi-label data
        self.mlb = MultiLabelBinarizer()
        processed_labels = [label.split(';') for label in y]
        y_encoded = self.mlb.fit_transform(processed_labels)
        num_classes = len(self.mlb.classes_)
        
        print(f"Total samples: {len(X)}")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {self.mlb.classes_}")
        print(f"Using transformer with {self.num_layers} layers, {self.num_heads} heads")
        print(f"Pooling strategy: {self.pooling_strategy}")
        
        # Create dataset for all data
        dataset = SequenceBGCDataset(X, y_encoded, self.max_seq_len)
        
        # Split off test set first
        test_size = int(test_split * len(dataset))
        train_val_size = len(dataset) - test_size
        
        train_val_dataset, test_dataset = random_split(
            dataset, 
            [train_val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        # Prepare test loader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
        
        print(f"Train/Val samples: {len(train_val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Initialize K-Fold cross-validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        # Get indices from train_val_dataset
        train_val_indices = train_val_dataset.indices
        
        # Track best model across all folds
        best_val_loss = float('inf')
        best_model_state = None
        fold_histories = []
        fold_metrics = []
        
        print(f"Starting {n_folds}-fold cross-validation...")
        
        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_val_indices)))):
            print(f"\nFold {fold+1}/{n_folds}")
            
            # Map indices back to original dataset
            train_indices = [train_val_indices[i] for i in train_idx]
            val_indices = [train_val_indices[i] for i in val_idx]
            
            # Create fold-specific datasets
            train_dataset_fold = torch.utils.data.Subset(dataset, train_indices)
            val_dataset_fold = torch.utils.data.Subset(dataset, val_indices)
            
            train_loader = DataLoader(
                train_dataset_fold, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=self._collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset_fold, 
                batch_size=self.batch_size,
                collate_fn=self._collate_fn
            )
            
            print(f"  Train samples: {len(train_dataset_fold)}")
            print(f"  Validation samples: {len(val_dataset_fold)}")
            
            # Build model for this fold
            self.model = self._create_model(num_classes)
            
            # Define loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=False
            )
            
            # History for this fold
            fold_history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
            
            # Training loop for this fold
            fold_best_val_loss = float('inf')
            fold_best_model_state = None
            no_improve_count = 0
            
            for epoch in tqdm(range(self.max_epochs), 
                             desc=f"Training Fold {fold+1}/{n_folds}",
                             disable=not show_progress):
                # Train
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validate
                val_loss, val_f1 = self._validate(val_loader, criterion)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Print results
                if show_progress and (epoch+1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.max_epochs}, "
                         f"Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                
                # Save fold history
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['val_f1'].append(val_f1)
                
                # Check if this is the best model for this fold
                if val_loss < fold_best_val_loss:
                    fold_best_val_loss = val_loss
                    fold_best_model_state = copy.deepcopy(self.model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stopping for this fold
                if no_improve_count >= self.early_stopping_patience:
                    if show_progress:
                        print(f"  No improvement for {self.early_stopping_patience} epochs. Stopping training for fold {fold+1}.")
                    break
            
            # Load best model for this fold and evaluate
            self.model.load_state_dict(fold_best_model_state)
            fold_val_results = self._evaluate(val_loader)
            
            # Record metrics for this fold
            fold_metrics.append({
                'fold': fold+1,
                'val_loss': fold_best_val_loss,
                'val_accuracy': fold_val_results['accuracy'],
                'val_hamming_loss': fold_val_results['hamming_loss'],
                'val_auroc_micro': fold_val_results['auroc_micro'],
                'val_auroc_macro': fold_val_results['auroc_macro']
            })
            
            print(f"  Fold {fold+1} best validation loss: {fold_best_val_loss:.4f}")
            print(f"  Fold {fold+1} validation accuracy: {fold_val_results['accuracy']:.4f}")
            print(f"  Fold {fold+1} validation AUROC (micro): {fold_val_results['auroc_micro']:.4f}")
            
            # Store fold history
            fold_histories.append(fold_history)
            
            # Update best overall model if this fold's model is better
            if fold_best_val_loss < best_val_loss:
                best_val_loss = fold_best_val_loss
                best_model_state = fold_best_model_state
                print(f"  New best model from fold {fold+1}")
        
        # Print cross-validation summary
        print("\nCross-Validation Summary:")
        avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) 
                      for key in ['val_loss', 'val_accuracy', 'val_hamming_loss', 
                                 'val_auroc_micro', 'val_auroc_macro']}
        
        print(f"Average validation loss: {avg_metrics['val_loss']:.4f}")
        print(f"Average validation accuracy: {avg_metrics['val_accuracy']:.4f}")
        print(f"Average validation hamming loss: {avg_metrics['val_hamming_loss']:.4f}")
        print(f"Average validation AUROC (micro): {avg_metrics['val_auroc_micro']:.4f}")
        print(f"Average validation AUROC (macro): {avg_metrics['val_auroc_macro']:.4f}")
        
        # FIX: More robust history aggregation
        if fold_histories:
            # Find minimum length across all fold histories (ensure it's not 0)
            min_train_len = min(len(h['train_loss']) for h in fold_histories)
            min_val_len = min(len(h['val_loss']) for h in fold_histories)
            min_val_f1_len = min(len(h['val_f1']) for h in fold_histories)
            
            # Use the minimum length that's above 0
            min_len = max(1, min(min_train_len, min_val_len, min_val_f1_len))
            
            # Truncate to same length and then average
            self.history = {
                'train_loss': np.mean([[h['train_loss'][i] for i in range(min(min_len, len(h['train_loss'])))] 
                                      for h in fold_histories], axis=0).tolist(),
                'val_loss': np.mean([[h['val_loss'][i] for i in range(min(min_len, len(h['val_loss'])))]
                                    for h in fold_histories], axis=0).tolist(),
                'val_acc': np.mean([[h['val_f1'][i] for i in range(min(min_len, len(h['val_f1'])))]
                                   for h in fold_histories], axis=0).tolist()
            }
            
            # Ensure train_acc exists with proper length
            self.history['train_acc'] = [0.0] * len(self.history['train_loss'])
        else:
            # Fallback if no histories available
            self.history = {
                'train_loss': [0.0],
                'val_loss': [0.0],
                'train_acc': [0.0],
                'val_acc': [0.0]
            }
            print("Warning: No training history available for plotting.")
        
        # Load best model from all folds for final evaluation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Evaluate on test set if available
        if test_split > 0.0 and len(test_dataset) > 0:
            test_results = self._evaluate(test_loader)
            print(f"\nTest Results (Best Model):")
            print(f"Test exact match accuracy: {test_results['accuracy']:.4f}")
            print(f"Test hamming loss: {test_results['hamming_loss']:.4f}")
            print(f"Test AUROC (micro-avg): {test_results['auroc_micro']:.4f}")
            print(f"Test AUROC (macro-avg): {test_results['auroc_macro']:.4f}")
        
        return self
        
    def _train_epoch(self, loader, optimizer, criterion):
        """Train for one epoch with sequence data."""
        self.model.train()
        running_loss = 0.0
        total = 0
        
        for inputs, labels, padding_mask in loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()
            padding_mask = padding_mask.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs, src_key_padding_mask=padding_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_size
            total += batch_size
        
        epoch_loss = running_loss / total
        return epoch_loss
    
    def _validate(self, loader, criterion):
        """Validate sequence model."""
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, padding_mask in loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()
                padding_mask = padding_mask.to(self.device)
                
                outputs = self.model(inputs, src_key_padding_mask=padding_mask)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * batch_size
                total += batch_size
                
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Calculate metrics
        outputs = torch.cat(all_outputs, dim=0).sigmoid().numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Apply threshold to get binary predictions
        predictions = (outputs > 0.5).astype(int)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        epoch_loss = running_loss / total
        return epoch_loss, f1
    
    def _evaluate(self, loader):
        """Evaluate sequence model with detailed metrics."""
        self.model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, padding_mask in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                padding_mask = padding_mask.to(self.device)
                
                outputs = self.model(inputs, src_key_padding_mask=padding_mask)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Convert to numpy for evaluation
        outputs = torch.sigmoid(torch.cat(all_outputs, dim=0)).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Apply threshold to get binary predictions
        predictions = (outputs > 0.5).astype(int)
        
        # Calculate metrics - same as in original implementation
        from sklearn.metrics import (hamming_loss, accuracy_score, roc_auc_score,
                                    precision_score, recall_score, f1_score)
        
        # Same metrics calculation as in the original implementation
        # This follows exactly the same pattern as the original _evaluate method
        
        # Return results dictionary with the same structure as in the original
        # implementation for compatibility
        
        # Sample-wise accuracy (exact matches)
        accuracy = accuracy_score(labels, predictions)
        
        # Calculate AUROC scores
        try:
            auroc_micro = roc_auc_score(labels, outputs, average='micro')
            auroc_macro = roc_auc_score(labels, outputs, average='macro')
            auroc_weighted = roc_auc_score(labels, outputs, average='weighted')
        except Exception as e:
            print(f"Warning: Could not compute AUROC: {e}")
            auroc_micro = auroc_macro = auroc_weighted = 0.0
            
        return {
            'accuracy': accuracy,
            'hamming_loss': hamming_loss(labels, predictions),
            'auroc_micro': auroc_micro,
            'auroc_macro': auroc_macro,
            'auroc_weighted': auroc_weighted,
        }
    
    def predict(self, X):
        """Make predictions on new sequence data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Create dataset for prediction
        dataset = SequenceBGCDataset(X, np.zeros((len(X), len(self.mlb.classes_))), self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)
        
        # Get predictions
        self.model.eval()
        all_outputs = []
        
        with torch.no_grad():
            for inputs, _, padding_mask in loader:
                inputs = inputs.to(self.device)
                padding_mask = padding_mask.to(self.device)
                
                outputs = self.model(inputs, src_key_padding_mask=padding_mask)
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
        
        # Concatenate all outputs
        probabilities = np.vstack(all_outputs)
        predictions = (probabilities > 0.5).astype(int)
        
        # Convert to original class names
        pred_labels = []
        for pred in predictions:
            classes = [self.mlb.classes_[i] for i in range(len(pred)) if pred[i]]
            pred_labels.append(';'.join(sorted(classes)) if classes else '')
            
        return pred_labels
    
    def plot_history(self):
        """Plot training history."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_acc'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Validation F1 Score')
        
        plt.tight_layout()
        plt.show()


class BiLSTMNetwork(nn.Module):
    """BiLSTM-based encoder for multi-label classification of sequence data."""
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 pooling_strategy: str = 'mean'):
        """
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension in LSTM and feedforward networks
            num_classes: Number of output classes
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pooling_strategy: How to pool sequence representations ('mean', 'last', or 'max')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pooling_strategy = pooling_strategy
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
            lengths: Optional list of sequence lengths for packed sequence
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        if lengths is not None:
            # Create packed sequence for variable length inputs
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, _) = self.lstm(packed_x)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            # Process as regular tensor
            lstm_output, (hidden, _) = self.lstm(x)
        
        # Apply pooling strategy
        if self.pooling_strategy == 'last':
            # Use hidden state of last layer
            # For bidirectional, concatenate both directions
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            pooled = last_hidden
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence dimension
            pooled, _ = torch.max(lstm_output, dim=1)
        else:  # Default to mean pooling
            if lengths is not None:
                # Create mask based on actual lengths
                mask = torch.zeros(batch_size, seq_len, device=x.device)
                for i, length in enumerate(lengths):
                    mask[i, :length] = 1
                mask = mask.unsqueeze(2).expand_as(lstm_output)
                # Apply mask and calculate mean only over actual sequence
                pooled = (lstm_output * mask).sum(dim=1) / lengths.unsqueeze(1).to(x.device)
            else:
                pooled = lstm_output.mean(dim=1)
        
        # Apply classification head
        logits = self.classifier(pooled)
        return logits


class MultiLabelBiLSTMClassifier:
    """Multi-label classifier using a BiLSTM for sequence inputs."""
    
    def __init__(self, 
                 embed_dim=256,
                 hidden_dim=512,
                 num_layers=2,
                 dropout_rate=0.2,
                 pooling_strategy='mean',
                 max_seq_len=None, 
                 lr=1e-4,
                 batch_size=32, 
                 early_stopping_patience=15,
                 max_epochs=100,
                 random_seed=42,
                 device=None):
        """Initialize a multi-label sequence classifier using a BiLSTM architecture."""
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.pooling_strategy = pooling_strategy
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Set random seed
        self.set_seed(random_seed)
        
        # Initialize other attributes
        self.model = None
        self.mlb = None  # MultiLabelBinarizer
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def _create_model(self, num_classes):
        """Create and return a BiLSTM model."""
        model = BiLSTMNetwork(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            pooling_strategy=self.pooling_strategy
        ).to(self.device)
        return model
    
    def _collate_fn(self, batch):
        """Custom collate function for handling variable-length sequences."""
        # Separate embeddings, labels, and lengths
        embeddings, labels, lengths = zip(*batch)
        
        # Get batch max length if not using a fixed max_seq_len
        if self.max_seq_len is None:
            max_len = max(lengths)
            # Pad all sequences to max length
            padded_embeddings = []
            for embed, length in zip(embeddings, lengths):
                if length < max_len:
                    pad = torch.zeros(max_len - length, embed.shape[1])
                    padded_embed = torch.cat([embed, pad], dim=0)
                else:
                    padded_embed = embed
                padded_embeddings.append(padded_embed)
        else:
            # Use predetermined max_seq_len
            max_len = self.max_seq_len
            padded_embeddings = []
            adjusted_lengths = []  # Track actual lengths after truncation
            for embed, length in zip(embeddings, lengths):
                if length > max_len:
                    # Truncate
                    padded_embed = embed[:max_len]
                    adjusted_lengths.append(max_len)
                elif length < max_len:
                    # Pad
                    pad = torch.zeros(max_len - length, embed.shape[1])
                    padded_embed = torch.cat([embed, pad], dim=0)
                    adjusted_lengths.append(length)
                else:
                    padded_embed = embed
                    adjusted_lengths.append(length)
                padded_embeddings.append(padded_embed)
            
            # Use adjusted lengths instead of corrupted lengths
            lengths = adjusted_lengths
        
        # Stack into tensors
        embeddings_tensor = torch.stack(padded_embeddings)
        labels_tensor = torch.stack(labels)
        lengths_tensor = torch.tensor(lengths)
        
        return embeddings_tensor, labels_tensor, lengths_tensor
    
    def fit(self, X, y, test_split=0.15, n_folds=5, n_repeats=3, show_progress=False):
        """Train the multi-label sequence model using repeated k-fold cross-validation."""
        print(f"Using device: {self.device}")
        
        # Process multi-label data
        self.mlb = MultiLabelBinarizer()
        processed_labels = [label.split(';') for label in y]
        y_encoded = self.mlb.fit_transform(processed_labels)
        num_classes = len(self.mlb.classes_)
        
        print(f"Total samples: {len(X)}")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {self.mlb.classes_}")
        print(f"Using BiLSTM with {self.num_layers} layers")
        print(f"Pooling strategy: {self.pooling_strategy}")
        
        # Create dataset for all data
        dataset = SequenceBGCDataset(X, y_encoded, self.max_seq_len)
        
        # Split off test set first
        test_size = int(test_split * len(dataset))
        train_val_size = len(dataset) - test_size
        
        train_val_dataset, test_dataset = random_split(
            dataset, 
            [train_val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        # Prepare test loader
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size,
                collate_fn=self._collate_fn
            )
        else:
            test_loader = None
        
        print(f"Train/Val samples: {len(train_val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Special case: when n_folds=1, use all training data without cross-validation
        if n_folds <= 1:
            print("Using full training set without cross-validation (n_folds<=1)...")
            
            # Create data loaders for training
            train_loader = DataLoader(
                train_val_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=self._collate_fn
            )
            
            # Build model
            self.model = self._create_model(num_classes)
            
            # Define loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=False
            )
            
            # Training history
            history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
            
            # Training loop
            best_train_loss = float('inf')
            best_model_state = None
            no_improve_count = 0
            
            for epoch in tqdm(range(self.max_epochs), 
                             desc="Training",
                             disable=not show_progress):
                # Train
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                
                # Update learning rate
                scheduler.step(train_loss)
                
                # Print results
                if show_progress and (epoch+1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.max_epochs}, "
                         f"Train Loss: {train_loss:.4f}")
                
                # Save history
                history['train_loss'].append(train_loss)
                # No validation set, so use placeholder values
                history['val_loss'].append(0.0)
                history['val_f1'].append(0.0)
                
                # Check if this is the best model
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stopping
                if no_improve_count >= self.early_stopping_patience:
                    if show_progress:
                        print(f"  No improvement for {self.early_stopping_patience} epochs. Stopping training.")
                    break
            
            # Load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            
            self.history = {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'train_acc': [0.0] * len(history['train_loss']),
                'val_acc': history['val_f1']
            }
            
            # Evaluate on test set
            if test_loader:
                test_results = self._evaluate(test_loader)
                print(f"\nTest Results (Best Model):")
                print(f"Test exact match accuracy: {test_results['accuracy']:.4f}")
                print(f"Test hamming loss: {test_results['hamming_loss']:.4f}")
                print(f"Test AUROC (micro-avg): {test_results['auroc_micro']:.4f}")
                print(f"Test AUROC (macro-avg): {test_results['auroc_macro']:.4f}")
                print(f"Test AUROC (weighted-avg): {test_results['auroc_weighted']:.4f}")

            return self
        
        # Regular case: Perform repeated k-fold cross-validation
        print(f"Starting {n_repeats} repeats of {n_folds}-fold cross-validation...")
        
        # Get indices from train_val_dataset
        train_val_indices = train_val_dataset.indices
        
        # Track best model across all folds and repeats
        best_val_loss = float('inf')
        best_model_state = None
        fold_histories = []
        all_fold_metrics = []
        
        for repeat in range(n_repeats):
            print(f"\n----- Repeat {repeat + 1}/{n_repeats} -----")
            
            # Initialize K-Fold cross-validation for this repeat
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed + repeat)

            # Iterate through folds
            for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_val_indices)))):
                print(f"\nFold {fold+1}/{n_folds} (Repeat {repeat+1})")
                
                # Map indices back to original dataset
                train_indices = [train_val_indices[i] for i in train_idx]
                val_indices = [train_val_indices[i] for i in val_idx]
                
                # Create fold-specific datasets
                train_dataset_fold = torch.utils.data.Subset(dataset, train_indices)
                val_dataset_fold = torch.utils.data.Subset(dataset, val_indices)
                
                train_loader = DataLoader(
                    train_dataset_fold, 
                    batch_size=self.batch_size, 
                    shuffle=True,
                    collate_fn=self._collate_fn
                )
                
                val_loader = DataLoader(
                    val_dataset_fold, 
                    batch_size=self.batch_size,
                    collate_fn=self._collate_fn
                )
                
                print(f"  Train samples: {len(train_dataset_fold)}")
                print(f"  Validation samples: {len(val_dataset_fold)}")
                
                # Build model for this fold
                self.model = self._create_model(num_classes)
                
                # Define loss function and optimizer
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3, verbose=False
                )
                
                # History for this fold
                fold_history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
                
                # Training loop for this fold
                fold_best_val_loss = float('inf')
                fold_best_model_state = None
                no_improve_count = 0
                
                for epoch in tqdm(range(self.max_epochs), 
                                 desc=f"Training Fold {fold+1}/{n_folds} (Repeat {repeat+1})",
                                 disable=not show_progress):
                    # Train
                    train_loss = self._train_epoch(train_loader, optimizer, criterion)
                    
                    # Validate
                    val_loss, val_f1 = self._validate(val_loader, criterion)
                    
                    # Update learning rate
                    scheduler.step(val_loss)
                    
                    # Print results
                    if show_progress and (epoch+1) % 5 == 0:
                        print(f"  Epoch {epoch+1}/{self.max_epochs}, "
                             f"Train Loss: {train_loss:.4f}, "
                             f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                    
                    # Save fold history
                    fold_history['train_loss'].append(train_loss)
                    fold_history['val_loss'].append(val_loss)
                    fold_history['val_f1'].append(val_f1)
                    
                    # Check if this is the best model for this fold
                    if val_loss < fold_best_val_loss:
                        fold_best_val_loss = val_loss
                        fold_best_model_state = copy.deepcopy(self.model.state_dict())
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    # Early stopping for this fold
                    if no_improve_count >= self.early_stopping_patience:
                        if show_progress:
                            print(f"  No improvement for {self.early_stopping_patience} epochs. Stopping training for fold {fold+1} in repeat {repeat+1}.")
                        break
                
                # Load best model for this fold and evaluate
                self.model.load_state_dict(fold_best_model_state)
                fold_val_results = self._evaluate(val_loader)
                
                # Record metrics for this fold
                all_fold_metrics.append({
                    'fold': fold+1,
                    'repeat': repeat + 1,
                    'val_loss': fold_best_val_loss,
                    'val_accuracy': fold_val_results['accuracy'],
                    'val_hamming_loss': fold_val_results['hamming_loss'],
                    'val_auroc_micro': fold_val_results['auroc_micro'],
                    'val_auroc_macro': fold_val_results['auroc_macro']
                })
                
                print(f"  Fold {fold+1} (Repeat {repeat+1}) best validation loss: {fold_best_val_loss:.4f}")
                print(f"  Fold {fold+1} (Repeat {repeat+1}) validation accuracy: {fold_val_results['accuracy']:.4f}")
                print(f"  Fold {fold+1} (Repeat {repeat+1}) validation AUROC (micro): {fold_val_results['auroc_micro']:.4f}")
                
                # Store fold history
                fold_histories.append(fold_history)
                
                # Update best overall model if this fold's model is better
                if fold_best_val_loss < best_val_loss:
                    best_val_loss = fold_best_val_loss
                    best_model_state = fold_best_model_state
                    print(f"  New best model from fold {fold+1} of repeat {repeat+1}")
        
        # Print cross-validation summary and handle history
        print(f"\n{n_repeats}x{n_folds}-Fold Cross-Validation Summary:")
        if all_fold_metrics:
            avg_metrics = {key: np.mean([fold[key] for fold in all_fold_metrics]) 
                          for key in ['val_loss', 'val_accuracy', 'val_hamming_loss', 
                                     'val_auroc_micro', 'val_auroc_macro']}
            
            print(f"Average validation loss: {avg_metrics['val_loss']:.4f}")
            print(f"Average validation accuracy: {avg_metrics['val_accuracy']:.4f}")
            print(f"Average validation hamming loss: {avg_metrics['val_hamming_loss']:.4f}")
            print(f"Average validation AUROC (micro): {avg_metrics['val_auroc_micro']:.4f}")
            print(f"Average validation AUROC (macro): {avg_metrics['val_auroc_macro']:.4f}")
        
        # Handle history aggregation
        if fold_histories:
            min_train_len = min((len(h['train_loss']) for h in fold_histories), default=0)
            min_val_len = min((len(h['val_loss']) for h in fold_histories), default=0)
            min_val_f1_len = min((len(h['val_f1']) for h in fold_histories), default=0)
            
            min_len = max(1, min(min_train_len, min_val_len, min_val_f1_len))
            
            if min_len > 1:
                self.history = {
                    'train_loss': np.mean([h['train_loss'][:min_len] for h in fold_histories], axis=0).tolist(),
                    'val_loss': np.mean([h['val_loss'][:min_len] for h in fold_histories], axis=0).tolist(),
                    'val_acc': np.mean([h['val_f1'][:min_len] for h in fold_histories], axis=0).tolist()
                }
                self.history['train_acc'] = [0.0] * len(self.history['train_loss'])
            else:
                 self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Load best model from all folds for final evaluation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Evaluate on test set if available
        if test_loader:
            test_results = self._evaluate(test_loader)
            print(f"\nTest Results (Best Model):")
            print(f"Test exact match accuracy: {test_results['accuracy']:.4f}")
            print(f"Test hamming loss: {test_results['hamming_loss']:.4f}")
            print(f"Test AUROC (micro-avg): {test_results['auroc_micro']:.4f}")
            print(f"Test AUROC (macro-avg): {test_results['auroc_macro']:.4f}")
            print(f"Test AUROC (weighted-avg): {test_results['auroc_weighted']:.4f}")
        else:
            print("\nNo test set to evaluate on.")
        
        return self

    def _train_epoch(self, loader, optimizer, criterion):
        """Train for one epoch with sequence data."""
        self.model.train()
        running_loss = 0.0
        total = 0
        
        for inputs, labels, lengths in loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()
            lengths = lengths.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs, lengths)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_size
            total += batch_size
        
        epoch_loss = running_loss / total
        return epoch_loss
    
    def _validate(self, loader, criterion):
        """Validate sequence model."""
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, lengths in loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()
                lengths = lengths.to(self.device)
                
                outputs = self.model(inputs, lengths)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * batch_size
                total += batch_size
                
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Calculate metrics
        outputs = torch.cat(all_outputs, dim=0).sigmoid().numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Apply threshold to get binary predictions
        predictions = (outputs > 0.5).astype(int)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        epoch_loss = running_loss / total
        return epoch_loss, f1
    
    def _evaluate(self, loader):
        """Evaluate sequence model with detailed metrics."""
        self.model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, lengths in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(inputs, lengths)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Convert to numpy for evaluation
        outputs = torch.sigmoid(torch.cat(all_outputs, dim=0)).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Apply threshold to get binary predictions
        predictions = (outputs > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import (hamming_loss, accuracy_score, roc_auc_score)
        
        # Sample-wise accuracy (exact matches)
        accuracy = accuracy_score(labels, predictions)
        
        # Calculate AUROC scores
        try:
            auroc_micro = roc_auc_score(labels, outputs, average='micro')
            auroc_macro = roc_auc_score(labels, outputs, average='macro')
            auroc_weighted = roc_auc_score(labels, outputs, average='weighted')
        except Exception as e:
            print(f"Warning: Could not compute AUROC: {e}")
            auroc_micro = auroc_macro = auroc_weighted = 0.0
            
        return {
            'accuracy': accuracy,
            'hamming_loss': hamming_loss(labels, predictions),
            'auroc_micro': auroc_micro,
            'auroc_macro': auroc_macro,
            'auroc_weighted': auroc_weighted,
        }
    
    def predict(self, X):
        """Make predictions on new sequence data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Create dataset for prediction
        dataset = SequenceBGCDataset(X, np.zeros((len(X), len(self.mlb.classes_))), self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)
        
        # Get predictions
        self.model.eval()
        all_outputs = []
        
        with torch.no_grad():
            for inputs, _, lengths in loader:
                inputs = inputs.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(inputs, lengths)
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
        
        # Concatenate all outputs
        probabilities = np.vstack(all_outputs)
        predictions = (probabilities > 0.5).astype(int)
        
        # Convert to original class names
        pred_labels = []
        for pred in predictions:
            classes = [self.mlb.classes_[i] for i in range(len(pred)) if pred[i]]
            pred_labels.append(';'.join(sorted(classes)) if classes else '')
            
        return pred_labels
    
    def plot_history(self):
        """Plot training history."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_acc'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Validation F1 Score')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _batch_iter(items, batch_size):
        """Yield successive mini-batches from *items*."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def predict_proba(self, sequences):
        """
        Return ndarray [N, C] with class probabilities.
        Handles variable-length sequence embeddings AND flat vectors.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()
        device = self.device
        probs_all = []

        with torch.no_grad():
            for batch in self._batch_iter(sequences, self.batch_size):
                # to tensor list
                tensors = [torch.tensor(s, dtype=torch.float32) for s in batch]
                # lengths for BiLSTM
                lengths = torch.tensor(
                    [t.shape[0] if t.ndim == 2 else 1 for t in tensors],
                    dtype=torch.long, device=device
                )

                # pad to max-len in this batch
                if tensors[0].ndim == 2:                 # (seq_len, dim)
                    pad = torch.nn.utils.rnn.pad_sequence(
                        tensors, batch_first=True
                    )
                else:                                    # 1-D vectors
                    pad = torch.stack(tensors)

                pad = pad.to(device)

                # forward – BiLSTM expects (x, lengths)
                try:
                    logits = self.model(pad, lengths)
                except TypeError:
                    logits = self.model(pad)             # 如果模型不收 lengths

                probs_all.append(torch.sigmoid(logits).cpu().numpy())

        return np.vstack(probs_all)