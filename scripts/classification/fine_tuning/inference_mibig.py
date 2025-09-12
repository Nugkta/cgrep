"""
Inference script for fine-tuned BigCarp MIBiG classifier

This script loads a fine-tuned model and performs inference on BGC sequences,
predicting the 7 MIBiG classes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths for BigCarp imports
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')

from finetune_mibig_classification import BigCarpClassifier, MIBiGClassMapper, load_pretrained_bigcarp


class MIBiGInference:
    """Inference pipeline for MIBiG BGC classification"""
    
    def __init__(self, model_path: str, vocab_path: str, class_mapping_path: str):
        
        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location='cpu')
        self.class_mapper = self.checkpoint['class_mapper']
        self.args = self.checkpoint['args']
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_info = json.load(f)
        self.specials = vocab_info['specials']
        self.domains = vocab_info['domains']
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the fine-tuned model"""
        
        # Load BigCarp encoder
        vocab_size = len(self.domains) + len(self.specials)
        mask_idx = self.specials['#']
        bigcarp_model = load_pretrained_bigcarp(
            self.args.checkpoint, vocab_size, mask_idx
        )
        
        # Create classifier
        self.model = BigCarpClassifier(
            bigcarp_model,
            num_classes=len(self.class_mapper.mibig_classes),
            freeze_encoder=self.args.freeze_encoder,
            pooling=self.args.pooling
        )
        
        # Load fine-tuned weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def predict_sequence(self, domain_sequence: List[str], threshold: float = 0.5) -> Dict:
        """Predict classes for a single domain sequence"""
        
        # Tokenize sequence
        tokens = []
        for domain in domain_sequence:
            if domain in self.domains:
                tokens.append(self.domains[domain])
            else:
                tokens.append(self.domains.get('UNK', 3))
        
        if len(tokens) == 0:
            return {
                'predicted_classes': [],
                'probabilities': np.zeros(len(self.class_mapper.mibig_classes)),
                'confidence': 0.0
            }
        
        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        mask_idx = self.specials['#']
        input_mask = (tokens_tensor != mask_idx).float().unsqueeze(-1)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tokens_tensor, input_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # Get predictions
        predictions = (probabilities > threshold).astype(int)
        predicted_classes = [self.class_mapper.mibig_classes[i] 
                           for i, pred in enumerate(predictions) if pred == 1]
        
        # Confidence as max probability
        confidence = probabilities.max()
        
        return {
            'predicted_classes': predicted_classes,
            'probabilities': probabilities,
            'confidence': confidence,
            'all_classes': self.class_mapper.mibig_classes
        }
    
    def predict_dataframe(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Predict classes for a DataFrame with domain sequences"""
        
        results = []
        for _, row in df.iterrows():
            prediction = self.predict_sequence(row['domain_sequence'], threshold)
            
            result = {
                'bgc_id': row.get('bgc_id', 'unknown'),
                'predicted_classes': ';'.join(prediction['predicted_classes']),
                'confidence': prediction['confidence']
            }
            
            # Add individual class probabilities
            for i, class_name in enumerate(prediction['all_classes']):
                result[f'prob_{class_name}'] = prediction['probabilities'][i]
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_on_test_set(self, test_df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """Evaluate model on test set with ground truth labels"""
        
        all_predictions = []
        all_labels = []
        
        for _, row in test_df.iterrows():
            # Get prediction
            prediction = self.predict_sequence(row['domain_sequence'], threshold)
            pred_vector = (prediction['probabilities'] > threshold).astype(int)
            
            # Get ground truth
            true_classes = [cls.strip() for cls in str(row['product_class']).split(';')]
            true_vector = self.class_mapper.encode_classes(true_classes)
            
            all_predictions.append(pred_vector)
            all_labels.append(true_vector)
        
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        # Compute metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        micro_f1 = f1_score(all_labels, all_predictions, average='micro')
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        micro_precision = precision_score(all_labels, all_predictions, average='micro')
        micro_recall = recall_score(all_labels, all_predictions, average='micro')
        
        # Per-class metrics
        report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_mapper.mibig_classes,
            output_dict=True
        )
        
        return {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels
        }


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot confusion matrix for each class"""
    
    # Compute multilabel confusion matrix
    cm = confusion_matrix(y_true.ravel(), y_pred.ravel())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{output_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class confusion matrices
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            cm_class = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(cm_class, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{class_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
    
    # Hide the last subplot if not needed
    if len(class_names) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/per_class_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(predictions_df, output_path):
    """Plot predicted class distribution"""
    
    # Count predicted classes
    all_predicted = []
    for pred_classes in predictions_df['predicted_classes']:
        if pred_classes:  # Not empty
            all_predicted.extend(pred_classes.split(';'))
    
    if all_predicted:
        class_counts = pd.Series(all_predicted).value_counts()
        
        plt.figure(figsize=(10, 6))
        class_counts.plot(kind='bar')
        plt.title('Distribution of Predicted Classes')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_path}/predicted_class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--input_data', type=str, required=True,
                       help='Path to input data (pickle or CSV)')
    parser.add_argument('--vocab_path', type=str,
                       default='/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json')
    parser.add_argument('--class_mapping', type=str,
                       default='/home/u5bb/han00.u5bb/workspace/tg_learn/data/raw/bgc_class_mapping.json')
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on test set (requires ground truth labels)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize inference pipeline
    print("Loading model...")
    inference = MIBiGInference(args.model_path, args.vocab_path, args.class_mapping)
    
    # Load input data
    print("Loading input data...")
    if args.input_data.endswith('.pkl'):
        input_df = pd.read_pickle(args.input_data)
    else:
        input_df = pd.read_csv(args.input_data)
    
    print(f"Loaded {len(input_df)} sequences")
    
    # Make predictions
    print("Making predictions...")
    predictions_df = inference.predict_dataframe(input_df, args.threshold)
    
    # Save predictions
    predictions_df.to_csv(f"{args.output_dir}/predictions.csv", index=False)
    print(f"Predictions saved to {args.output_dir}/predictions.csv")
    
    # Plot class distribution
    plot_class_distribution(predictions_df, args.output_dir)
    
    # Evaluate if ground truth is available
    if args.evaluate and 'product_class' in input_df.columns:
        print("Evaluating model...")
        eval_results = inference.evaluate_on_test_set(input_df, args.threshold)
        
        print(f"\\nEvaluation Results:")
        print(f"Micro-F1: {eval_results['micro_f1']:.4f}")
        print(f"Macro-F1: {eval_results['macro_f1']:.4f}")
        print(f"Micro-Precision: {eval_results['micro_precision']:.4f}")
        print(f"Micro-Recall: {eval_results['micro_recall']:.4f}")
        
        # Save detailed results
        with open(f"{args.output_dir}/evaluation_results.json", 'w') as f:
            json.dump({k: v for k, v in eval_results.items() 
                      if k not in ['predictions', 'labels']}, f, indent=2, default=str)
        
        # Plot confusion matrices
        plot_confusion_matrix(
            eval_results['labels'], eval_results['predictions'],
            inference.class_mapper.mibig_classes, args.output_dir
        )
        
        print(f"Evaluation results saved to {args.output_dir}")
    
    print("Inference completed!")


if __name__ == '__main__':
    main()