#!/usr/bin/env python
"""
Quick Architecture Test
=======================
Test just one fold with minimal data to verify the architecture works.
"""

import sys
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

import torch
import numpy as np
from bigcarp_architecture_comparison import (
    load_mibig3_data, prepare_single_label_data, create_stratified_splits,
    train_and_evaluate_fold
)
import json

def quick_test():
    print("🚀 Quick Architecture Test")
    print("=" * 40)
    
    # Load minimal data
    print("📂 Loading data...")
    data = load_mibig3_data()
    if data is None:
        print("❌ Data loading failed")
        return
    
    # Take a very small sample for quick testing
    small_data = data.head(100).copy()  # Just 100 samples
    single_data, label_encoder = prepare_single_label_data(small_data)
    
    print(f"   Using {len(single_data)} samples for quick test")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Create simple train/test split (not full CV)
    np.random.seed(42)
    indices = np.arange(len(single_data))
    np.random.shuffle(indices)
    
    split_point = int(0.8 * len(single_data))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    
    train_data = single_data.iloc[train_idx].copy()
    test_data = single_data.iloc[test_idx].copy()
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Load vocabulary
    print("📚 Loading vocabulary...")
    vocab_path = "/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json"
    with open(vocab_path, 'r') as f:
        vocab_info = json.load(f)
    
    specials = vocab_info['specials']
    domains = vocab_info['domains']
    
    # Create simple args object
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.epochs = 1
            self.lr = 1e-4
            self.patience = 1
            self.freeze_encoder = True  # Freeze to make training faster
    
    args = Args()
    
    # Test both architectures
    model_path = "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt"
    
    print("\n🔬 Testing MEAN pooling architecture...")
    try:
        result_mean = train_and_evaluate_fold(
            train_data, test_data, specials, domains, label_encoder,
            model_path, 0, args, pooling_strategy='mean'
        )
        print(f"✅ Mean pooling - Accuracy: {result_mean['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Mean pooling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🔬 Testing CLASS TOKEN architecture...")
    try:
        result_cls = train_and_evaluate_fold(
            train_data, test_data, specials, domains, label_encoder,
            model_path, 0, args, pooling_strategy='cls'
        )
        print(f"✅ Class token - Accuracy: {result_cls['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Class token failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    print(f"   Mean Pooling:  Accuracy = {result_mean['accuracy']:.4f}, F1 = {result_mean['f1_macro']:.4f}")
    print(f"   Class Token:   Accuracy = {result_cls['accuracy']:.4f}, F1 = {result_cls['f1_macro']:.4f}")
    
    if result_mean['accuracy'] > result_cls['accuracy']:
        print("🏆 Mean pooling performed better!")
    elif result_cls['accuracy'] > result_mean['accuracy']:
        print("🏆 Class token performed better!")
    else:
        print("🤝 Both performed equally!")
    
    print("\n✅ Quick test completed successfully!")

if __name__ == "__main__":
    quick_test()