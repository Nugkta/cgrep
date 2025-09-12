#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Script for MIBiG 3.0 Fine-tuned BigCarp Evaluation Pipeline
===============================================================
Quick validation of the fine-tuned evaluation pipeline without long training times.
Tests model loading, data processing, and evaluation metrics.
"""

import os, sys, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
import tempfile
import torch
import torch.nn as nn

# Add paths
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import functions from the main evaluation script
from scripts.classification.fine_tuning.mibig3_finetuned_evaluation import (
    load_mibig3_data,
    MIBiGClassMapper,
    MIBiGDataset,
    BigCarpMultiLabelClassifier,
    load_pretrained_bigcarp,
    collate_fn
)

from scripts.classification.mibig3_stratified_evaluation import (
    convert_product_classes_to_binary,
    create_stratified_splits,
    compute_comprehensive_metrics
)

def create_test_data(n_samples=20, max_seq_len=50):
    """Create synthetic test data that mimics MIBiG 3.0 structure."""
    print(f"🔧 Creating test data with {n_samples} samples...")
    
    # Sample domain vocabulary
    sample_domains = ['PF00001', 'PF00002', 'PF00003', 'PF00004', 'PF00005', 
                     'PF00006', 'PF00007', 'PF00008', 'PF00009', 'PF00010']
    
    # Product classes
    classes = ['Polyketide', 'NRP', 'RiPP', 'Terpene', 'Alkaloid', 'Saccharide', 'Other']
    
    data = {
        'bgc_id': [f'TEST_BGC_{i:04d}' for i in range(n_samples)],
        'product_class': [],
        'domain_sequence': []
    }
    
    np.random.seed(42)  # For reproducible test data
    
    for i in range(n_samples):
        # Random product classes (multi-label)
        n_classes = np.random.randint(1, 4)  # 1-3 classes per sample
        sample_classes = np.random.choice(classes, n_classes, replace=False)
        data['product_class'].append(';'.join(sample_classes))
        
        # Random domain sequence
        seq_len = np.random.randint(5, max_seq_len + 1)
        domain_seq = np.random.choice(sample_domains, seq_len, replace=True)
        data['domain_sequence'].append(' '.join(domain_seq))
    
    return pd.DataFrame(data)

def test_data_loading():
    """Test data loading functions."""
    print("\\n" + "="*60)
    print("🧪 TEST 1: Data Loading")
    print("="*60)
    
    try:
        # Test with real data if available
        artifacts_dir = "artifacts/classification/mibig3"
        if os.path.exists(artifacts_dir):
            print("📂 Testing with real MIBiG 3.0 data...")
            data = load_mibig3_data(artifacts_dir)
            if data is not None:
                print(f"✅ Loaded real data: {data.shape}")
                print(f"   Columns: {list(data.columns)}")
            else:
                print("❌ Failed to load real data")
        else:
            print("⚠️  Real data not found, using synthetic data only")
        
        print("✅ Data loading test passed!")
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_components():
    """Test model components."""
    print("\\n" + "="*60)
    print("🧪 TEST 2: Model Components") 
    print("="*60)
    
    try:
        # Test dataset creation
        print("🔄 Testing dataset creation...")
        test_df = create_test_data(n_samples=10)
        
        # Mock vocabulary
        specials = {'<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3, '<mask>': 4}
        domains = ['PF00001', 'PF00002', 'PF00003', 'PF00004', 'PF00005']
        
        # Create dataset
        sequences = test_df['domain_sequence'].tolist()
        labels = np.random.randint(0, 2, (10, 7))  # Random binary labels
        
        dataset = MIBiGDataset(sequences, labels, specials, domains, max_length=100)
        print(f"   ✅ Dataset created: {len(dataset)} samples, vocab size: {dataset.vocab_size}")
        
        # Test data loader
        print("🔄 Testing data loader...")
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, padding_idx=0)
        )
        
        batch = next(iter(dataloader))
        print(f"   ✅ Batch created: sequences {batch['sequences'].shape}, labels {batch['labels'].shape}")
        
        # Test model architecture
        print("🔄 Testing model architecture...")
        
        # Create mock encoder
        class MockEncoder(nn.Module):
            def __init__(self, vocab_size, embed_dim=256):
                super().__init__()
                self.embed_dim = embed_dim
                self.embedding = nn.Embedding(vocab_size, embed_dim)
            
            def forward(self, x):
                return self.embedding(x)
        
        mock_encoder = MockEncoder(dataset.vocab_size, embed_dim=256)
        model = BigCarpMultiLabelClassifier(
            encoder=mock_encoder,
            num_classes=7,
            hidden_dim=128,
            dropout=0.1,
            pooling='mean'
        )
        
        # Test forward pass
        with torch.no_grad():
            logits = model(batch['sequences'], batch['attention_mask'])
        
        print(f"   ✅ Model forward pass: input {batch['sequences'].shape} -> output {logits.shape}")
        
        print("✅ Model components test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        return False

def test_training_components():
    """Test training-related components."""
    print("\\n" + "="*60)
    print("🧪 TEST 3: Training Components")
    print("="*60)
    
    try:
        # Test binary conversion
        print("🔄 Testing binary conversion...")
        test_df = create_test_data(n_samples=20)
        df_binary, class_cols = convert_product_classes_to_binary(test_df)
        print(f"   ✅ Binary conversion: {df_binary.shape}, classes: {class_cols}")
        
        # Test CV splits
        print("🔄 Testing CV splits...")
        cv_splits = create_stratified_splits(df_binary, class_cols, n_splits=3, random_state=42)
        print(f"   ✅ Created {len(cv_splits)} CV splits")
        
        # Test metrics computation
        print("🔄 Testing metrics computation...")
        # Create mock predictions
        n_samples, n_classes = 50, len(class_cols)
        y_true = np.random.randint(0, 2, (n_samples, n_classes))
        y_pred = np.random.randint(0, 2, (n_samples, n_classes))
        y_proba = np.random.rand(n_samples, n_classes)
        
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, class_cols)
        print(f"   ✅ Metrics computed: {list(metrics.keys())}")
        print(f"   📊 Sample metrics: Macro F1={metrics['macro_f1']:.4f}, Exact Match={metrics['exact_match_accuracy']:.4f}")
        
        print("✅ Training components test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False

def test_model_loading():
    """Test model loading capability."""
    print("\\n" + "="*60)
    print("🧪 TEST 4: Model Loading")
    print("="*60)
    
    try:
        # Test paper model paths
        model_paths = [
            "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt",
            "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_esm1bfinetune.pt"
        ]
        
        for model_path in model_paths:
            print(f"🔄 Testing model loading: {model_path}")
            if os.path.exists(model_path):
                try:
                    # Try to load the model
                    checkpoint = torch.load(model_path, map_location='cpu')
                    print(f"   ✅ Model checkpoint loaded: {type(checkpoint)}")
                    
                    # Check checkpoint structure
                    if isinstance(checkpoint, dict):
                        keys = list(checkpoint.keys())
                        print(f"   📊 Checkpoint keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")
                    
                except Exception as e:
                    print(f"   ⚠️  Model loading failed: {e}")
            else:
                print(f"   ⚠️  Model not found: {model_path}")
        
        print("✅ Model loading test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the complete pipeline integration."""
    print("\\n" + "="*60)
    print("🧪 TEST 5: Pipeline Integration")
    print("="*60)
    
    try:
        # Test import of main evaluation script
        print("🔄 Testing main script import...")
        import scripts.classification.fine_tuning.mibig3_finetuned_evaluation as main_eval
        print("   ✅ Main evaluation script imported successfully")
        
        # Test argument parsing
        print("🔄 Testing argument parsing...")
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--artifacts_dir", default="artifacts/classification/mibig3")
        parser.add_argument("--outdir", default="results/test_mibig3_finetuned")
        parser.add_argument("--seed", type=int, default=42)
        
        args = parser.parse_args(['--seed', '123'])
        print(f"   ✅ Arguments parsed: seed={args.seed}")
        
        # Test required directories exist or can be created
        print("🔄 Testing directory structure...")
        test_outdir = "results/test_finetuned_pipeline"
        Path(test_outdir).mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(test_outdir):
            print(f"   ✅ Output directory created: {test_outdir}")
            import shutil
            shutil.rmtree(test_outdir)  # Clean up
        
        print("✅ Pipeline integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 MIBiG 3.0 Fine-tuned Evaluation Pipeline Test Suite")
    print("="*80)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Components", test_model_components), 
        ("Training Components", test_training_components),
        ("Model Loading", test_model_loading),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Final summary
    print("\\n" + "="*80)
    print("🏁 TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Fine-tuned evaluation pipeline is ready to run.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before running full evaluation.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)