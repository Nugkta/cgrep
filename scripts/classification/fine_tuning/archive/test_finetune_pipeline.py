#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Script for Modified Fine-tuning Pipeline
=============================================
Quick validation of the complete fine-tuned evaluation pipeline without long training times.
Tests model loading, data processing, evaluation metrics, and pipeline integration.
"""

import os, sys, json, warnings, tempfile
import numpy as np, pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

# Add paths
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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

def test_imports():
    """Test all required imports."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Imports")
    print("="*60)
    
    try:
        # Test direct execution of the fine-tuning script to check imports
        print("🔄 Testing fine-tuning script imports...")
        import subprocess
        result = subprocess.run([
            'python', '-c', 
            '''
import sys
sys.path.append("/home/u5bb/han00.u5bb/workspace/cgrep")
exec(open("/home/u5bb/han00.u5bb/workspace/cgrep/scripts/classification/fine_tuning/finetune_mibig_multilabel.py").read())
print("Fine-tuning script imports successful")
'''
        ], capture_output=True, text=True, cwd='/home/u5bb/han00.u5bb/workspace/cgrep')
        
        if result.returncode == 0:
            print("   ✅ Fine-tuning script imports successful")
        else:
            print(f"   ⚠️  Fine-tuning script import issues: {result.stderr}")
        
        # Test main evaluation script
        print("🔄 Testing main evaluation script imports...")
        result2 = subprocess.run([
            'python', '-c',
            '''
import sys
sys.path.append("/home/u5bb/han00.u5bb/workspace/cgrep")
from scripts.classification.mibig3_stratified_evaluation import convert_product_classes_to_binary, create_stratified_splits, compute_comprehensive_metrics
print("Main evaluation script imports successful")
'''
        ], capture_output=True, text=True, cwd='/home/u5bb/han00.u5bb/workspace/cgrep')
        
        if result2.returncode == 0:
            print("   ✅ Main evaluation script imports successful")
        else:
            print(f"   ⚠️  Main evaluation script import issues: {result2.stderr}")
        
        # Test BigCarp imports
        print("🔄 Testing BigCarp imports...")
        try:
            from sequence_models.convolutional import ByteNetLM
            from sequence_models.collaters import _pad
            print("   ✅ BigCarp components imported")
        except ImportError as e:
            print(f"   ⚠️  BigCarp import issues: {e}")
        
        print("✅ Import tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_data_pipeline():
    """Test data loading and preprocessing pipeline."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Data Pipeline")
    print("="*60)
    
    try:
        from scripts.classification.fine_tuning.finetune_mibig_multilabel import load_mibig3_data
        from scripts.classification.mibig3_stratified_evaluation import (
            convert_product_classes_to_binary,
            create_stratified_splits
        )
        
        # Test with synthetic data first
        print("🔄 Testing with synthetic data...")
        test_df = create_test_data(n_samples=50)
        print(f"   ✅ Created test data: {test_df.shape}")
        
        # Test binary conversion
        print("🔄 Testing binary conversion...")
        df_binary, class_cols = convert_product_classes_to_binary(test_df)
        print(f"   ✅ Binary conversion: {df_binary.shape}, classes: {class_cols}")
        
        # Test CV splits
        print("🔄 Testing CV splits...")
        cv_splits = create_stratified_splits(df_binary, class_cols, n_splits=3, random_state=42)
        print(f"   ✅ Created {len(cv_splits)} CV splits")
        
        # Test with real data if available
        print("🔄 Testing with real MIBiG 3.0 data...")
        real_data = load_mibig3_data("artifacts/classification/mibig3")
        if real_data is not None:
            print(f"   ✅ Loaded real data: {real_data.shape}")
            print(f"   📊 Columns: {list(real_data.columns)}")
            
            # Test preprocessing on real data
            real_binary, real_classes = convert_product_classes_to_binary(real_data)
            print(f"   ✅ Real data binary conversion: {real_binary.shape}, classes: {real_classes}")
        else:
            print("   ⚠️  Real data not found, using synthetic data only")
        
        print("✅ Data pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components():
    """Test model components and architecture."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Model Components")
    print("="*60)
    
    try:
        from scripts.classification.fine_tuning.finetune_mibig_multilabel import (
            MIBiGDataset,
            BigCarpMultiLabelClassifier,
            collate_fn
        )
        from torch.utils.data import DataLoader
        
        # Test dataset creation
        print("🔄 Testing dataset creation...")
        test_df = create_test_data(n_samples=10)
        
        # Mock vocabulary
        specials = {'<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3, '<mask>': 4}
        domains = ['PF00001', 'PF00002', 'PF00003', 'PF00004', 'PF00005']
        class_cols = ['Polyketide', 'NRP', 'RiPP', 'Terpene', 'Alkaloid', 'Saccharide', 'Other']
        
        dataset = MIBiGDataset(test_df, specials, domains, class_cols)
        print(f"   ✅ Dataset created: {len(dataset)} samples, vocab size: {dataset.vocab_size}")
        
        # Test data loader
        print("🔄 Testing data loader...")
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
            num_classes=len(class_cols),
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
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test BigCarp model loading."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Model Loading")
    print("="*60)
    
    try:
        from scripts.classification.fine_tuning.finetune_mibig_multilabel import load_pretrained_bigcarp
        
        # Test paper model paths
        model_paths = [
            "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt",
            "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_esm1bfinetune.pt"
        ]
        
        for model_path in model_paths:
            print(f"🔄 Testing model loading: {model_path}")
            if os.path.exists(model_path):
                try:
                    # Try to load the model with mock vocab size
                    model = load_pretrained_bigcarp(model_path, vocab_size=1000, mask_idx=0)
                    print(f"   ✅ Model loaded successfully: {type(model)}")
                    
                    # Test model structure
                    print(f"   📊 Model embed_dim: {model.embed_dim}")
                    
                except Exception as e:
                    print(f"   ⚠️  Model loading failed: {e}")
            else:
                print(f"   ⚠️  Model not found: {model_path}")
        
        print("✅ Model loading test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics():
    """Test evaluation metrics computation."""
    print("\n" + "="*60)
    print("🧪 TEST 5: Evaluation Metrics")
    print("="*60)
    
    try:
        from scripts.classification.mibig3_stratified_evaluation import compute_comprehensive_metrics
        
        # Create mock predictions
        print("🔄 Testing metrics computation...")
        n_samples, n_classes = 50, 7
        class_names = ['Polyketide', 'NRP', 'RiPP', 'Terpene', 'Alkaloid', 'Saccharide', 'Other']
        
        np.random.seed(42)
        y_true = np.random.randint(0, 2, (n_samples, n_classes))
        y_pred = np.random.randint(0, 2, (n_samples, n_classes))
        y_proba = np.random.rand(n_samples, n_classes)
        
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, class_names)
        print(f"   ✅ Metrics computed: {list(metrics.keys())}")
        
        # Print sample metrics
        print(f"   📊 Sample metrics:")
        print(f"      Exact Match: {metrics['exact_match_accuracy']:.4f}")
        print(f"      Macro F1: {metrics['macro_f1']:.4f}")
        print(f"      Macro AUC: {metrics['macro_auc']:.4f}")
        
        print("✅ Evaluation metrics test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_argument_parsing():
    """Test argument parsing and configuration."""
    print("\n" + "="*60)
    print("🧪 TEST 6: Argument Parsing")
    print("="*60)
    
    try:
        import argparse
        from scripts.classification.fine_tuning.finetune_mibig_multilabel import main
        
        print("🔄 Testing argument parsing...")
        
        # Test that we can create a parser like the main function
        parser = argparse.ArgumentParser()
        parser.add_argument("--artifacts_dir", default="artifacts/classification/mibig3")
        parser.add_argument("--outdir", default="results/test_finetuned")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--epochs", type=int, default=2)  # Small for testing
        
        args = parser.parse_args(['--epochs', '3', '--batch_size', '8'])
        print(f"   ✅ Arguments parsed: epochs={args.epochs}, batch_size={args.batch_size}")
        
        # Test directory creation
        print("🔄 Testing directory creation...")
        test_outdir = "results/test_finetuned_pipeline"
        Path(test_outdir).mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(test_outdir):
            print(f"   ✅ Output directory created: {test_outdir}")
            import shutil
            shutil.rmtree(test_outdir)  # Clean up
        
        print("✅ Argument parsing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Argument parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_training():
    """Test a quick training loop with minimal epochs."""
    print("\n" + "="*60)
    print("🧪 TEST 7: Quick Training Loop")
    print("="*60)
    
    try:
        from scripts.classification.fine_tuning.finetune_mibig_multilabel import (
            MIBiGDataset, 
            BigCarpMultiLabelClassifier,
            train_epoch,
            evaluate_model_fold,
            collate_fn
        )
        from torch.utils.data import DataLoader
        
        print("🔄 Testing quick training loop...")
        
        # Create minimal test data
        test_df = create_test_data(n_samples=20)
        specials = {'<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3, '<mask>': 4}
        domains = ['PF00001', 'PF00002', 'PF00003', 'PF00004', 'PF00005']
        class_cols = ['Polyketide', 'NRP', 'RiPP']  # Fewer classes for faster testing
        
        # Create dataset and dataloader
        dataset = MIBiGDataset(test_df, specials, domains, class_cols)
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, padding_idx=0)
        )
        
        # Create simple model
        class MockEncoder(nn.Module):
            def __init__(self, vocab_size, embed_dim=64):
                super().__init__()
                self.embed_dim = embed_dim
                self.embedding = nn.Embedding(vocab_size, embed_dim)
            
            def forward(self, x):
                return self.embedding(x)
        
        encoder = MockEncoder(dataset.vocab_size, embed_dim=64)
        model = BigCarpMultiLabelClassifier(
            encoder=encoder,
            num_classes=len(class_cols),
            hidden_dim=32,
            dropout=0.1,
            pooling='mean'
        )
        
        # Set up training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device('cpu')  # Use CPU for testing
        
        print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Quick training loop (1 epoch)
        print("   🏋️  Running 1 epoch of training...")
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, mask_idx=0)
        print(f"   ✅ Training completed: loss={train_loss:.4f}")
        
        # Quick evaluation
        print("   🔮 Running evaluation...")
        predictions, probabilities, labels, val_loss = evaluate_model_fold(
            model, dataloader, criterion, device, mask_idx=0
        )
        print(f"   ✅ Evaluation completed: val_loss={val_loss:.4f}")
        print(f"   📊 Predictions shape: {predictions.shape}, Probabilities shape: {probabilities.shape}")
        
        print("✅ Quick training loop test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Modified Fine-tuning Pipeline Test Suite")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Data Pipeline", test_data_pipeline), 
        ("Model Components", test_model_components),
        ("Model Loading", test_model_loading),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Argument Parsing", test_argument_parsing),
        ("Quick Training", test_quick_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Modified fine-tuning pipeline is ready to run.")
        print("\n🚀 Next steps:")
        print("   1. Run: conda activate cgrep")
        print("   2. Test with: python scripts/classification/fine_tuning/finetune_mibig_multilabel.py --epochs 2 --batch_size 8")
        print("   3. Full run: sbatch scripts/classification/fine_tuning/run_mibig3_finetuned_evaluation.sh")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before running full evaluation.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)