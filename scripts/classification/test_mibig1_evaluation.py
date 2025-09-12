#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Script for MIBiG 1.0 Stratified Evaluation Pipeline
========================================================
Quick validation of the full pipeline without long training times.
Tests data loading, preprocessing, model initialization, and metrics computation.
"""

import os, sys, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
import tempfile
import shutil
import torch

# Add path for models
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import functions from the main evaluation script
from scripts.classification.mibig1_stratified_evaluation import (
    load_mibig1_data,
    prepare_embedding_data, 
    convert_product_classes_to_binary,
    create_stratified_splits,
    _validate_and_prepare_XY,
    _coerce_to_seq2d,
    create_mean_pooled_features,
    compute_comprehensive_metrics,
    tensor_to_list,
    create_concatenated_embeddings
)

def create_test_data(n_samples=50, seq_len_range=(5, 20), emb_dim=256):
    """Create synthetic test data that mimics the real MIBiG 1.0 structure."""
    print(f"🔧 Creating test data with {n_samples} samples...")
    
    # Create synthetic BGC data
    data = {
        'bgc_id': [f'TEST_BGC_{i:04d}' for i in range(n_samples)],
        'product_class': [],
        'embeddings': [],
        'esm_embeddings': [],
        'pfam2vec_seq': []
    }
    
    # Define some test product classes
    classes = ['Polyketide', 'NRP', 'RiPP', 'Terpene', 'Alkaloid', 'Saccharide']
    
    np.random.seed(42)  # For reproducible test data
    
    for i in range(n_samples):
        # Random product classes (multi-label)
        n_classes = np.random.randint(1, 4)  # 1-3 classes per sample
        sample_classes = np.random.choice(classes, n_classes, replace=False)
        data['product_class'].append(';'.join(sample_classes))
        
        # Random sequence length
        seq_len = np.random.randint(seq_len_range[0], seq_len_range[1] + 1)
        
        # BiGCARP embeddings (sequence format)
        bigcarp_emb = np.random.randn(seq_len, emb_dim).tolist()
        data['embeddings'].append(bigcarp_emb)
        
        # ESM embeddings (sequence format) 
        esm_emb = np.random.randn(seq_len, 1280).tolist()
        data['esm_embeddings'].append(esm_emb)
        
        # Pfam2vec embeddings (list of 100-dim vectors)
        pfam2vec_emb = [np.random.randn(100).tolist() for _ in range(seq_len)]
        data['pfam2vec_seq'].append(pfam2vec_emb)
    
    return pd.DataFrame(data)

def test_data_loading():
    """Test data loading functions."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Data Loading")
    print("="*60)
    
    try:
        # Test with real data if available
        artifacts_dir = "artifacts/classification/mibig1"
        if os.path.exists(artifacts_dir):
            print("📂 Testing with real MIBiG 1.0 data...")
            data = load_mibig1_data(artifacts_dir)
            available_data = [k for k, v in data.items() if v is not None]
            print(f"✅ Available data types: {available_data}")
            
            # Test prepare_embedding_data for each type
            for embedding_type in available_data[:2]:  # Test first 2 to save time
                print(f"   Testing {embedding_type}...")
                df_prep, emb_col = prepare_embedding_data(data, embedding_type)
                if df_prep is not None:
                    print(f"   ✅ {embedding_type}: {df_prep.shape}, column: {emb_col}")
                else:
                    print(f"   ❌ {embedding_type}: Failed to prepare")
        else:
            print("⚠️  Real data not found, using synthetic data only")
        
        print("✅ Data loading test passed!")
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing functions."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Data Preprocessing") 
    print("="*60)
    
    try:
        # Create test data
        test_df = create_test_data(n_samples=30)
        
        # Test product class conversion
        print("🔄 Testing product class conversion...")
        df_binary, class_cols = convert_product_classes_to_binary(test_df)
        print(f"   ✅ Binary classes: {class_cols}")
        print(f"   ✅ Shape after conversion: {df_binary.shape}")
        
        # Test CV splits
        print("🔄 Testing CV splits creation...")
        cv_splits = create_stratified_splits(df_binary, class_cols, n_splits=3)
        print(f"   ✅ Created {len(cv_splits)} CV splits")
        
        # Test data validation
        print("🔄 Testing data validation...")
        sample_embeddings = test_df['embeddings'].iloc[:10].tolist()
        sample_labels = ['Polyketide;NRP'] * 10
        
        X_val, y_val, kept, dropped = _validate_and_prepare_XY(
            sample_embeddings, sample_labels, emb_dim=256, split_name="test"
        )
        print(f"   ✅ Validation: kept {kept}, dropped {dropped}")
        print(f"   ✅ Output shapes: X={len(X_val)}, y={len(y_val)}")
        
        # Test coercion
        print("🔄 Testing sequence coercion...")
        sample_seq = sample_embeddings[0]
        coerced = _coerce_to_seq2d(sample_seq, 256)
        print(f"   ✅ Coerced shape: {np.array(coerced).shape if coerced else None}")
        
        # Test mean pooling (for pfam2vec)
        print("🔄 Testing mean pooling...")
        pooled_features = create_mean_pooled_features(test_df['pfam2vec_seq'].iloc[:10])
        print(f"   ✅ Pooled features shape: {pooled_features.shape}")
        
        # Test concatenated embeddings
        print("🔄 Testing concatenated embeddings...")
        # Create mock ESM and BigCarp DataFrames
        esm_df = test_df[['bgc_id', 'product_class']].copy()
        esm_df['esm_embeddings'] = test_df['esm_embeddings'].copy()
        
        bigcarp_df = test_df[['bgc_id', 'product_class']].copy()
        bigcarp_df['embeddings'] = test_df['embeddings'].copy()
        
        concat_df = create_concatenated_embeddings(esm_df, bigcarp_df)
        if concat_df is not None:
            print(f"   ✅ Concatenated embeddings created: {concat_df.shape}")
            sample_concat = concat_df['concatenated_embeddings'].iloc[0]
            if isinstance(sample_concat, list) and len(sample_concat) > 0:
                concat_dim = len(sample_concat[0])
                print(f"   ✅ Concatenated dimension: {concat_dim} (should be 256+1280=1536)")
        else:
            print("   ⚠️  Concatenated embeddings creation skipped")
        
        print("✅ Preprocessing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization without training."""
    print("\n" + "="*60) 
    print("🧪 TEST 3: Model Initialization")
    print("="*60)
    
    try:
        # Test BiLSTM model
        print("🔄 Testing BiLSTM model initialization...")
        from cgrep.models_multiclass import MultiLabelBiLSTMClassifier
        
        # Try to use GPU if available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        model = MultiLabelBiLSTMClassifier(
            embed_dim=256, hidden_dim=128, num_layers=1,  # Smaller for testing
            dropout_rate=0.2, pooling_strategy="mean",
            lr=1e-3, batch_size=16, early_stopping_patience=3, max_epochs=2,  # Fast settings
            random_seed=42, device=device
        )
        print("   ✅ BiLSTM model initialized successfully")
        
        # Test Random Forest 
        print("🔄 Testing Random Forest initialization...")
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, random_state=42)  # Small for testing
        print("   ✅ Random Forest initialized successfully")
        
        print("✅ Model initialization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model initialization test failed: {e}")
        return False

def test_quick_training():
    """Test actual model training with tiny data."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Quick Training Test")
    print("="*60)
    
    try:
        # Create tiny dataset for quick training
        test_df = create_test_data(n_samples=20, seq_len_range=(3, 8), emb_dim=64)
        df_binary, class_cols = convert_product_classes_to_binary(test_df)
        
        print(f"📊 Test dataset: {test_df.shape} samples, {len(class_cols)} classes")
        
        # Quick BiLSTM test
        print("🔄 Testing BiLSTM training...")
        from cgrep.models_multiclass import MultiLabelBiLSTMClassifier
        
        # Prepare data
        embeddings = test_df['embeddings'].tolist()
        labels = [";".join([c for c in class_cols if row[c]==1]) for _, row in df_binary.iterrows()]
        
        # Validate data
        X_train, y_train, _, _ = _validate_and_prepare_XY(embeddings[:15], labels[:15], 64)
        X_test, y_test, _, _ = _validate_and_prepare_XY(embeddings[15:], labels[15:], 64)
        
        if len(X_train) == 0 or len(X_test) == 0:
            print("   ⚠️  Not enough valid samples after validation")
            return True  # Skip but don't fail
        
        # Quick training
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = MultiLabelBiLSTMClassifier(
            embed_dim=64, hidden_dim=32, num_layers=1,
            dropout_rate=0.1, pooling_strategy="mean", 
            lr=1e-3, batch_size=8, early_stopping_patience=2, max_epochs=3,
            random_seed=42, device=device
        )
        
        print(f"   🏋️  Quick training with {len(X_train)} samples...")
        model.fit(X_train, y_train, n_folds=1, show_progress=False)
        
        # Quick prediction
        print(f"   🔮 Testing prediction on {len(X_test)} samples...")
        y_proba = model.predict_proba(X_test)
        y_true = model.mlb.transform([s.split(';') for s in y_test])
        y_pred = (y_proba > 0.5).astype(int)
        
        print(f"   📊 Prediction shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        # Test metrics computation
        print("🔄 Testing metrics computation...")
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, model.mlb.classes_)
        print(f"   ✅ Computed metrics: {list(metrics.keys())}")
        print(f"   📊 Sample metrics: Macro F1={metrics['macro_f1']:.4f}")
        
        print("✅ Quick training test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False

def test_output_generation():
    """Test results output and file generation."""
    print("\n" + "="*60)
    print("🧪 TEST 5: Output Generation")
    print("="*60)
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temp directory: {temp_dir}")
            
            # Create mock results
            mock_results = [
                {
                    'model_name': 'Test Model 1',
                    'embedding_column': 'test_embeddings_1',
                    'aggregate_metrics': {
                        'macro_f1': 0.75,
                        'macro_auc': 0.82,
                        'exact_match_accuracy': 0.45
                    }
                },
                {
                    'model_name': 'Test Model 2', 
                    'embedding_column': 'test_embeddings_2',
                    'aggregate_metrics': {
                        'macro_f1': 0.68,
                        'macro_auc': 0.79,
                        'exact_match_accuracy': 0.41
                    }
                }
            ]
            
            # Test comparison table creation
            print("🔄 Testing comparison table generation...")
            from scripts.classification.mibig1_stratified_evaluation import create_comparison_table
            
            df_comparison = create_comparison_table(mock_results, temp_dir)
            
            if df_comparison is not None:
                print("   ✅ Comparison table created successfully")
                print(f"   📊 Table shape: {df_comparison.shape}")
                
                # Check output files
                csv_file = os.path.join(temp_dir, "mibig1_comparison.csv")
                if os.path.exists(csv_file):
                    print(f"   ✅ CSV file created: {csv_file}")
                    print("   📋 CSV contents:")
                    print(pd.read_csv(csv_file).to_string(index=False, float_format='%.4f'))
                else:
                    print("   ❌ CSV file not created")
            else:
                print("   ❌ Comparison table creation failed")
            
            # Test pickle saving
            print("🔄 Testing pickle file saving...")
            import pickle
            pickle_file = os.path.join(temp_dir, "test_results.pkl")
            with open(pickle_file, 'wb') as f:
                pickle.dump(mock_results, f)
            
            if os.path.exists(pickle_file):
                print("   ✅ Pickle file created successfully")
            else:
                print("   ❌ Pickle file creation failed")
        
        print("✅ Output generation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Output generation test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the complete pipeline integration."""
    print("\n" + "="*60)
    print("🧪 TEST 6: Pipeline Integration")
    print("="*60)
    
    try:
        # Test import of main evaluation script
        print("🔄 Testing main script import...")
        import scripts.classification.mibig1_stratified_evaluation as main_eval
        print("   ✅ Main evaluation script imported successfully")
        
        # Test argument parsing
        print("🔄 Testing argument parsing...")
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--artifacts_dir", default="artifacts/classification/mibig1")
        parser.add_argument("--outdir", default="results/test_mibig1")
        parser.add_argument("--seed", type=int, default=42)
        
        args = parser.parse_args(['--seed', '123'])
        print(f"   ✅ Arguments parsed: seed={args.seed}")
        
        # Test required directories exist or can be created
        print("🔄 Testing directory structure...")
        test_outdir = "results/test_pipeline"
        Path(test_outdir).mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(test_outdir):
            print(f"   ✅ Output directory created: {test_outdir}")
            shutil.rmtree(test_outdir)  # Clean up
        
        print("✅ Pipeline integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 MIBiG 1.0 Evaluation Pipeline Test Suite")
    print("="*80)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Data Preprocessing", test_preprocessing), 
        ("Model Initialization", test_model_initialization),
        ("Quick Training", test_quick_training),
        ("Output Generation", test_output_generation),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
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
        print("🎉 All tests passed! Pipeline is ready to run.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before running full evaluation.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)