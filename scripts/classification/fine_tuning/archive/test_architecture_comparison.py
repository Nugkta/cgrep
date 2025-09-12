#!/usr/bin/env python
"""
Test Architecture Comparison Script
===================================
Quick test to validate the architecture comparison functionality.
"""

import sys
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

def test_data_loading():
    print("🧪 Testing data loading...")
    
    from bigcarp_architecture_comparison import load_mibig3_data, prepare_single_label_data
    
    # Test data loading
    data = load_mibig3_data()
    if data is not None:
        print(f"✅ Data loaded: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        
        # Test single-label conversion
        single_data, label_encoder = prepare_single_label_data(data)
        print(f"✅ Single-label data: {single_data.shape}")
        print(f"   Classes: {list(label_encoder.classes_)}")
        
        return True
    else:
        print("❌ Data loading failed")
        return False

def test_vocabulary_loading():
    print("\n🧪 Testing vocabulary loading...")
    
    import json
    
    vocab_path = "/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json"
    
    try:
        with open(vocab_path, 'r') as f:
            vocab_info = json.load(f)
        
        specials = vocab_info['specials']
        domains = vocab_info['domains']
        
        print(f"✅ Vocabulary loaded")
        print(f"   Specials: {len(specials)} tokens")
        print(f"   Domains: {len(domains)} tokens")
        print(f"   Total vocab size: {len(specials) + len(domains)}")
        
        # Check for important tokens
        print(f"   Class tokens available:")
        classes = ['Alkaloid', 'NRP', 'Other', 'Polyketide', 'RiPP', 'Saccharide', 'Terpene']
        for cls in classes:
            if cls in specials:
                print(f"     ✅ {cls}: {specials[cls]}")
            else:
                print(f"     ❌ {cls}: missing")
        
        return True, specials, domains
    except Exception as e:
        print(f"❌ Vocabulary loading failed: {e}")
        return False, None, None

def test_model_loading():
    print("\n🧪 Testing model loading...")
    
    import os
    
    model_paths = [
        "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt",
        "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_esm1bfinetune.pt"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Found: {model_path}")
        else:
            print(f"❌ Missing: {model_path}")

def test_dataset_creation():
    print("\n🧪 Testing dataset creation...")
    
    from bigcarp_architecture_comparison import (
        load_mibig3_data, prepare_single_label_data, 
        MIBiGSingleLabelDataset
    )
    
    # Load small sample
    data = load_mibig3_data()
    if data is None:
        print("❌ Cannot test dataset - data loading failed")
        return False
    
    # Take small sample
    sample_data = data.head(10).copy()
    single_data, label_encoder = prepare_single_label_data(sample_data)
    
    # Load vocab
    import json
    vocab_path = "/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json"
    with open(vocab_path, 'r') as f:
        vocab_info = json.load(f)
    
    specials = vocab_info['specials']
    domains = vocab_info['domains']
    
    try:
        # Test dataset creation with class tokens
        dataset_with_tokens = MIBiGSingleLabelDataset(
            single_data, specials, domains, 
            include_class_token=True, use_unknown_token=False
        )
        
        # Test dataset creation without class tokens  
        dataset_unknown_tokens = MIBiGSingleLabelDataset(
            single_data, specials, domains,
            include_class_token=True, use_unknown_token=True
        )
        
        print(f"✅ Dataset with class tokens: {len(dataset_with_tokens)} samples")
        print(f"✅ Dataset with unknown tokens: {len(dataset_unknown_tokens)} samples")
        
        # Test sample
        sample = dataset_with_tokens[0]
        print(f"   Sample sequence length: {len(sample['sequence'])}")
        print(f"   Sample label: {sample['label'].item()} ({sample['primary_class']})")
        
        return True
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Testing BigCarp Architecture Comparison Components")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Vocabulary Loading", test_vocabulary_loading), 
        ("Model Paths", test_model_loading),
        ("Dataset Creation", test_dataset_creation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    passed = sum(isinstance(r, bool) and r for r in results.values())
    total = len(results)
    print(f"\n📊 {passed}/{total} tests passed")
    
    if passed >= 3:
        print("🎉 Core functionality verified!")
        print("\n🚀 Try running the full comparison:")
        print("   python scripts/classification/fine_tuning/bigcarp_architecture_comparison.py --epochs 2 --batch_size 4")
    else:
        print("⚠️  Critical issues found. Please fix before proceeding.")

if __name__ == "__main__":
    main()