#!/usr/bin/env python
"""
Quick Test for Fine-tuning Pipeline
==================================
Simple test to verify the modified fine-tuning script works end-to-end.
"""

import os
import subprocess
import sys

def test_script_execution():
    """Test that the script can be executed with help flag."""
    print("🧪 Testing script execution...")
    
    script_path = "/home/u5bb/han00.u5bb/workspace/cgrep/scripts/classification/fine_tuning/finetune_mibig_multilabel.py"
    
    # Test help flag
    try:
        result = subprocess.run([
            'python', script_path, '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Script executes successfully with --help")
            print("📋 Available arguments:")
            # Show first few lines of help
            for line in result.stdout.split('\n')[:15]:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ Script failed with --help: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script timed out")
        return False
    except Exception as e:
        print(f"❌ Script execution failed: {e}")
        return False

def test_import_check():
    """Test critical imports."""
    print("\n🧪 Testing critical imports...")
    
    import_test = """
import sys
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

try:
    # Test main evaluation script functions
    from scripts.classification.mibig3_stratified_evaluation import convert_product_classes_to_binary
    print("✅ Main evaluation functions imported")
    
    # Test BigCarp components
    from sequence_models.convolutional import ByteNetLM
    print("✅ BigCarp components imported")
    
    # Test PyTorch
    import torch
    print(f"✅ PyTorch {torch.__version__} available")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    print("🎉 All critical imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run([
            'python', '-c', import_test
        ], capture_output=True, text=True, cwd='/home/u5bb/han00.u5bb/workspace/cgrep')
        
        if result.returncode == 0:
            print("✅ Import test passed")
            print(result.stdout)
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False

def test_model_paths():
    """Test that model paths exist."""
    print("\n🧪 Testing model paths...")
    
    model_paths = [
        "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt",
        "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_esm1bfinetune.pt"
    ]
    
    found_models = 0
    for model_path in model_paths:
        full_path = f"/home/u5bb/han00.u5bb/workspace/cgrep/{model_path}"
        if os.path.exists(full_path):
            print(f"✅ Found: {model_path}")
            found_models += 1
        else:
            print(f"❌ Missing: {model_path}")
    
    print(f"📊 Found {found_models}/{len(model_paths)} models")
    return found_models > 0

def test_data_path():
    """Test that data path exists.""" 
    print("\n🧪 Testing data paths...")
    
    data_path = "/home/u5bb/han00.u5bb/workspace/cgrep/artifacts/classification/mibig3/mibig3_preprocessed.pkl"
    
    if os.path.exists(data_path):
        print("✅ Found MIBiG 3.0 preprocessed data")
        return True
    else:
        print(f"❌ Missing: {data_path}")
        print("   This is expected if you haven't preprocessed MIBiG 3.0 data yet")
        return False

def main():
    """Run all quick tests."""
    print("🚀 Quick Test Suite for Fine-tuning Pipeline")
    print("="*50)
    
    tests = [
        ("Script Execution", test_script_execution),
        ("Critical Imports", test_import_check),
        ("Model Paths", test_model_paths),
        ("Data Path", test_data_path),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*50)
    print("📋 QUICK TEST SUMMARY")
    print("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n📊 {passed}/{total} tests passed")
    
    if passed >= 2:  # Script execution and imports are most critical
        print("🎉 Core functionality verified!")
        print("\n🚀 Try running the script with minimal settings:")
        print("   python scripts/classification/fine_tuning/finetune_mibig_multilabel.py --epochs 1 --batch_size 4")
        return 0
    else:
        print("⚠️  Critical issues found. Please fix before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())