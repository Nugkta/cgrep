#!/usr/bin/env python
"""
Debug BigCarp Model Dimensions
==============================
Quick script to understand the actual output dimensions from BigCarp models.
"""

import sys
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

import torch
import json
from sequence_models.convolutional import ByteNetLM
from pathlib import Path

def debug_bigcarp_dimensions():
    """Debug BigCarp model output dimensions."""
    print("🔍 Debugging BigCarp Model Dimensions")
    print("=" * 50)
    
    # Load vocabulary
    vocab_path = "/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json"
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"📚 Vocabulary size: {len(vocab)}")
    
    # Model paths
    model_paths = [
        "/home/u5bb/han00.u5bb/workspace/cgrep/artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt",
        "/home/u5bb/han00.u5bb/workspace/cgrep/artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_esm1bfinetune.pt"
    ]
    
    for model_path in model_paths:
        print(f"\n🧠 Testing model: {Path(model_path).name}")
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            
            # Get model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            print(f"   State dict type: {type(state_dict)}")
            if isinstance(state_dict, str):
                print(f"   State dict is string: {state_dict}")
                continue
                
            # Find embedding/output dimensions from state dict
            print(f"   Model layers:")
            for key, tensor in state_dict.items():
                if any(x in key for x in ['embed', 'norm', 'final', 'output']):
                    print(f"     {key}: {tensor.shape}")
                    
            # Create model instance to test
            # First, let's find the hidden dimension from the state dict
            hidden_dim = None
            for key, tensor in state_dict.items():
                if 'embedder.embedder.weight' in key:
                    hidden_dim = tensor.shape[1]
                    break
                elif 'embed' in key and len(tensor.shape) == 2:
                    hidden_dim = tensor.shape[1]
                    
            if hidden_dim:
                print(f"   Detected hidden_dim: {hidden_dim}")
                
                # Create ByteNet model
                model = ByteNetLM(
                    n_tokens=len(vocab), 
                    d_embedding=hidden_dim,
                    d_model=hidden_dim,
                    n_layers=30,
                    kernel_size=3,
                    r=128
                )
                
                # Load weights
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                
                # Test with dummy input
                batch_size = 2
                seq_len = 100
                dummy_tokens = torch.randint(0, len(vocab), (batch_size, seq_len))
                dummy_mask = torch.ones(batch_size, seq_len, 1)
                
                print(f"   Input shape: {dummy_tokens.shape}")
                print(f"   Mask shape: {dummy_mask.shape}")
                
                with torch.no_grad():
                    # Test embedder output
                    embedder_output = model.embedder(dummy_tokens, input_mask=dummy_mask)
                    print(f"   Embedder output shape: {embedder_output.shape}")
                    
                    # Test after layer norm
                    if hasattr(model, 'last_norm'):
                        normed_output = model.last_norm(embedder_output)
                        print(f"   After last_norm shape: {normed_output.shape}")
                    else:
                        print("   No last_norm layer found")
                        normed_output = embedder_output
                    
                    # Test pooling
                    mask = dummy_mask.squeeze(-1)
                    pooled = (normed_output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                    print(f"   After mean pooling shape: {pooled.shape}")
                    
                print("   ✅ Model test successful")
            else:
                print("   ❌ Could not determine hidden dimension")
                
        except Exception as e:
            print(f"   ❌ Error testing model: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_bigcarp_dimensions()