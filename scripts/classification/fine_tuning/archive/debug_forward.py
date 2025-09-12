#!/usr/bin/env python
"""
Debug BigCarp Forward Pass
===========================
Diagnose the tensor size mismatch in the forward pass.
"""

import sys
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

import torch
import json
import pickle
import pandas as pd
from sequence_models.convolutional import ByteNetLM

def load_sample_data():
    """Load a small sample of data to debug with."""
    data_path = "data/processed/bgc_product_classification/processed_mibig3/mibig3_preprocessed.pkl"
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📊 Data shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Take just a few samples
    sample_data = data.head(3).copy()
    print(f"   Sample sequences lengths: {[len(seq) for seq in sample_data['pfam_sequence']]}")
    
    return sample_data

def debug_forward_pass():
    """Debug the forward pass step by step."""
    print("🔍 Debugging BigCarp Forward Pass")
    print("=" * 50)
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Load vocabulary
    vocab_path = "/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json"
    with open(vocab_path, 'r') as f:
        vocab_info = json.load(f)
    
    specials = vocab_info['specials']
    domains = vocab_info['domains']
    vocab_size = len(domains) + len(specials)
    
    print(f"📚 Vocabulary: {len(domains)} domains + {len(specials)} specials = {vocab_size}")
    
    # Load model
    model_path = "artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with correct config
    d_model = checkpoint.get('d_model', 256)
    d_embed = checkpoint.get('d_embed', 1280)
    n_layers = checkpoint.get('n_layers', 30)
    
    print(f"🧠 Model config: d_model={d_model}, d_embed={d_embed}, n_layers={n_layers}")
    
    model = ByteNetLM(
        n_tokens=vocab_size,
        d_embedding=d_embed,
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=3,
        r=128,
        slim=True,
        padding_idx=specials['-'],
        causal=False,
        final_ln=True,
        activation='gelu'
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return
    
    # Convert sequences to tokens
    mask_idx = specials['#']
    unk_idx = specials.get('UNK', specials.get('<unk>', 0))
    
    print(f"🔑 Special tokens: mask={mask_idx}, unk={unk_idx}")
    
    # Process one sequence at a time to isolate the issue
    for i, row in sample_data.iterrows():
        sequence = row['pfam_sequence']
        print(f"\n🧬 Sample {i}: sequence length = {len(sequence)}")
        
        # Convert to tokens
        tokens = []
        for domain in sequence:
            if domain in domains:
                tokens.append(domains[domain] + len(specials))
            else:
                tokens.append(unk_idx)
        
        if len(tokens) == 0:
            tokens = [mask_idx]  # Fallback for empty sequences
            
        print(f"   Token sequence length: {len(tokens)}")
        print(f"   First few tokens: {tokens[:5]}")
        
        # Create tensor
        tokens_tensor = torch.tensor([tokens])  # Add batch dimension
        print(f"   Input tensor shape: {tokens_tensor.shape}")
        
        # Create attention mask
        attention_mask = torch.ones(1, len(tokens), 1)  # (batch, seq_len, 1)
        print(f"   Attention mask shape: {attention_mask.shape}")
        
        # Try forward pass step by step
        model.eval()
        with torch.no_grad():
            try:
                print("   🔸 Testing embedder...")
                features = model.embedder(tokens_tensor, input_mask=attention_mask)
                print(f"   ✅ Embedder output shape: {features.shape}")
                
                print("   🔸 Testing last_norm...")
                if hasattr(model, 'last_norm'):
                    normed_features = model.last_norm(features)
                    print(f"   ✅ After last_norm shape: {normed_features.shape}")
                else:
                    print("   ⚠️  No last_norm found")
                    normed_features = features
                
                print("   🔸 Testing pooling...")
                # Mean pooling
                mask = attention_mask.squeeze(-1)  # (batch, seq_len)
                print(f"   Mask for pooling shape: {mask.shape}")
                pooled = (normed_features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                print(f"   ✅ After pooling shape: {pooled.shape}")
                
            except Exception as e:
                print(f"   ❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Let's check the exact point of failure
                print("\n   📋 Detailed error analysis:")
                print(f"   Input tensor: {tokens_tensor.shape} = {tokens_tensor}")
                print(f"   Input mask: {attention_mask.shape} = {attention_mask}")
                
                break

if __name__ == "__main__":
    debug_forward_pass()