#!/bin/bash --login

# Extract Average Embeddings from Epoch 89 Checkpoints
# This script extracts both embedder and last layer embeddings from checkpoint_epoch89.tar
# for ESM init, ESM init frozen, and Random init models

set -e  # Exit on any error

# Define paths
SCRIPT_PATH="scripts/umap/extract_average_embeddings.py"
VOCAB_PATH="data/processed/vocabularies/pfam_vocab_present.json"
CORPUS_PATH="data/processed/bgc_corpus/antidb_pfam_corpus.csv"
SAVE_DIR="artifacts/bigcarp/average_embeddings"

# Create output directory
mkdir -p "$SAVE_DIR"
echo "Created output directory: $SAVE_DIR"

# Define checkpoint paths and their corresponding output prefixes
declare -A CHECKPOINTS
CHECKPOINTS["artifacts/bigcarp/bigcarp_models/run_esm_init/checkpoint_epoch89.tar"]="esm_init_epoch89"
CHECKPOINTS["artifacts/bigcarp/bigcarp_models/run_esm_init_frozen/checkpoint_epoch89.tar"]="esm_init_frozen_epoch89"
CHECKPOINTS["artifacts/bigcarp/bigcarp_models/run_random_init/checkpoint_epoch89.tar"]="random_init_epoch89"

# Layer indices to extract
LAYERS=("embedder" "last")

echo "🚀 Starting embedding extraction from epoch 89 checkpoints..."
echo "📁 Vocabulary: $VOCAB_PATH"
echo "📁 Corpus: $CORPUS_PATH"
echo "📁 Output directory: $SAVE_DIR"
echo ""

# Function to extract embeddings for a single checkpoint and layer
extract_embeddings() {
    local checkpoint_path="$1"
    local output_prefix="$2"
    local layer="$3"
    local frozen_flag="$4"
    
    echo "🔧 Extracting $layer embeddings from: $(basename $checkpoint_path)"
    echo "   Output prefix: $output_prefix"
    
    # Construct the command
    local cmd="python $SCRIPT_PATH single \
        --checkpoint-path \"$checkpoint_path\" \
        --vocab-path \"$VOCAB_PATH\" \
        --corpus-path \"$CORPUS_PATH\" \
        --save-dir \"$SAVE_DIR\" \
        --layer-indices $layer"
    
    # Add frozen flag if needed
    if [[ "$frozen_flag" == "true" ]]; then
        cmd="$cmd --frozen-embeddings"
    fi
    
    echo "   Command: $cmd"
    
    # Execute the command
    if eval $cmd; then
        echo "   ✅ Successfully extracted $layer embeddings"
        
        # Rename the output file to include the model type and epoch info
        local original_file="$SAVE_DIR/embeddings_checkpoint_epoch89_$layer.pt"
        local new_file="$SAVE_DIR/embeddings_${output_prefix}_${layer}.pt"
        
        if [[ -f "$original_file" ]]; then
            mv "$original_file" "$new_file"
            echo "   📄 Renamed to: $(basename $new_file)"
        else
            echo "   ⚠️  Expected output file not found: $original_file"
        fi
    else
        echo "   ❌ Failed to extract $layer embeddings from $checkpoint_path"
        return 1
    fi
    
    echo ""
}

# Process each checkpoint
total_extractions=0
successful_extractions=0

for checkpoint_path in "${!CHECKPOINTS[@]}"; do
    output_prefix="${CHECKPOINTS[$checkpoint_path]}"
    
    # Check if checkpoint exists
    if [[ ! -f "$checkpoint_path" ]]; then
        echo "❌ Checkpoint not found: $checkpoint_path"
        continue
    fi
    
    echo "📦 Processing checkpoint: $(basename $checkpoint_path)"
    echo "   Full path: $checkpoint_path"
    
    # Determine if this is a frozen model
    frozen_flag="false"
    if [[ "$checkpoint_path" == *"frozen"* ]]; then
        frozen_flag="true"
        echo "   🔒 Using frozen embeddings flag"
    fi
    
    # Extract embeddings for each layer
    for layer in "${LAYERS[@]}"; do
        total_extractions=$((total_extractions + 1))
        
        if extract_embeddings "$checkpoint_path" "$output_prefix" "$layer" "$frozen_flag"; then
            successful_extractions=$((successful_extractions + 1))
        fi
    done
    
    echo "─────────────────────────────────────────────────────────────────────"
done

# Final summary
echo ""
echo "🎯 EXTRACTION SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo "📊 Total extractions attempted: $total_extractions"
echo "✅ Successful extractions: $successful_extractions"
echo "❌ Failed extractions: $((total_extractions - successful_extractions))"

if [[ $successful_extractions -eq $total_extractions ]]; then
    echo "🎉 All extractions completed successfully!"
else
    echo "⚠️  Some extractions failed. Check the logs above for details."
fi

echo ""
echo "📁 Output files saved in: $SAVE_DIR"
echo "📋 Generated files:"
ls -la "$SAVE_DIR"/*.pt 2>/dev/null || echo "   No .pt files found in output directory"

echo ""
echo "🏁 Embedding extraction script completed!"

# Exit with appropriate code
if [[ $successful_extractions -eq $total_extractions ]]; then
    exit 0
else
    exit 1
fi