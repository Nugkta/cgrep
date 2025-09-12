# Cross-granularity Representation Project (cgrep)

## Environment Setup

### Quick Installation

```bash
# 1. Create conda environment with Python 3.10
conda create --name cgrep python=3.10
conda activate cgrep

# 2. Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio -c pytorch

# 3. Install core scientific computing packages
conda install numpy scipy pandas matplotlib seaborn scikit-learn statsmodels -c conda-forge

# 4. Install bioinformatics and additional packages
conda install biopython tqdm pyyaml joblib -c conda-forge -c bioconda
pip install fair-esm rdkit plotly umap-learn numba

# 5. Install multi-label classification packages
pip install scikit-multilearn

# 6. Clone and install protein-sequence-models locally
git clone https://github.com/nugkta/protein-sequence-models.git external/protein-sequence-models
pip install -e external/protein-sequence-models/

# 7. Install other required custom packages
pip install git+https://github.com/RistoAle97/centered-kernel-alignment
# pip install maspr  # Install when needed

# 8. Install cgrep package in development mode
pip install -e .

```

### Package Versions Reference
- Python 3.10.18
- PyTorch 2.7.1+ with CUDA support
- NumPy 2.1.2+
- Pandas 2.3.2+
- Matplotlib 3.10.5+
- Seaborn 0.13.2+
- scikit-learn 1.7.1+
- Plotly 6.3.0+ (for interactive UMAP plots)



