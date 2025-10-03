#!/usr/bin/env python3
"""
UMAP Interactive Plotting Script for Pfam Domain Embeddings

Generates UMAP plots with multiple visualization options.
By default: generates clan plots and interactive HTML plots only.

Arguments:
  vocab_file              Path to domain vocabulary JSON file
  clans_file              Path to PFAM clans CSV file
  embeddings              Path to embeddings .pt file or directory containing multiple .pt files
  results_dir             Output directory where plots will be saved

  --technique             Dimensionality reduction method: 'pca', 'tsne', or 'umap' (default: 'umap')
  --tag                   String tag added to output filenames for organization (default: empty)
  --plot_by_presence      Create plots colored by domain presence/absence
  --plot_by_product_class Create plots colored by product class (requires --product_class_file)
  --plot_by_function_label Create plots colored by functional labels (requires --function_labels_csv)
  --product_class_file    JSON file mapping domains to product classes
  --function_labels_csv   CSV file with functional annotations for domains
  --no_interactive        Skip generating interactive HTML plots (only create static PDFs)

Usage Example:

python scripts/umap/plot_UMAP_interact.py \
  data/processed/vocabularies/pfam_vocab_present_pid.json \
  data/raw/Pfam-A.clans.csv \
  artifacts/bigcarp/average_embeddings/random_init/embeddings_checkpoint_best_last.pt \
  results \
  --plot_by_product_class \
  --tag random_init \
  --product_class_file data/processed/umap_label/pfam_product_classes_label.json \
  --plot_by_function_label \
  --function_labels_csv data/processed/umap_label/pfam_function_labels.csv
"""

import argparse
import json
import os
import glob
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import datetime
from tqdm import tqdm

# Set reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Plotting parameters
plt.style.use('default')  
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 20,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.grid': False,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white'
})


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Plot PFAM domain embeddings')
    parser.add_argument('vocab_file', help='Path to domain vocabulary JSON')
    parser.add_argument('clans_file', help='Path to PFAM clans CSV file')
    parser.add_argument('embeddings', help='Path to embeddings .pt file or directory')
    parser.add_argument('results_dir', help='Output directory for plots')
    parser.add_argument('--technique', default='umap', choices=['pca', 'tsne', 'umap'])
    parser.add_argument('--tag', default='', help='Tag for output filenames')
    parser.add_argument('--plot_by_presence', action='store_true')
    parser.add_argument('--plot_by_product_class', action='store_true')
    parser.add_argument('--plot_by_function_label', action='store_true')
    parser.add_argument('--product_class_file', help='JSON file with product class mappings')
    parser.add_argument('--function_labels_csv', help='CSV with function labels')
    parser.add_argument('--no_interactive', action='store_true', help='Skip HTML plots')

    args = parser.parse_args()
    
    
    return args


def load_embeddings(emb_path):
    """Load embeddings from file or directory"""
    if os.path.isdir(emb_path):
        files = sorted(glob.glob(os.path.join(emb_path, '*.pt')))
    else:
        files = [emb_path]
    
    embeddings = []
    for f in files:
        try:
            emb = torch.load(f, map_location='cpu', weights_only=True)
        except TypeError:
            emb = torch.load(f, map_location='cpu')
        embeddings.append((f, emb.cpu().numpy()))
    return embeddings


def load_metadata(args):
    """Load all metadata (domains, clans, product classes, function labels)"""
    # Load vocabulary
    with open(args.vocab_file) as f:
        vocab = json.load(f)
    domains = list(vocab['domains'].keys())
    
    # Load clans
    clans_df = pd.read_csv(args.clans_file)
    domain_to_clan = dict(zip(clans_df['pfam_id'], clans_df['clan_id']))
    
    metadata = {
        'domains': domains,
        'clans': [domain_to_clan.get(d) for d in domains]
    }
    
    # Load product classes if requested
    if args.plot_by_product_class and args.product_class_file:
        with open(args.product_class_file) as f:
            prod_data = json.load(f)
        prod_mapping = prod_data.get('pfam_to_product_class', {})
        metadata['product_classes'] = [prod_mapping.get(d) for d in domains]
    
    # Load function labels if requested
    if args.plot_by_function_label and args.function_labels_csv:
        func_df = pd.read_csv(args.function_labels_csv)
        func_mapping = dict(zip(func_df['pfam_id'], func_df['primary_role']))
        metadata['function_labels'] = [func_mapping.get(d) for d in domains]
    
    return metadata


def apply_dimensionality_reduction(embeddings, technique):
    """Apply dimensionality reduction technique"""
    if technique == 'tsne':
        return TSNE(n_components=2, random_state=SEED).fit_transform(embeddings)
    elif technique == 'umap':
        scaled = StandardScaler().fit_transform(embeddings)
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=10, min_dist=0.05, metric='cosine') # using cosine distance for embeddings
        return reducer.fit_transform(scaled)
    else:  # pca
        scaled = StandardScaler().fit_transform(embeddings)
        return PCA(n_components=2, random_state=SEED).fit_transform(scaled)


def save_plot(fig_or_path, results_dir, tag, emb_name, plot_type, timestamp):
    """Save plot with consistent naming and error handling"""
    # Create subdirectory structure: results_dir/umap/tag/emb_name/
    subdir = os.path.join(results_dir, 'umap', tag, emb_name)
    os.makedirs(subdir, exist_ok=True)
    save_path = os.path.join(subdir, f"umap_{plot_type}_{timestamp}.pdf")
    
    try:
        if hasattr(fig_or_path, 'savefig'):  # matplotlib figure
            fig_or_path.savefig(save_path, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
        else:  # plotly figure
            fig_or_path.write_html(save_path.replace('.pdf', '.html'))
        print(f"Saved {plot_type} plot: {save_path}")
    except Exception as e:
        print(f"Error saving {plot_type} plot: {e}")
    finally:
        plt.close()


def create_static_plot(embed_df, technique, hue_col, title, palette='Paired'):
    """Create a static matplotlib scatter plot for dimensionality-reduced embeddings.

    Args:
        embed_df: DataFrame containing coordinates, domains, and metadata columns
        technique: Dimensionality reduction technique ('umap', 'pca', 'tsne')
        hue_col: Column name to use for color-coding points
        title: Plot title (currently unused)
        palette: Seaborn color palette name (default: 'Paired')

    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter_size = 50

    # Prepare data and colors based on hue column type
    plot_df, colors = _prepare_plot_data(embed_df, hue_col, palette)

    # Create scatter plot
    if plot_df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=16)
    elif hue_col in plot_df.columns:
        sns.scatterplot(data=plot_df, x=f'{technique}_1', y=f'{technique}_2',
                      hue=hue_col, palette=colors, s=scatter_size, alpha=0.8,
                      edgecolor='white', linewidth=0.3, ax=ax)
    else:
        ax.scatter(plot_df[f'{technique}_1'], plot_df[f'{technique}_2'],
                  s=scatter_size, alpha=0.7, c='#3498DB', edgecolor='white', linewidth=0.3)

    # Format axes and legend
    _format_plot(ax, technique)

    plt.tight_layout()
    return fig


def _prepare_plot_data(embed_df, hue_col, palette):
    """Prepare plot data and color palette based on hue column type"""
    # Function labels: filter out 'other' and use specific colors
    if hue_col == 'function_labels' and 'function_labels' in embed_df.columns:
        plot_df = embed_df[embed_df['function_labels'].notna()]
        plot_df = plot_df[plot_df['function_labels'].astype(str).str.lower() != 'other']
        colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']
        return plot_df, colors

    # No hue column
    if hue_col not in embed_df.columns:
        return embed_df, None

    # Limit to top N categories for clans and product_classes
    top_n = 8 if hue_col == 'clans' else 15
    top_categories = embed_df[hue_col].value_counts().head(top_n).index
    plot_df = embed_df[embed_df[hue_col].isin(top_categories)]

    # Select color palette
    if hue_col == 'clans':
        clan_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#17becf']
        colors = clan_colors[:len(plot_df[hue_col].unique())]
    elif hue_col == 'product_classes':
        colors = sns.color_palette("Set2", n_colors=len(plot_df[hue_col].unique()))
    else:
        colors = sns.color_palette(palette, n_colors=len(plot_df[hue_col].unique()))

    return plot_df, colors


def _format_plot(ax, technique):
    """Format plot axes and legend"""
    ax.set_xlabel(f'{technique.upper()} 1', fontsize=26)
    ax.set_ylabel(f'{technique.upper()} 2', fontsize=26)

    # Remove ticks but keep axis lines clean
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Enhance legend if present
    if ax.get_legend():
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                          frameon=True, fancybox=False, shadow=False,
                          title_fontsize=22, fontsize=20)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)


def create_interactive_plot(embed_df, technique, hue_col, title):
    """Create interactive plotly plot"""
    try:
        import plotly.express as px
        
        hover_data = ['domains']
        if hue_col != 'domains' and hue_col in embed_df.columns:
            hover_data.append(hue_col)
            
        fig = px.scatter(embed_df, x=f'{technique}_1', y=f'{technique}_2',
                        color=hue_col, hover_data=hover_data)
        fig.update_traces(marker=dict(size=6, opacity=0.8))

        # Update layout for larger fonts
        fig.update_layout(
            font=dict(size=20),
            xaxis=dict(title=dict(text=f'{technique.upper()} 1', font=dict(size=26))),
            yaxis=dict(title=dict(text=f'{technique.upper()} 2', font=dict(size=26))),
            legend=dict(font=dict(size=20))
        )
        return fig
    except ImportError:
        print("Plotly not available, skipping interactive plot")
        return None


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load data
    print("Loading embeddings and metadata...")
    embeddings_list = load_embeddings(args.embeddings)
    metadata = load_metadata(args)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for emb_file, embeddings in tqdm(embeddings_list, desc="Processing embeddings"):
        emb_name = os.path.splitext(os.path.basename(emb_file))[0]
        print(f"\nProcessing: {emb_file}")
        
        # Adjust for UNK token if needed
        if len(embeddings) != len(metadata['domains']):
            if len(metadata['domains']) == len(embeddings) + 1 and metadata['domains'][0] == "UNK":
                print("Removing UNK token from domains list")
                for key in metadata:
                    if isinstance(metadata[key], list):
                        metadata[key] = metadata[key][1:]
        
        # Apply dimensionality reduction
        coords = apply_dimensionality_reduction(embeddings, args.technique)
        
        # Create DataFrame
        embed_df = pd.DataFrame({
            f'{args.technique}_1': coords[:, 0],
            f'{args.technique}_2': coords[:, 1],
            'domains': metadata['domains']
        })
        
        # Add metadata columns
        for key, values in metadata.items():
            if key != 'domains':
                embed_df[key] = values
        
        # Generate plots
        plots_to_make = []
        
        # Default: always make clan plot
        plots_to_make.append(('clans', 'Plot by Clan', 'Paired'))
        
        # Add specific plot types based on arguments
        if args.plot_by_product_class and 'product_classes' in metadata:
            plots_to_make.append(('product_classes', 'Plot by Product Class', 'Set1'))
        
        if args.plot_by_function_label and 'function_labels' in metadata:
            plots_to_make.append(('function_labels', 'Plot by Function Label', 'tab10'))
        
        if args.plot_by_presence:
            # Add presence information (simplified)
            embed_df['presence'] = 'Present'  # Default, can be customized
            plots_to_make.append(('presence', 'Plot by Presence', {'Present': 'blue', 'Absent': 'red'}))
        
        # Create static plots
        for hue_col, title, palette in plots_to_make:
            if hue_col in embed_df.columns:
                fig = create_static_plot(embed_df, args.technique, hue_col, title, palette)
                save_plot(fig, args.results_dir, args.tag, emb_name, hue_col, timestamp)
        
        # Create interactive plot
        if not args.no_interactive:
            # Use the first available categorical column for coloring
            color_col = 'clans'
            for col in ['function_labels', 'product_classes', 'presence']:
                if col in embed_df.columns:
                    color_col = col
                    break
            
            print(f"Interactive plot using label: {color_col}")
            interactive_fig = create_interactive_plot(embed_df, args.technique, color_col,
                                                    f"Interactive {args.technique.upper()} Plot")
            if interactive_fig:
                save_plot(interactive_fig, args.results_dir, args.tag, emb_name, 'interactive', timestamp)
    
    print("\nDone!")


if __name__ == "__main__":
    main()