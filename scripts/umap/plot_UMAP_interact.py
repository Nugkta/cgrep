#!/usr/bin/env python3
"""
UMAP Interactive Plotting Script for PFAM Domain Embeddings

Generates UMAP plots with multiple visualization options.
By default: generates clan plots and interactive HTML plots only.

Usage Example:

python scripts/umap/plot_UMAP_interact.py \
  data/processed/vocabularies/pfam_vocab_present_pid.json \
  data/raw/Pfam-A.clans.csv \
  artifacts/bigcarp/average_embeddings/random_init/embeddings_checkpoint_best_last.pt \
  results \
  --plot_by_product_class \
  --product_class_file data/processed/umap_label/pfam_product_classes_label.json \
  --plot_by_function_label \
  --function_labels_csv data/processed/umap_label/pfam_function_labels.csv
"""

import argparse
import json
import os
import glob
# from collections import Counter  # Not needed in this version
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

# Publication-quality plotting setup
plt.style.use('default')  # Use clean default style instead of ggplot
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
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
    domains = vocab.get('domains_array', list(vocab.get('domains', {})))
    
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
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=10, min_dist=0.05, metric='cosine')
        return reducer.fit_transform(scaled)
    else:  # pca
        scaled = StandardScaler().fit_transform(embeddings)
        return PCA(n_components=2, random_state=SEED).fit_transform(scaled)


def save_plot(fig_or_path, results_dir, tag, emb_name, plot_type, timestamp):
    """Save plot with consistent naming and error handling"""
    # Create subdirectory structure: results_dir/umap/tag/emb_name/
    subdir = os.path.join(results_dir, 'umap', tag, emb_name)
    os.makedirs(subdir, exist_ok=True)
    save_path = os.path.join(subdir, f"umap_{plot_type}_{timestamp}.png")
    
    try:
        if hasattr(fig_or_path, 'savefig'):  # matplotlib figure
            fig_or_path.savefig(save_path, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
        else:  # plotly figure
            fig_or_path.write_html(save_path.replace('.png', '.html'))
        print(f"Saved {plot_type} plot: {save_path}")
    except Exception as e:
        print(f"Error saving {plot_type} plot: {e}")
    finally:
        plt.close()


def create_static_plot(embed_df, technique, hue_col, title, palette='Paired'):
    """Create a static matplotlib plot"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Handle function labels with special color mapping
    if hue_col == 'function_labels' and 'function_labels' in embed_df.columns:
        # Filter out 'other' labels and get top categories
        filtered_df = embed_df[embed_df['function_labels'].notna()]
        filtered_df = filtered_df[filtered_df['function_labels'].astype(str).str.lower() != 'other']
        
        if not filtered_df.empty:
            top_functions = filtered_df['function_labels'].value_counts().head(12).index
            plot_df = filtered_df[filtered_df['function_labels'].isin(top_functions)]
            
            # Enhanced color palette for functions
            colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD', '#17A2B8',
                     '#FFC107', '#6F42C1', '#20C997', '#FD7E14', '#DC3545', '#6C757D']
            
            sns.scatterplot(data=plot_df, x=f'{technique}_1', y=f'{technique}_2', 
                          hue='function_labels', palette=colors[:len(top_functions)], 
                          s=50, alpha=0.8, edgecolor='white', linewidth=0.3, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No function data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
    else:
        # Standard plotting for other categories
        if hue_col in embed_df.columns:
            # Get top categories for cleaner visualization
            if embed_df[hue_col].dtype == 'object':
                top_categories = embed_df[hue_col].value_counts().head(15).index
                plot_df = embed_df[embed_df[hue_col].isin(top_categories)]
            else:
                plot_df = embed_df
            
            # Enhanced color palettes
            if hue_col == 'clans':
                colors = sns.color_palette("Set3", n_colors=len(plot_df[hue_col].unique()))
            elif hue_col == 'product_classes':
                colors = sns.color_palette("Set2", n_colors=len(plot_df[hue_col].unique()))
            else:
                colors = sns.color_palette(palette, n_colors=len(plot_df[hue_col].unique()))
                
            sns.scatterplot(data=plot_df, x=f'{technique}_1', y=f'{technique}_2', 
                          hue=hue_col, palette=colors, s=50, alpha=0.8, 
                          edgecolor='white', linewidth=0.3, ax=ax)
        else:
            ax.scatter(embed_df[f'{technique}_1'], embed_df[f'{technique}_2'], 
                      s=50, alpha=0.7, c='#3498DB', edgecolor='white', linewidth=0.3)
    
    # Enhanced title and axis formatting
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(f'{technique.upper()} 1', fontsize=14, fontweight='semibold')
    ax.set_ylabel(f'{technique.upper()} 2', fontsize=14, fontweight='semibold')
    
    # Remove ticks but keep axis lines clean
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Enhance legend if present
    if ax.get_legend():
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          frameon=True, fancybox=False, shadow=False,
                          title_fontsize=12, fontsize=10)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)
    
    plt.tight_layout()
    return fig


def create_interactive_plot(embed_df, technique, hue_col, title):
    """Create interactive plotly plot"""
    try:
        import plotly.express as px
        
        hover_data = ['domains']
        if 'function_labels' in embed_df.columns:
            hover_data.append('function_labels')
            
        fig = px.scatter(embed_df, x=f'{technique}_1', y=f'{technique}_2',
                        color=hue_col, hover_data=hover_data, title=title)
        fig.update_traces(marker=dict(size=6, opacity=0.8))
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
            for col in ['product_classes', 'function_labels', 'presence']:
                if col in embed_df.columns:
                    color_col = col
                    break
            
            interactive_fig = create_interactive_plot(embed_df, args.technique, color_col, 
                                                    f"Interactive {args.technique.upper()} Plot")
            if interactive_fig:
                save_plot(interactive_fig, args.results_dir, args.tag, emb_name, 'interactive', timestamp)
    
    print("\nDone!")


if __name__ == "__main__":
    main()