from collections import OrderedDict, Counter, defaultdict

import pickle
import json
import pandas as pd
import os
import glob
import subprocess
import re
import numpy as np
from Bio import SeqIO
from matplotlib import pyplot as plt
import torch
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

try:
    from sequence_models.collaters import _pad
except ImportError:
    print("Warning: 'sequence_models.collaters._pad' is not available in the environment. Continuing without it.")


# --------------------- This part is for compare with known subclusters and reading files ---------------------

def compare_known_subclusters(known_subcl, bgc, bgc_class, matches, cutoff):
    """
    Calculate the overlap percentage between each match and known subclusters.

    Parameters:
      known_subcl : dict
          Dictionary mapping bgc to list of subcluster entries, where each entry is 
          [first_info_element, ..., domains] and the last element is a comma-separated
          string of domain names.
      bgc : str
          BGC name.
      bgc_class : str
          Class of the bgc.
      matches : list
          List of matches, each a list structured as [topic_num, topic_prob, gene_list, overlap_score],
          where gene_list is a list of (gene, prob) tuples.
      cutoff : float
          Overlap fraction cutoff for reporting a match.

    Returns:
      List of results, each entry is:
          [first_info_element, overlap_fraction, total_overlap, bgc, bgc_class,
           topic_num, topic_prob, overlap_score, overlapping_genes, non_overlapping_genes]
    """
    matches_overlap = []
    print('\nComparing matches to known subclusters')

    # Flatten all known subclusters from the dictionary values.
    all_known_subclusters = [sub for sublist in known_subcl.values() for sub in sublist]

    for match in matches:
        topic_num, topic_prob, gene_list, match_overlap_score = match
        # Extract the set of gene names from the match.
        genes_set = {gene for gene, _ in gene_list}

        for k_sub in all_known_subclusters:
            sub_info = k_sub[0]
            # Assume the last element of k_sub is a comma‐separated string of domains.
            raw_domains = k_sub[-1].split(',')
            # Remove '-' entries.
            k_domains = [dom for dom in raw_domains if dom != '-']
            k_domain_set = set(k_domains)

            # Basic overlap: domains present in both the match and the known subcluster.
            overlap_domains = genes_set & k_domain_set
            l_overlap = len(overlap_domains)

            # Adjust for duplicate domains in the known subcluster.
            domain_counts = Counter(k_domains)
            for dom, count in domain_counts.items():
                if count > 1 and dom in genes_set:
                    # Check if any occurrence in the match has a rounded probability meeting the threshold.
                    # If so, count the extra copies (i.e. count - 1) toward the overlap.
                    if any(round(prob) >= count for gene, prob in gene_list if gene == dom):
                        l_overlap += count - 1

            # Compute the fraction of overlap.
            if k_domains:
                overlap_fraction = l_overlap / len(k_domains)
            else:
                overlap_fraction = 0

            # Only report matches with sufficient overlap and more than one domain.
            if overlap_fraction > cutoff and len(k_domains) > 1:
                overlapping_formatted = ','.join(
                    sorted('{}:{:.2f}'.format(g, p) for g, p in gene_list if g in overlap_domains)
                )
                non_overlapping_formatted = ','.join(
                    sorted('{}:{:.2f}'.format(g, p) for g, p in gene_list if g not in overlap_domains)
                )
                matches_overlap.append([
                    sub_info,
                    round(overlap_fraction, 3),
                    l_overlap,
                    bgc,
                    bgc_class,
                    topic_num,
                    round(topic_prob, 3),
                    round(match_overlap_score, 3),
                    overlapping_formatted,
                    non_overlapping_formatted
                ])

    return matches_overlap



def clean_corpus(vocab_json_path, corpus_csv_path):
    """
    Load and clean a corpus by removing any domain words that are not present in the vocabulary.

    This function loads a vocabulary from a JSON file and a corpus from a CSV file. The vocabulary file is
    expected to contain a key 'domains' mapping to a dictionary of domain tokens, and the first token (typically
    'UNK') is removed from the vocabulary list. The corpus file should have a column named 'domains' where each
    entry is a semicolon-separated string of domain words. The function then cleans the corpus by eliminating any
    words that are not found in the provided vocabulary.

    Args:
        vocab_json_path (str): Path to the JSON file containing the vocabulary.
        corpus_csv_path (str): Path to the CSV file containing the corpus.

    Returns:
        tuple: A tuple (corpus_clean, removed_domains) where:
            - corpus_clean (list of list of str): The cleaned corpus with each document represented as a list of domain words.
            - removed_domains (set): The set of domain words that were removed from the corpus.
    """
    # Load the vocabulary JSON file
    with open(vocab_json_path, 'r') as f:
        tokens = json.load(f)
    
    # Extract the domain tokens and remove the first element ('UNK')
    domains = tokens['domains']
    vocab = list(domains.keys())
    vocab = vocab[1:]
    
    # Load the corpus CSV file
    corpus_df = pd.read_csv(corpus_csv_path)
    corpus_list_raw = corpus_df['domains'].tolist()
    corpus_bgcid = corpus_df['unique_id'].tolist()
    
    # Split each document by semicolon to create a list of domain words per document
    corpus = [doc.split(';') for doc in corpus_list_raw]
    
    # Build a set of all domain words present in the corpus
    vocab_corpus = set(word for doc in corpus for word in doc)
    vocab_set = set(vocab)
    
    # Identify domain words in the corpus that are not in the vocabulary
    removed_domains = vocab_corpus.difference(vocab_set)
    print('Domains removed from corpus (not in vocabulary):', removed_domains)
    
    # Clean the corpus by removing words not present in the vocabulary
    corpus_clean = []
    for doc in corpus:
        cleaned_doc = [word for word in doc if word not in removed_domains]
        corpus_clean.append(cleaned_doc)
    
    print('The corpus has been cleaned, number of domains removed:', len(removed_domains))
    # create a dictionary with bgc id as key and cleaned corpus as value
    corpus_clean = dict(zip(corpus_bgcid, corpus_clean))

    return corpus_clean, removed_domains


def read_clusterfile(infile, m_gens=0, verbose=False, fpath_output=None):
    """
    Read a cluster file and extract biosynthetic gene cluster (BGC) domains.

    Each line in the input file should be formatted as:
        cluster_name, gene1, gene2, ..., geneN

    Each gene is represented as a string of domains separated by semicolons (';'). The '-' entry
    is considered invalid and will be removed from the domain list. For each cluster, if there are at least
    `m_gens` genes that contain at least one valid domain, the function aggregates all valid domains
    (removing any gene boundaries) into a single list.

    Parameters
    ----------
    infile : str
        Path to the input cluster file.
    m_gens : int, optional
        Minimum number of genes (with at least one valid domain) required for a cluster to be included.
        Default is 0.
    verbose : bool, optional
        If True, prints additional filtering information. Default is False.
    fpath_output : str, optional
        Directory path where the resulting dictionary will be saved as a pickle file.
        If None, no file is saved. Default is None.

    Returns
    -------
    clusters : OrderedDict
        An OrderedDict mapping each cluster (BGC) name to a list of valid domains (flattened across all genes)
        with '-' entries removed.
    """
    print(f"\nReading {infile}")
    filtered = 0
    clusters = OrderedDict()

    with open(infile, 'r') as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            cluster = parts[0]
            genes = parts[1:]

            valid_gene_count = 0
            cluster_domains = []
            for gene in genes:
                # Split the gene's domain string and filter out invalid entries.
                domains = [d for d in gene.split(';') if d != '-']
                if domains:
                    valid_gene_count += 1
                    cluster_domains.extend(domains)

            if valid_gene_count < m_gens:
                filtered += 1
                if verbose:
                    print(f"  Excluding cluster '{cluster}': only {valid_gene_count} valid gene(s) (< {m_gens})")
                continue

            if cluster in clusters:
                print(f"Warning: Cluster name '{cluster}' is not unique; skipping duplicate.")
            else:
                clusters[cluster] = cluster_domains

    print(f"Done. Read {len(clusters)} clusters.")
    print(f"{filtered} clusters have fewer than {m_gens} valid gene(s) and were excluded.")

    # keep only the name after / and before . of the infile
    infile = infile.split('/')[-1]
    infile = infile.split('.')[0]


    # Optionally save the cluster dictionary as a pickle file.
    if fpath_output is not None:
        output_filepath = os.path.join(fpath_output, f'bgc_corpus_{infile}.pkl')
        with open(output_filepath, 'wb') as f:
            pickle.dump(clusters, f)
        print(f"Cluster dictionary saved to {output_filepath}")

    return clusters



def read_known_clusters(file_path):
    """
    Reads a tab-delimited known clusters file and extracts domain information,
    while removing any tokens that are '-' from the domain field.
    
    The file is expected to have at least four columns:
        Column 0: BGC identifier
        Column 1: Additional info (e.g. compound name)
        Column 2: (Unused here)
        Column 3: Domain information, as a comma-separated string
    
    Parameters:
        file_path (str): The path to the known clusters file.
    
    Returns:
        List[List[str]]: A list where each element is a list of domain tokens (strings)
                         for that row, with '-' tokens removed.
    """
    domains_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            fields = line.split('\t')
            if len(fields) < 4:
                continue  # Skip lines that don't have enough fields
            # Extract the domain field (column 3)
            domains_field = fields[3]
            domains_field = domains_field.replace(';', ',')
            tokens = [token.strip() for token in domains_field.split(',') if token.strip() and token.strip() != '-']
            domains_list.append(tokens)
    
    return domains_list




def identify_known_subclusters(subcluster_domains, known_subclusters, cutoff=0.5):
    """
    Compare a subcluster domain list to a list of known subclusters and identify matches.

    Parameters:
        subcluster_domains (list of str): List of domain names in the subcluster.
        known_subclusters (list of list of str): List where each entry is a known subcluster (list of domains).
        cutoff (float): Overlap fraction threshold for identifying a match.

    Returns:
        tuple: (flag_same_as_known, index_of_first_match)
    """
    identified_subclusters = []
    subcluster_set = set(subcluster_domains)
    flag_same_as_known = False

    for idx, known_sub in enumerate(known_subclusters):
        known_set = set(known_sub)

        # Compute the number of overlapping domains
        overlap_domains = subcluster_set & known_set
        l_overlap = len(overlap_domains)

        # Handle duplicate domain cases
        domain_counts = Counter(known_sub)
        for dom, count in domain_counts.items():
            if count > 1 and dom in subcluster_set:
                l_overlap += count - 1

        # Compute overlap fraction
        overlap_fraction = l_overlap / len(known_set) if known_set else 0

        # Store result if overlap exceeds cutoff
        if overlap_fraction > cutoff:
            flag_same_as_known = True
            identified_subclusters.append((idx, round(overlap_fraction, 3), list(overlap_domains)))

        if len(identified_subclusters) > 1:
            print('one detected subcluster has multiple matches to known subclusters, now keeping the first detected subcluster, the user need to handle the threshold')
    if len(identified_subclusters)>=1:
        return flag_same_as_known, identified_subclusters[0][0]
    else:
        return False, None

# --------------------- This part is for obtaining the MSA therefore embeddings for the subpfams ---------------------
def extract_protein_id(header):
    """
    Extracts the Protein_ID from a structured FASTA header.
    Example header:
      ">BGC0000001_gid:orfP_pid:AEK75490.1_loc:0;1083;+_1/29"
    Returns "AEK75490.1" (the value after "pid:" and before "_loc:").
    Raises an error if no "pid:" field is found.
    """
    if header.startswith(">"):
        header = header[1:]
    parts = header.split("_")
    for part in parts:
        if part.startswith("pid:"):
            return part.split("pid:")[1]
    
    raise ValueError(f"Error: No 'pid:' field found in header: {header}")


def load_bgc_proteins(bgc_dir):
    """
    Load all protein sequences from FASTA files in the specified directory.
    Each FASTA header is parsed using extract_protein_id to obtain the Protein_ID.
    Returns a dictionary mapping Protein_ID to its sequence.

    Example output:
    {
        "BAJ07841.1": "MKTLLFTLS...",
        "BAJ07842.1": "MSDFKLLGS...",
        ...
    }
    """
    proteins = {}
    fasta_files = glob.glob(os.path.join(bgc_dir, "*.fa")) + glob.glob(os.path.join(bgc_dir, "*.fasta"))
    if not fasta_files:
        raise ValueError(f"No FASTA files found in directory: {bgc_dir}")
    for fasta_file in fasta_files:
        for record in SeqIO.parse(fasta_file, "fasta"):
            pid = extract_protein_id(record.id)
            proteins[pid] = str(record.seq)
    return proteins


def get_header_indices(hit_file):
    """
    Reads the header (first line) of the hit file and returns a dictionary
    mapping column names (in lowercase) to their indices, along with the delimiter.
    Expected header example (columns):
      bgc, g_id, p_id, location, orf_num, tot_orf, domain, q_range, bitscore
    This function is used because the gid sometimes appears and sometimes not.
    """
    with open(hit_file, "r") as f:
        header_line = f.readline().strip()
    if "\t" in header_line:
        delimiter = "\t"
        headers = header_line.split("\t")
    else:
        delimiter = " "
        headers = header_line.split()
    header_indices = {h.lower(): i for i, h in enumerate(headers)}
    return header_indices, delimiter


def parse_hit_line(line, header_indices, delimiter="\t"):
    """
    Parse a single line from the hit file using header_indices.
    Expected columns (based on header):
      - "bgc": BGC ID
      - "p_id": Protein ID
      - "domain": Domain name
      - "q_range": Domain coordinates (e.g. "80;205")
      - "bitscore": Score (if present)
    
    Converts domain coordinates from 1-indexed to 0-indexed.
    If the q_range field does not contain a semicolon, an error is reported and None is returned.
    """
    parts = line.strip().split(delimiter)
    if len(parts) < 5:
        return None
    try:
        bgc_id = parts[header_indices["bgc"]]
        protein_id = parts[header_indices["p_id"]]
        domain_name = parts[header_indices["domain"]]
        domain_coords = parts[header_indices["q_range"]]
        score = ""
        if "bitscore" in header_indices:
            idx = header_indices["bitscore"]
            if idx < len(parts):
                score = parts[idx]
    except KeyError as e:
        print(f"Missing expected header field: {e}")
        return None

    if ";" not in domain_coords:
        print(f"Error: Expected q_range to contain ';' but got '{domain_coords}' in line: {line.strip()}")
        return None

    try:
        start_str, end_str = domain_coords.split(";")
        start = int(start_str) - 1  # Convert from 1-indexed to 0-indexed
        end = int(end_str)
    except Exception as e:
        print(f"Error parsing domain coordinates '{domain_coords}': {e}")
        return None

    return {
        "bgc_id": bgc_id,
        "protein_id": protein_id,
        "domain_name": domain_name,
        "domain_start": start,
        "domain_end": end,
        "score": score
    }


def group_hits(hit_file, proteins):
    """
    Reads the hit file, uses get_header_indices to determine column positions and delimiter,
    and for each hit extracts the corresponding protein subsequence from the proteins dictionary.
    Groups hits by domain name.
    
    Returns a dictionary mapping each domain name to a list of tuples:
      (protein_id, extracted_sequence, header_string)
    Example output:
      {
    "Cyclase_polyket": [
        ("BAJ07841.1", "MKTLLFTLS...", "BAJ07841.1|1-108"),
        ("BAJ07842.1", "MSDFKLLGS...", "BAJ07842.1|4-253")
    ],
    "ketoacyl-synt": [
        ("BAJ07844.1", "MTEFHLVLSS...", "BAJ07844.1|5-242")
    ]
}
    """
    groups = defaultdict(list)
    header_indices, delimiter = get_header_indices(hit_file)
    with open(hit_file, "r") as f:
        # Skip header
        header_line = f.readline()
        for line in f:
            if not line.strip():
                continue
            hit = parse_hit_line(line, header_indices, delimiter)
            if not hit:
                continue
            pid = hit["protein_id"]
            if pid not in proteins:
                print(f"Warning: Protein {pid} not found in the BGC database.")
                continue
            full_seq = proteins[pid]
            domain_seq = full_seq[hit["domain_start"]:hit["domain_end"]]
            header_str = f"{pid}|{hit['domain_start']+1}-{hit['domain_end']}"
            groups[hit["domain_name"]].append((pid, domain_seq, header_str))
    return groups


def write_fasta(seqs, output_file):
    """Write a list of (header, sequence) tuples to a FASTA file."""
    with open(output_file, "w") as f:
        for header, seq in seqs:
            f.write(f">{header}\n{seq}\n")


def run_mafft(input_fasta, output_fasta):
    """
    Run MAFFT to generate a multiple sequence alignment (MSA) from the input FASTA file.
    Assumes that MAFFT is installed and accessible in the system PATH.
    """
    cmd = ["mafft", "--auto", input_fasta]
    with open(output_fasta, "w") as out_f:
        subprocess.run(cmd, stdout=out_f, check=True)

# --------------------- This part is for the others ---------------------
def analyze_variances(model):
    # Get the current variances
    with torch.no_grad():
        current_variances = torch.exp(model.decoder.log_diag).cpu().numpy()
    
    # Print basic statistics
    print(f"Variance shape: {current_variances.shape}")
    print(f"Statistics: min={current_variances.min():.2e}, max={current_variances.max():.2e}, "
          f"mean={current_variances.mean():.2e}, median={np.median(current_variances):.2e}")
    
    # Create visualizations
    plt.figure(figsize=(16, 10))
    
    # 1. Overall histogram of all variance values
    plt.subplot(2, 2, 1)
    plt.hist(current_variances.flatten(), bins=50)
    plt.title("Distribution of All Variances")
    plt.xlabel("Variance Value")
    plt.ylabel("Frequency")
    plt.yscale('log')  # Log scale often helps see the distribution better
    
    # 2. Boxplot of variances by topic
    plt.subplot(2, 2, 2)
    plt.boxplot([current_variances[i] for i in range(current_variances.shape[0])])
    plt.title("Variance Distribution by Topic")
    plt.xlabel("Topic #")
    plt.ylabel("Variance Value")
    
    # 3. Heatmap of variances
    plt.subplot(2, 2, 3)
    plt.imshow(current_variances, aspect='auto', cmap='viridis')
    plt.colorbar(label="Variance")
    plt.title("Variance Heatmap (Topics x Dimensions)")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Topic #")
    
    # 4. Distribution of log-variances (for better visualization of small values)
    plt.subplot(2, 2, 4)
    plt.hist(np.log10(current_variances.flatten()), bins=50)
    plt.title("Distribution of Log10(Variances)")
    plt.xlabel("Log10(Variance)")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    return current_variances



def analyze_variances_script(model, save_path):
    """
    Analyzes the variance of the topic-word distributions in the model.

    Args:
        model: The trained TNTM model.
        save_path: Path object where figures and statistics should be saved.

    Returns:
        current_variances: NumPy array of variance values.
    """
    save_path = Path(save_path)  # Ensure save_path is a Path object
    save_path.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    
    # Extract variances
    with torch.no_grad():
        current_variances = torch.exp(model.decoder.log_diag).cpu().numpy()
    
    # Compute statistics
    variance_stats = {
        "min": float(current_variances.min()),
        "max": float(current_variances.max()),
        "mean": float(current_variances.mean()),
        "median": float(np.median(current_variances))
    }
    
    # print(f"Variance shape: {current_variances.shape}")
    print(f"Statistics: min={variance_stats['min']:.2e}, max={variance_stats['max']:.2e}, "
          f"mean={variance_stats['mean']:.2e}, median={variance_stats['median']:.2e}")
    
    # Save statistics as JSON
    stats_path = save_path / "variance_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(variance_stats, f, indent=4)
    
    # Create and save variance visualizations
    plt.figure(figsize=(16, 10))
    
    # 1. Overall histogram of variance values
    plt.subplot(2, 2, 1)
    plt.hist(current_variances.flatten(), bins=50)
    plt.title("Distribution of All Variances")
    plt.xlabel("Variance Value")
    plt.ylabel("Frequency")
    plt.yscale('log')  # Log scale helps visualize distribution
    plt.grid(True)
    plt.savefig(save_path / "variance_histogram.png")
    
    
    return current_variances

def find_duplicate_values(dictionary, include_none=False):
    """
    Find and report duplicate values in a dictionary.
    
    Args:
        dictionary (dict): The dictionary to analyze
        include_none (bool): Whether to include None values in duplicate analysis
    
    Returns:
        dict: A dictionary where keys are duplicate values and values are lists of 
              keys from original dictionary that have that value
    """
    from collections import Counter
    
    # Extract all values from the dictionary
    values = list(dictionary.values())
    
    # Count occurrences of each value
    value_counts = Counter(values)
    
    # Find values that appear more than once
    duplicates = {}
    for value, count in value_counts.items():
        # Skip None values if include_none is False
        if count > 1 and (value is not None or include_none):
            # Find all keys where this value appears
            keys_with_value = [key for key, val in dictionary.items() if val == value]
            duplicates[value] = keys_with_value
    
    # Print summary
    print(f"Found {len(duplicates)} value(s) that appear multiple times:")
    
    for value, keys in duplicates.items():
        value_str = "None" if value is None else value
        print(f"Value {value_str} appears {len(keys)} times at indices: {keys}")
    
    if not duplicates:
        print("No duplicates found - all values are unique")
    
    return duplicates





# --------------------- This part is for the BiGCARP related ---------------------
class ListDataset_extraction(Dataset):
    def __init__(self, list_of_token_tensors):
        self.data = list_of_token_tensors
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

def mlm_collate_fn_extraction(batch, mask_idx, padding_idx, mask_frac=0.15): # !!!this collate function does not apply masking (for getting embeddings)
    src_list, mask_list = [], []
    for t in batch:
        s = t.clone()
        if len(s) == 0:
            continue
        # No masking is applied: local_mask is all zeros.
        local_mask = torch.zeros_like(s, dtype=torch.float)
        src_list.append(s)
        mask_list.append(local_mask)
    src = _pad(src_list, padding_idx)
    mask = _pad(mask_list, 0)
    return src, mask



def extract_layer_embeddings(model, src, input_mask=None, layer_indices=None, layer_weights=None):
    """
    Extract embeddings from the model.

    Args:
        model: The neural network model (ByteNetLM)
        src: Input token IDs (shape: [batch_size, seq_len])
        input_mask: Attention mask (shape: [batch_size, seq_len, 1])
        layer_indices: Either ["embedder"] for raw embedding layer or ["last"] for final contextualized layer
                       If None, defaults to ["last"]
        layer_weights: Not used (kept for backward compatibility)

    Returns:
        Embeddings from specified layer (shape: [batch_size, seq_len, hidden_dim])
    """
    # Special case: extract only from embedding layer (raw, non-contextualized)
    if layer_indices == ["embedder"]:
        hidden = model.embedder.embedder(src)
        return hidden

    # Default case: extract from last layer (fully contextualized)
    # This handles both layer_indices == ["last"] and layer_indices == None
    hidden = model.embedder(src, input_mask=input_mask)
    return model.last_norm(hidden)


def generate_average_embeddings(model, dl, n_tokens, MIN_SPECIAL_ID, padding_idx, mask_idx,
                               layer_indices=None, layer_weights=None, device=None):
    """
    Generate average embeddings for tokens by extracting from the model.

    Args:
        model: The neural network model (ByteNetLM)
        dl: DataLoader with input data (from mlm_collate_fn_extraction)
        n_tokens: Total number of tokens in vocabulary
        MIN_SPECIAL_ID: Minimum token ID to include in extraction (typically 4, to skip special tokens)
        padding_idx: Token ID of padding token to ignore
        mask_idx: Token ID of mask token (used to create input_mask)
        layer_indices: Either ["embedder"] for embedder-layer embeddings or ["last"] for last-layer
                       If None, defaults to ["last"]
        layer_weights: Not used (kept for backward compatibility)
        device: Device to run model on. If None, uses model's current device
]
    Returns:
        Dictionary mapping token IDs to their average embeddings (torch.Tensor)
    """
    from collections import defaultdict
    embedding_sums = defaultdict(lambda: None)
    embedding_counts = defaultdict(int)
    
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for src, mask in tqdm(dl, desc="Processing Batches", total=len(dl), 
                             leave=False, position=1, ncols=80, mininterval=1.0):
            # Move data to device
            src = src.to(device)
            mask = mask.to(device)
            input_mask = (src != mask_idx).float().unsqueeze(-1)
            hidden = extract_layer_embeddings(
                model, src, input_mask, layer_indices, layer_weights
            )

            bs, seq_len, hidden_dim = hidden.shape

            if all(v is None for v in embedding_sums.values()):
                for t_id in range(n_tokens):
                    if t_id >= MIN_SPECIAL_ID and t_id != padding_idx:
                        embedding_sums[t_id] = torch.zeros(hidden_dim, device='cpu')  # Keep sums on CPU to save GPU memory

            for i in range(bs):
                for j in range(seq_len):
                    t_id = src[i, j].item()
                    if t_id < MIN_SPECIAL_ID or t_id == padding_idx:
                        continue
                    embedding_sums[t_id] += hidden[i, j].cpu()
                    embedding_counts[t_id] += 1

    avg_embeddings = {}
    never_appearing_tokens = []  # List to store tokens that never appear
    for t_id, sum_vec in embedding_sums.items():
        count = embedding_counts[t_id]
        if count > 0:
            avg_embeddings[t_id] = sum_vec / count
        else:
            avg_embeddings[t_id] = torch.zeros_like(sum_vec)
            never_appearing_tokens.append(t_id)  # Track tokens that never appear

    # Print the never appearing tokens
    if never_appearing_tokens:
        print(f"Tokens that never appeared in the corpus number: {len(never_appearing_tokens)}")
        print("Never appearing tokens (first ten):", never_appearing_tokens[0:10])  # Print first 10 for brevity
    else:
        print("All tokens appeared at least once in the corpus.")

    return avg_embeddings



# --------------------This part is for file conversion --------------------

def convert_pfam_dat_to_json(pfam_file, output_file):
    """
    Converts a Pfam-A HMM file to a JSON mapping of domain IDs to Pfam accessions.

    The script parses the Pfam-A HMM file, extracts the domain ID (`#=GF ID`) and 
    Pfam accession (`#=GF AC`) from each record, and creates a mapping. The mapping 
    is saved as a JSON file.

    Args:
        input_path (str): Path to the input Pfam-A HMM file.
        output_path (str): Path to the output JSON file containing the domain-to-Pfam mapping.

    Example:
        Input Pfam-A HMM file snippet:
            #=GF ID   1-cysPrx_C
            #=GF AC   PF10417.14
            //
        
        Output JSON file:
            {
                "1-cysPrx_C": "PF10417.14"
            }
    """
    mapping = {}
    current_id = None
    current_ac = None

    with open(pfam_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#=GF ID"):
                # Expected format: "#=GF ID   <identifier>"
                parts = line.split()
                if len(parts) >= 3:
                    current_id = parts[2]
            elif line.startswith("#=GF AC"):
                # Expected format: "#=GF AC   <accession>"
                parts = line.split()
                if len(parts) >= 3:
                    current_ac = parts[2]
            elif line == "//":
                # End of a block; if both ID and AC are found, store the mapping
                if current_id and current_ac:
                    mapping[current_id] = current_ac
                # Reset for the next block
                current_id = None
                current_ac = None

    # Write the mapping to a JSON file
    with open(output_file, 'w') as out:
        json.dump(mapping, out, indent=4)
    print(f"Mapping file written to {output_file}")


def convert_pfam_vocab_to_pid(vocab_file, mapping_file, output_file):
    """
    Transforms a domain vocabulary file to use Pfam IDs instead of domain names
    using a provided mapping file.

    Args:
        vocab_file (str): Path to the domain vocabulary file.
        mapping_file (str): Path to the domain to Pfam mapping file.
        output_file (str): Path to save the PID-based vocabulary file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the domain vocabulary
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    domains = vocab["domains"]

    # Load the domain to Pfam mapping
    with open(mapping_file, "r") as f:
        domain_to_pfam = json.load(f)

    # Create a new domains dictionary with Pfam IDs
    new_domains = {}
    unmapped_domains = 0
    
    for domain_name, index in domains.items():
        if domain_name in domain_to_pfam:
            # Extract Pfam ID (remove version number after the dot if present)
            pfam_id = domain_to_pfam[domain_name].split('.')[0]
            new_domains[pfam_id] = index
        else:
            # Keep the original domain name if no mapping exists
            new_domains[domain_name] = index
            unmapped_domains += 1

    # Update the vocabulary with the new domains
    vocab["domains"] = new_domains

    # Save the updated vocabulary to a JSON file
    with open(output_file, "w") as f:
        json.dump(vocab, f, indent=4)

    print(f"Updated vocabulary with {len(new_domains)} domains.")
    print(f"Found {unmapped_domains} domains without a Pfam mapping.")
    print(f"PID-based vocabulary saved to {output_file}")


def convert_subpfam_vocab_to_pid(vocab_file, mapping_file, output_file):
    """
    Creates the final vocabulary file in the format:
    
    {
      "specials": { ... },
      "domains_array": [ ... ],
      "size": <number>
    }
    
    Steps:
    1. Loads the original vocabulary (which includes a "domains" dictionary)
       and a mapping file that maps domain names to Pfam IDs.
    2. For each domain in the vocabulary:
         - Remove any trailing '_c<number>' suffix.
         - If a mapping exists for the base domain name, replace it with the corresponding Pfam ID
           (ignoring any version number after a dot).
         - Otherwise, keep the original domain name.
    3. Builds an array (list) using the domain→index mapping.
    4. Removes any leading null values and the "UNK" token from the beginning of the array.
    5. Forward-fills any remaining None values with the last valid domain.
    6. Constructs a final dictionary with "specials" (from the original vocabulary), the processed
       "domains_array", and its "size", then saves it to the output file.
    """
    # Step 1: Load the vocabulary and mapping files.
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    domains = vocab["domains"]

    with open(mapping_file, "r") as f:
        domain_to_pfam = json.load(f)

    # Step 2: Create a new domains dictionary with Pfam IDs.
    new_domains = {}
    for domain_name, index in domains.items():
        # Remove the '_c<number>' suffix if present.
        match = re.match(r"^(.*)_c\d+$", domain_name)
        base_domain = match.group(1) if match else domain_name

        if base_domain in domain_to_pfam:
            # Replace with Pfam ID, removing any version after the dot.
            pfam_id = domain_to_pfam[base_domain].split('.')[0]
            new_domains[pfam_id] = index
        else:
            new_domains[domain_name] = index

    # Step 3: Convert the domains dictionary into an array based on index.
    max_index = max(new_domains.values())
    temp_array = [None] * (max_index + 1)
    for domain, index in new_domains.items():
        temp_array[index] = domain

    # Step 4: Remove leading nulls and the "UNK" token, if present.
    unk_index = None
    for i, domain in enumerate(temp_array):
        if domain == "UNK":
            unk_index = i
            break

    if unk_index is not None:
        start_index = unk_index + 1
    else:
        start_index = 0
        while start_index < len(temp_array) and temp_array[start_index] is None:
            start_index += 1

    domains_array = temp_array[start_index:]

    # Step 5: Forward-fill any remaining None values.
    last_valid = None
    for i in range(len(domains_array)):
        if domains_array[i] is not None:
            last_valid = domains_array[i]
        elif last_valid is not None:
            domains_array[i] = last_valid

    # Step 6: Build the final vocabulary dictionary.
    final_vocab = {
        "specials": vocab.get("specials", {}),
        "domains_array": domains_array,
        "size": len(domains_array)
    }

    # Save the final vocabulary to the output file.
    with open(output_file, "w") as f:
        json.dump(final_vocab, f, indent=2)

    print(f"Final vocabulary saved to {output_file}")
    print(f"Domains array size: {len(domains_array)}")


