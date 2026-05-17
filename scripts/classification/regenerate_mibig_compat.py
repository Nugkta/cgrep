"""
Quick script to regenerate MIBiG preprocessed data in compatible format.
This loads the existing embedding file and TSV data to create a compatible pickle.
"""
import pandas as pd
import os

# Load existing embedding to get bgc_ids and product classes
print("Loading existing embedding file...")
emb_file = "artifacts/classification/mibig3/random_init/mibig3_bigcarp_last.pkl"
df_emb = pd.read_pickle(emb_file)
print(f"Loaded {len(df_emb)} BGCs with embeddings")

# Load TSV to get domain sequences
print("Loading TSV domain data...")
tsv_file = "data/processed/bgc_product_classification/mibig_gbk_3.0_modified.tsv"
df_tsv = pd.read_csv(tsv_file, sep='\t')

# Group by BGC and concatenate domains
print("Grouping domains by BGC...")
bgc_domains = df_tsv.groupby('sequence_id')['pfam_id'].apply(lambda x: ';'.join(x)).reset_index()
bgc_domains.columns = ['bgc_id', 'domains']

# Merge with embedding data to get product classes
print("Merging with product class info...")
df_final = bgc_domains.merge(df_emb[['bgc_id', 'product_class']], on='bgc_id', how='inner')

print(f"\nFinal dataset: {len(df_final)} BGCs")
print(f"Columns: {df_final.columns.tolist()}")
print("\nSample:")
print(df_final.head(2))

# Save in compatible format
output_path = "data/processed/bgc_product_classification/processed_mibig3/mibig3_preprocessed_compat.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_final.to_pickle(output_path)
print(f"\nSaved to: {output_path}")
