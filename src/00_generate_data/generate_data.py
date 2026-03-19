import numpy as np
import pandas as pd
import argparse
import sys
import os

# 1. Parse arguments
parser = argparse.ArgumentParser(description="GWAS Simulator")
parser.add_argument("--n_individuals", type=int, default=1000, help="Total number of individuals")
parser.add_argument("--n_snps", type=int, default=5000, help="Total number of SNPs")
args = parser.parse_args()

n_individuals = args.n_individuals
n_snps = args.n_snps

# 1. Define dimensions
#n_individuals = 1000
#n_snps = 5000  # Start small for CSV; real GWAS would be 1M+

if n_snps <= 1200:
    if rank == 0:
        print(f"Error: n_snps ({n_snps}) must be greater than 1200 for the hidden interaction logic to work.")
    sys.exit(1)

# Make sure the files does not exist, otherwise stop immediatelly
if os.path.exists("genotypes.csv"):
    print(f"Error: the file genotypes.csv already exist.")
    sys.exit(1)

# Make sure the files does not exist, otherwise stop immediatelly
if os.path.exists("phenotypes.csv"):
    print(f"Error: the file phenotypes.csv already exist.")
    sys.exit(1)
    
# 2. Generate Genotypes: 0 (Ref/Ref), 1 (Het), 2 (Alt/Alt)
# Using a realistic Allele Frequency (e.g., 20% for the risk allele)
genotypes = np.random.choice([0, 1, 2], size=(n_individuals, n_snps), p=[0.64, 0.32, 0.04])

# Create SNP IDs (rs1, rs2...) and Individual IDs
snp_ids = [f"rs{i}" for i in range(n_snps)]
ind_ids = [f"Indiv_{i}" for i in range(n_individuals)]

# 3. Create the "Hidden" Epistasis (The Ground Truth)
# Let's say if a person has TWO copies of the Alt allele at rs500 
# AND at least ONE copy at rs1200, they have a 90% chance of being a "Case".
phenotypes = np.zeros(n_individuals, dtype=int)
for i in range(n_individuals):
    # The "Biological Logic"
    if genotypes[i, 500] == 2 and genotypes[i, 1200] >= 1:
        # Probabilistic outcome to simulate real-world noise
        phenotypes[i] = np.random.choice([0, 1], p=[0.1, 0.9])
    else:
        # Baseline disease risk in population (5%)
        phenotypes[i] = np.random.choice([0, 1], p=[0.95, 0.05])

# 4. Save to CSV
# Genotype Matrix: Rows = Individuals, Cols = SNPs
df_gen = pd.DataFrame(genotypes, index=ind_ids, columns=snp_ids)
df_gen.to_csv("genotypes.csv")

# Phenotype Vector: 0 = Control, 1 = Case
df_pheno = pd.DataFrame(phenotypes, index=ind_ids, columns=["Trait_Status"])
df_pheno.to_csv("phenotypes.csv")

print(f"Done! Created 'genotypes.csv' ({df_gen.shape}) and 'phenotypes.csv'.")
print(f"Hidden Interaction: rs500 and rs1200.")
