# BioFusionDTI
**BioFusionDTI**: A robust deep learning framework for drug–target interaction (DTI) prediction by integrating graph and sequence modalities.

---

## Installation

1. Create the conda environment:
   ```bash
   conda env create -f biofusiondti.yaml
   ```

2. Activate the environment:
   ```bash
   conda activate biofusiondti
   ```

3. Install BLAST+ (required for generating PSSM features).  
   You can install via **conda**:
   ```bash
   conda install -c bioconda blast
   ```
   Or via **apt**:
   ```bash
   sudo apt update
   sudo apt install ncbi-blast+
   ```
   Reference: [NCBI BLAST Documentation](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html)

---

## Data Preparation
1. Download all human protein sequences from **UniProt**.  
2. Switch to the `src/` directory:
   ```bash
   cd src/
   ```
3. Run the following script to generate a FASTA file containing all human proteins:
   ```bash
   python get_fasta.py
   ```
This will produce a file with all human protein sequences in FASTA format.

4. Preprocessed datasets can be organized under the `datasets/` folder.

---

## Feature Extraction

We provide scripts to generate different types of features for drugs and proteins.
1. Switch to the `src/` directory:
   ```bash
   cd src/
   ```

2. **PSSM Features**  
   Generate Position-Specific Scoring Matrix (PSSM) features:
   ```bash
   python get_pssm.py
   ```
   **Note:** The results will be saved in the `pssm` directory.

3. **Contact Maps** (using `esm2_t33_650M_UR50D`)  
   Generate protein contact maps:
   ```bash
   python get_contact_map.py
   ```
   **Note:** The results will be saved as `.pt` files in the `esm2_contact` directory.

4. **ESM-2 Embeddings**  
   Extract ESM-2 embeddings for proteins:
   ```bash
   python get_esm2_feature.py
   ```
   **Note:** The results will be saved as `.pt` files in the `esm2_feature` directory.

5. **Drug Embeddings (ChemBERTa)**  
   Generate embeddings for drugs using `ChemBERTa-zinc-base-v1`:
   ```bash
   python get_drug_feature.py
   ```
   **Note:** The results will be saved as `.pkl` files in the `chembert_feature` directory.

---

## Training & Evaluation

We provide three training settings:

- **Warm setting** (random split):
  ```bash
  python train_warm.py
  ```

- **Cold drug setting** (unseen drugs):
  ```bash
  python train_drug.py
  ```

- **Cold protein setting** (unseen proteins):
  ```bash
  python train_prot.py
  ```

Training outputs (logs, checkpoints, and results) will be stored under the `logs/`, `models/`, and `results/` directories.

---

## Visualization

To generate plots and analyze results:
```bash
python visualization.py
```

---

## Repository Structure

```bash
BioFusionDTI/
│── datasets/          # Raw and preprocessed datasets
│── logs/              # Training logs
│── models/            # Saved models
│── results/           # Evaluation results
│── src/               # Source code for feature extraction & training
│── README.md          # Documentation
│── biofusiondti.yaml  # Conda environment file
```

---

## Results

- BioFusionDTI consistently outperforms baseline methods across multiple benchmark datasets (SNAP, DRH, Kinase).  
- Detailed experimental results and ablation studies are reported in the manuscript.  

---
