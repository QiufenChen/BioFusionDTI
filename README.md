# BioFusionDTI
BioFusionDTI: A robust deep learning framework for drug-target interaction (DTI) prediction by integrating graph and sequence modalities.


---

## Installation
1. Create the conda environment:
   ```bash
   conda env create -f biofusiondti.yaml
   ```

2. Activate the environment
```bash
  conda activate biofusiondti
```
3. Install BLAST+ (for generating PSSM features):
You can install via conda:
```bash
conda install -c bioconda blast 
```
or via apt:
```bash
sudo apt update
sudo apt install ncbi-blast+
```
Reference: [NCBI BLAST Documentation](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html)

## Data Preparation
Place protein FASTA sequences in datasets/proteins.fasta
Place SMILES drug files in datasets/drugs.smi
Preprocessed datasets can be organized under the datasets/ folder.

## Feature Extraction
We provide scripts to generate different types of features for drugs and proteins.
### 1. PSSM features
```
python src/get_pssm.py --input datasets/proteins.fasta --output pssm/
```
### 2. Contact map (via esm2_t33_650M_UR50D)
```
python src/get_contact_map.py --input datasets/proteins.fasta --output esm2_contact/
```
### 3. ESM-2 embeddings
```
python src/get_esm2_feature.py --input datasets/proteins.fasta --output esm2_feature/
```
### 4. Drug embeddings (ChemBERTa)
```
python src/get_drug_feature.py --input datasets/drugs.smi --output chembert_feature/
```
## Training & Evaluation
We provide three training settings:
Warm setting (random split):
```python
python train_warm.py
```
Cold drug setting (unseen drugs):
```python
python train_drug.py
```
Cold protein setting (unseen proteins):
```python
python train_prot.py
```
Results (logs, models, results) will be stored under the logs/ , models/ and results/directories.
## Visualization
For plotting and analyzing results:
```python
python visualization.py
```
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
## Results
BioFusionDTI consistently outperforms baseline methods across multiple datasets (SNAP, DRH, Kinase).
Detailed results and ablation studies are reported in the manuscript.
