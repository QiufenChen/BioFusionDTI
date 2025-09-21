# BioFusionDTI
BioFusionDTI: A robust deep learning framework for drug-target interaction (DTI) prediction by integrating graph and sequence modalities.


---

## Installation
1. Create the conda environment:
   ```bash
   conda env create -f biofusiondti.yaml
   conda activate biofusiondti
   ```

## Feature Extraction
We provide scripts to generate different types of features for drugs and proteins.
### 1. PSSM features
```
python src/generate_pssm.py --input datasets/proteins.fasta --output pssm/
```
### 2. Contact map
```
python src/generate_contact.py --input datasets/proteins.fasta --output esm2_contact/
```
### 3. ESM-2 embeddings
```
python src/generate_esm2.py --input datasets/proteins.fasta --output esm2_feature/
```
### 4. Drug embeddings (ChemBERTa)
```
python src/generate_drug_emb.py --input datasets/drugs.smi --output chembert_feature/
```
## Citation
If you use this code, please cite:
