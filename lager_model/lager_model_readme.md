# LAGER Model Resources

This folder contains the resources and instructions for using pre-trained models in the **LAGER** framework for feature extraction. The models include **protein embeddings**, **ProtBert-BFD**, **ProtBert**, **ESM-2**, and **ChemBERTa**.  


---

## 1. ProtBert-BFD

**ProtBert-BFD** is a transformer-based model for protein sequence embeddings.  
- **Purpose:** Provides high-dimensional embeddings that capture structural and functional information from protein sequences.  
- **Pre-trained model:** Trained on the **BFD** database containing >2 billion protein sequences.  
- **Source:** [Hugging Face – ProtBert-BFD](https://huggingface.co/Rostlab/prot_bert_bfd)  
- **Download and Usage Example:**  
```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
```

---

## 2. ProtBert

**ProtBert** is another transformer-based protein embedding model trained on UniRef100.  
- **Purpose:** Captures functional and evolutionary features from protein sequences.  
- **Source:** [Hugging Face – ProtBert](https://huggingface.co/Rostlab/prot_bert)  
- **Download and Usage Example:**  
```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
```

---

## 3. ESM-2

**ESM-2** is a protein language model from Meta AI that provides high-quality embeddings for protein sequences.  
- **Purpose:** Captures long-range dependencies and structural information of proteins.  
- **Source:** [ESM GitHub](https://github.com/facebookresearch/esm)  
- **Download and Usage Example:**  
```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50S()
tokenizer = alphabet.get_batch_converter()
```

---

## 4. ChemBERTa

**ChemBERTa** is a transformer-based model for molecular (SMILES) embeddings.  
- **Purpose:** Encodes chemical structures into high-dimensional vectors for downstream tasks such as DTI prediction or property prediction.  
- **Pre-trained model:** Trained on large chemical databases like PubChem.  
- **Source:** [Hugging Face – ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)  
- **Download and Usage Example:**  
```python
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
```

---

## References

- ProtBert-BFD: Elnaggar et al., *ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Deep Learning and High-Performance Computing*, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021.  
- ProtBert: Elnaggar et al., *ProtTrans: Transformer Models for Protein Sequence Representation*, *arXiv:2007.06225*, 2020.  
- ESM-2: Lin et al., *Language models of protein sequences at the scale of evolution enable accurate structure prediction*, *Science*, 2023.  
- ChemBERTa: Chithrananda et al., *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction*, *arXiv:2010.09885*, 2020.