import os
import pandas as pd

output_dir = "../fasta"
os.mkdir(output_dir, exist_ok=True)

df = pd.read_excel("../datasets/uniprotkb_2025_02_13.xlsx")
for idx, row in df.iterrows():
    entry = row['Entry']
    sequence = row['Sequence']
    
    fasta_content = f">{entry}\n{sequence}\n"
    
    file_path = output_dir / f"{entry}.fasta"
    with open(file_path, "w") as f:
        f.write(fasta_content)

print(f"FASTA files saved in {output_dir}/")
