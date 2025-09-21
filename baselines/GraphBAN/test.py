import pandas as pd

file_path = r"E:\ProteinDrugInter\CompareModel\GraphBAN\result\test_output_182.pkl"


df = pd.read_pickle(file_path)
print(df)