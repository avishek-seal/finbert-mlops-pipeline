import pandas as pd
import os
from transformers import AutoTokenizer

input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

input_file = [f for f in os.listdir(input_dir) if f.endswith(".parquet")][0]
df = pd.read_parquet(os.path.join(input_dir, input_file))

df["input_ids"] = df["text"].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=128))
df["label"] = df["label"].astype(int)

df.to_parquet(os.path.join(output_dir, "data_tokenized.parquet"))
