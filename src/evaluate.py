from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
from sklearn.metrics import f1_score
import json
import os

model_dir = "/opt/ml/processing/model"
data_dir = "/opt/ml/processing/data"
eval_output_dir = "/opt/ml/processing/eval"

model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

input_file = [f for f in os.listdir(data_dir) if f.endswith(".parquet")][0]
df = pd.read_parquet(os.path.join(data_dir, input_file))
X = df["text"].tolist()
y_true = df["label"].tolist()

y_pred = []
for text in X:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    y_pred.append(torch.argmax(outputs.logits, dim=1).item())

score = f1_score(y_true, y_pred, average="weighted")
os.makedirs(eval_output_dir, exist_ok=True)
with open(os.path.join(eval_output_dir, "evaluation.json"), "w") as f:
    json.dump({"metrics": {"f1": score}}, f)
