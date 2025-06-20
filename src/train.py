from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import pandas as pd
import torch
import os

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

input_dir = "/opt/ml/input/data"
model_dir = "/opt/ml/model"
input_file = [f for f in os.listdir(input_dir) if f.endswith(".parquet")][0]
df = pd.read_parquet(os.path.join(input_dir, input_file))

class FinBertDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.input_ids = df["input_ids"].tolist()
        self.labels = df["label"].tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

dataset = FinBertDataset(df)
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="/opt/ml/output/logs"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)
trainer.train()
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
