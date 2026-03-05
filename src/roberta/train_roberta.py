# roberta_train.py
import os
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import argparse
import joblib

class SICCodeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_and_prepare_data(dataset_name, split, tokenizer, max_length):
    data_dir = f"data/{dataset_name}"
    COL_MAP = {
        "gsnip": "google_snippet",
        "llamasummary": "llama-summary",
        "gptsummary": "gpt_response"
    }
    
    if '+' in dataset_name:
        dir1,dir2 = dataset_name.split('+')
        data_dir1 = f"data/{dir1}"
        data_dir2 = f"data/{dir2}"
        df1 = pd.read_csv(os.path.join(data_dir1, f"{split}.csv"))
        df2 = pd.read_csv(os.path.join(data_dir2, f"{split}.csv"))
        df1["combined_text"] = df1[COL_MAP[dir1]] + " " + df2[COL_MAP[dir2]]
        texts = df1["combined_text"].tolist()
        labels = df1["label"].tolist()
    elif dataset_name == "gptsummary":
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        texts = df["gpt_response"].tolist()
        labels = df["label"].tolist()
    elif dataset_name == "llamasummary":
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        texts = df["llama-summary"].tolist()
        labels = df["label"].tolist()
    else:
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        texts = df["google_snippet"].tolist()
        labels = df["label"].tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return encodings, labels

def main(args):
    dataset = args.dataset
    output_dir = f"models/roberta/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_encodings, train_labels = load_and_prepare_data(dataset, "train", tokenizer, 512)
    dev_encodings, dev_labels = load_and_prepare_data(dataset, "dev", tokenizer, 512)

    label_encoder = LabelEncoder()
    train_encoded = label_encoder.fit_transform(train_labels)
    dev_encoded = label_encoder.transform(dev_labels)

    train_dataset = SICCodeDataset(train_encodings, train_encoded)
    dev_dataset = SICCodeDataset(dev_encodings, dev_encoded)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_encoder.classes_))

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "results"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "saved_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "saved_model"))
    joblib.dump(label_encoder, os.path.join(output_dir, "saved_model", "label_encoder.pkl"))
    print(f"\u2705 Training complete. Model saved to {output_dir}/saved_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name like gsnip, gptsummary, gsnip+gptsummary")
    args = parser.parse_args()
    main(args)
