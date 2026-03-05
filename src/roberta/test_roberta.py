import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import argparse
import joblib
from tqdm import tqdm

class SICCodeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_and_prepare_data(dataset_name, tokenizer, label_encoder, max_length=512):
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
        df1 = pd.read_csv(os.path.join(data_dir1, f"test.csv"))
        df2 = pd.read_csv(os.path.join(data_dir2, f"test.csv"))
        df1["combined_text"] = df1[COL_MAP[dir1]] + " " + df2[COL_MAP[dir2]]
        texts = df1["combined_text"].tolist()
        labels = df1["label"].tolist()
        orgs = df1['organization'].tolist()
    elif dataset_name == "gptsummary":
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        texts = df["gpt_response"].tolist()
        labels = df["label"].tolist()
        orgs = df['organization'].tolist()
    elif dataset_name == "llamasummary":
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        texts = df["llama-summary"].tolist()
        labels = df["label"].tolist()
        orgs = df['organization'].tolist()
    else:
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        texts = df["google_snippet"].tolist()
        labels = df["label"].tolist()
        orgs = df['organization'].tolist()

    encoded_labels = label_encoder.transform(labels)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return SICCodeDataset(encodings, encoded_labels), labels, orgs, texts

def main(args):
    dataset = args.dataset
    model_dir = f"models/roberta/{dataset}/saved_model"

    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset, gold_labels_str, orgs, texts = load_and_prepare_data(dataset, tokenizer, label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    predictions = []
    confidences = []
    gold_labels_encoded = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

            predictions.extend(preds.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            gold_labels_encoded.extend(batch['labels'].cpu().numpy())

    pred_labels_str = label_encoder.inverse_transform(predictions)

    # Save detailed report
    report = classification_report(gold_labels_encoded, predictions, target_names=label_encoder.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.loc['micro avg'] = [
        precision_score(gold_labels_encoded, predictions, average='micro'),
        recall_score(gold_labels_encoded, predictions, average='micro'),
        f1_score(gold_labels_encoded, predictions, average='micro'),
        len(gold_labels_encoded)
    ]
    df_report.loc['macro avg'] = [
        precision_score(gold_labels_encoded, predictions, average='macro'),
        recall_score(gold_labels_encoded, predictions, average='macro'),
        f1_score(gold_labels_encoded, predictions, average='macro'),
        len(gold_labels_encoded)
    ]

    output_base = f"models/roberta/{dataset}"
    os.makedirs(output_base, exist_ok=True)

    df_report.to_csv(os.path.join(output_base, "classification_report.csv"))
    print("📊 Classification report saved.")

    df_preds = pd.DataFrame({
        "organization": orgs,
        "text": texts,
        "gold": gold_labels_str,
        "pred": pred_labels_str,
        "confidence": confidences
    })
    df_preds.to_csv(os.path.join(output_base, "label_predictions.csv"), index=False)
    print("Predictions saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="e.g., gsnip, gptsummary, gsnip+gptsummary")
    args = parser.parse_args()
    main(args)
