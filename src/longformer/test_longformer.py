# test_longformer.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import argparse
import joblib
from tqdm import tqdm

class SICCodeDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def load_test_data(dataset_name, tokenizer, max_length):
    data_dir = f"data/{dataset_name}"
    #Always loading test.csv
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
        texts = df1["combined_text"].tolist()
        labels = df1["label"].tolist()
        org_names = df1["organization"].tolist()
    elif dataset_name == "gptsummary":
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        texts = df["gpt_response"].tolist()
        labels = df["label"].tolist()
        org_names = df["organization"].tolist()
    elif dataset_name == "llamasummary":
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        texts = df["llama-summary"].tolist()
        labels = df["label"].tolist()
        org_names = df["organization"].tolist()
    else:
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        texts = df["google_snippet"].tolist()
        labels = df["label"].tolist()
        org_names = df["organization"].tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return SICCodeDataset(encodings), labels, org_names, texts

def main(args):
    dataset = args.dataset
    model_dir = f"models/longformer/{dataset}/saved_model"

    tokenizer = LongformerTokenizer.from_pretrained(model_dir)
    model = LongformerForSequenceClassification.from_pretrained(model_dir)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset, gold_labels, org_names, texts = load_test_data(dataset, tokenizer, max_length=1024)
    gold_encoded = label_encoder.transform(gold_labels)

    predictions, probs = [], []
    for i in tqdm(range(len(test_dataset)), desc="Running inference"):
        batch = {k: v.unsqueeze(0).to(device) for k, v in test_dataset[i].items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            conf = torch.softmax(logits, dim=1).max().item()
        predictions.append(pred)
        probs.append(conf)

    pred_labels = label_encoder.inverse_transform(predictions)

    # Save label predictions
    df_out = pd.DataFrame({
        "organization": org_names,
        "text": texts,
        "true_label": gold_labels,
        "pred_label": pred_labels,
        "confidence": probs
    })
    df_out.to_csv(f"models/longformer/{dataset}/label_predictions.csv", index=False)

    # Save classification report
    report = classification_report(gold_encoded, predictions, target_names=label_encoder.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    micro_p = precision_score(gold_encoded, predictions, average='micro')
    micro_r = recall_score(gold_encoded, predictions, average='micro')
    micro_f = f1_score(gold_encoded, predictions, average='micro')

    df_report.loc['micro avg'] = [micro_p, micro_r, micro_f, '']
    df_report.to_csv(f"models/longformer/{dataset}/classification_report.csv")
    print(f"Report and predictions saved to models/longformer/{dataset}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name like gsnip, gptsummary, gsnip+gptsummary")
    args = parser.parse_args()
    main(args)
