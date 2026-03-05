import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
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
        combined_text = df1[COL_MAP[dir1]] + " " + df2[COL_MAP[dir2]]
        texts = combined_text.tolist()
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
    model_dir = f"models/bert/{dataset}/saved_model"
    output_dir = f"models/bert/{dataset}"

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset, gold_labels, org_names, texts = load_test_data(dataset, tokenizer, max_length=512)
    test_loader = DataLoader(test_dataset, batch_size=16)

    predictions = []
    confidences = []
    true_encoded = label_encoder.transform(gold_labels)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            confidences.extend(confs.cpu().numpy())

    pred_labels = label_encoder.inverse_transform(predictions)

    # Save label predictions with confidence
    output_df = pd.DataFrame({
        'organization': org_names,
        'text': texts,
        'true_label': gold_labels,
        'pred_label': pred_labels,
        'confidence': confidences
    })
    output_df.to_csv(os.path.join(output_dir, "label_predictions.csv"), index=False)

    # Save classification report (macro/micro)
    report = classification_report(true_encoded, predictions, output_dict=True, target_names=label_encoder.classes_)
    macro = {
        "macro_precision": precision_score(true_encoded, predictions, average='macro'),
        "macro_recall": recall_score(true_encoded, predictions, average='macro'),
        "macro_f1": f1_score(true_encoded, predictions, average='macro')
    }
    micro = {
        "micro_precision": precision_score(true_encoded, predictions, average='micro'),
        "micro_recall": recall_score(true_encoded, predictions, average='micro'),
        "micro_f1": f1_score(true_encoded, predictions, average='micro')
    }
    report_df = pd.DataFrame(report).transpose()
    summary_df = pd.DataFrame([macro, micro])
    full_report = pd.concat([report_df, summary_df], axis=0)
    full_report.to_csv(os.path.join(output_dir, "classification_report.csv"))

    print(f"Saved predictions and classification report to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="e.g., gsnip, gptsummary, gsnip+gptsummary")
    args = parser.parse_args()
    main(args)
