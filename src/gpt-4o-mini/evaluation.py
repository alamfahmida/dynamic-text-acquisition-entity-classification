import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from fractions import Fraction

# ==== Configure Your Input File ====
file_path = "path/to/your/predictions.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Rename columns if necessary
df.rename(columns={'Gold Label': 'label', 'Predicted Label': 'predicted_SIC'}, inplace=True)

# Ensure both columns are strings
df['label'] = df['label'].astype(str)
df['predicted_SIC'] = df['predicted_SIC'].astype(str)

# Compute micro scores
micro_precision = precision_score(df['label'], df['predicted_SIC'], average='micro')
micro_recall = recall_score(df['label'], df['predicted_SIC'], average='micro')
micro_f1 = f1_score(df['label'], df['predicted_SIC'], average='micro')

# Compute macro scores
macro_precision = precision_score(df['label'], df['predicted_SIC'], average='macro')
macro_recall = recall_score(df['label'], df['predicted_SIC'], average='macro')
macro_f1 = f1_score(df['label'], df['predicted_SIC'], average='macro')

# Convert micro precision and recall to fraction
precision_fraction = Fraction(micro_precision).limit_denominator()
recall_fraction = Fraction(micro_recall).limit_denominator()

# Print the results
print(f"Micro Precision: {micro_precision:.3f} ({precision_fraction.numerator}/{precision_fraction.denominator})")
print(f"Micro Recall: {micro_recall:.3f} ({recall_fraction.numerator}/{recall_fraction.denominator})")
print(f"Micro F1 Score: {micro_f1:.3f}")

print(f"Macro Precision: {macro_precision:.3f}")
print(f"Macro Recall: {macro_recall:.3f}")
print(f"Macro F1 Score: {macro_f1:.3f}")
