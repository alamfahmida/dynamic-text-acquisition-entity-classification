import openai
import json
import pandas as pd
import re
import time
import random
from tqdm import tqdm  # Progress bar

# Set your OpenAI API key
openai.api_key = "your-api-key"

# ==== Configure your paths ====
test_file = "path/to/your/test_file.jsonl"  # JSONL file with ChatML format test data
output_file = "path/to/save/predictions.csv"

# Provide your fine-tuned model ID
model_id = "your-fine-tuned-model-id"

# Load full test dataset
test_data = []
try:
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]  # Load ALL rows
except Exception as e:
    print(f"Error loading test file: {e}")
    test_data = []

# Prepare a list to store predictions
predictions = []

print(f"\nRunning Inference on {len(test_data)} records with Fine-Tuned GPT-4o Mini...\n")

# Loop through test data and get predictions with a progress bar
for entry in tqdm(test_data, desc="Processing", unit="query"):
    user_message = entry["messages"][1]["content"]  # Extract user message

    # Extract Organization Name safely
    org_match = re.search(r'Organization:\s*(.*?)\s*(?:\n|$)', user_message)
    organization_name = org_match.group(1) if org_match else "Unknown Organization"

    # Extract Description safely
    desc_match = re.search(r'Description:\s*(.*)', user_message, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else "No Description"

    # API request with retries (Exponential Backoff)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that classifies organizations into SIC codes. Your response must be a valid SIC code index, nothing else."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=10,
                temperature=0
            )
            break  # Exit loop if successful
        except Exception as e:
            print(f"API request failed (attempt {attempt + 1} of {max_retries}): {e}")
            time.sleep(2 ** attempt + random.uniform(0, 1))  # Exponential backoff
    else:
        print(f"Failed to get a response after multiple retries for: {organization_name}")
        predictions.append({"Organization": organization_name, "Description": description, "Predicted_SIC": "Error"})
        continue

    # Extract model response
    prediction_text = response["choices"][0]["message"]["content"].strip()

    # Debugging output: Print raw response
    print(f"\nRAW RESPONSE: {prediction_text}\n")

    # Store the raw SIC code as predicted (no filtering)
    sic_code = prediction_text

    # Print real-time progress
    print(f"Organization: {organization_name} → Predicted SIC Code: {sic_code}")

    # Store results (only Organization, Description, and Prediction)
    predictions.append({
        "Organization": organization_name,
        "Description": description,
        "Predicted_SIC": sic_code
    })

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions to CSV
predictions_df.to_csv(output_file, index=False)

print(f"\nPredictions saved to {output_file}")
print("\nInference Completed ✅")
