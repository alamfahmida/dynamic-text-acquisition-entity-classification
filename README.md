# Dynamically Acquiring Text Content to Enable the Classification of Lesser-known Entities for Real-world Tasks
## Overview

Existing Natural Language Processing (NLP) resources often lack the task-specific information required for real-world problems and provide limited coverage of lesser-known or newly introduced entities. For example, business organizations and healthcare providers may need to be classified into a variety of different taxonomic schemes for specific application tasks.

Our goal is to enable domain experts to easily create a task-specific classifier for entities by providing only entity names and gold labels as training data. Our framework dynamically acquires descriptive text about each entity, which is subsequently used to build a text-based classifier.

We propose a text acquisition method that leverages both web-based information retrieval and large language models (LLMs).

We evaluate our framework on two classification tasks:

- **Standard Industrial Classification (SIC) code classification** for organizations
- **Healthcare provider taxonomy code classification** for medical providers

## How-To Guides

### Repository Clone

To get started, first clone the GitHub repository:

```bash
git clone https://github.com/alamfahmida/dynamic-text-acquisition-entity-classification.git
cd dynamic-text-acquisition-entity-classification
```
### Create and Activate Virtual Environment

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment.

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

Install all required Python packages:

```bash
pip install -r requirements.txt
```
## Dataset

This repository includes two datasets used in our experiments:

1. **SIC Code Dataset** – for organization classification  
2. **Healthcare Provider Taxonomy Dataset** – for healthcare provider classification  

### SIC Code Dataset

The **SIC-code dataset** is used for classifying organizations into **Standard Industrial Classification (SIC) codes**.

The dataset is available on Hugging Face and must be downloaded before running any training or testing script.

### Dataset Download Instructions

```bash
git lfs install
git clone https://huggingface.co/datasets/ICICLE-AI/organization-sic-code_smart-foodsheds
```

After downloading, extract and place the unzipped `data/sic-code` folder in the root directory (next to `src/`).

### Dataset Variants

The dataset includes multiple variants based on the source of the organization descriptions:

- **gsnip** – Google search snippets  
- **gptsummary** – GPT-4o-mini generated summaries  
- **llamasummary** – LLaMA 3.1–8B Instruct generated summaries  
- **gsnip+gptsummary** – Combined inputs of Google snippets + GPT-4o-mini generated summaries  
- **gsnip+llamasummary** – Combined inputs of Google snippets + LLaMA 3.1–8B Instruct generated summaries  

Each variant includes the following splits:

- `train.csv`
- `dev.csv`
- `test.csv`

### Healthcare Provider Taxonomy Dataset

The **Healthcare Provider Taxonomy dataset** is used for classifying healthcare providers into **healthcare provider taxonomy codes**, which represent a provider’s medical specialty and area of practice.

Due to ethical considerations related to identifiable healthcare providers, we do **not release certain derived data directly** in this repository.

Specifically:

- We **do not release the raw Google search snippets (GSnip)** associated with provider names.
- Instead, we provide detailed instructions on how others can reconstruct the dataset by following our procedure.
- We **do not release the LLM-generated summaries** of healthcare providers.

### Instructions for Reconstructing Google Snippet Passages

In our experiments, we use **text snippets**, which are the small blocks of text that appear underneath a link to a webpage in a search engine results page. These snippets are typically around **100–200 characters long** and provide a short description of the webpage content.

The snippets used in our experiments were originally obtained using **SerpAPI**.

To avoid redistributing third-party content, we do **not release the raw snippets directly**. Instead, we provide instructions for reproducing similar snippets.

In this release, we provide only the **entity list (provider names)** and the corresponding **gold taxonomy labels**.

The dataset is released as:

```
data/
└── healthcare-taxonomy-code/
    └── healthcare-taxonomy-dataset.zip
```

After extraction, the directory structure will be:

```
data/healthcare-taxonomy-code/
├── dev/
├── test/
└── train/
```


Each split contains the same column structure. Each row corresponds to a healthcare provider and includes:

- `provider_name` – the name of the healthcare provider  
- `Healthcare Provider Taxonomy Code_1` – the taxonomy code label (gold label)  
- `Grouping`, `Classification` – taxonomy hierarchy information  

To reconstruct similar snippets, users can follow these steps:

1. Download the `healthcare-taxonomy-dataset.zip` file and extract it.
2. Use the **provider name** as the search query.
3. Use a web search API (e.g., **SerpAPI**, **Google Custom Search API**, or another search service) to collect the small blocks of text that appear underneath links in the search engine results page.
4. Collect the **top 10 snippets** returned by the search results.
5. **Concatenate the top 10 snippets** and store them as **GSnip**.
6. Repeat this process for all providers in the **train, dev, and test** sets.

## Reconstructing LLM-Generated Text

In addition to web-retrieved snippets (GSnip), our framework also generates **task-specific descriptive summaries using Large Language Models (LLMs)**. These summaries provide structured descriptions of the entities and are used as an additional source of text for training the classification models.

In our experiments, we generated summaries using two LLMs:

- **GPT-4o-mini** (`gpt-4o-mini-2024-07-18`)
- **LLaMA-3.1-8B-Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)

Due to redistribution restrictions on generated content, we **do not release the LLM-generated summaries directly**. Instead, we provide the prompts used in our experiments so that users can reproduce them.

---

## GPT-4o-mini Summary Generation

We generated summaries using **GPT-4o-mini** through the OpenAI API.

**Prompt:**
 Summarize the healthcare specialization, scope of practice, and typical services provided by [PROVIDER_NAME]. The summary should describe the clinician’s professional type and main field of practice, following standard U.S. healthcare taxonomy conventions.

 ## LLaMA-3.1-8B Summary Generation

We also generated summaries using **LLaMA-3.1-8B-Instruct**, which benefits from more explicit prompting to reduce hallucinations and ensure factual descriptions.

**Prompt:**
You are a research assistant writing a factual summary about a healthcare provider’s specialty.

Given the [PROVIDER_NAME], your goal is to identify and describe their medical specialty, professional focus, qualifications, and the healthcare sector they operate in.

Use only publicly verifiable information. The description should be informative, objective, and around 250–300 words. Do not add any assumptions or speculative content.

## Models

Each model directory under `src/` contains separate training and testing scripts.

You only need to specify the dataset variant using the `--dataset` argument.

Accepted options include:

- `gsnip`
- `gptsummary`
- `llamasummary`
- `gsnip+gptsummary`
- `gsnip+llamasummary`

---

## Training

Run the training scripts for each model as follows:

```bash
python src/bert/train_bert.py --dataset gsnip
python src/roberta/train_roberta.py --dataset gptsummary
python src/longformer/train_longformer.py --dataset gsnip+llamasummary
```

### Saved Model

After training, the model and related artifacts are saved to: models/<model_type>/<dataset>/saved_model/


Where:

- `model_type` – Name of the model used (e.g., `bert`, `roberta`, `longformer`)
- `dataset` – Dataset variant passed via `--dataset` (e.g., `gsnip`, `gptsummary`, `llamasummary`, `gsnip+gptsummary`, `gsnip+llamasummary`)

The folder structure is created automatically by the script; no manual setup is required.

### Test

Run the testing scripts as follows:

```bash
python src/bert/test_bert.py --dataset gsnip
python src/roberta/test_roberta.py --dataset gptsummary
python src/longformer/test_longformer.py --dataset gsnip+llamasummary
```
All scripts are designed to automatically handle variations in file naming and input formats.

### Output Files

After running the classification pipeline, the following output files will be generated:

**label_predictions.csv**

- `org_name` – The name of the organization  
- `true_label` – The ground-truth SIC category label  
- `predicted_label` – The label (SIC code) predicted by the model  
- `confidence_score` – The model's confidence in the prediction  

**classification_report.csv**

This file reports the overall performance of the model across all SIC categories using standard evaluation metrics:

- `precision` – Correct positive predictions out of all predicted positives  
- `recall` – Correct positive predictions out of all actual positives  
- `f1-score` – Harmonic mean of precision and recall  
- `support` – Number of true instances for each class  

The final row provides **macro, micro, and weighted averages** for a comprehensive summary of model performance.

### Ethics Statement

All healthcare providers included in our benchmark are based in the United States. We obtained the provider names and their corresponding taxonomy codes from the **National Plan and Provider Enumeration System (NPPES)**, maintained by the **Centers for Medicare & Medicaid Services (CMS)**. The NPPES registry is publicly accessible and downloadable in the United States, and the data are released by CMS as publicly available records. Healthcare providers submit this information themselves as part of the **National Provider Identifier (NPI)** registration process, which is required to become HIPAA-covered healthcare providers in the United States.

All experiments were conducted for research purposes only. The dataset does not contain sensitive attributes beyond publicly available professional information. We believe these measures ensure compliance with scientific integrity standards while minimizing potential ethical risks related to data redistribution.

### Acknowledgements

This research was supported in part by the **ICICLE project** through **NSF award OAC-2112606**.
