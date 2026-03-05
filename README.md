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

- We **do not release the raw Google search snippets (gsnip)** associated with provider names.
- Instead, we release the **URLs of the retrieved sources** so that users can reproduce the retrieval process.
- We **do not release the LLM-generated summaries** of healthcare providers.
