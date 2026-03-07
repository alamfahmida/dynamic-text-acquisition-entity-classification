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



