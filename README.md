# Dynamically Acquiring Text Content to Enable the Classification of Lesser-known Entities for Real-world Tasks
## Overview

Existing Natural Language Processing (NLP) resources often lack the task-specific information required for real-world problems and provide limited coverage of lesser-known or newly introduced entities. For example, business organizations and healthcare providers may need to be classified into a variety of different taxonomic schemes for specific application tasks.

Our goal is to enable domain experts to easily create a task-specific classifier for entities by providing only entity names and gold labels as training data. Our framework dynamically acquires descriptive text about each entity, which is subsequently used to build a text-based classifier.

We propose a text acquisition method that leverages both web-based information retrieval and large language models (LLMs).

We evaluate our framework on two classification tasks:

- **Standard Industrial Classification (SIC) code classification** for organizations
- **Healthcare provider taxonomy code classification** for medical providers
