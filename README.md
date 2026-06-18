# Document Similarity using TF-IDF and Cosine Similarity

This repository contains a natural language processing (NLP) pipeline for calculating semantic document similarity across a large text corpus. It extracts textual features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and calculates pairwise cosine similarity scores.

## Project Overview

- **Pipeline (`code.ipynb`)**:
  - Preprocesses document texts (lowercasing, punctuation removal, stopword filtering, stemming/lemmatization).
  - Generates TF-IDF vector representations of document matrices using `scikit-learn`.
  - Computes a full pairwise cosine similarity matrix to measure semantic overlap.
  - Exports calculations and exports similarity scores.
- **Outputs**:
  - `cs.txt` & `cs.npy`: Serialized similarity vectors for specific query documents.
  - `cos_sim.csv` (ignored): Pairwise cosine similarity matrix (890MB, excluded from repository due to size limits).
- **Corpus**:
  - `Dataset.csv` (ignored): Raw document text data (66MB, excluded from repository).

## Repository Structure

```text
├── code.ipynb       # NLP TF-IDF vectorization and cosine similarity calculations
├── cs.txt           # Extracted query similarity scores
├── cs.npy           # Serialized numpy query vectors
└── .gitignore       # Excludes large CSV datasets and pairwise similarity files from Git
```

## Getting Started

### 1. Setup Environment
Ensure you have Python 3.x and the required libraries installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn matplotlib jupyterlab
```

### 2. Execution
Start Jupyter and run the notebook:
```powershell
jupyter lab
```
1. Run `code.ipynb` to process document text from `Dataset.csv`.
2. Generate TF-IDF sparse matrices and compute cosine similarities.
3. Output query similarities to `cs.txt` and `cs.npy`.
