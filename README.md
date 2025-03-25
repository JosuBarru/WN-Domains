# Automatic Domain Assignment to Definitions

This project implements an automatic approach for assigning domains to WordNet synset definitions using pre-trained transformer models. The work focuses on fine-tuning state-of-the-art models to classify glosses into BabelDomains labels.

## Overview

The goal is to address a multi-class text classification problem by assigning domain labels to synset definitions. The project involves:
- **Data Cleaning and Preparation:** Processing raw WordNet and BabelDomains files to generate reliable training, development, and test sets.
- **Modeling:** Fine-tuning transformer-based architectures (DistilBERT, BERT, and RoBERTa) for the classification task.
- **Evaluation:** Measuring performance using accuracy, precision, recall, and F1-score, and conducting experiments including few-shot learning.

## Data

- **Training/Development Data:** Extracted and cleaned from `babeldomains_wordnet.txt`, filtered for 100% reliability and with duplicates removed.
- **Test Data:** Derived from `wordnet_dataset_gold.txt` ensuring that test instances are unseen during training.
- **Final Datasets:** Approximately 11,849 instances for training/development and 1,540 for testing.

## Methodology

- **Preprocessing:**  
  - Used shell commands (e.g., `awk`, `sed`) to clean and filter the raw data.
  - Mapped synset IDs to their corresponding glosses by merging various WordNet files.
  - Generated `train.tsv` and `test.tsv` files containing tuples of (gloss, domain).

- **Model Training:**  
  - Fine-tuned three transformer architectures:
    - **DistilBERT**
    - **BERT**
    - **RoBERTa**
  - Performed hyperparameter tuning (learning rate, batch size, epochs, scheduling) and applied early stopping to mitigate overfitting.
  - Explored few-shot learning scenarios with 5, 10, and 20 instances per class.

- **Evaluation:**  
  - Metrics computed include Accuracy, Precision, Recall, and F1-score.
  - Employed a multi-dimensional confusion matrix approach to handle the multi-class setup.
  - Compared results against a zero-shot baseline from prior work (Sainz and Rigau, 2021).

## Results

- **Best Performance:** DistilBERT outperformed BERT and RoBERTa on the test set.
- **Few-Shot Learning:** Models trained with as few as 20 instances per class achieved performance close to that of full-data training.
- **Insights:**  
  - Similar error patterns were observed across models, especially in domains with fewer training instances.
  - Fine-tuning led to notable improvements over the zero-shot approach.
  

## Files and Structure

- **WN-Domains.ipynb:**  
  Contains the complete code for data preprocessing, model fine-tuning and evaluation.
  
- **Informe.pdf:**  
  Provides a detailed report on the methodology, experimental setup, results and analysis.

- **train.tsv & test.tsv:**  
  Processed datasets used for training and testing the models.

## References

- Camacho-Collados, J. & Navigli, R. (2017). *BabelDomains: Large-scale domain labeling of lexical resources*.
- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*.
- Sainz, O. & Rigau, G. (2021). *Ask2Transformers: Zero-shot domain labelling with pretrained language models*.
