# DeepLearning-EntityRecognition
Deep learning NER with the CoNLL-2003 corpus. Uses bidirectional LSTM &amp; GloVe embeddings, addressing case-sensitivity for improved NER accuracy. Repository offers training scripts, model blueprints, and evaluation tools. Optimized for precision, recall, and F1 on dev data.

This repository contains a deep learning model built for Named Entity Recognition (NER) on the CoNLL-2003 corpus.

## Overview

The project is divided into two main tasks:

- **Task 1**: Building a simple bidirectional LSTM model for NER.
- **Task 2**: Enhancing the LSTM model by incorporating GloVe word embeddings.

## Data

The dataset is divided into train, dev, and test sets. The train and dev sets contain sentences with human-annotated NER tags. The test set contains only raw sentences.

GloVe embeddings provided: `glove.6B.100d.gz`

## Results

*For detailed results, see the Results folder.*

- Task 1: Achieved an F1 score of 75% on the dev data.
- Task 2: Achieved an F1 score of 80% on the dev data after incorporating GloVe embeddings.

## Usage

To evaluate the results, use the provided `conll03eval` script.
