# Clinical Summary Generation Project

## Overview

This project focuses on generating and evaluating clinical summaries using different natural language processing (NLP) models. The main objective is to compare the performance of various summarization techniques and evaluate their effectiveness using ROUGE scores.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Models and Techniques](#models-and-techniques)
- [Evaluation](#evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Getting Started

To get started with this project, follow these steps:

### Usage

#### Preparing Clinical Reports

Ensure your clinical reports are in a text file named `clinical_report.txt`. Each section should be clearly defined with headers.

#### Generating Summaries

The project uses several NLP models for summarization:

- **BERT-based Extractive Summarizer**: Utilizes BERT for extractive summarization, selecting sentences from the input text that are deemed most important.
- **T5 Model for Abstractive Summarization**: Uses the T5 model to generate abstractive summaries of the clinical text.

## Models and Techniques

### BERT Extractive Summarizer

Utilizes BERT for extractive summarization, selecting sentences from the input text that are deemed most important.

### T5 Abstractive Summarization

Uses the T5 model to generate abstractive summaries of the clinical text.

## Evaluation

The evaluation of the generated summaries is done using ROUGE scores. The ROUGE scores measure the overlap between generated summaries and reference summaries, providing insights into the quality of the summaries.

## Results

### BERT Extractive Summarizer

- Average ROUGE-1 F1: 0.6129
- Average ROUGE-2 F1: 0.3770
- Average ROUGE-L F1: 0.5000

### T5 Abstractive Summarization

- Average ROUGE-1 F1: 0.9037
- Average ROUGE-2 F1: 0.8271
- Average ROUGE-L F1: 0.9037

## Dependencies

- `transformers`: For working with BERT and T5 models
- `torch`: PyTorch library for tensor operations
- `rouge_score`: For ROUGE evaluation
- `summarizer`: BERT extractive summarizer

## License

See the [Documentation](https://ieeexplore.ieee.org/document/10576168) file for details.
